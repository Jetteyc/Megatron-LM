# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
import os

import torch

from megatron.core.network_engine import (
    get_global_network_engine,
    is_network_engine_stream_ownership_enabled,
)
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_global_memory_buffer,
    get_tensor_model_parallel_group,
)
from megatron.core.utils import get_tensor_model_parallel_group_if_none

try:
    from megatron.core.network_engine.enums import ParallelDomain
except Exception:
    ParallelDomain = None

from .utils import split_tensor_along_last_dim


def _ne_stream_ctx(domain, group):
    """Get a ``torch.cuda.stream`` context manager for *domain* + *group*.

    Returns ``nullcontext()`` when the NetworkEngine is unavailable or
    stream resolution fails, so the caller always falls back to the
    default stream.
    """
    if (
        os.getenv("STAGGERED_1F1B", "0") != "1"
        or not is_network_engine_stream_ownership_enabled()
    ):
        return nullcontext()
    try:
        ne = get_global_network_engine()
        stream = ne.get_comm_stream_for_domain(domain=domain, group=group)
        if stream is not None:
            return torch.cuda.stream(stream)
    except Exception:
        pass
    return nullcontext()


def _infer_domain_for_group(group, default=ParallelDomain.TP):
    if ParallelDomain is None:
        return default
    try:
        tp_group = get_tensor_model_parallel_group(check_initialized=False)
        if group is tp_group:
            return ParallelDomain.TP
    except Exception:
        pass
    try:
        cp_group = get_context_parallel_group(check_initialized=False)
        if group is cp_group:
            return ParallelDomain.CP
    except Exception:
        pass
    return default


def _reduce(input_, group):
    """All-reduce the input tensor across model parallel group."""
    assert group is not None, "group should not be None"

    if group.size() == 1:
        return input_

    domain = ParallelDomain.TP if ParallelDomain is not None else None
    with _ne_stream_ctx(domain, group):
        torch.distributed.all_reduce(input_.contiguous(), group=group)
    return input_


def _split_along_last_dim(input_, group):
    """Split the tensor along its last dimension and keep the corresponding slice."""
    assert group is not None, "group should not be None"

    world_size = group.size()
    if world_size == 1:
        return input_

    input_list = split_tensor_along_last_dim(input_, world_size)
    rank = group.rank()
    output = input_list[rank].contiguous()
    return output


def _split_along_first_dim(input_, group):
    """Split the tensor along its first dimension and keep the corresponding slice."""
    assert group is not None, "group should not be None"

    world_size = group.size()
    if world_size == 1:
        return input_

    dim_size = input_.size()[0]
    assert (
        dim_size % world_size == 0
    ), "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = group.rank()
    dim_offset = rank * local_dim_size

    output = input_[dim_offset : dim_offset + local_dim_size].contiguous()
    return output


def _gather_along_last_dim(input_, group):
    """Gather tensors and concatinate along the last dimension."""
    world_size = group.size()
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    input_contiguous = input_.contiguous()
    output_tensor_list = list(torch.split(output, input_contiguous.shape[0], dim=0))
    domain = ParallelDomain.TP if ParallelDomain is not None else None
    with _ne_stream_ctx(domain, group):
        torch.distributed.all_gather(output_tensor_list, input_contiguous, group=group)
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=-1).contiguous()
    return output


def _reduce_scatter_along_last_dim(input_, group):
    """Reduce-scatter tensors on the last dimension."""
    world_size = group.size()
    target_shape = list(input_.size())
    target_shape[-1] = target_shape[-1] // world_size
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = _reduce_scatter_along_first_dim(concat_tensor, group=group).reshape(target_shape)
    return output


def _gather_along_first_dim(input_, group, output_split_sizes=None, use_global_buffer=False):
    """Gather tensors and concatenate along the first dimension."""
    assert group is not None, "group should not be None"
    world_size = group.size()
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        input_contiguous = input_.contiguous()
        output_tensor_list = list(torch.split(output, input_contiguous.shape[0], dim=0))
        domain = _infer_domain_for_group(group, default=ParallelDomain.CP)
        with _ne_stream_ctx(domain, group):
            torch.distributed.all_gather(output_tensor_list, input_contiguous, group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        domain = _infer_domain_for_group(group, default=ParallelDomain.CP)
        with _ne_stream_ctx(domain, group):
            torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def _reduce_scatter_along_first_dim(input_, group, input_split_sizes=None, use_global_buffer=False):
    """Reduce-scatter the input tensor across model parallel group."""
    assert group is not None, "group should not be None"
    world_size = group.size()
    if world_size == 1:
        return input_

    if input_split_sizes is None:
        dim_size = list(input_.size())
        assert (
            dim_size[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"

        dim_size[0] = dim_size[0] // world_size

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        input_contiguous = input_.contiguous()
        chunk_size = input_contiguous.shape[0] // world_size
        input_tensor_list = list(torch.split(input_contiguous, chunk_size, dim=0))
        domain = _infer_domain_for_group(group, default=ParallelDomain.CP)
        with _ne_stream_ctx(domain, group):
            torch.distributed.reduce_scatter(output, input_tensor_list, group=group)
    else:
        rank = group.rank()
        input_tensor_list = list(torch.split(input_, input_split_sizes, dim=0))

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(
                input_tensor_list[rank].shape, input_.dtype, "mpu"
            )
        else:
            output = torch.empty_like(input_tensor_list[rank])
        domain = _infer_domain_for_group(group, default=ParallelDomain.CP)
        with _ne_stream_ctx(domain, group):
            torch.distributed.reduce_scatter(output, input_tensor_list, group=group)
    return output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return input_

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _reduce(grad_output, ctx.group), None


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _reduce(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return grad_output, None


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _split_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return _split_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _gather_along_last_dim(grad_output, ctx.group), None


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _split_along_last_dim(grad_output, ctx.group), None


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _split_along_first_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return _split_along_first_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _gather_along_first_dim(grad_output, ctx.group), None


class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate."""

    @staticmethod
    def symbolic(
        graph,
        input_,
        group,
        tensor_parallel_output_grad=True,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        """Symbolic function for tracing."""
        return _gather_along_first_dim(input_, group, output_split_sizes, use_global_buffer)

    @staticmethod
    def forward(
        ctx,
        input_,
        group,
        tensor_parallel_output_grad=True,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        """Forward function."""
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.use_global_buffer = use_global_buffer
        return _gather_along_first_dim(input_, group, output_split_sizes, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce
        # scattered and whereas if the computation is duplicated,
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return (
                _reduce_scatter_along_first_dim(
                    grad_output, ctx.group, ctx.output_split_sizes, ctx.use_global_buffer
                ),
                None,
                None,
                None,
                None,
            )
        else:
            assert ctx.output_split_sizes is None
            return (_split_along_first_dim(grad_output, ctx.group), None, None, None, None)


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group, input_split_sizes=None, use_global_buffer=False):
        """Symbolic function for tracing."""
        return _reduce_scatter_along_first_dim(input_, group, input_split_sizes, use_global_buffer)

    @staticmethod
    def forward(ctx, input_, group, input_split_sizes=None, use_global_buffer=False):
        """Forward function."""
        ctx.group = group
        ctx.input_split_sizes = input_split_sizes
        ctx.use_global_buffer = use_global_buffer
        return _reduce_scatter_along_first_dim(input_, group, input_split_sizes, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        input_split_sizes = ctx.input_split_sizes
        use_global_buffer = ctx.use_global_buffer
        return (
            _gather_along_first_dim(grad_output, ctx.group, input_split_sizes, use_global_buffer),
            None,
            None,
            None,
        )


class _AllGatherFromTensorParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatenate."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return _gather_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _reduce_scatter_along_last_dim(grad_output, ctx.group), None


class _ReduceScatterToTensorParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_, group):
        """Symbolic function for tracing."""
        return _reduce_scatter_along_last_dim(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        """Forward function."""
        ctx.group = group
        return _reduce_scatter_along_last_dim(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward function."""
        return _gather_along_last_dim(grad_output, ctx.group), None


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group, input, output_split_sizes, input_split_sizes, domain):
        """Forward function."""
        if output_split_sizes is not None:
            output_split_sizes = tuple(int(x) for x in output_split_sizes)
        if input_split_sizes is not None:
            input_split_sizes = tuple(int(x) for x in input_split_sizes)

        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.domain = domain

        world_size = group.size()
        if world_size == 1:
            return input

        input = input.contiguous()
        if output_split_sizes is None:
            output = torch.empty_like(input)
            output_chunks = list(output.chunk(world_size, dim=0))
            input_chunks = list(input.chunk(world_size, dim=0))
        else:
            output = input.new_empty(
                size=[sum(output_split_sizes)] + list(input.size()[1:]),
                dtype=input.dtype,
                device=torch.cuda.current_device(),
            )
            output_chunks = list(torch.split(output, output_split_sizes, dim=0))
            input_chunks = list(torch.split(input, input_split_sizes, dim=0))

        ne = get_global_network_engine()
        ne_domain = domain if domain is not None else ParallelDomain.TP
        with _ne_stream_ctx(ne_domain, group):
            torch.distributed.all_to_all(output_chunks, input_chunks, group=group)
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        """Backward function."""
        return (
            None,
            _AllToAll.apply(
                ctx.group,
                *grad_output,
                ctx.input_split_sizes,
                ctx.output_split_sizes,
                ctx.domain,
            ),
            None,
            None,
            None,
        )


# -----------------
# Helper functions.
# -----------------


def copy_to_tensor_model_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: copy, backward allreduce"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _CopyToModelParallelRegion.apply(input_, group)


def reduce_from_tensor_model_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: all reduce, backward copy"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _ReduceFromModelParallelRegion.apply(input_, group)


def scatter_to_tensor_model_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: RS, backward: AG <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _ScatterToModelParallelRegion.apply(input_, group)


def gather_from_tensor_model_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: AG, backward: split <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _GatherFromModelParallelRegion.apply(input_, group)


def scatter_to_sequence_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: split, backward: AG <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _ScatterToSequenceParallelRegion.apply(input_, group)


def gather_from_sequence_parallel_region(
    input_,
    tensor_parallel_output_grad=True,
    group=None,
    output_split_sizes=None,
    use_global_buffer=False,
):
    """Wrapper for autograd function: forward: AG, backward: RS <first dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _GatherFromSequenceParallelRegion.apply(
        input_, group, tensor_parallel_output_grad, output_split_sizes, use_global_buffer
    )


def reduce_scatter_to_sequence_parallel_region(
    input_, group=None, input_split_sizes=None, use_global_buffer=False
):
    """Wrapper for autograd function: forward: RS, backward AG <fisrt dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _ReduceScatterToSequenceParallelRegion.apply(
        input_, group, input_split_sizes, use_global_buffer
    )


def all_gather_last_dim_from_tensor_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: AG, backward RS <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _AllGatherFromTensorParallelRegion.apply(input_, group)


def reduce_scatter_last_dim_to_tensor_parallel_region(input_, group=None):
    """Wrapper for autograd function: forward: RS, backward AG: AG <last dim>"""
    group = get_tensor_model_parallel_group_if_none(group)
    return _ReduceScatterToTensorParallelRegion.apply(input_, group)


def all_to_all(
    group,
    input_,
    output_split_sizes_=None,
    input_split_sizes=None,
    domain=None,
):
    """Wrapper for autograd function"""
    assert group is not None, "group should not be None"
    return _AllToAll.apply(group, input_, output_split_sizes_, input_split_sizes, domain)


def all_to_all_sp2hp(input_, group=None):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape
    [num_tokens/TP, H] to [num_tokens, H/TP].

    Args:
        input_ (torch.Tensor):
            The input tensor which has been distributed along the sequence
            dimension.
        group (torch.distributed.ProcessGroup, optional):
            The process group to work on. If None, the tensor model parallel group
            will be used.

    Returns:
        torch.Tensor: The output tensor with shape [num_tokens, H/TP].

    """
    group = get_tensor_model_parallel_group_if_none(group)

    world_size = group.size()
    input_ = input_.reshape(-1, input_.shape[-1])
    split_tensors = torch.split(
        input_, split_size_or_sections=input_.shape[-1] // world_size, dim=1
    )
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = all_to_all(group, concat_tensor)
    return output


def all_to_all_hp2sp(input_, group=None):
    """
    Perform AlltoAll communication on tensor parallel group, transform the input tensor from shape
    [num_tokens, H/TP] to [num_tokens/TP, H].

    Args:
        input_ (torch.Tensor):
            The input tensor which has been distributed along the hidden
            dimension.
        group (torch.distributed.ProcessGroup, optional):
            The process group to work on. If None, the tensor model parallel group
            will be used.

    Returns:
        torch.Tensor: The output tensor with shape [num_tokens/TP, H].
    """
    group = get_tensor_model_parallel_group_if_none(group)

    world_size = group.size()
    input_ = input_.reshape(-1, input_.shape[-1])
    input_exchanged = all_to_all(group, input_)
    input_reshaped = input_exchanged.reshape(-1, input_exchanged.shape[-1])
    split_tensors = torch.split(
        input_reshaped, split_size_or_sections=input_reshaped.shape[0] // world_size, dim=0
    )
    output = torch.cat(split_tensors, dim=-1)
    return output
