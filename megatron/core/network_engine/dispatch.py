from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.distributed as dist

from .engine import resolve_backend, resolve_backend_for_group, resolve_backend_for_ranks
from .enums import CommBackend, ParallelDomain
def _resolve_torch_dist_backend(
    domain: ParallelDomain,
    *,
    group: Optional[dist.ProcessGroup] = None,
    ranks: Optional[Iterable[int]] = None,
    intranode: Optional[bool] = None,
) -> None:
    if intranode is not None:
        decision = resolve_backend(domain, intranode=intranode)
    elif ranks is not None:
        decision = resolve_backend_for_ranks(domain, ranks)
    elif group is not None:
        decision = resolve_backend_for_group(domain, group)
    else:
        decision = resolve_backend(
            domain,
            intranode=(domain in (ParallelDomain.TP, ParallelDomain.CP, ParallelDomain.EP)),
        )

    if decision.backend != CommBackend.TORCH_DIST:
        raise RuntimeError(
            f"NetworkEngine routed {domain.value} communication to {decision.backend.value}, "
            "but this call path only supports torch_dist"
        )


def all_reduce(
    tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    domain: ParallelDomain,
    async_op: bool = False,
):
    _resolve_torch_dist_backend(domain, group=group)
    return dist.all_reduce(tensor, group=group, async_op=async_op)


def all_gather_into_tensor(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    domain: ParallelDomain,
    async_op: bool = False,
):
    _resolve_torch_dist_backend(domain, group=group)
    if hasattr(dist, "all_gather_into_tensor"):
        return dist.all_gather_into_tensor(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )
    return dist._all_gather_base(output_tensor, input_tensor, group=group, async_op=async_op)


def all_gather(
    output_tensors,
    input_tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    domain: ParallelDomain,
) -> None:
    _resolve_torch_dist_backend(domain, group=group)
    dist.all_gather(output_tensors, input_tensor, group=group)


def reduce_scatter_tensor(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    domain: ParallelDomain,
    async_op: bool = False,
):
    _resolve_torch_dist_backend(domain, group=group)
    if hasattr(dist, "reduce_scatter_tensor"):
        return dist.reduce_scatter_tensor(
            output_tensor,
            input_tensor,
            group=group,
            async_op=async_op,
        )
    return dist._reduce_scatter_base(output_tensor, input_tensor, group=group, async_op=async_op)


def reduce_scatter(
    output_tensor: torch.Tensor,
    input_tensors,
    *,
    group: dist.ProcessGroup,
    domain: ParallelDomain,
) -> None:
    _resolve_torch_dist_backend(domain, group=group)
    dist.reduce_scatter(output_tensor, input_tensors, group=group)


def all_to_all_single(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup,
    domain: ParallelDomain,
    output_split_sizes=None,
    input_split_sizes=None,
    async_op: bool = False,
):
    _resolve_torch_dist_backend(domain, group=group)
    return dist.all_to_all_single(
        output_tensor,
        input_tensor,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=async_op,
    )


def isend(
    tensor: torch.Tensor,
    *,
    dst: int,
    group: dist.ProcessGroup,
    domain: ParallelDomain,
    intranode: Optional[bool] = None,
):
    _resolve_torch_dist_backend(domain, group=group, intranode=intranode)
    return dist.isend(tensor=tensor, dst=dst, group=group)


def irecv(
    tensor: torch.Tensor,
    *,
    src: int,
    group: dist.ProcessGroup,
    domain: ParallelDomain,
    intranode: Optional[bool] = None,
):
    _resolve_torch_dist_backend(domain, group=group, intranode=intranode)
    return dist.irecv(tensor=tensor, src=src, group=group)


def batch_isend_irecv(
    ops,
    *,
    domain: ParallelDomain,
    group: Optional[dist.ProcessGroup] = None,
    intranode: Optional[bool] = None,
):
    _resolve_torch_dist_backend(domain, group=group, intranode=intranode)
    return dist.batch_isend_irecv(ops)


def ring_exchange(
    *,
    domain: ParallelDomain,
    group: dist.ProcessGroup,
    tensor_send_prev=None,
    tensor_recv_prev=None,
    tensor_send_next=None,
    tensor_recv_next=None,
):
    _resolve_torch_dist_backend(domain, group=group, intranode=False)
    return dist.ring_exchange(
        tensor_send_prev=tensor_send_prev,
        tensor_recv_prev=tensor_recv_prev,
        tensor_send_next=tensor_send_next,
        tensor_recv_next=tensor_recv_next,
        group=group,
    )


def p2p_op(
    op,
    tensor: torch.Tensor,
    peer: int,
    *,
    group: dist.ProcessGroup,
    domain: ParallelDomain,
    intranode: Optional[bool] = None,
):
    _resolve_torch_dist_backend(domain, group=group, intranode=intranode)
    return dist.P2POp(op, tensor, peer, group)
