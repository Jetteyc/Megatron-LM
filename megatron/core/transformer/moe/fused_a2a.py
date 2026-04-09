# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Portions of this code are from DeepSeek DeepEP project
# Copyright (c) 2025 DeepSeek
# Licensed under the MIT License - https://github.com/deepseek-ai/DeepEP/blob/main/LICENSE


import sys
from contextlib import contextmanager

try:
    from deep_ep import Buffer
    from deep_ep.utils import EventHandle, EventOverlap

    HAVE_DEEP_EP = True
    print(f"[DeepEP] Successfully imported DeepEP from Megatron-LM-Enhanced", file=sys.stderr)
except ImportError as e:
    HAVE_DEEP_EP = False
    print(f"[DeepEP] Failed to import DeepEP: {e}", file=sys.stderr)

import torch
import os
import time
from functools import lru_cache

from megatron.core.network_engine.topology import get_group_global_ranks, get_local_world_size


@lru_cache(maxsize=1)
def _get_local_rank() -> int:
    return int(os.getenv("LOCAL_RANK", "-1"))


# Module-level cache for DEEPEP_* debug env vars (read once at import time).
_DEEPEP_DUMP_DISPATCH = os.getenv("DEEPEP_DUMP_DISPATCH", "0") == "1"
_DEEPEP_DUMP_ONCE = os.getenv("DEEPEP_DUMP_ONCE", "1") != "0"
_DEEPEP_DUMP_DIR = os.getenv("DEEPEP_DUMP_DIR", "outputs/deepep_runtime_dump")
_DEEPEP_INPUT_CLONE = os.getenv("DEEPEP_INPUT_CLONE", "0") == "1"
_DEEPEP_PRE_DISPATCH_SYNC = os.getenv("DEEPEP_PRE_DISPATCH_SYNC", "0") == "1"
_DEEPEP_MAX_NVL_BYTES = int(os.getenv("DEEPEP_MAX_NVL_BYTES", "0"))

# NOTE:
# DeepEP NVSHMEM runtime is sensitive to repeated init/destroy patterns.
# Keep per-group buffer cache keyed by global-rank tuple to avoid accidental
# re-initialization when ProcessGroup object identity changes.
_buffer = None
_buffer_pool = {}
_group_nvl_cap_bytes = {}
_group_dispatch_cfg = {}
_group_combine_cfg = {}
_dumped_dispatch_keys = set()
_dumped_group_topology = set()


def _deepep_comm_nvtx_enabled() -> bool:
    return os.getenv("NE_NVTX_DISABLE", "0") != "1"


def _deepep_comm_record_function_enabled() -> bool:
    return os.getenv("NE_RECORD_FUNCTION_DISABLE", "0") != "1"


@contextmanager
def _deepep_comm_nvtx(name: str):
    if _deepep_comm_record_function_enabled():
        with torch.profiler.record_function(name):
            yield
    else:
        yield


def _config_to_debug_str(cfg) -> str:
    if cfg is None:
        return "None"
    keys = [
        "num_sms",
        "num_max_nvl_chunked_send_tokens",
        "num_max_nvl_chunked_recv_tokens",
        "num_max_rdma_chunked_send_tokens",
        "num_max_rdma_chunked_recv_tokens",
    ]
    parts = []
    for k in keys:
        parts.append(f"{k}={getattr(cfg, k, '<n/a>')}")
    return ", ".join(parts)


def _debug_tensor_stats(name: str, tensor: torch.Tensor) -> str:
    """Return a compact string of tensor stats for debug logging."""
    if tensor is None:
        return f"{name}=None"
    try:
        with torch.no_grad():
            return (
                f"{name}(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, "
                f"contig={tensor.is_contiguous()}, min={tensor.min().item()}, max={tensor.max().item()}, "
                f"nan={torch.isnan(tensor).any().item()}, inf={torch.isinf(tensor).any().item()})"
            )
    except Exception as exc:
        return f"{name}(shape={tuple(tensor.shape)}, dtype={tensor.dtype}, device={tensor.device}, contig={tensor.is_contiguous()}, stats_error={exc})"


def _maybe_dump_dispatch_case(
    *,
    x: torch.Tensor,
    token_indices: torch.Tensor,
    token_probs: torch.Tensor,
    num_experts: int,
    group: torch.distributed.ProcessGroup,
    num_tokens_per_rank: torch.Tensor,
    num_tokens_per_rdma_rank: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    is_token_in_rank: torch.Tensor,
    buffer,
):
    if not _DEEPEP_DUMP_DISPATCH:
        return

    world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    local_rank = _get_local_rank()
    try:
        group_rank = group.rank()
    except Exception:
        group_rank = -1

    dump_key = (world_rank, local_rank, group_rank)
    if _DEEPEP_DUMP_ONCE and dump_key in _dumped_dispatch_keys:
        return

    os.makedirs(_DEEPEP_DUMP_DIR, exist_ok=True)
    ts = int(time.time() * 1000)
    dump_path = os.path.join(
        _DEEPEP_DUMP_DIR,
        f"dispatch_dump_rank{world_rank}_local{local_rank}_group{group_rank}_{ts}.pt",
    )

    payload = {
        "meta": {
            "world_rank": world_rank,
            "local_rank": local_rank,
            "group_rank": group_rank,
            "group_size": int(group.size()),
            "num_experts": int(num_experts),
            "num_nvl_bytes": int(getattr(buffer, "num_nvl_bytes", -1)),
            "num_rdma_bytes": int(getattr(buffer, "num_rdma_bytes", -1)),
            "hidden_bytes": int(get_hidden_bytes(x)),
            "x_shape": tuple(x.shape),
            "x_dtype": str(x.dtype),
            "token_indices_shape": tuple(token_indices.shape),
            "token_indices_dtype": str(token_indices.dtype),
            "token_probs_shape": tuple(token_probs.shape),
            "token_probs_dtype": str(token_probs.dtype),
        },
        "x": x.detach().cpu(),
        "token_indices": token_indices.detach().cpu(),
        "token_probs": token_probs.detach().cpu(),
        "num_tokens_per_rank": num_tokens_per_rank.detach().cpu() if num_tokens_per_rank is not None else None,
        "num_tokens_per_rdma_rank": num_tokens_per_rdma_rank.detach().cpu() if num_tokens_per_rdma_rank is not None else None,
        "num_tokens_per_expert": num_tokens_per_expert.detach().cpu() if num_tokens_per_expert is not None else None,
        "is_token_in_rank": is_token_in_rank.detach().cpu() if is_token_in_rank is not None else None,
    }
    torch.save(payload, dump_path)
    print(f"[DeepEP] dispatch dump saved: {dump_path}", file=sys.stderr)
    _dumped_dispatch_keys.add(dump_key)


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x (torch.Tensor): Input tensor

    Returns:
        int: Number of hidden bytes
    """
    return x.size(1) * max(x.element_size(), 2)


def _serialize_buffer_init_by_group_leader(
    group: torch.distributed.ProcessGroup, group_ranks_key: tuple[int, ...]
) -> None:
    """Lightweight sync for first-time DeepEP Buffer init inside the current EP group.

    IMPORTANT: do NOT synchronize on WORLD here, because not all ranks necessarily
    enter DeepEP path at the same time, which can deadlock multinode jobs.
    """
    if not torch.distributed.is_initialized():
        return

    if group is None or group.size() <= 1:
        return

    world_rank = torch.distributed.get_rank()
    print(
        f"[DeepEP] serialize-init enter: world_rank={world_rank} "
        f"group_ranks={list(group_ranks_key)}",
        file=sys.stderr,
    )
    with _deepep_comm_nvtx("deepep.group_barrier.serialize_init"):
        torch.distributed.barrier(group=group)
    print(
        f"[DeepEP] serialize-init proceed: world_rank={world_rank} "
        f"group_ranks={list(group_ranks_key)}",
        file=sys.stderr,
    )


def _build_nvl_retry_sizes(initial_nvl_bytes: int) -> list[int]:
    """Build descending NVL buffer sizes for retry on allocation failure."""
    sizes: list[int] = []
    size = int(initial_nvl_bytes)
    # 64MB lower bound for practical communication workspace.
    min_size = 64 * 1024 * 1024
    while size >= min_size:
        if not sizes or sizes[-1] != size:
            sizes.append(size)
        size //= 2
    if not sizes:
        sizes = [max(int(initial_nvl_bytes), min_size)]
    return sizes


def _create_buffer_with_retry(
    group: torch.distributed.ProcessGroup,
    group_ranks_key: tuple[int, ...],
    num_nvl_bytes: int,
    num_rdma_bytes: int,
):
    """Try creating DeepEP buffer; on NVL allocation failure, reduce NVL bytes and retry."""
    world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    local_rank = _get_local_rank()

    retry_sizes = _build_nvl_retry_sizes(num_nvl_bytes)
    last_exc = None
    for attempt, nvl_bytes_try in enumerate(retry_sizes, start=1):
        try:
            print(
                f"[DeepEP] Buffer alloc attempt={attempt} world_rank={world_rank} local_rank={local_rank} "
                f"group_ranks={list(group_ranks_key)} nvl_bytes={nvl_bytes_try} rdma_bytes={num_rdma_bytes}",
                file=sys.stderr,
            )
            buf = Buffer(group, nvl_bytes_try, num_rdma_bytes)
            if nvl_bytes_try < num_nvl_bytes:
                _group_nvl_cap_bytes[group_ranks_key] = nvl_bytes_try
                print(
                    f"[DeepEP] Buffer alloc succeeded with reduced nvl_bytes={nvl_bytes_try} "
                    f"(requested={num_nvl_bytes}) for group_ranks={list(group_ranks_key)}",
                    file=sys.stderr,
                )
            return buf
        except Exception as exc:
            last_exc = exc
            print(
                f"[DeepEP] Buffer alloc failed attempt={attempt} nvl_bytes={nvl_bytes_try} "
                f"rdma_bytes={num_rdma_bytes} err={exc}",
                file=sys.stderr,
            )

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("DeepEP buffer allocation failed with unknown error")


def _scale_nvl_config(cfg, scale: float):
    """Scale NVL chunk limits in DeepEP config while preserving validity constraints."""
    if cfg is None:
        return None
    if scale >= 0.999:
        return cfg

    # NOTE:
    # deep_ep_cpp.Config (pybind) doesn't reliably expose readable attributes in this env.
    # Use stable defaults + current Buffer.num_sms instead of getattr(cfg, ...).
    recv_orig = 32768
    send_orig = 6
    recv_new = max(1024, int(recv_orig * max(scale, 0.01)))
    send_new = min(send_orig, max(1, recv_new // 2))
    if send_new >= recv_new:
        recv_new = send_new + 1

    num_sms = int(getattr(Buffer, "num_sms", 20))
    if num_sms <= 0:
        num_sms = 20
    if num_sms % 2 != 0:
        num_sms -= 1
    if num_sms <= 0:
        num_sms = 2

    cfg_type = type(cfg)
    return cfg_type(
        num_sms,
        int(send_new),
        int(recv_new),
        6,
        32768,
    )


def _get_group_deepep_configs(group: torch.distributed.ProcessGroup):
    """Return per-group dispatch/combine configs aligned with allocated NVL buffer size."""
    group_ranks_key = tuple(get_group_global_ranks(group))
    dispatch_cfg = _group_dispatch_cfg.get(group_ranks_key)
    combine_cfg = _group_combine_cfg.get(group_ranks_key)
    if dispatch_cfg is None:
        dispatch_cfg = Buffer.get_dispatch_config(group.size())
    if combine_cfg is None:
        combine_cfg = Buffer.get_combine_config(group.size())
    return dispatch_cfg, combine_cfg


def get_buffer(group: torch.distributed.ProcessGroup, hidden_bytes: int):
    """Get or create a buffer for all-to-all communication.

    Args:
        group (torch.distributed.ProcessGroup): Process group for communication
        hidden_bytes (int): Number of hidden bytes needed

    Returns:
        Buffer: Communication buffer
    """
    print(f"[DeepEP] get_buffer called: group.size()={group.size()}, hidden_bytes={hidden_bytes}", file=sys.stderr)

    try:
        group_ranks = tuple(get_group_global_ranks(group))
        if group_ranks not in _dumped_group_topology:
            _dumped_group_topology.add(group_ranks)
            lws = get_local_world_size()
            node_ids = [int(r) // lws for r in group_ranks]
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
            local_rank = _get_local_rank()
            print(
                f"[DeepEP] group topology: world_rank={world_rank} local_rank={local_rank} "
                f"world_size={world_size} local_world_size={lws} group_ranks={list(group_ranks)} node_ids={node_ids}",
                file=sys.stderr,
            )
    except Exception as exc:
        print(f"[DeepEP] group topology debug failed: {exc}", file=sys.stderr)
    global _buffer
    global _buffer_pool
    num_nvl_bytes, num_rdma_bytes = 0, 0
    dispatch_cfg = Buffer.get_dispatch_config(group.size())
    combine_cfg = Buffer.get_combine_config(group.size())

    print(
        f"[DeepEP] group.size()={group.size()} dispatch_cfg=({_config_to_debug_str(dispatch_cfg)}) "
        f"combine_cfg=({_config_to_debug_str(combine_cfg)})",
        file=sys.stderr,
    )

    for config in (dispatch_cfg, combine_cfg):
        # Split long line for PEP8 compliance
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTES: the adaptive routing configuration of the network **must be off**
    group_ranks_key = tuple(get_group_global_ranks(group))

    # Optional user cap for intranode NVSHMEM workspace.
    if _DEEPEP_MAX_NVL_BYTES > 0:
        num_nvl_bytes = min(num_nvl_bytes, _DEEPEP_MAX_NVL_BYTES)

    # Sticky per-group cap discovered by previous successful retry.
    prior_cap = _group_nvl_cap_bytes.get(group_ranks_key)
    if prior_cap is not None and prior_cap > 0:
        num_nvl_bytes = min(num_nvl_bytes, int(prior_cap))

    cached = _buffer_pool.get(group_ranks_key)
    need_realloc = (
        cached is None
        or cached.num_nvl_bytes < num_nvl_bytes
        or cached.num_rdma_bytes < num_rdma_bytes
    )

    if need_realloc:
        # Serialize deep_ep.Buffer creation to avoid multinode init races.
        _serialize_buffer_init_by_group_leader(group, group_ranks_key)

        world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        local_rank = _get_local_rank()
        print(
            f"[DeepEP] Allocating new Buffer: world_rank={world_rank} local_rank={local_rank} "
            f"group_ranks={list(group_ranks_key)} "
            f"nvl_bytes={num_nvl_bytes}, rdma_bytes={num_rdma_bytes}",
            file=sys.stderr,
        )
        cached = _create_buffer_with_retry(
            group=group,
            group_ranks_key=group_ranks_key,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
        )

        scale = float(cached.num_nvl_bytes) / float(max(num_nvl_bytes, 1))
        _group_dispatch_cfg[group_ranks_key] = _scale_nvl_config(dispatch_cfg, scale)
        _group_combine_cfg[group_ranks_key] = _scale_nvl_config(combine_cfg, scale)

        _buffer_pool[group_ranks_key] = cached
    else:
        world_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        local_rank = _get_local_rank()
        print(
            f"[DeepEP] Reusing Buffer: world_rank={world_rank} local_rank={local_rank} "
            f"group_ranks={list(group_ranks_key)} "
            f"nvl_bytes={cached.num_nvl_bytes}, rdma_bytes={cached.num_rdma_bytes}",
            file=sys.stderr,
        )

        # Ensure configs exist for reused buffers.
        if group_ranks_key not in _group_dispatch_cfg:
            _group_dispatch_cfg[group_ranks_key] = dispatch_cfg
        if group_ranks_key not in _group_combine_cfg:
            _group_combine_cfg[group_ranks_key] = combine_cfg

    # Keep backward compatibility for any debug path referencing _buffer.
    _buffer = cached
    return cached


class FusedDispatch(torch.autograd.Function):
    """Fused dispatch operation for MoE routing combining computation and communication."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Forward pass of fused dispatch."""
        print(
            f"[DeepEP] FusedDispatch.forward() called: x.shape={x.shape}, token_indices.shape={token_indices.shape}, "
            f"token_probs.shape={token_probs.shape}, num_experts={num_experts}, async_finish={async_finish}, "
            f"allocate_on_comm_stream={allocate_on_comm_stream}, group.size()={group.size()}",
            file=sys.stderr,
        )
        print(
            f"[DeepEP] { _debug_tensor_stats('x', x) }; { _debug_tensor_stats('token_indices', token_indices) }; "
            f"{ _debug_tensor_stats('token_probs', token_probs) }",
            file=sys.stderr,
        )
        try:
            with torch.no_grad():
                max_idx = token_indices.max().item()
                min_idx = token_indices.min().item()
                invalid_low = (token_indices < -1).any().item()
                invalid_high = (token_indices >= num_experts).any().item()
                print(
                    f"[DeepEP] token_indices range: min={min_idx}, max={max_idx}, "
                    f"invalid_low(<-1)={invalid_low}, invalid_high(>=num_experts)={invalid_high}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"[DeepEP] token_indices range check failed: {exc}", file=sys.stderr)

        # Optional hard isolation for runtime debugging:
        # - clone inputs to fresh storage to rule out aliasing/upstream memory corruption
        # - synchronize before dispatch to surface earlier async kernel errors
        if _DEEPEP_INPUT_CLONE:
            print("[DeepEP] DEEPEP_INPUT_CLONE=1: cloning x/token_indices/token_probs", file=sys.stderr)
            x = x.contiguous().clone()
            token_indices = token_indices.contiguous().clone()
            token_probs = token_probs.contiguous().clone()

        if _DEEPEP_PRE_DISPATCH_SYNC:
            print("[DeepEP] DEEPEP_PRE_DISPATCH_SYNC=1: torch.cuda.synchronize()", file=sys.stderr)
            torch.cuda.synchronize()

        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        # Calculate layout before actual dispatch
        buffer = get_buffer(group, get_hidden_bytes(x))
        dispatch_cfg, _ = _get_group_deepep_configs(group)
        print(f"[DeepEP] Calling buffer.get_dispatch_layout()", file=sys.stderr)
        with _deepep_comm_nvtx("deepep.get_dispatch_layout"):
            print("------------------------------------------------------", file=sys.stderr)
            (
                num_tokens_per_rank,
                num_tokens_per_rdma_rank,
                num_tokens_per_expert,
                is_token_in_rank,
                event,
            ) = buffer.get_dispatch_layout(
                token_indices,
                num_experts,
                previous_event=previous_event,
                async_finish=async_finish,
                allocate_on_comm_stream=allocate_on_comm_stream,
            )

        try:
            rank = group.rank()
        except Exception:
            rank = -1
        try:
            print(
                f"[DeepEP] rank={rank} get_dispatch_layout: "
                f"num_tokens_per_rank.shape={tuple(num_tokens_per_rank.shape)} "
                f"sum={int(num_tokens_per_rank.sum().item())} "
                f"num_tokens_per_rank={num_tokens_per_rank.tolist()}",
                file=sys.stderr,
            )
            if num_tokens_per_expert is not None:
                print(
                    f"[DeepEP] rank={rank} get_dispatch_layout: "
                    f"num_tokens_per_expert.shape={tuple(num_tokens_per_expert.shape)} "
                    f"sum={int(num_tokens_per_expert.sum().item())}",
                    file=sys.stderr,
                )
            if is_token_in_rank is not None:
                print(
                    f"[DeepEP] rank={rank} get_dispatch_layout: "
                    f"is_token_in_rank.shape={tuple(is_token_in_rank.shape)} dtype={is_token_in_rank.dtype}",
                    file=sys.stderr,
                )
        except Exception as exc:
            print(f"[DeepEP] get_dispatch_layout debug failed: {exc}", file=sys.stderr)

        _maybe_dump_dispatch_case(
            x=x,
            token_indices=token_indices,
            token_probs=token_probs,
            num_experts=num_experts,
            group=group,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            is_token_in_rank=is_token_in_rank,
            buffer=buffer,
        )

        # Do MoE dispatch
        # NOTES: the CPU will wait for GPU's signal to arrive,
        # so this is not compatible with CUDA graph
        (
            recv_x,
            recv_token_indices,
            recv_token_probs,
            num_recv_tokens_per_expert_list,
            handle,
            after_event_overlap,
        ) = buffer.dispatch(
            x,
            topk_idx=token_indices,
            topk_weights=token_probs,  # DeepEP only supports float32 probs
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=event,  # wait in deepep::intra/inter_dispatch
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )

        # Make sure current stream is synchronized
        if async_finish:
            after_event_overlap.current_stream_wait()

        # Save for backward
        ctx.group = group
        ctx.handle = handle
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        tokens_per_expert = torch.tensor(num_recv_tokens_per_expert_list)

        print(f"[DeepEP] FusedDispatch.forward() completed: recv_x.shape={recv_x.shape}, tokens_per_expert={tokens_per_expert}", file=sys.stderr)
        return (recv_x, recv_token_indices, recv_token_probs, tokens_per_expert, handle)

    @staticmethod
    def backward(
        ctx, grad_output, grad_token_indices, grad_token_probs, grad_tokens_per_expert, grad_handle
    ):
        """Backward pass of fused dispatch."""
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        handle = ctx.handle
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        grad_x, grad_token_probs, after_event = buffer.combine(
            grad_output.contiguous(),
            handle,
            topk_weights=grad_token_probs.float(),
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, grad_token_probs, None, None, None, None


class FusedCombine(torch.autograd.Function):
    """Fused combine operation for MoE output combining computation and communication."""

    @staticmethod
    def forward(ctx, x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Forward pass of fused combine."""
        previous_event = None
        if async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(group, get_hidden_bytes(x))
        combined_x, _, after_event = buffer.combine(
            x,
            handle=handle,
            async_finish=async_finish,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if async_finish:
            after_event.current_stream_wait()

        ctx.handle = handle
        ctx.group = group
        ctx.async_finish = async_finish
        ctx.allocate_on_comm_stream = allocate_on_comm_stream
        return combined_x, None

    @staticmethod
    def backward(ctx, grad_output, previous_event=None):
        """Backward pass of fused combine."""
        previous_event = None
        if ctx.async_finish:
            previous_event = EventOverlap(EventHandle())
        buffer = get_buffer(ctx.group, get_hidden_bytes(grad_output))
        grad_x, _, _, _, _, after_event = buffer.dispatch(
            grad_output.contiguous(),
            handle=ctx.handle,
            previous_event=previous_event,
            async_finish=ctx.async_finish,
            allocate_on_comm_stream=ctx.allocate_on_comm_stream,
        )
        # Make sure current stream is synchronized
        if ctx.async_finish:
            after_event.current_stream_wait()
        return grad_x, None, None, None, None


if HAVE_DEEP_EP:
    print(f"[DeepEP] HAVE_DEEP_EP=True, registering fused_dispatch function", file=sys.stderr)

    def fused_dispatch(
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        async_finish=False,
        allocate_on_comm_stream=False,
    ):
        """Perform fused dispatch operation if deep_ep is available.

        Args:
            x: Input tensor [num_tokens, hidden_size]
            token_indices: Token routing indices [num_tokens, topk]
            token_probs: Token routing probabilities [num_tokens, topk]
            num_experts: Number of experts
            group: Process group
            previous_event: Previous CUDA event

        Returns:
            Result of FusedDispatch
        """
        print(f"[DeepEP] fused_dispatch() wrapper called - invoking FusedDispatch.apply()", file=sys.stderr)
        return FusedDispatch.apply(
            x.contiguous(),
            token_indices,
            token_probs,
            num_experts,
            group,
            async_finish,
            allocate_on_comm_stream,
        )

    def fused_combine(x, group, handle, async_finish=False, allocate_on_comm_stream=False):
        """Perform fused combine operation if deep_ep is available.

        Args:
            x: Input tensor
            group: Process group
            handle: Communication handle
            previous_event: Previous CUDA event

        Returns:
            Result of FusedCombine
        """
        return FusedCombine.apply(x, group, handle, async_finish, allocate_on_comm_stream)

    def set_deepep_num_sms(num_sms):
        """Sets the number of SMs to use for DeepEP"""
        Buffer.set_num_sms(num_sms)

else:
    print(f"[DeepEP] HAVE_DEEP_EP=False, fused_dispatch functions set to None", file=sys.stderr)
    fused_dispatch = None
    fused_combine = None
    set_deepep_num_sms = None
