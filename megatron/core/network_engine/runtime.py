from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass

import torch

from .engine import resolve_backend, resolve_backend_for_group, resolve_backend_for_ranks
from .enums import ParallelDomain


_LOGGER = logging.getLogger(__name__)
_ONCE_KEYS: set[tuple[str, str]] = set()


def _log_once(level: str, key: tuple[str, str], message: str) -> None:
    if key in _ONCE_KEYS:
        return
    _ONCE_KEYS.add(key)
    getattr(_LOGGER, level)(message)
    print(message, file=sys.stderr)


@dataclass(frozen=True)
class CommStreamSpec:
    domain: ParallelDomain
    intranode: bool
    backend: str


@dataclass(frozen=True)
class CommStreamHandle:
    spec: CommStreamSpec
    stream: torch.cuda.Stream


class GlobalNetworkEngine:
    """Minimal global NetworkEngine facade for incremental integration.

    This object provides a stable query point for backend and stream selection.
    It does not execute communication or own higher-level schedule policy yet.
    """

    def __init__(self) -> None:
        self._comm_stream_handles: dict[tuple[str, bool], CommStreamHandle] = {}

    def is_stream_routing_enabled(self) -> bool:
        return os.getenv("SCHEDULE_USE_NETWORK_ENGINE_STREAM", "0") == "1"

    def resolve_backend(self, domain, *, intranode: bool):
        return resolve_backend(domain, intranode=intranode)

    def resolve_backend_for_group(self, domain, group):
        return resolve_backend_for_group(domain, group)

    def resolve_backend_for_ranks(self, domain, global_ranks):
        return resolve_backend_for_ranks(domain, global_ranks)

    def _default_intranode(self, domain) -> bool:
        return domain in (ParallelDomain.TP, ParallelDomain.CP, ParallelDomain.EP)

    def _get_default_group_for_domain(self, domain):
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return None

        from megatron.core import parallel_state

        if domain == ParallelDomain.EP:
            return parallel_state.get_expert_model_parallel_group(check_initialized=False)
        if domain == ParallelDomain.CP:
            return parallel_state.get_context_parallel_group(check_initialized=False)
        if domain == ParallelDomain.PP:
            return parallel_state.get_pipeline_model_parallel_group(check_initialized=False)
        if domain == ParallelDomain.TP:
            return parallel_state.get_tensor_model_parallel_group(check_initialized=False)
        if domain == ParallelDomain.DP:
            return parallel_state.get_data_parallel_group(check_initialized=False)
        return None

    def resolve_comm_stream_spec(self, domain, group=None, intranode=None):
        if group is None:
            group = self._get_default_group_for_domain(domain)

        if intranode is None:
            if group is not None:
                decision = self.resolve_backend_for_group(domain, group)
                intranode = decision.intranode
                backend = decision.backend.value
            else:
                intranode = self._default_intranode(domain)
                backend = self.resolve_backend(domain, intranode=intranode).backend.value
        else:
            backend = self.resolve_backend(domain, intranode=intranode).backend.value

        return CommStreamSpec(
            domain=domain,
            intranode=bool(intranode),
            backend=backend,
        )

    def get_comm_stream_handle_for_domain(self, domain, group=None, intranode=None):
        if not self.is_stream_routing_enabled():
            return None
        if not torch.cuda.is_available():
            return None

        spec = self.resolve_comm_stream_spec(domain, group=group, intranode=intranode)
        stream_key = (spec.domain.value, spec.intranode)
        handle = self._comm_stream_handles.get(stream_key)
        if handle is None:
            stream = torch.cuda.Stream(device="cuda")
            handle = CommStreamHandle(spec=spec, stream=stream)
            self._comm_stream_handles[stream_key] = handle
            _log_once(
                "info",
                ("create_stream", f"{spec.domain.value}:{int(spec.intranode)}:{spec.backend}"),
                "[NetworkEngine] create comm stream "
                f"domain={spec.domain.value} intranode={spec.intranode} backend={spec.backend}",
            )
        return handle

    def get_comm_stream_for_domain(self, domain, group=None, intranode=None):
        handle = self.get_comm_stream_handle_for_domain(
            domain=domain,
            group=group,
            intranode=intranode,
        )
        if handle is None:
            return None
        return handle.stream

    def get_comm_stream_or_fallback(
        self,
        domain,
        fallback_stream,
        *,
        group=None,
        intranode=None,
        consumer: str = "unknown",
    ):
        try:
            stream = self.get_comm_stream_for_domain(
                domain=domain,
                group=group,
                intranode=intranode,
            )
            if stream is not None:
                return stream
        except Exception as exc:
            _log_once(
                "warning",
                ("stream_fallback_exc", f"{consumer}:{domain.value}:{type(exc).__name__}"),
                "[NetworkEngine][Fallback] "
                f"{consumer} {domain.value} stream resolve failed; fallback to local stream; "
                f"reason={type(exc).__name__}: {exc}",
            )
        return fallback_stream

    def get_registered_comm_stream_specs(self):
        return tuple(handle.spec for _, handle in sorted(self._comm_stream_handles.items()))


_GLOBAL_NETWORK_ENGINE = GlobalNetworkEngine()


def get_global_network_engine() -> GlobalNetworkEngine:
    return _GLOBAL_NETWORK_ENGINE
