import logging
import os
import sys
from dataclasses import dataclass
from typing import Iterable

from .enums import CommBackend, ParallelDomain
from .topology import are_ranks_in_single_node, get_local_world_size


_LOGGER = logging.getLogger(__name__)
_ONCE_KEYS: set[tuple[str, str]] = set()


def _log_once(level: str, key: tuple[str, str], message: str) -> None:
    if key in _ONCE_KEYS:
        return
    _ONCE_KEYS.add(key)
    getattr(_LOGGER, level)(message)
    print(message, file=sys.stderr)


@dataclass(frozen=True)
class BackendDecision:
    backend: CommBackend
    intranode: bool


def _parse_backend_env(env_name: str, default: CommBackend) -> CommBackend:
    raw = os.getenv(env_name, "").strip().lower()
    if not raw:
        return default
    parsed = CommBackend.parse(raw, default)
    if parsed == default and raw != default.value:
        _log_once(
            "warning",
            ("invalid_env", f"{env_name}={raw}"),
            f"[NetworkEngine][Fallback] invalid backend env {env_name}={raw}, fallback to {default.value}",
        )
    return parsed


def _resolve_backend(domain: ParallelDomain, intranode: bool) -> CommBackend:
    if domain == ParallelDomain.TP:
        backend = _parse_backend_env("TP_INTRANODE_BACKEND", CommBackend.TORCH_DIST)
        if backend != CommBackend.TORCH_DIST:
            _log_once(
                "warning",
                ("tp_fixed", backend.value),
                "[NetworkEngine][Fallback] TP only supports torch_dist for now, fallback applied",
            )
        return CommBackend.TORCH_DIST

    if domain == ParallelDomain.CP:
        env_name = "CP_INTRANODE_BACKEND" if intranode else "CP_INTERNODE_BACKEND"
        backend = _parse_backend_env(env_name, CommBackend.TORCH_DIST)
        if not intranode and backend != CommBackend.TORCH_DIST:
            _log_once(
                "warning",
                ("cp_internode_fixed", backend.value),
                "[NetworkEngine][Fallback] CP internode backend is not enabled yet, fallback to torch_dist",
            )
            return CommBackend.TORCH_DIST
        return backend

    if domain == ParallelDomain.EP:
        env_name = "EP_INTRANODE_BACKEND" if intranode else "EP_INTERNODE_BACKEND"
        backend = _parse_backend_env(env_name, CommBackend.TORCH_DIST)
        if not intranode and backend != CommBackend.TORCH_DIST:
            _log_once(
                "warning",
                ("ep_internode_fixed", backend.value),
                "[NetworkEngine][Fallback] EP internode accelerator backend is not enabled yet, fallback to torch_dist",
            )
            return CommBackend.TORCH_DIST
        return backend

    if domain in (ParallelDomain.PP, ParallelDomain.DP):
        env_name = "PP_INTERNODE_BACKEND" if domain == ParallelDomain.PP else "DP_INTERNODE_BACKEND"
        backend = _parse_backend_env(env_name, CommBackend.TORCH_DIST)
        if backend != CommBackend.TORCH_DIST:
            _log_once(
                "warning",
                (f"{domain.value}_fixed", backend.value),
                f"[NetworkEngine][Fallback] {domain.value.upper()} only supports torch_dist for now, fallback applied",
            )
        return CommBackend.TORCH_DIST

    return CommBackend.TORCH_DIST


def resolve_backend(domain: ParallelDomain, *, intranode: bool) -> BackendDecision:
    backend = _resolve_backend(domain, intranode)
    _log_once(
        "info",
        ("resolve", f"{domain.value}:{backend.value}:{int(intranode)}"),
        f"[NetworkEngine] resolve domain={domain.value} intranode={intranode} backend={backend.value}",
    )
    return BackendDecision(backend=backend, intranode=intranode)


def resolve_cp_backend_name_for_ranks(cp_global_ranks: Iterable[int]) -> str:
    intranode = are_ranks_in_single_node(cp_global_ranks, get_local_world_size())
    return resolve_backend(ParallelDomain.CP, intranode=intranode).backend.value


def resolve_backend_for_ranks(domain: ParallelDomain, global_ranks: Iterable[int]) -> BackendDecision:
    intranode = are_ranks_in_single_node(global_ranks, get_local_world_size())
    return resolve_backend(domain, intranode=intranode)
