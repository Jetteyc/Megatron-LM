# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NetworkEngine v1 – pure resource router.
# Provides ONLY stream acquisition and backend/traffic resolution.
# All actual communication calls stay in the original Megatron code paths.

from __future__ import annotations

import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import torch

from .backends import DeepEPBackend, NvshmemBackend, TorchDistributedBackend
from .enums import CommBackend, ParallelDomain, TrafficClass, traffic_for_backend
from .topology import get_local_world_size, is_group_intranode, is_intranode_rank_pair


_INTER_NODE_ACCEL_FALLBACK_LOGGED: set[tuple[str, str]] = set()
_BACKEND_UNAVAILABLE_FALLBACK_LOGGED: set[str] = set()
_INFER_INTRNODE_FALLBACK_LOGGED = False
_CP_RANKS_INFER_FALLBACK_LOGGED = False
_INVALID_BACKEND_ENV_FALLBACK_LOGGED: set[tuple[str, str]] = set()
_TRAFFIC_DOWNGRADE_FALLBACK_LOGGED: set[tuple[str, str, bool]] = set()
_ROUTE_PLAN_MISS_LOGGED: set[tuple[str, bool]] = set()
_GET_STREAM_SCOPE_FALLBACK_LOGGED = False
_STREAM_INIT_FALLBACK_LOGGED = False
_POLICY_FIXED_BACKEND_FALLBACK_LOGGED: set[tuple[str, str]] = set()
_DOMAIN_SCOPE_FALLBACK_LOGGED: set[str] = set()
_P2P_STREAM_SELECT_LOG_COUNT = 0
_P2P_STREAM_SELECT_LOG_LIMIT = 128



@dataclass
class BackendDecision:
    backend: CommBackend
    traffic: TrafficClass


@dataclass
class NetworkEnginePolicy:
    """Backend + stream routing policy for 5D parallel communication.

    v0 policy summary:
    - TP: intranode only, use torch.distributed P2P path directly
    - CP: intranode prefer NVSHMEM, internode fallback torch.dist
    - EP: intranode prefer DeepEP, internode fallback torch.dist
    - PP/DP: internode torch.dist (NCCL)
    """

    local_world_size: int | None = None

    def __post_init__(self) -> None:
        if self.local_world_size is None:
            self.local_world_size = get_local_world_size()

    @staticmethod
    def _parse_backend(
        name: str,
        default: CommBackend,
    ) -> CommBackend:
        value = os.getenv(name, "").strip().lower()
        if not value:
            return default
        print("-----backend env", name, "=", value, file=sys.stderr)
        parsed = CommBackend.parse(value, default)
        if parsed == default and value != default.value:
            key = (name, value)
            if key not in _INVALID_BACKEND_ENV_FALLBACK_LOGGED:
                _INVALID_BACKEND_ENV_FALLBACK_LOGGED.add(key)
                msg = (
                    "[NetworkEngine][Fallback] invalid backend env value "
                    f"{name}={value}; fallback to {default.value}"
                )
                logging.getLogger(__name__).warning(msg)
                print(msg, file=sys.stderr)
        return parsed

    def is_intranode_rank_pair(self, rank_a: int, rank_b: int) -> bool:
        return is_intranode_rank_pair(
            rank_a,
            rank_b,
            local_world_size=self.local_world_size,
        )

    def decide(
        self,
        domain: ParallelDomain,
        *,
        intranode: bool,
    ) -> BackendDecision:
        if domain == ParallelDomain.TP:
            backend = self._parse_backend("TP_INTRANODE_BACKEND", CommBackend.TORCH_DIST)
            if backend != CommBackend.TORCH_DIST:
                key = (domain.value, backend.value)
                if key not in _POLICY_FIXED_BACKEND_FALLBACK_LOGGED:
                    _POLICY_FIXED_BACKEND_FALLBACK_LOGGED.add(key)
                    msg = (
                        "[NetworkEngine][Policy] TP backend is fixed to torch_dist; "
                        f"fallback from {backend.value} to torch_dist"
                    )
                    logging.getLogger(__name__).warning(msg)
                    print(msg, file=sys.stderr)
                backend = CommBackend.TORCH_DIST
            return BackendDecision(backend, TrafficClass.INTRANODE)

        if domain == ParallelDomain.CP:
            if intranode:
                backend = self._parse_backend("CP_INTRANODE_BACKEND", CommBackend.TORCH_DIST)
                return BackendDecision(backend, TrafficClass.INTRANODE)
            backend = self._parse_backend("CP_INTERNODE_BACKEND", CommBackend.TORCH_DIST)
            traffic = traffic_for_backend(
                backend,
                torch_dist_traffic=TrafficClass.INTERNODE,
                # Current phase does not use all_bandwidth streams.
                accel_traffic=TrafficClass.INTERNODE,
            )
            return BackendDecision(backend, traffic)

        if domain == ParallelDomain.EP:
            if intranode:
                backend = self._parse_backend("EP_INTRANODE_BACKEND", CommBackend.TORCH_DIST)
                # DeepEP intranode uses NVSHMEM over NVLink/NVSwitch only,
                # so it belongs to the INTRANODE traffic class (not ALL_BANDWIDTH
                # which is reserved for cross-node accelerators using NVLink+RDMA).
                traffic = traffic_for_backend(
                    backend,
                    torch_dist_traffic=TrafficClass.INTRANODE,
                    accel_traffic=TrafficClass.INTRANODE,
                )
                return BackendDecision(backend, traffic)
            backend = self._parse_backend("EP_INTERNODE_BACKEND", CommBackend.TORCH_DIST)
            if backend != CommBackend.TORCH_DIST:
                key = (domain.value, backend.value)
                if key not in _INTER_NODE_ACCEL_FALLBACK_LOGGED:
                    _INTER_NODE_ACCEL_FALLBACK_LOGGED.add(key)
                    msg = (
                        "[NetworkEngine][Policy] EP internode accelerator backend "
                        f"{backend.value} is not enabled yet; fallback to torch_dist"
                    )
                    logging.getLogger(__name__).warning(msg)
                    print(msg, file=sys.stderr)
                backend = CommBackend.TORCH_DIST
            traffic = traffic_for_backend(backend, torch_dist_traffic=TrafficClass.INTERNODE)
            return BackendDecision(backend, traffic)

        if domain in (ParallelDomain.PP, ParallelDomain.DP):
            key = "PP_INTERNODE_BACKEND" if domain == ParallelDomain.PP else "DP_INTERNODE_BACKEND"
            backend = self._parse_backend(key, CommBackend.TORCH_DIST)
            if backend != CommBackend.TORCH_DIST:
                policy_key = (domain.value, backend.value)
                if policy_key not in _POLICY_FIXED_BACKEND_FALLBACK_LOGGED:
                    _POLICY_FIXED_BACKEND_FALLBACK_LOGGED.add(policy_key)
                    msg = (
                        f"[NetworkEngine][Policy] {domain.value.upper()} backend is fixed to "
                        "torch_dist; fallback from "
                        f"{backend.value} to torch_dist"
                    )
                    logging.getLogger(__name__).warning(msg)
                    print(msg, file=sys.stderr)
                backend = CommBackend.TORCH_DIST
            traffic = traffic_for_backend(backend, torch_dist_traffic=TrafficClass.INTERNODE)
            return BackendDecision(backend, traffic)

        return BackendDecision(CommBackend.TORCH_DIST, TrafficClass.INTERNODE)



class NetworkStreamManager:
    """Manage dedicated CUDA streams for communication traffic isolation.

    Two levels of stream isolation are provided:

    1. **Per-traffic-class streams** (legacy) – one stream for each of the
       three :class:`TrafficClass` values (INTRANODE / INTERNODE /
       ALL_BANDWIDTH).  These are kept for backward compatibility and for
       callers that only know a traffic class.

     2. **Per-domain streams** – one stream for each
         :class:`ParallelDomain` + scope pair.  Scope is intranode/internode.
         Domains that can touch both scopes (CP/EP) get two distinct streams.

    :meth:`get_domain_stream` returns the per-domain(+scope) stream, and
    :meth:`get_stream` returns the per-traffic-class stream.
    """

    def __init__(self, config: 'NetworkEngineConfig'):
        self._config = config
        self._streams: Dict[TrafficClass, Optional[torch.cuda.Stream]] = {
            TrafficClass.INTRANODE: None,
            TrafficClass.INTERNODE: None,
            TrafficClass.ALL_BANDWIDTH: None,
        }
        # Per-domain streams are keyed by (domain, intranode).
        # CP/EP own two streams: intranode + internode.
        # TP owns intranode stream only; PP/DP own internode stream only.
        self._domain_streams: Dict[tuple[ParallelDomain, bool], Optional[torch.cuda.Stream]] = {}
        self._debug_emitted = False
        self._domain_debug_emitted = False

    def _debug(self, message: str) -> None:
        full = f"[NetworkEngine][Stream] {message}"
        logging.getLogger(__name__).info(full)
        print(full, file=sys.stderr)

    _TRAFFIC_PRIORITY_ATTR = {
        TrafficClass.INTRANODE: "stream_intranode_priority",
        TrafficClass.INTERNODE: "stream_internode_priority",
        TrafficClass.ALL_BANDWIDTH: "stream_all_bandwidth_priority",
    }

    _TRAFFIC_STREAM_LABEL = {
        TrafficClass.INTRANODE: "NE_intranode_comm_stream",
        TrafficClass.INTERNODE: "NE_internode_comm_stream",
        TrafficClass.ALL_BANDWIDTH: "NE_all_bandwidth_comm_stream",
    }

    _DOMAIN_STREAM_LABEL = {
        (ParallelDomain.TP, True): "NE_tp_intranode_comm_stream",
        (ParallelDomain.CP, True): "NE_cp_intranode_comm_stream",
        (ParallelDomain.CP, False): "NE_cp_internode_comm_stream",
        (ParallelDomain.EP, True): "NE_ep_intranode_comm_stream",
        (ParallelDomain.EP, False): "NE_ep_internode_comm_stream",
        (ParallelDomain.PP, False): "NE_pp_internode_comm_stream",
        (ParallelDomain.DP, False): "NE_dp_internode_comm_stream",
    }

    _DOMAIN_SCOPE_MATRIX = {
        ParallelDomain.TP: (True,),
        ParallelDomain.CP: (True, False),
        ParallelDomain.EP: (True, False),
        ParallelDomain.PP: (False,),
        ParallelDomain.DP: (False,),
    }

    # Map each (domain, scope) to the config priority attribute it should inherit.
    _DOMAIN_PRIORITY_ATTR = {
        (ParallelDomain.TP, True): "stream_intranode_priority",
        (ParallelDomain.CP, True): "stream_intranode_priority",
        (ParallelDomain.CP, False): "stream_internode_priority",
        (ParallelDomain.EP, True): "stream_intranode_priority",
        (ParallelDomain.EP, False): "stream_internode_priority",
        (ParallelDomain.PP, False): "stream_internode_priority",
        (ParallelDomain.DP, False): "stream_internode_priority",
    }

    @staticmethod
    def _stream_labeling_enabled() -> bool:
        return os.getenv("NE_STREAM_LABEL_DISABLE", "1") != "1"

    def _lazy_create(self) -> None:
        if not torch.cuda.is_available():
            if not self._debug_emitted:
                self._debug("cuda not available, stream isolation disabled")
                self._debug_emitted = True
            return
        # Legacy per-traffic-class streams
        for tc, attr in self._TRAFFIC_PRIORITY_ATTR.items():
            if self._streams[tc] is None:
                s = torch.cuda.Stream(priority=getattr(self._config, attr))
                label = self._TRAFFIC_STREAM_LABEL.get(tc, f"NE_{tc.value}_stream")
                if self._stream_labeling_enabled():
                    try:
                        torch.cuda.set_stream_add_label(s, label)
                    except AttributeError:
                        self._debug(f"torch.cuda.set_stream_add_label not available, skip labeling {label}")
                self._streams[tc] = s
        if not self._debug_emitted:
            self._debug(
                "created intranode/internode/all_bandwidth streams with priorities "
                f"{self._config.stream_intranode_priority}/{self._config.stream_internode_priority}/"
                f"{self._config.stream_all_bandwidth_priority}"
            )
            self._debug_emitted = True

    def _lazy_create_domain_streams(self) -> None:
        """Create per-domain CUDA streams on first access."""
        if not torch.cuda.is_available():
            return
        for domain in ParallelDomain:
            scopes = self._DOMAIN_SCOPE_MATRIX.get(domain, (True,))
            for intranode in scopes:
                key = (domain, intranode)
                if key in self._domain_streams and self._domain_streams[key] is not None:
                    continue
                attr = self._DOMAIN_PRIORITY_ATTR.get(
                    key,
                    "stream_intranode_priority" if intranode else "stream_internode_priority",
                )
                priority = getattr(self._config, attr)
                s = torch.cuda.Stream(priority=priority)
                label = self._DOMAIN_STREAM_LABEL.get(
                    key,
                    f"NE_{domain.value}_{'intranode' if intranode else 'internode'}_comm_stream",
                )
                if self._stream_labeling_enabled():
                    try:
                        torch.cuda.set_stream_add_label(s, label)
                    except AttributeError:
                        pass
                self._domain_streams[key] = s
        if not self._domain_debug_emitted:
            self._domain_debug_emitted = True
            parts = []
            for d in ParallelDomain:
                scopes = self._DOMAIN_SCOPE_MATRIX.get(d, (True,))
                for intranode in scopes:
                    key = (d, intranode)
                    parts.append(
                        f"{d.value}.{ 'intranode' if intranode else 'internode' }={self._domain_streams.get(key)}"
                    )
            self._debug("created per-domain streams: " + ", ".join(parts))

    def get_stream(self, traffic: TrafficClass) -> Optional[torch.cuda.Stream]:
        """Get the legacy per-traffic-class stream."""
        self._lazy_create()
        return self._streams[traffic]

    def get_domain_stream(
        self,
        domain: ParallelDomain,
        *,
        intranode: Optional[bool] = None,
    ) -> Optional[torch.cuda.Stream]:
        """Get the per-domain(+scope) CUDA stream.

        CP/EP own both intranode and internode streams. TP owns intranode
        only. PP/DP own internode only.
        """
        self._lazy_create_domain_streams()
        scopes = self._DOMAIN_SCOPE_MATRIX.get(domain, (True,))
        if intranode is None:
            if len(scopes) == 1:
                intranode = scopes[0]
            else:
                # Ambiguous scope, default to internode for safety and log.
                intranode = False
                domain_key = domain.value
                if domain_key not in _DOMAIN_SCOPE_FALLBACK_LOGGED:
                    _DOMAIN_SCOPE_FALLBACK_LOGGED.add(domain_key)
                    msg = (
                        "[NetworkEngine][Fallback] ambiguous domain stream scope; "
                        f"domain={domain.value} default to internode stream"
                    )
                    logging.getLogger(__name__).warning(msg)
                    print(msg, file=sys.stderr)
        key = (domain, bool(intranode))
        return self._domain_streams.get(key)

    @contextmanager
    def use(self, traffic: TrafficClass) -> Iterator[None]:
        stream = self.get_stream(traffic)
        if stream is None:
            yield
            return
        with torch.cuda.stream(stream):
            yield

    def synchronize(self, traffic: TrafficClass) -> None:
        stream = self.get_stream(traffic)
        if stream is not None:
            self._debug(f"synchronize traffic={traffic.value}")
            stream.synchronize()



@dataclass
class NetworkEngineConfig:
    """Global config for network engine v0."""

    local_world_size: int | None = None
    # Stream priorities (lower = higher CUDA priority).
    stream_intranode_priority: int = 0
    stream_internode_priority: int = 0
    stream_all_bandwidth_priority: int = -1


class NetworkEngine:
    """Pure resource router for 5D parallel communication.

    Provides ONLY:
    - Backend + traffic resolution via policy (resolve / resolve_for_group)
    - CUDA stream acquisition (get_comm_stream_for_domain / get_p2p_stream_for_peer)

    All actual communication calls (all_reduce, all_gather, isend, irecv, etc.)
    remain in their original Megatron code paths.  NetworkEngine only decides
    *which stream* and *which backend* to use, then the caller wraps its own
    torch.distributed calls in ``torch.cuda.stream(stream)`` context.
    """

    def __init__(self, config: Optional[NetworkEngineConfig] = None):
        self.config = config or NetworkEngineConfig()
        self.streams = NetworkStreamManager(self.config)
        self.policy = NetworkEnginePolicy(
            local_world_size=self.config.local_world_size,
        )
        self.backends = {
            CommBackend.TORCH_DIST: TorchDistributedBackend(),
            CommBackend.NVSHMEM: NvshmemBackend(),
            CommBackend.DEEPEP: DeepEPBackend(),
        }
        self._route_debug_seen: set[tuple[str, bool, str, str]] = set()
        self._route_plan: Dict[tuple[ParallelDomain, bool], BackendDecision] = {}

        backend_status = {
            k.value: self.backends[k].available for k in self.backends
        }
        msg = (
            f"[NetworkEngine] init (resource-router mode) "
            f"local_world_size={self.policy.local_world_size} "
            f"backends={backend_status}"
        )
        logging.getLogger(__name__).info(msg)
        print(msg, file=sys.stderr)

        self._build_route_plan()
        self._initialize_stream_plan()

    def _build_route_plan(self) -> None:
        """Build and log static route plan at initialization time."""
        for domain in ParallelDomain:
            for intranode in (True, False):
                self._route_plan[(domain, intranode)] = self.policy.decide(
                    domain,
                    intranode=intranode,
                )

        lines = ["[NetworkEngine] static route plan (domain,intranode)->(backend,traffic):"]
        for domain in ParallelDomain:
            for intranode in (True, False):
                decision = self._route_plan[(domain, intranode)]
                lines.append(
                    f"  ({domain.value},{intranode}) -> ({decision.backend.value},{decision.traffic.value})"
                )
        msg = "\n".join(lines)
        logging.getLogger(__name__).info(msg)
        print(msg, file=sys.stderr)

    def _initialize_stream_plan(self) -> None:
        """Eagerly create stream plan for traffic-class and per-domain streams."""
        global _STREAM_INIT_FALLBACK_LOGGED
        intranode_stream = self.streams.get_stream(TrafficClass.INTRANODE)
        internode_stream = self.streams.get_stream(TrafficClass.INTERNODE)
        all_bandwidth_stream = self.streams.get_stream(TrafficClass.ALL_BANDWIDTH)

        msg = (
            "[NetworkEngine] legacy traffic-class stream plan "
            f"intranode={intranode_stream} "
            f"internode={internode_stream} "
            f"all_bandwidth={all_bandwidth_stream}"
        )
        logging.getLogger(__name__).info(msg)
        print(msg, file=sys.stderr)

        # Eagerly create per-domain streams
        domain_parts = []
        for domain in ParallelDomain:
            scopes = self.streams._DOMAIN_SCOPE_MATRIX.get(domain, (True,))
            for intranode in scopes:
                ds = self.streams.get_domain_stream(domain, intranode=intranode)
                domain_parts.append(
                    f"{domain.value}.{ 'intranode' if intranode else 'internode' }={ds}"
                )
        msg = "[NetworkEngine] per-domain stream plan " + ", ".join(domain_parts)
        logging.getLogger(__name__).info(msg)
        print(msg, file=sys.stderr)

        if (
            intranode_stream is None
            or internode_stream is None
            or all_bandwidth_stream is None
        ) and not _STREAM_INIT_FALLBACK_LOGGED:
            _STREAM_INIT_FALLBACK_LOGGED = True
            msg = (
                "[NetworkEngine][Fallback] static stream plan is incomplete; "
                f"intranode={intranode_stream is not None} "
                f"internode={internode_stream is not None} "
                f"all_bandwidth={all_bandwidth_stream is not None}"
            )
            logging.getLogger(__name__).warning(msg)
            print(msg, file=sys.stderr)

    def _get_backend(self, backend: CommBackend):
        impl = self.backends[backend]
        if impl.available:
            return impl
        # v0 fallback path: all unavailable backends fall back to torch.distributed.
        if backend.value not in _BACKEND_UNAVAILABLE_FALLBACK_LOGGED:
            _BACKEND_UNAVAILABLE_FALLBACK_LOGGED.add(backend.value)
            msg = (
                "[NetworkEngine][Fallback] requested backend "
                f"{backend.value} is unavailable; fallback to torch_dist"
            )
            logging.getLogger(__name__).warning(msg)
            print(msg, file=sys.stderr)
        return self.backends[CommBackend.TORCH_DIST]

    def _resolve(self, domain: ParallelDomain, intranode: bool):
        key = (domain, intranode)
        decision = self._route_plan.get(key)
        if decision is None:
            decision = self.policy.decide(domain, intranode=intranode)
            if key not in _ROUTE_PLAN_MISS_LOGGED:
                _ROUTE_PLAN_MISS_LOGGED.add(key)
                msg = (
                    "[NetworkEngine][Fallback] route plan cache miss; "
                    f"recompute decision for domain={domain.value} intranode={intranode}"
                )
                logging.getLogger(__name__).warning(msg)
                print(msg, file=sys.stderr)
        backend = self._get_backend(decision.backend)
        traffic = decision.traffic
        requested_backend = decision.backend
        if backend.name == "torch_dist" and decision.backend != CommBackend.TORCH_DIST:
            # Backend fallback keeps traffic isolation semantics.
            if traffic == TrafficClass.ALL_BANDWIDTH:
                key = (domain.value, decision.backend.value, intranode)
                if key not in _TRAFFIC_DOWNGRADE_FALLBACK_LOGGED:
                    _TRAFFIC_DOWNGRADE_FALLBACK_LOGGED.add(key)
                    msg = (
                        "[NetworkEngine][Fallback] traffic downgraded from all_bandwidth "
                        f"to torch_dist stream class for domain={domain.value} "
                        f"requested_backend={decision.backend.value} intranode={intranode}"
                    )
                    logging.getLogger(__name__).warning(msg)
                    print(msg, file=sys.stderr)
                traffic = traffic_for_backend(
                    CommBackend.TORCH_DIST,
                    torch_dist_traffic=(
                        TrafficClass.INTRANODE if intranode else TrafficClass.INTERNODE
                    ),
                )

        route_key = (
            domain.value,
            intranode,
            requested_backend.value,
            traffic.value,
        )
        if route_key not in self._route_debug_seen:
            self._route_debug_seen.add(route_key)
            msg = (
                f"[NetworkEngine] resolve domain={domain.value} intranode={intranode} "
                f"backend={requested_backend.value}->{backend.name} "
                f"traffic={traffic.value}"
            )
            logging.getLogger(__name__).info(msg)
            print(msg, file=sys.stderr)
        return backend, traffic

    def resolve(self, domain: ParallelDomain, intranode: bool):
        """Public resolver for integration points that need policy + stream routing.

        This keeps scheduler/communicator side decoupled from backend selection details.
        """
        return self._resolve(domain, intranode=intranode)

    def infer_intranode(self, group: Optional[torch.distributed.ProcessGroup]) -> bool:
        """Best-effort intranode inference from process-group membership."""
        global _INFER_INTRNODE_FALLBACK_LOGGED
        if group is None:
            return False
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return True
        try:
            return is_group_intranode(group, local_world_size=self.policy.local_world_size)
        except Exception:
            if not _INFER_INTRNODE_FALLBACK_LOGGED:
                _INFER_INTRNODE_FALLBACK_LOGGED = True
                msg = (
                    "[NetworkEngine][Fallback] failed to infer intranode from process group; "
                    "fallback to intranode=False"
                )
                logging.getLogger(__name__).warning(msg)
                print(msg, file=sys.stderr)
            return False

    def resolve_for_group(
        self,
        *,
        domain: ParallelDomain,
        group: Optional[torch.distributed.ProcessGroup],
        intranode: Optional[bool] = None,
    ):
        if intranode is None:
            intranode = self.infer_intranode(group)
        return self._resolve(domain, intranode=intranode)

    def get_comm_stream_for_domain(
        self,
        *,
        domain: ParallelDomain,
        group: Optional[torch.distributed.ProcessGroup] = None,
        intranode: Optional[bool] = None,
    ) -> Optional[torch.cuda.Stream]:
        """Get communication stream for a parallel domain.

        Returns the **per-domain + scope** CUDA stream. Scope is decided by
        ``intranode`` (or inferred from ``group``).

        - TP: intranode stream
        - CP/EP: intranode or internode stream
        - PP/DP: internode stream

        The ``group`` / ``intranode`` arguments are still used for policy
        resolution (backend selection, logging), and also determine scope.
        """
        global _GET_STREAM_SCOPE_FALLBACK_LOGGED

        if intranode is None:
            if group is not None:
                intranode = self.infer_intranode(group)
            else:
                # Domain-based fallback when no topology info is provided.
                if domain == ParallelDomain.TP:
                    intranode = True
                elif domain in (ParallelDomain.PP, ParallelDomain.DP):
                    intranode = False
                else:
                    # CP/EP are ambiguous without group/intranode.
                    intranode = False
                    if not _GET_STREAM_SCOPE_FALLBACK_LOGGED:
                        _GET_STREAM_SCOPE_FALLBACK_LOGGED = True
                        msg = (
                            "[NetworkEngine][Fallback] get_comm_stream_for_domain called "
                            "without group/intranode for CP/EP; "
                            "fallback to internode stream"
                        )
                        logging.getLogger(__name__).warning(msg)
                        print(msg, file=sys.stderr)

        # Still resolve for logging / backend decision side-effects.
        _, traffic = self.resolve(
            domain,
            intranode=bool(intranode),
        )

        # Return per-domain stream (preferred) with traffic-class fallback.
        domain_stream = self.streams.get_domain_stream(
            domain,
            intranode=bool(intranode),
        )
        if domain_stream is not None:
            return domain_stream
        # Fallback to legacy traffic-class stream.
        return self.streams.get_stream(traffic)

    def get_p2p_stream_for_peer(
        self,
        *,
        domain: ParallelDomain,
        peer_rank: int,
        local_rank: Optional[int] = None,
    ) -> Optional[torch.cuda.Stream]:
        """Resolve stream for a single P2P peer and emit bounded diagnostics.

        Prefers the per-domain stream for proper isolation across parallel
        domains.  Falls back to the legacy per-traffic-class stream when
        the domain stream is unavailable.
        """
        global _P2P_STREAM_SELECT_LOG_COUNT

        # CP attention in TransformerEngine already carries a domain-owned
        # `cp_stream` through the full algorithm and synchronizes against that
        # single stream. Returning a peer-specific stream here would create a
        # second stream source for the same CP operation and break the current
        # synchronization model. In that case, let callers fall back to their
        # existing CP stream.
        if domain == ParallelDomain.CP:
            return None

        if local_rank is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                return None
            local_rank = torch.distributed.get_rank()

        intranode = self.policy.is_intranode_rank_pair(int(local_rank), int(peer_rank))
        _, traffic = self.resolve(
            domain,
            intranode=intranode,
        )
        # Prefer per-domain stream; fall back to traffic-class stream.
        stream = self.streams.get_domain_stream(
            domain,
            intranode=intranode,
        )
        if stream is None:
            stream = self.streams.get_stream(traffic)

        if _P2P_STREAM_SELECT_LOG_COUNT < _P2P_STREAM_SELECT_LOG_LIMIT:
            _P2P_STREAM_SELECT_LOG_COUNT += 1
            msg = (
                "[NetworkEngine][P2P-Stream] "
                f"domain={domain.value} local_rank={int(local_rank)} peer_rank={int(peer_rank)} "
                f"intranode={intranode} traffic={traffic.value} stream={stream}"
            )
            logging.getLogger(__name__).info(msg)
            print(msg, file=sys.stderr)

        return stream


_GLOBAL_NETWORK_ENGINE: Optional[NetworkEngine] = None


def resolve_cp_backend_name_for_ranks(cp_global_ranks) -> str:
    """Resolve CP backend name for a CP group rank list.

    This helper allows external modules (e.g. TransformerEngine CP attention path)
    to reuse NetworkEngine policy/fallback decision with minimal coupling.
    """
    global _CP_RANKS_INFER_FALLBACK_LOGGED
    intranode = True
    if cp_global_ranks:
        try:
            lws = max(int(get_local_world_size()), 1)
            ranks = [int(r) for r in cp_global_ranks]
            node_ids = {r // lws for r in ranks}
            intranode = len(node_ids) <= 1
        except Exception:
            intranode = False
            if not _CP_RANKS_INFER_FALLBACK_LOGGED:
                _CP_RANKS_INFER_FALLBACK_LOGGED = True
                msg = (
                    "[NetworkEngine][Fallback] failed to infer CP intranode from cp_global_ranks; "
                    "fallback to intranode=False"
                )
                logging.getLogger(__name__).warning(msg)
                print(msg, file=sys.stderr)

    backend, _ = get_global_network_engine().resolve(
        ParallelDomain.CP,
        intranode=intranode,
    )
    return backend.name


def get_global_network_engine() -> NetworkEngine:
    """Get a process-local singleton NetworkEngine instance."""
    global _GLOBAL_NETWORK_ENGINE
    if _GLOBAL_NETWORK_ENGINE is None:
        _GLOBAL_NETWORK_ENGINE = NetworkEngine(NetworkEngineConfig())
    return _GLOBAL_NETWORK_ENGINE
