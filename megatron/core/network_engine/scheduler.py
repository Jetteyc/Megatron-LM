from __future__ import annotations

import logging
import sys
from typing import Optional

import torch

from .engine import get_global_network_engine
from .enums import ParallelDomain, TrafficClass

logger = logging.getLogger(__name__)
_SCHEDULER_STREAM_FALLBACK_WARNED = False


class SchedulerStreamPool:

    def __init__(
        self,
        comp_stream: Optional[torch.cuda.Stream] = None,
        comm_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        global _SCHEDULER_STREAM_FALLBACK_WARNED
        if comp_stream is None:
            comp_stream = torch.cuda.current_stream()
        if comm_stream is None:
            fallback_reason = None
            try:
                ne = get_global_network_engine()
                # Scheduler does not create its own comm streams. It aliases the
                # three NetworkEngine traffic-class streams.
                self._intranode_comm_stream = ne.streams.get_stream(TrafficClass.INTRANODE)
                self._internode_comm_stream = ne.streams.get_stream(TrafficClass.INTERNODE)
                self._all_bandwidth_comm_stream = ne.streams.get_stream(TrafficClass.ALL_BANDWIDTH)
                # Keep backward-compatible default comm stream behavior:
                # scheduler comm stream defaults to PP/internode class.
                comm_stream = self._internode_comm_stream
            except Exception as exc:
                self._intranode_comm_stream = None
                self._internode_comm_stream = None
                self._all_bandwidth_comm_stream = None
                comm_stream = None
                fallback_reason = f"exception={type(exc).__name__}: {exc}"
            if comm_stream is None:
                if fallback_reason is None:
                    fallback_reason = "network_engine returned no stream"
                if not _SCHEDULER_STREAM_FALLBACK_WARNED:
                    _SCHEDULER_STREAM_FALLBACK_WARNED = True
                    msg = (
                        "[NetworkEngine][Fallback] SchedulerStreamPool comm stream "
                        "failed to get network_engine stream; "
                        f"reason={fallback_reason}"
                    )
                    logger.warning(msg)
                    print(msg, file=sys.stderr)
                raise RuntimeError(
                    "SchedulerStreamPool requires network_engine comm stream "
                    "(strict stream ownership mode)"
                )
        else:
            self._intranode_comm_stream = None
            self._internode_comm_stream = comm_stream
            self._all_bandwidth_comm_stream = None
        self._comp_stream: torch.cuda.Stream = comp_stream
        self._comm_stream: torch.cuda.Stream = comm_stream

        msg = (
            f"[NetworkEngine][Scheduler] stream pool created: "
            f"comp={self._comp_stream} comm(default={self._comm_stream}) "
            f"intranode={self._intranode_comm_stream} "
            f"internode={self._internode_comm_stream} "
            f"all_bandwidth={self._all_bandwidth_comm_stream}"
        )
        logger.info(msg)
        print(msg, file=sys.stderr)


    @property
    def comp_stream(self) -> torch.cuda.Stream:
        return self._comp_stream

    @property
    def comm_stream(self) -> torch.cuda.Stream:
        return self._comm_stream

    @property
    def intranode_comm_stream(self) -> Optional[torch.cuda.Stream]:
        return self._intranode_comm_stream

    @property
    def internode_comm_stream(self) -> Optional[torch.cuda.Stream]:
        return self._internode_comm_stream

    @property
    def all_bandwidth_comm_stream(self) -> Optional[torch.cuda.Stream]:
        return self._all_bandwidth_comm_stream
    
_POOL: Optional[SchedulerStreamPool] = None


def get_scheduler_stream_pool() -> SchedulerStreamPool:
    global _POOL
    if _POOL is None:
        _POOL = SchedulerStreamPool()
    return _POOL


def get_comp_stream() -> torch.cuda.Stream:
    return get_scheduler_stream_pool().comp_stream


def get_comm_stream() -> torch.cuda.Stream:
    return get_scheduler_stream_pool().comm_stream


def get_comm_stream_for_domain(
    *,
    domain: ParallelDomain,
    group: Optional[torch.distributed.ProcessGroup] = None,
    intranode: Optional[bool] = None,
) -> Optional[torch.cuda.Stream]:
    return get_global_network_engine().get_comm_stream_for_domain(
        domain=domain,
        group=group,
        intranode=intranode,
    )


def get_comm_stream_by_traffic(traffic: TrafficClass) -> Optional[torch.cuda.Stream]:
    pool = get_scheduler_stream_pool()
    if traffic == TrafficClass.INTRANODE:
        return pool.intranode_comm_stream
    if traffic == TrafficClass.INTERNODE:
        return pool.internode_comm_stream
    if traffic == TrafficClass.ALL_BANDWIDTH:
        return pool.all_bandwidth_comm_stream
    return None