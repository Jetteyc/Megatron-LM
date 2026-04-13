import os

from .backends import (
    BaseCommBackend,
    DeepEPBackend,
    NvshmemBackend,
    TorchDistributedBackend,
)
from .engine import (
    BackendDecision,
    NetworkEngine,
    NetworkEngineConfig,
    NetworkEnginePolicy,
    NetworkStreamManager,
    get_global_network_engine,
    resolve_cp_backend_name_for_ranks,
)
from .enums import CommBackend, ParallelDomain, TrafficClass
from .scheduler import (
    SchedulerStreamPool,
    get_comm_stream,
    get_comp_stream,
    get_scheduler_stream_pool,
)


def is_network_engine_stream_ownership_enabled() -> bool:
    return os.getenv("MEGATRON_DISABLE_NETWORK_ENGINE_STREAM_OWNERSHIP", "0") != "1"

__all__ = [
    # Engine
    "NetworkEngine",
    "NetworkEngineConfig",
    "NetworkEnginePolicy",
    "NetworkStreamManager",
    "BackendDecision",
    "get_global_network_engine",
    "resolve_cp_backend_name_for_ranks",
    "is_network_engine_stream_ownership_enabled",
    # Backends
    "BaseCommBackend",
    "TorchDistributedBackend",
    "NvshmemBackend",
    "DeepEPBackend",
    # Enums
    "ParallelDomain",
    "TrafficClass",
    "CommBackend",
    # Scheduler stream pool
    "SchedulerStreamPool",
    "get_scheduler_stream_pool",
    "get_comp_stream",
    "get_comm_stream",
]
