# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from enum import Enum


class ParallelDomain(str, Enum):
    """Communication domain in 5D parallelism."""

    TP = "tp"
    CP = "cp"
    EP = "ep"
    PP = "pp"
    DP = "dp"


class TrafficClass(str, Enum):
    """CUDA stream class for communication isolation."""

    INTRANODE = "intranode"
    INTERNODE = "internode"
    ALL_BANDWIDTH = "all_bandwidth"


class CommBackend(str, Enum):
    """Backend kind used by network engine."""

    TORCH_DIST = "torch_dist"
    NVSHMEM = "nvshmem"
    DEEPEP = "deepep"

    @classmethod
    def parse(cls, value: str, default: 'CommBackend') -> 'CommBackend':
        try:
            return cls(value)
        except ValueError:
            return default


def traffic_for_backend(
    backend: CommBackend,
    *,
    torch_dist_traffic: TrafficClass,
    accel_traffic: TrafficClass = TrafficClass.ALL_BANDWIDTH,
) -> TrafficClass:
    return torch_dist_traffic if backend == CommBackend.TORCH_DIST else accel_traffic
