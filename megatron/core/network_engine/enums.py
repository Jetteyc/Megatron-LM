from enum import Enum


class ParallelDomain(str, Enum):
    TP = "tp"
    CP = "cp"
    EP = "ep"
    PP = "pp"
    DP = "dp"


class CommBackend(str, Enum):
    TORCH_DIST = "torch_dist"
    NVSHMEM = "nvshmem"
    DEEPEP = "deepep"

    @classmethod
    def parse(cls, value: str, default: "CommBackend") -> "CommBackend":
        try:
            return cls(value)
        except ValueError:
            return default
