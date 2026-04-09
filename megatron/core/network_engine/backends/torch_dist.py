from __future__ import annotations

import torch

from .base import BaseCommBackend


class TorchDistributedBackend(BaseCommBackend):
    """Backend descriptor for torch.distributed (NCCL).

    Only provides ``name`` and ``available`` – all actual communication
    calls remain in the original Megatron code paths.
    """

    name = "torch_dist"

    @property
    def available(self) -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()
