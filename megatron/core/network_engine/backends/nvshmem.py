from __future__ import annotations

import logging
import sys

from .base import BaseCommBackend
from functools import cached_property

class NvshmemBackend(BaseCommBackend):

    name = "nvshmem"

    @cached_property
    def available(self) -> bool:
        try:
            from transformer_engine.pytorch.attention.dot_product_attention import (  # noqa: F401
                context_parallel_nvshmem,
            )

            return True
        except Exception as exc:
            msg = (
                "[NetworkEngine][Fallback] nvshmem backend unavailable; "
                f"fallback to torch_dist when requested. reason={type(exc).__name__}: {exc}"
            )
            logging.getLogger(__name__).warning(msg)
            print(msg, file=sys.stderr)
            return False
