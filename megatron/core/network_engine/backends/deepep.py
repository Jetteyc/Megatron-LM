from __future__ import annotations

import logging
import sys

from .base import BaseCommBackend
from functools import cached_property

class DeepEPBackend(BaseCommBackend):

    name = "deepep"

    @cached_property
    def available(self) -> bool:
        try:
            from deep_ep import Buffer
            return True
        except Exception as exc:
            msg = (
                "[NetworkEngine][Fallback] deepep backend unavailable; "
                f"fallback to torch_dist when requested. reason={type(exc).__name__}: {exc}"
            )
            logging.getLogger(__name__).warning(msg)
            print(msg, file=sys.stderr)
            return False
