from __future__ import annotations

from abc import ABC, abstractmethod


class BaseCommBackend(ABC):
    """Minimal backend descriptor – only declares identity and availability.

    All actual communication calls (all_reduce, isend, irecv, …) remain in
    the original Megatron code paths.  The backend object is consulted only
    for its ``name`` and ``available`` properties by the NetworkEngine
    resource-router.
    """

    name: str = "base"

    @property
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError
