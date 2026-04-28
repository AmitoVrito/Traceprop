"""Granularity levels for provenance tracking, using ContextVar for thread-safety."""

from __future__ import annotations

from contextvars import ContextVar
from enum import IntEnum


class Granularity(IntEnum):
    """Controls the level of detail in provenance tracking."""

    NONE = 0       # No tracking
    OP = 1         # Track operations only (op graph, no source_id propagation)
    BATCH = 2      # Track op graph + source_id propagation through edges
    TENSOR = 3     # Track tensor-level lineage (default)
    ELEMENT = 4    # Track element-level lineage


_granularity: ContextVar[Granularity] = ContextVar("_granularity", default=Granularity.TENSOR)


def get_granularity() -> Granularity:
    return _granularity.get()


def set_granularity(level: Granularity | int) -> None:
    _granularity.set(Granularity(level))
