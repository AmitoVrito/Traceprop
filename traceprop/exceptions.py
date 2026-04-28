"""Traceprop exceptions and safe_provenance decorator (non-hot-path only)."""

from __future__ import annotations

import functools
import logging
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable)


class TracepropError(Exception):
    """Base exception for all traceprop errors."""


class StoreError(TracepropError):
    """Raised when a store operation fails."""


class ExportError(TracepropError):
    """Raised when an export operation fails."""


class QueryError(TracepropError):
    """Raised when a query operation fails."""


class BackendNotInstalledError(TracepropError):
    """Raised when a required backend (e.g. JAX) is not installed."""


class StoreUnavailableWarning(UserWarning):
    """Warning emitted when a store backend is not available."""


def safe_provenance(fn: F) -> F:
    """Decorator that catches provenance errors so they never break user code.

    Use ONLY on non-hot-path functions (store ops, exports, queries).
    Hot-path code uses inline try/except instead.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.debug("Provenance error in %s", fn.__name__, exc_info=True)
            return None

    return wrapper  # type: ignore[return-value]
