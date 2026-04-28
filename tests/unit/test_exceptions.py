"""Tests for exceptions and safe_provenance decorator."""

from traceprop.exceptions import (
    BackendNotInstalledError,
    ExportError,
    QueryError,
    StoreError,
    StoreUnavailableWarning,
    TracepropError,
    safe_provenance,
)


def test_exception_hierarchy():
    assert issubclass(StoreError, TracepropError)
    assert issubclass(ExportError, TracepropError)
    assert issubclass(QueryError, TracepropError)
    assert issubclass(BackendNotInstalledError, TracepropError)
    assert issubclass(StoreUnavailableWarning, UserWarning)


def test_safe_provenance_returns_value():
    @safe_provenance
    def good():
        return 42

    assert good() == 42


def test_safe_provenance_catches_errors():
    @safe_provenance
    def bad():
        raise RuntimeError("boom")

    assert bad() is None
