"""Tests for backends."""

from traceprop.backends.numpy_backend import get_array_module, is_available


def test_is_available():
    assert is_available() is True


def test_get_array_module():
    import numpy as np
    assert get_array_module() is np
