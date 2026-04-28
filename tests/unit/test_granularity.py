"""Tests for granularity settings."""

import pytest

from traceprop.granularity import Granularity, get_granularity, set_granularity


@pytest.fixture(autouse=True)
def _reset():
    set_granularity(Granularity.TENSOR)
    yield
    set_granularity(Granularity.TENSOR)


def test_default():
    assert get_granularity() == Granularity.TENSOR


def test_set_none():
    set_granularity(Granularity.NONE)
    assert get_granularity() == Granularity.NONE


def test_set_by_int():
    set_granularity(1)
    assert get_granularity() == Granularity.OP


def test_ordering():
    assert Granularity.NONE < Granularity.OP < Granularity.BATCH < Granularity.TENSOR < Granularity.ELEMENT
