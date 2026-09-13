"""Acceptance tests for the isolated python-fix task."""

import pytest

from calculator import divide


def test_divide_returns_equal_share():
    assert divide(12, 3) == 4


def test_divide_rejects_zero_parts():
    with pytest.raises(ValueError):
        divide(12, 0)
