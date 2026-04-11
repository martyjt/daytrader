"""Tests for the execution adapter registry."""

from __future__ import annotations

import pytest

from daytrader.execution.base import ExecutionAdapter
from daytrader.execution.paper import PaperExecutor
from daytrader.execution.registry import ExecutionRegistry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with a fresh registry."""
    ExecutionRegistry.clear()
    yield
    ExecutionRegistry.clear()


def test_register_and_get():
    adapter = PaperExecutor()
    ExecutionRegistry.register(adapter)
    assert ExecutionRegistry.get("paper") is adapter


def test_get_unknown_raises():
    with pytest.raises(KeyError, match="not registered"):
        ExecutionRegistry.get("nonexistent")


def test_available_returns_sorted_names():
    ExecutionRegistry.register(PaperExecutor())
    assert ExecutionRegistry.available() == ["paper"]


def test_clear_removes_all():
    ExecutionRegistry.register(PaperExecutor())
    ExecutionRegistry.clear()
    assert ExecutionRegistry.available() == []


def test_auto_register_always_adds_paper():
    """auto_register should always register PaperExecutor."""
    ExecutionRegistry.auto_register()
    assert "paper" in ExecutionRegistry.available()
    adapter = ExecutionRegistry.get("paper")
    assert isinstance(adapter, PaperExecutor)


def test_auto_register_idempotent():
    """Calling auto_register twice doesn't duplicate adapters."""
    ExecutionRegistry.auto_register()
    first = ExecutionRegistry.get("paper")
    ExecutionRegistry.auto_register()
    assert ExecutionRegistry.get("paper") is first
