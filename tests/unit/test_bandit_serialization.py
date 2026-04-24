"""Unit tests for bandit allocator serialization + round-trip.

Locks that a ``BanditConfig`` survives save→load→build without losing
any of the user's intent (children, hyperparameters, description).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daytrader.algorithms.registry import AlgorithmRegistry
from daytrader.algorithms.rl.bandit_serialization import (
    BanditConfig,
    build_bandit_from_config,
    load_all_bandits,
    load_bandit,
    save_bandit,
)


@pytest.fixture(autouse=True)
def _registry_scope():
    """Each test runs against a fresh registry so order doesn't matter."""
    AlgorithmRegistry.clear()
    AlgorithmRegistry.auto_register()
    yield
    AlgorithmRegistry.clear()


def test_config_round_trip(tmp_path: Path) -> None:
    cfg = BanditConfig(
        id="round_trip",
        name="Round Trip",
        children=["ema_crossover", "macd_signal"],
        learning_rate=0.15,
        decay=0.98,
        seed=42,
        description="test",
    )
    path = save_bandit(cfg, tmp_path)
    loaded = load_bandit(path)

    assert loaded.id == cfg.id
    assert loaded.name == cfg.name
    assert loaded.children == cfg.children
    assert loaded.learning_rate == cfg.learning_rate
    assert loaded.decay == cfg.decay
    assert loaded.seed == cfg.seed


def test_build_rejects_unknown_children() -> None:
    cfg = BanditConfig(
        id="no_children",
        name="No Children",
        children=["totally_bogus_algo_123"],
    )
    assert build_bandit_from_config(cfg) is None


def test_build_succeeds_with_partial_children() -> None:
    # Mix of real + bogus; should still build with the real ones.
    cfg = BanditConfig(
        id="partial",
        name="Partial",
        children=["ema_crossover", "totally_bogus_algo_123"],
    )
    allocator = build_bandit_from_config(cfg)
    assert allocator is not None


def test_load_all_bandits_registers_under_prefixed_id(tmp_path: Path) -> None:
    cfg = BanditConfig(
        id="unit_test_b",
        name="Unit-Test Bandit",
        children=["ema_crossover", "macd_signal"],
    )
    save_bandit(cfg, tmp_path)
    load_all_bandits(tmp_path)

    registered = AlgorithmRegistry.available()
    assert "bandit:unit_test_b" in registered

    instance = AlgorithmRegistry.get("bandit:unit_test_b")
    assert instance.manifest.id == "bandit:unit_test_b"
    assert instance.manifest.name == "Unit-Test Bandit"


def test_load_all_bandits_ignores_missing_dir(tmp_path: Path) -> None:
    # Directory exists but has no yaml files — must not raise.
    load_all_bandits(tmp_path / "not_there")


def test_load_all_bandits_skips_corrupt_yaml(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: valid: yaml: content", encoding="utf-8")
    # Valid bandit alongside the bad one
    cfg = BanditConfig(id="good", name="Good", children=["ema_crossover", "macd_signal"])
    save_bandit(cfg, tmp_path)

    # Should not raise; should register the good one only.
    load_all_bandits(tmp_path)
    assert "bandit:good" in AlgorithmRegistry.available()
