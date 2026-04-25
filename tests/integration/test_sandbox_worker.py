"""End-to-end tests for the plugin worker subprocess.

These actually spawn ``python -m daytrader.algorithms.sandbox.worker_main``,
load tiny on-the-fly plugins, and verify load/run/crash/recovery flows.
The tests are slow-ish (subprocess start latency) so we batch into one
file with shared fixtures.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from daytrader.algorithms.sandbox import (
    PluginRuntimeError,
    PluginWorkerHandle,
    PluginWorkerManager,
    PluginWorkerTimeout,
)
from daytrader.algorithms.sandbox import protocol
from daytrader.core.context import AlgorithmContext
from daytrader.core.types.bars import Bar, Timeframe
from daytrader.core.types.symbols import AssetClass, Symbol


def _ctx(algo_id: str, tenant_id) -> AlgorithmContext:
    closes = np.array([100.0, 101.0, 102.0, 103.0, 105.0], dtype="float64")
    return AlgorithmContext(
        tenant_id=tenant_id,
        persona_id=uuid4(),
        algorithm_id=algo_id,
        symbol=Symbol("BTC", "USDT", AssetClass.CRYPTO, "binance"),
        timeframe=Timeframe.D1,
        now=datetime(2026, 4, 25, tzinfo=timezone.utc),
        bar=Bar(
            timestamp=datetime(2026, 4, 25, tzinfo=timezone.utc),
            open=Decimal("104"), high=Decimal("106"), low=Decimal("103"),
            close=Decimal("105"), volume=Decimal("1000"),
        ),
        history_arrays={
            "close": closes, "open": closes, "high": closes,
            "low": closes, "volume": np.ones(5),
        },
        features={"rsi_14": 0.5},
        params={},
        emit_fn=lambda s: None,
        log_fn=lambda m, f: None,
    )


@pytest.fixture
def manager(tmp_path):
    mgr = PluginWorkerManager(base_dir=tmp_path)
    yield mgr


@pytest.fixture
async def handle(manager):
    tenant_id = uuid4()
    h = await manager.get_handle(tenant_id)
    yield h
    await manager.shutdown_all()


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


_GOOD_PLUGIN = """
from daytrader.algorithms.base import Algorithm, AlgorithmManifest, AlgorithmParam
from daytrader.core.context import AlgorithmContext
from daytrader.core.types.signals import Signal


class MovingAvg(Algorithm):
    @property
    def manifest(self):
        return AlgorithmManifest(
            id="ma_test", name="Test MA",
            params=[AlgorithmParam(name="window", type="int", default=3)],
        )

    def warmup_bars(self):
        return 3

    def on_bar(self, ctx):
        window = ctx.param("window", 3)
        closes = ctx.history(window, "close")
        avg = float(closes.mean())
        score = 1.0 if ctx.bar.close > avg else -1.0
        return ctx.emit(score=score, confidence=0.7, reason=f"ma{window}")
"""


async def test_ping(handle):
    assert await handle.ping() is True


async def test_load_run_unload(handle, tmp_path):
    plugin_path = tmp_path / "ma_test.py"
    plugin_path.write_text(_GOOD_PLUGIN)
    info = await handle.load_plugin(plugin_path, "ma_test")
    assert info["manifest"]["id"] == "ma_test"
    assert info["warmup_bars"] == 3

    payload = protocol.serialize_context(_ctx("ma_test", handle.tenant_id))
    result = await handle.on_bar("ma_test", payload)
    assert len(result["signals"]) == 1
    sig = protocol.deserialize_signal(result["signals"][0])
    assert sig.source == "ma_test"
    assert sig.reason == "ma3"
    assert sig.score in (1.0, -1.0)

    await handle.unload("ma_test")
    # After unload, on_bar fails with "not loaded"
    with pytest.raises(PluginRuntimeError) as exc:
        await handle.on_bar("ma_test", payload)
    assert "not loaded" in exc.value.error_message


async def test_replay_bars(handle, tmp_path):
    plugin_path = tmp_path / "ma_test.py"
    plugin_path.write_text(_GOOD_PLUGIN)
    await handle.load_plugin(plugin_path, "ma_test")

    contexts = [
        protocol.serialize_context(_ctx("ma_test", handle.tenant_id))
        for _ in range(20)
    ]
    results = await handle.replay_bars("ma_test", contexts)
    assert len(results) == 20
    for r in results:
        assert "signals" in r
        assert len(r["signals"]) == 1


# ---------------------------------------------------------------------------
# Crash + recovery
# ---------------------------------------------------------------------------


_CRASH_PLUGIN = """
from daytrader.algorithms.base import Algorithm, AlgorithmManifest


class Crash(Algorithm):
    @property
    def manifest(self):
        return AlgorithmManifest(id="crash", name="Crash")

    def on_bar(self, ctx):
        raise ValueError("plugin boom")
"""


async def test_plugin_exception_returns_error_not_crash(handle, tmp_path):
    plugin_path = tmp_path / "crash.py"
    plugin_path.write_text(_CRASH_PLUGIN)
    await handle.load_plugin(plugin_path, "crash")

    payload = protocol.serialize_context(_ctx("crash", handle.tenant_id))
    with pytest.raises(PluginRuntimeError) as exc:
        await handle.on_bar("crash", payload)
    assert "plugin boom" in exc.value.error_message
    # Worker is still alive after a plugin-level exception.
    assert await handle.ping() is True


_HARD_EXIT_PLUGIN = """
import os
from daytrader.algorithms.base import Algorithm, AlgorithmManifest


class HardExit(Algorithm):
    @property
    def manifest(self):
        return AlgorithmManifest(id="hard_exit", name="HardExit")

    def on_bar(self, ctx):
        os._exit(99)
"""


async def test_worker_crash_and_respawn(handle, tmp_path):
    plugin_path = tmp_path / "hard_exit.py"
    plugin_path.write_text(_HARD_EXIT_PLUGIN)
    await handle.load_plugin(plugin_path, "hard_exit")

    payload = protocol.serialize_context(_ctx("hard_exit", handle.tenant_id))
    # The manager auto-restarts once on crash. The retry hits a fresh worker
    # that doesn't have the plugin loaded → PluginRuntimeError("not loaded").
    with pytest.raises(PluginRuntimeError) as exc:
        await handle.on_bar("hard_exit", payload)
    assert "not loaded" in exc.value.error_message
    # And after the respawn the worker is alive again.
    assert await handle.ping() is True


# ---------------------------------------------------------------------------
# Forbidden imports
# ---------------------------------------------------------------------------


def _forbidden_plugin(target: str) -> str:
    return f"""
from {target} import *  # type: ignore
from daytrader.algorithms.base import Algorithm, AlgorithmManifest


class Bad(Algorithm):
    @property
    def manifest(self):
        return AlgorithmManifest(id="bad_{target.replace('.', '_')}", name="Bad")

    def on_bar(self, ctx):
        return None
"""


@pytest.mark.parametrize("target", [
    "daytrader.storage.session",
    "daytrader.execution.registry",
    "daytrader.core.crypto",
    "daytrader.auth.session",
])
async def test_forbidden_import_blocked(handle, tmp_path, target):
    plugin_path = tmp_path / f"bad_{target.replace('.', '_')}.py"
    plugin_path.write_text(_forbidden_plugin(target))
    with pytest.raises(PluginRuntimeError) as exc:
        await handle.load_plugin(
            plugin_path, f"bad_{target.replace('.', '_')}"
        )
    msg = exc.value.error_message
    assert "blocked by the sandbox" in msg or "ImportError" in exc.value.error_type


# ---------------------------------------------------------------------------
# Env scrub
# ---------------------------------------------------------------------------


_ENV_PROBE_PLUGIN = """
import os, json
from daytrader.algorithms.base import Algorithm, AlgorithmManifest


class Probe(Algorithm):
    @property
    def manifest(self):
        return AlgorithmManifest(id="env_probe", name="Probe")

    def on_bar(self, ctx):
        # Phone home through metadata - the only legit way out.
        return ctx.emit(
            score=0.0, confidence=1.0, reason="probe",
            metadata={
                "has_db": os.environ.get("DATABASE_URL") is not None,
                "has_key": os.environ.get("APP_ENCRYPTION_KEY") is not None,
                "has_binance": any(k.startswith("BINANCE") for k in os.environ),
            },
        )
"""


async def test_env_secrets_are_scrubbed(handle, tmp_path, monkeypatch):
    # Inject a secret into the parent's env so we can verify the worker
    # didn't inherit it.
    monkeypatch.setenv("DATABASE_URL", "postgresql://probe:probe@x/y")
    monkeypatch.setenv("APP_ENCRYPTION_KEY", "shouldnotleak")
    monkeypatch.setenv("BINANCE_API_KEY", "shouldnotleak")

    plugin_path = tmp_path / "env_probe.py"
    plugin_path.write_text(_ENV_PROBE_PLUGIN)
    await handle.load_plugin(plugin_path, "env_probe")

    payload = protocol.serialize_context(_ctx("env_probe", handle.tenant_id))
    result = await handle.on_bar("env_probe", payload)
    sig = protocol.deserialize_signal(result["signals"][0])
    assert sig.metadata["has_db"] is False
    assert sig.metadata["has_key"] is False
    assert sig.metadata["has_binance"] is False


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


_HANG_PLUGIN = """
import time
from daytrader.algorithms.base import Algorithm, AlgorithmManifest


class Hang(Algorithm):
    @property
    def manifest(self):
        return AlgorithmManifest(id="hang", name="Hang")

    def on_bar(self, ctx):
        time.sleep(60)
"""


async def test_hung_plugin_times_out(handle, tmp_path):
    plugin_path = tmp_path / "hang.py"
    plugin_path.write_text(_HANG_PLUGIN)
    await handle.load_plugin(plugin_path, "hang")

    payload = protocol.serialize_context(_ctx("hang", handle.tenant_id))
    with pytest.raises(PluginWorkerTimeout):
        await handle.call("on_bar", timeout=0.5, algo_id="hang", ctx=payload)
    # After timeout, worker is killed; subsequent ping respawns.
    assert await handle.ping() is True


# ---------------------------------------------------------------------------
# Cross-tenant isolation (manager-level)
# ---------------------------------------------------------------------------


async def test_two_tenants_get_separate_workers(manager, tmp_path):
    t1, t2 = uuid4(), uuid4()
    h1 = await manager.get_handle(t1)
    h2 = await manager.get_handle(t2)
    assert h1 is not h2
    # Each gets its own working directory under base_dir.
    assert h1._cwd != h2._cwd
    assert manager.tenant_dir(t1) != manager.tenant_dir(t2)
    await manager.shutdown_all()


async def test_shutdown_tenant_cleans_only_one(manager):
    t1, t2 = uuid4(), uuid4()
    await manager.get_handle(t1)
    await manager.get_handle(t2)
    await manager.shutdown_tenant(t1)
    assert not manager.has_handle(t1)
    assert manager.has_handle(t2)
    await manager.shutdown_all()
