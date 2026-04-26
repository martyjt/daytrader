"""Smoke tests for the Phase 13 example plugins.

These plugins live under ``plugins/examples/`` and aren't auto-loaded —
they're meant to be copied into ``plugins/``. The tests load each one
through the same :class:`PluginLoader` that the runtime uses, so any
regression in the plugin contract surfaces immediately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daytrader.algorithms.plugin_loader import PluginLoader

EXAMPLES_DIR = (
    Path(__file__).resolve().parents[2] / "plugins" / "examples"
)
EXPECTED_IDS = {
    "buy_hold": "example_buy_hold",
    "rsi_threshold": "example_rsi_threshold",
    "template": "example_template",
}


@pytest.mark.parametrize("folder, algo_id", EXPECTED_IDS.items())
def test_example_plugin_loads(folder, algo_id):
    """Each example must declare its manifest and expose an Algorithm subclass."""
    loader = PluginLoader(EXAMPLES_DIR)
    plugins = {info.algorithm_id: info for info in loader.discover()}
    assert algo_id in plugins, f"{algo_id} not discovered in {EXAMPLES_DIR}"

    result = loader.load(plugins[algo_id])
    assert result.success, f"{algo_id} failed to load: {result.error}"
    assert result.algorithm is not None
    manifest = result.algorithm.manifest
    assert manifest.id == algo_id


def test_examples_dir_layout():
    """Every example folder has the three required artefacts."""
    for folder in EXPECTED_IDS:
        path = EXAMPLES_DIR / folder
        assert path.is_dir(), f"missing {path}"
        assert (path / "manifest.yaml").is_file()
        assert (path / "algorithm.py").is_file()
        assert (path / "__init__.py").is_file()
