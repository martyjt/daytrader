"""Tests for plugin auto-discovery and loading."""

import textwrap
from pathlib import Path

from daytrader.algorithms.plugin_loader import PluginLoader, _parse_params


def _create_plugin(tmp_path: Path, algo_id: str = "test_plugin") -> Path:
    """Create a minimal valid plugin in tmp_path."""
    plugin_dir = tmp_path / algo_id
    plugin_dir.mkdir()

    # manifest.yaml
    (plugin_dir / "manifest.yaml").write_text(textwrap.dedent(f"""\
        id: {algo_id}
        name: "Test Plugin"
        version: "0.1.0"
        author: "Test"
        description: "A test plugin"
        asset_classes: [crypto]
        timeframes: [1d]
        params:
          - name: period
            type: int
            default: 14
            min: 2
            max: 100
    """))

    # __init__.py
    (plugin_dir / "__init__.py").write_text("")

    # algorithm.py
    (plugin_dir / "algorithm.py").write_text(textwrap.dedent("""\
        from daytrader.algorithms.base import Algorithm, AlgorithmManifest, AlgorithmParam
        from daytrader.core.context import AlgorithmContext
        from daytrader.core.types.signals import Signal

        class TestPluginAlgorithm(Algorithm):
            @property
            def manifest(self) -> AlgorithmManifest:
                return AlgorithmManifest(
                    id="test_plugin",
                    name="Test Plugin",
                    version="0.1.0",
                    params=[AlgorithmParam("period", "int", 14, min=2, max=100)],
                )

            def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
                return ctx.emit(score=0.5, confidence=0.5, reason="test")

            def warmup_bars(self) -> int:
                return 14
    """))

    return plugin_dir


def test_discover_finds_valid_plugin(tmp_path):
    _create_plugin(tmp_path, "my_algo")
    loader = PluginLoader(tmp_path)
    plugins = loader.discover()
    assert len(plugins) == 1
    assert plugins[0].algorithm_id == "my_algo"


def test_discover_skips_missing_manifest(tmp_path):
    (tmp_path / "bad_plugin").mkdir()
    loader = PluginLoader(tmp_path)
    plugins = loader.discover()
    assert len(plugins) == 0


def test_discover_empty_dir(tmp_path):
    loader = PluginLoader(tmp_path)
    plugins = loader.discover()
    assert len(plugins) == 0


def test_discover_nonexistent_dir():
    loader = PluginLoader(Path("/nonexistent/path"))
    plugins = loader.discover()
    assert len(plugins) == 0


def test_load_valid_plugin(tmp_path):
    _create_plugin(tmp_path, "test_plugin")
    loader = PluginLoader(tmp_path)
    plugins = loader.discover()
    result = loader.load(plugins[0])
    assert result.success
    assert result.algorithm is not None
    assert result.algorithm.manifest.id == "test_plugin"
    assert result.algorithm.warmup_bars() == 14


def test_load_missing_algorithm_py(tmp_path):
    plugin_dir = tmp_path / "broken"
    plugin_dir.mkdir()
    (plugin_dir / "manifest.yaml").write_text("id: broken\nname: Broken\n")
    loader = PluginLoader(tmp_path)
    plugins = loader.discover()
    result = loader.load(plugins[0])
    assert not result.success
    assert "algorithm.py not found" in result.error


def test_load_all(tmp_path):
    _create_plugin(tmp_path, "algo_a")
    _create_plugin(tmp_path, "algo_b")
    # Fix the id in algo_b's algorithm.py
    algo_b_py = tmp_path / "algo_b" / "algorithm.py"
    algo_b_py.write_text(algo_b_py.read_text().replace("test_plugin", "algo_b"))

    loader = PluginLoader(tmp_path)
    results = loader.load_all()
    assert len(results) == 2
    assert all(r.success for r in results)


def test_parse_params():
    raw = [
        {"name": "period", "type": "int", "default": 14, "min": 2, "max": 100},
        {"name": "threshold", "type": "float", "default": 0.5},
    ]
    params = _parse_params(raw)
    assert len(params) == 2
    assert params[0].name == "period"
    assert params[0].type == "int"
    assert params[0].default == 14
    assert params[1].name == "threshold"


def test_parse_params_empty():
    assert _parse_params(None) == []
    assert _parse_params([]) == []
