"""Plugin auto-discovery and loading from the plugins/ directory.

Scans for subdirectories containing ``manifest.yaml``, parses the
manifest, dynamically imports the algorithm module, finds the
``Algorithm`` subclass, and registers it with ``AlgorithmRegistry``.

Errors are handled gracefully per-plugin — a bad plugin never crashes
the app.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .base import Algorithm, AlgorithmParam

logger = logging.getLogger(__name__)


@dataclass
class PluginInfo:
    """Discovered plugin metadata before loading."""

    path: Path
    manifest_data: dict[str, Any]
    algorithm_id: str


@dataclass
class PluginLoadResult:
    """Result of loading a single plugin."""

    plugin_id: str
    success: bool
    algorithm: Algorithm | None = None
    error: str | None = None


def _parse_params(raw_params: list[dict[str, Any]] | None) -> list[AlgorithmParam]:
    """Convert raw YAML param dicts to AlgorithmParam objects."""
    if not raw_params:
        return []
    result = []
    for p in raw_params:
        result.append(
            AlgorithmParam(
                name=p["name"],
                type=p.get("type", "float"),
                default=p.get("default", 0),
                min=p.get("min"),
                max=p.get("max"),
                step=p.get("step"),
                description=p.get("description", ""),
                choices=p.get("choices"),
            )
        )
    return result


class PluginLoader:
    """Discover and load algorithm plugins from a directory."""

    def __init__(self, plugin_dir: Path | str) -> None:
        self._plugin_dir = Path(plugin_dir)

    def discover(self) -> list[PluginInfo]:
        """Scan for subdirectories containing manifest.yaml."""
        if not self._plugin_dir.exists():
            return []

        plugins = []
        for child in sorted(self._plugin_dir.iterdir()):
            if not child.is_dir():
                continue
            manifest_path = child / "manifest.yaml"
            if not manifest_path.exists():
                continue
            try:
                with open(manifest_path) as f:
                    data = yaml.safe_load(f)
                if not isinstance(data, dict) or "id" not in data:
                    logger.warning("Invalid manifest in %s: missing 'id'", child)
                    continue
                plugins.append(
                    PluginInfo(
                        path=child,
                        manifest_data=data,
                        algorithm_id=data["id"],
                    )
                )
            except Exception as exc:
                logger.warning("Failed to parse manifest in %s: %s", child, exc)
        return plugins

    def load(self, info: PluginInfo) -> PluginLoadResult:
        """Load a single plugin by importing its algorithm module."""
        try:
            # Find the algorithm module
            algo_path = info.path / "algorithm.py"
            if not algo_path.exists():
                return PluginLoadResult(
                    plugin_id=info.algorithm_id,
                    success=False,
                    error=f"algorithm.py not found in {info.path}",
                )

            # Import the module dynamically
            module_name = f"daytrader_plugin_{info.algorithm_id}"
            spec = importlib.util.spec_from_file_location(module_name, algo_path)
            if spec is None or spec.loader is None:
                return PluginLoadResult(
                    plugin_id=info.algorithm_id,
                    success=False,
                    error=f"Could not create module spec for {algo_path}",
                )

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find the Algorithm subclass
            algo_class = None
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Algorithm) and obj is not Algorithm:
                    algo_class = obj
                    break

            if algo_class is None:
                return PluginLoadResult(
                    plugin_id=info.algorithm_id,
                    success=False,
                    error=f"No Algorithm subclass found in {algo_path}",
                )

            # Instantiate and validate
            instance = algo_class()
            if instance.manifest.id != info.algorithm_id:
                logger.warning(
                    "Plugin %s: manifest.id (%s) does not match yaml id (%s)",
                    info.path,
                    instance.manifest.id,
                    info.algorithm_id,
                )

            return PluginLoadResult(
                plugin_id=info.algorithm_id,
                success=True,
                algorithm=instance,
            )

        except Exception as exc:
            return PluginLoadResult(
                plugin_id=info.algorithm_id,
                success=False,
                error=str(exc),
            )

    def load_all(self) -> list[PluginLoadResult]:
        """Discover and load all plugins."""
        results = []
        for info in self.discover():
            result = self.load(info)
            if result.success:
                logger.info("Loaded plugin: %s", result.plugin_id)
            else:
                logger.warning(
                    "Failed to load plugin %s: %s",
                    result.plugin_id,
                    result.error,
                )
            results.append(result)
        return results
