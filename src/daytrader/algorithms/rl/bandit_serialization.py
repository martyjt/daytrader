"""Serialize / load Bandit Allocator configurations to YAML.

Parallels the DAG serialization — lets users compose bandit strategies
interactively (via the Bandit Builder page) and save them. On startup
the algorithm registry picks up every YAML in ``data/bandits/`` and
exposes each as a registered algorithm under the id ``bandit:<config_id>``.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..registry import AlgorithmRegistry
from .bandit_allocator import BanditAllocator


@dataclass
class BanditConfig:
    """User-facing bandit configuration."""

    id: str
    name: str
    children: list[str] = field(default_factory=list)
    learning_rate: float = 0.1
    decay: float = 0.99
    seed: int = 0
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "children": list(self.children),
            "learning_rate": self.learning_rate,
            "decay": self.decay,
            "seed": self.seed,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BanditConfig":
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", data["id"])),
            children=[str(c) for c in data.get("children", [])],
            learning_rate=float(data.get("learning_rate", 0.1)),
            decay=float(data.get("decay", 0.99)),
            seed=int(data.get("seed", 0)),
            description=str(data.get("description", "")),
        )


def save_bandit(config: BanditConfig, bandits_dir: Path | str) -> Path:
    """Persist a ``BanditConfig`` as YAML under ``bandits_dir``."""
    bandits_dir = Path(bandits_dir)
    bandits_dir.mkdir(parents=True, exist_ok=True)
    path = bandits_dir / f"{config.id}.yaml"
    path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    return path


def load_bandit(path: Path | str) -> BanditConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return BanditConfig.from_dict(data)


def build_bandit_from_config(config: BanditConfig) -> BanditAllocator | None:
    """Instantiate a ``BanditAllocator`` with children pulled from the registry.

    Returns ``None`` if any child isn't registered yet (registry order matters —
    this function is called after all builtin algorithms register).
    """
    children = []
    for child_id in config.children:
        try:
            template = AlgorithmRegistry.get(child_id)
        except KeyError:
            # Silently skip missing children; user can re-save with valid IDs.
            continue
        children.append(copy.deepcopy(template))
    if not children:
        return None
    allocator = BanditAllocator(
        children=children,
        learning_rate=config.learning_rate,
        decay=config.decay,
        seed=config.seed,
    )
    return allocator


def load_all_bandits(bandits_dir: Path | str) -> None:
    """Load every bandit YAML from ``bandits_dir`` and register it.

    Each saved bandit appears in the algorithm registry under
    ``bandit:<config_id>`` so it can be used anywhere a normal algorithm
    can be: Strategy Lab, Shadow Tournament, Charts Workbench, DAGs.
    """
    bandits_dir = Path(bandits_dir)
    if not bandits_dir.exists():
        return
    for yaml_file in sorted(bandits_dir.glob("*.yaml")):
        try:
            config = load_bandit(yaml_file)
        except Exception:  # noqa: BLE001 — one bad file shouldn't break startup
            continue
        algo = build_bandit_from_config(config)
        if algo is None:
            continue

        # Rebrand the manifest so each saved bandit gets a unique id/name.
        # We subclass on the fly to override the manifest property without
        # mutating the template's AlgorithmManifest.
        _install_named_bandit(algo, config)


def _install_named_bandit(allocator: BanditAllocator, config: BanditConfig) -> None:
    """Register ``allocator`` under a unique id derived from ``config``.

    BanditAllocator's base manifest always reports id=``bandit_allocator``.
    For saved instances we need a distinct id so the registry can
    differentiate between e.g. ``bandit:btc_meta`` and ``bandit:equities_switch``.
    """
    from ..base import AlgorithmManifest

    algo_id = f"bandit:{config.id}"

    # Capture the base manifest and replace its id/name.
    base = allocator.manifest
    new_manifest = AlgorithmManifest(
        id=algo_id,
        name=config.name or algo_id,
        version=base.version,
        description=(
            config.description
            or f"Bandit allocator over: {', '.join(config.children) or '(no children)'}"
        ),
        asset_classes=base.asset_classes,
        timeframes=base.timeframes,
        params=base.params,
        author=base.author,
        suitable_regimes=base.suitable_regimes,
    )
    allocator.__class__ = type(
        f"Bandit_{config.id}",
        (BanditAllocator,),
        {"manifest": property(lambda _self: new_manifest)},
    )
    AlgorithmRegistry.register(allocator)
