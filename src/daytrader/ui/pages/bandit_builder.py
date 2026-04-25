"""Bandit Builder — interactive page to compose and save a bandit allocator.

A bandit allocator picks ONE child algorithm per bar via Thompson
sampling over each arm's posterior reward distribution. Unlike a
combinator that merges signals, the bandit *routes* capital to the
child currently predicted to do best.

Workflow:
    1. Pick a unique id + human name.
    2. Select 2-6 child algorithms from the registry.
    3. Tune learning rate (EW update speed) and decay (unused-arm
       regression to prior).
    4. Save — writes ``data/bandits/<id>.yaml`` and registers the
       allocator under id ``bandit:<id>``, ready to use in Strategy Lab,
       Shadow Tournament, Charts Workbench.
"""

from __future__ import annotations

import re
from pathlib import Path

from nicegui import ui

from ..shell import page_layout


_BANDITS_DIR = Path(__file__).resolve().parents[4] / "data" / "bandits"
_VALID_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")


@ui.page("/bandit-builder")
async def bandit_builder_page() -> None:
    if not page_layout("Bandit Builder"):
        return

    from ...algorithms.registry import AlgorithmRegistry

    algo_ids = AlgorithmRegistry.available()
    # Don't allow a bandit to wrap itself or another bandit (keep it shallow).
    pickable = {
        aid: AlgorithmRegistry.get(aid).manifest.name
        for aid in algo_ids
        if not aid.startswith("bandit")
    }

    ui.label("Bandit Allocator Builder").classes("text-h5 q-pb-sm")
    ui.label(
        "Compose a Thompson-sampling allocator over N child algorithms. "
        "On each bar the bandit picks whichever arm has the highest "
        "sampled posterior reward; arms update on credit-assigned PnL."
    ).classes("text-caption text-grey-5 q-pb-md")

    with ui.card().classes("w-full"):
        with ui.row().classes("w-full gap-4 items-end flex-wrap"):
            id_in = ui.input(
                "ID", value="my_bandit",
                placeholder="letters, digits, underscores, hyphens only",
            ).classes("min-w-[200px]")
            name_in = ui.input(
                "Display name", value="My Bandit",
            ).classes("min-w-[220px]")

        children_in = ui.select(
            options=pickable,
            value=[],
            label="Child algorithms (pick 2–6)",
            multiple=True,
        ).props("use-chips").classes("w-full q-pt-sm")

        with ui.row().classes("w-full gap-4 items-end q-pt-sm"):
            lr_in = ui.number(
                "Learning rate (posterior EW update)",
                value=0.1, min=0.01, max=0.9, step=0.01,
            ).classes("min-w-[260px]")
            decay_in = ui.number(
                "Decay (unused-arm drift per bar)",
                value=0.99, min=0.90, max=1.0, step=0.005,
            ).classes("min-w-[260px]")
            seed_in = ui.number(
                "Seed", value=0, min=0, max=2**31,
            ).classes("min-w-[100px]")

        desc_in = ui.textarea(
            "Description (optional)",
            placeholder="Why this bandit, what it's designed for...",
        ).classes("w-full q-pt-sm")

    status = ui.row().classes("w-full q-pt-sm")
    existing = ui.column().classes("w-full q-pt-md")

    async def save_bandit() -> None:
        status.clear()

        cid = (id_in.value or "").strip()
        if not cid or not _VALID_ID.match(cid):
            with status:
                ui.label(
                    "Invalid id — use letters, digits, underscores, or hyphens only."
                ).classes("text-negative")
            return

        selected = list(children_in.value or [])
        if not (2 <= len(selected) <= 6):
            with status:
                ui.label(
                    f"Pick 2–6 children (currently {len(selected)})."
                ).classes("text-negative")
            return

        try:
            from ...algorithms.rl.bandit_serialization import (
                BanditConfig,
                build_bandit_from_config,
                save_bandit as _save_bandit,
                _install_named_bandit,
            )
            from ...algorithms.registry import AlgorithmRegistry

            config = BanditConfig(
                id=cid,
                name=(name_in.value or cid).strip(),
                children=selected,
                learning_rate=float(lr_in.value or 0.1),
                decay=float(decay_in.value or 0.99),
                seed=int(seed_in.value or 0),
                description=(desc_in.value or "").strip(),
            )
            _save_bandit(config, _BANDITS_DIR)

            # Hot-register the new bandit so it appears without restart.
            algo_id = f"bandit:{cid}"
            if algo_id in AlgorithmRegistry.available():
                # Replace by deleting and re-registering.
                AlgorithmRegistry._algorithms.pop(algo_id, None)  # noqa: SLF001
            algo = build_bandit_from_config(config)
            if algo is not None:
                _install_named_bandit(algo, config)
        except Exception as exc:  # noqa: BLE001
            with status:
                ui.icon("error", color="negative")
                ui.label(f"Save failed: {exc}").classes("text-negative")
            return

        with status:
            ui.icon("check_circle", color="positive")
            ui.label(
                f"Saved bandit:{cid} — now available in algorithm pickers."
            ).classes("text-positive")
        _refresh_existing()

    def _refresh_existing() -> None:
        existing.clear()
        with existing:
            ui.separator().classes("q-my-md")
            ui.label("Saved bandits").classes("text-subtitle1 q-pb-xs")
            if not _BANDITS_DIR.exists():
                ui.label("None yet.").classes("text-caption text-grey-6")
                return
            rows = []
            for path in sorted(_BANDITS_DIR.glob("*.yaml")):
                try:
                    from ...algorithms.rl.bandit_serialization import load_bandit

                    cfg = load_bandit(path)
                    rows.append({
                        "id": cfg.id,
                        "name": cfg.name,
                        "children": ", ".join(cfg.children) or "(none)",
                        "lr": f"{cfg.learning_rate:.2f}",
                        "decay": f"{cfg.decay:.2f}",
                    })
                except Exception:  # noqa: BLE001
                    continue
            if not rows:
                ui.label("None yet.").classes("text-caption text-grey-6")
                return
            cols = [
                {"name": "id", "label": "ID", "field": "id"},
                {"name": "name", "label": "Name", "field": "name"},
                {"name": "children", "label": "Children", "field": "children"},
                {"name": "lr", "label": "LR", "field": "lr"},
                {"name": "decay", "label": "Decay", "field": "decay"},
            ]
            ui.aggrid({
                "columnDefs": cols,
                "rowData": rows,
                "domLayout": "autoHeight",
                "defaultColDef": {"sortable": True, "resizable": True},
            }).classes("w-full")

    ui.button(
        "Save bandit", icon="save", on_click=save_bandit,
    ).props("color=primary unelevated size=lg").classes("q-mt-sm")

    _refresh_existing()
