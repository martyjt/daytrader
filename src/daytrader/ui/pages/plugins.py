"""Plugins page — manage algorithm plugins, upload custom files.

Shows every algorithm currently registered (built-in + loaded plugins)
with a card view for each. Includes a hot-rescan button that re-runs
``AlgorithmRegistry.load_plugins()`` so newly-dropped plugin directories
appear without restarting the server, and a file upload that drops a
single-file plugin into ``plugins/uploads/`` with a safety confirmation.

Plugin upload contract (single-file ``.py``):
    * Must define a subclass of ``Algorithm``.
    * Must expose a module-level attribute ``plugin`` = instance, OR
      a class named ``Plugin`` that instantiates cleanly.
    * Gets saved to ``plugins/uploads/<filename>.py`` — the user confirms
      before the file is written.
    * After save, a "Rescan" picks it up via ``importlib`` and registers.
"""

from __future__ import annotations

import re
from pathlib import Path

from nicegui import ui, events

from ..shell import page_layout


_PLUGINS_DIR = Path(__file__).resolve().parents[4] / "plugins"
_UPLOADS_DIR = _PLUGINS_DIR / "uploads"
_VALID_FILENAME = re.compile(r"^[A-Za-z0-9_\-]+\.py$")


@ui.page("/plugins")
async def plugins_page() -> None:
    if not page_layout("Plugins"):
        return

    ui.label("Algorithm Library").classes("text-h5 q-pb-sm")
    ui.label(
        "Every algorithm below is registered and available in "
        "Strategy Lab, Research Lab, Charts Workbench, and DAG Composer. "
        "Drop new plugin directories into ``plugins/`` or upload a "
        "single-file plugin via the panel below."
    ).classes("text-body2 text-grey-6 q-pb-md")

    grid_area = ui.column().classes("w-full q-pt-md")

    def _render_registry() -> None:
        from ...algorithms.registry import AlgorithmRegistry

        grid_area.clear()
        registered = AlgorithmRegistry.all()
        with grid_area:
            with ui.row().classes("items-center gap-2 q-pb-sm"):
                ui.label(f"{len(registered)} algorithm(s) registered").classes(
                    "text-subtitle2 text-grey-5"
                )
                ui.space()
                ui.button(
                    "Rescan plugins",
                    icon="refresh",
                    on_click=_rescan,
                ).props("flat dense color=primary")

            with ui.row().classes("w-full gap-3 flex-wrap"):
                for algo_id, algo in registered.items():
                    m = algo.manifest
                    with ui.card().classes("w-72"):
                        with ui.row().classes("w-full items-center justify-between"):
                            ui.label(m.name).classes("text-h6")
                            ui.badge("Active", color="positive")
                        if m.description:
                            ui.label(m.description[:140]).classes(
                                "text-body2 text-grey-5 q-py-xs"
                            )
                        with ui.row().classes("gap-1 q-pb-xs flex-wrap"):
                            for ac in m.asset_classes:
                                ui.chip(ac, color="teal").props("dense outline")
                            for tf in m.timeframes[:3]:
                                ui.chip(tf, color="blue").props("dense outline")
                            if len(m.timeframes) > 3:
                                ui.chip(f"+{len(m.timeframes) - 3} more",
                                        color="grey").props("dense outline")
                        if m.params:
                            ui.label(
                                f"{len(m.params)} configurable parameter(s)"
                            ).classes("text-caption text-grey-7")
                        ui.label(
                            f"id: {m.id} · v{m.version}"
                            + (f" · {m.author}" if m.author else "")
                        ).classes("text-caption text-grey-7")

            if not registered:
                ui.label(
                    "No algorithms registered. Check that "
                    "AlgorithmRegistry.auto_register() ran at startup."
                ).classes("text-body2 text-grey-6")

    def _rescan() -> None:
        from ...algorithms.registry import AlgorithmRegistry

        before = len(AlgorithmRegistry.all())
        try:
            AlgorithmRegistry.load_plugins(_PLUGINS_DIR)
        except Exception as exc:  # noqa: BLE001
            ui.notify(f"Rescan failed: {exc}", type="negative")
            return
        after = len(AlgorithmRegistry.all())
        delta = after - before
        ui.notify(
            f"Rescan complete — {after} registered ({delta:+d})",
            type="positive" if delta > 0 else "info",
        )
        _render_registry()

    _render_registry()

    ui.separator().classes("q-my-md")
    ui.label("Upload single-file plugin").classes("text-subtitle1 q-pb-xs")
    ui.label(
        "Saves a ``.py`` file into ``plugins/uploads/``. Arbitrary Python "
        "code is NOT auto-executed — after upload, restart the server or "
        "click Rescan to register the plugin. The file must define an "
        "``Algorithm`` subclass."
    ).classes("text-caption text-grey-6 q-pb-sm")

    upload_status = ui.row().classes("w-full q-pt-xs")

    def _handle_upload(e: events.UploadEventArguments) -> None:
        name = e.name or ""
        if not _VALID_FILENAME.match(name):
            upload_status.clear()
            with upload_status:
                ui.icon("error", color="negative")
                ui.label(
                    f"Rejected '{name}' — filename must match "
                    f"[A-Za-z0-9_-]+.py"
                ).classes("text-negative")
            return
        try:
            _UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            payload = e.content.read()
            (_UPLOADS_DIR / name).write_bytes(payload)
        except Exception as exc:  # noqa: BLE001
            upload_status.clear()
            with upload_status:
                ui.icon("error", color="negative")
                ui.label(f"Save failed: {exc}").classes("text-negative")
            return

        upload_status.clear()
        with upload_status:
            ui.icon("check_circle", color="positive")
            ui.label(
                f"Saved to {_UPLOADS_DIR / name}. Click 'Rescan plugins' "
                "above or restart the server to register."
            ).classes("text-positive")

    ui.upload(
        on_upload=_handle_upload,
        auto_upload=True,
        max_files=1,
    ).props("accept=.py color=primary").classes("w-full")
