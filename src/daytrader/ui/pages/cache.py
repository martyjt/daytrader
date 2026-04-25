"""Data Cache page — inspect and manage the Parquet OHLCV cache.

Every backtest reads through ``FeatureStore`` (Parquet-on-disk cache).
Over time the cache grows: stale symbols, outdated ranges, duplicate
windows. This page exposes it so users can see what's cached, clear
specific symbols, or wipe the whole thing before benchmarking.
"""

from __future__ import annotations

from pathlib import Path

from nicegui import ui

from ..shell import page_layout

_CACHE_DIR = Path(__file__).resolve().parents[4] / "data" / "features"


@ui.page("/cache")
async def cache_page() -> None:
    if not page_layout("Data Cache"):
        return

    ui.label("Data Cache").classes("text-h5 q-pb-xs")
    ui.label(
        "On-disk Parquet cache of OHLCV data fetched by adapters. "
        "Every backtest, Research Lab run, and Charts Workbench request "
        "reads through this cache — massive speedup on repeated windows."
    ).classes("text-caption text-grey-5 q-pb-md")

    status = ui.row().classes("w-full q-pb-sm")
    table_container = ui.column().classes("w-full")

    def _refresh() -> None:
        status.clear()
        table_container.clear()

        if not _CACHE_DIR.exists():
            with status:
                ui.label(
                    f"Cache directory does not exist: {_CACHE_DIR}"
                ).classes("text-grey-6")
            return

        files = sorted(_CACHE_DIR.glob("*.parquet"))
        total_bytes = sum(f.stat().st_size for f in files)

        with status:
            with ui.row().classes("gap-4 items-center"):
                ui.badge(f"{len(files)} files", color="primary").props("outline")
                ui.badge(
                    f"{_fmt_bytes(total_bytes)}", color="grey-8",
                ).props("outline")
                ui.space()
                ui.button(
                    "Clear all cache",
                    icon="delete_sweep",
                    on_click=_clear_all,
                ).props("color=negative outline dense")

        if not files:
            with table_container:
                ui.label("Cache is empty.").classes("text-caption text-grey-6")
            return

        rows = []
        for f in files:
            stat = f.stat()
            rows.append({
                "file": f.name,
                "size": _fmt_bytes(stat.st_size),
                "size_bytes": stat.st_size,
                "mtime": _fmt_mtime(stat.st_mtime),
            })

        with table_container:
            cols = [
                {"name": "file", "label": "File", "field": "file"},
                {"name": "size", "label": "Size", "field": "size"},
                {"name": "mtime", "label": "Modified", "field": "mtime"},
            ]
            ui.aggrid({
                "columnDefs": cols,
                "rowData": rows,
                "domLayout": "autoHeight",
                "defaultColDef": {"sortable": True, "resizable": True},
            }).classes("w-full")

    def _clear_all() -> None:
        if not _CACHE_DIR.exists():
            return
        removed = 0
        for f in _CACHE_DIR.glob("*.parquet"):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
        ui.notify(f"Cleared {removed} cache file(s)", type="warning")
        _refresh()

    _refresh()
    ui.button(
        "Refresh", icon="refresh", on_click=_refresh,
    ).props("flat dense color=primary").classes("q-mt-sm")


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    kb = n / 1024.0
    if kb < 1024:
        return f"{kb:.1f} KB"
    mb = kb / 1024.0
    if mb < 1024:
        return f"{mb:.2f} MB"
    gb = mb / 1024.0
    return f"{gb:.2f} GB"


def _fmt_mtime(ts: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
