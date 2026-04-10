"""Plugins page — view and manage algorithm plugins.

Phase 0: shows registered algorithms from the AlgorithmRegistry
plus placeholder cards for algorithms coming in Phase 2.
"""

from __future__ import annotations

from nicegui import ui

from ..shell import page_layout

_COMING_SOON = [
    ("EMA Crossover", "Trend-following with fast/slow EMA cross"),
    ("RSI Mean Revert", "Buy oversold, sell overbought (RSI 14)"),
    ("MACD Signal", "Momentum via MACD line / signal cross"),
    ("Bollinger Bands", "Mean reversion at 2-sigma Bollinger touches"),
    ("ADX Trend Filter", "Trend strength gate — only trade when ADX > 25"),
    ("Donchian Breakout", "Classic Turtle breakout on N-day high/low"),
    ("VWAP Reversion", "Intraday mean reversion to anchored VWAP"),
    ("Stochastic Oscillator", "Short-term overbought/oversold oscillator"),
    ("ATR Chandelier Exit", "Dynamic trailing stop based on ATR"),
]


@ui.page("/plugins")
async def plugins_page() -> None:
    page_layout("Plugins")

    ui.label("Algorithm Library").classes("text-h5 q-pb-sm")
    ui.label(
        "These algorithms are available for backtesting in the Strategy Lab. "
        "Phase 2 will add installable plugins from the plugins/ directory."
    ).classes("text-body2 text-grey-6 q-pb-md")

    # ---- Registered algorithms from the real registry ---------------------
    from ...algorithms.registry import AlgorithmRegistry

    registered = AlgorithmRegistry.all()

    with ui.row().classes("w-full gap-4 flex-wrap"):
        for algo_id, algo in registered.items():
            m = algo.manifest
            with ui.card().classes("w-72"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(m.name).classes("text-h6")
                    ui.badge("Active", color="positive")

                ui.label(m.description).classes(
                    "text-body2 text-grey-5 q-py-xs"
                )

                with ui.row().classes("gap-1 q-pb-xs flex-wrap"):
                    for ac in m.asset_classes:
                        ui.chip(ac, color="teal").props("dense outline")
                    for tf in m.timeframes[:4]:
                        ui.chip(tf, color="blue").props("dense outline")
                    if len(m.timeframes) > 4:
                        ui.chip(
                            f"+{len(m.timeframes) - 4} more", color="grey"
                        ).props("dense outline")

                with ui.row().classes("items-center gap-2"):
                    ui.label(f"v{m.version}").classes(
                        "text-caption text-grey-7"
                    )
                    if m.author:
                        ui.label(f"· {m.author}").classes(
                            "text-caption text-grey-7"
                        )

                ui.switch("Enabled", value=True)

    # ---- Coming soon ------------------------------------------------------
    if _COMING_SOON:
        ui.separator().classes("q-my-md")
        ui.label("Coming in Phase 2").classes("text-h6 text-grey-5 q-pb-sm")

        with ui.row().classes("w-full gap-4 flex-wrap"):
            for name, desc in _COMING_SOON:
                with ui.card().classes("w-72 opacity-60"):
                    ui.label(name).classes("text-h6 text-grey-5")
                    ui.label(desc).classes("text-body2 text-grey-7")
                    ui.badge("Coming soon", color="grey").props("outline")
