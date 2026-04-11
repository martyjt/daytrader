"""Sweep parameter range form for the Research Lab.

Renders per-parameter controls: a checkbox to sweep, and min/max/step
fields when sweeping is enabled. Outputs a dict suitable for
``expand_param_grid()``.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ...algorithms.base import AlgorithmParam
from ...algorithms.registry import AlgorithmRegistry


def render_sweep_form(
    algo_id: str,
    container: ui.element,
    sweep_config: dict[str, dict[str, Any]],
) -> None:
    """Render sweep parameter form for the selected algorithm.

    Clears the container and populates it with per-parameter sweep controls.
    Values are synced into ``sweep_config`` dict with the format expected by
    ``expand_param_grid()``.
    """
    container.clear()
    sweep_config.clear()

    try:
        algo = AlgorithmRegistry.get(algo_id)
    except KeyError:
        return

    params = algo.manifest.params
    if not params:
        return

    with container:
        ui.label("Parameter Ranges").classes("text-subtitle2 text-grey-5 q-pt-xs")
        for p in params:
            if p.type in ("int", "float") and p.min is not None and p.max is not None:
                _render_sweep_row(p, sweep_config)
            else:
                # Non-numeric or unbounded — fixed value only
                sweep_config[p.name] = {
                    "sweep": False,
                    "value": p.default,
                    "type": p.type,
                }


def _render_sweep_row(param: AlgorithmParam, config: dict[str, dict[str, Any]]) -> None:
    """Render one parameter row with sweep toggle and range fields."""
    label = param.name.replace("_", " ").title()
    default_step = 1 if param.type == "int" else (param.step or 0.1)

    # Initialize config entry
    config[param.name] = {
        "sweep": False,
        "value": param.default,
        "min": param.min,
        "max": param.max,
        "step": default_step,
        "type": param.type,
    }

    with ui.row().classes("w-full items-center gap-2 q-py-xs"):
        sweep_toggle = ui.switch(f"Sweep {label}", value=False).classes("min-w-[180px]")

        # Fixed value field (shown when not sweeping)
        fixed_field = ui.number(
            f"{label}",
            value=param.default,
            min=param.min,
            max=param.max,
            step=1 if param.type == "int" else (param.step or 0.01),
        ).classes("min-w-[80px]")

        # Range fields (shown when sweeping)
        min_field = ui.number("Min", value=param.min, step=default_step).classes("min-w-[70px]")
        max_field = ui.number("Max", value=param.max, step=default_step).classes("min-w-[70px]")
        step_field = ui.number("Step", value=default_step, min=0.001).classes("min-w-[70px]")

        min_field.visible = False
        max_field.visible = False
        step_field.visible = False

        def _on_toggle(e, name=param.name):
            sweeping = e.value
            config[name]["sweep"] = sweeping
            fixed_field.visible = not sweeping
            min_field.visible = sweeping
            max_field.visible = sweeping
            step_field.visible = sweeping

        def _on_fixed(e, name=param.name, ptype=param.type):
            val = e.value
            if val is not None:
                config[name]["value"] = int(val) if ptype == "int" else float(val)

        def _on_min(e, name=param.name, ptype=param.type):
            if e.value is not None:
                config[name]["min"] = int(e.value) if ptype == "int" else float(e.value)

        def _on_max(e, name=param.name, ptype=param.type):
            if e.value is not None:
                config[name]["max"] = int(e.value) if ptype == "int" else float(e.value)

        def _on_step(e, name=param.name, ptype=param.type):
            if e.value is not None:
                config[name]["step"] = int(e.value) if ptype == "int" else float(e.value)

        sweep_toggle.on_value_change(_on_toggle)
        fixed_field.on_value_change(_on_fixed)
        min_field.on_value_change(_on_min)
        max_field.on_value_change(_on_max)
        step_field.on_value_change(_on_step)
