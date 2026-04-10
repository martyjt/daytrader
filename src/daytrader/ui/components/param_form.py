"""Auto-generated parameter form from AlgorithmParam declarations.

Renders NiceGUI widgets based on the parameter type, default, min/max,
and choices defined in an algorithm's manifest.
"""

from __future__ import annotations

from typing import Any

from nicegui import ui

from ...algorithms.base import AlgorithmParam
from ...algorithms.registry import AlgorithmRegistry


def render_param_form(
    algo_id: str,
    container: ui.element,
    param_values: dict[str, Any],
) -> None:
    """Render parameter form for the selected algorithm.

    Clears the container and populates it with widgets matching the
    algorithm's ``manifest.params``.  Widget values are synced into
    ``param_values`` dict.
    """
    container.clear()

    try:
        algo = AlgorithmRegistry.get(algo_id)
    except KeyError:
        return

    params = algo.manifest.params
    if not params:
        return

    with container:
        ui.label("Algorithm Parameters").classes("text-subtitle2 text-grey-5 q-pt-xs")
        with ui.row().classes("w-full gap-4 items-end flex-wrap"):
            for p in params:
                _render_widget(p, param_values)


def _render_widget(param: AlgorithmParam, values: dict[str, Any]) -> None:
    """Render a single parameter widget and bind its value to the dict."""
    # Initialize with default
    values.setdefault(param.name, param.default)

    label = param.name.replace("_", " ").title()
    if param.description:
        label = f"{label}"

    if param.type == "bool":
        sw = ui.switch(label, value=param.default)
        sw.on_value_change(lambda e, n=param.name: values.__setitem__(n, e.value))

    elif param.type == "str" and param.choices:
        sel = ui.select(
            param.choices,
            value=param.default,
            label=label,
        ).classes("min-w-[120px]")
        sel.on_value_change(lambda e, n=param.name: values.__setitem__(n, e.value))

    elif param.type == "str":
        inp = ui.input(label, value=str(param.default))
        inp.on_value_change(lambda e, n=param.name: values.__setitem__(n, e.value))

    elif param.type == "int":
        num = ui.number(
            label,
            value=param.default,
            min=param.min,
            max=param.max,
            step=1,
        ).classes("min-w-[100px]")
        num.on_value_change(
            lambda e, n=param.name: values.__setitem__(n, int(e.value) if e.value is not None else param.default)
        )

    else:  # float
        num = ui.number(
            label,
            value=param.default,
            min=param.min,
            max=param.max,
            step=param.step or 0.01,
        ).classes("min-w-[100px]")
        num.on_value_change(
            lambda e, n=param.name: values.__setitem__(n, float(e.value) if e.value is not None else param.default)
        )
