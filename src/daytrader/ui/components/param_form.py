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

    # Group params by the "[Name] ..." prefix in their description if
    # present (used by CompositeAlgorithm to expose child params).
    groups: dict[str, list[AlgorithmParam]] = {}
    for p in params:
        desc = p.description or ""
        if desc.startswith("[") and "]" in desc:
            group = desc[1 : desc.index("]")]
        else:
            group = ""
        groups.setdefault(group, []).append(p)

    with container:
        ui.label("Algorithm Parameters").classes("text-subtitle2 text-grey-5 q-pt-xs")
        # Render ungrouped first, then each group under its own header
        if "" in groups:
            with ui.row().classes("w-full gap-4 items-end flex-wrap"):
                for p in groups[""]:
                    _render_widget(p, param_values)
        for group_name, group_params in groups.items():
            if not group_name:
                continue
            ui.label(group_name).classes(
                "text-caption text-grey-6 q-pt-sm q-mb-xs"
            )
            with ui.row().classes("w-full gap-4 items-end flex-wrap"):
                for p in group_params:
                    _render_widget(p, param_values)


def _render_widget(param: AlgorithmParam, values: dict[str, Any]) -> None:
    """Render a single parameter widget and bind its value to the dict."""
    # Seed with the manifest default unless the caller pre-populated a
    # value (e.g. Strategy Lab loading a saved strategy recipe). Widgets
    # then render using the current value, not the manifest default.
    values.setdefault(param.name, param.default)
    initial = values[param.name]

    # For composite child params the internal name is ``{node}__{param}``;
    # show just the final param for a clean label.
    display_name = param.name.split("__")[-1]
    label = display_name.replace("_", " ").title()

    if param.type == "bool":
        sw = ui.switch(label, value=bool(initial))
        sw.on_value_change(lambda e, n=param.name: values.__setitem__(n, e.value))

    elif param.type == "str" and param.choices:
        sel = ui.select(
            param.choices,
            value=initial,
            label=label,
        ).classes("min-w-[120px]")
        sel.on_value_change(lambda e, n=param.name: values.__setitem__(n, e.value))

    elif param.type == "str":
        inp = ui.input(label, value=str(initial))
        inp.on_value_change(lambda e, n=param.name: values.__setitem__(n, e.value))

    elif param.type == "int":
        num = ui.number(
            label,
            value=initial,
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
            value=initial,
            min=param.min,
            max=param.max,
            step=param.step or 0.01,
        ).classes("min-w-[100px]")
        num.on_value_change(
            lambda e, n=param.name: values.__setitem__(n, float(e.value) if e.value is not None else param.default)
        )
