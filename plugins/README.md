# Plugins

This directory holds user-installable algorithm plugins. Each plugin is a
Python package with a `manifest.yaml`:

```
plugins/
└── my_algo/
    ├── manifest.yaml
    ├── __init__.py
    └── algorithm.py
```

## manifest.yaml schema

```yaml
id: my_algo
name: "My Algorithm"
version: 0.1.0
author: Your Name
description: "One-line description of what this algo does."
asset_classes: [crypto, equities]
timeframes: [1h, 4h, 1d]
params:
  - name: fast_period
    type: int
    default: 12
    min: 2
    max: 200
  - name: slow_period
    type: int
    default: 26
    min: 2
    max: 500
dependencies: []
```

## Algorithm contract

Plugins subclass `daytrader.algorithms.base.Algorithm` and implement a
single `on_bar(context) -> Signal | None` method. The `AlgorithmContext`
is the sandbox: it exposes features and a signal emitter, and nothing
else. No DB, no broker, no global state.

Full contract documented in Phase 1 when the `algorithms/base.py` module
lands.
