# Example plugins

Three starter plugins to copy and fork:

| Folder | What it shows |
| --- | --- |
| `buy_hold/` | Minimal valid plugin — `manifest.yaml` + a single-method `Algorithm` subclass. |
| `rsi_threshold/` | Reading parameters from the manifest, declaring `warmup_bars()`, computing an indicator from history. |
| `template/` | A no-op skeleton designed for forking. Has every manifest field filled in with a comment. |

## Loading these into your tenant

The runtime auto-loads any plugin under `plugins/<name>/manifest.yaml`
on startup. Examples live one level deeper (`plugins/examples/<name>/`)
so they don't auto-register — copy or symlink the folder you want into
`plugins/` (or upload the zipped folder via `Plugins → Upload` in the
UI) and restart.

## The AlgorithmContext API

Every plugin implements one method:

```python
def on_bar(self, ctx: AlgorithmContext) -> Signal | None:
    ...
```

`ctx` is a narrow sandbox. You **can**:

- Read the current bar: `ctx.bar.open / .high / .low / .close / .volume`
- Read recent history: `ctx.history(n, "close")` returns the last `n`
  closes as a NumPy array. Other fields: `"open"`, `"high"`, `"low"`,
  `"volume"`.
- Read precomputed features: `ctx.feature("sma_50")`, with
  `ctx.has_feature(name)` to test first.
- Read declared parameters: `ctx.param("threshold")`.
- Emit a signal: `ctx.emit(score, confidence=..., reason=..., metadata=...)`.
  - `score`: float in `[-1, 1]`. Positive is bullish, negative bearish.
  - `confidence`: float in `[0, 1]`. How sure you are.
- Log: `ctx.log("rsi update", rsi=42.0, threshold=30.0)`.

You **cannot**:

- Touch the database, broker, filesystem, or other algorithms.
- Import from `daytrader.storage` or `daytrader.execution`.
- Mutate global state — instances are recreated per persona per run.

If you need a feature that's not exposed via `ctx.feature()`, add a
declaration to your `manifest.yaml`'s `features:` list (Phase 6+) and
the runtime will compute it before invoking your `on_bar`.

## manifest.yaml schema (full)

```yaml
id: my_algo                # unique snake_case id
name: "My Algorithm"       # human-readable
version: 0.1.0
author: Your Name
description: "One-liner."
asset_classes: [crypto, equities]
timeframes: [1h, 4h, 1d]
suitable_regimes: [bull, sideways]   # optional; omit for "agnostic"
params:
  - name: fast_period
    type: int                         # int | float | bool | str
    default: 12
    min: 2
    max: 200
    description: "Lookback for the fast moving average"
```

The runtime calls `instance.warmup_bars()` before serving the first bar
to your algo. Override it if you need any history:

```python
def warmup_bars(self) -> int:
    return self.manifest.param_defaults()["period"] + 1
```
