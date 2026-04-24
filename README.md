# Daytrader

Multi-asset algorithmic trading platform — crypto first, equities next.

Successor to CryptoTrader. Built for solo operators (initially) who want
to go from **idea → backtest → paper-trade → confidence** without friction,
then grow into a multi-tenant tool with isolated workspaces.

## Design pillars

1. **Isolation of tasks.** Every algorithm is a sandboxed plugin; no
   global state, no direct broker access. Composition is safe by
   construction.
2. **Simplicity of forming algorithms.** Form-based composer for 90% of
   users, visual DAG composer for power users — both produce the same YAML.
3. **Usable screens.** Pages are organized around *jobs* (create a
   persona, prove an algo, watch the book) not *tools* (backtest,
   forward-test, walk-forward).

Everything pivots around **the Ritual**:

```
Idea → Configure → Backtest → Walk-forward → Paper → Promote → Live
```

One page (Strategy Lab), one timeline, one Promote button per gate.

## Stack

Python 3.12 · NiceGUI · Postgres + TimescaleDB · Redis · mlflow ·
Polars · backtesting.py (Phase 0) / vectorbt (Phase 3+) · XGBoost ·
ccxt · alpaca-py · yfinance · AG Grid Enterprise · AG Charts Enterprise.

## Quickstart — Docker (recommended)

```bash
cp .env.example .env
# edit .env, especially APP_SECRET_KEY, APP_ENCRYPTION_KEY, AG_GRID_LICENSE_KEY
docker compose up -d
# open http://localhost:8080
```

## Quickstart — native (Windows)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
python -m daytrader
```

> **Windows note:** use `python`, not `python3`. `python3` on Windows
> resolves to the Store stub and fails.

## Architecture

Layered, plugin-first, multi-tenant from day one:

```
┌─────────────────────────────────────────────────────────┐
│  UI (NiceGUI pages — 8 job-oriented screens)           │
├─────────────────────────────────────────────────────────┤
│  Application services (tenant-scoped)                  │
├─────────────────────────────────────────────────────────┤
│  Personas · Strategy Engine · Risk Guardrails          │
├─────────────────────────────────────────────────────────┤
│  Algorithms (plugins) · Indicators · ML Models         │
├─────────────────────────────────────────────────────────┤
│  Feature Store (Polars + Parquet cache)                │
├─────────────────────────────────────────────────────────┤
│  Data Adapters (ccxt · alpaca · yfinance · fred · ...) │
├─────────────────────────────────────────────────────────┤
│  Execution Adapters (paper · Binance · Alpaca · ...)   │
└─────────────────────────────────────────────────────────┘
```

## Phase plan

| Phase | Scope |
|---|---|
| **0** | Scaffolding, one data adapter (yfinance), one dummy algo, end-to-end paper trade, tenant-aware schema |
| **1** | The Ritual end-to-end · first ML algo (XGBoost trend classifier) · risk layers 1–2 · explainability tree · mlflow |
| **2** | 10 baseline algorithms salvaged from CryptoTrader · Plugins page · Strategy Lab tier-1 form composer · Alpaca equities adapter · real auth |
| **3** | DAG composer (tier 2) · full algo library · regime-switching HMM |
| **4** | Live execution (Binance, Alpaca) · global risk layer 3 · kill-switch · journal |
| **5** | Deep learning (PyTorch) · advanced research tooling |
| **6** | Charts Workbench · data expansion (FRED/NewsAPI/Binance public/Alpha Vantage/Twelve Data) · composite param exposure · data cache |
| **7** | Reinforcement learning (PPO, SAC, Bandit Allocator) · Exploration Agent · drift monitor · shadow tournament · portfolio backtest · alert center · live regime watcher |

## Feature map (current)

14 pages in the left nav:

| Page | Job |
|---|---|
| Home | Dashboard — personas, equity, regime, alerts, top discoveries |
| Personas | Create / manage personas; bind to a saved strategy |
| Persona detail (`/persona/<id>`) | Single persona: bound strategy + activity |
| Strategy Lab | The Ritual (backtest → walk-forward → paper → promote) with `?strategy=<uuid>` preload |
| Strategies | Save & reuse named algo+params+symbol+venue recipes |
| Charts | Multi-algorithm charting workbench with agreement ribbon + DAG attribution |
| DAG Composer | Visual DAG editor for composite strategies |
| Bandit Builder | Thompson-sampling allocator composer — saves YAML, hot-registers |
| Universes | Reusable symbol watchlists for Portfolio + Shadow |
| Plugins | Algorithm library · single-file plugin upload · rescan |
| Risk Center | Kill switch · per-persona risk · cross-persona correlation monitor |
| Signals | Live auto-refreshing feed of every signal across personas |
| Journal | Full event log with attribution drawer · CSV export |
| Research Lab | 7 tabs: Model Comparison · Parameter Sweep · Feature Attribution · WF Stability · **Discoveries** · **Shadow** · **Portfolio** |
| Data Cache | Parquet cache inspector + clear |

### Always-visible shell widgets

- **Regime badge** — live HMM-inferred regime (bull/bear/sideways) with 30-min auto-refresh
- **Alerts bell** — in-process ring buffer; regime changes, drift, winners, correlation breaches
- **Kill-all** — global emergency halt

### Background workers (opt-in via env)

- **RegimeWatcher** — periodic HMM refresh (default 30 min, always on)
- **ExplorationScheduler** — periodic feature-lift scans (`EXPLORATION_SCHEDULE_HOURS=6`)
- **ShadowScheduler** — periodic tournament runs (`SHADOW_SCHEDULE_HOURS=24`)

## Optional extras

```bash
pip install daytrader[rl]         # stable-baselines3 + gymnasium (PPO, SAC)
pip install daytrader[dl]         # PyTorch (LSTM/Transformer/CNN-LSTM)
pip install daytrader[sentiment]  # VADER inline scoring for NewsAPI
pip install daytrader[dev]        # pytest + ruff + mypy
```

## Environment

See `.env.example`. Most adapters are key-gated and skip cleanly when the
key is absent. Two adapters (`yfinance`, `binance_public`) work with no
configuration.
