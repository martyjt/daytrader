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
