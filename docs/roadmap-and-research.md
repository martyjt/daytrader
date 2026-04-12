# Daytrader — Roadmap & Research

Ideas to steal from competitors, missing features, and data source options.
Not a commitment — a menu to pick from.

---

## 1. Features to borrow from competitors

### Natural-language DAG explainer (from Composer.trade)

**What it does:** Composer shows a plain-English walkthrough of any strategy —
"if RSI(14) is below 30 over the last 10 days, buy BTC; if MACD crosses down,
exit." Non-technical users understand what the DAG actually does without
reading graph nodes.

**Why we want it:** The DAG Composer is great for builders but opaque for
reviewers. A natural-language summary auto-generated from the DAG structure
would make compositions auditable and shareable.

**Rough approach:**
- Walk the DAG in topological order
- Each algorithm and combinator implements a `explain()` method that returns
  a sentence fragment ("MACD histogram crossed from negative to positive")
- Compose fragments under the combinator logic ("…AND ADX is above 20 AND
  RSI was recently below 30")
- Render as a card above the canvas + inside the backtest result

**Effort:** 1-2 days. Pure presentation layer, no backend changes.

---

### Chart-first live market view (from TradingView)

**What it does:** Candle chart with algorithm signals drawn on top — buy/sell
markers, indicator overlays, position highlight bars. Live-updating.

**Why we want it:** We have no live market screen at all. TradingView's UX is
the standard everyone expects from a trading app. Currently users have no way
to visually see what their algo is "seeing".

**Rough approach:**
- New page `/live-market` with ECharts candlestick + volume sub-pane
- Subscribe to WebSocket market data (or poll with our existing data adapters)
- Overlay algorithm signal markers using `dag_attribution` metadata
- Toggle indicators on/off via checkboxes
- Link to active personas so their open positions show as horizontal bands

**Effort:** 3-5 days. Needs live data streaming layer (polling is a cheap
start).

**Dependencies:** Better real-time data source (see section 2).

---

### Universe selection (from QuantConnect)

**What it does:** Strategies don't just run on one symbol — they run on a
filtered universe. Example: "S&P 500 members where 20-day volume > X", or
"top 50 crypto by market cap excluding stablecoins".

**Why we want it:** Right now every backtest is one symbol. Universe selection
turns one strategy into a portfolio-style backtest: best signals across many
instruments, with position sizing across them.

**Rough approach:**
- New `Universe` abstraction: takes a filter expression + data source, returns
  a list of `Symbol` instances refreshed daily/weekly
- Built-in universes: "All S&P 500", "Top 20 crypto", "Forex majors"
- Strategy Lab gets a toggle: "Single symbol" vs "Universe"
- Backtest engine loops over symbols and aggregates results

**Effort:** 1-2 weeks. Needs portfolio-level position sizing and cross-symbol
capital allocation. Real engineering work.

---

### Fast parameter sweep via vectorization (from vectorbt)

**What it does:** vectorbt can backtest 100,000+ parameter combinations in
seconds by running the entire simulation as numpy array operations instead of
a Python for-loop.

**Why we want it:** Our current sweep is the same simulation loop N times.
At ~5 seconds per backtest, a 1000-point sweep is an hour. vectorbt would do
the same sweep in seconds.

**Rough approach:**
- Not a full rewrite — add a parallel "fast path" for algorithms that expose
  a `vectorized_scores(closes, highs, lows, volumes) -> np.array` method
- Sweep engine uses the fast path when available, falls back to iterative
  simulation otherwise
- Only indicator-based algos can go vectorized (ML/DL can't)

**Effort:** 1 week. Biggest wins on the existing sweep UI.

---

### Analyzers and post-backtest reports (from backtrader)

**What it does:** Pluggable post-backtest analyzers that compute extra metrics:
Sortino ratio, Calmar ratio, drawdown distribution, trade MFE/MAE (maximum
favorable/adverse excursion), time in market, exposure analysis, rolling
Sharpe.

**Why we want it:** Our KPIs are solid but limited. Serious strategy review
needs more than Sharpe + max DD. MFE/MAE in particular is great for spotting
"strategies that would have been profitable if they'd held longer".

**Rough approach:**
- `BacktestAnalyzer` ABC with `compute(result)` → dict of metrics
- Built-in analyzers: Sortino, Calmar, rolling Sharpe, trade MFE/MAE,
  exposure %, win streak/loss streak
- Strategy Lab renders a second KPI row below the main cards

**Effort:** 2-3 days. Pure analysis on top of existing `BacktestResult`.

---

### Strategy sharing & fork system (from Kryll / Composer)

**What it does:** Users publish strategies to a shared library. Others browse,
preview, fork, and modify. Creates a social layer.

**Why we want it:** Low priority for local-first use, but a nice-to-have.
Could be as simple as "export/import a shareable URL encoding the DAG YAML".

**Rough approach:**
- Phase 1: "Export as shareable link" → base64-encoded DAG YAML in URL
- Phase 2: Public strategy index on a shared backend

**Effort:** 1 day for Phase 1. Phase 2 is indefinite (backend, moderation,
etc.).

---

### Realistic execution modelling (from every pro platform)

**What it does:** Proper simulation of bid-ask spread, partial fills, market
impact, slippage curves, corporate actions (splits, dividends), after-hours
gaps, funding rate costs for perpetuals.

**Why we want it:** Our fee model is solid but everything else is idealized.
Fill-at-bar-close is fictitious. Serious backtests need serious execution
models or they lie to you.

**Rough approach:**
- Order book simulator (can be approximate): assume bid = mid × (1 - spread/2),
  ask = mid × (1 + spread/2), fill limit orders deterministically, market
  orders eat spread
- Partial-fill logic for orders larger than `volume * X%`
- Corporate action handling from yfinance (already returns adjusted prices)

**Effort:** 1-2 weeks for a first pass. Requires rethinking the simulation
loop.

---

### Algorithm-level grouping in the DAG Composer

**What it does:** Collapse a group of related nodes into a single "meta" node.
E.g. put MACD + ADX + RSI + combinator into a single "Consensus" group that
other DAGs can reuse.

**Why we want it:** Currently composites are per-file. Reusable sub-DAGs would
let us build libraries of common patterns ("trend filter", "mean reversion
setup", "regime classifier").

**Rough approach:**
- Add `group_id` to `DAGNode`
- "Group selected" action in the composer
- When saving, a grouped set can be saved as a reusable "Algorithm Group"
  that appears in the Add Algorithm dropdown

**Effort:** 3-5 days. Non-trivial validation logic (nested DAGs, cycle
detection through groups).

---

### In-browser algorithm editor (Monaco)

**What it does:** Write a new Algorithm in Python directly in the app, with
syntax highlighting, autocomplete for `ctx.emit/ctx.history`, and "save as
plugin" to hot-reload it into the registry.

**Why we want it:** Right now users have to drop Python files into `plugins/`
manually. A first-class in-app editor is friendlier and keeps everything
inside the tool.

**Rough approach:**
- New page `/algo-editor` with CodeMirror (NiceGUI has a wrapper)
- Template starter: prefilled `class MyAlgo(Algorithm)` skeleton
- "Save & Load" button writes to `plugins/<id>.py` and calls
  `AlgorithmRegistry.load_plugins()` to register it

**Effort:** 2-3 days. Main risk is sandboxing — user code can do anything.

---

### Trade MFE/MAE inspection UI

**What it does:** For each trade, show a sparkline of price movement from
entry to exit with markers for the Maximum Favorable Excursion and Maximum
Adverse Excursion. Reveals "I exited too early" and "the stop was too tight"
patterns.

**Why we want it:** After the 6-trade narrative walkthrough we did earlier,
it's clear we need a way to dive into individual trades and see what they
looked like intra-position.

**Rough approach:**
- Backtest engine already captures bar index per trade — add per-bar P&L
  tracking during a held position
- New "Trades" section in Strategy Lab results with expandable rows: click a
  trade, see the intra-trade chart

**Effort:** 3-5 days.

---

## 2. Better data sources to investigate

Our current yfinance adapter is limited:
- 1h data capped at ~730 days
- 1m data capped at ~7 days
- No streaming/live feed
- No fundamental data
- Rate-limited (unofficial, but real)

### Crypto

**CCXT (already partially supported)**
- Unified API across 100+ exchanges
- Historical candles via exchange-native endpoints (free, no auth for most)
- Live websocket streams via `ccxt.pro` (paid)
- Best for: BTC/ETH/alts, 1m-1w, real-time
- Caveat: Binance.US, Coinbase, etc. have different coverage

**Binance API (direct)**
- Free, no auth needed for historical candles
- Very deep history (years of 1m data)
- Native websocket streams for free
- Best for: single-venue deep history and live data
- Already wired via existing BinanceAdapter (Phase 4)

**Coinbase API (direct)**
- Free, no auth for public market data
- Similar depth to Binance
- Good for: US-regulated venue pricing

**Kaiko (paid)**
- Institutional-grade crypto market data
- Orderbook snapshots, trades, L2 data
- Best for: realistic execution modelling
- $$$$ — probably overkill for retail-ish use

### Equities

**Alpaca Markets (already supported via AlpacaAdapter)**
- Free IEX tier: 15-min delayed
- Paid SIP tier: real-time, full-market
- Historical: 5+ years of daily, intraday depends on plan
- Built-in execution — our existing Alpaca adapter already uses this
- Best current option for US equities

**Polygon.io**
- $30/month for 2 years of historical intraday
- $200/month for real-time + websockets
- Much better than yfinance for equities
- Clean API, good docs
- Fundamental data (earnings, splits, corporate actions)
- **Strong candidate** for the next adapter

**IEX Cloud** (shutting down / transitioning)
- Was a great free tier, now being wound down. Skip.

**Financial Modeling Prep (FMP)**
- $15/month tier covers most retail needs
- Historical intraday, fundamentals, earnings
- Good for: backtesting with fundamentals
- Worth a trial

**Databento**
- Pay-per-use institutional data
- Order book L2/L3, tick data
- Best for: serious intraday strategies
- Probably overkill for now

### Forex & commodities

**OANDA v20 API**
- Free for personal use
- Real-time tick data via streaming
- Historical 1m back ~5 years
- Best for: forex majors
- Our existing code has no forex adapter — this would be the obvious first

**TwelveData**
- Free tier: 800 requests/day, most asset classes
- $29/month for real-time
- Unified API across stocks, forex, crypto, commodities
- Best if we want one adapter for everything

### Fundamental / news / alternative data

**Yahoo Finance Fundamentals (via yfinance)**
- Already available as a byproduct of yfinance
- Free but limited and unreliable
- Good enough for "filter by P/E ratio" type universe rules

**FRED (St. Louis Fed)**
- Macroeconomic series: interest rates, unemployment, CPI, GDP
- Already wired via existing FREDAdapter
- Great for: regime-switching strategies, macro filters

**News / sentiment**
- NewsAPI.org — free tier 100 requests/day
- RavenPack — institutional ($$$$)
- Twitter/X API — expensive post-changes
- Reddit API — free, good for crypto sentiment

### Recommended next adapters

1. **Polygon.io** — best equities upgrade from yfinance. Would fix the "1h
   data capped at 730 days" issue permanently. Probably $30/mo.

2. **Binance direct REST/WebSocket** — already partially done via
   BinanceAdapter but needs the live streaming layer for the live market
   screen. Free.

3. **OANDA v20** — fills the forex gap entirely and is free.

4. **CCXT Pro** (later) — unified real-time crypto across all exchanges.
   Costs money but covers every crypto venue at once.

---

## 3. Unprioritized shopping list

Small things worth doing when there's 30 minutes free:

- **Chart the trade P&L distribution** — histogram of per-trade returns
- **Export backtest result as JSON/CSV** — for external analysis
- **Comparison against buy-and-hold baseline** automatically on every
  backtest
- **"Copy algorithm to plugins/" button** — for forking built-in algos
- **Keyboard shortcuts in the DAG Composer** — `Delete` to remove selected
  node, `Ctrl+S` for save
- **Dark/light theme toggle** — currently dark-only
- **DAG version control** — save multiple versions of a DAG under one name,
  roll back easily
- **Email/SMS alerts on fills** — important for live mode
- **Max concurrent positions per persona** — currently configured, not
  enforced
- **CSV import adapter** — let users backtest on their own OHLCV files

---

## 4. Known pain points (not really features — fix list)

Things that already bit us in this session, worth a proper fix:

- **Shutdown bug with `app.state.get()`** — already patched, but exposes
  brittleness around NiceGUI state access. Audit all `app.state` usage.
- **Browser cache defeats code changes** — every code change requires a
  hard-refresh. Look into NiceGUI's cache-busting options or serve an
  always-stale `_dt_version` string that changes on restart.
- **WebSocket disconnect during long backtests** — fixed by running `_simulate`
  in a worker thread + periodic GIL release. Consider the same fix
  proactively for any other long-running handler.
- **Warmup + walk-forward fold math is fragile** — auto-scales now, but the
  scaling logic is opinionated. Document the heuristic and expose it in the
  UI so users understand why they see 2 folds instead of 5.
- **"Connection lost" toast stays in DOM** after reconnect — minor cosmetic
  issue but looks alarming.

---

_Last updated: 2026-04-13_
