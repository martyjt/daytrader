# Dev Runbook

Get Daytrader running locally in under 5 minutes. Two paths: **native**
(fastest for iteration) or **Docker** (closest to production).

---

## Option A: Native (recommended for daily dev)

Uses SQLite — no Docker, no Postgres, no Redis. Everything in one process.

### Prerequisites

- Python 3.12+ (tested on 3.14.3)
- Git
- Windows: use `python`, not `python3`

### Steps

```bash
# 1. Clone and enter
git clone https://github.com/zimpla/daytrader.git
cd daytrader

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

# 3. Install (editable + dev tools)
pip install -e ".[dev]"
```

> **If the full install fails** (some ML/data deps may not have wheels
> for Python 3.14 yet), install the minimum needed for Phase 0:
>
> ```bash
> pip install nicegui polars yfinance pandas numpy sqlalchemy[asyncio] \
>             asyncpg aiosqlite pydantic pydantic-settings pyyaml \
>             cryptography pytest pytest-asyncio aiosqlite
> pip install -e . --no-deps
> ```

```bash
# 4. Create your .env (SQLite for native dev)
cp .env.example .env
```

Now edit `.env` — set these two values for native dev:

```ini
DATABASE_URL=sqlite+aiosqlite:///daytrader.db
APP_ENV=dev
APP_SECRET_KEY=any-string-at-least-32-chars-long
```

Everything else can stay at defaults.

```bash
# 5. Launch
python -m daytrader
```

Open **http://localhost:8080** in your browser.

### What happens on first launch

1. SQLite database `daytrader.db` is created in the project root
2. All tables are auto-created (dev mode skips Alembic)
3. A default tenant and operator user are seeded
4. Built-in algorithms and data adapters are registered
5. NiceGUI serves the app on port 8080

### Run tests

```bash
# All tests (unit + integration) — no network, no Docker needed
pytest tests/ -v

# Just unit tests
pytest tests/unit -v

# Just the end-to-end ritual flow
pytest tests/integration/test_e2e.py -v
```

---

## Option B: Docker (production-like)

Runs the full stack: app + Postgres/TimescaleDB + Redis + mlflow.

### Prerequisites

- Docker Desktop (running)
- Git

### Steps

```bash
# 1. Clone and enter
git clone https://github.com/zimpla/daytrader.git
cd daytrader

# 2. Create .env from example
cp .env.example .env
```

Edit `.env`:

```ini
# These are fine for local Docker:
DATABASE_URL=postgresql+asyncpg://daytrader:daytrader@postgres:5432/daytrader
APP_ENV=dev
APP_SECRET_KEY=change-me-to-something-long

# IMPORTANT: if you have other containers using these ports, change them:
APP_HOST_PORT=8080
POSTGRES_HOST_PORT=5432
REDIS_HOST_PORT=6379
MLFLOW_HOST_PORT=5000
```

```bash
# 3. Build and start
docker compose up -d

# 4. Check status
docker compose ps
docker compose logs app --tail 20
```

Open **http://localhost:8080** (or whatever APP_HOST_PORT you set).

### Check port conflicts

If startup fails, check if something else is using the ports:

```bash
# Windows
netstat -ano | findstr :5432
netstat -ano | findstr :6379
netstat -ano | findstr :5000
netstat -ano | findstr :8080
```

Fix by changing `*_HOST_PORT` values in `.env`.

### Run Alembic migrations (production path)

```bash
# Inside the container:
docker compose exec app alembic upgrade head

# Or from host (if you have the venv):
alembic upgrade head
```

For dev mode (`APP_ENV=dev`), tables are auto-created on boot — you
don't need to run migrations manually.

### Stop everything

```bash
docker compose down          # stop containers, keep data
docker compose down -v       # stop AND delete data volumes (fresh start)
```

---

## Walkthrough: the 5-minute demo

Once the app is running (either native or Docker):

1. **Home** (`/`) — empty dashboard, shows "No personas yet"

2. **Personas** (`/personas`) — click **+ New Persona**:
   - Name: `BTC Paper Tester`
   - Asset Class: `crypto`
   - Base Currency: `USDT`
   - Initial Capital: `10000`
   - Risk Profile: `balanced`
   - Click **Create**

3. **Strategy Lab** (`/strategy-lab`):
   - Algorithm: `Buy & Hold`
   - Symbol: `BTC-USD`
   - Timeframe: `1d`
   - Start Date: `2024-01-01`
   - End Date: `2024-12-31`
   - Capital: `10000`
   - Click **Run Backtest**
   - Wait ~3 seconds (fetches real data from Yahoo Finance)
   - See: equity curve, Sharpe ratio, max drawdown, total return

4. **Plugins** (`/plugins`) — see Buy & Hold registered, plus
   placeholders for Phase 2 algorithms

5. **Risk Center** (`/risk`) — kill switch, risk rules, persona table

6. **Home** (`/`) — now shows your persona card with equity

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python3` fails with exit code 49 | Use `python` on Windows (the `python3` alias points to the Store stub) |
| `ModuleNotFoundError: No module named 'X'` | Install the missing dep: `pip install X`, or use the minimal install command above |
| Port 8080 already in use | Set `APP_HOST_PORT=8081` in `.env` (native: change `APP_PORT` instead) |
| yfinance backtest says "no data" | Check the symbol format: crypto = `BTC-USD`, equity = `AAPL`, forex = `EURUSD=X` |
| Docker Postgres won't start | Another container is on port 5432. Set `POSTGRES_HOST_PORT=5433` in `.env` |
| Tests fail with `aiosqlite` error | `pip install aiosqlite` — it's needed for SQLite async tests |
| `cryptography` build fails | On Windows, install via `pip install cryptography` — wheels exist for most Python versions |

## Project structure (quick reference)

```
daytrader/
├── src/daytrader/
│   ├── core/           ← domain types, events, AlgorithmContext, settings
│   ├── storage/        ← SQLAlchemy models, tenant-scoped repository, migrations
│   ├── data/           ← DataAdapter ABC, yfinance adapter, feature store
│   ├── algorithms/     ← Algorithm ABC, registry, built-in (buy_hold)
│   ├── backtest/       ← BacktestEngine (bar-by-bar simulation + KPIs)
│   ├── execution/      ← ExecutionAdapter ABC, PaperExecutor
│   └── ui/             ← NiceGUI app, shell, 5 pages, services layer
├── tests/
│   ├── unit/           ← 65 tests (no network, no DB, fast)
│   └── integration/    ← 3 tests (SQLite in-memory, end-to-end ritual)
├── plugins/            ← user-installable algorithm plugins (Phase 2)
├── config/             ← default.yaml (risk rules, ritual gates, UI config)
└── docker/             ← Postgres init scripts
```
