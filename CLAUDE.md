# CLAUDE.md — Pairs Trading Platform

Guidance for Claude working in this repo. Read this first, then the docs it points to.

## What this project is

Official topic (set by the mentor):
**"Projekt i implementacja modułowej platformy handlu algorytmicznego wykorzystującej strategię Pairs Trading oraz uczenie maszynowe do optymalizacji decyzji inwestycyjnych"**

A modular algorithmic-trading **research/simulation** platform in Python: pick S&P 500 stocks → find cointegrated pairs → run a pairs-trading strategy (optionally filtered by an ML model) → backtest walk-forward → inspect results via API + Streamlit UI. It never trades real money.

Main research question: **does ML improve classical pairs trading?** A negative result ("ML adds nothing") is a valid conclusion if it is measured correctly. Methodology matters more than nice numbers.

- University team project (Poland), 3 people, two semesters: **09.11.2026 – 31.05.2027** (~29 weeks, ~24 effective after breaks and the exam session).
- Mentor requirements (non-negotiable, do not drop any of them):
  - The project must cover **three domains**: software engineering, AI, and algorithmics/data analysis.
  - The platform must let the user: (1) choose the instruments to analyse, (2) automatically find statistically related pairs, (3) choose a base strategy and an ML model, (4) configure trading and risk parameters, (5) run a simulation on historical data, (6) view results, charts and trade history.
  - O3 scope from the mentor: backend, architecture, database, **own backtesting engine / exchange simulator ("najlepiej samodzielnie")**, API, frontend, automated tests, deployment.
- Universe: **S&P 500 stocks**, designed so other instruments (ETFs etc.) can be added later. No deployment target exists: "deployment" = Docker Compose, one command runs everything locally.

## Team and roles

- **O1 — strategy and data analysis:** pair selection (PCA/clustering, sectors, cointegration), hedge ratios, Z-score signals, strategy plugins, risk/position sizing, performance metrics, statistical validation, pair charts in the UI.
- **O2 — machine learning:** feature engineering, triple-barrier labeling, models (logistic regression baseline, Random Forest, XGBoost), purged + embargoed walk-forward CV, Optuna tuning, probability calibration, ML-impact analysis (ablations, SHAP, random-filter control), ML charts in the UI.
- **O3 — software engineering:** architecture and contracts (`core/`), repo/CI, data loader (with O1), backtester (vectorized, then event-driven), PostgreSQL DB, FastAPI, Streamlit app skeleton, automated tests incl. leakage tests, Docker Compose.
- **The user (Emilia) is O3.** Default to the O3 perspective. When touching O1/O2 modules, stick to the contracts and do not redesign their internals.
- Dependencies: O2 needs O1's **trade log** to build training data (critical path). O1 and O2 need O3's **data loader** and a **vectorized backtester** early so they are never blocked. O3 needs O1/O2 code that matches the contracts.

## Source of truth

- `README.md` — overview, goal, stack, structure, roles, milestones, workflow rules (Polish).
- `docs/contract.md` — inter-module contracts v0.1 (Polish). **Read it before changing any interface, schema or config.** It calls itself `docs/CONTRACTS.md` and the README links `docs/CONTRACTS.md` and `docs/ARCHITECTURE.md`; on disk the file is `docs/contract.md` and `ARCHITECTURE.md` does not exist yet. Ask before renaming files or fixing links.
- LaTeX report `sprawozdanie.tex` (Polish; not in the repo yet) — living project report: goals, research questions PB1–PB4, functional requirements FR-01..FR-15, non-functional NFR-01..NFR-07, architecture, stack, RACI role matrix, schedule, experiments E1–E7, risks, **status table D1–D12**, **weekly progress log**, results, bibliography. `\todo{}` marks open decisions. Keep it consistent with the contract (e.g. pair id `KO__PEP`, trade log column `ret`).

## Planned repo layout

```
docs/            plan, architecture, contracts, report
src/ptp/
  core/          interfaces (ABCs), Pydantic config/types, plugin registry      [O3]
  data/          download, cleaning, Parquet storage, get_prices()              [O3+O1]
  pairs/         candidate generation, cointegration tests, ranking             [O1]
  strategies/    Strategy plugins (zscore.py, ...)                              [O1]
  risk/          PositionSizer, RiskParams                                      [O1]
  metrics/       compute_metrics()                                              [O1]
  ml/            features, labels, SignalFilter plugins, validation             [O2]
  backtest/      vectorized.py, engine.py (event), costs.py, portfolio.py       [O3]
  db/            SQLAlchemy models, repository, Alembic migrations              [O3]
  api/           FastAPI app, routers, background jobs                          [O3]
frontend/        Streamlit app (app.py, pages/)          [O3 skeleton, O1/O2 charts]
experiments/     YAML configs (one per experiment) + scripts                    [O1+O2]
notebooks/       exploration only, never imported by src/
tests/           unit/, integration/, leakage/
data/            local Parquet files — gitignored, never committed
docker/          api.Dockerfile, frontend.Dockerfile
docker-compose.yml   postgres + api + frontend
pyproject.toml, .github/workflows/ci.yml, .env.example
```

## Stack

Python 3.11 · pandas, numpy, yfinance, Parquet (pyarrow) · statsmodels, scipy · scikit-learn, XGBoost, Optuna, SHAP · experiment tracking via YAML configs + saved results (MLflow optional) · Pydantic · FastAPI + Uvicorn · PostgreSQL, SQLAlchemy, Alembic · Streamlit, Plotly · pytest, ruff, GitHub Actions · Docker / Docker Compose · GitHub Issues + Projects.

Deliberately **not** used (keep it simple, it is an academic project): Celery/Redis (use FastAPI `BackgroundTasks`), React/custom JS frontend, paid data, intraday data, broker integration/live trading, multi-user auth.

## Architecture

```
Data (Parquet) → Pair selection → Strategy (signals) → ML filter → Backtester → DB
                                                                                  ↓
                                    Frontend (Streamlit) ← API (FastAPI) ←────────┘
```

- Strategies, pair selectors, ML filters, feature builders, sizers and backtesters are **plugins** behind ABCs (`PairSelector`, `Strategy`, `SignalFilter`, `FeatureBuilder`, `PositionSizer`, `Backtester`), registered with `@register(category, name)` in `ptp.core.registry` and looked up by name from `BacktestConfig`. Each plugin has a Pydantic `params_model`; the API exposes it as JSON Schema (`GET /plugins/{category}`) and the frontend auto-generates forms from it. Adding a strategy/model = new class + decorator + import, nothing else changes.
- One `BacktestConfig` (Pydantic) describes a whole simulation and is shared by API, frontend, `experiments/*.yaml` and the DB (JSONB column).
- **Prices live in Parquet, not in the DB.** The DB stores what the system produces: `backtest_runs` (config JSONB, status, progress, data_version, duration, error), `trades`, `equity_curve`, `metrics`, `pairs`, optional `signals`.
- O1/O2 modules never touch the DB; they take and return DataFrames. Persistence goes through `ptp.db.repository` (O3).
- **Streamlit talks only to the API**, never directly to the DB or to O1/O2 code. This keeps the "modular platform" real and lets the frontend be swapped.
- Long runs: `POST /backtests` returns `run_id` (HTTP 202) and runs in the background; status `pending → running → done | failed`; the frontend polls `GET /backtests/{id}`. Other endpoints: `/health`, `/universes`, `/plugins/{category}`, `/pairs/discover`, `/backtests` (list), `/backtests/{id}/trades|equity|signals`, `/backtests/compare?ids=`, `DELETE /backtests/{id}`. Errors are `{"detail": ...}`, 422 for bad config, 404 for unknown run.

## Backtester (O3's main deliverable)

- **Vectorized engine first** (phase F1) so O1/O2 are not blocked; **event-driven engine** later (F2–F3) = the mentor's "own exchange simulator". Both implement `Backtester.run(config, progress) -> BacktestResult` and **must produce the same trade log** for the same config (integration test, numeric tolerance).
- `run` flow: split `[start, end]` into walk-forward windows → per window: `PairSelector.select` on the formation window only → `Strategy.generate_signals` → optionally train the ML filter on events with `t1 < trading_start` and query it → execute → update portfolio. Positions are closed at the end of each trading window (`exit_reason = "end_of_window"`). Then `compute_metrics(equity, trades)`.
- Execution: a signal decided at the close of `t` executes at the **open of `t+1`**. Costs: commission and slippage per side (fractions of notional/price), annual borrow rate on the short leg. Defaults in the contract: commission 0.0005, slippage 0.0005, borrow 0.01/yr (final values still `\todo` in the report).
- ML filter: applied **only to entry events**; if `predict_proba < config.ml.threshold`, the entry is skipped and the pair stays flat until the next entry event. `ml_proba` is written to the trade log. The random filter (experiment E3) implements the same interface with a fixed seed.

## Domain knowledge (enough to work with O1/O2 code)

- Pairs: candidates via PCA on returns + clustering (K-Means/DBSCAN) + sector grouping. Tests: OLS hedge ratio, Engle–Granger cointegration, ADF on the spread, half-life (`HL = -ln2/λ` from `Δs_t = λ s_{t-1} + ε`), Hurst exponent. **Benjamini–Hochberg** multiple-testing control. Spread `s_t = log A − β log B − α`; `Z_t` = rolling z-score.
- Walk-forward everywhere: formation window ~12 months, trading window ~6 months, step 6 months (defaults, may change).
- Baseline Z-score rules (defaults): lookback 60, entry |z| ≥ 2.0, exit |z| ≤ 0.5, stop |z| ≥ 3.5, max holding ≈ 3 × half-life.
- Risk defaults: dollar-neutral legs, capital_per_pair 0.1 of equity, max 10 open pairs, max gross leverage 2.0.
- ML = **meta-labeling**: the strategy decides direction, the model decides whether to take the trade (`P(success)`). Labels via triple barrier (profit / stop / time). Validation: purged + embargoed walk-forward CV. Sparse signals → pool trades across many pairs and years.
- Clustering uses only price-derived features + static sector labels. **Never yfinance fundamentals** (not point-in-time → leakage).
- Experiments: E1 baseline · E2 ML filter at several thresholds · E3 random-filter control · E4 ablations (feature groups, models) · E5 regimes (2020 crash, 2022 bear market) · E6 cost sensitivity · E7 bootstrap CIs on Sharpe difference. Each experiment = one YAML in `experiments/` and one run in the DB.

## Data

- Source: yfinance, daily OHLCV, split/dividend adjusted. **Date range: 2010-01-01 to the day before the download** (decided 12.11.2026; defaults in `ptp.data.download`, end = yesterday so no unfinished session is stored). Current team version: `v20261112` = 2010-01-04 .. 2026-11-11, 503 tickers, 0 failed. Known Yahoo anomalies (cleaning rules still open with O1): 15 daily moves > 50%, most real events; DHR 2016-07-05 (+61%, Fortive spin-off) looks like a data glitch. Long-format Parquet: `data/prices/v{YYYYMMDD}/prices.parquet` (`date, ticker, open, high, low, close, adj_open, adj_close, volume`); `data/context/v{YYYYMMDD}/context.parquet` (`date, vix, spx`); `data/meta/sectors.csv` (`ticker, sector, industry`). The data version is stored with every run (reproducibility).
- Loader API: `get_prices(tickers, start, end, field="adj_close", data_version=None)` → wide DataFrame (index `date`, columns tickers, inclusive range, NaN kept); `get_context`, `get_universe(name)` (`sp500`, `sector:<name>`, `custom:<file>`), `get_sectors`, `trading_days`.
- Known limitation: **survivorship bias** (yfinance gives today's index members). Document it, don't hide it.

## Key conventions (from the contract, abbreviated)

- Code, identifiers, docstrings, comments, commit messages: **English**. Project docs, README and report: **Polish**.
- Dates: `pd.Timestamp` without timezone, NYSE trading days only. Time index ascending, unique, named `date`.
- Tickers: uppercase, Yahoo format (`BRK-B`). Pair id `"{ticker_a}__{ticker_b}"` (double underscore). `ticker_a` is the dependent variable in `A = α + β·B`. Event id `"{pair_id}|{YYYY-MM-DD}"` links trades to ML features.
- Fractions are decimals, never percent (`0.001` = 0.1%). Returns simple unless named `log_`. Money is `float` USD.
- Direction: `+1` long spread (buy A, sell β·B), `-1` short spread, `0` flat.
- Signals are **target positions** at the close of day `t` (columns `date, pair_id, target, zscore, spread, hedge_ratio, event`); the backtester derives orders from changes in `target`.
- Trade log (contract §9) and `equity`/metric schemas (§10) are fixed. Metric keys may be added, never renamed.
- The loader does not fill NaNs or filter tickers; the caller decides.
- Anything random takes `seed: int`. All public functions have type hints.

## No-lookahead rules (L1–L7, enforced by tests in `tests/leakage/`)

1. Pair selection sees only the formation window (O3 slices the data). 2. A signal on day `t` uses data ≤ `t`. 3. Execution after `t`: `entry_date > signal_date`. 4. ML features on day `t` use data ≤ `t`. 5. Models train only on labels known before `trading_start` (`t1 < trading_start`). 6. Scalers/normalizers fit on the training set only. 7. No `bfill()` or `shift(-n)` in production code (grep check in CI).
Standard leakage test: perturb the data after `t` and assert outputs up to `t` are unchanged. Never write code that violates these, even in prototypes.

## Changing contracts

Contracts freeze on **29.11.2026**. After that, a change needs a PR touching `docs/contract.md` and `src/ptp/core/` together, with approval from all 3 people. Adding an *optional* field is non-breaking (one approval). Removing a field or changing its meaning is breaking: bump the version and add a changelog entry. Before the freeze, propose contract changes to the user rather than making them silently.

## Workflow

- `main` always works. New work goes on `feat/...` branches and reaches `main` via PR. CI (pytest + ruff) must pass and at least one teammate reviews.
- Commit messages: `feat:`, `fix:`, `docs:`, `test:` prefixes.
- Every task is a GitHub Issue with an assignee, tracked on the GitHub Projects board.
- Notebooks are for exploration only. Code the platform uses lives in `src/`.
- **Definition of Done:** merged to `main` after review · tests where applicable and CI green · matches the contracts · short docs/docstrings · status updated in the report (D1–D12 table / weekly log).
- Run `pytest` and `ruff` before declaring work done.

## Timeline

| Milestone | Date | Done when |
|---|---|---|
| M0 foundations | 29.11.2026 | repo + CI, contracts frozen, S&P 500 data downloaded, Docker Compose with DB |
| M1 baseline (demo for mentor) | 31.01.2027 | one config runs end to end: pairs → signals → vectorized backtest → result on screen; first trade log for O2 |
| M2 ML integration | 28.03.2027 | model plugged into the backtest, first baseline vs ML comparison, event-driven engine, results in DB |
| M3 complete platform | 09.05.2027 | all 6 mentor functions available from the UI, experiments done, leakage tests pass |
| M4 final | 31.05.2027 | report, presentation, one-command startup |

Phases: F0 09.11–29.11 · F1 30.11–31.01 (winter break ~21.12–06.01) · exam session in February (dates TBD, light work only) · F2 22.02–28.03 · F3 29.03–09.05 · F4 10.05–31.05 (report, polish, buffer).

**Current phase: F0 / M0.** O3 next steps: repo skeleton (`pyproject.toml`, `src/ptp/` packages, ruff/pytest config, CI workflow, `.gitignore` incl. `data/`, `.env.example`), contracts as real code in `src/ptp/core/` (ABCs, Pydantic types/config, registry), data loader + download script, `docker-compose.yml` with PostgreSQL, then the vectorized backtester and first API endpoints.

## Open items (do not decide silently, ask)

- Contract file name (`docs/contract.md` vs `docs/CONTRACTS.md`) and missing `docs/ARCHITECTURE.md`.
- Report `\todo`s: final cost values, ML thresholds, regime periods, NFR-06 performance target, team member names, meeting schedule, communication channel, exam session dates.
- Whether the LaTeX report lives in the repo (e.g. `docs/report/`) or only on Overleaf.

## How to work with the user

- If a request can be read more than one way, confirm the interpretation before building.
- It is an academic project: prefer simple, well-documented, free tools over clever or heavy solutions, but never cut any item the mentor required.
