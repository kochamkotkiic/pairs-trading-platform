"""DataFrame schemas (contract sections 4.3, 9 and 10).

Column lists for the tables that move between modules, plus a helper that
checks a DataFrame has them. Use it at module boundaries, e.g. the backtester
checks a strategy's output before using it.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

# Section 4.3: strategy output
SIGNAL_COLUMNS: tuple[str, ...] = (
    "date",
    "pair_id",
    "target",
    "zscore",
    "spread",
    "hedge_ratio",
    "event",
)
SIGNAL_EVENTS: frozenset[str] = frozenset(
    {"entry", "exit_mean_revert", "exit_stop_loss", "exit_time_limit", "hold", "flat"}
)
TARGET_VALUES: frozenset[int] = frozenset({-1, 0, 1})

# Section 9: trade log, the most important contract in the project
TRADE_COLUMNS: tuple[str, ...] = (
    "trade_id",
    "event_id",
    "pair_id",
    "ticker_a",
    "ticker_b",
    "direction",
    "signal_date",
    "entry_date",
    "exit_signal_date",
    "exit_date",
    "holding_days",
    "hedge_ratio",
    "qty_a",
    "qty_b",
    "entry_price_a",
    "entry_price_b",
    "exit_price_a",
    "exit_price_b",
    "entry_z",
    "exit_z",
    "gross_pnl",
    "costs",
    "pnl",
    "ret",
    "exit_reason",
    "window_id",
    "ml_proba",
)
EXIT_REASONS: frozenset[str] = frozenset(
    {"mean_revert", "stop_loss", "time_limit", "end_of_window"}
)

# Section 10.1: equity curve
EQUITY_COLUMNS: tuple[str, ...] = (
    "date",
    "equity",
    "cash",
    "gross_exposure",
    "n_open_pairs",
    "daily_ret",
)

# Section 10.2: metric keys. New keys may be added; existing ones must never change.
METRIC_KEYS: tuple[str, ...] = (
    "total_return",
    "cagr",
    "volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "calmar",
    "n_trades",
    "win_rate",
    "avg_trade_ret",
    "avg_holding_days",
    "turnover",
    "total_costs",
)


class SchemaError(ValueError):
    """A DataFrame does not match its contract schema."""


def check_columns(df: pd.DataFrame, required: Iterable[str], what: str = "DataFrame") -> None:
    """Raise ``SchemaError`` if any required column is missing. Extra columns are allowed."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"{what} is missing columns: {', '.join(missing)}")


def check_signals(df: pd.DataFrame) -> None:
    """Validate a strategy's ``signals`` output (contract 4.3)."""
    check_columns(df, SIGNAL_COLUMNS, "signals")
    bad_targets = set(df["target"].dropna().unique()) - TARGET_VALUES
    if bad_targets:
        raise SchemaError(f"signals.target has invalid values: {sorted(bad_targets)}")
    bad_events = set(df["event"].dropna().unique()) - SIGNAL_EVENTS
    if bad_events:
        raise SchemaError(f"signals.event has invalid values: {sorted(bad_events)}")
    if df.duplicated(subset=["date", "pair_id"]).any():
        raise SchemaError("signals has duplicate (date, pair_id) rows")


def check_trades(df: pd.DataFrame) -> None:
    """Validate a trade log (contract 9), including rule L3 (entry after signal)."""
    check_columns(df, TRADE_COLUMNS, "trades")
    if df["trade_id"].duplicated().any():
        raise SchemaError("trades.trade_id is not unique")
    bad_reasons = set(df["exit_reason"].dropna().unique()) - EXIT_REASONS
    if bad_reasons:
        raise SchemaError(f"trades.exit_reason has invalid values: {sorted(bad_reasons)}")
    lookahead = pd.to_datetime(df["entry_date"]) <= pd.to_datetime(df["signal_date"])
    if lookahead.any():
        raise SchemaError(
            f"{int(lookahead.sum())} trade(s) entered on or before their signal date (rule L3)"
        )


def check_equity(df: pd.DataFrame) -> None:
    """Validate an equity curve (contract 10.1)."""
    check_columns(df, EQUITY_COLUMNS, "equity")
    dates = pd.to_datetime(df["date"])
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise SchemaError("equity.date must be increasing and unique")


def check_metrics(metrics: dict[str, float]) -> None:
    """Validate that all fixed metric keys are present (contract 10.2)."""
    missing = [k for k in METRIC_KEYS if k not in metrics]
    if missing:
        raise SchemaError(f"metrics is missing keys: {', '.join(missing)}")
