"""Vectorized backtester (contract section 8). Owner: O3.

Flow of ``run`` (same for both engines):
1. Split ``[start, end]`` into walk-forward windows.
2. Per window: ``PairSelector.select`` on the formation window ONLY (rule L1) ->
   ``Strategy.generate_signals`` -> signals checked against the contract.
3. Target changes become trades, executed at the open of the NEXT trading day
   (rule L3). Positions still open at the end of a trading window are closed at the
   close of its last day (``exit_reason = "end_of_window"``).
4. Trades are accepted in time order: risk limits, sizing, costs.
5. Daily equity curve (marked at the close), then metrics.

Simplifications (documented for the report):
- Sizing and the leverage limit use *realized* equity (initial capital + closed P&L).
- Borrow cost is charged when a trade closes, not accrued daily in the equity curve.
- A price missing on an execution day: entry is skipped; exit uses the last known close.
- ML filter is not supported yet (phase F2): a config with ``ml.filter`` raises.
"""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ptp.backtest.costs import borrow_cost, commission, fill_price
from ptp.backtest.provisional_metrics import provisional_metrics
from ptp.backtest.windows import Window, make_windows
from ptp.core.config import BacktestConfig
from ptp.core.interfaces import Backtester, BacktestResult, PositionSizer
from ptp.core.registry import get, register
from ptp.core.schemas import (
    EQUITY_COLUMNS,
    SIGNAL_COLUMNS,
    TRADE_COLUMNS,
    SchemaError,
    check_equity,
    check_signals,
    check_trades,
)
from ptp.core.types import Pair, PortfolioState, make_event_id
from ptp.data.loader import get_prices, get_universe
from ptp.data.storage import resolve_version

# Packages whose modules register plugins when imported.
PLUGIN_PACKAGES = ("ptp.pairs", "ptp.strategies", "ptp.risk", "ptp.ml")

# Bookkeeping columns used between execution and the equity curve, dropped afterwards.
INTERNAL_TRADE_COLUMNS = ["_entry_idx", "_exit_idx", "_exit_at_close", "_entry_commission"]

EXIT_REASON_BY_EVENT = {
    "exit_mean_revert": "mean_revert",
    "exit_stop_loss": "stop_loss",
    "exit_time_limit": "time_limit",
}


def load_plugins() -> None:
    """Import plugin packages so their ``@register`` decorators run."""
    for package in PLUGIN_PACKAGES:
        importlib.import_module(package)


@dataclass
class _Candidate:
    """A trade the strategy asked for, before sizing and risk checks."""

    window_id: int
    pair: Pair
    direction: int
    signal_date: pd.Timestamp
    entry_idx: int  # position in the trading calendar
    exit_signal_date: pd.Timestamp
    exit_idx: int
    exit_at_close: bool  # True only for end-of-window exits
    exit_reason: str
    hedge_ratio: float
    entry_z: float
    exit_z: float


@register("backtester", "vectorized")
class VectorizedBacktester(Backtester):
    name = "vectorized"

    def run(
        self,
        config: BacktestConfig,
        progress: Callable[[float], None] | None = None,
    ) -> BacktestResult:
        started = time.perf_counter()
        if config.ml.filter is not None:
            raise NotImplementedError("ML filter integration is planned for phase F2")

        load_plugins()
        selector = get("pair_selector", config.pair_selector)()
        selector_params = selector.params_model(**config.pair_selector_params)
        strategy = get("strategy", config.strategy)()
        strategy_params = strategy.params_model(**config.strategy_params)
        sizer: PositionSizer = get("sizer", config.sizer)()

        windows = make_windows(config.walk_forward)
        if not windows:
            raise ValueError("walk_forward range is too short for one formation + trading window")

        version = resolve_version("prices", config.data_version)
        tickers = get_universe(config.universe)
        first, last = windows[0].formation_start, windows[-1].trading_end
        close = get_prices(tickers, first, last, "adj_close", version)
        open_ = get_prices(tickers, first, last, "adj_open", version)
        calendar = close.index

        candidates: list[_Candidate] = []
        signals: list[pd.DataFrame] = []
        pairs_all: list[Pair] = []
        sim_mask = np.zeros(len(calendar), dtype=bool)  # days inside any trading window
        for n, window in enumerate(windows, start=1):
            in_window = (calendar >= window.trading_start) & (calendar <= window.trading_end)
            if in_window.any():
                sim_mask |= in_window
                result = self._run_window(
                    window, close, calendar, selector, selector_params, strategy, strategy_params
                )
                if result is not None:
                    window_pairs, window_signals, window_candidates = result
                    pairs_all += window_pairs
                    signals.append(window_signals)
                    candidates += window_candidates
            if progress is not None:
                progress(0.9 * n / len(windows))

        trades = self._execute(candidates, open_, close, calendar, config, sizer)
        equity = self._equity_curve(trades, close, calendar, sim_mask, config.initial_capital)
        trades = trades.drop(columns=INTERNAL_TRADE_COLUMNS, errors="ignore")
        check_trades(trades)
        check_equity(equity)

        try:  # O1's official metrics, once they exist
            from ptp.metrics import compute_metrics  # type: ignore[attr-defined]

            metrics = compute_metrics(equity, trades)
        except ImportError:
            metrics = provisional_metrics(equity, trades, config.initial_capital)

        all_signals = (
            pd.concat(signals, ignore_index=True)
            if signals
            else pd.DataFrame(columns=[*SIGNAL_COLUMNS, "window_id"])
        )
        if progress is not None:
            progress(1.0)
        return BacktestResult(
            config=config,
            data_version=version,
            trades=trades,
            equity=equity,
            signals=all_signals,
            pairs=pairs_all,
            metrics=metrics,
            duration_s=time.perf_counter() - started,
        )

    # ------------------------------------------------------------------
    # One window: selection + signals -> candidates
    # ------------------------------------------------------------------

    def _run_window(
        self, window, close, calendar, selector, selector_params, strategy, strategy_params
    ):
        # Rule L1: the selector sees ONLY the formation window. Tickers without a full
        # formation history (e.g. listed later) are left out here, by the caller.
        formation = close.loc[window.formation_start : window.formation_end]
        formation = formation.dropna(axis=1, how="any")
        if formation.empty:
            return None
        pairs = selector.select(
            formation, window.formation_start, window.formation_end, selector_params
        )
        if not pairs:
            return None

        needed = sorted({t for p in pairs for t in (p.ticker_a, p.ticker_b)})
        prices = close.loc[window.formation_start : window.trading_end, needed]
        sig = strategy.generate_signals(
            prices, pairs, window.trading_start, window.trading_end, strategy_params
        )
        check_signals(sig)
        _check_signal_scope(sig, window, pairs)
        sig = sig.sort_values(["pair_id", "date"]).reset_index(drop=True)
        sig["window_id"] = window.window_id
        return pairs, sig, _make_candidates(sig, pairs, window, calendar)

    # ------------------------------------------------------------------
    # Candidates -> trades (sizing, risk limits, costs)
    # ------------------------------------------------------------------

    def _execute(self, candidates, open_, close, calendar, config, sizer) -> pd.DataFrame:
        risk, costs = config.risk, config.costs
        opens, closes = open_.to_numpy(), close.ffill().to_numpy()
        col = {t: i for i, t in enumerate(close.columns)}

        realized = 0.0
        open_trades: list[dict] = []
        trades: list[dict] = []
        for c in sorted(candidates, key=lambda c: (c.entry_idx, c.pair.pair_id)):
            i = c.entry_idx
            # Exits at the open of the same day happen before new entries.
            still_open = []
            for t in open_trades:
                if t["_exit_idx"] < i or (t["_exit_idx"] == i and not t["_exit_at_close"]):
                    realized += t["pnl"]
                else:
                    still_open.append(t)
            open_trades = still_open

            ia, ib = col[c.pair.ticker_a], col[c.pair.ticker_b]
            pa, pb = opens[i, ia], opens[i, ib]
            if not (np.isfinite(pa) and np.isfinite(pb)):
                continue
            if len(open_trades) >= risk.max_open_pairs:
                continue

            equity_now = config.initial_capital + realized
            gross_now = sum(t["_entry_notional"] for t in open_trades)
            net_value = sum(t["_entry_net_value"] for t in open_trades)
            state = PortfolioState(
                date=calendar[i].date(),
                cash=equity_now - net_value,
                equity=equity_now,
                open_positions={t["pair_id"]: (t["qty_a"], t["qty_b"]) for t in open_trades},
                gross_exposure=gross_now,
            )
            qty_a, qty_b = sizer.size(c.pair, c.direction, float(pa), float(pb), state, risk)
            if qty_a == 0 and qty_b == 0:
                continue
            entry_notional = abs(qty_a) * pa + abs(qty_b) * pb
            if gross_now + entry_notional > risk.max_gross_leverage * equity_now:
                continue

            prices_at_exit = closes if c.exit_at_close else opens
            xa, xb = prices_at_exit[c.exit_idx, ia], prices_at_exit[c.exit_idx, ib]
            if not np.isfinite(xa):
                xa = closes[c.exit_idx, ia]
            if not np.isfinite(xb):
                xb = closes[c.exit_idx, ib]

            trade = _price_trade(c, qty_a, qty_b, pa, pb, xa, xb, calendar, costs)
            trade["_entry_notional"] = entry_notional
            trade["_entry_net_value"] = qty_a * pa + qty_b * pb
            trades.append(trade)
            open_trades.append(trade)

        df = pd.DataFrame(trades)
        if df.empty:
            return pd.DataFrame(columns=[*TRADE_COLUMNS, *INTERNAL_TRADE_COLUMNS])
        df["trade_id"] = np.arange(1, len(df) + 1)
        return df.loc[:, [*TRADE_COLUMNS, *INTERNAL_TRADE_COLUMNS]]

    # ------------------------------------------------------------------
    # Equity curve
    # ------------------------------------------------------------------

    def _equity_curve(self, trades, close, calendar, sim_mask, initial_capital) -> pd.DataFrame:
        n = len(calendar)
        closes = close.ffill().to_numpy()
        col = {t: i for i, t in enumerate(close.columns)}
        realized = np.zeros(n)
        unrealized = np.zeros(n)
        net_value = np.zeros(n)
        gross = np.zeros(n)
        n_open = np.zeros(n, dtype=int)

        for t in trades.to_dict("records"):
            a, b = closes[:, col[t["ticker_a"]]], closes[:, col[t["ticker_b"]]]
            qa, qb = t["qty_a"], t["qty_b"]
            # Held from the entry day (bought at the open, marked at the close) until the
            # day before the exit; on the exit day the realized P&L takes over.
            held = slice(t["_entry_idx"], t["_exit_idx"])
            unrealized[held] += (
                qa * (a[held] - t["entry_price_a"])
                + qb * (b[held] - t["entry_price_b"])
                - t["_entry_commission"]
            )
            net_value[held] += qa * a[held] + qb * b[held]
            gross[held] += np.abs(qa * a[held]) + np.abs(qb * b[held])
            n_open[held] += 1
            realized[t["_exit_idx"]] += t["pnl"]

        equity = initial_capital + np.cumsum(realized) + unrealized
        mask = sim_mask
        out = pd.DataFrame(
            {
                "date": calendar[mask],
                "equity": equity[mask],
                "cash": (equity - net_value)[mask],
                "gross_exposure": gross[mask],
                "n_open_pairs": n_open[mask],
            }
        )
        prev = np.r_[initial_capital, out["equity"].to_numpy()[:-1]]
        out["daily_ret"] = out["equity"].to_numpy() / prev - 1
        return out.loc[:, list(EQUITY_COLUMNS)].reset_index(drop=True)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _check_signal_scope(sig: pd.DataFrame, window: Window, pairs: list[Pair]) -> None:
    dates = pd.to_datetime(sig["date"])
    if len(sig) and (dates.min() < window.trading_start or dates.max() > window.trading_end):
        raise SchemaError(
            f"signals for window {window.window_id} contain dates outside "
            f"[{window.trading_start.date()}, {window.trading_end.date()}]"
        )
    unknown = set(sig["pair_id"]) - {p.pair_id for p in pairs}
    if unknown:
        raise SchemaError(f"signals contain pairs that were not selected: {sorted(unknown)}")


def _make_candidates(
    sig: pd.DataFrame, pairs: list[Pair], window: Window, calendar: pd.DatetimeIndex
) -> list[_Candidate]:
    """Turn target positions into entry/exit dates (execution at the next open)."""
    by_id = {p.pair_id: p for p in pairs}
    in_window = np.flatnonzero(
        (calendar >= window.trading_start) & (calendar <= window.trading_end)
    )
    last_idx = int(in_window[-1])
    out: list[_Candidate] = []

    for pair_id, g in sig.groupby("pair_id", sort=True):
        pair = by_id[pair_id]
        idx = calendar.get_indexer(pd.to_datetime(g["date"]))
        if (idx < 0).any():
            raise SchemaError(f"signals for {pair_id} contain non-trading days")
        prev_target, current = 0, None
        for i, row in zip(idx, g.itertuples(index=False), strict=True):
            target = int(row.target)
            if current is not None and target != current["direction"]:
                if target == -current["direction"]:
                    reason = "reversal"
                elif row.event in EXIT_REASON_BY_EVENT:
                    reason = EXIT_REASON_BY_EVENT[row.event]
                else:
                    raise SchemaError(
                        f"{pair_id} {row.date}: target went to 0 with event {row.event!r}, "
                        "expected exit_mean_revert / exit_stop_loss / exit_time_limit"
                    )
                at_close = i + 1 > last_idx
                out.append(
                    _Candidate(
                        **current,
                        exit_signal_date=pd.Timestamp(row.date),
                        exit_idx=last_idx if at_close else int(i) + 1,
                        exit_at_close=at_close,
                        exit_reason=reason,
                        exit_z=float(row.zscore),
                    )
                )
                current = None
            if current is None and target != 0 and target != prev_target and i + 1 <= last_idx:
                current = dict(
                    window_id=window.window_id,
                    pair=pair,
                    direction=target,
                    signal_date=pd.Timestamp(row.date),
                    entry_idx=int(i) + 1,
                    hedge_ratio=float(row.hedge_ratio),
                    entry_z=float(row.zscore),
                )
            prev_target = target

        if current is not None:  # still open: close at the end of the trading window
            last_row = g.iloc[-1]
            out.append(
                _Candidate(
                    **current,
                    exit_signal_date=calendar[last_idx],
                    exit_idx=last_idx,
                    exit_at_close=True,
                    exit_reason="end_of_window",
                    exit_z=float(last_row["zscore"]),
                )
            )
    return out


def _price_trade(c: _Candidate, qty_a, qty_b, pa, pb, xa, xb, calendar, costs) -> dict:
    """All trade-log fields for one trade (contract section 9)."""
    entry_date, exit_date = calendar[c.entry_idx], calendar[c.exit_idx]
    s = costs.slippage
    entry_a, entry_b = fill_price(pa, qty_a, s), fill_price(pb, qty_b, s)
    exit_a, exit_b = fill_price(xa, -qty_a, s), fill_price(xb, -qty_b, s)

    entry_notional = abs(qty_a) * pa + abs(qty_b) * pb
    exit_notional = abs(qty_a) * xa + abs(qty_b) * xb
    short_notional = (abs(qty_a) * pa if qty_a < 0 else 0.0) + (
        abs(qty_b) * pb if qty_b < 0 else 0.0
    )
    entry_commission = commission(entry_notional, costs.commission)
    total_commission = entry_commission + commission(exit_notional, costs.commission)
    total_slippage = s * (entry_notional + exit_notional)
    borrow = borrow_cost(short_notional, costs.borrow_rate_annual, (exit_date - entry_date).days)

    gross_pnl = qty_a * (xa - pa) + qty_b * (xb - pb)
    total_costs = total_commission + total_slippage + borrow
    pnl = gross_pnl - total_costs
    return {
        "event_id": make_event_id(c.pair.pair_id, c.signal_date),
        "pair_id": c.pair.pair_id,
        "ticker_a": c.pair.ticker_a,
        "ticker_b": c.pair.ticker_b,
        "direction": c.direction,
        "signal_date": c.signal_date,
        "entry_date": entry_date,
        "exit_signal_date": c.exit_signal_date,
        "exit_date": exit_date,
        "holding_days": c.exit_idx - c.entry_idx,
        "hedge_ratio": c.hedge_ratio,
        "qty_a": float(qty_a),
        "qty_b": float(qty_b),
        "entry_price_a": float(entry_a),
        "entry_price_b": float(entry_b),
        "exit_price_a": float(exit_a),
        "exit_price_b": float(exit_b),
        "entry_z": c.entry_z,
        "exit_z": c.exit_z,
        "gross_pnl": float(gross_pnl),
        "costs": float(total_costs),
        "pnl": float(pnl),
        "ret": float(pnl / entry_notional) if entry_notional else math.nan,
        "exit_reason": c.exit_reason,
        "window_id": c.window_id,
        "ml_proba": math.nan,
        "_entry_idx": c.entry_idx,
        "_exit_idx": c.exit_idx,
        "_exit_at_close": c.exit_at_close,
        "_entry_commission": entry_commission,
    }
