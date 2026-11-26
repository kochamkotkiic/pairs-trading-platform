"""TEMPORARY metrics, used only until O1 delivers ``ptp.metrics.compute_metrics``.

The official implementation belongs to O1 (contract section 10.2). This version exists
so the backtester can return a complete ``BacktestResult`` (and the M1 demo can show
numbers) before then. Keys match ``ptp.core.schemas.METRIC_KEYS`` exactly.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def provisional_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int = 252,
) -> dict[str, float]:
    nan = float("nan")
    values = equity["equity"].to_numpy(dtype=float)
    rets = equity["daily_ret"].to_numpy(dtype=float)
    n = len(values)
    years = n / periods_per_year if n else 0.0

    total_return = values[-1] / initial_capital - 1 if n else 0.0
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 and total_return > -1 else nan
    std = float(np.std(rets, ddof=1)) if n > 1 else 0.0
    mean = float(np.mean(rets)) if n else 0.0
    volatility = std * math.sqrt(periods_per_year)
    sharpe = mean / std * math.sqrt(periods_per_year) if std > 0 else nan
    downside = math.sqrt(float(np.mean(np.minimum(rets, 0.0) ** 2))) if n else 0.0
    sortino = mean / downside * math.sqrt(periods_per_year) if downside > 0 else nan
    if n:
        peak = np.maximum.accumulate(np.r_[initial_capital, values])[1:]
        max_drawdown = float(np.max(1 - values / peak))
    else:
        max_drawdown = 0.0
    calmar = cagr / max_drawdown if max_drawdown > 0 and not math.isnan(cagr) else nan

    n_trades = len(trades)
    if n_trades:
        traded = (
            trades["qty_a"].abs() * (trades["entry_price_a"] + trades["exit_price_a"])
            + trades["qty_b"].abs() * (trades["entry_price_b"] + trades["exit_price_b"])
        ).sum()
        avg_equity = float(np.mean(values)) if n else initial_capital
        turnover = traded / avg_equity / years if years > 0 else nan
    else:
        turnover = 0.0

    return {
        "total_return": float(total_return),
        "cagr": float(cagr),
        "volatility": float(volatility),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "n_trades": float(n_trades),
        "win_rate": float((trades["pnl"] > 0).mean()) if n_trades else nan,
        "avg_trade_ret": float(trades["ret"].mean()) if n_trades else nan,
        "avg_holding_days": float(trades["holding_days"].mean()) if n_trades else nan,
        "turnover": float(turnover),
        "total_costs": float(trades["costs"].sum()) if n_trades else 0.0,
    }
