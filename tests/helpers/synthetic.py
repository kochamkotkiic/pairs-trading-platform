"""Synthetic market data for tests.

AAA and BBB are cointegrated by construction:
    log(AAA) = ALPHA + BETA * log(BBB) + x_t,   x_t mean-reverting (AR(1), phi < 1)
CCC is an independent random walk. Because we know the true relationship, tests can
check that the platform behaves sensibly without downloading real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ptp.data import storage

ALPHA = 0.5
BETA = 0.8
VERSION = "v20190101"


def make_market(
    start: str = "2015-01-01", end: str = "2018-12-31", seed: int = 7
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(prices_long, context, sectors)`` in the storage formats."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end, name="date")
    n = len(days)

    log_b = np.log(50) + np.cumsum(rng.normal(0, 0.01, n))
    x = np.zeros(n)
    for t in range(1, n):  # AR(1) spread with half-life of a few days
        x[t] = 0.85 * x[t - 1] + rng.normal(0, 0.01)
    log_a = ALPHA + BETA * log_b + x
    log_c = np.log(80) + np.cumsum(rng.normal(0, 0.012, n))

    frames = []
    for ticker, log_p in [("AAA", log_a), ("BBB", log_b), ("CCC", log_c)]:
        close = np.exp(log_p)
        # Open = previous close with a small overnight move.
        open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.002, n))
        frames.append(
            pd.DataFrame(
                {
                    "date": days,
                    "ticker": ticker,
                    "open": open_,
                    "high": np.maximum(open_, close) * 1.005,
                    "low": np.minimum(open_, close) * 0.995,
                    "close": close,
                    "adj_open": open_,  # no dividends/splits in the fake market
                    "adj_close": close,
                    "volume": 1_000_000,
                }
            )
        )
    prices = pd.concat(frames, ignore_index=True)
    context = pd.DataFrame({"date": days, "vix": 20.0, "spx": np.linspace(2000, 2500, n)})
    sectors = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "sector": ["Test", "Test", "Other"],
            "industry": ["Test", "Test", "Other"],
        }
    )
    return prices, context, sectors


def write_market(
    data_dir: Path,
    prices: pd.DataFrame,
    context: pd.DataFrame,
    sectors: pd.DataFrame,
    version: str = VERSION,
) -> None:
    """Write a market into ``data_dir`` (caller must point PTP_DATA_DIR at it)."""
    assert storage.data_dir() == data_dir, "set PTP_DATA_DIR before calling write_market"
    storage.write_prices(prices, version)
    storage.write_context(context, version)
    storage.write_sectors(sectors)
