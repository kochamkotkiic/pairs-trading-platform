"""Shared fixtures: a tiny fake dataset written to a temporary data directory."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptp.data import storage

TICKERS = ["KO", "PEP", "XOM"]


def make_prices(days: pd.DatetimeIndex, tickers: list[str], seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i, ticker in enumerate(tickers):
        close = 50 + i * 10 + rng.normal(0, 1, len(days)).cumsum()
        rows.append(
            pd.DataFrame(
                {
                    "date": days,
                    "ticker": ticker,
                    "open": close - 0.1,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "adj_open": (close - 0.1) * 0.98,
                    "adj_close": close * 0.98,
                    "volume": 1_000_000 + i,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def days() -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-04", "2021-03-31", name="date")


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, days: pd.DatetimeIndex) -> Path:
    """Point PTP_DATA_DIR at a temp folder holding one data version."""
    monkeypatch.setenv("PTP_DATA_DIR", str(tmp_path))
    prices = make_prices(days, TICKERS)
    # XOM is missing one day, to check NaNs are kept (not filled).
    prices = prices[~((prices["ticker"] == "XOM") & (prices["date"] == days[5]))]
    storage.write_prices(prices, "v20210401")
    storage.write_context(
        pd.DataFrame({"date": days, "vix": 20.0, "spx": np.linspace(3700, 3900, len(days))}),
        "v20210401",
    )
    storage.write_sectors(
        pd.DataFrame(
            {
                "ticker": TICKERS,
                "sector": ["Consumer Staples", "Consumer Staples", "Energy"],
                "industry": ["Soft Drinks", "Soft Drinks", "Integrated Oil & Gas"],
            }
        )
    )
    return tmp_path
