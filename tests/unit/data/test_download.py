"""Tests for the download transformations. No network: input frames are built by hand
in the same shape yfinance and Wikipedia return."""

import numpy as np
import pandas as pd
import pytest

from ptp.data.download import normalize_ticker, parse_sp500_table, yf_to_context, yf_to_long
from ptp.data.storage import PRICE_COLUMNS


def _yf_frame(tickers: list[str], days: pd.DatetimeIndex) -> pd.DataFrame:
    fields = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers], names=["Price", "Ticker"])
    data = np.ones((len(days), len(cols)))
    df = pd.DataFrame(data, index=days, columns=cols)
    for t in tickers:
        df[("Close", t)] = 100.0
        df[("Adj Close", t)] = 90.0  # adjustment factor 0.9
        df[("Open", t)] = 50.0
        df[("Volume", t)] = 1234.0
    return df


@pytest.mark.parametrize(
    ("raw", "expected"), [("brk.b", "BRK-B"), (" aapl ", "AAPL"), ("BF.B", "BF-B")]
)
def test_normalize_ticker(raw: str, expected: str) -> None:
    assert normalize_ticker(raw) == expected


def test_yf_to_long_shape_and_adj_open() -> None:
    days = pd.bdate_range("2021-01-04", periods=3)
    long = yf_to_long(_yf_frame(["KO", "BRK.B"], days))
    assert list(long.columns) == list(PRICE_COLUMNS)
    assert len(long) == 6
    assert set(long["ticker"]) == {"KO", "BRK-B"}
    assert long["adj_open"].tolist() == pytest.approx([45.0] * 6)  # 50 * 90/100
    assert str(long["volume"].dtype) == "Int64"


def test_yf_to_long_drops_non_trading_rows() -> None:
    days = pd.bdate_range("2021-01-04", periods=3)
    raw = _yf_frame(["KO", "NEW"], days)
    for field in ["Adj Close", "Close", "High", "Low", "Open", "Volume"]:
        raw.loc[days[0], (field, "NEW")] = np.nan  # NEW listed on day 2
    long = yf_to_long(raw)
    assert (long["ticker"] == "NEW").sum() == 2


def test_yf_to_long_empty() -> None:
    assert list(yf_to_long(pd.DataFrame()).columns) == list(PRICE_COLUMNS)


def test_yf_to_context() -> None:
    days = pd.bdate_range("2021-01-04", periods=3)
    ctx = yf_to_context(_yf_frame(["^VIX", "^GSPC"], days))
    assert list(ctx.columns) == ["date", "vix", "spx"]
    assert len(ctx) == 3


def test_parse_sp500_table() -> None:
    table = pd.DataFrame(
        {
            "Symbol": ["MSFT", "BRK.B"],
            "Security": ["Microsoft", "Berkshire"],
            "GICS Sector": ["Information Technology", "Financials"],
            "GICS Sub-Industry": ["Systems Software", "Multi-Sector Holdings"],
        }
    )
    out = parse_sp500_table(table)
    assert list(out.columns) == ["ticker", "sector", "industry"]
    assert out["ticker"].tolist() == ["BRK-B", "MSFT"]
