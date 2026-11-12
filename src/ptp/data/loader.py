"""Read access to local market data (contract section 2.2).

Rules from the contract:
- The loader does NOT fill missing values and does NOT drop tickers. The caller decides.
- The loader returns exactly the inclusive range ``[start, end]``. Making sure ``end``
  is not in the future relative to a decision is the caller's job (rule L1/L2).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, get_args

import pandas as pd

from ptp.data.storage import (
    DataNotFoundError,
    data_dir,
    dataset_path,
    resolve_version,
    sectors_path,
)

logger = logging.getLogger(__name__)

PriceField = Literal["adj_close", "adj_open", "close", "open", "volume"]


def _date_filters(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple]:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end < start:
        raise ValueError(f"end ({end.date()}) is before start ({start.date()})")
    return [("date", ">=", start), ("date", "<=", end)]


def trading_days(
    start: pd.Timestamp, end: pd.Timestamp, data_version: str | None = None
) -> pd.DatetimeIndex:
    """NYSE trading days in ``[start, end]``.

    Taken from the S&P 500 index series in the context data (the index trades on every
    NYSE session), so no extra calendar library is needed. Only covers downloaded dates.
    """
    version = resolve_version("context", data_version)
    df = pd.read_parquet(
        dataset_path("context", version), columns=["date"], filters=_date_filters(start, end)
    )
    return pd.DatetimeIndex(sorted(df["date"].unique()), name="date")


def get_prices(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    field: PriceField = "adj_close",
    data_version: str | None = None,  # None = newest
) -> pd.DataFrame:
    """Wide format: index = date, columns = tickers. Inclusive [start, end].

    Missing values stay NaN (no forward fill here). Tickers without any data in the
    range are still returned, as all-NaN columns, with a warning.
    """
    if field not in get_args(PriceField):
        raise ValueError(f"Unknown field {field!r}. Valid: {get_args(PriceField)}")
    tickers = list(dict.fromkeys(tickers))  # remove duplicates, keep order
    version = resolve_version("prices", data_version)
    filters = _date_filters(start, end)
    long = pd.read_parquet(
        dataset_path("prices", version),
        columns=["date", "ticker", field],
        filters=[*filters, ("ticker", "in", tickers)] if tickers else filters,
    )
    wide = long.pivot(index="date", columns="ticker", values=field)

    # Same calendar for every ticker: all trading days in the range. Prefer the context
    # data of the same version; otherwise use every date present in the prices file.
    if dataset_path("context", version).exists():
        days = trading_days(start, end, data_version=version)
    else:
        all_dates = pd.read_parquet(
            dataset_path("prices", version), columns=["date"], filters=filters
        )["date"]
        days = pd.DatetimeIndex(sorted(all_dates.unique()), name="date")
    wide = wide.reindex(index=pd.DatetimeIndex(days, name="date"), columns=tickers)
    wide.columns.name = None

    empty = [t for t in tickers if wide[t].isna().all()]
    if empty:
        logger.warning("No %s data in range for %d ticker(s): %s", field, len(empty), empty[:10])
    return wide


def get_context(
    start: pd.Timestamp, end: pd.Timestamp, data_version: str | None = None
) -> pd.DataFrame:
    """index = date, columns = ['vix', 'spx']."""
    version = resolve_version("context", data_version)
    df = pd.read_parquet(dataset_path("context", version), filters=_date_filters(start, end))
    return df.set_index("date").sort_index().loc[:, ["vix", "spx"]]


def get_sectors() -> pd.Series:
    """index = ticker, value = sector."""
    path = sectors_path()
    if not path.exists():
        raise DataNotFoundError(f"{path} not found. Run: python -m ptp.data.download")
    df = pd.read_csv(path)
    return df.set_index("ticker")["sector"].rename("sector")


def get_universe(name: str) -> list[str]:
    """Named universes: 'sp500', 'sector:Energy', 'custom:<file>'.

    ``custom:<file>`` reads a CSV with a ``ticker`` column (or one ticker per line),
    from ``data/universes/<file>`` or from a path.
    """
    if name == "sp500":
        return sorted(get_sectors().index)
    if name.startswith("sector:"):
        sector = name.removeprefix("sector:")
        sectors = get_sectors()
        tickers = sorted(sectors[sectors.str.lower() == sector.lower()].index)
        if not tickers:
            raise DataNotFoundError(
                f"No tickers in sector {sector!r}. Known: {sorted(sectors.unique())}"
            )
        return tickers
    if name.startswith("custom:"):
        return _read_custom_universe(name.removeprefix("custom:"))
    raise ValueError(f"Unknown universe {name!r}. Use 'sp500', 'sector:<name>' or 'custom:<file>'")


def _read_custom_universe(file: str) -> list[str]:
    path = Path(file)
    if not path.exists():
        path = data_dir() / "universes" / file
    if not path.exists():
        raise DataNotFoundError(f"Custom universe file {file!r} not found")
    df = pd.read_csv(path)
    column = df["ticker"] if "ticker" in df.columns else pd.read_csv(path, header=None)[0]
    tickers = [str(t).strip().upper().replace(".", "-") for t in column.dropna()]
    return list(dict.fromkeys(t for t in tickers if t))
