"""Download S&P 500 daily data from Yahoo Finance and store it as a new data version.

Usage (from the repo root, with the virtual environment active):

    python -m ptp.data.download                        # full S&P 500, 2010-01-01 .. today
    python -m ptp.data.download --start 2015-01-01
    python -m ptp.data.download --tickers KO PEP XOM   # quick test with a few tickers

Known limitation (document it in the report): the ticker list is TODAY's S&P 500
members, so stocks that left the index are missing (survivorship bias).
"""

from __future__ import annotations

import argparse
import logging
import time
from io import StringIO
from typing import Any

import pandas as pd

from ptp.data.storage import (
    PRICE_COLUMNS,
    data_dir,
    make_version,
    write_context,
    write_manifest,
    write_prices,
    write_sectors,
)

logger = logging.getLogger(__name__)

SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CONTEXT_SYMBOLS = {"^VIX": "vix", "^GSPC": "spx"}
DEFAULT_START = "2010-01-01"  # agreed data range: 2010-01-01 .. download date
CHUNK_SIZE = 100

_YF_FIELDS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


# ---------------------------------------------------------------------------
# Pure transformations (no network, unit-tested)
# ---------------------------------------------------------------------------


def normalize_ticker(ticker: str) -> str:
    """Yahoo format: uppercase, dot -> dash (``brk.b`` -> ``BRK-B``)."""
    return ticker.strip().upper().replace(".", "-")


def parse_sp500_table(table: pd.DataFrame) -> pd.DataFrame:
    """Wikipedia constituents table -> ``ticker, sector, industry``."""
    out = pd.DataFrame(
        {
            "ticker": table["Symbol"].astype(str).map(normalize_ticker),
            "sector": table["GICS Sector"].astype(str),
            "industry": table["GICS Sub-Industry"].astype(str),
        }
    )
    return out.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)


def yf_to_long(raw: pd.DataFrame) -> pd.DataFrame:
    """yfinance ``download`` output (wide, 2-level columns) -> contract long format.

    Adds ``adj_open = open * adj_close / close`` (same adjustment factor as the close).
    Rows where the stock did not trade (all prices NaN) are dropped.
    """
    if raw.empty:
        return pd.DataFrame(columns=list(PRICE_COLUMNS))
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Expected yfinance output with (field, ticker) column levels")
    # Find which column level holds the field names ("Close", ...); the other is tickers.
    field_level = 0 if "Close" in raw.columns.get_level_values(0) else 1
    ticker_level = 1 - field_level

    frames = []
    for ticker in raw.columns.get_level_values(ticker_level).unique():
        sub = raw.xs(ticker, axis=1, level=ticker_level)
        sub = sub.rename(columns=_YF_FIELDS).loc[:, list(_YF_FIELDS.values())]
        sub = sub.dropna(subset=["open", "close", "adj_close"], how="all")
        if sub.empty:
            continue
        sub = sub.assign(ticker=normalize_ticker(str(ticker)))
        frames.append(sub)
    if not frames:
        return pd.DataFrame(columns=list(PRICE_COLUMNS))

    long = pd.concat(frames)
    long.index = pd.to_datetime(long.index).tz_localize(None).normalize()
    long = long.rename_axis("date").reset_index()
    long["adj_open"] = long["open"] * long["adj_close"] / long["close"]
    long["volume"] = long["volume"].round().astype("Int64")
    return long.loc[:, list(PRICE_COLUMNS)].sort_values(["date", "ticker"]).reset_index(drop=True)


def yf_to_context(raw: pd.DataFrame) -> pd.DataFrame:
    """yfinance output for ^VIX and ^GSPC -> ``date, vix, spx`` (index closes)."""
    field_level = 0 if "Close" in raw.columns.get_level_values(0) else 1
    close = raw.xs("Close", axis=1, level=field_level).rename(columns=CONTEXT_SYMBOLS)
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    # Trading days come from the S&P 500 index: keep only days it traded.
    close = close.dropna(subset=["spx"])
    return close.rename_axis("date").reset_index().loc[:, ["date", "vix", "spx"]]


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


def fetch_sp500_constituents() -> pd.DataFrame:
    """Current S&P 500 members with GICS sector and sub-industry, from Wikipedia."""
    import requests

    # Wikipedia rejects requests without a User-Agent.
    resp = requests.get(
        SP500_WIKI_URL, headers={"User-Agent": "ptp-data-downloader/0.1"}, timeout=30
    )
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text), attrs={"id": "constituents"})
    return parse_sp500_table(tables[0])


def _yf_download(symbols: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    # yfinance treats `end` as exclusive; add one day so the range is inclusive.
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return yf.download(
        symbols,
        start=start,
        end=end_excl,
        auto_adjust=False,
        actions=False,
        group_by="column",
        progress=False,
        threads=True,
        multi_level_index=True,
    )


def fetch_prices(tickers: list[str], start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    """Download prices in chunks. Returns ``(long_df, failed_tickers)``."""
    frames = []
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        logger.info("Downloading tickers %d-%d of %d", i + 1, i + len(chunk), len(tickers))
        frames.append(yf_to_long(_yf_download(chunk, start, end)))
        time.sleep(1)  # be polite to Yahoo
    long = pd.concat(frames, ignore_index=True) if frames else yf_to_long(pd.DataFrame())

    failed = sorted(set(tickers) - set(long["ticker"]))
    if failed:
        logger.info("Retrying %d failed ticker(s) once", len(failed))
        retry = yf_to_long(_yf_download(failed, start, end))
        long = pd.concat([long, retry], ignore_index=True)
        failed = sorted(set(tickers) - set(long["ticker"]))
    return long, failed


def run(start: str, end: str, tickers: list[str] | None = None, version: str | None = None) -> str:
    """Download everything and write a new data version. Returns the version string."""
    version = version or make_version()
    started = pd.Timestamp.now()

    if tickers:
        tickers = [normalize_ticker(t) for t in tickers]
        logger.info("Using %d ticker(s) from the command line", len(tickers))
    else:
        sectors = fetch_sp500_constituents()
        write_sectors(sectors)
        tickers = sectors["ticker"].tolist()
        logger.info("Fetched %d S&P 500 constituents", len(tickers))

    prices, failed = fetch_prices(tickers, start, end)
    write_prices(prices, version)
    context = yf_to_context(_yf_download(list(CONTEXT_SYMBOLS), start, end))
    write_context(context, version)

    info: dict[str, Any] = {
        "version": version,
        "source": "yfinance",
        "start": start,
        "end": end,
        "created_at": started.isoformat(timespec="seconds"),
        "duration_s": round((pd.Timestamp.now() - started).total_seconds(), 1),
        "n_tickers_requested": len(tickers),
        "n_tickers_ok": int(prices["ticker"].nunique()),
        "failed_tickers": failed,
        "n_rows": len(prices),
        "universe_note": "Current S&P 500 members (survivorship bias)",
    }
    write_manifest("prices", version, info)
    logger.info(
        "Saved %s: %d rows, %d tickers ok, %d failed -> %s",
        version,
        len(prices),
        info["n_tickers_ok"],
        len(failed),
        data_dir().resolve(),
    )
    return version


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--start", default=DEFAULT_START, help="first date (YYYY-MM-DD)")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--tickers", nargs="*", help="only these tickers (default: S&P 500)")
    parser.add_argument("--version", help="version name, default v<today>")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.start, args.end, args.tickers, args.version)


if __name__ == "__main__":
    main()
