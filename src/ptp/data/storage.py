"""Where data lives on disk and how it is versioned (contract section 2.1).

Layout under the data directory (default ``./data``, override with ``PTP_DATA_DIR``):

    prices/v20261112/prices.parquet     long format: date, ticker, open, ..., volume
    prices/v20261112/manifest.json      what was downloaded, when, and what failed
    context/v20261112/context.parquet   date, vix, spx
    meta/sectors.csv                    ticker, sector, industry
    universes/<name>.csv                optional custom ticker lists

A version is the folder name (``v`` + download date). ``None`` means "newest".
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd

Kind = Literal["prices", "context"]

PRICE_COLUMNS: tuple[str, ...] = (
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_open",
    "adj_close",
    "volume",
)
CONTEXT_COLUMNS: tuple[str, ...] = ("date", "vix", "spx")
VERSION_RE = re.compile(r"^v\d{8}$")


class DataNotFoundError(FileNotFoundError):
    """Requested data (version, file or universe) does not exist locally."""


def data_dir() -> Path:
    """Root data directory: ``$PTP_DATA_DIR`` if set, else ``./data``."""
    return Path(os.environ.get("PTP_DATA_DIR", "data"))


def make_version(date: pd.Timestamp | None = None) -> str:
    """Version string for a download made on ``date`` (default: today), e.g. ``v20261112``."""
    return f"v{pd.Timestamp(date or pd.Timestamp.today()):%Y%m%d}"


def list_versions(kind: Kind) -> list[str]:
    """All available versions of a dataset, oldest first."""
    root = data_dir() / kind
    if not root.is_dir():
        return []
    return sorted(
        p.name
        for p in root.iterdir()
        if VERSION_RE.match(p.name) and (p / f"{kind}.parquet").exists()
    )


def resolve_version(kind: Kind, version: str | None) -> str:
    """Turn ``None`` into the newest version and check the version exists."""
    versions = list_versions(kind)
    if not versions:
        raise DataNotFoundError(
            f"No {kind} data in {data_dir().resolve()}. Run: python -m ptp.data.download"
        )
    if version is None:
        return versions[-1]
    if version not in versions:
        raise DataNotFoundError(f"{kind} version {version!r} not found. Available: {versions}")
    return version


def dataset_path(kind: Kind, version: str) -> Path:
    return data_dir() / kind / version / f"{kind}.parquet"


def _write(df: pd.DataFrame, kind: Kind, columns: tuple[str, ...], version: str) -> Path:
    if not VERSION_RE.match(version):
        raise ValueError(f"Invalid version {version!r}, expected e.g. 'v20261112'")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{kind} data is missing columns: {missing}")
    out = df.loc[:, list(columns)].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    sort_keys = ["date", "ticker"] if "ticker" in out.columns else ["date"]
    if out.duplicated(subset=sort_keys).any():
        raise ValueError(f"{kind} data has duplicate {sort_keys} rows")
    out = out.sort_values(sort_keys).reset_index(drop=True)
    path = dataset_path(kind, version)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def write_prices(df: pd.DataFrame, version: str) -> Path:
    """Save long-format prices as a new version."""
    return _write(df, "prices", PRICE_COLUMNS, version)


def write_context(df: pd.DataFrame, version: str) -> Path:
    """Save market context (VIX, S&P 500 index level) as a new version."""
    return _write(df, "context", CONTEXT_COLUMNS, version)


def write_manifest(kind: Kind, version: str, info: dict[str, Any]) -> Path:
    path = data_dir() / kind / version / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path


def sectors_path() -> Path:
    return data_dir() / "meta" / "sectors.csv"


def write_sectors(df: pd.DataFrame) -> Path:
    """Save the ticker -> sector/industry table."""
    path = sectors_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.loc[:, ["ticker", "sector", "industry"]].sort_values("ticker").to_csv(path, index=False)
    return path
