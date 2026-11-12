"""Shared data types (contract sections 1, 3.1 and 6).

Every object here is a Pydantic model: creating one validates the input, so bad
data fails loudly at the boundary instead of deep inside a backtest.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

PAIR_SEPARATOR = "__"
EVENT_SEPARATOR = "|"


# ---------------------------------------------------------------------------
# Identifiers (contract section 1)
# ---------------------------------------------------------------------------


def make_pair_id(ticker_a: str, ticker_b: str) -> str:
    """Build a pair id, e.g. ``make_pair_id("KO", "PEP") == "KO__PEP"``."""
    return f"{ticker_a}{PAIR_SEPARATOR}{ticker_b}"


def parse_pair_id(pair_id: str) -> tuple[str, str]:
    """Split a pair id back into ``(ticker_a, ticker_b)``."""
    parts = pair_id.split(PAIR_SEPARATOR)
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Invalid pair_id {pair_id!r}, expected 'A{PAIR_SEPARATOR}B'")
    return parts[0], parts[1]


def make_event_id(pair_id: str, date: dt.date | pd.Timestamp) -> str:
    """Build an ML event id, e.g. ``"KO__PEP|2021-03-04"`` (contract section 5.1)."""
    return f"{pair_id}{EVENT_SEPARATOR}{pd.Timestamp(date):%Y-%m-%d}"


def parse_event_id(event_id: str) -> tuple[str, pd.Timestamp]:
    """Split an event id back into ``(pair_id, date)``."""
    pair_id, sep, date_str = event_id.rpartition(EVENT_SEPARATOR)
    if not sep or not pair_id:
        raise ValueError(f"Invalid event_id {event_id!r}, expected 'PAIR{EVENT_SEPARATOR}DATE'")
    return pair_id, pd.Timestamp(date_str)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    """Base for all contract models: unknown fields are an error, not ignored.

    This catches typos such as ``z_entri: 2.0`` in a YAML config.
    """

    model_config = ConfigDict(extra="forbid")


class Pair(StrictModel):
    """A selected pair with statistics from its formation window (contract 3.1)."""

    pair_id: str  # "KO__PEP"
    ticker_a: str  # dependent variable in A = alpha + beta * B
    ticker_b: str
    hedge_ratio: float  # beta from formation window
    intercept: float  # alpha
    half_life: float  # in trading days
    pvalue: float = Field(ge=0.0, le=1.0)  # raw cointegration test p-value
    pvalue_adj: float | None = Field(default=None, ge=0.0, le=1.0)  # after Benjamini-Hochberg
    hurst: float | None = None
    spread_mean: float  # mean of spread in formation window
    spread_std: float = Field(ge=0.0)  # std of spread in formation window
    formation_start: dt.date
    formation_end: dt.date
    score: float | None = None  # ranking score (higher = better)

    @model_validator(mode="after")
    def _check_consistency(self) -> Pair:
        expected = make_pair_id(self.ticker_a, self.ticker_b)
        if self.pair_id != expected:
            raise ValueError(f"pair_id {self.pair_id!r} does not match tickers ({expected!r})")
        if self.formation_end <= self.formation_start:
            raise ValueError("formation_end must be after formation_start")
        return self


class PortfolioState(StrictModel):
    """Snapshot of the portfolio passed to a PositionSizer (contract section 6)."""

    date: dt.date
    cash: float
    equity: float
    open_positions: dict[str, tuple[float, float]] = Field(default_factory=dict)
    gross_exposure: float = Field(default=0.0, ge=0.0)


class RiskParams(StrictModel):
    """Risk limits (contract section 6).

    Defined in ``core`` (not ``ptp.risk``) because ``BacktestConfig`` contains it
    and ``core`` must not depend on plugin packages.
    """

    capital_per_pair: float = Field(default=0.1, gt=0.0, le=1.0)  # fraction of equity
    max_open_pairs: int = Field(default=10, ge=1)
    max_gross_leverage: float = Field(default=2.0, gt=0.0)
    dollar_neutral: bool = True
    allow_fractional_shares: bool = True
