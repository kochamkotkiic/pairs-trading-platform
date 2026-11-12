"""Plugin interfaces (contract sections 3.2, 4.2, 5.2, 5.4, 6 and 8).

Each class is an abstract base class (ABC): a template that says which methods a
plugin must have. Python refuses to create an object of a subclass that forgets
one of the ``@abstractmethod`` methods, so a missing method is caught immediately.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pandas as pd
from pydantic import BaseModel

from ptp.core.config import BacktestConfig
from ptp.core.types import Pair, PortfolioState, RiskParams


class PairSelector(ABC):
    """Finds pairs in the formation window (owner: O1)."""

    name: ClassVar[str]
    params_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def select(
        self,
        prices: pd.DataFrame,  # wide adj_close, ONLY formation window
        formation_start: pd.Timestamp,
        formation_end: pd.Timestamp,
        params: BaseModel,
    ) -> list[Pair]:
        """Return selected pairs sorted by score (best first), at most ``params.max_pairs``."""


class Strategy(ABC):
    """Turns prices into target positions per pair (owner: O1)."""

    name: ClassVar[str]
    params_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,  # wide adj_close, from formation_start to trading_end
        pairs: list[Pair],
        trading_start: pd.Timestamp,
        trading_end: pd.Timestamp,
        params: BaseModel,
    ) -> pd.DataFrame:
        """One row per (date, pair_id) for every trading day in [trading_start, trading_end].

        Columns: see ``ptp.core.schemas.SIGNAL_COLUMNS``. Row for day ``t`` may use
        only data up to ``t`` (rule L2).
        """


class FeatureBuilder(ABC):
    """Builds ML features for entry events (owner: O2)."""

    name: ClassVar[str]

    @abstractmethod
    def build(
        self,
        events: pd.DataFrame,  # index = event_id; cols: date, pair_id, zscore, hedge_ratio, ...
        prices: pd.DataFrame,  # wide adj_close
        volume: pd.DataFrame,  # wide volume
        context: pd.DataFrame,  # vix, spx
        pairs: dict[str, Pair],
    ) -> pd.DataFrame:
        """Return index = event_id, numeric feature columns, no NaN (rule L4)."""


class SignalFilter(ABC):
    """ML model deciding whether to take an entry (owner: O2)."""

    name: ClassVar[str]
    params_model: ClassVar[type[BaseModel]]

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, t1: pd.Series) -> None:
        """Train. ``t1`` = date each label became known (exit date), for purged CV."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Return index = event_id, values = P(success) in [0, 1]."""

    def save(self, path: Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement save()")

    @classmethod
    def load(cls, path: Path) -> SignalFilter:
        raise NotImplementedError(f"{cls.__name__} does not implement load()")


class PositionSizer(ABC):
    """Decides how many shares of each leg to trade (owner: O1)."""

    name: ClassVar[str]

    @abstractmethod
    def size(
        self,
        pair: Pair,
        direction: int,  # +1 / -1
        price_a: float,  # execution price (open t+1)
        price_b: float,
        portfolio: PortfolioState,
        params: RiskParams,
    ) -> tuple[float, float]:
        """Return ``(qty_a, qty_b)`` in shares, signed. ``(0, 0)`` = reject."""


@dataclass
class BacktestResult:
    """Everything a backtest produces (contract section 8)."""

    config: BacktestConfig
    data_version: str
    trades: pd.DataFrame  # schema: TRADE_COLUMNS
    equity: pd.DataFrame  # schema: EQUITY_COLUMNS
    signals: pd.DataFrame  # schema: SIGNAL_COLUMNS (all windows concatenated)
    pairs: list[Pair]  # all selected pairs (with formation dates)
    metrics: dict[str, float]  # keys: METRIC_KEYS
    duration_s: float


class Backtester(ABC):
    """Runs a whole simulation (owner: O3)."""

    name: ClassVar[str]  # "vectorized" | "event"

    @abstractmethod
    def run(
        self,
        config: BacktestConfig,
        progress: Callable[[float], None] | None = None,
    ) -> BacktestResult:
        """Run the walk-forward simulation. ``progress`` receives values in [0, 1]."""
