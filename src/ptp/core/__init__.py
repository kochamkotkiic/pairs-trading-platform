"""Contracts shared by all modules: ABCs, Pydantic types/config, plugin registry. Owner: O3.

Import from here, e.g. ``from ptp.core import Pair, Strategy, register``.
The full specification is ``docs/contract.md``.
"""

from ptp.core.config import (
    BacktestConfig,
    CostConfig,
    MLConfig,
    WalkForwardConfig,
    load_config,
    save_config,
)
from ptp.core.interfaces import (
    Backtester,
    BacktestResult,
    FeatureBuilder,
    PairSelector,
    PositionSizer,
    SignalFilter,
    Strategy,
)
from ptp.core.registry import CATEGORIES, RegistryError, available, get, register
from ptp.core.schemas import SchemaError
from ptp.core.types import (
    Pair,
    PortfolioState,
    RiskParams,
    make_event_id,
    make_pair_id,
    parse_event_id,
    parse_pair_id,
)

__all__ = [
    "CATEGORIES",
    "BacktestConfig",
    "BacktestResult",
    "Backtester",
    "CostConfig",
    "FeatureBuilder",
    "MLConfig",
    "Pair",
    "PairSelector",
    "PortfolioState",
    "PositionSizer",
    "RegistryError",
    "RiskParams",
    "SchemaError",
    "SignalFilter",
    "Strategy",
    "WalkForwardConfig",
    "available",
    "get",
    "load_config",
    "make_event_id",
    "make_pair_id",
    "parse_event_id",
    "parse_pair_id",
    "register",
    "save_config",
]
