"""Simulation configuration (contract section 7).

One ``BacktestConfig`` describes a whole simulation. The same schema is used by
the API, the frontend, ``experiments/*.yaml`` and the database (JSONB column).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from ptp.core.types import RiskParams, StrictModel


class WalkForwardConfig(StrictModel):
    start: dt.date
    end: dt.date
    formation_months: int = Field(default=12, ge=1)
    trading_months: int = Field(default=6, ge=1)
    step_months: int = Field(default=6, ge=1)

    @model_validator(mode="after")
    def _check_dates(self) -> WalkForwardConfig:
        if self.end <= self.start:
            raise ValueError("walk_forward.end must be after walk_forward.start")
        return self


class CostConfig(StrictModel):
    commission: float = Field(default=0.0005, ge=0.0)  # per side, fraction of notional
    slippage: float = Field(default=0.0005, ge=0.0)  # per side, fraction of price
    borrow_rate_annual: float = Field(default=0.01, ge=0.0)  # on short leg


class MLConfig(StrictModel):
    filter: str | None = None  # registry name, None = baseline
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    params: dict[str, Any] = Field(default_factory=dict)


class BacktestConfig(StrictModel):
    name: str = Field(min_length=1)
    universe: str = "sp500"
    data_version: str | None = None
    initial_capital: float = Field(default=1_000_000.0, gt=0.0)
    walk_forward: WalkForwardConfig
    pair_selector: str = "coint_bh"
    pair_selector_params: dict[str, Any] = Field(default_factory=dict)
    strategy: str = "zscore"
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    sizer: str = "equal_dollar"
    risk: RiskParams = Field(default_factory=RiskParams)
    costs: CostConfig = Field(default_factory=CostConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    engine: Literal["vectorized", "event"] = "vectorized"
    seed: int = 42


def load_config(path: str | Path) -> BacktestConfig:
    """Load and validate a ``BacktestConfig`` from a YAML file."""
    with Path(path).open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    return BacktestConfig.model_validate(raw)


def save_config(config: BacktestConfig, path: str | Path) -> None:
    """Write a ``BacktestConfig`` to a YAML file (round-trips with ``load_config``)."""
    data = config.model_dump(mode="json")
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
