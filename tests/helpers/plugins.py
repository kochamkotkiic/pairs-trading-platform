"""Minimal plugins for testing the backtester before O1/O2 deliver the real ones.

They follow the contract interfaces exactly but are deliberately simple. They live in
tests/ and use "test_" names, so they never clash with the real plugins.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from ptp.core import Pair, PairSelector, PositionSizer, Strategy, make_pair_id, register
from ptp.core.types import PortfolioState, RiskParams

# ---------------------------------------------------------------------------
# Pair selector
# ---------------------------------------------------------------------------


class FixedPairsParams(BaseModel):
    pairs: list[tuple[str, str]] = Field(default_factory=lambda: [("AAA", "BBB")])
    max_pairs: int = 10


@register("pair_selector", "test_fixed_pairs")
class FixedPairSelector(PairSelector):
    """Returns the given pairs, with OLS statistics from the formation window."""

    name = "test_fixed_pairs"
    params_model = FixedPairsParams

    def select(self, prices, formation_start, formation_end, params) -> list[Pair]:
        out = []
        for a, b in params.pairs[: params.max_pairs]:
            if a not in prices.columns or b not in prices.columns:
                continue
            la, lb = np.log(prices[a].to_numpy()), np.log(prices[b].to_numpy())
            if np.std(lb) > 1e-12:
                beta, alpha = np.polyfit(lb, la, 1)
            else:  # flat prices (hand-made test data)
                beta, alpha = 1.0, float(np.mean(la - lb))
            spread = la - beta * lb - alpha
            half_life = _half_life(spread)
            out.append(
                Pair(
                    pair_id=make_pair_id(a, b),
                    ticker_a=a,
                    ticker_b=b,
                    hedge_ratio=float(beta),
                    intercept=float(alpha),
                    half_life=half_life,
                    pvalue=0.0,
                    pvalue_adj=None,
                    hurst=None,
                    spread_mean=float(np.mean(spread)),
                    spread_std=float(np.std(spread)),
                    formation_start=pd.Timestamp(formation_start).date(),
                    formation_end=pd.Timestamp(formation_end).date(),
                )
            )
        return out


def _half_life(spread: np.ndarray) -> float:
    lagged, diff = spread[:-1], np.diff(spread)
    if np.std(lagged) < 1e-12:
        return math.inf
    lam = np.polyfit(lagged, diff, 1)[0]
    return float(-math.log(2) / lam) if lam < 0 else math.inf


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class SimpleZParams(BaseModel):
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 4.0


@register("strategy", "test_zscore")
class SimpleZScoreStrategy(Strategy):
    """Z-score from formation-window mean/std (static, so no look-ahead)."""

    name = "test_zscore"
    params_model = SimpleZParams

    def generate_signals(self, prices, pairs, trading_start, trading_end, params) -> pd.DataFrame:
        days = prices.loc[trading_start:trading_end].index
        rows = []
        for p in pairs:
            spread = (
                np.log(prices.loc[days, p.ticker_a])
                - p.hedge_ratio * np.log(prices.loc[days, p.ticker_b])
                - p.intercept
            )
            z = (spread - p.spread_mean) / p.spread_std
            target = 0
            for d in days:
                zt = float(z[d])
                prev, event = target, "flat"
                if prev == 0 and abs(zt) >= params.z_entry:
                    target, event = (-1 if zt > 0 else 1), "entry"
                elif prev != 0 and abs(zt) >= params.z_stop:
                    target, event = 0, "exit_stop_loss"
                elif prev != 0 and abs(zt) <= params.z_exit:
                    target, event = 0, "exit_mean_revert"
                elif prev != 0:
                    event = "hold"
                rows.append((d, p.pair_id, target, zt, float(spread[d]), p.hedge_ratio, event))
        return pd.DataFrame(
            rows, columns=["date", "pair_id", "target", "zscore", "spread", "hedge_ratio", "event"]
        )


class ScriptedParams(BaseModel):
    # date (YYYY-MM-DD) -> new target from that day on, for every pair
    schedule: dict[str, int] = Field(default_factory=dict)
    exit_event: str = "exit_mean_revert"


@register("strategy", "test_scripted")
class ScriptedStrategy(Strategy):
    """Targets dictated by the test, for hand-calculated results."""

    name = "test_scripted"
    params_model = ScriptedParams

    def generate_signals(self, prices, pairs, trading_start, trading_end, params) -> pd.DataFrame:
        days = prices.loc[trading_start:trading_end].index
        changes = {pd.Timestamp(k): v for k, v in params.schedule.items()}
        rows = []
        for p in pairs:
            target = 0
            for d in days:
                prev = target
                target = changes.get(d, target)
                if prev == 0 and target != 0:
                    event = "entry"
                elif prev != 0 and target == 0:
                    event = params.exit_event
                elif prev != 0 and target != prev:
                    event = "entry"  # reversal
                else:
                    event = "hold" if target else "flat"
                rows.append((d, p.pair_id, target, 0.0, 0.0, p.hedge_ratio, event))
        return pd.DataFrame(
            rows, columns=["date", "pair_id", "target", "zscore", "spread", "hedge_ratio", "event"]
        )


# ---------------------------------------------------------------------------
# Position sizer
# ---------------------------------------------------------------------------


@register("sizer", "test_equal_dollar")
class EqualDollarSizer(PositionSizer):
    """capital_per_pair * equity on each leg, long A / short B for direction +1."""

    name = "test_equal_dollar"

    def size(
        self,
        pair: Pair,
        direction: int,
        price_a: float,
        price_b: float,
        portfolio: PortfolioState,
        params: RiskParams,
    ) -> tuple[float, float]:
        notional = params.capital_per_pair * portfolio.equity
        qty_a = direction * notional / price_a
        qty_b = -direction * notional / price_b
        if not params.allow_fractional_shares:
            qty_a, qty_b = float(math.trunc(qty_a)), float(math.trunc(qty_b))
        return qty_a, qty_b
