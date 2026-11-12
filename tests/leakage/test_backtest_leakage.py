"""No-lookahead tests for the backtester (contract section 14: L1 and L3).

L1: the pair selector only ever sees formation-window prices.
L3: every trade is executed after its signal day.
Plus the standard perturbation test: changing prices after day T must not change any
trade that was finished by day T.
"""

from pathlib import Path

import pandas as pd
import pytest

import tests.helpers.plugins  # noqa: F401
from ptp.backtest import VectorizedBacktester
from ptp.core import BacktestConfig, register
from tests.helpers.plugins import FixedPairSelector, SimpleZScoreStrategy
from tests.helpers.synthetic import make_market, write_market

SEEN: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {"selector": [], "strategy": []}


@register("pair_selector", "test_spy_selector")
class SpySelector(FixedPairSelector):
    name = "test_spy_selector"

    def select(self, prices, formation_start, formation_end, params):
        SEEN["selector"].append((prices.index.max(), pd.Timestamp(formation_end)))
        return super().select(prices, formation_start, formation_end, params)


@register("strategy", "test_spy_strategy")
class SpyStrategy(SimpleZScoreStrategy):
    name = "test_spy_strategy"

    def generate_signals(self, prices, pairs, trading_start, trading_end, params):
        SEEN["strategy"].append((prices.index.max(), pd.Timestamp(trading_end)))
        return super().generate_signals(prices, pairs, trading_start, trading_end, params)


def _config(**overrides) -> BacktestConfig:
    data = {
        "name": "leakage",
        "walk_forward": {"start": "2015-01-01", "end": "2018-12-31"},
        "pair_selector": "test_fixed_pairs",
        "pair_selector_params": {"pairs": [["AAA", "BBB"], ["AAA", "CCC"]]},
        "strategy": "test_zscore",
        "sizer": "test_equal_dollar",
    }
    data.update(overrides)
    return BacktestConfig.model_validate(data)


@pytest.mark.leakage
def test_l1_selector_sees_only_formation_window(synthetic_market: Path) -> None:
    SEEN["selector"].clear()
    SEEN["strategy"].clear()
    VectorizedBacktester().run(
        _config(pair_selector="test_spy_selector", strategy="test_spy_strategy")
    )
    assert SEEN["selector"], "selector was never called"
    for last_seen, formation_end in SEEN["selector"]:
        assert last_seen <= formation_end
    for last_seen, trading_end in SEEN["strategy"]:
        assert last_seen <= trading_end


@pytest.mark.leakage
def test_l3_entry_after_signal(synthetic_market: Path) -> None:
    trades = VectorizedBacktester().run(_config()).trades
    assert len(trades) > 0
    assert (trades["entry_date"] > trades["signal_date"]).all()
    assert (trades["exit_date"] >= trades["entry_date"]).all()
    not_forced = trades["exit_reason"] != "end_of_window"
    assert (trades.loc[not_forced, "exit_date"] > trades.loc[not_forced, "exit_signal_date"]).all()


@pytest.mark.leakage
def test_changing_the_future_does_not_change_the_past(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cutoff = pd.Timestamp("2017-06-30")
    prices, context, sectors = make_market()

    def run(prices_df: pd.DataFrame, name: str) -> pd.DataFrame:
        data_dir = tmp_path / name
        monkeypatch.setenv("PTP_DATA_DIR", str(data_dir))
        write_market(data_dir, prices_df, context, sectors)
        return VectorizedBacktester().run(_config()).trades

    base = run(prices, "base")
    perturbed = prices.copy()
    future = perturbed["date"] > cutoff
    for col in ["open", "high", "low", "close", "adj_open", "adj_close"]:
        perturbed.loc[future, col] *= 1.5  # change the future
    after = run(perturbed, "perturbed")

    done_before = base[base["exit_date"] <= cutoff].reset_index(drop=True)
    done_after = after[after["exit_date"] <= cutoff].reset_index(drop=True)
    assert len(done_before) > 0
    pd.testing.assert_frame_equal(done_before, done_after)
