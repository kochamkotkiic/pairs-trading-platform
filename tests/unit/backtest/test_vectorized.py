"""Tests for the vectorized backtester.

Part 1 uses a tiny hand-made market where every number can be checked on paper.
Part 2 runs the full engine on the synthetic market (tests/helpers/synthetic.py).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tests.helpers.plugins  # noqa: F401  (registers the test plugins)
from ptp.backtest import VectorizedBacktester
from ptp.core import BacktestConfig, get
from ptp.core.schemas import METRIC_KEYS, SchemaError, check_equity, check_trades
from tests.helpers.synthetic import write_market

INITIAL = 1_000_000.0

# ---------------------------------------------------------------------------
# Part 1: hand-made market
# ---------------------------------------------------------------------------


@pytest.fixture
def hand_market(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """XA = 100 until 2020-02-20, then 110 (from the open of 2020-02-21). XB = 50 always."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("PTP_DATA_DIR", str(data_dir))
    days = pd.bdate_range("2020-01-01", "2020-03-31", name="date")
    a = np.where(days >= pd.Timestamp("2020-02-21"), 110.0, 100.0)
    frames = []
    for ticker, px in [("XA", a), ("XB", np.full(len(days), 50.0))]:
        frames.append(
            pd.DataFrame(
                {
                    "date": days,
                    "ticker": ticker,
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "adj_open": px,
                    "adj_close": px,
                    "volume": 1000,
                }
            )
        )
    context = pd.DataFrame({"date": days, "vix": 20.0, "spx": 3000.0})
    sectors = pd.DataFrame({"ticker": ["XA", "XB"], "sector": ["T", "T"], "industry": ["T", "T"]})
    write_market(data_dir, pd.concat(frames, ignore_index=True), context, sectors)
    return data_dir


def _hand_config(schedule: dict[str, int], **overrides) -> BacktestConfig:
    data = {
        "name": "hand",
        "walk_forward": {
            "start": "2020-01-01",
            "end": "2020-03-31",
            "formation_months": 1,
            "trading_months": 1,
            "step_months": 1,
        },
        "pair_selector": "test_fixed_pairs",
        "pair_selector_params": {"pairs": [["XA", "XB"]]},
        "strategy": "test_scripted",
        "strategy_params": {"schedule": schedule},
        "sizer": "test_equal_dollar",
        "costs": {"commission": 0.001, "slippage": 0.001, "borrow_rate_annual": 0.0365},
    }
    data.update(overrides)
    return BacktestConfig.model_validate(data)


def test_single_trade_matches_hand_calculation(hand_market: Path) -> None:
    # Signal Mon 02-10 -> buy at the open of Tue 02-11 (XA 100, XB 50).
    # Exit signal Thu 02-20 -> sell at the open of Fri 02-21 (XA 110, XB 50).
    result = VectorizedBacktester().run(_hand_config({"2020-02-10": 1, "2020-02-20": 0}))
    assert len(result.trades) == 1
    t = result.trades.iloc[0]

    # Sizing: 10% of 1,000,000 per leg -> +1000 XA, -2000 XB.
    assert t.qty_a == pytest.approx(1000)
    assert t.qty_b == pytest.approx(-2000)
    assert t.signal_date == pd.Timestamp("2020-02-10")
    assert t.entry_date == pd.Timestamp("2020-02-11")
    assert t.exit_signal_date == pd.Timestamp("2020-02-20")
    assert t.exit_date == pd.Timestamp("2020-02-21")
    assert t.holding_days == 8  # trading days 02-11 -> 02-21
    assert t.event_id == "XA__XB|2020-02-10"
    assert t.window_id == 0  # window 0 = formation January, trading February

    # Fills with 0.1% slippage: buy XA 100.1, short XB 49.95, sell XA 109.89, cover XB 50.05
    assert t.entry_price_a == pytest.approx(100.1)
    assert t.entry_price_b == pytest.approx(49.95)
    assert t.exit_price_a == pytest.approx(109.89)
    assert t.exit_price_b == pytest.approx(50.05)

    # gross = 1000*(110-100) + (-2000)*(50-50) = 10,000
    # commission = 0.1% * (200,000 entry + 210,000 exit) = 410
    # slippage   = 0.1% * (200,000 + 210,000)            = 410
    # borrow     = 100,000 short * 3.65% * 10 days / 365  = 100
    assert t.gross_pnl == pytest.approx(10_000)
    assert t.costs == pytest.approx(920)
    assert t.pnl == pytest.approx(9_080)
    assert t.ret == pytest.approx(9_080 / 200_000)
    assert t.exit_reason == "mean_revert"
    assert np.isnan(t.ml_proba)

    eq = result.equity.set_index("date")["equity"]
    # Entry day: marked at the close (XA 100, XB 50) minus entry commission and slippage.
    # 1000*(100-100.1) + (-2000)*(50-49.95) - 200 = -400
    assert eq[pd.Timestamp("2020-02-11")] == pytest.approx(INITIAL - 400)
    assert eq[pd.Timestamp("2020-02-21")] == pytest.approx(INITIAL + 9_080)
    assert eq.iloc[-1] == pytest.approx(INITIAL + 9_080)


def test_window_ids_and_months(hand_market: Path) -> None:
    result = VectorizedBacktester().run(_hand_config({}))
    # start 2020-01: window 0 = formation Jan / trading Feb, window 1 = formation Feb / trading Mar
    assert result.equity["date"].min() == pd.Timestamp("2020-02-03")
    assert result.equity["date"].max() == pd.Timestamp("2020-03-31")
    assert result.trades.empty


def test_reversal_closes_and_reopens(hand_market: Path) -> None:
    schedule = {"2020-02-10": 1, "2020-02-14": -1, "2020-02-20": 0}
    trades = VectorizedBacktester().run(_hand_config(schedule)).trades
    assert trades["direction"].tolist() == [1, -1]
    assert trades["exit_reason"].tolist() == ["reversal", "mean_revert"]
    # Fri 02-14 decision -> both the exit and the new entry happen at the open of Mon 02-17
    assert trades.iloc[0].exit_date == trades.iloc[1].entry_date == pd.Timestamp("2020-02-17")


def test_open_position_closed_at_end_of_window(hand_market: Path) -> None:
    trades = VectorizedBacktester().run(_hand_config({"2020-02-26": 1})).trades
    assert len(trades) == 1
    t = trades.iloc[0]
    assert t.exit_reason == "end_of_window"
    assert t.exit_date == pd.Timestamp("2020-02-28")  # last trading day of February


def test_entry_signal_on_last_day_is_ignored(hand_market: Path) -> None:
    # No next day inside the window to execute on.
    trades = VectorizedBacktester().run(_hand_config({"2020-02-28": 1})).trades
    assert trades.empty


def test_exit_without_exit_event_is_rejected(hand_market: Path) -> None:
    cfg = _hand_config(
        {"2020-02-10": 1, "2020-02-20": 0},
        strategy_params={"schedule": {"2020-02-10": 1, "2020-02-20": 0}, "exit_event": "flat"},
    )
    with pytest.raises(SchemaError, match="expected exit_"):
        VectorizedBacktester().run(cfg)


# ---------------------------------------------------------------------------
# Part 2: synthetic market
# ---------------------------------------------------------------------------


def _synthetic_config(**overrides) -> BacktestConfig:
    data = {
        "name": "synthetic",
        "walk_forward": {"start": "2015-01-01", "end": "2018-12-31"},
        "pair_selector": "test_fixed_pairs",
        "pair_selector_params": {"pairs": [["AAA", "BBB"], ["AAA", "CCC"]]},
        "strategy": "test_zscore",
        "sizer": "test_equal_dollar",
    }
    data.update(overrides)
    return BacktestConfig.model_validate(data)


def test_full_run_produces_valid_result(synthetic_market: Path) -> None:
    progress: list[float] = []
    result = VectorizedBacktester().run(_synthetic_config(), progress=progress.append)

    check_trades(result.trades)
    check_equity(result.equity)
    assert set(METRIC_KEYS) <= set(result.metrics)
    assert result.data_version == "v20190101"
    assert len(result.trades) > 10
    assert (result.trades["entry_date"] > result.trades["signal_date"]).all()
    assert result.trades["trade_id"].tolist() == list(range(1, len(result.trades) + 1))
    assert progress == sorted(progress)
    assert progress[-1] == 1.0
    # Money adds up: final equity = initial + every trade's net P&L (all closed at the end)
    assert result.equity["equity"].iloc[-1] == pytest.approx(INITIAL + result.trades["pnl"].sum())
    assert result.equity["n_open_pairs"].iloc[-1] == 0


def test_true_pair_mean_reverts_fake_pair_stops_out(synthetic_market: Path) -> None:
    trades = VectorizedBacktester().run(_synthetic_config()).trades
    reasons = trades.groupby("pair_id")["exit_reason"].agg(lambda s: s.value_counts().idxmax())
    assert reasons["AAA__BBB"] == "mean_revert"
    assert reasons["AAA__CCC"] == "stop_loss"


def test_results_are_reproducible(synthetic_market: Path) -> None:
    a = VectorizedBacktester().run(_synthetic_config())
    b = VectorizedBacktester().run(_synthetic_config())
    pd.testing.assert_frame_equal(a.trades, b.trades)
    pd.testing.assert_frame_equal(a.equity, b.equity)


def test_max_open_pairs_respected(synthetic_market: Path) -> None:
    result = VectorizedBacktester().run(_synthetic_config(risk={"max_open_pairs": 1}))
    assert result.equity["n_open_pairs"].max() <= 1
    assert len(result.trades) > 0


def test_leverage_limit_blocks_trades(synthetic_market: Path) -> None:
    # Each trade needs 2 x 10% = 20% of equity gross; the limit allows only 10%.
    result = VectorizedBacktester().run(_synthetic_config(risk={"max_gross_leverage": 0.1}))
    assert result.trades.empty
    assert (result.equity["equity"] == INITIAL).all()


def test_higher_costs_lower_pnl(synthetic_market: Path) -> None:
    cheap = VectorizedBacktester().run(
        _synthetic_config(costs={"commission": 0.0, "slippage": 0.0})
    )
    pricey = VectorizedBacktester().run(_synthetic_config(costs={"commission": 0.002}))
    assert pricey.trades["pnl"].sum() < cheap.trades["pnl"].sum()
    # pnl = gross_pnl - costs for every trade
    t = pricey.trades
    assert (t["pnl"] - (t["gross_pnl"] - t["costs"])).abs().max() < 1e-6
    # Without commission and slippage, only the borrow cost is left (short leg only)
    c = cheap.trades
    short_notional = c["qty_b"].abs() * c["entry_price_b"]  # direction +1 shorts B
    short_notional = short_notional.where(c["qty_b"] < 0, c["qty_a"].abs() * c["entry_price_a"])
    days = (c["exit_date"] - c["entry_date"]).dt.days
    expected_borrow = short_notional * 0.01 * days / 365
    assert c["costs"].to_numpy() == pytest.approx(expected_borrow.to_numpy())


def test_ml_filter_not_supported_yet(synthetic_market: Path) -> None:
    with pytest.raises(NotImplementedError, match="F2"):
        VectorizedBacktester().run(_synthetic_config(ml={"filter": "xgboost"}))


def test_engine_registered() -> None:
    assert get("backtester", "vectorized") is VectorizedBacktester
