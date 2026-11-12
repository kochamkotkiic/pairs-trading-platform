import pandas as pd

from ptp.backtest.windows import make_windows
from ptp.core.config import WalkForwardConfig


def _wf(**kw) -> WalkForwardConfig:
    data = {"start": "2015-01-01", "end": "2018-12-31"}
    data.update(kw)
    return WalkForwardConfig.model_validate(data)


def test_default_windows() -> None:
    windows = make_windows(_wf())
    # trading starts 2016-01, 2016-07, ..., 2018-07 -> 6 windows
    assert len(windows) == 6
    w0 = windows[0]
    assert w0.formation_start == pd.Timestamp("2015-01-01")
    assert w0.formation_end == pd.Timestamp("2015-12-31")
    assert w0.trading_start == pd.Timestamp("2016-01-01")
    assert w0.trading_end == pd.Timestamp("2016-06-30")
    assert windows[-1].trading_end == pd.Timestamp("2018-12-31")
    assert [w.window_id for w in windows] == list(range(6))


def test_formation_always_before_trading() -> None:
    for w in make_windows(_wf(formation_months=7, trading_months=2, step_months=1)):
        assert w.formation_start < w.formation_end < w.trading_start <= w.trading_end


def test_trading_windows_do_not_overlap_with_default_step() -> None:
    windows = make_windows(_wf())
    for prev, nxt in zip(windows, windows[1:], strict=False):
        assert prev.trading_end < nxt.trading_start


def test_last_window_cut_at_end() -> None:
    windows = make_windows(_wf(end="2016-03-15"))
    assert len(windows) == 1
    assert windows[0].trading_end == pd.Timestamp("2016-03-15")


def test_range_too_short_gives_no_windows() -> None:
    assert make_windows(_wf(end="2015-06-30")) == []
