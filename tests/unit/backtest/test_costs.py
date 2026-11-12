import pytest

from ptp.backtest.costs import borrow_cost, commission, fill_price


def test_buy_fills_above_sell_below() -> None:
    assert fill_price(100.0, 10, 0.001) == pytest.approx(100.1)
    assert fill_price(100.0, -10, 0.001) == pytest.approx(99.9)
    assert fill_price(100.0, 0, 0.001) == 100.0


def test_commission_on_absolute_notional() -> None:
    assert commission(-50_000.0, 0.0005) == pytest.approx(25.0)


def test_borrow_cost_per_calendar_day() -> None:
    # 100k short, 3.65% a year, 10 days -> 100 USD
    assert borrow_cost(100_000.0, 0.0365, 10) == pytest.approx(100.0)
    assert borrow_cost(100_000.0, 0.0365, 0) == 0.0
