import datetime as dt

import pandas as pd
import pytest
from pydantic import ValidationError

from ptp.core.types import (
    Pair,
    PortfolioState,
    RiskParams,
    make_event_id,
    make_pair_id,
    parse_event_id,
    parse_pair_id,
)


def _pair(**overrides) -> Pair:
    data = dict(
        pair_id="KO__PEP",
        ticker_a="KO",
        ticker_b="PEP",
        hedge_ratio=0.8,
        intercept=0.1,
        half_life=12.0,
        pvalue=0.01,
        pvalue_adj=0.04,
        hurst=0.4,
        spread_mean=0.0,
        spread_std=0.05,
        formation_start=dt.date(2020, 1, 1),
        formation_end=dt.date(2020, 12, 31),
    )
    data.update(overrides)
    return Pair(**data)


def test_pair_and_event_ids_round_trip() -> None:
    assert make_pair_id("KO", "PEP") == "KO__PEP"
    assert parse_pair_id("BRK-B__MSFT") == ("BRK-B", "MSFT")
    event_id = make_event_id("KO__PEP", pd.Timestamp("2021-03-04"))
    assert event_id == "KO__PEP|2021-03-04"
    assert parse_event_id(event_id) == ("KO__PEP", pd.Timestamp("2021-03-04"))


@pytest.mark.parametrize("bad", ["KOPEP", "KO__", "A__B__C"])
def test_parse_pair_id_rejects_bad_ids(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_pair_id(bad)


def test_valid_pair() -> None:
    assert _pair().score is None


def test_pair_id_must_match_tickers() -> None:
    with pytest.raises(ValidationError, match="does not match"):
        _pair(pair_id="KO__MSFT")


def test_pair_formation_dates_ordered() -> None:
    with pytest.raises(ValidationError, match="formation_end"):
        _pair(formation_end=dt.date(2019, 1, 1))


def test_pair_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _pair(hedge_ratoi=1.0)


def test_risk_params_defaults_and_limits() -> None:
    risk = RiskParams()
    assert risk.capital_per_pair == 0.1
    assert risk.max_open_pairs == 10
    with pytest.raises(ValidationError):
        RiskParams(capital_per_pair=10)  # percent instead of fraction


def test_portfolio_state() -> None:
    state = PortfolioState(date=dt.date(2021, 1, 4), cash=1e6, equity=1e6)
    assert state.open_positions == {}
