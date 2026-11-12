import pandas as pd
import pytest

from ptp.core.schemas import (
    METRIC_KEYS,
    TRADE_COLUMNS,
    SchemaError,
    check_equity,
    check_metrics,
    check_signals,
    check_trades,
)


def _signals(**overrides) -> pd.DataFrame:
    data = {
        "date": pd.to_datetime(["2021-01-04", "2021-01-05"]),
        "pair_id": ["KO__PEP", "KO__PEP"],
        "target": [1, 0],
        "zscore": [-2.1, -0.3],
        "spread": [-0.1, -0.01],
        "hedge_ratio": [0.8, 0.8],
        "event": ["entry", "exit_mean_revert"],
    }
    data.update(overrides)
    return pd.DataFrame(data)


def _trades(signal_date: str, entry_date: str) -> pd.DataFrame:
    row = dict.fromkeys(TRADE_COLUMNS, 0)
    row.update(
        trade_id=1,
        signal_date=pd.Timestamp(signal_date),
        entry_date=pd.Timestamp(entry_date),
        exit_reason="mean_revert",
    )
    return pd.DataFrame([row])


def test_valid_signals_pass() -> None:
    check_signals(_signals())


def test_missing_column_reported() -> None:
    with pytest.raises(SchemaError, match="zscore"):
        check_signals(_signals().drop(columns="zscore"))


def test_invalid_target_rejected() -> None:
    with pytest.raises(SchemaError, match="target"):
        check_signals(_signals(target=[2, 0]))


def test_duplicate_signal_rows_rejected() -> None:
    with pytest.raises(SchemaError, match="duplicate"):
        check_signals(_signals(date=pd.to_datetime(["2021-01-04", "2021-01-04"])))


def test_trades_pass_when_entry_after_signal() -> None:
    check_trades(_trades("2021-01-04", "2021-01-05"))


@pytest.mark.leakage
def test_trades_rule_l3_same_day_entry_rejected() -> None:
    with pytest.raises(SchemaError, match="L3"):
        check_trades(_trades("2021-01-04", "2021-01-04"))


def test_equity_dates_must_increase() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-05", "2021-01-04"]),
            "equity": [1.0, 1.0],
            "cash": [1.0, 1.0],
            "gross_exposure": [0.0, 0.0],
            "n_open_pairs": [0, 0],
            "daily_ret": [0.0, 0.0],
        }
    )
    with pytest.raises(SchemaError):
        check_equity(df)


def test_metrics_keys() -> None:
    check_metrics(dict.fromkeys(METRIC_KEYS, 0.0))
    with pytest.raises(SchemaError, match="sharpe"):
        check_metrics({k: 0.0 for k in METRIC_KEYS if k != "sharpe"})
