from pathlib import Path

import pandas as pd
import pytest

from ptp.data import (
    DataNotFoundError,
    get_context,
    get_prices,
    get_sectors,
    get_universe,
    list_versions,
    storage,
    trading_days,
)

from .conftest import TICKERS, make_prices


def test_get_prices_wide_format(data_root: Path, days: pd.DatetimeIndex) -> None:
    df = get_prices(["KO", "PEP"], days[0], days[-1])
    assert list(df.columns) == ["KO", "PEP"]
    assert df.index.name == "date"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.is_monotonic_increasing and df.index.is_unique
    assert len(df) == len(days)


def test_range_is_inclusive(data_root: Path, days: pd.DatetimeIndex) -> None:
    df = get_prices(["KO"], days[10], days[20])
    assert df.index[0] == days[10]
    assert df.index[-1] == days[20]
    assert len(df) == 11


def test_missing_values_stay_nan(data_root: Path, days: pd.DatetimeIndex) -> None:
    df = get_prices(["XOM"], days[0], days[10])
    assert pd.isna(df.loc[days[5], "XOM"])  # not forward-filled
    assert df["XOM"].notna().sum() == 10


def test_unknown_ticker_kept_as_nan_column(data_root: Path, days: pd.DatetimeIndex) -> None:
    df = get_prices(["KO", "NOPE"], days[0], days[5])
    assert list(df.columns) == ["KO", "NOPE"]
    assert df["NOPE"].isna().all()


def test_field_selection(data_root: Path, days: pd.DatetimeIndex) -> None:
    adj = get_prices(["KO"], days[0], days[3], field="adj_close")
    raw = get_prices(["KO"], days[0], days[3], field="close")
    assert (adj["KO"] < raw["KO"]).all()
    with pytest.raises(ValueError, match="Unknown field"):
        get_prices(["KO"], days[0], days[3], field="price")  # type: ignore[arg-type]


def test_end_before_start_rejected(data_root: Path, days: pd.DatetimeIndex) -> None:
    with pytest.raises(ValueError, match="before start"):
        get_prices(["KO"], days[5], days[0])


def test_newest_version_is_default(data_root: Path, days: pd.DatetimeIndex) -> None:
    newer = make_prices(days, TICKERS, seed=1)
    storage.write_prices(newer, "v20210501")
    assert list_versions("prices") == ["v20210401", "v20210501"]
    old = get_prices(["KO"], days[0], days[0], data_version="v20210401")
    new = get_prices(["KO"], days[0], days[0])
    assert old.iloc[0, 0] != new.iloc[0, 0]


def test_missing_version_raises(data_root: Path, days: pd.DatetimeIndex) -> None:
    with pytest.raises(DataNotFoundError, match="not found"):
        get_prices(["KO"], days[0], days[1], data_version="v19990101")


def test_no_data_gives_helpful_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PTP_DATA_DIR", str(tmp_path))
    with pytest.raises(DataNotFoundError, match="python -m ptp.data.download"):
        get_prices(["KO"], pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05"))


def test_trading_days_and_context(data_root: Path, days: pd.DatetimeIndex) -> None:
    td = trading_days(days[0], days[9])
    assert list(td) == list(days[:10])
    ctx = get_context(days[0], days[9])
    assert list(ctx.columns) == ["vix", "spx"]
    assert len(ctx) == 10


def test_sectors_and_universes(data_root: Path, tmp_path: Path) -> None:
    assert get_sectors()["XOM"] == "Energy"
    assert get_universe("sp500") == sorted(TICKERS)
    assert get_universe("sector:Energy") == ["XOM"]
    assert get_universe("sector:consumer staples") == ["KO", "PEP"]

    custom = tmp_path / "universes" / "drinks.csv"
    custom.parent.mkdir(parents=True)
    custom.write_text("ticker\nko\nPEP\nbrk.b\n", encoding="utf-8")
    assert get_universe("custom:drinks.csv") == ["KO", "PEP", "BRK-B"]

    with pytest.raises(ValueError, match="Unknown universe"):
        get_universe("nasdaq")


@pytest.mark.leakage
def test_loader_never_returns_data_after_end(data_root: Path, days: pd.DatetimeIndex) -> None:
    end = days[30]
    df = get_prices(TICKERS, days[0], end)
    assert df.index.max() == end
    assert get_context(days[0], end).index.max() == end
