"""Fixtures shared by all test folders."""

from pathlib import Path

import pytest

from tests.helpers.synthetic import make_market, write_market


@pytest.fixture
def synthetic_market(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Fake market (AAA/BBB cointegrated, CCC independent, 2015-2018) as the data dir."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("PTP_DATA_DIR", str(data_dir))
    write_market(data_dir, *make_market())
    return data_dir
