import datetime as dt
from pathlib import Path

import pytest
from pydantic import ValidationError

from ptp.core.config import BacktestConfig, load_config, save_config

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "e1_baseline.yaml"


def _config(**overrides) -> BacktestConfig:
    data = {
        "name": "test",
        "walk_forward": {"start": "2015-01-01", "end": "2020-12-31"},
    }
    data.update(overrides)
    return BacktestConfig.model_validate(data)


def test_defaults_match_contract() -> None:
    cfg = _config()
    assert cfg.initial_capital == 1_000_000.0
    assert cfg.walk_forward.formation_months == 12
    assert cfg.walk_forward.trading_months == 6
    assert cfg.costs.commission == 0.0005
    assert cfg.ml.filter is None
    assert cfg.ml.threshold == 0.5
    assert cfg.engine == "vectorized"
    assert cfg.seed == 42


def test_load_contract_example_yaml() -> None:
    cfg = load_config(FIXTURE)
    assert cfg.name == "e1_baseline"
    assert cfg.walk_forward.start == dt.date(2012, 1, 1)
    assert cfg.strategy_params["z_entry"] == 2.0
    assert cfg.engine == "event"


def test_yaml_round_trip(tmp_path: Path) -> None:
    cfg = load_config(FIXTURE)
    out = tmp_path / "copy.yaml"
    save_config(cfg, out)
    assert load_config(out) == cfg


def test_json_round_trip() -> None:
    # The API and the DB (JSONB column) store configs as JSON.
    cfg = _config()
    assert BacktestConfig.model_validate_json(cfg.model_dump_json()) == cfg


@pytest.mark.parametrize(
    "overrides",
    [
        {"costs": {"comission": 0.001}},  # typo
        {"costs": {"commission": -0.001}},  # negative cost
        {"ml": {"threshold": 1.5}},  # probability above 1
        {"engine": "magic"},  # unknown engine
        {"initial_capital": 0},
        {"walk_forward": {"start": "2020-01-01", "end": "2019-01-01"}},
    ],
)
def test_invalid_configs_are_rejected(overrides: dict) -> None:
    with pytest.raises(ValidationError):
        _config(**overrides)


def test_mutable_defaults_are_not_shared() -> None:
    a, b = _config(), _config()
    a.strategy_params["x"] = 1
    assert b.strategy_params == {}


def test_json_schema_available_for_frontend() -> None:
    schema = BacktestConfig.model_json_schema()
    assert "walk_forward" in schema["properties"]
