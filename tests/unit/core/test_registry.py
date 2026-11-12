from collections.abc import Iterator

import pandas as pd
import pytest
from pydantic import BaseModel

from ptp.core import registry
from ptp.core.interfaces import Strategy
from ptp.core.registry import RegistryError, available, get, register


class _Params(BaseModel):
    lookback: int = 60


@pytest.fixture
def clean_name() -> Iterator[str]:
    name = "_test_strategy"
    yield name
    registry.unregister("strategy", name)


def _make_strategy(name: str) -> type[Strategy]:
    class _S(Strategy):
        params_model = _Params

        def generate_signals(self, prices, pairs, trading_start, trading_end, params):
            return pd.DataFrame()

    _S.name = name
    return _S


def test_register_and_get(clean_name: str) -> None:
    cls = register("strategy", clean_name)(_make_strategy(clean_name))
    assert get("strategy", clean_name) is cls
    assert clean_name in available("strategy")
    # The usage pattern from the contract:
    params = get("strategy", clean_name).params_model(lookback=30)
    assert params.lookback == 30


def test_unknown_name_lists_available() -> None:
    with pytest.raises(RegistryError, match="No strategy named 'nope'"):
        get("strategy", "nope")


def test_unknown_category() -> None:
    with pytest.raises(RegistryError, match="Unknown category"):
        get("strategy_typo", "x")  # type: ignore[arg-type]


def test_wrong_base_class_rejected() -> None:
    class NotAStrategy:
        name = "x"

    with pytest.raises(TypeError, match="must subclass Strategy"):
        register("strategy", "x")(NotAStrategy)


def test_name_attribute_must_match(clean_name: str) -> None:
    with pytest.raises(ValueError, match="registered as"):
        register("strategy", clean_name)(_make_strategy("other_name"))


def test_duplicate_name_from_different_class_rejected(clean_name: str) -> None:
    register("strategy", clean_name)(_make_strategy(clean_name))

    class Other(Strategy):
        name = clean_name
        params_model = _Params

        def generate_signals(self, prices, pairs, trading_start, trading_end, params):
            return pd.DataFrame()

    with pytest.raises(RegistryError, match="already registered"):
        register("strategy", clean_name)(Other)
