from pathlib import Path

import pandas as pd
import pytest
from pydantic import BaseModel

from ptp.core.interfaces import SignalFilter, Strategy


class _Params(BaseModel):
    pass


def test_incomplete_plugin_cannot_be_created() -> None:
    class Incomplete(Strategy):
        name = "incomplete"
        params_model = _Params

    with pytest.raises(TypeError, match="generate_signals"):
        Incomplete()


def test_complete_plugin_can_be_created() -> None:
    class Complete(Strategy):
        name = "complete"
        params_model = _Params

        def generate_signals(self, prices, pairs, trading_start, trading_end, params):
            return pd.DataFrame()

    assert isinstance(Complete(), Strategy)


def test_signal_filter_save_not_implemented_by_default(tmp_path: Path) -> None:
    class Dummy(SignalFilter):
        name = "dummy"
        params_model = _Params

        def fit(self, X, y, t1):
            pass

        def predict_proba(self, X):
            return pd.Series(0.5, index=X.index)

    with pytest.raises(NotImplementedError):
        Dummy().save(tmp_path / "model.pkl")
