"""Walk-forward windows (contract section 8, step 1).

``walk_forward.start`` is the start of the FIRST FORMATION window, so the first trading
day is ``formation_months`` later. Windows move forward by ``step_months``; the last
trading window is cut at ``walk_forward.end``.

    |-- formation (12m) --|-- trading (6m) --|
                |-- formation (12m) --|-- trading (6m) --|      (step 6m)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ptp.core.config import WalkForwardConfig


@dataclass(frozen=True)
class Window:
    window_id: int
    formation_start: pd.Timestamp
    formation_end: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp


def make_windows(wf: WalkForwardConfig) -> list[Window]:
    """Split ``[wf.start, wf.end]`` into consecutive formation/trading windows."""
    start, end = pd.Timestamp(wf.start), pd.Timestamp(wf.end)
    one_day = pd.Timedelta(days=1)
    windows: list[Window] = []
    k = 0
    while True:
        formation_start = start + pd.DateOffset(months=k * wf.step_months)
        trading_start = formation_start + pd.DateOffset(months=wf.formation_months)
        if trading_start > end:
            break
        trading_end = min(trading_start + pd.DateOffset(months=wf.trading_months) - one_day, end)
        windows.append(
            Window(
                window_id=k,
                formation_start=formation_start,
                formation_end=trading_start - one_day,
                trading_start=trading_start,
                trading_end=trading_end,
            )
        )
        k += 1
    return windows
