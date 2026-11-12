"""Backtesting engines (vectorized and event-driven), costs, portfolio. Owner: O3.

Importing this package registers the engines, e.g. ``get("backtester", "vectorized")``.
"""

from ptp.backtest.vectorized import VectorizedBacktester
from ptp.backtest.windows import Window, make_windows

__all__ = ["VectorizedBacktester", "Window", "make_windows"]
