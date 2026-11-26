"""Transaction cost model (contract section 7, ``CostConfig``).

- Slippage: a buy fills ``slippage`` above the quoted price, a sell ``slippage`` below.
- Commission: ``commission`` x traded notional, on entry and on exit.
- Borrow: ``borrow_rate_annual`` x short notional at entry x calendar days held / 365.
"""

from __future__ import annotations

import math


def fill_price(price: float, qty: float, slippage: float) -> float:
    """Execution price for trading ``qty`` shares (``qty > 0`` buy, ``qty < 0`` sell)."""
    return price * (1 + math.copysign(slippage, qty)) if qty else price


def commission(notional: float, rate: float) -> float:
    return abs(notional) * rate


def borrow_cost(short_notional: float, rate_annual: float, calendar_days: int) -> float:
    return abs(short_notional) * rate_annual * max(calendar_days, 0) / 365.0
