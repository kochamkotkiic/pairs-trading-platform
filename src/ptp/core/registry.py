"""Plugin registry (contract section 11).

The registry is a "phone book": plugins sign up under a category and a name, and
the backtester / API look them up by the name written in ``BacktestConfig``.

    @register("strategy", "zscore")
    class ZScoreStrategy(Strategy):
        name = "zscore"
        ...

    strategy_cls = get("strategy", config.strategy)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal, TypeVar, get_args

from ptp.core.interfaces import (
    Backtester,
    FeatureBuilder,
    PairSelector,
    PositionSizer,
    SignalFilter,
    Strategy,
)

Category = Literal[
    "pair_selector", "strategy", "signal_filter", "feature_builder", "sizer", "backtester"
]
CATEGORIES: tuple[str, ...] = get_args(Category)

# Each category requires plugins to subclass this interface.
BASE_CLASSES: dict[str, type] = {
    "pair_selector": PairSelector,
    "strategy": Strategy,
    "signal_filter": SignalFilter,
    "feature_builder": FeatureBuilder,
    "sizer": PositionSizer,
    "backtester": Backtester,
}

_REGISTRY: dict[str, dict[str, type]] = {c: {} for c in CATEGORIES}

T = TypeVar("T", bound=type)


class RegistryError(KeyError):
    """Raised for unknown categories/names or conflicting registrations."""

    def __str__(self) -> str:  # KeyError would otherwise print the message in quotes
        return str(self.args[0]) if self.args else ""


def _check_category(category: str) -> None:
    if category not in _REGISTRY:
        raise RegistryError(f"Unknown category {category!r}. Valid: {', '.join(CATEGORIES)}")


def register(category: Category, name: str) -> Callable[[T], T]:
    """Class decorator that adds a plugin to the registry."""
    _check_category(category)

    def decorator(cls: T) -> T:
        base = BASE_CLASSES[category]
        if not issubclass(cls, base):
            raise TypeError(f"{cls.__name__} must subclass {base.__name__} to be a {category}")
        cls_name = getattr(cls, "name", None)
        if cls_name != name:
            raise ValueError(
                f"{cls.__name__}.name is {cls_name!r} but it is registered as {name!r}"
            )
        existing = _REGISTRY[category].get(name)
        # Re-importing the same module (e.g. Streamlit reruns) is fine; two different
        # classes with the same name are not.
        if existing is not None and (existing.__module__, existing.__qualname__) != (
            cls.__module__,
            cls.__qualname__,
        ):
            raise RegistryError(
                f"{category} {name!r} is already registered by "
                f"{existing.__module__}.{existing.__qualname__}"
            )
        _REGISTRY[category][name] = cls
        return cls

    return decorator


def get(category: Category, name: str) -> type:
    """Look up a plugin class by category and name."""
    _check_category(category)
    try:
        return _REGISTRY[category][name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY[category])) or "(none registered)"
        raise RegistryError(f"No {category} named {name!r}. Available: {known}") from None


def available(category: Category) -> list[str]:
    """Names of all plugins registered in a category, sorted."""
    _check_category(category)
    return sorted(_REGISTRY[category])


def unregister(category: Category, name: str) -> None:
    """Remove a plugin (mainly for tests)."""
    _check_category(category)
    _REGISTRY[category].pop(name, None)
