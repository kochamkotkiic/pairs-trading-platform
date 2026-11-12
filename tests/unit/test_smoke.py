"""Smoke tests: the package and all module packages import cleanly."""

import importlib

import pytest

import ptp

MODULES = [
    "core",
    "data",
    "pairs",
    "strategies",
    "risk",
    "metrics",
    "ml",
    "backtest",
    "db",
    "api",
]


def test_version_is_set() -> None:
    assert isinstance(ptp.__version__, str)
    assert ptp.__version__


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(f"ptp.{module}")
