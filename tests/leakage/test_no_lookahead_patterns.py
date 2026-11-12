"""Rule L7 (contract section 14): no bfill() and no shift(-n) in production code.

Both patterns pull future values into the past and are the most common source of
look-ahead bias. This is a static check over every .py file in src/ptp.
"""

import re
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[2] / "src" / "ptp"

FORBIDDEN = {
    "bfill()": re.compile(r"\.bfill\s*\("),
    "backfill()": re.compile(r"\.backfill\s*\("),
    "fillna(method='bfill')": re.compile(r"method\s*=\s*['\"](bfill|backfill)['\"]"),
    "shift(-n)": re.compile(r"\.shift\s*\(\s*(periods\s*=\s*)?-"),
}


def _violations() -> list[str]:
    found = []
    for path in sorted(SRC_DIR.rglob("*.py")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            for name, pattern in FORBIDDEN.items():
                if pattern.search(code):
                    rel = path.relative_to(SRC_DIR.parent)
                    found.append(f"{rel}:{lineno}: {name}: {line.strip()}")
    return found


@pytest.mark.leakage
def test_src_dir_exists() -> None:
    assert SRC_DIR.is_dir()


@pytest.mark.leakage
def test_no_forbidden_lookahead_patterns() -> None:
    violations = _violations()
    assert not violations, "Look-ahead patterns found (rule L7):\n" + "\n".join(violations)


@pytest.mark.leakage
@pytest.mark.parametrize(
    "line",
    ["df.bfill()", "s.fillna(method='bfill')", "x.shift(-1)", "x.shift(periods=-5)"],
)
def test_patterns_catch_known_bad_code(line: str) -> None:
    assert any(p.search(line) for p in FORBIDDEN.values())


@pytest.mark.leakage
@pytest.mark.parametrize("line", ["df.ffill()", "x.shift(1)", "x.shift(periods=5)"])
def test_patterns_allow_safe_code(line: str) -> None:
    assert not any(p.search(line) for p in FORBIDDEN.values())
