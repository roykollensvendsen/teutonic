"""Tests for low-cost trivial helpers across modules.

Bundled into one PR because each helper is too small to warrant its
own test file. Coverage value is low per-helper but cheap to maintain
and fills small gaps in the foundation.
"""
import time
from datetime import datetime

from validator import _monotonic_now, _now

# `_now()` — used as ISO timestamp in State fields like "started_at",
# "timestamp", "phase_since". No type annotation; expected: ISO str.

def test_now_returns_string():
    assert isinstance(_now(), str)


def test_now_returns_parseable_iso_datetime():
    parsed = datetime.fromisoformat(_now())
    assert isinstance(parsed, datetime)


def test_now_is_monotonically_non_decreasing():
    a = _now()
    time.sleep(0.001)
    b = _now()
    assert b >= a


# `_monotonic_now() -> float` — used to measure elapsed time
# (idle = _monotonic_now() - last_event_monotonic). Wraps
# time.monotonic().

def test_monotonic_now_returns_float():
    assert isinstance(_monotonic_now(), float)


def test_monotonic_now_is_strictly_monotonic():
    a = _monotonic_now()
    time.sleep(0.001)
    b = _monotonic_now()
    assert b > a


def test_monotonic_now_elapsed_is_small_for_immediate_calls():
    # Sanity: back-to-back calls should yield sub-second deltas.
    deltas = [_monotonic_now() - _monotonic_now() for _ in range(10)]
    # All deltas non-positive (since a is sampled before b → a-b ≤ 0)
    # — but the magnitude should be tiny.
    assert max(abs(d) for d in deltas) < 1.0
