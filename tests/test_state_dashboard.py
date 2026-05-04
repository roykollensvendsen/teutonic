"""Tests for State.flush_dashboard.

Dashboard flush is presentational and rate-limited
(DASHBOARD_FLUSH_MIN_INTERVAL seconds). force=True bypasses the
rate-limit. Critically, it MUST NEVER raise into the main eval loop —
even on R2/Hippius outages the validator must keep evaluating.
"""
import pytest

from validator import State


@pytest.fixture
def r2_mock(mocker):
    storage = {}
    appended = {}
    dashboards = {}
    r2 = mocker.MagicMock()
    r2.get.side_effect = lambda key: storage.get(key)
    r2.put.side_effect = lambda key, data: storage.update({key: data})
    r2.append_jsonl.side_effect = lambda key, rec: appended.setdefault(key, []).append(rec)
    r2.put_dashboard.side_effect = lambda key, data: dashboards.update({key: data})
    r2._dashboards = dashboards
    return r2


# ---------------------------------------------------------------------
# Rate-limit semantics.

def test_flush_dashboard_first_call_succeeds(r2_mock):
    s = State(r2_mock)
    result = s.flush_dashboard()
    # First call (no prior flush) — should run, not be skipped.
    assert result is not False


def test_flush_dashboard_immediate_second_call_returns_false(r2_mock):
    s = State(r2_mock)
    s.flush_dashboard()
    # Immediate second call — within rate-limit window, should be skipped.
    result = s.flush_dashboard()
    assert result is False


def test_flush_dashboard_force_bypasses_rate_limit(r2_mock):
    s = State(r2_mock)
    s.flush_dashboard()
    # force=True must override the rate-limit.
    result = s.flush_dashboard(force=True)
    assert result is not False


# ---------------------------------------------------------------------
# Side effects.

def test_flush_dashboard_calls_put_dashboard_on_r2(r2_mock):
    s = State(r2_mock)
    s.flush_dashboard()
    r2_mock.put_dashboard.assert_called()
    # And the captured payload must be non-empty (something was actually written).
    assert r2_mock._dashboards


def test_flush_dashboard_updates_watchdog_timestamp(r2_mock):
    s = State(r2_mock)
    initial = s.watchdog.get("last_dashboard_flush_at")
    s.flush_dashboard()
    assert s.watchdog["last_dashboard_flush_at"] is not None
    assert s.watchdog["last_dashboard_flush_at"] != initial


# ---------------------------------------------------------------------
# Robustness — must NEVER raise per docstring contract.

def test_flush_dashboard_swallows_r2_exceptions(r2_mock):
    # Simulate an R2 outage during put_dashboard.
    r2_mock.put_dashboard.side_effect = RuntimeError("R2 outage")
    s = State(r2_mock)
    # Must not raise.
    s.flush_dashboard()


def test_flush_dashboard_swallows_internal_exceptions(r2_mock):
    # Even if internal logic blows up (e.g. market data malformed),
    # the call must not propagate.
    s = State(r2_mock)
    # Inject a poison payload that would trip the price-conversion code.
    s.market = {"sn3_alpha_price_tao": "not-a-number"}
    # Must not raise.
    s.flush_dashboard()
