import asyncio
from datetime import UTC, datetime, timedelta

from validator import (
    _age_seconds,
    _is_transient_eval_error,
    _rank_sort_key,
    _safe_block,
)


def _entry(**fields):
    """Helper: build a minimal entry dict with the given fields set."""
    return fields


# _rank_sort_key — sort key returning lower-is-better tuples for ranking
# accepted challenger evaluations.

def test_rank_sort_key_higher_mu_hat_sorts_first():
    better = _entry(mu_hat=0.9)
    worse = _entry(mu_hat=0.5)
    assert _rank_sort_key(better) < _rank_sort_key(worse)


def test_rank_sort_key_higher_lcb_breaks_mu_hat_tie():
    better = _entry(mu_hat=0.5, lcb=0.4)
    worse = _entry(mu_hat=0.5, lcb=0.1)
    assert _rank_sort_key(better) < _rank_sort_key(worse)


def test_rank_sort_key_lower_loss_breaks_lcb_tie():
    better = _entry(mu_hat=0.5, lcb=0.4, avg_challenger_loss=2.0)
    worse = _entry(mu_hat=0.5, lcb=0.4, avg_challenger_loss=3.0)
    assert _rank_sort_key(better) < _rank_sort_key(worse)


def test_rank_sort_key_earlier_timestamp_breaks_loss_tie():
    better = _entry(mu_hat=0.5, lcb=0.4, avg_challenger_loss=2.0, timestamp="2026-01-01T00:00:00")
    worse = _entry(mu_hat=0.5, lcb=0.4, avg_challenger_loss=2.0, timestamp="2026-02-01T00:00:00")
    assert _rank_sort_key(better) < _rank_sort_key(worse)


def test_rank_sort_key_missing_fields_sort_last():
    populated = _entry(mu_hat=0.5, lcb=0.4, avg_challenger_loss=2.0, timestamp="2026-01-01T00:00:00")
    empty = _entry()
    assert _rank_sort_key(populated) < _rank_sort_key(empty)


def test_rank_sort_key_none_fields_sort_last():
    populated = _entry(mu_hat=0.5, lcb=0.4, avg_challenger_loss=2.0, timestamp="2026-01-01T00:00:00")
    nones = _entry(mu_hat=None, lcb=None, avg_challenger_loss=None, timestamp=None)
    assert _rank_sort_key(populated) < _rank_sort_key(nones)


def test_rank_sort_key_used_with_sorted_returns_best_first():
    entries = [
        _entry(mu_hat=0.3),
        _entry(mu_hat=0.9),
        _entry(mu_hat=0.6),
    ]
    ranked = sorted(entries, key=_rank_sort_key)
    assert [e["mu_hat"] for e in ranked] == [0.9, 0.6, 0.3]


# _age_seconds — convert ISO-timestamp string to age-in-seconds. Returns
# None for None input. Used to measure how long ago the validator last
# performed an audit / state flush.

def test_age_seconds_none_input_returns_none():
    assert _age_seconds(None) is None


def test_age_seconds_recent_timestamp_returns_small_positive():
    ts = (datetime.now(UTC) - timedelta(seconds=2)).isoformat()
    age = _age_seconds(ts)
    assert age is not None
    assert 1 < age < 10


def test_age_seconds_one_hour_old_returns_about_3600():
    ts = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    age = _age_seconds(ts)
    assert age is not None
    assert 3590 < age < 3610


def test_age_seconds_invalid_string_returns_none():
    assert _age_seconds("not-a-timestamp") is None


def test_age_seconds_empty_string_returns_none():
    assert _age_seconds("") is None


def test_age_seconds_future_timestamp_clamps_to_zero():
    # Spec was ambiguous; impl clamps future timestamps to 0.0 (max(0, ...)).
    ts = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    assert _age_seconds(ts) == 0.0


# _is_transient_eval_error — classify exceptions/error messages as
# retry-able (transient) vs permanent. Returns (is_transient, reason_str).
# Markers known from commit history: CancelledError (162345d),
# SSE-truncation (a1bc502), streamconsumed/closed/error (8b2e2bd).

def test_is_transient_returns_tuple_of_bool_and_str():
    is_transient, reason = _is_transient_eval_error(Exception("anything"))
    assert isinstance(is_transient, bool)
    assert isinstance(reason, str)


def test_is_transient_for_cancelled_error():
    is_transient, _reason = _is_transient_eval_error(asyncio.CancelledError())
    assert is_transient is True


def test_is_transient_for_streamconsumed_message():
    is_transient, _reason = _is_transient_eval_error(Exception("StreamConsumed"))
    assert is_transient is True


def test_is_transient_for_streamclosed_message():
    is_transient, _reason = _is_transient_eval_error(Exception("StreamClosed"))
    assert is_transient is True


def test_is_transient_for_streamerror_message():
    is_transient, _reason = _is_transient_eval_error(Exception("StreamError"))
    assert is_transient is True


def test_is_not_transient_for_value_error():
    is_transient, _reason = _is_transient_eval_error(ValueError("bad arg"))
    assert is_transient is False


def test_is_not_transient_for_assertion_error():
    is_transient, _reason = _is_transient_eval_error(AssertionError("nope"))
    assert is_transient is False


def test_is_transient_accepts_string_input():
    # Sig is Exception | str — passing a raw string should also classify.
    is_transient, _reason = _is_transient_eval_error("StreamConsumed")
    assert is_transient is True


def test_is_not_transient_for_unrelated_string():
    is_transient, _reason = _is_transient_eval_error("unrelated message")
    assert is_transient is False


# _safe_block — best-effort current-block reader. Returns 0 on any
# subtensor RPC failure so the dethrone path can still record a king
# transition without losing state. The fallback matters because losing
# the block number would otherwise abort the whole transition path.

class _FakeSubtensor:
    """Minimal stub: only the .block attribute matters to _safe_block."""
    def __init__(self, block):
        self._block = block

    @property
    def block(self):
        if isinstance(self._block, BaseException):
            raise self._block
        return self._block


def test_safe_block_returns_int_value_on_happy_path():
    assert _safe_block(_FakeSubtensor(12345)) == 12345


def test_safe_block_returns_zero_on_rpc_error():
    # Any exception from accessing .block must collapse to 0 so the
    # dethrone path keeps moving — see docstring rationale.
    assert _safe_block(_FakeSubtensor(RuntimeError("rpc down"))) == 0


def test_safe_block_returns_zero_on_attribute_error():
    class NoBlock:
        pass
    assert _safe_block(NoBlock()) == 0
