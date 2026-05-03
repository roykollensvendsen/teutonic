"""Tests for State queue + error-path methods.

Covers:
- enqueue(reveal, defer_flush): add to queue, optionally flush
- requeue_front(entry, *, reason, ...): retry transient failures
- record_failure(entry, error_code, error_detail): note failure
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
    r2._storage = storage
    r2._appended = appended
    r2._dashboards = dashboards
    return r2


def _reveal(hotkey, repo, *, challenge_id="ch-1"):
    return {
        "hotkey": hotkey,
        "hf_repo": repo,
        "challenge_id": challenge_id,
    }


# ---------------------------------------------------------------------
# enqueue — adds reveal to queue; flushes by default; defer_flush skips.

def test_enqueue_appends_to_queue(r2_mock):
    s = State(r2_mock)
    s.enqueue(_reveal("hk1", "user/repo"))
    assert len(s.queue) == 1
    assert s.queue[0]["hotkey"] == "hk1"


def test_enqueue_defer_flush_skips_r2_state_put(r2_mock):
    # When deferring, the state/validator_state.json key should not be
    # written. Caller is expected to flush explicitly later.
    s = State(r2_mock)
    s.enqueue(_reveal("hk1", "user/repo"), defer_flush=True)
    assert "state/validator_state.json" not in r2_mock._storage


def test_enqueue_default_writes_state_to_r2(r2_mock):
    s = State(r2_mock)
    s.enqueue(_reveal("hk1", "user/repo"))
    # Default (no defer) flushes — state should be persisted.
    assert "state/validator_state.json" in r2_mock._storage


def test_enqueue_multiple_with_defer_does_not_flush(r2_mock):
    s = State(r2_mock)
    for i in range(5):
        s.enqueue(_reveal(f"hk{i}", f"user/repo{i}"), defer_flush=True)
    assert len(s.queue) == 5
    # Storage should not have validator_state since all enqueues deferred.
    assert "state/validator_state.json" not in r2_mock._storage


# ---------------------------------------------------------------------
# requeue_front — retry transient failures by inserting at queue front.

def test_requeue_front_inserts_at_front(r2_mock):
    s = State(r2_mock)
    # Pre-existing item in queue
    s.queue.append({"hotkey": "first", "hf_repo": "user/first", "challenge_id": "old"})
    entry = {"hotkey": "retry", "hf_repo": "user/retry", "challenge_id": "ch-x"}
    s.requeue_front(entry, reason="transient", error_code="net")
    # The retry should be at index 0, original at index 1.
    assert s.queue[0]["hotkey"] == "retry"
    assert s.queue[1]["hotkey"] == "first"


def test_requeue_front_increments_retry_count(r2_mock):
    s = State(r2_mock)
    entry = {"hotkey": "hk", "hf_repo": "user/r", "challenge_id": "ch"}
    s.requeue_front(entry, reason="transient")
    assert s.queue[0]["retry_count"] >= 1


def test_requeue_front_increments_existing_retry_count(r2_mock):
    s = State(r2_mock)
    entry = {"hotkey": "hk", "hf_repo": "user/r", "challenge_id": "ch", "retry_count": 2}
    s.requeue_front(entry, reason="transient")
    assert s.queue[0]["retry_count"] == 3


def test_requeue_front_skips_if_repo_already_pending(r2_mock):
    # Per docstring: "avoids duplicating the same repo if it's already
    # pending elsewhere in the queue".
    s = State(r2_mock)
    s.queue.append({"hotkey": "hk", "hf_repo": "user/dup", "challenge_id": "ch-1"})
    entry = {"hotkey": "hk", "hf_repo": "user/dup", "challenge_id": "ch-2"}
    s.requeue_front(entry, reason="transient")
    # Should NOT have added the duplicate — queue length unchanged.
    assert len(s.queue) == 1


# ---------------------------------------------------------------------
# record_failure — insert failure record at front of self.history (in-mem).

def test_record_failure_inserts_into_history(r2_mock):
    s = State(r2_mock)
    entry = {"hotkey": "hk", "hf_repo": "user/r", "challenge_id": "ch"}
    s.record_failure(entry, error_code="bad_config")
    assert len(s.history) == 1


def test_record_failure_marks_accepted_false(r2_mock):
    s = State(r2_mock)
    entry = {"hotkey": "hk", "hf_repo": "user/r", "challenge_id": "ch"}
    s.record_failure(entry, error_code="bad_config")
    assert s.history[0]["accepted"] is False


def test_record_failure_inserts_at_front(r2_mock):
    # Per leaked impl (self.history.insert(0, ...)), most-recent failures
    # appear first in history.
    s = State(r2_mock)
    s.record_failure({"hotkey": "old", "hf_repo": "user/old", "challenge_id": "1"},
                     error_code="e1")
    s.record_failure({"hotkey": "new", "hf_repo": "user/new", "challenge_id": "2"},
                     error_code="e2")
    # Newer failure first.
    assert s.history[0]["hotkey"] == "new"
    assert s.history[1]["hotkey"] == "old"


def test_record_failure_propagates_error_code_and_detail(r2_mock):
    s = State(r2_mock)
    entry = {"hotkey": "hk", "hf_repo": "user/r", "challenge_id": "ch"}
    s.record_failure(entry, error_code="net_timeout", error_detail="connection reset")
    rec = s.history[0]
    # Record should include both error_code and error_detail somehow.
    assert "net_timeout" in str(rec.values())
    assert "connection reset" in str(rec.values())
