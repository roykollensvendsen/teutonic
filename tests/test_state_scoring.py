"""Tests for State scoring/ranking methods.

Score-window lifecycle: accepted verdicts are recorded into
score_window["accepted_by_hotkey"], ranked into score_window["topk"]
on demand, and the window resets via reset_score_window with a
monotonically-incrementing window_id.
"""
from validator import TOPK_WEIGHTS, State


def _verdict(challenge_id="ch-1", **fields):
    """Build a verdict dict with sensible defaults for accepted-path tests."""
    base = {
        "challenge_id": challenge_id,
        "mu_hat": 0.5,
        "lcb": 0.4,
        "avg_king_loss": 2.0,
        "avg_challenger_loss": 1.5,
        "wall_time_s": 30.0,
        "timestamp": "2026-01-01T00:00:00",
        "challenger_revision": "abc123",
    }
    base.update(fields)
    return base


# ---------------------------------------------------------------------
# record_accepted_result — write entry into score_window's
# accepted_by_hotkey dict, keyed by hotkey.

def test_record_accepted_result_adds_entry_keyed_by_hotkey(r2_mock):
    s = State(r2_mock)
    s.uid_map = {"hk1": 7}
    s.record_accepted_result(_verdict(), "user/repo", "hk1", block=100)
    accepted = s.score_window["accepted_by_hotkey"]
    assert "hk1" in accepted


def test_record_accepted_result_propagates_mu_hat(r2_mock):
    s = State(r2_mock)
    s.uid_map = {"hk1": 7}
    s.record_accepted_result(_verdict(mu_hat=0.85), "user/repo", "hk1")
    entry = s.score_window["accepted_by_hotkey"]["hk1"]
    assert entry["mu_hat"] == 0.85


def test_record_accepted_result_overwrites_previous_entry_for_same_hotkey(r2_mock):
    s = State(r2_mock)
    s.uid_map = {"hk1": 7}
    s.record_accepted_result(_verdict(mu_hat=0.3), "user/repo", "hk1")
    s.record_accepted_result(_verdict(mu_hat=0.9), "user/repo", "hk1")
    accepted = s.score_window["accepted_by_hotkey"]
    assert len(accepted) == 1
    assert accepted["hk1"]["mu_hat"] == 0.9


# ---------------------------------------------------------------------
# recompute_topk — sort accepted_by_hotkey values by _rank_sort_key,
# truncate to len(TOPK_WEIGHTS), store in score_window["topk"], return it.

def test_recompute_topk_returns_empty_for_empty_window(r2_mock):
    s = State(r2_mock)
    assert s.recompute_topk() == []


def test_recompute_topk_truncates_to_topk_weights_length(r2_mock):
    s = State(r2_mock)
    s.score_window["accepted_by_hotkey"] = {
        f"hk{i}": {"hotkey": f"hk{i}", "mu_hat": 0.1 * i}
        for i in range(10)  # 10 entries
    }
    topk = s.recompute_topk()
    assert len(topk) == len(TOPK_WEIGHTS)


def test_recompute_topk_orders_by_rank_sort_key(r2_mock):
    s = State(r2_mock)
    s.score_window["accepted_by_hotkey"] = {
        "low": {"hotkey": "low", "mu_hat": 0.2},
        "high": {"hotkey": "high", "mu_hat": 0.9},
        "mid": {"hotkey": "mid", "mu_hat": 0.5},
    }
    topk = s.recompute_topk()
    assert [e["hotkey"] for e in topk] == ["high", "mid", "low"]


def test_recompute_topk_persists_to_score_window(r2_mock):
    s = State(r2_mock)
    s.score_window["accepted_by_hotkey"] = {
        "hk": {"hotkey": "hk", "mu_hat": 0.5},
    }
    s.recompute_topk()
    assert s.score_window["topk"] == [{"hotkey": "hk", "mu_hat": 0.5}]


# ---------------------------------------------------------------------
# note_weight_set — record metadata about a chain weight-set call.

def test_note_weight_set_stores_in_score_window(r2_mock):
    s = State(r2_mock)
    s.note_weight_set(block=42, ranked_hotkeys=["A", "B"],
                      ranked_weights=[0.6, 0.4], reason="periodic")
    last = s.score_window["last_weight_set"]
    assert last["block"] == 42
    assert last["ranked_hotkeys"] == ["A", "B"]
    assert last["ranked_weights"] == [0.6, 0.4]
    assert last["reason"] == "periodic"


def test_note_weight_set_includes_timestamp(r2_mock):
    s = State(r2_mock)
    s.note_weight_set(block=1, ranked_hotkeys=[], ranked_weights=[], reason="r")
    assert "timestamp" in s.score_window["last_weight_set"]


# ---------------------------------------------------------------------
# reset_score_window — start a fresh window, increment window_id,
# preserve last_weight_set, clear evaluated_repos.

def test_reset_score_window_increments_window_id(r2_mock):
    s = State(r2_mock)
    # Default window_id starts at "window-0000".
    s.reset_score_window(current_block=100)
    assert s.score_window["window_id"] == "window-0001"


def test_reset_score_window_clears_accepted_by_hotkey(r2_mock):
    s = State(r2_mock)
    s.score_window["accepted_by_hotkey"] = {"hk": {"hotkey": "hk", "mu_hat": 0.5}}
    s.reset_score_window(current_block=200)
    assert s.score_window["accepted_by_hotkey"] == {}


def test_reset_score_window_clears_topk(r2_mock):
    s = State(r2_mock)
    s.score_window["topk"] = [{"hotkey": "hk", "mu_hat": 0.5}]
    s.reset_score_window(current_block=200)
    assert s.score_window["topk"] == []


def test_reset_score_window_preserves_last_weight_set(r2_mock):
    s = State(r2_mock)
    prev_weight_set = {"block": 99, "reason": "r"}
    s.score_window["last_weight_set"] = prev_weight_set
    s.reset_score_window(current_block=200)
    assert s.score_window["last_weight_set"] == prev_weight_set


def test_reset_score_window_clears_evaluated_repos(r2_mock):
    s = State(r2_mock)
    s.evaluated_repos = {"user/a", "user/b"}
    s.reset_score_window(current_block=200)
    assert s.evaluated_repos == set()


def test_reset_score_window_records_started_block(r2_mock):
    s = State(r2_mock)
    s.reset_score_window(current_block=12345)
    assert s.score_window["started_block"] == 12345


# ---------------------------------------------------------------------
# record_verdict — build verdict entry from verdict + hotkey, write
# somewhere durable. Probe-rejected verdicts may lack some fields.

def test_record_verdict_handles_partial_probe_rejected_verdict(r2_mock):
    # Probe-rejected verdicts lack avg_*_loss / wall_time_s / timestamp.
    # Per docstring: "Default everything so a partial verdict still
    # records cleanly".
    s = State(r2_mock)
    partial = {"challenge_id": "ch-1", "ok": False, "reason": "probe_failed"}
    # Should not raise.
    s.record_verdict(partial, "user/repo", "hk1")


def test_record_verdict_handles_full_verdict(r2_mock):
    s = State(r2_mock)
    s.record_verdict(_verdict(), "user/repo", "hk1")
