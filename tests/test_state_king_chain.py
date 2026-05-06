"""Tests for State king-chain mutation methods.

State.set_king links each new king to the previous via `previous_king`,
forming a chain capped at KING_CHAIN_DEPTH (= 5) for the rolling
emission window. State._reconcile_chain_from_history rebuilds the
top of that chain from history.jsonl on startup, recovering from
crashed dethrones.
"""
from validator import KING_CHAIN_DEPTH, State

# ---------------------------------------------------------------------
# set_king — install new king, link to previous, cap chain depth.

def test_set_king_initial_seed_has_no_previous_king(r2_mock):
    s = State(r2_mock)
    s.set_king("hk1", "user/repo", "hash-x", block=100, challenge_id="seed")
    # Initial king has no prior reign to chain to.
    assert s.king.get("previous_king") in (None, {})
    assert s.king["hotkey"] == "hk1"


def test_set_king_records_basic_fields(r2_mock):
    s = State(r2_mock)
    s.set_king("hk1", "user/repo", "hash-x", block=100, challenge_id="ch-1")
    assert s.king["hotkey"] == "hk1"
    assert s.king["hf_repo"] == "user/repo"
    assert s.king["king_hash"] == "hash-x"


def test_set_king_links_previous_on_subsequent_call(r2_mock):
    s = State(r2_mock)
    s.set_king("A", "user/A", "hashA", block=100, challenge_id="seed")
    s.set_king("B", "user/B", "hashB", block=200, challenge_id="ch-1")
    # New king's previous_king should be the seed king (A).
    assert s.king["hotkey"] == "B"
    assert s.king["previous_king"]["hotkey"] == "A"


def test_set_king_clears_failed_repos(r2_mock):
    s = State(r2_mock)
    s.failed_repos = {"user/old1", "user/old2"}
    s.set_king("hk", "user/r", "hash", block=1, challenge_id="ch-1")
    assert s.failed_repos == set()


def test_set_king_clears_evaluated_repos(r2_mock):
    s = State(r2_mock)
    s.evaluated_repos = {"user/eval1"}
    s.set_king("hk", "user/r", "hash", block=1, challenge_id="ch-1")
    assert s.evaluated_repos == set()


def test_set_king_chain_capped_at_king_chain_depth(r2_mock):
    s = State(r2_mock)
    # Add KING_CHAIN_DEPTH + 2 kings to overflow.
    for i in range(KING_CHAIN_DEPTH + 2):
        s.set_king(f"hk{i}", f"user/r{i}", f"hash{i}", block=i,
                   challenge_id=("seed" if i == 0 else f"ch-{i}"))

    # Walk the chain and count; it should be at most KING_CHAIN_DEPTH.
    chain_len = 0
    node = s.king
    while node:
        chain_len += 1
        node = node.get("previous_king")
    assert chain_len <= KING_CHAIN_DEPTH


def test_set_king_consecutive_same_hotkey_keeps_both_entries(r2_mock):
    # Per docstring: per-reign stacking is intentional. Same hotkey
    # winning twice gets two slots in the chain, not deduped.
    s = State(r2_mock)
    s.set_king("A", "user/A1", "hash1", block=100, challenge_id="seed")
    s.set_king("A", "user/A2", "hash2", block=200, challenge_id="ch-1")
    assert s.king["hotkey"] == "A"
    assert s.king["previous_king"]["hotkey"] == "A"


def test_set_king_seed_reign_number_does_not_increment(r2_mock):
    # Per leaked impl: reign_number increments by 0 if challenge_id="seed",
    # else by 1.
    s = State(r2_mock)
    s.set_king("A", "user/r", "hash", block=1, challenge_id="seed")
    s.set_king("A", "user/r2", "hash2", block=2, challenge_id="seed")
    # Both seed calls — reign_number should not have grown beyond 0.
    assert s.king.get("reign_number", 0) == 0


def test_set_king_increments_reign_for_dethrone(r2_mock):
    s = State(r2_mock)
    s.set_king("A", "user/A", "hashA", block=1, challenge_id="seed")
    # reign_number after seed is 0
    s.set_king("B", "user/B", "hashB", block=2, challenge_id="ch-1")
    # First dethrone: reign_number goes from 0 to 1.
    assert s.king["reign_number"] == 1


# ---------------------------------------------------------------------
# _reconcile_chain_from_history — early-return when no king or no history.

def test_reconcile_no_king_is_noop(r2_mock):
    s = State(r2_mock)
    # No king set — should return without crashing.
    s._reconcile_chain_from_history()
    assert s.king == {}


def test_reconcile_no_history_is_noop(r2_mock):
    s = State(r2_mock)
    s.set_king("A", "user/A", "hashA", block=1, challenge_id="seed")
    pre_state = dict(s.king)
    s._reconcile_chain_from_history()
    # With no history, the chain stays exactly as it was.
    assert s.king == pre_state


def test_reconcile_is_idempotent(r2_mock):
    # Calling twice should produce the same result as calling once.
    s = State(r2_mock)
    s.set_king("A", "user/A", "hashA", block=1, challenge_id="seed")
    s.history = [{"accepted": True, "challenge_id": "ch-1", "hotkey": "B",
                  "challenger_repo": "user/B", "block": 2}]
    s._reconcile_chain_from_history()
    after_first = dict(s.king)
    s._reconcile_chain_from_history()
    after_second = dict(s.king)
    assert after_first == after_second
