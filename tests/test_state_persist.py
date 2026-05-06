"""Round-trip + schema-evolution tests for validator.State.

State persists itself to R2 via `r2.put(key, data)` and loads via
`r2.get(key)` (auto-JSON serialization). These tests use a dict-backed
mock R2 so save/load happens entirely in memory.

Assertions focus on per-field round-trip identity rather than full-dict
equality — keeps the tests resilient to upstream PRs that add new
fields to the State schema.
"""
from validator import State


def test_fresh_state_load_with_empty_storage_does_not_crash(r2_mock):
    s = State(r2_mock)
    s.load()  # storage is empty — load should populate defaults gracefully

    # Default invariants from State.__init__:
    assert s.king == {}
    assert s.queue == []
    assert s.counter == 0


def test_round_trip_king(r2_mock):
    s = State(r2_mock)
    s.king = {"hotkey": "abc", "hf_repo": "miner/model", "block": 42}
    s.flush()

    s2 = State(r2_mock)
    s2.load()
    assert s2.king == {"hotkey": "abc", "hf_repo": "miner/model", "block": 42}


def test_round_trip_counter(r2_mock):
    s = State(r2_mock)
    s.counter = 7
    s.flush()

    s2 = State(r2_mock)
    s2.load()
    assert s2.counter == 7


def test_round_trip_last_weight_block(r2_mock):
    s = State(r2_mock)
    s.last_weight_block = 1234567
    s.flush()

    s2 = State(r2_mock)
    s2.load()
    assert s2.last_weight_block == 1234567


def test_round_trip_seen_set_serializes_as_collection(r2_mock):
    s = State(r2_mock)
    s.seen = {"hotkey1", "hotkey2", "hotkey3"}
    s.flush()

    s2 = State(r2_mock)
    s2.load()
    # seen is a set — ordering may differ across save/load (JSON
    # collections), but membership must round-trip.
    assert s2.seen == {"hotkey1", "hotkey2", "hotkey3"}


def test_failed_and_evaluated_repos_are_intentionally_not_persisted(r2_mock):
    # The persist key is "state/seen_hotkeys.json" — hotkeys, singular
    # concept. failed_repos / evaluated_repos hold *repo names* (a
    # different category) and are intentionally per-process state used
    # for in-memory dedup during a single validator run; they reset on
    # restart so a previously-failed repo gets a fresh chance.
    s = State(r2_mock)
    s.failed_repos = {"miner_a/repo", "miner_b/repo"}
    s.evaluated_repos = {"miner_c/repo"}
    s.flush()

    s2 = State(r2_mock)
    s2.load()
    assert s2.failed_repos == set()
    assert s2.evaluated_repos == set()


def test_load_tolerates_missing_optional_keys(r2_mock):
    """Schema evolution: an older state payload may lack newer keys.
    State.load should not crash — missing parts default to empty."""
    # Only the king is persisted; queue/seen/validator_state/etc. are absent.
    r2_mock._storage["king/current.json"] = {"hotkey": "x", "hf_repo": "r/m"}

    s = State(r2_mock)
    s.load()
    assert s.king == {"hotkey": "x", "hf_repo": "r/m"}
    # Missing-key fields default to their __init__ values.
    assert s.queue == []
    assert s.seen == set()
