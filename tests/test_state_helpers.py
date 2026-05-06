"""Tests for State pure getters + small helpers.

These methods are individually small but together form the bookkeeping
substrate that the heavier State algorithms rely on. All testable with
the existing r2_mock pattern + manual State construction.
"""
from validator import State

# ---------------------------------------------------------------------
# next_id — increments counter, returns formatted "eval-NNNN" string.

def test_next_id_returns_eval_dash_format(r2_mock):
    s = State(r2_mock)
    assert s.next_id() == "eval-0001"


def test_next_id_strictly_increasing(r2_mock):
    s = State(r2_mock)
    ids = [s.next_id() for _ in range(5)]
    assert ids == ["eval-0001", "eval-0002", "eval-0003", "eval-0004", "eval-0005"]


def test_next_id_zero_pads_to_four_digits(r2_mock):
    s = State(r2_mock)
    s.counter = 9  # next will be 10 → "eval-0010"
    assert s.next_id() == "eval-0010"


def test_next_id_persists_counter_mutation(r2_mock):
    s = State(r2_mock)
    s.next_id()
    s.next_id()
    assert s.counter == 2


# ---------------------------------------------------------------------
# event(data) — adds timestamp default, appends record to
# state/history.jsonl via r2.append_jsonl.

def test_event_calls_append_jsonl(r2_mock):
    s = State(r2_mock)
    s.event({"kind": "test"})
    assert "state/history.jsonl" in r2_mock._appended
    assert len(r2_mock._appended["state/history.jsonl"]) == 1


def test_event_adds_timestamp_when_missing(r2_mock):
    s = State(r2_mock)
    s.event({"kind": "test"})
    record = r2_mock._appended["state/history.jsonl"][0]
    assert "timestamp" in record


def test_event_preserves_explicit_timestamp(r2_mock):
    s = State(r2_mock)
    s.event({"kind": "test", "timestamp": "2026-01-01T00:00:00"})
    record = r2_mock._appended["state/history.jsonl"][0]
    assert record["timestamp"] == "2026-01-01T00:00:00"


# ---------------------------------------------------------------------
# coldkey_for(hotkey) — lookup hotkey -> coldkey (None if unknown).

def test_coldkey_for_returns_stored_value(r2_mock):
    s = State(r2_mock)
    s.hotkey_coldkey = {"hk1": "ck1"}
    assert s.coldkey_for("hk1") == "ck1"


def test_coldkey_for_unknown_hotkey_returns_none(r2_mock):
    s = State(r2_mock)
    assert s.coldkey_for("never-seen") is None


# ---------------------------------------------------------------------
# expected_coldkey_prefix(hotkey) — first N chars of coldkey ss58.

def test_expected_coldkey_prefix_for_known_hotkey(r2_mock):
    s = State(r2_mock)
    coldkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    s.hotkey_coldkey = {"hk1": coldkey}
    prefix = s.expected_coldkey_prefix("hk1")
    assert isinstance(prefix, str)
    # Must be a prefix of the actual coldkey.
    assert coldkey.startswith(prefix)


def test_expected_coldkey_prefix_unknown_hotkey_returns_none(r2_mock):
    s = State(r2_mock)
    assert s.expected_coldkey_prefix("unknown") is None


# ---------------------------------------------------------------------
# remember_revision(hotkey, repo, revision) — store latest revision.

def test_remember_revision_stores_under_hotkey(r2_mock):
    s = State(r2_mock)
    s.remember_revision("hk1", "user/repo", "abc123")
    # The data should be retrievable somehow — either via best_known_revision
    # or directly from known_revisions.
    assert "hk1" in s.known_revisions


def test_remember_revision_ignores_falsy_hotkey(r2_mock):
    s = State(r2_mock)
    s.remember_revision("", "user/repo", "abc123")
    s.remember_revision(None, "user/repo", "abc123")
    assert s.known_revisions == {}


# ---------------------------------------------------------------------
# best_known_revision(hotkey, repo="") — lookup recorded revision.

def test_best_known_revision_returns_stored(r2_mock):
    s = State(r2_mock)
    s.remember_revision("hk1", "user/repo", "abc123")
    assert s.best_known_revision("hk1", "user/repo") == "abc123"


def test_best_known_revision_unknown_returns_empty_string(r2_mock):
    # Spec found during testing: returns "" not None for unknown hotkey.
    # Defensible — caller can use truthy checks; avoids None-handling churn.
    s = State(r2_mock)
    assert s.best_known_revision("never-seen") == ""


def test_best_known_revision_returns_empty_when_repo_mismatches(r2_mock):
    # If a different repo is requested than the one we recorded, treat
    # the known revision as not applicable.
    s = State(r2_mock)
    s.remember_revision("hk1", "user/repo-A", "abc123")
    assert s.best_known_revision("hk1", "user/repo-B") == ""


def test_best_known_revision_no_repo_arg_returns_stored_revision(r2_mock):
    # If repo arg is empty string (default), no repo-match check
    # applies — return the stored revision regardless.
    s = State(r2_mock)
    s.remember_revision("hk1", "user/repo-A", "abc123")
    assert s.best_known_revision("hk1") == "abc123"


# ---------------------------------------------------------------------
# _best_of(entries) — return highest-ranked entry or None.

def test_best_of_empty_returns_none(r2_mock):
    s = State(r2_mock)
    assert s._best_of([]) is None


def test_best_of_picks_highest_mu_hat(r2_mock):
    s = State(r2_mock)
    entries = [
        {"hotkey": "A", "mu_hat": 0.3},
        {"hotkey": "B", "mu_hat": 0.9},
        {"hotkey": "C", "mu_hat": 0.6},
    ]
    best = s._best_of(entries)
    assert best["hotkey"] == "B"


# ---------------------------------------------------------------------
# _truncate_king_chain(node, max_depth) — staticmethod that returns a
# copy with previous_king chain truncated at max_depth layers.

def test_truncate_king_chain_at_max_depth():
    chain = {
        "hotkey": "E",
        "previous_king": {
            "hotkey": "D",
            "previous_king": {
                "hotkey": "C",
                "previous_king": {
                    "hotkey": "B",
                    "previous_king": {"hotkey": "A"},
                },
            },
        },
    }
    truncated = State._truncate_king_chain(chain, max_depth=3)
    # Depth 1 (E) -> 2 (D) -> 3 (C). Beyond should be truncated.
    assert truncated["hotkey"] == "E"
    assert truncated["previous_king"]["hotkey"] == "D"
    assert truncated["previous_king"]["previous_king"]["hotkey"] == "C"
    # The chain ends at C — no further previous_king.
    assert truncated["previous_king"]["previous_king"].get("previous_king") in (None, {})


def test_truncate_king_chain_does_not_mutate_input():
    chain = {
        "hotkey": "B",
        "previous_king": {"hotkey": "A"},
    }
    State._truncate_king_chain(chain, max_depth=1)
    # Input must still have the full original chain intact.
    assert chain["previous_king"]["hotkey"] == "A"


def test_truncate_king_chain_handles_short_chain_below_max_depth():
    chain = {"hotkey": "A"}
    result = State._truncate_king_chain(chain, max_depth=10)
    assert result["hotkey"] == "A"


# ---------------------------------------------------------------------
# _with_fresh_uid(entry) — copy entry, refresh uid from current uid_map.

def test_with_fresh_uid_updates_uid_from_uid_map(r2_mock):
    s = State(r2_mock)
    s.uid_map = {"hk1": 42}
    entry = {"hotkey": "hk1", "uid": 999}  # stale uid
    fresh = s._with_fresh_uid(entry)
    assert fresh["uid"] == 42


def test_with_fresh_uid_does_not_mutate_input(r2_mock):
    s = State(r2_mock)
    s.uid_map = {"hk1": 42}
    entry = {"hotkey": "hk1", "uid": 999}
    s._with_fresh_uid(entry)
    assert entry["uid"] == 999  # original unchanged


def test_with_fresh_uid_refreshes_coldkey_from_metagraph(r2_mock):
    # The dashboard hotkey -> coldkey link must point at the *current*
    # coldkey for a hotkey rather than whatever was on file when the
    # duel was recorded — see docstring on validator.py:1414.
    s = State(r2_mock)
    s.uid_map = {"hk1": 42}
    s.hotkey_coldkey = {"hk1": "ck_current"}
    entry = {"hotkey": "hk1", "uid": 0, "coldkey": "ck_stale"}
    fresh = s._with_fresh_uid(entry)
    assert fresh["coldkey"] == "ck_current"


def test_with_fresh_uid_falls_back_to_persisted_coldkey_when_deregistered(r2_mock):
    # The docstring says: "fall back to the persisted value only if
    # the hotkey has been deregistered out of the metagraph". Pin
    # that contract — if hotkey_coldkey doesn't have hk1, return the
    # entry's stored coldkey untouched.
    s = State(r2_mock)
    s.uid_map = {"hk1": 42}
    s.hotkey_coldkey = {}  # hk1 has been deregistered
    entry = {"hotkey": "hk1", "uid": 0, "coldkey": "ck_archived"}
    fresh = s._with_fresh_uid(entry)
    assert fresh["coldkey"] == "ck_archived"


def test_with_fresh_uid_returns_entry_unchanged_when_not_dict(r2_mock):
    # Defensive branch: if entry isn't a dict (e.g. None, str, int),
    # _with_fresh_uid returns it as-is rather than crashing on .get().
    s = State(r2_mock)
    assert s._with_fresh_uid(None) is None
    assert s._with_fresh_uid("not-a-dict") == "not-a-dict"


def test_with_fresh_uid_returns_entry_unchanged_when_no_hotkey(r2_mock):
    # Same branch from a different angle: dict without "hotkey" key.
    s = State(r2_mock)
    entry = {"challenge_id": "x", "uid": 5}  # no hotkey field
    fresh = s._with_fresh_uid(entry)
    assert fresh is entry  # identity, not a copy
