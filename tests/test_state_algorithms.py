"""Tests for State methods with substantive algorithm logic.

These methods only read internal state — no external IO — so they're
testable with the same dict-backed r2_mock used in PR 4 + manual
State construction.

Coverage targets:
- State.recent_king_chain(depth): walk king + previous_king chain
- State.topk_for_weight_set(): rank score_window's accepted verdicts
"""
import pytest

from validator import State


@pytest.fixture
def r2_mock(mocker):
    """In-memory dict-backed mock of validator.R2."""
    storage = {}
    r2 = mocker.MagicMock()
    r2.get.side_effect = lambda key: storage.get(key)
    r2.put.side_effect = lambda key, data: storage.update({key: data})
    return r2


def _king(hotkey, previous=None, **extra):
    """Build a king dict; optionally chain via previous_king."""
    king = {"hotkey": hotkey}
    king.update(extra)
    if previous is not None:
        king["previous_king"] = previous
    return king


# ---------------------------------------------------------------------
# recent_king_chain — walks king.previous_king up to `depth` layers,
# preserves repeats, returns [current, prev1, prev2, ...].

def test_recent_king_chain_empty_king_returns_empty(r2_mock):
    s = State(r2_mock)
    # __init__ sets s.king = {} — falsy
    assert s.recent_king_chain(5) == []


def test_recent_king_chain_depth_zero_returns_empty(r2_mock):
    s = State(r2_mock)
    s.king = _king("A")
    assert s.recent_king_chain(0) == []


def test_recent_king_chain_single_king_returns_one_entry(r2_mock):
    s = State(r2_mock)
    s.king = _king("A")  # no previous_king field
    chain = s.recent_king_chain(5)
    assert len(chain) == 1
    assert chain[0]["hotkey"] == "A"


def test_recent_king_chain_walks_previous_links_in_order(r2_mock):
    s = State(r2_mock)
    s.king = _king("C", previous=_king("B", previous=_king("A")))
    chain = s.recent_king_chain(5)
    assert [k["hotkey"] for k in chain] == ["C", "B", "A"]


def test_recent_king_chain_preserves_repeats(r2_mock):
    # Per docstring: "Repeats are intentional".
    s = State(r2_mock)
    s.king = _king("A", previous=_king("A", previous=_king("A")))
    chain = s.recent_king_chain(5)
    assert [k["hotkey"] for k in chain] == ["A", "A", "A"]


def test_recent_king_chain_respects_depth_limit(r2_mock):
    s = State(r2_mock)
    s.king = _king("E",
                   previous=_king("D",
                                  previous=_king("C",
                                                 previous=_king("B",
                                                                previous=_king("A")))))
    chain = s.recent_king_chain(3)
    assert [k["hotkey"] for k in chain] == ["E", "D", "C"]


def test_recent_king_chain_stops_at_chain_end_when_depth_exceeds(r2_mock):
    s = State(r2_mock)
    s.king = _king("B", previous=_king("A"))
    chain = s.recent_king_chain(99)
    assert [k["hotkey"] for k in chain] == ["B", "A"]


def test_recent_king_chain_with_none_previous_terminates(r2_mock):
    # Some payloads may set previous_king explicitly to None.
    s = State(r2_mock)
    s.king = _king("A")
    s.king["previous_king"] = None
    chain = s.recent_king_chain(5)
    assert [k["hotkey"] for k in chain] == ["A"]


# ---------------------------------------------------------------------
# topk_for_weight_set — read score_window's accepted_by_hotkey,
# rank by _rank_sort_key, dedupe by hotkey (first-seen wins per
# sort order), filter to entries whose hotkey is in self.uid_map.

def _accepted(hotkey, mu_hat):
    """Build a minimal accepted-verdict entry."""
    return {"hotkey": hotkey, "mu_hat": mu_hat}


def test_topk_empty_score_window_returns_empty(r2_mock):
    s = State(r2_mock)
    # __init__ already sets score_window with empty accepted_by_hotkey.
    assert s.topk_for_weight_set() == []


def test_topk_filters_entries_not_in_uid_map(r2_mock):
    s = State(r2_mock)
    s.score_window["accepted_by_hotkey"] = {
        "A": _accepted("A", 0.5),
        "B": _accepted("B", 0.7),
    }
    s.uid_map = {"A": 0}  # B is not registered
    ranked = s.topk_for_weight_set()
    assert [e["hotkey"] for e in ranked] == ["A"]


def test_topk_orders_by_rank_sort_key(r2_mock):
    # Higher mu_hat → lower sort key → comes first per _rank_sort_key.
    s = State(r2_mock)
    s.score_window["accepted_by_hotkey"] = {
        "A": _accepted("A", 0.3),
        "B": _accepted("B", 0.9),
        "C": _accepted("C", 0.6),
    }
    s.uid_map = {"A": 0, "B": 1, "C": 2}
    ranked = s.topk_for_weight_set()
    assert [e["hotkey"] for e in ranked] == ["B", "C", "A"]


def test_topk_dedupes_by_hotkey(r2_mock):
    # If two entries have the same hotkey, only the first (best by
    # sort-order) should appear in output.
    # Note: dict keys must differ to allow duplicate-hotkey entries.
    s = State(r2_mock)
    s.score_window["accepted_by_hotkey"] = {
        "key1": _accepted("A", 0.9),  # better
        "key2": _accepted("A", 0.3),  # worse — same hotkey
    }
    s.uid_map = {"A": 0}
    ranked = s.topk_for_weight_set()
    hotkeys = [e["hotkey"] for e in ranked]
    assert hotkeys == ["A"]
    assert ranked[0]["mu_hat"] == 0.9


def test_topk_skips_entries_with_missing_or_falsy_hotkey(r2_mock):
    s = State(r2_mock)
    s.score_window["accepted_by_hotkey"] = {
        "k1": {"hotkey": "", "mu_hat": 0.9},  # empty hotkey
        "k2": {"mu_hat": 0.5},  # missing hotkey
        "k3": _accepted("A", 0.8),
    }
    s.uid_map = {"A": 0, "": 1}  # even empty key in uid_map
    ranked = s.topk_for_weight_set()
    assert [e["hotkey"] for e in ranked] == ["A"]


def test_topk_returns_full_entry_dict_not_just_hotkey(r2_mock):
    # The returned items are full entry dicts (not stripped to just
    # the hotkey) so callers can read mu_hat / lcb / timestamp etc.
    s = State(r2_mock)
    s.score_window["accepted_by_hotkey"] = {
        "k1": _accepted("A", 0.5),
    }
    s.uid_map = {"A": 0}
    ranked = s.topk_for_weight_set()
    assert ranked[0] == _accepted("A", 0.5)
