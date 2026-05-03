"""Tests for validator.aggregate_chain_weights.

Per-reign aggregator for the rolling 5-king emission window.
Each chain entry earns 1/KING_CHAIN_DEPTH of total emission. Repeats
stack; entries not in uid_map are dropped. Returns
[(hotkey, weight)] in first-seen-newest-first order.

KING_CHAIN_DEPTH = 5, so slot_weight = 0.2.
"""
import pytest

from validator import KING_CHAIN_DEPTH, aggregate_chain_weights

SLOT = 1.0 / KING_CHAIN_DEPTH  # 0.2


def _entry(hotkey):
    return {"hotkey": hotkey}


def test_empty_chain_returns_empty_list():
    assert aggregate_chain_weights([], {"A": 1}) == []


def test_empty_uid_map_drops_everything():
    chain = [_entry("A"), _entry("B")]
    assert aggregate_chain_weights(chain, {}) == []


def test_single_registered_hotkey_gets_one_slot():
    chain = [_entry("A")]
    result = aggregate_chain_weights(chain, {"A": 0})
    assert result == [("A", SLOT)]


def test_distinct_hotkeys_each_get_one_slot():
    chain = [_entry("A"), _entry("B")]
    result = aggregate_chain_weights(chain, {"A": 0, "B": 1})
    assert result == [("A", SLOT), ("B", SLOT)]


def test_repeats_stack_for_same_hotkey():
    # A wins kingship twice in the last 5 reigns → 2 * SLOT
    chain = [_entry("A"), _entry("B"), _entry("A")]
    result = aggregate_chain_weights(chain, {"A": 0, "B": 1})
    assert result == [("A", 2 * SLOT), ("B", SLOT)]


def test_unregistered_hotkeys_are_dropped():
    chain = [_entry("A"), _entry("X"), _entry("B")]
    result = aggregate_chain_weights(chain, {"A": 0, "B": 1})  # X missing
    assert result == [("A", SLOT), ("B", SLOT)]


def test_order_is_first_seen_newest_first():
    # B appears first in the chain (newest), then A. Output preserves
    # this order — B comes before A.
    chain = [_entry("B"), _entry("A"), _entry("B")]
    result = aggregate_chain_weights(chain, {"A": 0, "B": 1})
    assert result == [("B", 2 * SLOT), ("A", SLOT)]


def test_full_chain_same_winner_gives_full_emission():
    # If one hotkey held kingship for all KING_CHAIN_DEPTH reigns,
    # they receive 100% of the slot weight (1.0 = full emission).
    chain = [_entry("A")] * KING_CHAIN_DEPTH
    result = aggregate_chain_weights(chain, {"A": 0})
    assert result == [("A", 1.0)]


def test_non_dict_entries_are_skipped():
    # Robustness: chain entries that aren't dicts (e.g., None or
    # malformed) should be silently skipped rather than crashing.
    chain = [_entry("A"), None, _entry("B"), "not-a-dict"]
    result = aggregate_chain_weights(chain, {"A": 0, "B": 1})
    assert result == [("A", SLOT), ("B", SLOT)]


def test_entries_missing_hotkey_field_are_skipped():
    chain = [_entry("A"), {"not_hotkey": "X"}, _entry("B")]
    result = aggregate_chain_weights(chain, {"A": 0, "B": 1})
    assert result == [("A", SLOT), ("B", SLOT)]


@pytest.mark.parametrize("count", [1, 2, 5, 10])
def test_n_repeats_yield_n_slot_weight(count):
    # Float precision: 10 * 0.2 ≈ 2.0 in IEEE 754 (off by ~1e-16),
    # so use approx equality.
    chain = [_entry("A")] * count
    result = aggregate_chain_weights(chain, {"A": 0})
    assert result[0][0] == "A"
    assert result[0][1] == pytest.approx(count * SLOT)
