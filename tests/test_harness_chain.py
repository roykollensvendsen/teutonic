"""Tests for tests._harness.chain.FakeChain.

Pin the contract documented in `chain.py`'s module + class docstrings:
the read API mirrors `bittensor.subtensor` for the surface the validator
consumes; the mutator API lets tests inject hotkeys, blocks, and reveals.

The chain is in-process, deterministic, and never raises spontaneously
— tests that need RPC failures monkeypatch the method directly.
"""
import pytest

from tests._harness.chain import FakeChain

# ---------------------------------------------------------------------
# block — current block, advances on demand.

def test_block_starts_at_constructor_value():
    assert FakeChain(block=12345).block == 12345


def test_block_default_starts_at_zero():
    assert FakeChain().block == 0


def test_advance_block_default_bumps_by_one():
    chain = FakeChain(block=10)
    chain.advance_block()
    assert chain.block == 11


def test_advance_block_with_n_bumps_by_n():
    chain = FakeChain(block=10)
    chain.advance_block(5)
    assert chain.block == 15


# ---------------------------------------------------------------------
# metagraph(netuid) — returns SimpleNamespace with hotkeys, emission, coldkeys.

def test_metagraph_empty_chain_has_empty_hotkey_list():
    meta = FakeChain().metagraph(netuid=3)
    assert meta.hotkeys == []
    assert meta.emission == []
    assert meta.coldkeys == []


def test_metagraph_hotkeys_indexed_by_uid():
    chain = FakeChain()
    chain.register("hk_alpha", uid=0)
    chain.register("hk_beta", uid=1)
    meta = chain.metagraph(netuid=3)
    assert meta.hotkeys == ["hk_alpha", "hk_beta"]


def test_metagraph_returns_a_copy_not_internal_list():
    # Validator code occasionally mutates lists pulled off metagraph.
    # The returned list must be independent so a test mutation doesn't
    # poison the chain's internal state.
    chain = FakeChain()
    chain.register("hk_alpha", uid=0)
    meta = chain.metagraph(netuid=3)
    meta.hotkeys.append("hk_injected_by_consumer")
    assert chain.metagraph(netuid=3).hotkeys == ["hk_alpha"]


def test_metagraph_emission_aligned_with_uid():
    chain = FakeChain()
    chain.register("hk_alpha", uid=0, emission=0.05)
    chain.register("hk_beta", uid=1, emission=0.1)
    meta = chain.metagraph(netuid=3)
    assert meta.emission == [0.05, 0.1]


def test_metagraph_coldkeys_aligned_with_uid():
    chain = FakeChain()
    chain.register("hk_alpha", uid=0, coldkey="ck_alpha")
    chain.register("hk_beta", uid=1, coldkey="ck_beta")
    meta = chain.metagraph(netuid=3)
    assert meta.coldkeys == ["ck_alpha", "ck_beta"]


def test_register_default_coldkey_is_none():
    chain = FakeChain()
    chain.register("hk_alpha", uid=0)
    meta = chain.metagraph(netuid=3)
    assert meta.coldkeys == [None]


# ---------------------------------------------------------------------
# Re-registering a uid replaces the hotkey at that slot — the
# deregister-and-re-register chain scenario the harness exists to drive.

def test_register_replaces_hotkey_at_existing_uid():
    chain = FakeChain()
    chain.register("hk_old", uid=0, coldkey="ck_old", emission=0.1)
    chain.register("hk_new", uid=0, coldkey="ck_new", emission=0.2)
    meta = chain.metagraph(netuid=3)
    assert meta.hotkeys == ["hk_new"]
    assert meta.coldkeys == ["ck_new"]
    assert meta.emission == [0.2]


# ---------------------------------------------------------------------
# Gap uids are filled with empty hotkeys so indexing-by-uid stays valid.
# The docstring promises this behaviour.

def test_register_at_non_contiguous_uid_pads_gaps():
    chain = FakeChain()
    chain.register("hk_at_three", uid=3, coldkey="ck", emission=0.5)
    meta = chain.metagraph(netuid=3)
    assert meta.hotkeys == ["", "", "", "hk_at_three"]
    assert meta.emission == [0.0, 0.0, 0.0, 0.5]
    assert meta.coldkeys == [None, None, None, "ck"]


# ---------------------------------------------------------------------
# set_emission — update a registered hotkey's emission.

def test_set_emission_updates_registered_hotkey():
    chain = FakeChain()
    chain.register("hk_alpha", uid=0, emission=0.0)
    chain.set_emission("hk_alpha", 0.42)
    meta = chain.metagraph(netuid=3)
    assert meta.emission == [0.42]


def test_set_emission_raises_for_unregistered_hotkey():
    # The docstring contract: "update a registered hotkey's emission".
    # A typo in a hotkey-name should surface loudly, not silently no-op
    # (otherwise tests pass with an invariant they didn't establish).
    chain = FakeChain()
    with pytest.raises(KeyError):
        chain.set_emission("hk_never_registered", 0.5)


# ---------------------------------------------------------------------
# commit_reveal + get_all_revealed_commitments.

def test_get_reveals_empty_for_fresh_chain():
    assert FakeChain().get_all_revealed_commitments(netuid=3) == {}


def test_commit_reveal_uses_current_block_by_default():
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_alpha", "king_hash:miner/repo:model_hash")
    reveals = chain.get_all_revealed_commitments(netuid=3)
    assert reveals == {
        "hk_alpha": [(100, "king_hash:miner/repo:model_hash")],
    }


def test_commit_reveal_with_explicit_block_overrides_default():
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_alpha", "payload_a", block=42)
    reveals = chain.get_all_revealed_commitments(netuid=3)
    assert reveals == {"hk_alpha": [(42, "payload_a")]}


def test_commit_reveal_appends_in_insertion_order():
    # Multiple reveals from the same hotkey are kept in the order the
    # test added them (scan_reveals picks the max-block entry; the
    # chain just records, doesn't sort).
    chain = FakeChain(block=10)
    chain.commit_reveal("hk_a", "first", block=10)
    chain.commit_reveal("hk_a", "second", block=20)
    chain.commit_reveal("hk_a", "third", block=15)
    reveals = chain.get_all_revealed_commitments(netuid=3)
    assert reveals == {
        "hk_a": [(10, "first"), (20, "second"), (15, "third")],
    }


def test_get_reveals_returns_a_copy_not_internal_dict():
    chain = FakeChain(block=10)
    chain.commit_reveal("hk_a", "x")
    reveals = chain.get_all_revealed_commitments(netuid=3)
    reveals["hk_b"] = [(99, "injected")]
    reveals["hk_a"].append((100, "injected_too"))
    fresh = chain.get_all_revealed_commitments(netuid=3)
    assert "hk_b" not in fresh
    assert fresh["hk_a"] == [(10, "x")]


# ---------------------------------------------------------------------
# get_subnet_hyperparameters — stubbed.

def test_get_subnet_hyperparameters_returns_simplenamespace():
    # Stubbed; tests that need specific fields can attach them per-test.
    hp = FakeChain().get_subnet_hyperparameters(netuid=3)
    assert hasattr(hp, "__dict__")
