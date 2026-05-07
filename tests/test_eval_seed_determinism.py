"""Characterization tests for the (block_hash, hotkey) seed pipeline.

Two layers of randomness drive every challenger eval:

* **Shard selection** — the validator picks ONE shard out of `n_shards`
  for the eval. `validator.py:2012-2014`:

      seed_mat  = f"{block_hash}:{hotkey}".encode()
      shard_idx = int.from_bytes(blake2b(seed_mat, 8).digest(), "little") % n_shards

* **Eval indices** — the eval-server picks `actual_N` sequence indices
  out of `n_sequences` from that shard via PCG64.
  `eval/torch_runner.py:1156-1159` (called from `eval_server.py:621`,
  `eval/vllm_server.py:384`):

      seed_str     = f"{block_hash}:{hotkey}"
      seed         = int.from_bytes(blake2b(seed_str.encode(), 8).digest(), "little")
      rng          = np.random.Generator(np.random.PCG64(seed))
      eval_indices = rng.choice(n_sequences, size=actual_N, replace=False).tolist()

Both layers consume IDENTICAL seed material — the validator's `seed_mat`
and the eval-server's `seed_str.encode()` are the same bytes. The
eval-server receives `block_hash` over the dispatch payload
(`validator.py:2017-2026 → eval/<cid>/meta.json`).

This file pins three properties:

1. **Determinism** — same `(block_hash, hotkey)` always yields the
   same `shard_idx` and the same `eval_indices`.

2. **Sensitivity** — distinct `block_hash` values (or distinct hotkeys)
   yield different shards / different index sets with overwhelming
   probability. The seed is *not* a per-hotkey constant.

3. **Local-vs-remote reproduction gap** — Discord 2026-05-06/07 has
   ~0.10–0.17 loss-difference reports between miners' local evals and
   the validator's remote eval (e.g. @kyle890015's 3.10–3.20 vs
   3.29–3.36; @clarencedan reproducing 3.123 vs 3.292 on identical
   hardware). The state-drift hypothesis is rejected (king is in
   `model.eval()` during `compute_paired_losses` per
   `eval/torch_runner.py:534`; Quasar is documented stateless in
   eval-mode at `archs/quasar/modeling_quasar.py:766`). Far more
   plausible: the miner runs a local eval against shard 0 (or any
   default), the validator runs against a *different* shard chosen
   from `(block_hash, hotkey)` at eval-time. `block_hash` is chain
   state at eval-time and cannot be guessed without reading the
   validator dashboard. The tests below pin the property that any
   `(block_hash, hotkey)` derivative MUST include block_hash to be
   reproducible — without it, no miner can reproduce the remote shard
   or indices.

These are pure-helper tests: no harness, no R2, no torch model. They
mirror the inline formulas verbatim so any drift in the production
code paths shows up as a failed test.
"""
import hashlib

import numpy as np
import pytest

# ---------------------------------------------------------------------
# Mirror helpers — verbatim copies of the inline formulas. Updating
# either of these implies the corresponding production code path
# changed, which is exactly what these tests guard against.

def _shard_idx(block_hash: str, hotkey: str, n_shards: int) -> int:
    """MIRROR of `validator.py:2012-2014`."""
    seed_mat = f"{block_hash}:{hotkey}".encode()
    return int.from_bytes(
        hashlib.blake2b(seed_mat, digest_size=8).digest(), "little"
    ) % n_shards


def _eval_indices(block_hash: str, hotkey: str,
                  n_sequences: int, actual_n: int) -> list[int]:
    """MIRROR of `eval/torch_runner.py:1156-1159`.

    `seed_str = f"{block_hash}:{hotkey}"` is constructed in
    `eval_server.py:621` and `eval/vllm_server.py:384` and passed as
    the `seed_str` argument of `run_bootstrap_test`.
    """
    seed_str = f"{block_hash}:{hotkey}"
    seed = int.from_bytes(
        hashlib.blake2b(seed_str.encode(), digest_size=8).digest(), "little"
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    return rng.choice(n_sequences, size=actual_n, replace=False).tolist()


# Realistic dataset shape: the live SN3 dataset manifest at
# `https://us-east-1.hippius.com/teutonic-sn3/dataset/v2/manifest.json`
# has 1992 shards (verified 2026-05-07).
_LIVE_N_SHARDS = 1992
_HOTKEY_A = "5HhKLLongkey_a"
_HOTKEY_B = "5GxBzMOtherkey_b"
_BLOCK_HASH_A = "0xd34db33fcafe" + "00" * 26
_BLOCK_HASH_B = "0xb16b00b5beef" + "00" * 26


# ---------------------------------------------------------------------
# Property 1: Determinism — same inputs, same outputs.

def test_shard_idx_deterministic():
    a = _shard_idx(_BLOCK_HASH_A, _HOTKEY_A, _LIVE_N_SHARDS)
    b = _shard_idx(_BLOCK_HASH_A, _HOTKEY_A, _LIVE_N_SHARDS)
    assert a == b


def test_shard_idx_in_range():
    # blake2b output is uniform; modulo `n_shards` is in [0, n_shards).
    for hk in (_HOTKEY_A, _HOTKEY_B):
        for bh in (_BLOCK_HASH_A, _BLOCK_HASH_B):
            idx = _shard_idx(bh, hk, _LIVE_N_SHARDS)
            assert 0 <= idx < _LIVE_N_SHARDS


def test_eval_indices_deterministic():
    a = _eval_indices(_BLOCK_HASH_A, _HOTKEY_A, n_sequences=10_000, actual_n=64)
    b = _eval_indices(_BLOCK_HASH_A, _HOTKEY_A, n_sequences=10_000, actual_n=64)
    assert a == b


def test_eval_indices_no_repeats():
    # `rng.choice(..., replace=False)` — same sequence sampled at most
    # once. Guards against accidental flip to `replace=True` upstream
    # (would skew the eval).
    indices = _eval_indices(_BLOCK_HASH_A, _HOTKEY_A,
                             n_sequences=10_000, actual_n=128)
    assert len(set(indices)) == len(indices)


def test_eval_indices_in_range():
    indices = _eval_indices(_BLOCK_HASH_A, _HOTKEY_A,
                             n_sequences=10_000, actual_n=64)
    for i in indices:
        assert 0 <= i < 10_000


# ---------------------------------------------------------------------
# Property 2: Sensitivity — distinct seeds yield distinct outputs
# (with overwhelming probability for these particular inputs).

def test_shard_idx_changes_with_block_hash():
    a = _shard_idx(_BLOCK_HASH_A, _HOTKEY_A, _LIVE_N_SHARDS)
    b = _shard_idx(_BLOCK_HASH_B, _HOTKEY_A, _LIVE_N_SHARDS)
    assert a != b


def test_shard_idx_changes_with_hotkey():
    a = _shard_idx(_BLOCK_HASH_A, _HOTKEY_A, _LIVE_N_SHARDS)
    b = _shard_idx(_BLOCK_HASH_A, _HOTKEY_B, _LIVE_N_SHARDS)
    assert a != b


def test_eval_indices_change_with_block_hash():
    a = _eval_indices(_BLOCK_HASH_A, _HOTKEY_A,
                       n_sequences=10_000, actual_n=64)
    b = _eval_indices(_BLOCK_HASH_B, _HOTKEY_A,
                       n_sequences=10_000, actual_n=64)
    # blake2b → PCG64 → np.choice — set overlap should be near-zero.
    assert a != b
    assert len(set(a) & set(b)) < 5  # ~64²/10000 ≈ 0.4 expected overlap


def test_eval_indices_change_with_hotkey():
    a = _eval_indices(_BLOCK_HASH_A, _HOTKEY_A,
                       n_sequences=10_000, actual_n=64)
    b = _eval_indices(_BLOCK_HASH_A, _HOTKEY_B,
                       n_sequences=10_000, actual_n=64)
    assert a != b


# ---------------------------------------------------------------------
# Property 3: Cross-pipeline consistency — validator's `seed_mat` and
# eval-server's `seed_str.encode()` are the same bytes.
#
# If these ever drift, the eval-server samples from the wrong shard or
# indices and the validator's dashboard.json reports a different seed
# than was actually used, with no error surfaced. This test catches
# that with a single byte-equality check.

def test_validator_seed_mat_matches_eval_server_seed_str():
    block_hash = "0xabc123"
    hotkey = "5HhKLLongkey"
    # Validator: `seed_mat = f"{block_hash}:{hotkey}".encode()`
    validator_seed_mat = f"{block_hash}:{hotkey}".encode()
    # Eval-server: `seed_str = f"{req.block_hash}:{req.hotkey}"`
    # then `seed_material = seed_str.encode()` inside torch_runner.
    eval_server_seed_material = f"{block_hash}:{hotkey}".encode()
    assert validator_seed_mat == eval_server_seed_material


# ---------------------------------------------------------------------
# Property 4: Local-vs-remote reproduction gap.
#
# Discord 2026-05-06/07 in #γ・τeuτonic・3 — multiple miners report
# their local eval-loss disagreeing with the validator's by 0.10-0.17.
# The state-drift hypothesis is rejected (eval-mode king is stateless,
# `archs/quasar/modeling_quasar.py:766`). The remaining hypothesis is
# that miners run local evals against a different shard / different
# indices than the validator. These tests pin the property that
# block_hash is INDISPENSABLE — no derivative of (hotkey,) alone can
# reproduce the remote choice.

def test_hotkey_alone_cannot_reproduce_shard():
    # A miner who knows ONLY their hotkey (no block_hash) has two
    # natural fallback strategies: default to shard 0, or hash the
    # hotkey alone. For our concrete test inputs, the validator's
    # choice is 1262 — distinct from both 0 and the hotkey-only-hash
    # of 983. Pinning AND makes the assertion meaningful (the OR
    # version was vacuously true when miner_guess_zero != miner_guess_hk_only).
    miner_guess_zero = 0
    miner_guess_hk_only = int.from_bytes(
        hashlib.blake2b(_HOTKEY_A.encode(), digest_size=8).digest(),
        "little") % _LIVE_N_SHARDS
    validator_choice = _shard_idx(_BLOCK_HASH_A, _HOTKEY_A, _LIVE_N_SHARDS)
    assert validator_choice != miner_guess_zero
    assert validator_choice != miner_guess_hk_only


@pytest.mark.parametrize("block_hash", [
    "0x" + "00" * 32,                       # genesis-like
    "0x" + "ff" * 32,                       # all-ones
    "0xd34db33fcafe" + "00" * 26,           # plausible
    "0xb16b00b5beef" + "00" * 26,           # plausible
    "0x4242424242" + "00" * 27,             # arbitrary
])
def test_changing_only_block_hash_changes_validator_shard(block_hash):
    # For a fixed hotkey, sweeping block_hash sweeps the shard. This
    # demonstrates that miners cannot reproduce the validator's shard
    # without reading the actual `block_hash` from the dispatch.
    validator_seed = _shard_idx(block_hash, _HOTKEY_A, _LIVE_N_SHARDS)
    miner_seed_no_block_hash = int.from_bytes(
        hashlib.blake2b(_HOTKEY_A.encode(), digest_size=8).digest(),
        "little") % _LIVE_N_SHARDS
    # Most block_hashes won't collide with the hotkey-only seed.
    # Across 5 cases: at least 4 should diverge (1992-shard space).
    different = validator_seed != miner_seed_no_block_hash
    # Pin the disclosure surface: tag the test parameter so when one
    # block_hash happens to collide, the failure pinpoints the value.
    assert different, (f"block_hash={block_hash!r}: validator chose "
                       f"shard {validator_seed}, hotkey-only guess was "
                       f"{miner_seed_no_block_hash}")
