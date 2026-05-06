"""Tests for eval.torch_runner._seed_for_iteration.

A 1-line helper that derives a per-iteration PRNG seed from the
module-level `PROBE_SEED` constant. Used by `trainability_probe` to give
each of the N seeds in a multi-seed probe a stable, distinct uint32
seed without collapsing to PROBE_SEED itself for i=0.

Contract from docstring + signature:
* Returns int.
* "Stable per-seed-index PRNG seed derived from PROBE_SEED" — same i
  must give same output, and the output must depend on PROBE_SEED.
"""
from eval.torch_runner import _seed_for_iteration


def test_seed_is_deterministic_for_same_index():
    # "Stable per-seed-index" — calling twice with the same i must yield
    # the same seed (no internal RNG state).
    assert _seed_for_iteration(0) == _seed_for_iteration(0)
    assert _seed_for_iteration(7) == _seed_for_iteration(7)


def test_seed_fits_in_uint32():
    # Every result must be a non-negative 32-bit value — trainability
    # probe seeds are passed downstream as torch RNG seeds, which the
    # PyTorch generator API requires to fit in [0, 2**32-1].
    for i in range(20):
        s = _seed_for_iteration(i)
        assert 0 <= s <= 0xFFFFFFFF


def test_seed_returns_int():
    assert isinstance(_seed_for_iteration(0), int)


def test_seed_distinct_for_neighbouring_indices():
    # The point of having a per-i seed at all is that neighbouring
    # iterations don't collapse to the same RNG state. The golden-ratio
    # multiplier (0x9E3779B1) gives well-separated outputs for adjacent
    # i; if a future refactor breaks this, every seed in the probe
    # would re-trace the same forward pass.
    seeds = {_seed_for_iteration(i) for i in range(8)}
    assert len(seeds) == 8


def test_seed_depends_on_probe_seed_constant(monkeypatch):
    # "derived from PROBE_SEED" — monkeypatching the module-level
    # constant must change the output, otherwise the docstring is
    # lying and operators can't override the seed via the env var.
    import eval.torch_runner as tr
    monkeypatch.setattr(tr, "PROBE_SEED", 0xDEADBEEF)
    s_a = _seed_for_iteration(0)
    monkeypatch.setattr(tr, "PROBE_SEED", 0x12345678)
    s_b = _seed_for_iteration(0)
    assert s_a != s_b
