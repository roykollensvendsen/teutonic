"""Tests for eval_penalized.py pure-math functions.

The three target functions are documented as pure (no GPU, no IO):
- `penalize`: asymmetric regression penalty on paired differences
- `score_from_diffs`: deterministic verdict from precomputed diffs
- `score_bootstrap_from_diffs`: bootstrap test on precomputed diffs (seeded)

Used by experiment scripts to re-score the same losses under different
(alpha, delta, beta) without re-running model inference.
"""
import numpy as np

from eval_penalized import penalize

# `penalize(d, beta)` — docstring: "Apply asymmetric regression penalty
# to paired differences. d: d_i = king_loss_i - challenger_loss_i
# (positive = challenger better). beta: penalty multiplier for
# regressions (beta=1 means regressions count double). Returns u:
# penalized scores where regressions are amplified."

def test_penalize_zeros_stay_zero():
    d = np.zeros(5)
    u = penalize(d, beta=1.0)
    assert np.all(u == 0)


def test_penalize_returns_array_same_length():
    d = np.array([1.0, -1.0, 0.5, -0.5])
    u = penalize(d, beta=1.0)
    assert len(u) == len(d)


def test_penalize_beta_zero_is_identity():
    # beta=0 means no penalty at all — output equals input.
    d = np.array([1.0, -1.0, 2.5, -3.5])
    u = penalize(d, beta=0.0)
    np.testing.assert_allclose(u, d)


def test_penalize_positive_values_unchanged():
    # Non-regressions (positive d_i = challenger is better) should not
    # be penalized regardless of beta.
    d = np.array([0.5, 1.0, 2.0])
    u = penalize(d, beta=1.0)
    np.testing.assert_allclose(u, d)


def test_penalize_beta_one_doubles_regressions():
    # Per docstring: "beta=1 means regressions count double".
    # For d_i < 0, the magnitude grows by factor (1+beta) = 2.
    d = np.array([-1.0, -2.5, -0.5])
    u = penalize(d, beta=1.0)
    np.testing.assert_allclose(u, d * 2.0)


def test_penalize_larger_beta_amplifies_more():
    d = np.array([-1.0])
    small = penalize(d, beta=0.5)
    large = penalize(d, beta=2.0)
    # Both push d more negative; larger beta gives larger magnitude.
    assert abs(large[0]) > abs(small[0])


def test_penalize_mixed_array_only_penalizes_negatives():
    d = np.array([1.0, -1.0, 2.0, -2.0])
    u = penalize(d, beta=1.0)
    assert u[0] == 1.0  # positive unchanged
    assert u[2] == 2.0  # positive unchanged
    assert u[1] == -2.0  # negative doubled
    assert u[3] == -4.0  # negative doubled
