"""Tests for eval_penalized.py pure-math functions.

The three target functions are documented as pure (no GPU, no IO):
- `penalize`: asymmetric regression penalty on paired differences
- `score_from_diffs`: deterministic verdict from precomputed diffs
- `score_bootstrap_from_diffs`: bootstrap test on precomputed diffs (seeded)

Used by experiment scripts to re-score the same losses under different
(alpha, delta, beta) without re-running model inference.
"""
import numpy as np

from eval_penalized import penalize, score_from_diffs

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


# `score_from_diffs(d, alpha, delta, beta)` — deterministic verdict
# from precomputed diffs. Returns a dict; call site
# (scripts/experiment_penalized.py:264) accesses "accepted" key.
# alpha = significance level (typical 0.05), delta = effect size
# threshold (validator uses delta = 1/N).

def test_score_from_diffs_returns_dict_with_accepted():
    d = np.array([0.5, 0.4, 0.6, 0.3, 0.5])  # all positive — clearly better
    verdict = score_from_diffs(d, alpha=0.05, delta=0.01, beta=0.0)
    assert isinstance(verdict, dict)
    assert "accepted" in verdict
    assert isinstance(verdict["accepted"], (bool, np.bool_))


def test_score_from_diffs_is_deterministic():
    d = np.array([0.1, -0.05, 0.2, -0.1, 0.3])
    a = score_from_diffs(d, alpha=0.05, delta=0.01, beta=0.0)
    b = score_from_diffs(d, alpha=0.05, delta=0.01, beta=0.0)
    assert a == b


def test_score_from_diffs_strongly_positive_accepted():
    # Clear win: 100 strongly positive diffs.
    rng = np.random.default_rng(0)
    d = rng.normal(loc=0.5, scale=0.05, size=100)
    verdict = score_from_diffs(d, alpha=0.05, delta=0.01, beta=0.0)
    assert verdict["accepted"] is True or verdict["accepted"] == np.True_


def test_score_from_diffs_strongly_negative_rejected():
    # Clear loss: 100 strongly negative diffs.
    rng = np.random.default_rng(0)
    d = rng.normal(loc=-0.5, scale=0.05, size=100)
    verdict = score_from_diffs(d, alpha=0.05, delta=0.01, beta=0.0)
    assert verdict["accepted"] is False or verdict["accepted"] == np.False_


def test_score_from_diffs_higher_beta_makes_acceptance_harder():
    # Marginal challenger (small positive mean with regressions mixed in).
    # Higher beta amplifies the regressions, so a barely-passing diff
    # may flip from accepted to rejected.
    rng = np.random.default_rng(1)
    d = rng.normal(loc=0.02, scale=0.1, size=200)  # marginal
    no_penalty = score_from_diffs(d, alpha=0.05, delta=0.005, beta=0.0)
    heavy_penalty = score_from_diffs(d, alpha=0.05, delta=0.005, beta=10.0)
    # Heavy penalty must be at least as strict as no penalty:
    # if no_penalty rejects, heavy_penalty must also reject.
    if not no_penalty["accepted"]:
        assert not heavy_penalty["accepted"]
