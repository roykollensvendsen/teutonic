"""Tests for `eval_torch._default_grad_norm_threshold(num_params)`.

Context: the trainability-probe rejects candidates whose finetune
gradient norm exceeds `FINETUNE_GRAD_NORM_MAX` (default 500). The
current default was calibrated for 1-8B-parameter models — a comment
in eval_torch.py:577-579 says "A healthy ~1B-7B model gradient on a
256-token random batch is typically O(1)–O(10); 500 is comfortably
above honest models".

After the Quasar/24B switch (2026-05-01), miners report false-positive
untrainability rejections for honestly-trained large models because
gradient norms naturally scale with parameter count. Sebastian
(2026-05-02 21:14) reported observed honest grad norms in the
435–725 range for 25B models — 500 puts honest models right at the
boundary.

Proposed fix: introduce a scaling helper that preserves backward
compatibility for ≤8B models and scales the cap upward for larger
models. The probe path then consults this helper instead of the
hardcoded constant.

These tests pin the helper's contract. They will fail at import
time until the helper is added, which is intentional (TDD red).
"""
import math

import pytest

from eval_torch import _default_grad_norm_threshold

# ---------------------------------------------------------------------
# Type and basic-shape contract.

def test_returns_a_positive_float():
    assert _default_grad_norm_threshold(8_000_000_000) > 0
    assert isinstance(_default_grad_norm_threshold(8_000_000_000), float)


def test_finite_for_realistic_sizes():
    for n in (1_000_000_000, 8_000_000_000, 25_000_000_000, 100_000_000_000):
        v = _default_grad_norm_threshold(n)
        assert math.isfinite(v)


# ---------------------------------------------------------------------
# Backward compatibility — small models get the historical 500 cap.

def test_8b_returns_baseline_500():
    # 8B is the calibration anchor — must equal the historical default.
    assert _default_grad_norm_threshold(8_000_000_000) == pytest.approx(500.0)


def test_below_baseline_does_not_tighten():
    # Smaller models must NOT get a tighter cap — that would retroactively
    # reject historically-accepted small models.
    for n in (100_000_000, 1_000_000_000, 4_000_000_000, 7_000_000_000):
        assert _default_grad_norm_threshold(n) >= 500.0, n


# ---------------------------------------------------------------------
# Scaling for larger models — Sebastian's empirical 435-725 range
# for 25B should comfortably pass.

def test_25b_threshold_clears_observed_honest_range():
    # Sebastian observed 435-725 for 25B honest models. The cap must be
    # above 725 with reasonable headroom (≥30%) so noise doesn't trip it.
    assert _default_grad_norm_threshold(25_000_000_000) >= 725.0 * 1.3


def test_100b_threshold_higher_than_25b():
    # Strictly increasing for the relevant range above the baseline.
    assert (_default_grad_norm_threshold(100_000_000_000)
            > _default_grad_norm_threshold(25_000_000_000))


# ---------------------------------------------------------------------
# Monotonicity (non-decreasing) across the full range — never get
# tighter as models grow.

def test_monotonically_non_decreasing():
    sizes = [
        100_000_000, 1_000_000_000, 4_000_000_000, 8_000_000_000,
        16_000_000_000, 25_000_000_000, 50_000_000_000, 100_000_000_000,
    ]
    thresholds = [_default_grad_norm_threshold(n) for n in sizes]
    for a, b in zip(thresholds, thresholds[1:], strict=True):
        assert b >= a


# ---------------------------------------------------------------------
# Sanity bounds — refuse pathological returns.

def test_25b_threshold_not_absurdly_high():
    # A linear-in-N scaling would give 25B → ~12500, which lets
    # exploits through. Cap headroom at a sane upper bound.
    assert _default_grad_norm_threshold(25_000_000_000) <= 5000.0


def test_zero_params_does_not_crash():
    # Defensive: `len(list(model.parameters()))` could in principle be
    # 0 for a malformed model. The helper should return *something*
    # finite, not raise.
    v = _default_grad_norm_threshold(0)
    assert math.isfinite(v)
    assert v > 0
