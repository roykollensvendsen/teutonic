"""Tests for eval.torch_runner norm-related helpers.

Three helpers operate over a model's *Norm-typed submodules:

* `_norm_modules(model)` — list (name, module) for any submodule whose
  type-name contains "Norm". One-liner; spec is the signature.
* `_check_norm_weight_cap(model) -> (ok, reason, max_seen)` — Layer-1
  probe: walks every *Norm `.weight` and rejects if any element exceeds
  `FINETUNE_NORM_WEIGHT_MAX` or is non-finite. Cap is module-level
  constant, env-overridable, so tests `monkeypatch` it.
* `norm_quantization_score(model) -> float | None` — fraction of *Norm
  `.weight` L2 norms (rounded to 4 decimals) that share the most common
  value. Returns None when no *Norm modules. 1.0 means every norm tensor
  has the same L2 (highly suspicious — possible quantization-fraud signal).
"""
import pytest
import torch
import torch.nn as nn

import eval.torch_runner as tr
from eval.torch_runner import (
    _check_norm_weight_cap,
    _norm_modules,
    norm_quantization_score,
)


class _BareModel(nn.Module):
    """Model with no *Norm modules — used for the empty-model branches."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)


class _TinyModelHolder(nn.Module):
    """Model with named LayerNorms whose weights the test sets directly.

    `n_norms` LayerNorm submodules are attached as `norm0`, `norm1`, ...
    so tests can locate them by attribute and override `.weight.data`
    to control L2 norms / max abs values.

    Class name deliberately avoids "Norm" as a substring so it doesn't
    self-match `_norm_modules`'s `"Norm" in type(m).__name__` filter
    and inflate the result list with the container itself.
    """

    def __init__(self, n_norms: int = 2, dim: int = 4):
        super().__init__()
        for i in range(n_norms):
            setattr(self, f"norm{i}", nn.LayerNorm(dim))


# ---------------------------------------------------------------------
# _norm_modules — list of (name, module) for *Norm-named types.

def test_norm_modules_empty_for_model_without_norms():
    assert _norm_modules(_BareModel()) == []


def test_norm_modules_finds_layernorms():
    m = _TinyModelHolder(n_norms=2)
    found = _norm_modules(m)
    names = [name for name, _ in found]
    assert "norm0" in names
    assert "norm1" in names


def test_norm_modules_returns_module_instances():
    m = _TinyModelHolder(n_norms=1)
    found = _norm_modules(m)
    assert len(found) == 1
    name, mod = found[0]
    assert isinstance(mod, nn.LayerNorm)
    assert name == "norm0"


def test_norm_modules_excludes_non_norm_submodules():
    # `_BareModel` has Embedding + Linear; neither has "Norm" in the
    # type name, so `_norm_modules` returns empty.
    m = _BareModel()
    found = _norm_modules(m)
    assert found == []


# ---------------------------------------------------------------------
# _check_norm_weight_cap — Layer-1 cap on *Norm `.weight` magnitudes.

def test_check_cap_empty_model_returns_ok_and_zero():
    # No *Norm modules → no weights walked → (True, None, 0.0).
    ok, reason, max_seen = _check_norm_weight_cap(_BareModel())
    assert ok is True
    assert reason is None
    assert max_seen == 0.0


def test_check_cap_passes_when_all_weights_under_cap(monkeypatch):
    monkeypatch.setattr(tr, "FINETUNE_NORM_WEIGHT_MAX", 30.0)
    m = _TinyModelHolder(n_norms=1)
    with torch.no_grad():
        m.norm0.weight.fill_(2.5)
    ok, reason, max_seen = _check_norm_weight_cap(m)
    assert ok is True
    assert reason is None
    assert max_seen == pytest.approx(2.5)


def test_check_cap_rejects_weight_over_cap(monkeypatch):
    monkeypatch.setattr(tr, "FINETUNE_NORM_WEIGHT_MAX", 5.0)
    m = _TinyModelHolder(n_norms=1)
    with torch.no_grad():
        m.norm0.weight.fill_(10.0)  # > cap=5.0
    ok, reason, _max_seen = _check_norm_weight_cap(m)
    assert ok is False
    assert reason is not None
    assert reason.startswith("norm_weight_cap:")


def test_check_cap_rejects_non_finite_weight(monkeypatch):
    monkeypatch.setattr(tr, "FINETUNE_NORM_WEIGHT_MAX", 30.0)
    m = _TinyModelHolder(n_norms=1)
    with torch.no_grad():
        m.norm0.weight[0] = float("nan")
    ok, reason, _ = _check_norm_weight_cap(m)
    assert ok is False
    assert reason is not None
    assert reason.startswith("norm_weight_non_finite:")


def test_check_cap_max_seen_tracks_max_abs_across_norms(monkeypatch):
    # max_seen aggregates the largest |weight| across every *Norm module
    # the walker visited (so callers can render "actual vs cap" diagnostics).
    monkeypatch.setattr(tr, "FINETUNE_NORM_WEIGHT_MAX", 30.0)
    m = _TinyModelHolder(n_norms=2)
    with torch.no_grad():
        m.norm0.weight.fill_(1.5)
        m.norm1.weight.fill_(7.5)
    ok, reason, max_seen = _check_norm_weight_cap(m)
    assert ok is True
    assert reason is None
    assert max_seen == pytest.approx(7.5)


def test_check_cap_uses_abs_for_negative_weights(monkeypatch):
    # Cap is on |weight|, not raw weight — a large-negative weight
    # triggers rejection.
    monkeypatch.setattr(tr, "FINETUNE_NORM_WEIGHT_MAX", 5.0)
    m = _TinyModelHolder(n_norms=1)
    with torch.no_grad():
        m.norm0.weight.fill_(-10.0)
    ok, reason, _ = _check_norm_weight_cap(m)
    assert ok is False
    assert reason is not None
    assert reason.startswith("norm_weight_cap:")


# ---------------------------------------------------------------------
# norm_quantization_score — fraction of L2 norms sharing the most common
# rounded value across all *Norm `.weight` tensors.

def test_score_returns_none_when_no_norms():
    # Docstring: "Returns None if the model has no norm-like modules."
    assert norm_quantization_score(_BareModel()) is None


def test_score_one_norm_yields_one_point_zero():
    # With a single norm, every L2 trivially clusters to itself → 1.0.
    m = _TinyModelHolder(n_norms=1)
    score = norm_quantization_score(m)
    assert score == pytest.approx(1.0)


def test_score_all_identical_weights_yield_one_point_zero():
    # All n_norms LayerNorms have identical .weight → identical L2 → 1.0.
    m = _TinyModelHolder(n_norms=4)
    with torch.no_grad():
        for i in range(4):
            getattr(m, f"norm{i}").weight.fill_(1.0)
    assert norm_quantization_score(m) == pytest.approx(1.0)


def test_score_half_clustered_yields_one_half():
    # 2 of 4 norms cluster to one L2 value, 2 to another (each unique
    # within itself) → most-common count is 2; total 4 → 0.5.
    m = _TinyModelHolder(n_norms=4)
    with torch.no_grad():
        m.norm0.weight.fill_(1.0)
        m.norm1.weight.fill_(1.0)
        m.norm2.weight.fill_(2.0)
        m.norm3.weight.fill_(3.0)
    score = norm_quantization_score(m)
    assert score == pytest.approx(0.5)


def test_score_all_distinct_yields_inverse_n():
    # Every L2 distinct → most-common count = 1; total = 4 → 0.25.
    m = _TinyModelHolder(n_norms=4)
    with torch.no_grad():
        m.norm0.weight.fill_(1.0)
        m.norm1.weight.fill_(2.0)
        m.norm2.weight.fill_(3.0)
        m.norm3.weight.fill_(4.0)
    score = norm_quantization_score(m)
    assert score == pytest.approx(0.25)


def test_score_rounds_to_four_decimals_for_clustering():
    # Docstring: L2 rounded to 4 decimals before counting. Two norms
    # whose L2 differs only in the 5th decimal must cluster as one.
    m = _TinyModelHolder(n_norms=2, dim=2)
    with torch.no_grad():
        # Both fills produce L2 = sqrt(2 * 1.0**2) = 1.4142135 ≈ 1.4142
        # The second is perturbed in the 6th decimal — rounds to same 1.4142.
        m.norm0.weight.fill_(1.0)
        m.norm1.weight.fill_(1.0 + 1e-7)
    score = norm_quantization_score(m)
    assert score == pytest.approx(1.0)
