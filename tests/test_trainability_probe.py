"""Tests for eval.torch_runner.trainability_probe public contract.

Docstring contract:

    Take one SGD step on a random-token batch; return a verdict dict.

    Result keys:
      ok: bool — True if model survived the step (loss didn't explode).
      reason: str | None — short rejection reason if not ok.
      loss_before: float
      loss_after: float
      delta: float (loss_after - loss_before)

    Restores all parameters to their pre-probe values via a per-param data
    snapshot (.clone() on each .data buffer) regardless of success or failure.

These tests pin only what the docstring promises:

* Result dict shape + types.
* `delta == loss_after - loss_before` (semantic from key list).
* Parameters bit-exactly restored on the success path.
* Parameters bit-exactly restored on the failure path
  (the docstring says "regardless of success or failure").
* `ok=True` for a well-behaved model.
* `ok=False` for a model whose forward produces non-finite loss.

Buffer immutability and training-mode restoration are NOT tested here —
the docstring is silent on both. A separate prod-PR documenting those
guarantees in the docstring would let a follow-up test PR pin them.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from eval.torch_runner import trainability_probe

VOCAB = 32
HIDDEN = 8


class _MicroLM(nn.Module):
    """Tiny CPU-only LM that satisfies trainability_probe's expectations.

    The probe needs:
      * `next(model.parameters()).device` — works on any nn.Module.
      * `model.config.vocab_size` — HF-style config attribute.
      * A forward pass returning either logits directly or an object
        with `.logits` (probe checks via `hasattr(out, "logits")`).
      * Parameters whose `.data` and `.grad` are mutable
        (standard nn.Module behavior).
    """

    def __init__(self, vocab_size: int = VOCAB):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.embed = nn.Embedding(vocab_size, HIDDEN)
        self.head = nn.Linear(HIDDEN, vocab_size)

    def forward(self, input_ids):
        h = self.embed(input_ids)
        return SimpleNamespace(logits=self.head(h))


def _snapshot_params(model: nn.Module) -> dict:
    return {n: p.data.clone() for n, p in model.named_parameters()}


def _params_equal(model: nn.Module, snapshot: dict) -> bool:
    """Bit-exact parameter comparison that treats NaN as equal-to-NaN.

    `torch.equal` follows IEEE 754: NaN != NaN, so a snapshot of a
    NaN-filled tensor never compares equal to itself. The restoration
    contract here is "byte-identical to snapshot" — np.array_equal
    with equal_nan=True is the cleanest stdlib-style match.
    """
    for n, p in model.named_parameters():
        a = p.data.detach().cpu().numpy()
        b = snapshot[n].detach().cpu().numpy()
        if not np.array_equal(a, b, equal_nan=True):
            return False
    return True


# ---------------------------------------------------------------------
# Result-dict shape.

def test_probe_returns_dict_with_documented_keys():
    result = trainability_probe(_MicroLM())

    assert isinstance(result, dict)
    assert set(result.keys()) >= {
        "ok", "reason", "loss_before", "loss_after", "delta",
    }


def test_probe_result_field_types():
    result = trainability_probe(_MicroLM())

    assert isinstance(result["ok"], bool)
    assert result["reason"] is None or isinstance(result["reason"], str)
    assert isinstance(result["loss_before"], float)
    assert isinstance(result["loss_after"], float)
    assert isinstance(result["delta"], float)


def test_probe_delta_equals_loss_after_minus_loss_before():
    result = trainability_probe(_MicroLM())

    assert result["delta"] == pytest.approx(
        result["loss_after"] - result["loss_before"]
    )


# ---------------------------------------------------------------------
# Parameter restoration (the docstring's central guarantee).

def test_probe_restores_parameters_on_success_path():
    model = _MicroLM()
    snapshot = _snapshot_params(model)

    result = trainability_probe(model)

    assert result["ok"] is True
    assert _params_equal(model, snapshot)


def test_probe_restores_parameters_on_failure_path():
    # Construct a model whose forward produces a non-finite loss so the
    # probe takes the failure path. The docstring's guarantee says
    # parameter restoration runs "regardless of success or failure".
    model = _MicroLM()
    with torch.no_grad():
        model.embed.weight.fill_(float("nan"))
    snapshot = _snapshot_params(model)

    result = trainability_probe(model)

    assert result["ok"] is False
    assert _params_equal(model, snapshot)


# ---------------------------------------------------------------------
# Verdict semantics.

def test_probe_returns_ok_true_for_well_behaved_model():
    # A freshly-initialized small LM has finite forward loss and a
    # tiny SGD step shouldn't blow it up.
    result = trainability_probe(_MicroLM())

    assert result["ok"] is True
    assert result["reason"] is None


def test_probe_returns_ok_false_when_forward_produces_non_finite_loss():
    model = _MicroLM()
    with torch.no_grad():
        model.embed.weight.fill_(float("nan"))

    result = trainability_probe(model)

    assert result["ok"] is False
    assert isinstance(result["reason"], str)
