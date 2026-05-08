"""Tests for sharded-eval helpers in eval/torch_runner.

Upstream commit 64fe9d2 (\"LXXX: arch package + sharded eval +
chain-override knob\") introduced sharding support so the 80B Qwen3-MoE
king fits across multiple GPUs. The new device-routing helper is:

* `_lm_head_device(model) -> torch.device` (eval/torch_runner.py:363) —
  returns where lm_head's weight lives. For tied-embedding models this
  equals the embed_tokens device; for untied/sharded models this is
  wherever accelerate placed the head.

Used by `compute_batch_losses` (and `compute_paired_losses`) to relocate
hidden states to the lm_head device before the chunked cross-entropy
loop. Wrong device routing → silent wrong-loss (per-GPU OOM if sharded
across the wrong devices, or zero gradient flow if hidden lives on a
device the head can't read).

The richer surface (`compute_batch_losses` sharded path, full
forward through chunked lm_head, `@torch.no_grad()` decoration) is
out-of-scope here — exercises real torch CUDA + accelerate + a model
of nontrivial size, which is GPU-rig territory. Tests below pin the
pure-helper part.
"""
from types import SimpleNamespace

import pytest
import torch

from eval.torch_runner import _lm_head_device


def _model_with_lm_head_on(device: torch.device, *,
                              n_params: int = 1):
    """Fake model where `lm_head.parameters()` yields tensors on `device`.

    Mirrors the duck-typed contract `_lm_head_device` reads:
    `next(model.lm_head.parameters()).device`. We don't need a real
    `nn.Module`; a SimpleNamespace whose `.lm_head.parameters()`
    returns the right iterable suffices.
    """
    params = [torch.zeros(1, device=device) for _ in range(n_params)]
    lm_head = SimpleNamespace(parameters=lambda: iter(params))
    return SimpleNamespace(lm_head=lm_head)


# ---------------------------------------------------------------------
# Happy paths: returns the device of the lm_head's first parameter.

def test_lm_head_device_returns_cpu_for_cpu_model():
    model = _model_with_lm_head_on(torch.device("cpu"))
    assert _lm_head_device(model) == torch.device("cpu")


def test_lm_head_device_returns_first_parameter_device():
    # Spec says "Where lm_head's weight lives" — singular. Impl reads
    # `next(model.lm_head.parameters())`; if multiple params exist
    # (e.g. weight + bias) only the first matters for routing.
    p0 = torch.zeros(1, device="cpu")
    p1 = torch.zeros(1, device="cpu")
    lm_head = SimpleNamespace(parameters=lambda: iter([p0, p1]))
    model = SimpleNamespace(lm_head=lm_head)
    assert _lm_head_device(model) == p0.device


def test_lm_head_device_returns_meta_device_for_meta_model():
    # accelerate's `init_empty_weights` uses meta tensors; size.py
    # builds a meta-device QuasarConfig+QuasarForCausalLM for sizing.
    # _lm_head_device must work the same way there (used for assertions
    # in the size/sizing path).
    p = torch.zeros(1, device="meta")
    lm_head = SimpleNamespace(parameters=lambda: iter([p]))
    model = SimpleNamespace(lm_head=lm_head)
    assert _lm_head_device(model) == torch.device("meta")


# ---------------------------------------------------------------------
# Error path: missing lm_head raises (impl detail — see SPEC_DEBT.md).
#
# Docstring doesn't specify behaviour when the model has no `lm_head`
# attribute. Impl raises AttributeError because of the bare attribute
# access. Pinning so a refactor that catches/wraps the error has to
# update the docstring + this test together.

def test_lm_head_device_raises_attribute_error_when_lm_head_missing():
    model = SimpleNamespace()  # no .lm_head at all
    with pytest.raises(AttributeError):
        _lm_head_device(model)


def test_lm_head_device_raises_when_parameters_iterator_empty():
    # `next()` on empty iterator raises StopIteration — pin so a
    # caller surrounding this with try/except has the right marker.
    lm_head = SimpleNamespace(parameters=lambda: iter([]))
    model = SimpleNamespace(lm_head=lm_head)
    with pytest.raises(StopIteration):
        _lm_head_device(model)
