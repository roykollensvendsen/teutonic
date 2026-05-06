"""Tests for eval.torch_runner._build_probe_verdict.

Wraps probe results into the dict shape every caller expects. Pure
aggregation: takes (ok, reason, status, max_norm_weight, per_seed,
norm_quant, warnings) and produces a dict that emits both the new
diagnostic fields and the legacy loss/grad/n_seeds fields for backward
compatibility with existing eval_server consumers.

These tests pin the shape and aggregation rules described in the
docstring. The function is keyword-only at every callsite so the tests
mirror that.
"""
import math

from eval.torch_runner import (
    FINETUNE_GRAD_NORM_MAX,
    FINETUNE_NORM_WEIGHT_MAX,
    FINETUNE_PARAM_GROUP_GRAD_MAX,
    _build_probe_verdict,
)


def _verdict(**overrides):
    """Build a verdict with sensible defaults, override what the test cares about."""
    defaults = {
        "ok": True,
        "reason": None,
        "status": "ok",
        "max_norm_weight": 1.5,
        "per_seed": [],
        "norm_quant": None,
        "warnings": None,
    }
    defaults.update(overrides)
    return _build_probe_verdict(**defaults)


# ---------------------------------------------------------------------
# Passthroughs — primary verdict fields are forwarded as-is.

def test_passthrough_ok_status_reason():
    v = _verdict(ok=False, status="anti_finetune", reason="grad_blew_up")
    assert v["ok"] is False
    assert v["status"] == "anti_finetune"
    assert v["reason"] == "grad_blew_up"


def test_passthrough_max_norm_weight():
    v = _verdict(max_norm_weight=2.7)
    assert v["max_norm_weight"] == 2.7


def test_passthrough_norm_quant():
    v = _verdict(norm_quant=0.83)
    assert v["norm_quantization"] == 0.83


def test_passthrough_per_seed():
    seeds = [{"loss": 1.0, "global_grad_norm": 0.5}]
    v = _verdict(per_seed=seeds)
    assert v["per_seed"] is seeds


# ---------------------------------------------------------------------
# warnings: None becomes empty list (callers can iterate without None-check).

def test_warnings_none_becomes_empty_list():
    v = _verdict(warnings=None)
    assert v["warnings"] == []


def test_warnings_list_is_passed_through():
    v = _verdict(warnings=["norm_quant_clustered"])
    assert v["warnings"] == ["norm_quant_clustered"]


# ---------------------------------------------------------------------
# Caps come from the module-level constants — every verdict reports the
# threshold that was applied so consumers can render "x > cap" diagnostics.

def test_caps_come_from_module_constants():
    v = _verdict()
    assert v["norm_weight_cap"] == FINETUNE_NORM_WEIGHT_MAX
    assert v["global_grad_norm_cap"] == FINETUNE_GRAD_NORM_MAX
    assert v["param_group_grad_norm_cap"] == FINETUNE_PARAM_GROUP_GRAD_MAX


# ---------------------------------------------------------------------
# Loss aggregation across per_seed.
#
# loss_before = first finite loss in per_seed (the model's loss on the
# first probe seed). loss_after = same value — no SGD step is ever taken
# in the new probe, so the legacy "before/after" fields are equal.
# min_loss_before / max_loss_after = min/max of finite losses across seeds.

def test_loss_aggregation_picks_first_min_max():
    per_seed = [
        {"loss": 2.5, "global_grad_norm": 0.1},
        {"loss": 4.0, "global_grad_norm": 0.2},
        {"loss": 1.8, "global_grad_norm": 0.3},
    ]
    v = _verdict(per_seed=per_seed)
    assert v["loss_before"] == 2.5  # first
    assert v["loss_after"] == 2.5  # no SGD step → same as loss_before
    assert v["min_loss_before"] == 1.8
    assert v["max_loss_after"] == 4.0


def test_non_finite_losses_are_filtered_out():
    # NaN / inf entries don't contribute to first/min/max.
    per_seed = [
        {"loss": float("nan"), "global_grad_norm": 0.1},
        {"loss": 3.0, "global_grad_norm": 0.2},
        {"loss": float("inf"), "global_grad_norm": 0.3},
    ]
    v = _verdict(per_seed=per_seed)
    assert v["loss_before"] == 3.0
    assert v["min_loss_before"] == 3.0
    assert v["max_loss_after"] == 3.0


def test_no_finite_losses_yields_nan_loss_fields():
    per_seed = [{"loss": float("nan"), "global_grad_norm": None}]
    v = _verdict(per_seed=per_seed)
    assert math.isnan(v["loss_before"])
    assert math.isnan(v["loss_after"])
    assert math.isnan(v["min_loss_before"])
    assert math.isnan(v["max_loss_after"])


# ---------------------------------------------------------------------
# Grad-norm aggregation: global_grad_norm = max over seeds of finite
# per-seed `global_grad_norm`. max_grad_norm is the legacy alias for the
# same value.

def test_global_grad_norm_is_max_over_seeds():
    per_seed = [
        {"loss": 1.0, "global_grad_norm": 0.7},
        {"loss": 1.1, "global_grad_norm": 1.4},
        {"loss": 1.2, "global_grad_norm": 0.9},
    ]
    v = _verdict(per_seed=per_seed)
    assert v["global_grad_norm"] == 1.4
    assert v["max_grad_norm"] == 1.4  # legacy alias


def test_no_finite_grads_yields_nan_grad_fields():
    per_seed = [{"loss": 1.0, "global_grad_norm": None}]
    v = _verdict(per_seed=per_seed)
    assert math.isnan(v["global_grad_norm"])
    assert math.isnan(v["max_grad_norm"])


# ---------------------------------------------------------------------
# param_group_grad_norms: per-category max across seeds. The docstring
# describes this as "max over seeds, per category" via `agg_groups`.

def test_param_group_aggregation_is_per_category_max():
    per_seed = [
        {
            "loss": 1.0,
            "global_grad_norm": 0.5,
            "param_group_grad_norms": {"attn": 0.3, "ffn": 0.7},
        },
        {
            "loss": 1.1,
            "global_grad_norm": 0.6,
            "param_group_grad_norms": {"attn": 0.9, "ffn": 0.4},
        },
    ]
    v = _verdict(per_seed=per_seed)
    assert v["param_group_grad_norms"] == {"attn": 0.9, "ffn": 0.7}


def test_param_group_filters_non_finite():
    # NaN / inf entries within a category don't replace a smaller finite max.
    per_seed = [
        {
            "loss": 1.0,
            "global_grad_norm": 0.5,
            "param_group_grad_norms": {"attn": 0.3},
        },
        {
            "loss": 1.1,
            "global_grad_norm": 0.6,
            "param_group_grad_norms": {"attn": float("nan")},
        },
    ]
    v = _verdict(per_seed=per_seed)
    assert v["param_group_grad_norms"] == {"attn": 0.3}


def test_param_group_returns_plain_dict_not_defaultdict():
    # Callers serialize this with json.dumps; defaultdict would deserialize
    # weird. Impl wraps with `dict(agg_groups)` — pin that contract.
    v = _verdict(per_seed=[{
        "loss": 1.0, "global_grad_norm": 0.5,
        "param_group_grad_norms": {"attn": 0.1},
    }])
    assert type(v["param_group_grad_norms"]) is dict


# ---------------------------------------------------------------------
# n_seeds / n_steps_per_seed: bookkeeping fields. n_seeds reports the
# number of per-seed records (whether or not their losses were finite);
# n_steps_per_seed is always 0 because the new probe takes no SGD step.

def test_n_seeds_is_len_per_seed():
    per_seed = [
        {"loss": 1.0, "global_grad_norm": 0.5},
        {"loss": 2.0, "global_grad_norm": 0.6},
    ]
    v = _verdict(per_seed=per_seed)
    assert v["n_seeds"] == 2


def test_n_seeds_zero_for_empty_per_seed():
    v = _verdict(per_seed=[])
    assert v["n_seeds"] == 0


def test_n_steps_per_seed_is_always_zero():
    # The new probe takes no SGD step — legacy field pinned at 0.
    v = _verdict(per_seed=[{"loss": 1.0, "global_grad_norm": 0.5}])
    assert v["n_steps_per_seed"] == 0


# ---------------------------------------------------------------------
# Legacy delta / max_ratio fields: pinned at 0.0 / 1.0 because no SGD
# step is taken. eval_server consumers still read these.

def test_legacy_delta_is_zero():
    v = _verdict(per_seed=[{"loss": 1.0, "global_grad_norm": 0.5}])
    assert v["delta"] == 0.0


def test_legacy_max_ratio_is_one():
    v = _verdict(per_seed=[{"loss": 1.0, "global_grad_norm": 0.5}])
    assert v["max_ratio"] == 1.0
