"""Tests for eval.torch_runner._classify_param.

The docstring promises a *coarse* classifier returning one of:
    {attn, ffn, embed, lm_head, norm, bias, other}

with priority ordering:
    lm_head/embed > norm > attn/ffn > bias > other

(lm_head/embed first because they sometimes share paths with `norm`;
bias is a fallback only for unattributed `.bias` tensors.)

These tests pin only the docstring-promised buckets and the docstring-
promised priority. The implementation also routes some Quasar-specific
strings into MoE / memory buckets (`moe_router`, `moe_routed`,
`moe_shared`, `moe_dcca`, `moe_smebu`, `memory`, `looped_inject`); those
buckets are not in the docstring's promised set, so we leave them
untested rather than impl-leak the test set.
"""
from eval.torch_runner import _classify_param

PROMISED_BUCKETS = {"attn", "ffn", "embed", "lm_head", "norm", "bias", "other"}


# ---------------------------------------------------------------------
# Each promised bucket has at least one canonical transformer param
# name that lands in it. These names are HF/PyTorch standard, not
# Quasar-specific — they're the names the contract is implicitly
# written against.

def test_classify_embed_tokens_returns_embed():
    assert _classify_param("model.embed_tokens.weight") == "embed"


def test_classify_lm_head_returns_lm_head():
    assert _classify_param("lm_head.weight") == "lm_head"


def test_classify_layernorm_returns_norm():
    assert _classify_param("model.layers.0.input_layernorm.weight") == "norm"


def test_classify_q_proj_returns_attn():
    assert _classify_param("model.layers.0.self_attn.q_proj.weight") == "attn"


def test_classify_mlp_gate_proj_returns_ffn():
    assert _classify_param("model.layers.0.mlp.gate_proj.weight") == "ffn"


def test_classify_unattributed_bias_returns_bias():
    # Docstring: "bias as a fallback for unattributed `.bias` tensors".
    # A bare `.bias` with no other recognised pattern lands in `bias`.
    assert _classify_param("some.unknown.module.bias") == "bias"


def test_classify_unrecognised_returns_other():
    # The catch-all bucket for names matching no pattern.
    assert _classify_param("model.unknown_param_name") == "other"


# ---------------------------------------------------------------------
# Every result is in the promised set for any of the canonical inputs
# above. (We can't claim it for *all* possible inputs — the impl returns
# Quasar-specific buckets too — but the docstring contract is that the
# generic transformer names route into the 7-bucket set.)

def test_classify_promised_inputs_stay_in_promised_set():
    for name in [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "some.unknown.module.bias",
        "model.unknown_param_name",
    ]:
        assert _classify_param(name) in PROMISED_BUCKETS


# ---------------------------------------------------------------------
# Priority: lm_head/embed are checked *before* norm. The docstring calls
# this out explicitly ("they sometimes share paths with `norm`"). A name
# containing both substrings must route to lm_head/embed, not norm.

def test_classify_priority_lm_head_over_norm():
    # A param path containing both "lm_head" and "norm" → lm_head wins.
    assert _classify_param("model.lm_head_norm.weight") == "lm_head"


def test_classify_priority_embed_over_norm():
    # A param path containing both "embed_tokens" and "norm" → embed wins.
    assert _classify_param("model.embed_tokens_norm.weight") == "embed"


# ---------------------------------------------------------------------
# Priority: norm before attn/ffn. A name with both "norm" and an attn/ffn
# pattern must route to `norm`. (Less load-bearing than the above — but
# the docstring's stated ordering implies it.)

def test_classify_priority_norm_over_attn():
    # `q_proj` would normally land in attn; layernorm-prefixed routes to norm.
    assert _classify_param("model.layers.0.layernorm.q_proj") == "norm"


# ---------------------------------------------------------------------
# Priority: bias is fallback. An attn-pattern param with `.bias` suffix
# stays in `attn`, not `bias`, because attn is checked first.

def test_classify_attn_bias_stays_attn():
    assert _classify_param("model.layers.0.self_attn.q_proj.bias") == "attn"


# Same for ffn.bias — the ffn pattern wins over the bias fallback.
def test_classify_ffn_bias_stays_ffn():
    assert _classify_param("model.layers.0.mlp.gate_proj.bias") == "ffn"
