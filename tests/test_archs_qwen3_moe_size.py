"""Tests for archs.qwen3_moe.size pure helpers.

Mirrors `tests/test_archs_quasar_size.py` adapted for Qwen3-MoE.
Three helpers:

* `build_config(args)` — argparse Namespace → Qwen3MoeConfig. Same
  fields as seed.build_config but driven by CLI args instead of
  module-level constants. Hardcodes `output_router_logits=False`
  and defaults `mlp_only_layers` to empty list when None.

* `_classify(name)` — parameter-name → bucket string. Buckets:
  embed, lm_head, moe_experts, moe_router, attn, mlp_dense, norm,
  other. Case-insensitive (impl lowercases the name first).
  Qwen3MoE-specific: experts live in a `ModuleList` so paths look
  like `layers.0.mlp.experts.{e}.{gate,up,down}_proj`; the router
  is `layers.0.mlp.gate.weight` (NOT inside `.experts.`).

* `count_params(model, cfg)` — walks the meta model, returns
  `(total, active, by_class, by_class_active)`. Active credits
  expert tensors as `total × top_k / num_experts`. Tied embeddings:
  lm_head excluded, embed_tokens deduplicated.
"""
import argparse
import importlib
import importlib.machinery
import sys
import types
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.arch_specific("archs.qwen3_moe")


def _stub_accelerate() -> None:
    """Stub `accelerate` before `archs.qwen3_moe.size` imports it.

    `size.py` imports `accelerate.init_empty_weights` at module load
    time for its `main` script path, but `accelerate` is not a test
    dependency and is only used inside `main`. The stub lets the pure
    helpers be exercised in isolation. Mirrors the pattern in
    `tests/test_archs_quasar_size.py`.
    """
    if "accelerate" in sys.modules:
        return
    stub = types.ModuleType("accelerate")
    stub.__spec__ = importlib.machinery.ModuleSpec("accelerate", loader=None)
    stub.__version__ = "1.0.0"
    stub.init_empty_weights = lambda *a, **kw: None
    sys.modules["accelerate"] = stub


_stub_accelerate()
size = importlib.import_module("archs.qwen3_moe.size")


# ---------------------------------------------------------------------
# build_config — argparse Namespace → Qwen3MoeConfig.

def _ns(**overrides):
    """Build an argparse.Namespace with sensible defaults."""
    base = {
        "vocab_size": 262144, "hidden": 4096, "n_layers": 36, "n_heads": 64,
        "n_kv_heads": 8, "head_dim": 128, "intermediate_size": 12288,
        "max_seq_len": 4096, "tie_word_embeddings": True,
        "num_experts": 80, "top_k": 8, "moe_intermediate_size": 1024,
        "decoder_sparse_step": 1, "norm_topk_prob": True,
        "router_aux_loss_coef": 0.001, "mlp_only_layers": None,
        "rope_theta": 1000000.0,
        "bos_token_id": 2, "eos_token_id": 1, "pad_token_id": 0,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_config_propagates_namespace_fields():
    cfg = size.build_config(_ns(hidden=2048, n_layers=12))
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 12


def test_build_config_propagates_moe_fields():
    cfg = size.build_config(_ns(num_experts=64, top_k=4,
                                  moe_intermediate_size=2048))
    assert cfg.num_experts == 64
    assert cfg.num_experts_per_tok == 4
    assert cfg.moe_intermediate_size == 2048


def test_build_config_mlp_only_layers_none_becomes_empty_list():
    # Spec: `args.mlp_only_layers or []` — None → [].
    cfg = size.build_config(_ns(mlp_only_layers=None))
    assert cfg.mlp_only_layers == []


def test_build_config_mlp_only_layers_explicit_passed_through():
    cfg = size.build_config(_ns(mlp_only_layers=[0, 1, 35]))
    assert cfg.mlp_only_layers == [0, 1, 35]


def test_build_config_forces_output_router_logits_false():
    # Hardcoded — no Namespace field for it. Eval-time forward should
    # never accumulate router-aux logits (eval=non-training).
    cfg = size.build_config(_ns())
    assert cfg.output_router_logits is False


def test_build_config_kv_heads_independent_of_attention_heads():
    cfg = size.build_config(_ns(n_heads=64, n_kv_heads=8))
    assert cfg.num_attention_heads == 64
    assert cfg.num_key_value_heads == 8


# ---------------------------------------------------------------------
# _classify — parameter-name buckets.

@pytest.mark.parametrize("name,bucket", [
    # embed / lm_head
    ("model.embed_tokens.weight", "embed"),
    ("lm_head.weight", "lm_head"),
    # MoE experts: ModuleList format
    ("model.layers.0.mlp.experts.0.gate_proj.weight", "moe_experts"),
    ("model.layers.5.mlp.experts.42.up_proj.weight", "moe_experts"),
    ("model.layers.10.mlp.experts.79.down_proj.weight", "moe_experts"),
    # MoE router (NOT inside .experts.)
    ("model.layers.0.mlp.gate.weight", "moe_router"),
    # attention
    ("model.layers.0.self_attn.q_proj.weight", "attn"),
    ("model.layers.0.self_attn.k_proj.weight", "attn"),
    ("model.layers.0.self_attn.v_proj.weight", "attn"),
    ("model.layers.0.self_attn.o_proj.weight", "attn"),
    ("model.layers.0.self_attn.q_norm.weight", "attn"),
    ("model.layers.0.self_attn.k_norm.weight", "attn"),
    # dense MLP (mlp_only_layers, or anything outside .experts.)
    ("model.layers.0.mlp.gate_proj.weight", "mlp_dense"),
    ("model.layers.0.mlp.up_proj.weight", "mlp_dense"),
    ("model.layers.0.mlp.down_proj.weight", "mlp_dense"),
    # norm
    ("model.norm.weight", "norm"),
    ("model.layers.0.input_layernorm.weight", "norm"),
    ("model.layers.0.post_attention_layernorm.weight", "norm"),
    # fallback
    ("some.unknown.thing.weight", "other"),
])
def test_classify_parameter_name_buckets(name, bucket):
    assert size._classify(name) == bucket


def test_classify_is_case_insensitive():
    # Impl lowercases the name first — uppercase variants must
    # classify the same as canonical lowercase.
    assert size._classify("Model.LM_Head.Weight") == "lm_head"
    assert size._classify("MODEL.LAYERS.0.MLP.EXPERTS.0.GATE_PROJ.WEIGHT") \
        == "moe_experts"


def test_classify_router_is_not_misread_as_expert():
    # Critical edge: the router gate (`mlp.gate.weight`) lives next to
    # experts in the layer. Without the negative `.experts. NOT in n`
    # check it could fall into "moe_experts". Pinning the disambiguator.
    assert size._classify("model.layers.0.mlp.gate.weight") == "moe_router"
    assert size._classify("model.layers.0.mlp.experts.0.gate_proj.weight") \
        == "moe_experts"


# ---------------------------------------------------------------------
# count_params — meta-model param accounting with MoE adjustment.

def _fake_param(numel: int):
    return SimpleNamespace(numel=lambda n=numel: n)


def _model(*named):
    return SimpleNamespace(named_parameters=lambda: iter(named))


def _cfg(*, num_experts=4, top_k=2, tie_word_embeddings=True):
    return SimpleNamespace(
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        tie_word_embeddings=tie_word_embeddings,
    )


def test_count_params_returns_four_tuple():
    total, active, by_class, by_class_active = size.count_params(
        _model(("model.embed_tokens.weight", _fake_param(1000))),
        _cfg(),
    )
    assert isinstance(total, int)
    assert isinstance(active, int)
    assert isinstance(by_class, dict)
    assert isinstance(by_class_active, dict)


def test_count_params_dense_active_equals_total():
    model = _model(
        ("model.layers.0.self_attn.q_proj.weight", _fake_param(1000)),
        ("model.layers.0.self_attn.k_proj.weight", _fake_param(500)),
    )
    total, active, _, _ = size.count_params(model, _cfg())
    assert total == 1500
    assert active == 1500


def test_count_params_applies_top_k_factor_to_experts():
    # 4 experts × 1000 numel each = 4000 total. top_k=2 → active=2000.
    model = _model(*[
        (f"model.layers.0.mlp.experts.{i}.gate_proj.weight", _fake_param(1000))
        for i in range(4)
    ])
    cfg = _cfg(num_experts=4, top_k=2)
    total, active, _, _ = size.count_params(model, cfg)
    assert total == 4000
    assert active == 2000


def test_count_params_skips_lm_head_when_tied():
    model = _model(
        ("model.embed_tokens.weight", _fake_param(1000)),
        ("lm_head.weight", _fake_param(1000)),
    )
    total, active, _, _ = size.count_params(model, _cfg(tie_word_embeddings=True))
    assert total == 1000
    assert active == 1000


def test_count_params_dedupes_tied_embed_tokens():
    model = _model(
        ("model.embed_tokens.weight", _fake_param(1000)),
        ("model.embed_tokens.weight", _fake_param(1000)),
    )
    total, active, _, _ = size.count_params(model, _cfg(tie_word_embeddings=True))
    assert total == 1000


def test_count_params_includes_lm_head_when_untied():
    model = _model(
        ("model.embed_tokens.weight", _fake_param(1000)),
        ("lm_head.weight", _fake_param(1000)),
    )
    total, active, _, _ = size.count_params(model, _cfg(tie_word_embeddings=False))
    assert total == 2000
    assert active == 2000


def test_count_params_by_class_breakdown():
    # by_class sums per-bucket params. Verify the breakdown matches
    # _classify's bucketing.
    model = _model(
        ("model.embed_tokens.weight", _fake_param(1000)),
        ("model.layers.0.self_attn.q_proj.weight", _fake_param(500)),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", _fake_param(2000)),
        ("model.layers.0.mlp.gate.weight", _fake_param(100)),
        ("model.norm.weight", _fake_param(50)),
    )
    cfg = _cfg(num_experts=2, top_k=1, tie_word_embeddings=True)
    _, _, by_class, by_class_active = size.count_params(model, cfg)
    assert by_class["embed"] == 1000
    assert by_class["attn"] == 500
    assert by_class["moe_experts"] == 2000
    assert by_class["moe_router"] == 100
    assert by_class["norm"] == 50
    # Active: experts get top_k / num_experts factor, others 1×.
    assert by_class_active["moe_experts"] == 1000  # 2000 × 1/2
    assert by_class_active["embed"] == 1000
    assert by_class_active["attn"] == 500
