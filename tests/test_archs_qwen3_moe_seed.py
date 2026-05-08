"""Tests for archs.qwen3_moe.seed pure helpers.

Mirrors the pattern from `tests/test_archs_quasar_seed.py`, adapted
for the Qwen3-MoE arch. Three pure helpers:

* `build_config()` — Qwen3MoeConfig factory. Takes no args; every
  field is read from module-level constants that snapshot `TEUTONIC_SEED_*`
  env vars at import time. Tests monkeypatch the constants directly
  to avoid env-reload cost.

* `_strip_auto_map(out_dir)` — mutates `config.json` on disk to drop
  the `auto_map` field. Documented as a no-op for vanilla Qwen3MoE
  (Auto* dispatches via `model_type`), but kept for parity with
  archs/quasar/seed for defense in depth.

* `_count_active(model, cfg)` — given a real model, returns
  `(total_params, active_params)`. Active = total for non-expert
  params; for `.experts.` tensors active = total × top_k / num_experts.
  Handles tied embeddings (lm_head skipped, embed_tokens deduplicated).
"""
import json
from types import SimpleNamespace

import pytest

from archs.qwen3_moe import seed

pytestmark = pytest.mark.arch_specific("archs.qwen3_moe")


# ---------------------------------------------------------------------
# build_config — module-constant snapshotting + invariants.

def test_build_config_uses_module_constants_for_dims(monkeypatch):
    monkeypatch.setattr(seed, "HIDDEN_SIZE", 2048)
    monkeypatch.setattr(seed, "NUM_LAYERS", 16)
    monkeypatch.setattr(seed, "INTERMEDIATE_SIZE", 8192)
    cfg = seed.build_config()
    assert cfg.hidden_size == 2048
    assert cfg.num_hidden_layers == 16
    assert cfg.intermediate_size == 8192


def test_build_config_hardcodes_tied_embeddings():
    # Per docstring/comment: the seed ships a tied-embedding 80B Qwen3MoE
    # with the Gemma3 token IDs (vocab=262144, BOS=2 EOS=1 PAD=0).
    cfg = seed.build_config()
    assert cfg.tie_word_embeddings is True


def test_build_config_default_token_ids_are_gemma3():
    # The defaults match Teutonic-I tokenizer (Gemma3-derived):
    # bos=2, eos=1, pad=0. Documented in build_config's inline comment.
    cfg = seed.build_config()
    assert cfg.bos_token_id == 2
    assert cfg.eos_token_id == 1
    assert cfg.pad_token_id == 0


def test_build_config_token_ids_overridable_via_env(monkeypatch):
    # TEUTONIC_SEED_{BOS,EOS,PAD}_ID env vars override the defaults.
    monkeypatch.setenv("TEUTONIC_SEED_BOS_ID", "151643")
    monkeypatch.setenv("TEUTONIC_SEED_EOS_ID", "151645")
    monkeypatch.setenv("TEUTONIC_SEED_PAD_ID", "151643")
    cfg = seed.build_config()
    assert cfg.bos_token_id == 151643
    assert cfg.eos_token_id == 151645
    assert cfg.pad_token_id == 151643


def test_build_config_propagates_moe_fields(monkeypatch):
    monkeypatch.setattr(seed, "NUM_EXPERTS", 80)
    monkeypatch.setattr(seed, "TOP_K", 8)
    monkeypatch.setattr(seed, "MOE_INTERMEDIATE_SIZE", 1024)
    monkeypatch.setattr(seed, "DECODER_SPARSE_STEP", 1)
    monkeypatch.setattr(seed, "ROUTER_AUX_LOSS_COEF", 0.001)
    cfg = seed.build_config()
    assert cfg.num_experts == 80
    assert cfg.num_experts_per_tok == 8
    assert cfg.moe_intermediate_size == 1024
    assert cfg.decoder_sparse_step == 1
    assert cfg.router_aux_loss_coef == pytest.approx(0.001)


def test_build_config_kv_heads_separate_from_attention_heads(monkeypatch):
    # GQA: num_key_value_heads is independent of num_attention_heads.
    # Spec doesn't lock this to mirror — distinct constants.
    monkeypatch.setattr(seed, "NUM_HEADS", 32)
    monkeypatch.setattr(seed, "NUM_KV_HEADS", 4)
    cfg = seed.build_config()
    assert cfg.num_attention_heads == 32
    assert cfg.num_key_value_heads == 4


def test_build_config_sets_architectures_attribute():
    # AutoModelForCausalLM dispatches to Qwen3MoeForCausalLM via
    # model_type, but `architectures` is also stamped for explicitness.
    cfg = seed.build_config()
    assert cfg.architectures == ["Qwen3MoeForCausalLM"]


def test_build_config_norm_topk_prob_hardcoded_true():
    # Always-renormalize routing weights — not exposed as env var.
    cfg = seed.build_config()
    assert cfg.norm_topk_prob is True


# ---------------------------------------------------------------------
# _strip_auto_map — mutate config.json on disk.

def test_strip_auto_map_removes_field_when_present(tmp_path):
    cfg = {
        "model_type": "qwen3_moe",
        "hidden_size": 4096,
        "auto_map": {"AutoConfig": "configuration_qwen3_moe.Qwen3MoeConfig"},
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    seed._strip_auto_map(tmp_path)
    after = json.loads(cfg_path.read_text())
    assert "auto_map" not in after
    # Other fields preserved.
    assert after["model_type"] == "qwen3_moe"
    assert after["hidden_size"] == 4096


def test_strip_auto_map_no_op_when_field_absent(tmp_path):
    # Vanilla Qwen3MoE doesn't ship auto_map — must not write the
    # file (no churn on cfg) and must not raise.
    cfg = {"model_type": "qwen3_moe", "hidden_size": 4096}
    cfg_path = tmp_path / "config.json"
    original = json.dumps(cfg)
    cfg_path.write_text(original)
    mtime_before = cfg_path.stat().st_mtime_ns
    seed._strip_auto_map(tmp_path)
    # Content unchanged; file not rewritten (mtime preserved).
    assert cfg_path.read_text() == original
    assert cfg_path.stat().st_mtime_ns == mtime_before


def test_strip_auto_map_writes_valid_json(tmp_path):
    cfg = {"model_type": "qwen3_moe", "auto_map": {"x": "y"}}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    seed._strip_auto_map(tmp_path)
    # Must round-trip — no trailing garbage, valid JSON.
    parsed = json.loads(cfg_path.read_text())
    assert parsed == {"model_type": "qwen3_moe"}


# ---------------------------------------------------------------------
# _count_active — handle MoE expert sampling + tied embeddings.

def _fake_param(numel: int):
    """Stub a torch parameter for named_parameters() — only .numel() matters."""
    return SimpleNamespace(numel=lambda n=numel: n)


def _model_with_named_params(named):
    """Build a fake model whose `.named_parameters()` yields (name, param)."""
    return SimpleNamespace(named_parameters=lambda: iter(named))


def _make_cfg(*, num_experts=4, top_k=2, tie_word_embeddings=True):
    return SimpleNamespace(
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        tie_word_embeddings=tie_word_embeddings,
    )


def test_count_active_counts_dense_params_in_full():
    # Non-expert params: active = total.
    model = _model_with_named_params([
        ("layers.0.self_attn.q_proj.weight", _fake_param(1000)),
        ("layers.0.self_attn.k_proj.weight", _fake_param(500)),
    ])
    cfg = _make_cfg()
    total, active = seed._count_active(model, cfg)
    assert total == 1500
    assert active == 1500


def test_count_active_applies_top_k_factor_to_experts():
    # Expert params: active = total × top_k / num_experts.
    # 4 experts × 1000 numel = 4000 total. top_k=2 → active=2000.
    model = _model_with_named_params([
        (f"layers.0.mlp.experts.{i}.gate_proj.weight", _fake_param(1000))
        for i in range(4)
    ])
    cfg = _make_cfg(num_experts=4, top_k=2)
    total, active = seed._count_active(model, cfg)
    assert total == 4000
    assert active == 2000


def test_count_active_skips_lm_head_when_tied():
    # tied: lm_head excluded from total + active (it shares storage with embed).
    model = _model_with_named_params([
        ("model.embed_tokens.weight", _fake_param(1000)),
        ("lm_head.weight", _fake_param(1000)),
    ])
    cfg = _make_cfg(tie_word_embeddings=True)
    total, active = seed._count_active(model, cfg)
    assert total == 1000  # lm_head skipped
    assert active == 1000


def test_count_active_dedupes_tied_embed_tokens():
    # Spec: a model may surface embed_tokens twice (e.g. via accelerate
    # sharding); when tied, only the first occurrence counts.
    model = _model_with_named_params([
        ("model.embed_tokens.weight", _fake_param(1000)),
        ("model.embed_tokens.weight", _fake_param(1000)),  # dup
    ])
    cfg = _make_cfg(tie_word_embeddings=True)
    total, active = seed._count_active(model, cfg)
    assert total == 1000  # second occurrence skipped


def test_count_active_includes_lm_head_when_untied():
    # untied: lm_head and embed_tokens both contribute.
    model = _model_with_named_params([
        ("model.embed_tokens.weight", _fake_param(1000)),
        ("lm_head.weight", _fake_param(1000)),
    ])
    cfg = _make_cfg(tie_word_embeddings=False)
    total, active = seed._count_active(model, cfg)
    assert total == 2000
    assert active == 2000
