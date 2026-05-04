"""Tests for archs.quasar.size pure helpers.

Two pure functions live in size.py:

* `_classify(name)` — maps a `model.named_parameters()` path string into
  one of 14 buckets. Used by `count_params` to break the total/active
  parameter count down per architectural component when sizing a model.
  First-match-wins on substring patterns; case-insensitive (lowercases
  the name before matching).

* `build_config(args)` — argparse Namespace → QuasarConfig mapping.
  Hardcodes `moe_type="bigmac"` and `num_key_value_heads=args.n_heads`;
  every other field is read from the namespace.
"""
import argparse
import importlib
import importlib.machinery
import sys
import types


def _stub_accelerate() -> None:
    """Stub `accelerate` before `archs.quasar.size` imports it.

    `archs.quasar.size` imports `accelerate.init_empty_weights` at
    module load time for its `main` script path, but `accelerate` is
    not a test dependency and is only used inside `main`. The stub
    lets the pure helpers be exercised in isolation without dragging
    in the optional dep.
    """
    if "accelerate" in sys.modules:
        return
    stub = types.ModuleType("accelerate")
    stub.__spec__ = importlib.machinery.ModuleSpec("accelerate", loader=None)
    # transformers' `is_accelerate_available()` reads __version__ via
    # importlib.metadata; absent that, it falls back to "N/A" which
    # `packaging.version.parse` then rejects. Pin a benign version so
    # transformers' availability gate stops complaining.
    stub.__version__ = "1.0.0"
    stub.init_empty_weights = lambda *a, **kw: None
    sys.modules["accelerate"] = stub


_stub_accelerate()
_size_mod = importlib.import_module("archs.quasar.size")
_classify = _size_mod._classify
build_config = _size_mod.build_config


# ---------------------------------------------------------------------
# _classify — bucket assignment for parameter names.
#
# Pin each pattern to its bucket so an accidental rename in size.py
# would surface as a test break rather than as a silent shift in the
# size-report breakdown.

def test_classify_embed_tokens():
    assert _classify("model.embed_tokens.weight") == "embed"


def test_classify_lm_head():
    assert _classify("lm_head.weight") == "lm_head"


def test_classify_routed_experts_w12():
    assert _classify("layers.0.moe.experts_w12") == "moe_experts_routed"


def test_classify_routed_experts_w3():
    assert _classify("layers.0.moe.experts_w3") == "moe_experts_routed"


def test_classify_shared_experts():
    assert _classify("layers.0.moe.shared_experts.gate.weight") \
        == "moe_experts_shared"


def test_classify_dcca_down_proj():
    assert _classify("layers.0.moe.w_down_proj") == "moe_dcca"


def test_classify_dcca_up_proj():
    assert _classify("layers.0.moe.w_up_proj") == "moe_dcca"


def test_classify_router():
    assert _classify("layers.0.moe.router.weight") == "moe_router"


def test_classify_smebu_buffer_moe_bias():
    assert _classify("layers.0.moe.moe_bias") == "moe_smebu_buffers"


def test_classify_smebu_buffer_moe_momentum():
    assert _classify("layers.0.moe.moe_momentum") == "moe_smebu_buffers"


def test_classify_smebu_buffer_max_vio():
    assert _classify("layers.0.moe.max_vio") == "moe_smebu_buffers"


def test_classify_latent_memory_via_memory_segment():
    # Real PyTorch param names contain `.memory.` for latent-memory params
    # (e.g. `layers.0.memory.W_alpha`); the `.memory.` substring is what
    # actually catches them in practice. (The `.W_alpha` / `.C_to_hidden`
    # patterns in `_classify` are unreachable after `n = name.lower()` —
    # noted as a separate cleanup, not load-bearing for sizing.)
    assert _classify("layers.0.memory.W_alpha") == "latent_memory"


def test_classify_ffn_dense_gate():
    assert _classify("layers.0.ffn.gate.weight") == "ffn_dense"


def test_classify_ffn_dense_up():
    assert _classify("layers.0.ffn.up.weight") == "ffn_dense"


def test_classify_ffn_dense_down():
    assert _classify("layers.0.ffn.down.weight") == "ffn_dense"


def test_classify_attn():
    assert _classify("layers.0.attn.q_proj.weight") == "attn"


def test_classify_attn_via_proj_substring():
    # "_proj" alone (no "attn.") still matches the attn bucket.
    assert _classify("layers.0.q_proj") == "attn"


def test_classify_norm():
    assert _classify("layers.0.input_layernorm.weight") == "norm"


def test_classify_ln1():
    assert _classify("layers.0.ln1.weight") == "norm"


def test_classify_rope():
    assert _classify("layers.0.rotary_emb.inv_freq") == "rope"


def test_classify_injection():
    assert _classify("layers.0.injection_gate") == "injection"


def test_classify_other_fallback():
    # No pattern matches → "other".
    assert _classify("model.unknown_param") == "other"


def test_classify_is_case_insensitive():
    assert _classify("MODEL.EMBED_TOKENS.WEIGHT") == "embed"


# Order of patterns in `_classify` matters: experts_w12 is checked
# before "router" and before "_proj"/"attn." — so a name that contains
# *both* "experts_w12" and "router" routes to moe_experts_routed.
def test_classify_priority_experts_over_router():
    assert _classify("layers.0.moe.router.experts_w12") \
        == "moe_experts_routed"


# embed_tokens is checked before lm_head — a name with both routes to
# embed (the tied-embedding case in count_params suppresses lm_head
# separately, so this priority is intentional).
def test_classify_priority_embed_over_lm_head():
    assert _classify("model.embed_tokens.lm_head") == "embed"


# ---------------------------------------------------------------------
# build_config — argparse Namespace → QuasarConfig.

def _full_args(**overrides):
    """Build the argparse namespace `build_config` reads from.

    Mirrors `archs.quasar.size.main`'s argparse defaults so each test
    can override only the field it cares about.
    """
    defaults = {
        "vocab_size": 262144,
        "hidden": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "d_ff": 11008,
        "head_dim": 128,
        "max_seq_len": 16384,
        "tie_word_embeddings": True,
        "quasar_layers": 4,
        "gated_layers": 2,
        "memory_slots": 128,
        "memory_dim": 128,
        "num_shared_experts": 1,
        "num_experts": 80,
        "top_k": 10,
        "shared_expert_size": 4096,
        "routed_expert_size": 2560,
        "dense_input_layers": 4,
        "bigmac_r": 0.25,
        "rope_theta": 1_000_000.0,
        "bos_token_id": 2,
        "eos_token_id": 1,
        "pad_token_id": 0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_config_propagates_basic_dims():
    cfg = build_config(_full_args(hidden=2048, n_layers=16, d_ff=8192))
    assert cfg.d_model == 2048
    assert cfg.n_layers == 16
    assert cfg.d_ff == 8192


def test_build_config_hardcodes_moe_type_bigmac():
    cfg = build_config(_full_args())
    assert cfg.moe_type == "bigmac"


def test_build_config_kv_heads_mirrors_n_heads():
    # build_config maps args.n_heads → both n_heads and num_key_value_heads
    # (i.e. MHA, not GQA). A divergence here would silently break the
    # sizing report for any args that intended GQA.
    cfg = build_config(_full_args(n_heads=24))
    assert cfg.n_heads == 24
    assert cfg.num_key_value_heads == 24


def test_build_config_routes_num_experts_to_num_routed_experts():
    # The CLI flag is `--num-experts` (args.num_experts) but the
    # QuasarConfig field is `num_routed_experts`.
    cfg = build_config(_full_args(num_experts=128))
    assert cfg.num_routed_experts == 128


def test_build_config_propagates_token_ids():
    cfg = build_config(_full_args(bos_token_id=5, eos_token_id=6,
                                  pad_token_id=7))
    assert cfg.bos_token_id == 5
    assert cfg.eos_token_id == 6
    assert cfg.pad_token_id == 7


def test_build_config_propagates_expert_sizes_and_top_k():
    cfg = build_config(_full_args(top_k=4, shared_expert_size=3072,
                                  routed_expert_size=1024))
    assert cfg.top_k == 4
    assert cfg.shared_expert_size == 3072
    assert cfg.routed_expert_size == 1024
