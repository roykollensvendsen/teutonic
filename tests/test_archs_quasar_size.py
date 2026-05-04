"""Tests for archs.quasar.size._classify.

Maps a `model.named_parameters()` path string into one of 14 buckets.
Used by `count_params` to break the total/active parameter count down
per architectural component when sizing a model. First-match-wins on
substring patterns; case-insensitive (lowercases the name before
matching).
"""
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
_classify = importlib.import_module("archs.quasar.size")._classify


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
