"""Smoke tests for the nano-gpt arch package.

archs.nano_gpt is a thin shim over `transformers.GPT2{Config,LMHeadModel}` —
no vendored modeling, no Auto* registration needed (GPT-2 is in-tree). The
tests pin three things so the shim can't silently rot:

1. The package re-exports GPT2Config / GPT2LMHeadModel under those names.
2. `size.build_config(args)` maps argparse fields onto the right HF config
   keys (n_embd vs hidden_size etc — easy to typo since we straddle GPT-2's
   "n_*" naming and Qwen3-MoE's "num_*_*" naming elsewhere in the codebase).
3. `size._classify(name)` buckets named-parameter paths the way
   `count_params` expects. Pin each bucket so an accidental rename surfaces.

Marked arch_specific so the CI matrix only runs them under a chain config
that actually selects nano-gpt; the live LXXX chain stays on qwen3_moe.
"""
import argparse
import importlib
import importlib.machinery
import sys
import types

import pytest

pytestmark = pytest.mark.arch_specific("archs.nano_gpt")


def _stub_accelerate() -> None:
    """Stub `accelerate` before `archs.nano_gpt.size` imports it.

    `archs.nano_gpt.size` imports `accelerate.init_empty_weights` at module
    load for its `main` script path, but `accelerate` is not a hard test
    dependency and is only used inside `main`. The stub lets the pure helpers
    be exercised in isolation without dragging in the optional dep —
    mirrors the same trick used by tests/test_archs_quasar_size.py.
    """
    if "accelerate" in sys.modules:
        return
    stub = types.ModuleType("accelerate")
    stub.__spec__ = importlib.machinery.ModuleSpec("accelerate", loader=None)
    # transformers' is_accelerate_available() reads __version__ via
    # importlib.metadata; absent that it falls back to "N/A" which
    # packaging.version.parse rejects. Pin a benign version.
    stub.__version__ = "1.0.0"
    stub.init_empty_weights = lambda *a, **kw: None
    sys.modules["accelerate"] = stub


_stub_accelerate()


# ---------------------------------------------------------------------
# Package import — proves chain_config.load_arch() will succeed when
# chain.toml selects this module.

def test_package_re_exports_gpt2_config_and_model():
    from transformers import GPT2Config as _Config, GPT2LMHeadModel as _Model

    from archs.nano_gpt import GPT2Config, GPT2LMHeadModel

    assert GPT2Config is _Config
    assert GPT2LMHeadModel is _Model


def test_package_dunder_all_lists_only_the_re_exports():
    import archs.nano_gpt as pkg

    assert set(pkg.__all__) == {"GPT2Config", "GPT2LMHeadModel"}


# ---------------------------------------------------------------------
# size.build_config — argparse → GPT2Config mapping.

def _ns(**overrides):
    """Build a Namespace with the size.py CLI defaults, then overlay overrides.

    Mirrors the production path: argparse parses CLI args using size.main()'s
    defaults, the namespace lands in build_config, build_config builds GPT2Config.
    """
    base = {
        "vocab_size": 50257,
        "n_embd": 128,
        "n_layer": 4,
        "n_head": 4,
        "n_positions": 256,
        "n_inner": None,
        "activation_function": "gelu_new",
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256,
        "tie_word_embeddings": True,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_config_propagates_dims():
    from archs.nano_gpt.size import build_config

    cfg = build_config(_ns(n_embd=192, n_layer=6, n_head=6, n_positions=512))
    assert cfg.n_embd == 192
    assert cfg.n_layer == 6
    assert cfg.n_head == 6
    assert cfg.n_positions == 512


def test_build_config_propagates_vocab_and_special_tokens():
    from archs.nano_gpt.size import build_config

    cfg = build_config(_ns(vocab_size=128, bos_token_id=1, eos_token_id=2,
                           pad_token_id=0))
    assert cfg.vocab_size == 128
    assert cfg.bos_token_id == 1
    assert cfg.eos_token_id == 2
    assert cfg.pad_token_id == 0


def test_build_config_preserves_tie_word_embeddings_flag():
    from archs.nano_gpt.size import build_config

    cfg_tied = build_config(_ns(tie_word_embeddings=True))
    cfg_untied = build_config(_ns(tie_word_embeddings=False))
    assert cfg_tied.tie_word_embeddings is True
    assert cfg_untied.tie_word_embeddings is False


def test_build_config_n_inner_none_falls_through_to_hf_default():
    """`n_inner=None` is HF-shorthand for "use 4 * n_embd". The size.py CLI
    keeps None as the default so callers don't have to know that magic; we
    pin that GPT2Config preserves it (HF computes 4*n_embd lazily inside
    forward). If HF ever changes the None semantics our sizer would silently
    misreport FF param counts.
    """
    from archs.nano_gpt.size import build_config

    cfg = build_config(_ns(n_inner=None))
    assert cfg.n_inner is None


# ---------------------------------------------------------------------
# size._classify — substring buckets for named_parameters() walks.
#
# GPT-2's parameter paths look like:
#   transformer.wte.weight                              # token embedding
#   transformer.wpe.weight                              # position embedding
#   transformer.h.0.ln_1.weight                         # pre-attn layernorm
#   transformer.h.0.attn.c_attn.weight                  # fused QKV proj
#   transformer.h.0.attn.c_proj.weight                  # attn output proj
#   transformer.h.0.mlp.c_fc.weight                     # FF up-proj
#   transformer.h.0.mlp.c_proj.weight                   # FF down-proj
#   lm_head.weight                                      # output head
# `count_params` calls `_classify(name)` on each. Pin first-match-wins so
# an accidental reorder doesn't shift bucket totals.

@pytest.mark.parametrize("name,bucket", [
    ("transformer.wte.weight", "embed"),
    ("transformer.wpe.weight", "pos_embed"),
    ("lm_head.weight", "lm_head"),
    ("transformer.h.0.attn.c_attn.weight", "attn"),
    ("transformer.h.0.attn.c_proj.weight", "attn"),
    ("transformer.h.0.mlp.c_fc.weight", "mlp"),
    ("transformer.h.0.mlp.c_proj.weight", "mlp"),
    ("transformer.h.0.ln_1.weight", "norm"),
    ("transformer.h.0.ln_2.weight", "norm"),
    ("transformer.ln_f.weight", "norm"),
    ("some.future.unknown.param", "other"),
])
def test_classify_buckets_each_param_path(name, bucket):
    from archs.nano_gpt.size import _classify

    assert _classify(name) == bucket


def test_classify_lowercases_so_capitalized_paths_still_match():
    """`_classify` calls `name.lower()` first. Pin so a future caller that
    feeds capitalized paths (e.g. some traced/exported model variant) still
    classifies correctly."""
    from archs.nano_gpt.size import _classify

    assert _classify("Transformer.Wte.Weight") == "embed"
    assert _classify("LM_HEAD.weight") == "lm_head"
