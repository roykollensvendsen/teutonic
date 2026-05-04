"""Tests for archs.quasar.seed.build_config.

`build_config()` is the seed-time QuasarConfig factory. Unlike the
sizing-script `archs.quasar.size.build_config(args)`, it takes no
arguments — every field is read from module-level constants that
themselves snapshot `TEUTONIC_SEED_*` env vars at import time.

The tests monkeypatch the module-level constants directly (rather
than mutating env + reloading) so each case isolates a single field
without paying the import cost of torch + transformers + chain_config
on every parameterization.
"""
from archs.quasar import seed


def test_build_config_uses_module_constants_for_dims(monkeypatch):
    monkeypatch.setattr(seed, "HIDDEN_SIZE", 2048)
    monkeypatch.setattr(seed, "NUM_LAYERS", 16)
    monkeypatch.setattr(seed, "D_FF", 8192)
    cfg = seed.build_config()
    assert cfg.d_model == 2048
    assert cfg.n_layers == 16
    assert cfg.d_ff == 8192


def test_build_config_hardcodes_invariants():
    # These are not env-overridable and must stay pinned: the seed
    # ships a tied-embedding BigMac MoE with the Gemma3 token IDs.
    cfg = seed.build_config()
    assert cfg.tie_word_embeddings is True
    assert cfg.moe_type == "bigmac"
    assert cfg.pad_token_id == 0
    assert cfg.eos_token_id == 1
    assert cfg.bos_token_id == 2


def test_build_config_kv_heads_mirrors_n_heads(monkeypatch):
    # NUM_HEADS feeds both n_heads and num_key_value_heads (MHA).
    monkeypatch.setattr(seed, "NUM_HEADS", 24)
    cfg = seed.build_config()
    assert cfg.n_heads == 24
    assert cfg.num_key_value_heads == 24


def test_build_config_propagates_moe_fields(monkeypatch):
    monkeypatch.setattr(seed, "NUM_ROUTED_EXPERTS", 32)
    monkeypatch.setattr(seed, "NUM_SHARED_EXPERTS", 2)
    monkeypatch.setattr(seed, "TOP_K", 6)
    monkeypatch.setattr(seed, "ROUTED_EXPERT_SIZE", 1024)
    monkeypatch.setattr(seed, "SHARED_EXPERT_SIZE", 3072)
    monkeypatch.setattr(seed, "BIGMAC_R", 0.5)
    cfg = seed.build_config()
    assert cfg.num_routed_experts == 32
    assert cfg.num_shared_experts == 2
    assert cfg.top_k == 6
    assert cfg.routed_expert_size == 1024
    assert cfg.shared_expert_size == 3072
    assert cfg.bigmac_r == 0.5


def test_build_config_sets_architectures_attribute():
    # build_config attaches `architectures = ["QuasarForCausalLM"]` so
    # AutoModelForCausalLM dispatches to the vendored class without
    # trust_remote_code on consumer load.
    cfg = seed.build_config()
    assert cfg.architectures == ["QuasarForCausalLM"]
