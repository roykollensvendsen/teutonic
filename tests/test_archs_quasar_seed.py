"""Tests for archs.quasar.seed pure helpers.

* `build_config()` — seed-time QuasarConfig factory. Takes no args;
  every field is read from module-level constants that themselves
  snapshot `TEUTONIC_SEED_*` env vars at import time. The build_config
  tests monkeypatch the module-level constants directly (rather than
  mutating env + reloading) so each case isolates a single field
  without paying the import cost of torch + transformers + chain_config
  on every parameterization.

* `_strip_auto_map(out_dir)` — mutates `config.json` on disk to drop
  the `auto_map` field. The vendored archs/quasar package handles
  loading via plain AutoModelForCausalLM dispatch once imported, so
  `auto_map` would only cause HF to attempt a dynamic import of the
  original silx-ai modules. Tested with a tmp_path-built config.json.
"""
import json

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


# ---------------------------------------------------------------------
# _strip_auto_map — mutate config.json on disk to drop auto_map field.

def _write_config(tmp_path, payload):
    """Helper: write a config.json into tmp_path and return the dir."""
    (tmp_path / "config.json").write_text(json.dumps(payload, indent=2))
    return tmp_path


def test_strip_auto_map_removes_field(tmp_path):
    out_dir = _write_config(tmp_path, {
        "model_type": "quasar",
        "auto_map": {"AutoConfig": "configuration_quasar.QuasarConfig"},
        "d_model": 4096,
    })
    seed._strip_auto_map(out_dir)
    after = json.loads((out_dir / "config.json").read_text())
    assert "auto_map" not in after


def test_strip_auto_map_is_noop_when_field_absent(tmp_path):
    payload = {"model_type": "quasar", "d_model": 4096}
    out_dir = _write_config(tmp_path, payload)
    seed._strip_auto_map(out_dir)
    after = json.loads((out_dir / "config.json").read_text())
    assert after == payload


def test_strip_auto_map_preserves_other_fields(tmp_path):
    out_dir = _write_config(tmp_path, {
        "model_type": "quasar",
        "auto_map": {"AutoConfig": "configuration_quasar.QuasarConfig"},
        "d_model": 4096,
        "n_layers": 32,
        "architectures": ["QuasarForCausalLM"],
        "torch_dtype": "bfloat16",
    })
    seed._strip_auto_map(out_dir)
    after = json.loads((out_dir / "config.json").read_text())
    assert after == {
        "model_type": "quasar",
        "d_model": 4096,
        "n_layers": 32,
        "architectures": ["QuasarForCausalLM"],
        "torch_dtype": "bfloat16",
    }


def test_strip_auto_map_is_idempotent(tmp_path):
    # Running twice on the same file must not corrupt it — second call
    # is the no-op-when-absent case.
    out_dir = _write_config(tmp_path, {
        "model_type": "quasar",
        "auto_map": {"AutoConfig": "x"},
    })
    seed._strip_auto_map(out_dir)
    seed._strip_auto_map(out_dir)
    after = json.loads((out_dir / "config.json").read_text())
    assert after == {"model_type": "quasar"}


def test_strip_auto_map_writes_valid_json(tmp_path):
    # The output must round-trip through json.load — pin the indent=2
    # contract so the on-disk file stays human-diffable when the seed
    # script later re-uploads it.
    out_dir = _write_config(tmp_path, {
        "model_type": "quasar",
        "auto_map": {"AutoConfig": "x"},
        "n_layers": 8,
    })
    seed._strip_auto_map(out_dir)
    text = (out_dir / "config.json").read_text()
    # indent=2 → newline + 2-space prefix on first key.
    assert "\n  " in text
    assert json.loads(text) == {"model_type": "quasar", "n_layers": 8}
