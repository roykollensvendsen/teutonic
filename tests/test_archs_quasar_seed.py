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

import pytest

from archs.quasar import seed

pytestmark = pytest.mark.arch_specific("archs.quasar")


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
    # The on-disk result must round-trip through json.load and contain
    # the expected post-strip payload. The exact formatting (indent /
    # key order) is impl detail — only the parsed value is contract.
    out_dir = _write_config(tmp_path, {
        "model_type": "quasar",
        "auto_map": {"AutoConfig": "x"},
        "n_layers": 8,
    })
    seed._strip_auto_map(out_dir)
    text = (out_dir / "config.json").read_text()
    assert json.loads(text) == {"model_type": "quasar", "n_layers": 8}


# ---------------------------------------------------------------------
# _count_active(model, cfg) — return (total, active_per_token).
# Mirrors archs/quasar/size.py count_params for the active-param math:
# routed MoE experts contribute `numel/num_experts * top_k` to active.

class _FakeParam:
    """Stand-in for a torch.nn.Parameter: only numel() + shape used."""
    def __init__(self, numel: int, shape=None):
        self._numel = numel
        self.shape = shape if shape is not None else (numel,)
    def numel(self) -> int:
        return self._numel


class _FakeModel:
    """Model with a controllable named_parameters() yield."""
    def __init__(self, params):
        self._params = list(params)
    def named_parameters(self):
        return iter(self._params)


def _cfg(*, tie_word_embeddings: bool = True, top_k: int = 8):
    """Tiny QuasarConfig stand-in — only fields _count_active reads."""
    from types import SimpleNamespace
    return SimpleNamespace(tie_word_embeddings=tie_word_embeddings, top_k=top_k)


def test_count_active_total_equals_active_for_dense_model():
    # No MoE routed experts → every parameter counts fully toward active.
    model = _FakeModel([
        ("layers.0.attn.q_proj.weight", _FakeParam(100)),
        ("layers.0.ffn.gate.weight", _FakeParam(200)),
    ])
    total, active = seed._count_active(model, _cfg(tie_word_embeddings=False))
    assert total == 300
    assert active == 300


def test_count_active_routed_experts_apply_topk_factor():
    # 80 routed experts, top_k=10 → active = (numel/80)*10 = numel/8.
    model = _FakeModel([
        ("layers.0.moe.experts_w12", _FakeParam(numel=8000, shape=(80, 100))),
    ])
    total, active = seed._count_active(model, _cfg(tie_word_embeddings=False, top_k=10))
    assert total == 8000
    assert active == 1000  # 8000/80 * 10


def test_count_active_handles_experts_w3_same_as_w12():
    # Both `experts_w12` and `experts_w3` are routed-expert weights;
    # the topk-factor applies identically.
    model = _FakeModel([
        ("layers.0.moe.experts_w3", _FakeParam(numel=4000, shape=(40, 100))),
    ])
    total, active = seed._count_active(model, _cfg(tie_word_embeddings=False, top_k=8))
    assert total == 4000
    assert active == 800  # 4000/40 * 8


def test_count_active_skips_lm_head_when_tied():
    model = _FakeModel([
        ("model.embed_tokens.weight", _FakeParam(50000)),
        ("lm_head.weight", _FakeParam(50000)),
    ])
    total, _active = seed._count_active(model, _cfg(tie_word_embeddings=True))
    assert total == 50000  # lm_head dropped


def test_count_active_keeps_lm_head_when_not_tied():
    model = _FakeModel([
        ("model.embed_tokens.weight", _FakeParam(50000)),
        ("lm_head.weight", _FakeParam(50000)),
    ])
    total, _active = seed._count_active(model, _cfg(tie_word_embeddings=False))
    assert total == 100000


def test_count_active_dedups_repeated_embed_tokens_when_tied():
    model = _FakeModel([
        ("model.embed_tokens.weight", _FakeParam(50000)),
        ("model.embed_tokens.weight", _FakeParam(50000)),
    ])
    total, _active = seed._count_active(model, _cfg(tie_word_embeddings=True))
    assert total == 50000  # second occurrence skipped
