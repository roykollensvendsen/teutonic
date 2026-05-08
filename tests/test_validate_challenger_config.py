"""Tests for validator.validate_challenger_config.

The function gates challenger uploads before they're accepted as a new
king. It downloads config.json for both king and challenger from
HuggingFace, compares architecture/vocab, and checks the repo contents
for safetensors files and security violations (no *.py).

These tests mock HfApi to avoid network access and to construct
specific failure scenarios.
"""
import json

import pytest

from validator import validate_challenger_config


@pytest.fixture(autouse=True)
def _reset_king_config_cache():
    """Clear validator's process-wide king config cache between tests.

    `get_king_config` caches by (repo, revision) — without resetting,
    the first test to populate the cache for KING_REPO@KING_REV
    poisons every later test that varies the king config under that
    same key.
    """
    import validator as v
    v._king_config = None
    v._king_config_key = None
    yield
    v._king_config = None
    v._king_config_key = None


@pytest.fixture
def fake_hf(mocker, tmp_path):
    """Mock validator.HfApi.

    Returns a setup function: setup(repo, revision, config=..., files=...,
    sizes=...) pre-registers what hf_hub_download / list_repo_files /
    repo_info should return for that (repo, revision) pair.
    `sizes` is optional `{filename: byte_count}` for the size pre-flight
    (introduced upstream 2026-05-08); when omitted, repo_info siblings
    return size 0 (skip-size-check semantics).
    """
    repo_state = {}  # (repo, revision) -> {"config": ..., "files": [...], "sizes": {...}}

    def hf_hub_download(repo_id, filename, revision=None, **_kwargs):
        key = (repo_id, revision)
        if key not in repo_state:
            raise FileNotFoundError(f"No fake state for {key}")
        if filename != "config.json":
            raise FileNotFoundError(f"Only config.json is faked, got {filename}")
        path = tmp_path / f"{repo_id.replace('/', '_')}_{revision}_config.json"
        path.write_text(json.dumps(repo_state[key]["config"]))
        return str(path)

    def list_repo_files(repo_id, revision=None, **_kwargs):
        return repo_state[(repo_id, revision)]["files"]

    def repo_info(repo_id, revision=None, files_metadata=False, **_kwargs):
        key = (repo_id, revision)
        if key not in repo_state:
            raise FileNotFoundError(f"No fake state for {key}")
        from types import SimpleNamespace
        files = repo_state[key]["files"]
        sizes = repo_state[key].get("sizes", {})
        siblings = [SimpleNamespace(rfilename=f, size=sizes.get(f, 0))
                    for f in files]
        return SimpleNamespace(siblings=siblings)

    api_instance = mocker.MagicMock()
    api_instance.hf_hub_download.side_effect = hf_hub_download
    api_instance.list_repo_files.side_effect = list_repo_files
    api_instance.repo_info.side_effect = repo_info
    mocker.patch("validator.HfApi", return_value=api_instance)

    def setup(repo, revision, *, config, files, sizes=None):
        repo_state[(repo, revision)] = {
            "config": config, "files": files,
            "sizes": sizes or {},
        }

    return setup


KING_REPO = "unconst/king"
KING_REV = "abc123"
CHALLENGER_REPO = "miner/challenger"
CHALLENGER_REV = "def456"


def _matching_config():
    return {
        "architectures": ["QuasarForCausalLM"],
        "vocab_size": 50304,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
    }


def _matching_files():
    return ["config.json", "model.safetensors", "tokenizer.json"]


def test_validate_returns_none_for_matching_challenger(fake_hf):
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV, config=_matching_config(), files=_matching_files())

    rejection = validate_challenger_config(CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)

    assert rejection is None


def test_validate_rejects_architecture_mismatch(fake_hf):
    challenger_cfg = _matching_config()
    challenger_cfg["architectures"] = ["GPTNeoForCausalLM"]
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV, config=challenger_cfg, files=_matching_files())

    rejection = validate_challenger_config(CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)

    assert isinstance(rejection, str)
    assert "arch" in rejection.lower()


def test_validate_rejects_vocab_size_mismatch(fake_hf):
    challenger_cfg = _matching_config()
    challenger_cfg["vocab_size"] = 32000  # different from king's 50304
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV, config=challenger_cfg, files=_matching_files())

    rejection = validate_challenger_config(CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)

    assert isinstance(rejection, str)
    assert "vocab" in rejection.lower()


def test_validate_rejects_missing_safetensors(fake_hf):
    no_safetensors = ["config.json", "tokenizer.json"]  # safetensors absent
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV, config=_matching_config(), files=no_safetensors)

    rejection = validate_challenger_config(CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)

    assert isinstance(rejection, str)
    assert "safetensors" in rejection.lower()


def test_validate_rejects_python_file_upload(fake_hf):
    # Module-level comment in validator.py: "we deny *.py uploads in
    # validate_challenger_config" — *.py would let a miner ship arbitrary
    # code that gets executed via auto_map.
    with_py = _matching_files() + ["modeling_custom.py"]
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV, config=_matching_config(), files=with_py)

    rejection = validate_challenger_config(CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)

    assert isinstance(rejection, str)
    assert ".py" in rejection.lower() or "python" in rejection.lower()


def test_validate_rejects_auto_map_field(fake_hf):
    # auto_map = HF dynamic-import dispatch. Even though we never set
    # trust_remote_code=True downstream, a defence-in-depth gate refuses
    # uploads that would attempt remote-code loading on consumer dispatch.
    challenger_cfg = _matching_config()
    challenger_cfg["auto_map"] = {"AutoConfig": "configuration_custom.X"}
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV, config=challenger_cfg, files=_matching_files())

    rejection = validate_challenger_config(CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)

    assert isinstance(rejection, str)
    assert "auto_map" in rejection


def test_validate_rejects_extra_lock_key_mismatch(fake_hf, monkeypatch):
    # chain.toml's [arch].extra_lock_keys names per-arch fields that must
    # also match (e.g. d_model, num_routed_experts). A challenger that
    # silently flips one would otherwise pass the generic-key check.
    import chain_config as cc

    monkeypatch.setattr(cc, "EXTRA_LOCK_KEYS", ("d_model",))

    king_cfg = _matching_config()
    king_cfg["d_model"] = 4096
    challenger_cfg = _matching_config()
    challenger_cfg["d_model"] = 2048
    fake_hf(KING_REPO, KING_REV, config=king_cfg, files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV, config=challenger_cfg, files=_matching_files())

    rejection = validate_challenger_config(CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)

    assert isinstance(rejection, str)
    assert "d_model" in rejection


def test_validate_returns_none_when_king_cfg_unfetchable(mocker):
    # If get_king_config returns falsy (HF lookup of king failed), the
    # function returns None — challenger is *not* rejected on a stale king
    # snapshot. This keeps the validator from rejecting everything when
    # the king-side HF endpoint blips. Returns None before challenger is
    # ever fetched, so no fake_hf state is needed.
    mocker.patch("validator.get_king_config", return_value={})

    rejection = validate_challenger_config(CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)

    assert rejection is None


def test_validate_returns_error_when_challenger_config_unfetchable(fake_hf):
    # If hf_hub_download for the challenger raises (network blip,
    # repo gone, revision missing), surface the error rather than
    # silently accepting the challenger.
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    # Note: deliberately do NOT register CHALLENGER_REPO@CHALLENGER_REV
    # so the fake's hf_hub_download raises FileNotFoundError.

    rejection = validate_challenger_config(CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)

    assert isinstance(rejection, str)
    assert "config.json" in rejection.lower()


# ---------------------------------------------------------------------
# Pre-flight checks (upstream 86ab8dd + 4240f44, M36):
#
# The validator pre-flights the challenger's safetensors layout and
# total size BEFORE dispatching the eval, to fail fast instead of
# burning ~5-15 min downloading 165 GB only to crash with the
# misleading "could not load model with any attention implementation"
# error. Two checks:
#
# 1. Naming: from_pretrained(use_safetensors=True) requires either
#    `model.safetensors` (single-shard) OR
#    `model.safetensors.index.json` + `model-NNNNN-of-NNNNN.safetensors`
#    shards (sharded with index). Anything else is rejected up-front.
#
# 2. Size: total .safetensors bytes (from repo_info.siblings) capped at
#    `TEUTONIC_MAX_CHALLENGER_SAFETENSORS_GB` (default 200 GB). Normal
#    Qwen3MoE 80B in bfloat16 is ~154 GB; cap leaves 30 percent headroom
#    while killing fp32 / duplicated / optimizer-state busts.

# --- Naming layout: accepted forms.

def test_validate_accepts_single_shard_layout(fake_hf):
    files = ["config.json", "model.safetensors", "tokenizer.json"]
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV,
            config=_matching_config(), files=files)
    rejection = validate_challenger_config(
        CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)
    assert rejection is None


def test_validate_accepts_sharded_with_index(fake_hf):
    files = [
        "config.json", "tokenizer.json",
        "model.safetensors.index.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV,
            config=_matching_config(), files=files)
    rejection = validate_challenger_config(
        CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)
    assert rejection is None


# --- Naming layout: rejected forms (live failures from commit messages).

def test_validate_rejects_sharded_without_index_json(fake_hf):
    # Caught live 2026-05-08 13:26 UTC (silvanus97930/...-vv-07):
    # shards uploaded without model.safetensors.index.json — miner
    # must add the index file.
    files = [
        "config.json", "tokenizer.json",
        "model-00001-of-00003.safetensors",
        "model-00002-of-00003.safetensors",
        "model-00003-of-00003.safetensors",
    ]
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV,
            config=_matching_config(), files=files)
    rejection = validate_challenger_config(
        CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)
    assert isinstance(rejection, str)
    assert "index.json" in rejection.lower()


def test_validate_rejects_non_canonical_safetensors_naming(fake_hf):
    # Caught live 2026-05-08 10:47 UTC (seed429/...-b4a9f514): a non-
    # canonical filename like `weights.safetensors` passes the
    # "no .safetensors files" check but transformers can't discover it.
    files = ["config.json", "tokenizer.json", "weights.safetensors"]
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV,
            config=_matching_config(), files=files)
    rejection = validate_challenger_config(
        CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)
    assert isinstance(rejection, str)
    # Rejection message lists the canonical layout choices for the miner.
    assert "canonical" in rejection.lower() or "model.safetensors" in rejection


def test_validate_rejects_shard_layout_with_off_pattern_filenames(fake_hf):
    # `pytorch_model-00001-of-00003.safetensors` and similar look
    # shard-y but don't match the strict `model-NNNNN-of-NNNNN`
    # pattern transformers expects.
    files = [
        "config.json", "tokenizer.json",
        "pytorch_model-00001-of-00003.safetensors",
        "pytorch_model-00002-of-00003.safetensors",
    ]
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV,
            config=_matching_config(), files=files)
    rejection = validate_challenger_config(
        CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)
    assert isinstance(rejection, str)


# --- Size pre-flight.

_GB = 1_000_000_000


def test_validate_accepts_normal_size(fake_hf):
    # Normal Qwen3MoE 80B in bfloat16 ~154 GB → well under 200 GB cap.
    files = ["config.json", "model.safetensors", "tokenizer.json"]
    sizes = {"model.safetensors": int(154 * _GB)}
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV,
            config=_matching_config(), files=files, sizes=sizes)
    rejection = validate_challenger_config(
        CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)
    assert rejection is None


def test_validate_rejects_oversized_safetensors(fake_hf):
    # Caught live 2026-05-08 10:54 UTC (silvanus97930/...-vv-05):
    # 293 GB on disk → ENOSPC mid-load.
    files = ["config.json", "model.safetensors", "tokenizer.json"]
    sizes = {"model.safetensors": int(293 * _GB)}
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV,
            config=_matching_config(), files=files, sizes=sizes)
    rejection = validate_challenger_config(
        CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)
    assert isinstance(rejection, str)
    assert "oversized" in rejection.lower() or "200" in rejection


def test_validate_size_cap_overridable_via_env(fake_hf, monkeypatch):
    # TEUTONIC_MAX_CHALLENGER_SAFETENSORS_GB lets ops raise/lower the cap.
    monkeypatch.setenv("TEUTONIC_MAX_CHALLENGER_SAFETENSORS_GB", "300")
    files = ["config.json", "model.safetensors", "tokenizer.json"]
    sizes = {"model.safetensors": int(250 * _GB)}  # under raised cap
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV,
            config=_matching_config(), files=files, sizes=sizes)
    rejection = validate_challenger_config(
        CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)
    assert rejection is None


def test_validate_size_cap_lowered_rejects_normal_size(fake_hf, monkeypatch):
    # Sanity check the env override actually takes effect: lower the
    # cap below the normal size and confirm the same model is rejected.
    monkeypatch.setenv("TEUTONIC_MAX_CHALLENGER_SAFETENSORS_GB", "100")
    files = ["config.json", "model.safetensors", "tokenizer.json"]
    sizes = {"model.safetensors": int(154 * _GB)}
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV,
            config=_matching_config(), files=files, sizes=sizes)
    rejection = validate_challenger_config(
        CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)
    assert isinstance(rejection, str)
    assert "oversized" in rejection.lower() or "100" in rejection


def test_validate_size_check_sums_sharded_files(fake_hf):
    # Sharded models: cap applies to total (sum across shards), not
    # per-shard. 4 × 50 GB = 200 GB total — exactly at cap, must NOT
    # reject (cap is `>`, not `>=`).
    files = [
        "config.json", "tokenizer.json",
        "model.safetensors.index.json",
    ] + [f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5)]
    sizes = {f"model-{i:05d}-of-00004.safetensors": int(50 * _GB)
             for i in range(1, 5)}
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV,
            config=_matching_config(), files=files, sizes=sizes)
    rejection = validate_challenger_config(
        CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)
    assert rejection is None  # 200 GB total exactly at cap; not rejected


def test_validate_size_check_skipped_on_repo_info_exception(fake_hf, mocker):
    # Spec: if `repo_info(files_metadata=True)` raises (e.g. transient
    # HF API error), the size check is skipped — the eval is allowed
    # to proceed. Logged at WARNING. Failing closed here would let a
    # transient HF glitch block all coronations.
    files = ["config.json", "model.safetensors", "tokenizer.json"]
    fake_hf(KING_REPO, KING_REV, config=_matching_config(), files=_matching_files())
    fake_hf(CHALLENGER_REPO, CHALLENGER_REV,
            config=_matching_config(), files=files)
    # Override the api fixture's repo_info to raise.
    import validator
    api = validator.HfApi.return_value
    api.repo_info.side_effect = RuntimeError("HF api transient")
    rejection = validate_challenger_config(
        CHALLENGER_REPO, CHALLENGER_REV, KING_REPO, KING_REV)
    assert rejection is None
