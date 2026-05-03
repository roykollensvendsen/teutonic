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


@pytest.fixture
def fake_hf(mocker, tmp_path):
    """Mock validator.HfApi.

    Returns a setup function: setup(repo, revision, config=..., files=...)
    pre-registers what hf_hub_download and list_repo_files should return
    for that (repo, revision) pair.
    """
    repo_state = {}  # (repo, revision) -> {"config": <dict>, "files": [<str>]}

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

    api_instance = mocker.MagicMock()
    api_instance.hf_hub_download.side_effect = hf_hub_download
    api_instance.list_repo_files.side_effect = list_repo_files
    mocker.patch("validator.HfApi", return_value=api_instance)

    def setup(repo, revision, *, config, files):
        repo_state[(repo, revision)] = {"config": config, "files": files}

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
