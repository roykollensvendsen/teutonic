"""Tests for miner-side validation helpers in miner.py.

`validate_local_config` is the miner-side equivalent of
`validator.validate_challenger_config`: it sanity-checks a challenger
checkpoint locally before the miner pushes it to HuggingFace, so the
miner doesn't waste an upload on a config the validator would reject.
"""
import json

import pytest

from miner import validate_local_config


def _write_config(d, **fields):
    (d / "config.json").write_text(json.dumps(fields))


def _write_safetensors(d, name="model.safetensors"):
    """Touch a fake safetensors file. Tests don't need its content."""
    (d / name).write_bytes(b"")


def _make_valid_challenger(d, **field_overrides):
    """Build a minimal valid challenger directory: config + safetensors."""
    fields = _matching_fields()
    fields.update(field_overrides)
    _write_config(d, **fields)
    _write_safetensors(d)


def _matching_fields():
    return {
        "architectures": ["QuasarForCausalLM"],
        "vocab_size": 50304,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
    }


@pytest.fixture
def king_chall_dirs(tmp_path):
    king = tmp_path / "king"
    chall = tmp_path / "chall"
    king.mkdir()
    chall.mkdir()
    return king, chall


def test_validate_local_config_matching_returns_none(king_chall_dirs):
    king, chall = king_chall_dirs
    _write_config(king, **_matching_fields())
    _make_valid_challenger(chall)

    assert validate_local_config(str(king), str(chall)) is None


def test_validate_local_config_missing_king_returns_none(king_chall_dirs):
    # Per docstring: can't validate without king config — returns None
    # so miner doesn't refuse to publish before the king is set up.
    king, chall = king_chall_dirs
    _make_valid_challenger(chall)

    assert validate_local_config(str(king), str(chall)) is None


def test_validate_local_config_arch_mismatch_rejects(king_chall_dirs):
    king, chall = king_chall_dirs
    _write_config(king, **_matching_fields())
    _make_valid_challenger(chall, architectures=["GPTNeoForCausalLM"])

    rejection = validate_local_config(str(king), str(chall))
    assert isinstance(rejection, str)
    assert "arch" in rejection.lower()


def test_validate_local_config_vocab_mismatch_rejects(king_chall_dirs):
    king, chall = king_chall_dirs
    _write_config(king, **_matching_fields())
    _make_valid_challenger(chall, vocab_size=32000)

    rejection = validate_local_config(str(king), str(chall))
    assert isinstance(rejection, str)
    assert "vocab" in rejection.lower()


def test_validate_local_config_missing_challenger_config(king_chall_dirs):
    king, chall = king_chall_dirs
    _write_config(king, **_matching_fields())
    # chall has no config.json (and no safetensors)

    rejection = validate_local_config(str(king), str(chall))
    assert isinstance(rejection, str)


def test_validate_local_config_missing_safetensors_rejects(king_chall_dirs):
    # Discovered during testing: validate_local_config also checks the
    # challenger directory contains at least one .safetensors file.
    king, chall = king_chall_dirs
    _write_config(king, **_matching_fields())
    _write_config(chall, **_matching_fields())
    # Note: no .safetensors written.

    rejection = validate_local_config(str(king), str(chall))
    assert isinstance(rejection, str)
    assert "safetensors" in rejection.lower()


def test_validate_local_config_python_file_rejects(king_chall_dirs):
    # Mirrors the security rule in validator.validate_challenger_config:
    # *.py uploads would let a miner ship arbitrary code via auto_map.
    king, chall = king_chall_dirs
    _write_config(king, **_matching_fields())
    _make_valid_challenger(chall)
    (chall / "modeling_custom.py").write_text("# evil code")

    rejection = validate_local_config(str(king), str(chall))
    assert isinstance(rejection, str)
    assert ".py" in rejection.lower() or "python" in rejection.lower()
