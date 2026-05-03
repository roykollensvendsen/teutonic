"""Tests for miner-side validation helpers in miner.py.

`validate_local_config` is the miner-side equivalent of
`validator.validate_challenger_config`: it sanity-checks a challenger
checkpoint locally before the miner pushes it to HuggingFace, so the
miner doesn't waste an upload on a config the validator would reject.
"""
import json

import pytest

from miner import sha256_dir, validate_local_config


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


# sha256_dir(path) — content hash of all *.safetensors files in `path`,
# read in sorted order. Used by miners to commit a hash on-chain
# matching what the validator computes for the king.

def test_sha256_dir_returns_64_char_hex(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    digest = sha256_dir(str(tmp_path))

    assert isinstance(digest, str)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


def test_sha256_dir_is_deterministic(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    assert sha256_dir(str(tmp_path)) == sha256_dir(str(tmp_path))


def test_sha256_dir_distinct_content_yields_distinct_hash(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"weights-A")
    digest_a = sha256_dir(str(tmp_path))
    (tmp_path / "model.safetensors").write_bytes(b"weights-B")
    digest_b = sha256_dir(str(tmp_path))
    assert digest_a != digest_b


def test_sha256_dir_ignores_non_safetensors_files(tmp_path):
    # README, config.json, etc. should not affect the hash — only
    # .safetensors content matters.
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    digest_before = sha256_dir(str(tmp_path))

    (tmp_path / "README.md").write_text("docs")
    (tmp_path / "config.json").write_text('{"x": 1}')
    digest_after = sha256_dir(str(tmp_path))

    assert digest_before == digest_after


def test_sha256_dir_combines_multiple_safetensors_in_sorted_order(tmp_path):
    # The miner and validator must agree on hash regardless of
    # filesystem ordering, so the impl sorts. Two equivalent dirs
    # built in different orders should produce the same hash.
    src1 = tmp_path / "src1"
    src2 = tmp_path / "src2"
    src1.mkdir()
    src2.mkdir()
    # src1: write a then b
    (src1 / "a.safetensors").write_bytes(b"chunk-A")
    (src1 / "b.safetensors").write_bytes(b"chunk-B")
    # src2: write b then a
    (src2 / "b.safetensors").write_bytes(b"chunk-B")
    (src2 / "a.safetensors").write_bytes(b"chunk-A")

    assert sha256_dir(str(src1)) == sha256_dir(str(src2))
