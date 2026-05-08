"""Bundled tests for small pure surfaces across modules.

Each function/constant here is too small to warrant its own test file;
bundled together with one commit per logical group inside the PR.
"""
import struct

import numpy as np
import pytest

from eval.torch_runner import extract_sequences, math_isfinite
from validator import _REPO_RE, REPO_PATTERN, parse_args

# `extract_sequences(shard_data, data_offset, indices, seq_len)` —
# docstring: "Extract sequences from a locally-cached shard."
# Each sequence is seq_len * 4 bytes (uint32 little-endian tokens).
# Returns dict mapping idx -> list of token ids.

def _build_shard(num_sequences: int, seq_len: int, header_size: int = 8) -> bytes:
    """Build fake shard bytes: header_size pad + num_sequences * seq_len uint32 tokens.
    Token at sequence i, position j = i * 1000 + j (deterministic, easy to check)."""
    header = b"\x00" * header_size
    body = bytearray()
    for i in range(num_sequences):
        for j in range(seq_len):
            body.extend(struct.pack("<I", i * 1000 + j))
    return header + bytes(body)


def test_extract_sequences_returns_dict():
    shard = _build_shard(num_sequences=3, seq_len=4)
    result = extract_sequences(shard, data_offset=8, indices=[0], seq_len=4)
    assert isinstance(result, dict)


def test_extract_sequences_correct_tokens_for_known_layout():
    # Sequence 0 has tokens [0, 1, 2, 3]; sequence 2 has [2000, 2001, 2002, 2003].
    shard = _build_shard(num_sequences=3, seq_len=4)
    result = extract_sequences(shard, data_offset=8, indices=[0, 2], seq_len=4)
    assert result[0] == [0, 1, 2, 3]
    assert result[2] == [2000, 2001, 2002, 2003]


def test_extract_sequences_skips_indices_not_requested():
    shard = _build_shard(num_sequences=3, seq_len=4)
    result = extract_sequences(shard, data_offset=8, indices=[1], seq_len=4)
    assert set(result.keys()) == {1}


def test_extract_sequences_empty_indices_returns_empty_dict():
    shard = _build_shard(num_sequences=3, seq_len=4)
    result = extract_sequences(shard, data_offset=8, indices=[], seq_len=4)
    assert result == {}


def test_extract_sequences_each_sequence_has_seq_len_tokens():
    shard = _build_shard(num_sequences=2, seq_len=10)
    result = extract_sequences(shard, data_offset=8, indices=[0, 1], seq_len=10)
    assert all(len(seq) == 10 for seq in result.values())


# `math_isfinite(x: float) -> bool` — module-local isfinite that
# handles bf16 .float() outputs.

def test_math_isfinite_normal_floats():
    assert math_isfinite(0.0) is True
    assert math_isfinite(1.5) is True
    assert math_isfinite(-1e10) is True


def test_math_isfinite_rejects_nan():
    assert math_isfinite(float("nan")) is False


def test_math_isfinite_rejects_inf():
    assert math_isfinite(float("inf")) is False
    assert math_isfinite(float("-inf")) is False


def test_math_isfinite_handles_numpy_floats():
    # bf16 .float() outputs are typically numpy/torch floats — must
    # work the same as native floats. Use truthiness (not `is True`)
    # because numpy may return np.True_ / np.False_ rather than
    # python bools.
    assert math_isfinite(np.float32(2.5))
    assert not math_isfinite(np.float64("nan"))


# `REPO_PATTERN` — regex r"^[^/]+/Teutonic-LXXX-.+$"
# Validates HF repo names: <user>/Teutonic-LXXX-<anything>
# Compiled into _REPO_RE at module load.

@pytest.mark.parametrize("repo", [
    "user1/Teutonic-LXXX-abc",
    "miner/Teutonic-LXXX-v2-roy",
    "org-with-dash/Teutonic-LXXX-1",
])
def test_repo_pattern_accepts_valid_names(repo):
    assert _REPO_RE.match(repo) is not None


@pytest.mark.parametrize("repo", [
    "Teutonic-LXXX-abc",                  # missing user/
    "user/Teutonic-XXIII-abc",            # wrong version
    "user/Teutonic-LXXX-",                # empty suffix
    "user/RandomRepo",                    # not Teutonic-LXXX
    "user/sub/Teutonic-LXXX-abc",         # extra slash
    "",
])
def test_repo_pattern_rejects_invalid_names(repo):
    assert _REPO_RE.match(repo) is None


def test_repo_pattern_constant_matches_compiled_regex():
    # _REPO_RE is just re.compile(REPO_PATTERN); verify they agree.
    import re
    direct = re.compile(REPO_PATTERN)
    sample = "user/Teutonic-LXXX-test"
    assert (direct.match(sample) is not None) == (_REPO_RE.match(sample) is not None)


# `parse_args()` — argparse with --seen flag (BooleanOptionalAction,
# default True). Tested by injecting sys.argv via monkeypatch.

def test_parse_args_default_seen_is_true(monkeypatch):
    monkeypatch.setattr("sys.argv", ["validator"])
    args = parse_args()
    assert args.seen is True


def test_parse_args_no_seen_flag_disables(monkeypatch):
    monkeypatch.setattr("sys.argv", ["validator", "--no-seen"])
    args = parse_args()
    assert args.seen is False


def test_parse_args_explicit_seen_enables(monkeypatch):
    monkeypatch.setattr("sys.argv", ["validator", "--seen"])
    args = parse_args()
    assert args.seen is True
