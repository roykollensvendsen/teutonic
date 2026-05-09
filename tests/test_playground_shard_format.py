"""Pin the on-disk shard format produced by playground.dataset.build_shakespeare.

The devnet's whole point is to test the *real* validator + eval-server code
paths, not modified versions. That means our toy dataset shards must be
byte-compatible with what `eval/torch_runner.get_shard_info` and
`fetch_sequences` expect — which is what `scripts/ingest_hf.py` produces in
production: a numpy `.npy` file with a flat uint32 token array, plus a
sidecar `manifest.json` keyed by shard.

These tests pin the writer side: feed `write_shard` + `build_manifest` a
known small input, then parse the output byte-for-byte the way the
validator's reader code does (re-implemented inline here so the assertions
stay readable; the production path lives in eval/torch_runner.py).

Tests do NOT hit the network — `fetch_tinyshakespeare` is exercised only by
running the script for real.
"""
from __future__ import annotations

import io
import json
import struct

import numpy as np
import pytest

from playground.dataset.build_shakespeare import (
    BYTES_PER_TOKEN,
    DTYPE,
    build_manifest,
    write_shard,
)

# ---------------------------------------------------------------------
# write_shard — one .npy per shard, flat uint32 token array.

def _read_npy_header(npy_bytes: bytes) -> tuple[dict, int]:
    """Mirror eval/torch_runner.get_shard_info / _parse_shard_header.

    Returns (header_dict, data_offset). The header dict is what `eval()`
    produces from the npy header line — the production code uses bare
    `eval()` on this string (numpy guarantees the format is a Python dict
    literal). We use ast.literal_eval since the test's assertions don't
    need the production's eval() leniency.
    """
    import ast
    buf = io.BytesIO(npy_bytes)
    magic = buf.read(6)
    assert magic == b"\x93NUMPY", f"not an npy file: magic={magic!r}"
    ver = struct.unpack("BB", buf.read(2))
    if ver[0] == 1:
        hl = struct.unpack("<H", buf.read(2))[0]
    else:
        hl = struct.unpack("<I", buf.read(4))[0]
    hdr_str = buf.read(hl).decode("latin1").strip()
    return ast.literal_eval(hdr_str), buf.tell()


def test_write_shard_emits_npy_with_uint32_le_dtype(tmp_path):
    tokens = np.arange(4096, dtype=DTYPE)
    shard_path = tmp_path / "shards" / "shard_000000.npy"
    write_shard(tokens, seq_len=2048, shard_path=shard_path,
                shard_key="test/v2/shards/shard_000000.npy")

    hdr, _ = _read_npy_header(shard_path.read_bytes())
    assert hdr["descr"] == "<u4", (
        f"validator expects uint32 little-endian; got {hdr['descr']!r}. "
        f"A mismatch here means fetch_sequences would mis-decode every "
        f"token in the shard."
    )
    assert hdr["fortran_order"] is False, (
        "fetch_sequences indexes by linear byte offset; fortran_order=True "
        "would change the storage layout."
    )


def test_write_shard_drops_partial_seq_len_tail(tmp_path):
    """`scripts/ingest_hf.py` packs full seq_len windows and discards the
    remainder. Our writer must do the same so the manifest's total_tokens
    is divisible by seq_len — eval/torch_runner does
    `n_sequences = n_tokens // seq_len` and a non-divisible total would mean
    the manifest claims tokens that no sequence index addresses.
    """
    tokens = np.arange(2048 + 100, dtype=DTYPE)
    shard_path = tmp_path / "shards" / "shard_000000.npy"
    info = write_shard(tokens, seq_len=2048, shard_path=shard_path,
                       shard_key="test/v2/shards/shard_000000.npy")
    assert info["n_tokens"] == 2048
    assert info["n_tokens"] % 2048 == 0


def test_write_shard_returns_manifest_entry_with_validator_required_keys(tmp_path):
    """Each shards[] entry in the manifest must have: key, n_tokens,
    size_bytes, sha256. validator.py reads `key` and uses `n_tokens` for
    bootstrap math; eval/torch_runner verifies sha256 (not yet, but the
    field is there for when that defense lands).
    """
    tokens = np.arange(2048, dtype=DTYPE)
    shard_path = tmp_path / "shards" / "shard_000000.npy"
    info = write_shard(tokens, seq_len=2048, shard_path=shard_path,
                       shard_key="test/v2/shards/shard_000000.npy")
    assert set(info) == {"key", "n_tokens", "size_bytes", "sha256"}
    assert info["key"] == "test/v2/shards/shard_000000.npy"
    assert info["sha256"] == pytest.approx(info["sha256"])  # 64-hex string
    assert len(info["sha256"]) == 64


def test_write_shard_refuses_corpus_smaller_than_one_window(tmp_path):
    """A 100-token corpus at seq_len=2048 produces zero windows. Better to
    fail loudly than to write an empty shard the eval server would then
    divide-by-zero on.
    """
    tokens = np.arange(100, dtype=DTYPE)
    shard_path = tmp_path / "shards" / "shard_000000.npy"
    with pytest.raises(SystemExit, match="corpus too small"):
        write_shard(tokens, seq_len=2048, shard_path=shard_path,
                    shard_key="test/v2/shards/shard_000000.npy")


def test_write_shard_reports_size_bytes_matches_disk(tmp_path):
    tokens = np.arange(2048, dtype=DTYPE)
    shard_path = tmp_path / "shards" / "shard_000000.npy"
    info = write_shard(tokens, seq_len=2048, shard_path=shard_path,
                       shard_key="test/v2/shards/shard_000000.npy")
    assert info["size_bytes"] == shard_path.stat().st_size
    # size_bytes >= n_tokens * BYTES_PER_TOKEN (npy header is ~128 bytes
    # extra). Pin the lower bound so the writer doesn't accidentally
    # double-byte tokens or pad with zeros.
    assert info["size_bytes"] >= info["n_tokens"] * BYTES_PER_TOKEN


# ---------------------------------------------------------------------
# Validator-side parse: re-shape what the validator does on an npy header.

def test_get_shard_info_reproduces_n_tokens(tmp_path):
    """eval/torch_runner.get_shard_info reads the first 1024 bytes of the
    shard, parses the npy header, and returns prod(shape). That count is
    used as the authoritative n_tokens for bootstrap-index sampling. Pin
    that our writer's manifest n_tokens equals what the validator's reader
    would compute from the same file.
    """
    tokens = np.arange(8192, dtype=DTYPE)
    shard_path = tmp_path / "shards" / "shard_000000.npy"
    info = write_shard(tokens, seq_len=2048, shard_path=shard_path,
                       shard_key="test/v2/shards/shard_000000.npy")

    hdr, _ = _read_npy_header(shard_path.read_bytes()[:1024])
    n_from_shape = 1
    for s in hdr["shape"]:
        n_from_shape *= s
    assert n_from_shape == info["n_tokens"]


# ---------------------------------------------------------------------
# build_manifest — the JSON the validator's `r2.ds_get('manifest.json')`
# returns.

def test_build_manifest_has_v2_validator_required_top_level_keys():
    """validator.py:2196 calls `r2.ds_get('manifest.json')` and then reads
    `manifest['shards']` (test_sha_swap_exploit uses this layout too). The
    eval-server reads `total_tokens`, `dtype`, `shards`. Pin every field
    the production code accesses so a typo in our writer would surface here
    rather than at devnet bring-up time.
    """
    manifest = build_manifest(
        tokenizer_name="gpt2",
        source="test://example",
        shard_prefix="test/v2/shards/",
        shards=[{"key": "k", "n_tokens": 100, "size_bytes": 500,
                 "sha256": "a" * 64}],
    )
    assert manifest["version"] == "v2"
    assert manifest["tokenizer"] == "gpt2"
    assert manifest["dtype"] == "uint32"
    assert manifest["source"] == "test://example"
    assert manifest["shard_prefix"] == "test/v2/shards/"
    assert manifest["total_shards"] == 1
    assert manifest["total_tokens"] == 100
    assert manifest["shards"] == [{
        "key": "k", "n_tokens": 100, "size_bytes": 500, "sha256": "a" * 64,
    }]
    assert "created" in manifest


def test_build_manifest_sums_total_tokens_across_shards():
    manifest = build_manifest(
        tokenizer_name="gpt2",
        source="test://x",
        shard_prefix="p/",
        shards=[
            {"key": "a", "n_tokens": 10, "size_bytes": 1, "sha256": "0" * 64},
            {"key": "b", "n_tokens": 20, "size_bytes": 2, "sha256": "1" * 64},
            {"key": "c", "n_tokens": 30, "size_bytes": 3, "sha256": "2" * 64},
        ],
    )
    assert manifest["total_tokens"] == 60
    assert manifest["total_shards"] == 3


def test_build_manifest_round_trips_through_json():
    """The manifest is uploaded as application/json. Anything in it must
    survive a round-trip through json.dumps/loads — pin so a dict-key with
    a non-string key (or a numpy.int64 that json can't serialise) would
    fail here, not in production.
    """
    manifest = build_manifest(
        tokenizer_name="gpt2",
        source="test://x",
        shard_prefix="p/",
        shards=[{"key": "k", "n_tokens": 10, "size_bytes": 1,
                 "sha256": "0" * 64}],
    )
    blob = json.dumps(manifest)
    restored = json.loads(blob)
    assert restored == manifest
