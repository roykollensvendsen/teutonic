"""Tests for eval/torch_runner pure helpers.

Three concerns covered here:

* `extract_sequences(shard_data, data_offset, indices, seq_len)` —
  pure numpy slicing. Reads `seq_len` uint32 tokens at byte offset
  `data_offset + idx * (seq_len * 4)` for each requested index.

* `R2` class in eval/torch_runner.py — boto3-backed S3 wrapper with
  optional dataset store. Smaller than validator's R2 (no Hippius
  dashboard fallback, no append_jsonl). Same defensive get/range_get
  pattern.

* `get_shard_info(r2, shard_key)` — parses the .npy header from R2
  range-get to compute the total token count.

These all run without GPU / without real network — pure
numpy/boto3-mock territory.
"""
import io
import json
from unittest.mock import MagicMock

import numpy as np
import pytest

from eval.torch_runner import R2, extract_sequences, get_shard_info

# ---------------------------------------------------------------------
# extract_sequences — pure numpy slicing.

def _build_shard(seqs: list[list[int]], *, header_offset: int = 0) -> bytes:
    """Build a shard-data bytes buffer from a list of token sequences.

    Each sequence is `seq_len` uint32 little-endian tokens. The
    `header_offset` parameter prefixes `\x00 * header_offset` bytes
    so tests can verify `data_offset` is honoured.
    """
    body_parts: list[bytes] = []
    for seq in seqs:
        body_parts.append(np.array(seq, dtype="<u4").tobytes())
    return b"\x00" * header_offset + b"".join(body_parts)


def test_extract_sequences_empty_indices_returns_empty_dict():
    data = _build_shard([[1, 2, 3], [4, 5, 6]])
    assert extract_sequences(data, 0, [], 3) == {}


def test_extract_sequences_single_index_returns_correct_tokens():
    data = _build_shard([[10, 20, 30, 40], [50, 60, 70, 80]])
    result = extract_sequences(data, 0, [0], 4)
    assert result == {0: [10, 20, 30, 40]}


def test_extract_sequences_multiple_indices_each_keyed_by_index():
    data = _build_shard([[1, 2], [3, 4], [5, 6], [7, 8]])
    result = extract_sequences(data, 0, [0, 2, 3], 2)
    assert result == {0: [1, 2], 2: [5, 6], 3: [7, 8]}


def test_extract_sequences_indices_in_arbitrary_order():
    # Output dict keys reflect the indices passed in; iteration order
    # within Python's dict preserves insertion order.
    data = _build_shard([[1, 2], [3, 4], [5, 6]])
    result = extract_sequences(data, 0, [2, 0, 1], 2)
    assert list(result.keys()) == [2, 0, 1]
    assert result == {0: [1, 2], 1: [3, 4], 2: [5, 6]}


def test_extract_sequences_honours_data_offset():
    # The header_offset bytes are skipped by passing data_offset=N.
    data = _build_shard([[1, 2, 3]], header_offset=128)
    result = extract_sequences(data, 128, [0], 3)
    assert result == {0: [1, 2, 3]}


def test_extract_sequences_returns_python_lists_not_arrays():
    # Caller's downstream code does JSON-encoding; lists serialise.
    data = _build_shard([[42]])
    result = extract_sequences(data, 0, [0], 1)
    assert isinstance(result[0], list)
    assert json.dumps(result) == '{"0": [42]}'


def test_extract_sequences_handles_large_uint32_values():
    # uint32 max = 2**32 - 1; tokens near vocab_size=262144 are fine.
    data = _build_shard([[262143, 0, 4294967295]])
    result = extract_sequences(data, 0, [0], 3)
    assert result == {0: [262143, 0, 4294967295]}


# ---------------------------------------------------------------------
# R2 class — boto3-backed S3 wrapper used by eval-side code.

@pytest.fixture
def fake_eval_r2(mocker, monkeypatch):
    """Mock boto3.client and required env vars so R2() can construct."""
    monkeypatch.setenv("TEUTONIC_R2_ENDPOINT", "https://r2.example/")
    monkeypatch.setenv("TEUTONIC_R2_ACCESS_KEY", "k")
    monkeypatch.setenv("TEUTONIC_R2_SECRET_KEY", "s")
    monkeypatch.setenv("TEUTONIC_R2_BUCKET", "test-bucket")
    monkeypatch.delenv("TEUTONIC_DS_ENDPOINT", raising=False)
    monkeypatch.delenv("TEUTONIC_DS_ACCESS_KEY", raising=False)
    monkeypatch.delenv("TEUTONIC_DS_SECRET_KEY", raising=False)

    clients = [MagicMock(), MagicMock()]
    iter_clients = iter(clients)
    mocker.patch("eval.torch_runner.boto3.client",
                 side_effect=lambda *a, **kw: next(iter_clients))
    return clients


def _body(payload: bytes) -> dict:
    body = MagicMock()
    body.read.return_value = payload
    return {"Body": body}


def test_r2_init_uses_single_client_when_ds_unset(fake_eval_r2):
    r2 = R2()
    # No DS endpoint → ds_client falls back to the same R2 client.
    assert r2.client is fake_eval_r2[0]
    assert r2.ds_client is fake_eval_r2[0]
    assert r2.bucket == "test-bucket"
    assert r2.ds_bucket == "test-bucket"


def test_r2_init_creates_separate_ds_client_when_env_set(mocker, monkeypatch):
    monkeypatch.setenv("TEUTONIC_R2_ENDPOINT", "https://r2.example/")
    monkeypatch.setenv("TEUTONIC_R2_ACCESS_KEY", "k")
    monkeypatch.setenv("TEUTONIC_R2_SECRET_KEY", "s")
    monkeypatch.setenv("TEUTONIC_R2_BUCKET", "r2-bucket")
    monkeypatch.setenv("TEUTONIC_DS_ENDPOINT", "https://ds.example/")
    monkeypatch.setenv("TEUTONIC_DS_ACCESS_KEY", "dk")
    monkeypatch.setenv("TEUTONIC_DS_SECRET_KEY", "ds")
    monkeypatch.setenv("TEUTONIC_DS_BUCKET", "ds-bucket")

    clients = [MagicMock(), MagicMock()]
    iter_clients = iter(clients)
    mocker.patch("eval.torch_runner.boto3.client",
                 side_effect=lambda *a, **kw: next(iter_clients))
    r2 = R2()
    assert r2.client is clients[0]
    assert r2.ds_client is clients[1]
    assert r2.bucket == "r2-bucket"
    assert r2.ds_bucket == "ds-bucket"


def test_r2_get_decodes_json_from_body(fake_eval_r2):
    fake_eval_r2[0].get_object.return_value = _body(b'{"k": "v"}')
    r2 = R2()
    assert r2.get("state/x.json") == {"k": "v"}


def test_r2_get_returns_none_on_boto_error(fake_eval_r2):
    fake_eval_r2[0].get_object.side_effect = Exception("NoSuchKey")
    r2 = R2()
    assert r2.get("missing") is None


def test_r2_range_get_passes_range_header(fake_eval_r2):
    fake_eval_r2[0].get_object.return_value = _body(b"PARTIAL_DATA")
    r2 = R2()
    result = r2.range_get("shards/0.npy", 0, 1023)
    kwargs = fake_eval_r2[0].get_object.call_args.kwargs
    assert kwargs["Range"] == "bytes=0-1023"
    assert result == b"PARTIAL_DATA"


def test_r2_ds_get_swallows_errors(fake_eval_r2):
    fake_eval_r2[0].get_object.side_effect = Exception("ds down")
    r2 = R2()
    assert r2.ds_get("manifest.json") is None


def test_r2_ds_range_get_uses_ds_bucket(mocker, monkeypatch):
    monkeypatch.setenv("TEUTONIC_R2_ENDPOINT", "https://r2.example/")
    monkeypatch.setenv("TEUTONIC_R2_ACCESS_KEY", "k")
    monkeypatch.setenv("TEUTONIC_R2_SECRET_KEY", "s")
    monkeypatch.setenv("TEUTONIC_R2_BUCKET", "r2-bucket")
    monkeypatch.setenv("TEUTONIC_DS_ENDPOINT", "https://ds.example/")
    monkeypatch.setenv("TEUTONIC_DS_ACCESS_KEY", "dk")
    monkeypatch.setenv("TEUTONIC_DS_SECRET_KEY", "ds")
    monkeypatch.setenv("TEUTONIC_DS_BUCKET", "ds-bucket")

    clients = [MagicMock(), MagicMock()]
    iter_clients = iter(clients)
    mocker.patch("eval.torch_runner.boto3.client",
                 side_effect=lambda *a, **kw: next(iter_clients))
    clients[1].get_object.return_value = _body(b"DS_BYTES")
    r2 = R2()
    result = r2.ds_range_get("shards/0.npy", 100, 200)
    # The DS client (clients[1]) was used, not the R2 client.
    clients[0].get_object.assert_not_called()
    clients[1].get_object.assert_called_once()
    kwargs = clients[1].get_object.call_args.kwargs
    assert kwargs["Bucket"] == "ds-bucket"
    assert kwargs["Range"] == "bytes=100-200"
    assert result == b"DS_BYTES"


# ---------------------------------------------------------------------
# get_shard_info — parses .npy header to compute total token count.

class _FakeR2WithDsRange:
    """Minimal stub: only ds_range_get is exercised."""
    def __init__(self, payload: bytes):
        self._payload = payload
        self.calls: list = []

    def ds_range_get(self, key: str, start: int, end: int) -> bytes:
        self.calls.append((key, start, end))
        return self._payload[start:end + 1]


def test_get_shard_info_returns_total_token_count():
    # Build a real .npy with shape (4, 8) → total = 32.
    arr = np.zeros((4, 8), dtype=np.uint32)
    raw = io.BytesIO()
    np.save(raw, arr)
    r2 = _FakeR2WithDsRange(raw.getvalue())
    assert get_shard_info(r2, "shards/0.npy") == 32


def test_get_shard_info_reads_first_1024_bytes_via_ds_range_get():
    arr = np.zeros((2, 4), dtype=np.uint32)
    raw = io.BytesIO()
    np.save(raw, arr)
    r2 = _FakeR2WithDsRange(raw.getvalue())
    get_shard_info(r2, "shards/0.npy")
    assert r2.calls == [("shards/0.npy", 0, 1023)]


def test_get_shard_info_handles_1d_array():
    arr = np.zeros(64, dtype=np.uint32)
    raw = io.BytesIO()
    np.save(raw, arr)
    r2 = _FakeR2WithDsRange(raw.getvalue())
    assert get_shard_info(r2, "shards/0.npy") == 64
