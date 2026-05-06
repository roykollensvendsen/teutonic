"""Tests for pure helpers in eval.torch_runner.

Note: many helpers tested here previously (`_classify_param`,
`_seed_for_iteration`, `_build_probe_verdict`) were removed upstream
when the trainability probe was rewritten from magnitude heuristics
to a loss-explosion approach (commit b868790). Only `_parse_npy_header`
survives; tests for the deleted helpers were dropped during the
restructure migration.
"""
import io

import numpy as np

from eval.torch_runner import _parse_npy_header, _parse_shard_header

# _parse_npy_header(raw: bytes) -> int
# Docstring: "Return the byte offset where data begins in a .npy file."
# Used by eval.torch_runner to skip past the .npy header when streaming
# raw tensor bytes from R2 shards.

def test_parse_npy_header_returns_int():
    raw = io.BytesIO()
    np.save(raw, np.array([1.0]))
    assert isinstance(_parse_npy_header(raw.getvalue()), int)


def test_parse_npy_header_v1_offset_locates_data():
    arr = np.arange(10, dtype=np.float32)
    raw = io.BytesIO()
    np.save(raw, arr)
    raw_bytes = raw.getvalue()

    offset = _parse_npy_header(raw_bytes)

    # Bytes from offset onward must equal the array's raw representation.
    assert raw_bytes[offset:offset + arr.nbytes] == arr.tobytes()


def test_parse_npy_header_v2_offset_locates_data():
    # Force v2.0 by writing the header explicitly (v2.0 uses a 4-byte
    # header length instead of 2 bytes — used when v1.0's 65535-byte
    # header would overflow, or written manually as below).
    arr = np.arange(5, dtype=np.int64)
    raw = io.BytesIO()
    np.lib.format.write_array(raw, arr, version=(2, 0))
    raw_bytes = raw.getvalue()

    offset = _parse_npy_header(raw_bytes)

    assert raw_bytes[offset:offset + arr.nbytes] == arr.tobytes()


def test_parse_npy_header_offset_at_least_minimum_v1_size():
    # v1.0 layout: 6 (magic) + 2 (version) + 2 (header_len_u16) = 10 bytes
    # before the header text, plus the header text itself.
    arr = np.array([1.0])
    raw = io.BytesIO()
    np.save(raw, arr)
    assert _parse_npy_header(raw.getvalue()) >= 10


def test_parse_npy_header_distinct_dtypes_yield_consistent_offsets():
    # Same shape but different dtype → header text differs slightly,
    # but offset must still locate the actual data bytes correctly.
    for dtype in (np.float32, np.float64, np.int32, np.uint8):
        arr = np.zeros(4, dtype=dtype)
        raw = io.BytesIO()
        np.save(raw, arr)
        raw_bytes = raw.getvalue()
        offset = _parse_npy_header(raw_bytes)
        assert raw_bytes[offset:offset + arr.nbytes] == arr.tobytes(), \
            f"offset mismatch for dtype={dtype}"


# _parse_shard_header(r2, shard_key) -> int
# Same parsing as _parse_npy_header but reads the first 1024 bytes from
# r2 via `r2.ds_range_get(shard_key, 0, 1023)`. fetch_sequences uses the
# returned offset to know where the data section begins so it can compute
# byte ranges for individual sequences.

class _FakeR2:
    """Tiny r2 stub with a programmable ds_range_get."""
    def __init__(self, data: bytes):
        self._data = data
        self.calls: list[tuple[str, int, int]] = []

    def ds_range_get(self, key: str, start: int, end: int) -> bytes:
        self.calls.append((key, start, end))
        return self._data[start:end + 1]


def test_parse_shard_header_returns_offset_matching_parse_npy_header():
    # The two parsers must agree on the offset for the same npy bytes.
    arr = np.zeros(8, dtype=np.float32)
    raw = io.BytesIO()
    np.save(raw, arr)
    raw_bytes = raw.getvalue()

    expected = _parse_npy_header(raw_bytes)
    r2 = _FakeR2(raw_bytes)
    actual = _parse_shard_header(r2, "shards/0.npy")
    assert actual == expected


def test_parse_shard_header_reads_first_1024_bytes():
    # The parser pulls a 1024-byte prefix via ds_range_get(0, 1023) —
    # consistent with how fetch_sequences locates the data section.
    arr = np.zeros(4, dtype=np.float32)
    raw = io.BytesIO()
    np.save(raw, arr)
    r2 = _FakeR2(raw.getvalue())
    _parse_shard_header(r2, "shards/0.npy")
    assert r2.calls == [("shards/0.npy", 0, 1023)]


def test_parse_shard_header_offset_locates_data_section():
    # Offset returned by the parser must let a caller slice straight to
    # the data bytes. Mirrors how fetch_sequences uses it.
    arr = np.arange(16, dtype=np.uint32)
    raw = io.BytesIO()
    np.save(raw, arr)
    raw_bytes = raw.getvalue()

    r2 = _FakeR2(raw_bytes)
    offset = _parse_shard_header(r2, "shards/0.npy")
    # The bytes starting at `offset` should be the data section.
    assert raw_bytes[offset:offset + arr.nbytes] == arr.tobytes()
