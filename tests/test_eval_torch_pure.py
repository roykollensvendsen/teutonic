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

from eval.torch_runner import _parse_npy_header

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
