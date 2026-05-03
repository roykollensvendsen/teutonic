import io

import numpy as np

from eval_torch import _classify_param, _parse_npy_header, _seed_for_iteration

# _parse_npy_header(raw: bytes) -> int
# Docstring: "Return the byte offset where data begins in a .npy file."
# Used by eval_torch to skip past the .npy header when streaming raw
# tensor bytes from R2 shards.

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


# _classify_param — bucket a parameter name into one of
# {attn, ffn, embed, lm_head, norm, bias, other}. Used to categorize
# state_dict parameter names from HF transformers models.

def test_classify_param_returns_one_of_known_buckets():
    valid = {"attn", "ffn", "embed", "lm_head", "norm", "bias", "other"}
    assert _classify_param("model.layers.0.self_attn.q_proj.weight") in valid


def test_classify_param_embed_tokens():
    assert _classify_param("model.embed_tokens.weight") == "embed"


def test_classify_param_lm_head():
    assert _classify_param("lm_head.weight") == "lm_head"


def test_classify_param_layernorm():
    assert _classify_param("model.layers.0.input_layernorm.weight") == "norm"


def test_classify_param_attention_q_proj():
    assert _classify_param("model.layers.0.self_attn.q_proj.weight") == "attn"


def test_classify_param_mlp_gate_proj():
    assert _classify_param("model.layers.0.mlp.gate_proj.weight") == "ffn"


def test_classify_param_unknown_falls_to_other():
    assert _classify_param("some_random_unrecognized_param") == "other"


# _seed_for_iteration — derive a deterministic PRNG seed from the
# iteration index. Docstring: "Stable per-seed-index PRNG seed derived
# from PROBE_SEED."

def test_seed_for_iteration_returns_int():
    assert isinstance(_seed_for_iteration(0), int)


def test_seed_for_iteration_is_deterministic():
    assert _seed_for_iteration(7) == _seed_for_iteration(7)
    assert _seed_for_iteration(42) == _seed_for_iteration(42)


def test_seed_for_iteration_distinct_indices_yield_distinct_seeds():
    # Across a small range, no collisions expected (the function is
    # explicitly named "per-seed-index").
    seeds = {_seed_for_iteration(i) for i in range(20)}
    assert len(seeds) == 20
