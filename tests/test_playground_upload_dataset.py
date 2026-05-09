"""Pin the pure-helper logic of playground.storage.upload_dataset.

The upload script has three concerns:

  1. `make_client` — wraps boto3.client with the same connection knobs the
     validator's R2 dataset client uses (signature_version=s3v4, path-style
     addressing). Pin those defaults so a future refactor of one side
     surfaces here, not at devnet bring-up.

  2. `plan_uploads` — given a manifest, return [(local_path, s3_key)] pairs.
     Pure function. The manifest at `dataset/v2/manifest.json` is the
     validator's hardcoded entrypoint (validator.py:2196); shard keys come
     from manifest.shards[].key. Pin: manifest first, then each shard,
     local paths resolve relative to manifest dir.

  3. `ensure_bucket` and `upload` — actual S3 calls. Skipped here; verified
     manually via `docker compose up -d && python -m
     playground.storage.upload_dataset` against the live minio. moto/mock_aws
     would let us pin these too, but it's not in [test] extras yet.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from playground.storage.upload_dataset import make_client, plan_uploads

# ---------------------------------------------------------------------
# make_client — boto3 config parity with eval/torch_runner.R2.

def test_make_client_uses_path_addressing_and_s3v4():
    """minio rejects virtual-host-style addressing; signature_version=s3v4
    is required for ranged GETs to validate. The validator's dataset
    client (eval/torch_runner.R2.__init__) sets both — we mirror.
    """
    client = make_client("http://localhost:9100", "minioadmin", "minioadmin")
    cfg = client.meta.config
    assert cfg.signature_version == "s3v4"
    assert cfg.s3["addressing_style"] == "path"


def test_make_client_endpoint_url_propagates():
    client = make_client("http://localhost:9100", "k", "s")
    assert client.meta.endpoint_url == "http://localhost:9100"


def test_make_client_uses_auto_region():
    """Auto region matches eval/torch_runner.R2's primary client setting.
    minio doesn't validate region; arbitrary string would work, but pinning
    'auto' keeps parity so a copy-paste from production works.
    """
    client = make_client("http://x", "k", "s")
    assert client.meta.region_name == "auto"


# ---------------------------------------------------------------------
# plan_uploads — manifest → [(local_path, s3_key)] pairs.

def _write_manifest(dir_path: Path, shards: list[dict]) -> Path:
    """Build a minimal manifest.json under dir_path/manifest.json."""
    manifest = {
        "version": "v2",
        "tokenizer": "gpt2",
        "dtype": "uint32",
        "total_tokens": sum(s.get("n_tokens", 0) for s in shards),
        "total_shards": len(shards),
        "shards": shards,
    }
    manifest_path = dir_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_plan_uploads_first_pair_is_manifest_at_validator_hardcoded_key(tmp_path):
    """validator.py:2196 calls `r2.ds_get('dataset/v2/manifest.json')` —
    that exact key must be the upload destination, regardless of where the
    manifest lives on disk locally.
    """
    manifest_path = _write_manifest(tmp_path, [])
    pairs = plan_uploads(manifest_path, json.loads(manifest_path.read_text()))
    assert pairs[0] == (manifest_path, "dataset/v2/manifest.json")


def test_plan_uploads_shard_paths_resolve_relative_to_manifest(tmp_path):
    """build_shakespeare writes shards under <manifest_dir>/shards/<filename>
    and records the full s3-style key (e.g. dataset/v2/shards/shard_000000.npy)
    in manifest.shards[].key. Plan must read the local file from the
    on-disk shards/ dir, but UPLOAD it to the manifest-recorded key.
    """
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    shard_a = shards_dir / "shard_000000.npy"
    shard_b = shards_dir / "shard_000001.npy"
    shard_a.write_bytes(b"\x93NUMPY...fake_a")
    shard_b.write_bytes(b"\x93NUMPY...fake_b")

    manifest_path = _write_manifest(tmp_path, [
        {"key": "dataset/v2/shards/shard_000000.npy", "n_tokens": 100,
         "size_bytes": 14, "sha256": "0" * 64},
        {"key": "dataset/v2/shards/shard_000001.npy", "n_tokens": 200,
         "size_bytes": 14, "sha256": "1" * 64},
    ])

    pairs = plan_uploads(manifest_path, json.loads(manifest_path.read_text()))
    assert pairs[0] == (manifest_path, "dataset/v2/manifest.json")
    assert pairs[1] == (shard_a, "dataset/v2/shards/shard_000000.npy")
    assert pairs[2] == (shard_b, "dataset/v2/shards/shard_000001.npy")


def test_plan_uploads_raises_when_referenced_shard_missing(tmp_path):
    """If the manifest mentions a shard that build_shakespeare never wrote,
    fail loudly here rather than silently uploading 0 bytes that the
    validator would later 404 on.
    """
    manifest_path = _write_manifest(tmp_path, [
        {"key": "dataset/v2/shards/missing.npy", "n_tokens": 100,
         "size_bytes": 14, "sha256": "0" * 64},
    ])
    with pytest.raises(SystemExit, match="local file .* is missing"):
        plan_uploads(manifest_path, json.loads(manifest_path.read_text()))


def test_plan_uploads_uses_basename_of_shard_key_for_local_lookup(tmp_path):
    """Shard keys may have arbitrary nested prefixes
    (dataset/v2/shards/foo.npy, custom/playground/v17/x.npy, etc), but on
    disk build_shakespeare always writes them flat under <data>/shards/.
    The planner must use just the basename when looking on disk.
    """
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    (shards_dir / "deeply.npy").write_bytes(b"x")

    manifest_path = _write_manifest(tmp_path, [
        {"key": "some/very/nested/prefix/deeply.npy", "n_tokens": 1,
         "size_bytes": 1, "sha256": "0" * 64},
    ])
    pairs = plan_uploads(manifest_path, json.loads(manifest_path.read_text()))
    assert pairs[1][0] == shards_dir / "deeply.npy"
    assert pairs[1][1] == "some/very/nested/prefix/deeply.npy"


def test_plan_uploads_handles_empty_shards_list(tmp_path):
    """Edge case: a manifest with no shards. The plan still includes the
    manifest itself — useful for a `--dry-run` style flow that uploads
    only metadata.
    """
    manifest_path = _write_manifest(tmp_path, [])
    pairs = plan_uploads(manifest_path, json.loads(manifest_path.read_text()))
    assert len(pairs) == 1
    assert pairs[0][1] == "dataset/v2/manifest.json"
