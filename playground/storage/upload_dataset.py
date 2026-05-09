#!/usr/bin/env python3
"""Upload the local dataset (built by playground.dataset.build_shakespeare)
to minio (or any S3-compatible endpoint) for the devnet's storage layer.

Phase-3a artifact: bridges Phase-2's on-disk shards into a bucket the
validator's R2 client (eval/torch_runner.R2) can read from. After this runs,
`r2.ds_get('dataset/v2/manifest.json')` returns the dict and
`r2.ds_range_get('<shard-key>', start, end)` streams uint32 tokens —
the exact code path production validators use, just pointed at localhost:9000.

Reads connection settings from the same TEUTONIC_R2_* env vars the validator
reads, so `source playground/env.devnet.sh` configures both this script and
a subsequent `python validator.py ...` invocation identically.

Run:
    source playground/env.devnet.sh
    python -m playground.dataset.build_shakespeare      # if not yet built
    python -m playground.storage.upload_dataset
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("upload-dataset")


def make_client(endpoint: str, access_key: str, secret_key: str):
    """Mirror eval/torch_runner.R2's `ds_client` config — `path` addressing
    style + s3v4 signature so minio (which prefers path-style URLs) is happy.
    """
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
        config=BotoConfig(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 5, "mode": "adaptive"},
        ),
    )


def ensure_bucket(client, bucket: str) -> bool:
    """Create the bucket if absent. Returns True if it was created.

    Idempotent — re-running is a no-op. minio rejects bucket names with
    underscores and uppercase letters; pass a lowercase-only bucket name.
    """
    try:
        client.head_bucket(Bucket=bucket)
        return False
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            raise
    client.create_bucket(Bucket=bucket)
    log.info("created bucket %s", bucket)
    return True


def plan_uploads(manifest_path: Path, manifest: dict) -> list[tuple[Path, str]]:
    """Return [(local_path, s3_key)] pairs to upload.

    Always includes the manifest at the validator-hardcoded key
    `dataset/v2/manifest.json`. Shard keys come from manifest.shards[].key
    so the writer (build_shakespeare) and reader (validator.py) agree byte-
    for-byte on where each shard lives. Local shard files are resolved
    relative to manifest_path's parent, mirroring build_shakespeare's
    `<out>/shards/<filename>` layout.
    """
    pairs: list[tuple[Path, str]] = []
    pairs.append((manifest_path, "dataset/v2/manifest.json"))

    data_root = manifest_path.parent
    for entry in manifest.get("shards", []):
        key = entry["key"]
        # The shard key is the full S3 key (e.g. "dataset/v2/shards/foo.npy");
        # on-disk we stored it under <data_root>/shards/<basename> per
        # build_shakespeare's layout. Read shards/ relative to data_root.
        local = data_root / "shards" / Path(key).name
        if not local.exists():
            raise SystemExit(
                f"manifest references shard {key!r} but local file "
                f"{local} is missing — re-run build_shakespeare?"
            )
        pairs.append((local, key))
    return pairs


def upload(client, bucket: str, pairs: list[tuple[Path, str]]) -> int:
    """Upload each (local, key) pair. Returns total bytes pushed.

    Uses `put_object` with the file body in memory rather than `upload_file`'s
    multipart path — devnet shards are small (~1 MB) and the simpler
    single-part PUT is enough. The manifest is uploaded as
    application/json so validator's `ds_get` (which calls json.loads on the
    body) doesn't need to second-guess content type.
    """
    total = 0
    for local, key in pairs:
        body = local.read_bytes()
        content_type = "application/json" if key.endswith(".json") else "application/octet-stream"
        client.put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
        log.info("PUT s3://%s/%s (%d bytes)", bucket, key, len(body))
        total += len(body)
    return total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="playground/dataset/data",
                   help="Where build_shakespeare wrote manifest.json + shards/.")
    args = p.parse_args()

    endpoint = os.environ.get("TEUTONIC_R2_ENDPOINT")
    access = os.environ.get("TEUTONIC_R2_ACCESS_KEY")
    secret = os.environ.get("TEUTONIC_R2_SECRET_KEY")
    bucket = os.environ.get("TEUTONIC_R2_BUCKET", "playground")
    if not (endpoint and access and secret):
        raise SystemExit(
            "TEUTONIC_R2_{ENDPOINT,ACCESS_KEY,SECRET_KEY} must be set. "
            "Run `source playground/env.devnet.sh` first."
        )

    manifest_path = Path(args.data_root) / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"no manifest at {manifest_path}; run "
            f"`python -m playground.dataset.build_shakespeare` first."
        )
    manifest = json.loads(manifest_path.read_text())

    pairs = plan_uploads(manifest_path, manifest)
    log.info("planned %d uploads to %s/%s", len(pairs), endpoint, bucket)

    client = make_client(endpoint, access, secret)
    ensure_bucket(client, bucket)
    total = upload(client, bucket, pairs)

    log.info("uploaded %d bytes across %d objects (manifest + %d shards)",
             total, len(pairs), len(pairs) - 1)
    log.info("validator.r2.ds_get('dataset/v2/manifest.json') will now resolve")


if __name__ == "__main__":
    sys.exit(main())
