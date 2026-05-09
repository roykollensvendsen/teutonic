#!/usr/bin/env python3
"""Build a tinyshakespeare-tokenised .npy shard + manifest for the devnet.

Mirrors `scripts/ingest_hf.py`'s shard format byte-for-byte (numpy .npy with
a flat uint32 token array, sidecar manifest.json) so the validator's
`r2.ds_get` / `get_shard_info` / `fetch_sequences` paths work without any
production-code modification.

This is the Phase-2 deliverable of the local-devnet experiment: feed the
nano-gpt king something to read. Tinyshakespeare (~1.1 MB) tokenises with
the gpt2 BPE down to ~280k tokens — enough for ~135 sequences at the
production SEQ_LEN=2048 (see validator.py:60), which is enough for eval-
server bootstrap math to function. Not enough to actually train a competent
model; the goal is wiring fidelity, not loss curves.

Output layout:
    playground/dataset/data/manifest.json
    playground/dataset/data/shards/shard_000000.npy

The script writes to disk; standing this up in a real R2 / minio bucket is
the Phase-3 (devnet bring-up) job. The validator's R2 client speaks plain
S3 + GET-range so any minio bucket containing the same key layout works.

Run:
    python -m playground.dataset.build_shakespeare
    # or with overrides:
    python -m playground.dataset.build_shakespeare --seq-len 2048 --tokenizer gpt2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build-shakespeare")

# Karpathy's tinyshakespeare. Single text file, 1.1 MB, public domain.
TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)

# Mirrors `scripts/ingest_hf.py`'s on-disk format.
DTYPE = np.uint32
BYTES_PER_TOKEN = 4


def fetch_tinyshakespeare(cache_path: Path) -> str:
    if cache_path.exists():
        log.info("using cached tinyshakespeare at %s (%d bytes)",
                 cache_path, cache_path.stat().st_size)
        return cache_path.read_text(encoding="utf-8")
    log.info("downloading tinyshakespeare from %s", TINYSHAKESPEARE_URL)
    with urllib.request.urlopen(TINYSHAKESPEARE_URL, timeout=30) as r:  # noqa: S310
        text = r.read().decode("utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    log.info("downloaded %d bytes to %s", len(text), cache_path)
    return text


def tokenize(text: str, tokenizer_name: str) -> np.ndarray:
    """Tokenise with HF AutoTokenizer; return a flat uint32 array.

    add_special_tokens=False matches scripts/ingest_hf.py — no extra BOS/EOS
    inserted, just the BPE pieces. The validator's bootstrap math doesn't
    care about specials; it samples random seq_len-windows from the flat
    token stream.
    """
    from transformers import AutoTokenizer
    log.info("loading tokenizer: %s", tokenizer_name)
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    log.info("encoding %d chars", len(text))
    t0 = time.time()
    ids = tok.encode(text, add_special_tokens=False)
    log.info("encoded %d tokens in %.1fs (compression %.2fx)",
             len(ids), time.time() - t0, len(text) / max(len(ids), 1))
    return np.asarray(ids, dtype=DTYPE)


def write_shard(tokens: np.ndarray, seq_len: int, shard_path: Path,
                shard_key: str) -> dict:
    """Save tokens as a .npy shard, return the manifest entry.

    Drops the tail tokens that don't fill a full seq_len window — the
    validator's `n_sequences = n_tokens // seq_len` math (eval/torch_runner
    line ~1417) does the same, so committing trailing partial-window tokens
    here would make manifest['total_tokens'] disagree with the validator's
    sequence count.
    """
    n_full = (len(tokens) // seq_len) * seq_len
    if n_full == 0:
        raise SystemExit(
            f"corpus too small: {len(tokens)} tokens but seq_len={seq_len}"
        )
    tokens = tokens[:n_full]
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(shard_path, tokens)

    file_hasher = hashlib.sha256()
    with open(shard_path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            file_hasher.update(chunk)

    return {
        "key": shard_key,
        "n_tokens": int(len(tokens)),
        "size_bytes": shard_path.stat().st_size,
        "sha256": file_hasher.hexdigest(),
    }


def build_manifest(tokenizer_name: str, source: str,
                   shard_prefix: str, shards: list[dict]) -> dict:
    """Mirrors scripts/ingest_hf.py:flush_shard + the manifest assembly there.

    The shape of this dict is what `validator.r2.ds_get('manifest.json')`
    returns; eval/torch_runner reads `version`, `dtype`, `shards`, and
    `total_tokens` from it.
    """
    return {
        "version": "v2",
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tokenizer": tokenizer_name,
        "dtype": "uint32",
        "source": source,
        "total_tokens": sum(s["n_tokens"] for s in shards),
        "total_shards": len(shards),
        "shard_prefix": shard_prefix,
        "shards": shards,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="playground/dataset/data",
                   help="Destination root. Manifest at <out>/manifest.json, "
                        "shards under <out>/shards/.")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Pack windows. Must match validator.py SEQ_LEN.")
    p.add_argument("--tokenizer", default="gpt2",
                   help="HF tokenizer repo. Must match nano-gpt seed king's "
                        "tokenizer (chain.toml [seed].tokenizer_repo).")
    p.add_argument("--shard-prefix", default="playground/v2/shards/",
                   help="Logical R2 key prefix recorded in the manifest. "
                        "When uploading to a minio/R2 bucket, this is what "
                        "the validator will GET; on disk we mirror it under "
                        "<out>/shards/. Trailing slash matters.")
    p.add_argument("--cache-dir", default="playground/dataset/cache",
                   help="Where to cache the downloaded tinyshakespeare text "
                        "between runs.")
    args = p.parse_args()

    out = Path(args.out_dir)
    cache = Path(args.cache_dir)

    text = fetch_tinyshakespeare(cache / "input.txt")
    tokens = tokenize(text, args.tokenizer)

    # tinyshakespeare fits comfortably in one shard. Production ingest_hf
    # rolls a new shard every ~2 GB; we have 1 MB.
    shard_filename = "shard_000000.npy"
    shard_path = out / "shards" / shard_filename
    shard_key = f"{args.shard_prefix}{shard_filename}"
    info = write_shard(tokens, args.seq_len, shard_path, shard_key)
    log.info("shard: %s (%d tokens, %d bytes, sha256 %s...)",
             info["key"], info["n_tokens"], info["size_bytes"],
             info["sha256"][:12])

    manifest = build_manifest(
        tokenizer_name=args.tokenizer,
        source=TINYSHAKESPEARE_URL,
        shard_prefix=args.shard_prefix,
        shards=[info],
    )
    manifest_path = out / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("manifest: %s (%d tokens across %d shards, %d sequences at "
             "seq_len=%d)",
             manifest_path, manifest["total_tokens"], manifest["total_shards"],
             manifest["total_tokens"] // args.seq_len, args.seq_len)


if __name__ == "__main__":
    sys.exit(main())
