#!/usr/bin/env python3
"""Build a tinyshakespeare-tokenized .npy shard at char level.

Mirrors `playground.dataset.build_shakespeare` byte-for-byte except for the
tokenizer: char-level vocab (69 tokens including 4 specials) instead of
gpt2's 50257-token BPE. Output goes to a different default path so both
tokenizations can coexist on disk; manifest format and shard file format
are identical so eval-server's `get_shard_info` and `fetch_sequences`
read it the same way.

Run:
    python -m playground.tokenizers.build_char_tokenizer --push  # tokenizer first
    python -m playground.dataset.build_shakespeare_char          # then shard
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build-shakespeare-char")

DTYPE = np.uint32
BYTES_PER_TOKEN = 4
TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def fetch_tinyshakespeare(cache_path: Path) -> str:
    if cache_path.exists():
        log.info("using cached tinyshakespeare at %s", cache_path)
        return cache_path.read_text(encoding="utf-8")
    import urllib.request
    log.info("downloading tinyshakespeare from %s", TINYSHAKESPEARE_URL)
    with urllib.request.urlopen(TINYSHAKESPEARE_URL, timeout=30) as r:  # noqa: S310
        text = r.read().decode("utf-8")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(text, encoding="utf-8")
    return text


def tokenize(text: str, tokenizer_repo: str) -> np.ndarray:
    from transformers import AutoTokenizer
    log.info("loading tokenizer: %s", tokenizer_repo)
    tok = AutoTokenizer.from_pretrained(tokenizer_repo)
    log.info("encoding %d chars (vocab=%d)", len(text), tok.vocab_size)
    t0 = time.time()
    ids = tok.encode(text, add_special_tokens=False)
    log.info("encoded %d tokens in %.1fs (compression %.3fx)",
             len(ids), time.time() - t0, len(text) / max(len(ids), 1))
    return np.asarray(ids, dtype=DTYPE)


def write_shard(tokens: np.ndarray, seq_len: int, shard_path: Path,
                shard_key: str) -> dict:
    """Save tokens as a flat-uint32 .npy. Drops tail tokens that don't fill
    a full seq_len window (matches build_shakespeare.py's contract).
    """
    n_full = (len(tokens) // seq_len) * seq_len
    if n_full == 0:
        raise SystemExit(f"corpus too small: {len(tokens)} but seq_len={seq_len}")
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


def build_manifest(tokenizer_name: str, source: str, shard_prefix: str,
                   shards: list[dict]) -> dict:
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="playground/dataset/data-char",
                   help="Destination root. Manifest + shards/ live here.")
    p.add_argument("--seq-len", type=int, default=2048,
                   help="Must match validator.py SEQ_LEN.")
    p.add_argument("--tokenizer",
                   default="ai-garage/Teutonic-Nano-char-tokenizer",
                   help="HF repo id of the char-level tokenizer.")
    p.add_argument("--shard-prefix", default="dataset/v2/shards/",
                   help="Logical R2 key prefix in the manifest. Validator "
                        "hardcodes `dataset/v2/manifest.json` so we keep "
                        "this consistent with baseline.")
    p.add_argument("--cache",
                   default="playground/dataset/cache/input.txt",
                   help="Where to cache the raw shakespeare text.")
    args = p.parse_args()

    out = Path(args.out_dir)
    text = fetch_tinyshakespeare(Path(args.cache))
    tokens = tokenize(text, args.tokenizer)

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
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("manifest: %s (%d tokens across %d shards, %d sequences "
             "at seq_len=%d)",
             manifest_path, manifest["total_tokens"], manifest["total_shards"],
             manifest["total_tokens"] // args.seq_len, args.seq_len)
    return 0


if __name__ == "__main__":
    sys.exit(main())
