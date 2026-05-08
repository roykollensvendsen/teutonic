#!/usr/bin/env python3
"""Seed a freshly initialised GPT-2 (nano-gpt) checkpoint.

Used to bootstrap the local devnet's genesis king. Random-init a tiny GPT-2,
save to disk, optionally push to HF. No GPU needed — ~7M params builds
instantly on CPU. Mirrors `archs/qwen3_moe/seed.py` so the devnet workflow
matches the production one byte-for-byte from the validator's perspective.

Defaults are sized for "fast on a laptop" (Shakespeare in seconds), not for
incentive realism. Override via `TEUTONIC_SEED_*` env vars or CLI flags.

    python -m archs.nano_gpt.seed                    # save to /tmp
    python -m archs.nano_gpt.seed --push             # save and push to HF
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import chain_config  # noqa: E402

chain_config.load_arch()
from archs.nano_gpt import GPT2Config, GPT2LMHeadModel  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("seed-nano-gpt")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
TARGET_REPO = os.environ.get("TEUTONIC_SEED_REPO_OVERRIDE", chain_config.SEED_REPO)
TOKENIZER_REPO = os.environ.get("TEUTONIC_SEED_TOKENIZER_OVERRIDE",
                                chain_config.SEED_TOKENIZER_REPO or "gpt2")
OUT_DIR = os.environ.get("TEUTONIC_SEED_DIR",
                         f"/tmp/{chain_config.NAME.lower()}")

VOCAB_SIZE = int(os.environ.get("TEUTONIC_SEED_VOCAB", "50257"))
N_EMBD = int(os.environ.get("TEUTONIC_SEED_N_EMBD", "128"))
N_LAYER = int(os.environ.get("TEUTONIC_SEED_N_LAYER", "4"))
N_HEAD = int(os.environ.get("TEUTONIC_SEED_N_HEAD", "4"))
N_POSITIONS = int(os.environ.get("TEUTONIC_SEED_N_POSITIONS", "256"))


def build_config() -> GPT2Config:
    cfg = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_embd=N_EMBD,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_positions=N_POSITIONS,
        # gpt2 BPE special-token convention: a single token (50256) does duty
        # as bos/eos/pad. Override via TEUTONIC_SEED_{BOS,EOS,PAD}_ID for a
        # custom tokenizer (e.g. char-level Shakespeare with vocab=128).
        bos_token_id=int(os.environ.get("TEUTONIC_SEED_BOS_ID", "50256")),
        eos_token_id=int(os.environ.get("TEUTONIC_SEED_EOS_ID", "50256")),
        pad_token_id=int(os.environ.get("TEUTONIC_SEED_PAD_ID", "50256")),
        tie_word_embeddings=True,
    )
    cfg.architectures = ["GPT2LMHeadModel"]
    return cfg


def _strip_auto_map(out_dir: Path):
    """Remove auto_map from config.json so consumers never invoke remote code.

    Vanilla GPT2Config doesn't set auto_map, but we keep parity with
    archs/qwen3_moe/seed.py and archs/quasar/seed.py for defense in depth —
    a future transformers refactor that adds auto_map by default would
    otherwise silently re-enable trust_remote_code execution paths.
    """
    cfg_path = out_dir / "config.json"
    with open(cfg_path) as f:
        cfg_data = json.load(f)
    if cfg_data.pop("auto_map", None) is not None:
        with open(cfg_path, "w") as f:
            json.dump(cfg_data, f, indent=2)
        log.info("stripped auto_map from %s", cfg_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--push", action="store_true",
                   help="Push to HF (requires HF_TOKEN with write access).")
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--repo", default=TARGET_REPO)
    p.add_argument("--tokenizer-repo", default=TOKENIZER_REPO,
                   help="HF repo with a compatible tokenizer to bundle "
                        "alongside the weights. 'gpt2' for the BPE default; "
                        "set to a custom repo for char-level or smaller vocab.")
    p.add_argument("--seed", type=int, default=42,
                   help="Torch RNG seed for reproducible random init.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config()
    log.info("config: vocab=%d n_embd=%d n_layer=%d n_head=%d n_pos=%d",
             cfg.vocab_size, cfg.n_embd, cfg.n_layer, cfg.n_head, cfg.n_positions)

    torch.manual_seed(args.seed)
    t0 = time.time()
    model = GPT2LMHeadModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("init complete: %.2fM params in %.1fs", n_params / 1e6, time.time() - t0)

    log.info("saving to %s", out_dir)
    model.save_pretrained(out_dir, safe_serialization=True)
    _strip_auto_map(out_dir)

    log.info("bundling tokenizer from %s", args.tokenizer_repo)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_repo, token=HF_TOKEN or None)
    tok.save_pretrained(out_dir)

    if args.push:
        if not HF_TOKEN:
            raise SystemExit("--push requires HF_TOKEN env var")
        api = HfApi(token=HF_TOKEN)
        log.info("pushing to %s", args.repo)
        api.create_repo(args.repo, exist_ok=True)
        api.upload_folder(folder_path=str(out_dir), repo_id=args.repo)
        log.info("pushed: https://huggingface.co/%s", args.repo)
    else:
        log.info("dry-run (use --push to publish to HF)")


if __name__ == "__main__":
    main()
