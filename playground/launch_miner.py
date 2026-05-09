#!/usr/bin/env python3
"""Submit a reveal-commitment from one of the registered miners, with
optional HF push so validator.process_challenge can actually download
and dispatch eval.

Steps:

  1. Generate a fresh nano-gpt checkpoint under
     playground/miners/<wallet_name>/<run_id>/.
  2. Compute sha256 over the safetensors files (matches miner.py:91's
     `sha256_dir` byte-for-byte).
  3. With --push: upload the checkpoint to HF under
     <hf-namespace>/Teutonic-Nano-<wallet>-<run_id>. Without --push:
     skip and use a fake repo name (validator.process_challenge will
     404 at HfApi.model_info — useful for proving just the chain side).
  4. Submit `set_reveal_commitment(wallet, netuid, "<king>:<repo>:<hash>",
     blocks_until_reveal=N)` against the local subtensor. After N blocks
     the chain auto-reveals; validator.scan_reveals returns the entry
     on its next tick.

Caveats by design (gates on later phases):

  - king_hash is a placeholder (zeros). Validator's scan_reveals stores
    it but never re-checks against the current king (M23's SHA-swap
    test pinned that as a known open exploit).
  - The local king ("ai-garage/Teutonic-Nano-devnet-king") is the
    pristine seed; no actual reign progression happens until eval-server
    finishes a real bootstrap test. That gates on the dataset shard
    being uploaded to minio (Phase 3a) and validator dispatching /eval
    (Phase 3h).

Run:
    docker compose -f playground/docker-compose.yml up -d
    source playground/env.devnet.sh
    python -m playground.chain.bootstrap_devnet           # if not bootstrapped
    python -m playground.launch_miner --wallet miner_alpha
    # ... wait blocks_until_reveal blocks (~24s at 12s block time, default 2) ...
    # validator's scan_reveals will then pick up this reveal.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
import time
from pathlib import Path

import bittensor as bt  # noqa: E402
import torch  # noqa: E402

# Shim FIRST so bt.wallet / bt.subtensor lowercase aliases resolve and
# wallet path is pinned to playground/wallets/.
import playground._shims  # noqa: F401, E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from archs.nano_gpt import GPT2Config, GPT2LMHeadModel  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("launch-miner")


PLAYGROUND_ROOT = Path(__file__).resolve().parent
MINERS_ROOT = PLAYGROUND_ROOT / "miners"


def _assert_safe() -> None:
    """Refuse to launch if env points at prod (mirrors the other launchers)."""
    safe = ("ws://localhost", "ws://127.0.0.1", "ws://[::1]")
    net = os.environ.get("TEUTONIC_NETWORK", "")
    if not any(net.startswith(p) for p in safe):
        raise SystemExit(
            f"REFUSING TO LAUNCH: TEUTONIC_NETWORK={net!r} doesn't look "
            f"like localhost. Run `source playground/env.devnet.sh` first."
        )


def sha256_dir(path: Path) -> str:
    """Mirror miner.py:91's sha256_dir byte-for-byte: concat all
    *.safetensors in sorted-name order, hash with sha256.
    """
    h = hashlib.sha256()
    for p in sorted(path.glob("*.safetensors")):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    return h.hexdigest()


def build_checkpoint(out_dir: Path, *, n_embd: int, n_layer: int, n_head: int,
                     vocab: int, seed: int) -> None:
    """Random-init a nano-gpt + save safetensors. Tied to archs.nano_gpt
    so any arch-side regression surfaces here.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = GPT2Config(
        vocab_size=vocab, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        n_positions=2048,                              # match SEQ_LEN=2048
        bos_token_id=50256, eos_token_id=50256, pad_token_id=50256,
        tie_word_embeddings=True,
    )
    cfg.architectures = ["GPT2LMHeadModel"]
    torch.manual_seed(seed)
    model = GPT2LMHeadModel(cfg)
    model.save_pretrained(out_dir, safe_serialization=True)
    log.info("saved checkpoint to %s (%.2fM params)",
             out_dir, sum(p.numel() for p in model.parameters()) / 1e6)


def main() -> int:
    _assert_safe()

    p = argparse.ArgumentParser()
    p.add_argument("--wallet", default="miner_alpha",
                   help="Wallet name under playground/wallets/. Must be "
                        "registered on the subnet (run bootstrap_devnet first).")
    p.add_argument("--hotkey", default="default")
    p.add_argument("--blocks-until-reveal", type=int, default=2,
                   help="How many blocks before the commit auto-reveals. "
                        "Default 2 (~24s at 12s block time) for fast demos.")
    p.add_argument("--king-hash", default="0" * 64,
                   help="Placeholder king hash. Real miner.py reads this from "
                        "the dashboard; we hardcode for chain-side demo.")
    p.add_argument("--seed", type=int, default=int(time.time()),
                   help="Torch RNG seed. Default = wall-clock so successive "
                        "runs produce different checkpoints (and thus "
                        "different model_hashes).")
    p.add_argument("--push", action="store_true",
                   help="Upload the checkpoint to HF before reveal. Without "
                        "this, the reveal payload's hf_repo is fictional and "
                        "validator.process_challenge will 404 — useful for "
                        "isolating chain-side bugs from HF-side ones.")
    p.add_argument("--hf-namespace", default="ai-garage",
                   help="HF account or org to push under. Repo name is "
                        "<namespace>/Teutonic-Nano-<wallet>-<run_id>.")
    args = p.parse_args()

    netuid = int(os.environ.get("TEUTONIC_NETUID", "2"))
    network = os.environ["TEUTONIC_NETWORK"]

    log.info("connecting to %s, netuid=%d", network, netuid)
    sub = bt.Subtensor(network=network)
    wallet = bt.wallet(name=args.wallet, hotkey=args.hotkey)
    log.info("wallet %s/%s  hotkey ss58=%s",
             args.wallet, args.hotkey, wallet.hotkey.ss58_address)

    # Pre-flight: hotkey must be registered on the subnet
    meta = sub.metagraph(netuid)
    if wallet.hotkey.ss58_address not in meta.hotkeys:
        raise SystemExit(
            f"hotkey {wallet.hotkey.ss58_address[:16]}... not registered on "
            f"netuid={netuid}. Run `python -m playground.chain.bootstrap_devnet`."
        )
    uid = meta.hotkeys.index(wallet.hotkey.ss58_address)
    log.info("hotkey registered as uid=%d on netuid=%d", uid, netuid)

    # 1. Build a nano-gpt checkpoint
    run_id = f"run-{int(time.time())}"
    ckpt_dir = MINERS_ROOT / args.wallet / run_id
    log.info("building nano-gpt checkpoint at %s", ckpt_dir)
    build_checkpoint(ckpt_dir, n_embd=128, n_layer=4, n_head=4, vocab=50257,
                     seed=args.seed)

    # 2. Hash the safetensors
    model_hash = sha256_dir(ckpt_dir)
    log.info("model_hash (sha256_dir): %s", model_hash)

    # 3. Optional: push to HF so validator.process_challenge can actually
    #    download. Repo naming embeds the coldkey ss58 prefix (first 8 chars)
    #    so validator's a36e71d coldkey-prefix check passes — without that
    #    embedded, process_challenge rejects with
    #    "hf repo ... must contain miner coldkey prefix '<prefix>'".
    coldkey_prefix = wallet.coldkeypub.ss58_address[:8]
    hf_repo = (f"{args.hf_namespace}/Teutonic-Nano-"
               f"{coldkey_prefix}-{args.wallet}-{run_id}")
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        log.info("pushing %s to HF (this is a real upload)", hf_repo)
        api.create_repo(hf_repo, exist_ok=True, private=False)
        api.upload_folder(folder_path=str(ckpt_dir), repo_id=hf_repo)
        log.info("pushed: https://huggingface.co/%s", hf_repo)
    else:
        log.info("--push not set; reveal points at fictional %s", hf_repo)

    # 4. Build the reveal payload — same colon-separated format scan_reveals
    #    parses (validator.py:647-650).
    payload = f"{args.king_hash}:{hf_repo}:{model_hash}"
    log.info("reveal payload: %s", payload)

    # 5. Submit set_reveal_commitment. Auto-reveals after blocks_until_reveal
    #    blocks; until then `subtensor.get_all_revealed_commitments` returns
    #    the prior state. After reveal, validator.scan_reveals sees this
    #    entry on its next tick.
    log.info("submitting reveal-commitment (blocks_until_reveal=%d)",
             args.blocks_until_reveal)
    resp = sub.set_reveal_commitment(
        wallet=wallet,
        netuid=netuid,
        data=payload,
        blocks_until_reveal=args.blocks_until_reveal,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    if not resp.success:
        raise SystemExit(f"set_reveal_commitment failed: {resp.error_message}")
    log.info("commit included; reveal scheduled")

    # 6. Optional: poll until revealed
    log.info("waiting for auto-reveal (~%ds at 12s block time) ...",
             args.blocks_until_reveal * 12 + 6)
    deadline = time.time() + args.blocks_until_reveal * 12 + 30
    while time.time() < deadline:
        time.sleep(6)
        all_reveals = sub.get_all_revealed_commitments(netuid) or {}
        entries = all_reveals.get(wallet.hotkey.ss58_address, [])
        if entries and any(payload in (e[1] if isinstance(e, tuple) else e)
                           for e in entries):
            log.info("reveal landed at block %d (entries=%d)",
                     sub.block, len(entries))
            return 0
    log.warning("reveal didn't appear within window; chain may need more blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
