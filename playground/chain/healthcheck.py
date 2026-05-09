#!/usr/bin/env python3
"""Verify the local subtensor stack is up and producing blocks.

Connects via the bittensor SDK (the same library validator.py uses) so any
SDK-level breakage surfaces here rather than at validator launch. Reads
TEUTONIC_NETWORK from the env so a single `source playground/env.devnet.sh`
configures both this and the validator identically.

Usage:
    source playground/env.devnet.sh
    python -m playground.chain.healthcheck

Exit code 0 if chain is healthy (block > 0 AND advancing); non-zero otherwise.
"""
from __future__ import annotations

import os
import sys
import time

import bittensor as bt


def main() -> int:
    network = os.environ.get("TEUTONIC_NETWORK")
    if not network:
        print("TEUTONIC_NETWORK is unset. Run `source playground/env.devnet.sh`.",
              file=sys.stderr)
        return 1

    print(f"connecting to {network} ...")
    # The SDK was renamed `bt.subtensor` -> `bt.Subtensor` in 10.x. validator.py
    # still uses the lowercase name (validator.py:2475: `bt.subtensor(...)`),
    # which fails on bittensor>=10. That's an upstream impedance to resolve
    # separately; this healthcheck uses the new name so it works against the
    # version actually installed in the venv.
    sub = bt.Subtensor(network=network)

    block_a = sub.block
    print(f"block:  {block_a}")
    print(f"endpoint: {sub.chain_endpoint}")

    # Wait one block period (12s for canonical bittensor block time, plus
    # buffer) and confirm the head moved. Catches the "chain started but
    # consensus stalled" case where block 0 is reachable forever.
    print("waiting ~14s to confirm advance ...")
    time.sleep(14)
    block_b = sub.block

    if block_b <= block_a:
        print(f"FAIL: block did not advance — still at {block_b}",
              file=sys.stderr)
        return 2

    print(f"block advanced: {block_a} -> {block_b} (Δ = {block_b - block_a})")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
