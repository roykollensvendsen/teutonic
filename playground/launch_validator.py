#!/usr/bin/env python3
"""Launch validator.py against the local devnet.

Wraps the unmodified validator.py with:

  1. The bittensor compat shim (playground._shims), so `bt.wallet(...)`
     and `bt.subtensor(...)` resolve and the wallet path is pinned to
     playground/wallets/.
  2. Devnet env-var defaults (NETUID=2, validator wallet name, etc.)
     that callers can still override from the shell. validator.py reads
     env at module-load, so the env injection happens before importing.

Usage:
    docker compose -f playground/docker-compose.yml up -d   # bring stack up
    source playground/env.devnet.sh                          # base env
    python -m playground.launch_validator                    # tick loop forever

Stops cleanly on Ctrl+C / SIGINT / SIGTERM. validator.py's main() is async
and runs until external interruption — `main_sync()` wraps it with the
asyncio.run() boilerplate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Inject devnet defaults BEFORE importing validator (it reads env at module
# load). Shell-set env wins via setdefault.
_DEFAULTS = {
    "TEUTONIC_NETUID": "2",
    "BT_WALLET_NAME": "validator",
    "BT_WALLET_HOTKEY": "default",
    # eval-server is its own process (`python -m playground.launch_eval_server`);
    # port 9200 by env.devnet.sh because 9000 collides with connexi-service
    # locally. Validator only POSTs to it when a reveal needs evaluating, so
    # this remains soft-required — the tick loop runs idle on a fresh subnet
    # even if eval-server isn't up yet.
    "TEUTONIC_EVAL_SERVER": "http://localhost:9200",
}
for k, v in _DEFAULTS.items():
    os.environ.setdefault(k, v)

# Sanity: refuse to launch if TEUTONIC_NETWORK looks like prod. The shim
# does the same check for wallet path; this catches the env-var case.
_safe = ("ws://localhost", "ws://127.0.0.1", "ws://[::1]")
_net = os.environ.get("TEUTONIC_NETWORK", "")
if not any(_net.startswith(p) for p in _safe):
    print(
        f"REFUSING TO LAUNCH: TEUTONIC_NETWORK={_net!r} doesn't look like "
        f"localhost. Run `source playground/env.devnet.sh` first.",
        file=sys.stderr,
    )
    sys.exit(2)

# Install shim — adds bt.wallet / bt.subtensor lowercase aliases, pins
# wallet path to playground/wallets/. Must happen before validator import.
import playground._shims  # noqa: F401, E402

# Path setup so `import validator` resolves to teutonic's top-level module.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import validator  # noqa: E402

if __name__ == "__main__":
    validator.main_sync()
