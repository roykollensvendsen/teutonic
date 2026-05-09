#!/usr/bin/env python3
"""Launch eval_server.py against the local devnet.

Wraps the unmodified eval_server.py with:

  1. The bittensor compat shim (playground._shims). Eval-server doesn't
     import bittensor directly, but the shim also pins the wallet path —
     belt-and-suspenders if eval_server's deps grow a bittensor import
     later.
  2. Devnet env-var defaults: EVAL_HOST, EVAL_PORT (9200 to dodge the
     local connexi-service collision on 9000). Caller env wins via
     setdefault.

Usage:
    docker compose -f playground/docker-compose.yml up -d   # storage + chain
    source playground/env.devnet.sh                          # base env
    python -m playground.dataset.build_shakespeare           # if not built
    python -m playground.storage.upload_dataset              # → minio
    python -m playground.launch_eval_server                  # FastAPI on :9200

Stops on SIGINT/SIGTERM. Health probe: GET http://localhost:9200/health.

Note: eval_server has hard imports of torch + transformers + accelerate.
The CPU-only torch wheel works for boot + module-level init; actual eval
on a king model would need a real model load (~minutes on CPU for our
~7M-param nano-gpt). The launch script just verifies boot here; running
an eval against a real reveal is Phase 3f territory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULTS = {
    "EVAL_HOST": "127.0.0.1",
    "EVAL_PORT": "9200",
    # GPUs: empty string forces CPU fallback in parse_gpu_ids("auto")?
    # Actually torch_runner.parse_gpu_ids returns [] on no-CUDA boxes;
    # eval-server then runs on CPU. No env var needed for that path.
}
for k, v in _DEFAULTS.items():
    os.environ.setdefault(k, v)

# Sanity: refuse to launch unless TEUTONIC_NETWORK looks like localhost.
# Eval-server doesn't talk to the chain directly, but the shared
# environment means a mis-set TEUTONIC_NETWORK would imply an unsafe
# devnet config — fail loud.
_safe = ("ws://localhost", "ws://127.0.0.1", "ws://[::1]")
_net = os.environ.get("TEUTONIC_NETWORK", "")
if not any(_net.startswith(p) for p in _safe):
    print(
        f"REFUSING TO LAUNCH: TEUTONIC_NETWORK={_net!r} doesn't look like "
        f"localhost. Run `source playground/env.devnet.sh` first.",
        file=sys.stderr,
    )
    sys.exit(2)

# Shim must come before any eval_server import — eval_server imports
# chain_config which calls load_arch(), and we want the shim's wallet
# path pin in place even though eval_server doesn't itself construct
# wallets. Defense in depth.
import playground._shims  # noqa: F401, E402

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# eval_server's `if __name__ == "__main__"` block reads EVAL_HOST/EVAL_PORT
# and calls uvicorn.run. Re-implement it here so we don't have to invoke
# eval_server.py as a script (which would bypass the shim).
import uvicorn  # noqa: E402

if __name__ == "__main__":
    host = os.environ["EVAL_HOST"]
    port = int(os.environ["EVAL_PORT"])
    print(f"starting eval-server on {host}:{port}")
    uvicorn.run("eval_server:app", host=host, port=port, log_level="info")
