"""Bittensor SDK compat shims for devnet launch.

Two impedances solved here:

  1. validator.py:2474-2475 uses `bt.wallet(...)` and `bt.subtensor(...)` —
     lowercase aliases that bittensor 10.x removed. The SDK now exposes
     `bt.Wallet` and `bt.Subtensor` (capital). We restore the lowercase
     aliases so unmodified validator.py can run.

  2. `bt.Wallet(name=..., hotkey=...)` defaults `path=~/.bittensor/wallets/`
     — the user's real wallet directory. Even reading from there is a
     foot-gun; if a wallet named "validator" / "teutonic" exists in
     production, validator.py would silently load it. We hard-pin the
     wallet path to playground/wallets/ for any caller that goes through
     our shim.

Activate by import:

    import playground._shims  # noqa: F401
    # then it's safe to import validator.py / miner.py

The shim is hasattr-gated and parent-of-each-other-checked, so:
  - importing twice is a no-op (idempotent)
  - if upstream lands the rename fix, our shim becomes a no-op
  - if the wallet path ever resolves under ~/.bittensor, we refuse to
    install (loud crash > silent leak)
"""
from __future__ import annotations

from pathlib import Path

import bittensor as bt

PLAYGROUND_ROOT = Path(__file__).resolve().parent
WALLET_PATH = PLAYGROUND_ROOT / "wallets"


def _assert_safe_wallet_path() -> None:
    """Refuse to install the shim if the wallet path overlaps real wallets."""
    home_bt = (Path.home() / ".bittensor").resolve()
    resolved = WALLET_PATH.resolve()
    if resolved == home_bt or home_bt in resolved.parents or resolved in home_bt.parents:
        raise RuntimeError(
            f"REFUSING TO INSTALL SHIM: wallet path {resolved} overlaps "
            f"with {home_bt}. The shim pins wallet path; if it pointed at "
            f"real wallets, validator.py would happily load them."
        )


_assert_safe_wallet_path()


# Resolve the underlying Wallet class. On a fresh import this is bt.Wallet
# itself; on a reload it's the original Wallet that the prior wrapper
# captured (via `_devnet_shimmed_wraps`). Walking through the marker
# attribute prevents double-wrapping, which would otherwise produce
# infinite recursion when the wrapper calls itself.
_OriginalWallet = getattr(bt.Wallet, "_devnet_shimmed_wraps", bt.Wallet)


def _wallet_with_devnet_path(*args, **kwargs):
    """`bt.Wallet` wrapper that pins `path=playground/wallets/` by default.

    Honors an explicit `path=` kwarg from the caller (escape hatch for
    tests that need a different scratch dir). Without that, every
    `bt.wallet(name=..., hotkey=...)` call in validator.py loads from
    playground/wallets/ instead of ~/.bittensor/wallets/.
    """
    if "path" not in kwargs:
        kwargs["path"] = str(WALLET_PATH)
    return _OriginalWallet(*args, **kwargs)


# Marker so a re-import sees this wrapper and skips re-wrapping.
_wallet_with_devnet_path._devnet_shimmed_wraps = _OriginalWallet  # type: ignore[attr-defined]


# Lowercase aliases. Hasattr-guarded so a future upstream rename or
# bittensor SDK that re-adds the alias makes this a no-op.
if not hasattr(bt, "subtensor"):
    bt.subtensor = bt.Subtensor

if not hasattr(bt, "wallet"):
    bt.wallet = _wallet_with_devnet_path


# Also redirect bt.Wallet itself, so even capital-W callers get the
# devnet path. The `_devnet_shimmed_wraps` marker makes this idempotent
# under reload — a second pass sees its own wrapper and skips.
if not hasattr(bt.Wallet, "_devnet_shimmed_wraps"):
    bt.Wallet = _wallet_with_devnet_path
