"""Pin the bittensor SDK shim that lets unmodified validator.py / miner.py
import-and-run against our local devnet.

The shim solves two impedances (see playground/_shims.py for the full
rationale):

  1. validator.py:2474-2475 calls `bt.wallet(...)` and `bt.subtensor(...)` —
     lowercase aliases that bittensor 10.x removed.
  2. `bt.Wallet(name=..., hotkey=...)` defaults `path=~/.bittensor/wallets/`
     — the user's real wallet directory.

Importing playground._shims is a side-effect that mutates the global
`bittensor` module. These tests run in an isolated subprocess (via
the `script_runner`-style pattern below) so one test's mutation doesn't
leak into another's. The `pytest -p no:cacheprovider -x` invocation
order across this file would otherwise produce inconsistent state.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_in_subprocess(code: str) -> tuple[int, str, str]:
    """Run a snippet in a fresh python so each test gets a clean
    bittensor module — the shim mutates global state and we don't
    want one test leaking into another within the same process.
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------
# Lowercase aliases — what validator.py:2474-2475 needs.

def test_shim_adds_lowercase_subtensor_alias():
    code = """
        import playground._shims
        import bittensor as bt
        assert hasattr(bt, 'subtensor'), 'bt.subtensor not aliased'
        assert bt.subtensor is bt.Subtensor, 'alias does not point at bt.Subtensor'
        print('OK')
    """
    rc, out, err = _run_in_subprocess(code)
    assert rc == 0, f"shim failed:\nstdout={out}\nstderr={err}"
    assert "OK" in out


def test_shim_adds_lowercase_wallet_alias():
    code = """
        import playground._shims
        import bittensor as bt
        assert hasattr(bt, 'wallet'), 'bt.wallet not aliased'
        # Not is-identity here because bt.wallet wraps bt.Wallet (path-injecting).
        assert callable(bt.wallet), 'bt.wallet is not callable'
        print('OK')
    """
    rc, out, err = _run_in_subprocess(code)
    assert rc == 0, f"shim failed:\nstdout={out}\nstderr={err}"
    assert "OK" in out


# ---------------------------------------------------------------------
# Wallet path injection — the actual safety guarantee.

def test_shim_pins_default_wallet_path_to_playground_wallets():
    """`bt.wallet(name=..., hotkey=...)` with no explicit path= must NEVER
    resolve to ~/.bittensor/wallets/. This is the core safety constraint
    behind the shim — if it broke, validator.py launches would silently
    read real wallets.
    """
    code = """
        from pathlib import Path
        import playground._shims
        import bittensor as bt
        w = bt.wallet(name='probe', hotkey='probe')
        playground_root = Path('playground').resolve()
        wallet_path = Path(w.path).resolve()
        assert wallet_path == playground_root / 'wallets', (
            f'expected {playground_root}/wallets, got {wallet_path}'
        )
        assert '.bittensor' not in str(wallet_path), (
            f'path {wallet_path} resolves under ~/.bittensor — UNSAFE'
        )
        print('OK')
    """
    rc, out, err = _run_in_subprocess(code)
    assert rc == 0, f"shim failed:\nstdout={out}\nstderr={err}"
    assert "OK" in out


def test_shim_redirects_capital_wallet_too():
    """Even `bt.Wallet(...)` (capital, the modern API) should be redirected
    to playground/wallets/. Otherwise a future caller that uses the
    correct API would still leak to real wallets.
    """
    code = """
        from pathlib import Path
        import playground._shims
        import bittensor as bt
        w = bt.Wallet(name='probe', hotkey='probe')
        wallet_path = Path(w.path).resolve()
        assert wallet_path == Path('playground').resolve() / 'wallets'
        print('OK')
    """
    rc, out, err = _run_in_subprocess(code)
    assert rc == 0, f"shim failed:\nstdout={out}\nstderr={err}"
    assert "OK" in out


def test_shim_honors_explicit_path_override():
    """Tests and devnet utilities should be able to override the path
    when they need a scratch dir — the shim only injects when path= is
    omitted. Pin so this escape hatch can't be removed without surfacing.
    """
    code = """
        from pathlib import Path
        import playground._shims
        import bittensor as bt
        w = bt.wallet(name='probe', hotkey='probe', path='/tmp/scratch-shim-test')
        assert w.path == '/tmp/scratch-shim-test'
        print('OK')
    """
    rc, out, err = _run_in_subprocess(code)
    assert rc == 0, f"shim failed:\nstdout={out}\nstderr={err}"
    assert "OK" in out


# ---------------------------------------------------------------------
# Idempotency — re-import must be a no-op.

def test_shim_is_idempotent_under_double_import():
    """Importing the shim twice (which happens when both validator.py and
    miner.py go through it via different entrypoints) must not crash or
    re-wrap the wallet class — the second wrap would create infinite
    recursion or double-shimmed state.
    """
    code = """
        import playground._shims
        first_wallet_class = __import__('bittensor').Wallet
        # Force a re-import. Python's module cache should make this a no-op,
        # but we want to be sure even if the cache is bypassed somehow.
        import importlib
        importlib.reload(__import__('playground._shims', fromlist=['_']))
        second_wallet_class = __import__('bittensor').Wallet
        assert first_wallet_class is second_wallet_class, (
            'reload changed bt.Wallet — shim is not idempotent'
        )
        print('OK')
    """
    rc, out, err = _run_in_subprocess(code)
    assert rc == 0, f"shim failed:\nstdout={out}\nstderr={err}"
    assert "OK" in out
