"""Tests for `validator._seed_king_hash` — the king-hash computation
that fast-paths through the eval-server when possible.

Spec-først: tests are written from the function's docstring + signature
+ callsites, NOT its body. If a test fails, it's one of:

* (a) test bug — fixture or assertion is wrong
* (b) spec ambiguity — docstring doesn't pin the behaviour the test
  asserts (escalate, do not silently relax)
* (c) impl bug — code does not match the documented contract

## Contract (from docstring at validator.py:805)

`_seed_king_hash(repo: str, revision: str) -> str` returns sha256 over
the repo's safetensors so it matches the `king_hash` miners encode in
their on-chain commits (`miner.sha256_dir`).

Two paths in priority order:

1. **Fast path** — query `GET /hash` on the eval-server (env
   `TEUTONIC_EVAL_SERVER`). Skipped if:
   * env var is unset / empty
   * server is unreachable (connection error, timeout, etc.)
   * server returns HTTP 404 (repo not in its cache)

2. **Slow path** — isolated subprocess that does `snapshot_download`
   + sha256 locally. Subprocess insulation guards against native
   crashes in the HF downloader (per 2026-04-26 xet-abort incident).

3. **Last-resort fallback** — if the subprocess also fails, returns
   the literal string `"seed"` as a placeholder. State.load() detects
   the placeholder and recomputes later.

## Callsites (validator.py:1009, 2367, 2460)

* State.load placeholder-recompute path
* coronation moment (`process_challenge`)
* startup with seed king

All three care only about the return value being a hex sha256 string
(or the `"seed"` placeholder). None depend on which path produced it.
"""
from unittest.mock import MagicMock

import pytest

import validator

# ---------------------------------------------------------------------
# Shared mocks: httpx for fast path, subprocess for slow path.

@pytest.fixture
def mock_httpx_get(mocker):
    """Patch `validator.httpx.get` so the fast-path HTTP call is observable.

    Returns the mock so a test can configure `.return_value =
    _resp({...})` or `.side_effect = ConnectionError(...)`.
    """
    return mocker.patch("validator.httpx.get")


@pytest.fixture
def mock_subprocess_run(mocker):
    """Patch the global `subprocess.run` so the slow-path call is
    observable. `_seed_king_hash` does a lazy `import subprocess` at
    call-time, so patching the module-global captures it.

    The slow-path mechanism (subprocess.run via sys.executable + a
    Python script constant) is impl detail — see tests/SPEC_DEBT.md
    entry for validator.py:805.
    """
    return mocker.patch("subprocess.run")


def _resp_200(json_payload: dict):
    """Build a sync httpx-shaped response."""
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = json_payload
    r.raise_for_status = MagicMock()
    return r


def _resp_404():
    r = MagicMock()
    r.status_code = 404
    return r


def _proc_ok(stdout: str):
    """Build a CompletedProcess-like return for subprocess.run."""
    p = MagicMock()
    p.returncode = 0
    p.stdout = stdout
    return p


def _proc_fail():
    p = MagicMock()
    p.returncode = 1
    p.stdout = ""
    p.stderr = "boom"
    return p


# ---------------------------------------------------------------------
# Fast path: eval-server returns a hash → function returns it without
# spawning a subprocess.

def test_fast_path_returns_hash_from_eval_server(
    monkeypatch, mock_httpx_get, mock_subprocess_run,
):
    monkeypatch.setenv("TEUTONIC_EVAL_SERVER", "http://eval-server:9000")
    expected_hash = "a" * 64
    mock_httpx_get.return_value = _resp_200({"sha256": expected_hash})
    result = validator._seed_king_hash("alice/king-repo", "rev_abc")
    assert result == expected_hash
    # Slow path was NOT taken.
    mock_subprocess_run.assert_not_called()


# ---------------------------------------------------------------------
# Fast path skipped: env unset → straight to slow path.

def test_falls_back_to_subprocess_when_env_unset(
    monkeypatch, mock_httpx_get, mock_subprocess_run,
):
    monkeypatch.delenv("TEUTONIC_EVAL_SERVER", raising=False)
    mock_subprocess_run.return_value = _proc_ok("b" * 64 + "\n")
    result = validator._seed_king_hash("alice/king-repo", "rev_abc")
    assert result == "b" * 64
    # Spec says "Skipped if TEUTONIC_EVAL_SERVER is unset" — must NOT
    # call httpx in that case.
    mock_httpx_get.assert_not_called()
    mock_subprocess_run.assert_called_once()


def test_falls_back_to_subprocess_when_env_empty(
    monkeypatch, mock_httpx_get, mock_subprocess_run,
):
    # Empty string is also "unset" semantically.
    monkeypatch.setenv("TEUTONIC_EVAL_SERVER", "")
    mock_subprocess_run.return_value = _proc_ok("c" * 64)
    validator._seed_king_hash("alice/repo", "rev")
    mock_httpx_get.assert_not_called()


# ---------------------------------------------------------------------
# Fast path failure modes: 404 / unreachable / connection error.

def test_falls_back_to_subprocess_on_eval_server_404(
    monkeypatch, mock_httpx_get, mock_subprocess_run,
):
    monkeypatch.setenv("TEUTONIC_EVAL_SERVER", "http://eval-server:9000")
    mock_httpx_get.return_value = _resp_404()
    mock_subprocess_run.return_value = _proc_ok("d" * 64)
    result = validator._seed_king_hash("alice/repo", "rev")
    assert result == "d" * 64
    mock_subprocess_run.assert_called_once()


def test_falls_back_to_subprocess_on_connection_error(
    monkeypatch, mock_httpx_get, mock_subprocess_run,
):
    monkeypatch.setenv("TEUTONIC_EVAL_SERVER", "http://eval-server:9000")
    import httpx
    mock_httpx_get.side_effect = httpx.ConnectError("connection refused")
    mock_subprocess_run.return_value = _proc_ok("e" * 64)
    result = validator._seed_king_hash("alice/repo", "rev")
    assert result == "e" * 64


def test_falls_back_to_subprocess_on_timeout(
    monkeypatch, mock_httpx_get, mock_subprocess_run,
):
    monkeypatch.setenv("TEUTONIC_EVAL_SERVER", "http://eval-server:9000")
    import httpx
    mock_httpx_get.side_effect = httpx.ReadTimeout("timed out")
    mock_subprocess_run.return_value = _proc_ok("f" * 64)
    result = validator._seed_king_hash("alice/repo", "rev")
    assert result == "f" * 64


# ---------------------------------------------------------------------
# Last-resort fallback: both paths fail → returns "seed" placeholder.

def test_returns_seed_placeholder_when_subprocess_fails(
    monkeypatch, mock_httpx_get, mock_subprocess_run,
):
    # Spec: "Falls back to literal 'seed' placeholder if the subprocess
    # fails; State.load()'s placeholder-recompute fixes it on..."
    monkeypatch.delenv("TEUTONIC_EVAL_SERVER", raising=False)
    mock_subprocess_run.return_value = _proc_fail()
    result = validator._seed_king_hash("alice/repo", "rev")
    assert result == "seed"


def test_returns_seed_placeholder_when_subprocess_raises(
    monkeypatch, mock_httpx_get, mock_subprocess_run,
):
    monkeypatch.delenv("TEUTONIC_EVAL_SERVER", raising=False)
    mock_subprocess_run.side_effect = OSError("no such executable")
    result = validator._seed_king_hash("alice/repo", "rev")
    assert result == "seed"


# ---------------------------------------------------------------------
# Undocumented fast-path fall-through cases (see tests/SPEC_DEBT.md
# entry for validator.py:805 — HTTP edge cases). Pinning so the
# behaviour stays stable until the docstring is tightened.

def test_falls_back_to_subprocess_on_200_with_empty_sha256(
    monkeypatch, mock_httpx_get, mock_subprocess_run,
):
    # 200 OK but the JSON body has empty sha256 → fall through.
    monkeypatch.setenv("TEUTONIC_EVAL_SERVER", "http://eval-server:9000")
    mock_httpx_get.return_value = _resp_200({"sha256": ""})
    mock_subprocess_run.return_value = _proc_ok("g" * 64)
    result = validator._seed_king_hash("alice/repo", "rev")
    assert result == "g" * 64
    mock_subprocess_run.assert_called_once()


def test_falls_back_to_subprocess_on_unexpected_http_status(
    monkeypatch, mock_httpx_get, mock_subprocess_run,
):
    # Non-200, non-404 (e.g. 500, 502) → fall through with warning.
    monkeypatch.setenv("TEUTONIC_EVAL_SERVER", "http://eval-server:9000")
    bad = MagicMock()
    bad.status_code = 503
    mock_httpx_get.return_value = bad
    mock_subprocess_run.return_value = _proc_ok("h" * 64)
    result = validator._seed_king_hash("alice/repo", "rev")
    assert result == "h" * 64
