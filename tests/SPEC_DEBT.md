# Spec debt — gaps in docstrings/specs we leaned on impl for

When writing a test, the discipline (per `tests/README.md`) is:

> Read docstring + signature + 1-3 callsites before the body. The
> "spec-først" discipline keeps tests honest about what's contract
> vs. impl.

In practice some functions' docstrings underspecify behaviour we
need to test. When that happens we read the body to write the test
AND log the gap here so a later round can either:

* tighten the docstring (local commit or upstream PR), OR
* accept the impl-detail as part of the contract by making it
  explicit in the docstring.

Format:

```
### <file:line> — <function/symbol>

* **Gap:** what the docstring says vs what the impl does
* **Observed:** the impl behaviour we ended up testing
* **Suggested:** how to tighten the spec
* **Touched by:** which test file / milestone
```

---

### validator.py:805 — `_seed_king_hash`

* **Gap:** Docstring describes the slow path as "isolated subprocess
  that does `snapshot_download` + sha256 locally" without specifying
  the mechanism. Could plausibly be `subprocess.run`,
  `multiprocessing.Process`, `os.fork+exec`, etc.
* **Observed:** Implementation uses lazy `import subprocess` followed
  by `subprocess.run([sys.executable, "-c", _SEED_KING_HASH_SUBPROCESS,
  repo, revision or ""], capture_output=True, text=True,
  timeout=int(os.environ.get("TEUTONIC_KING_HASH_TIMEOUT_S", "1200")),
  env={**os.environ, "HF_HUB_DISABLE_XET": "1"})`. The script body
  lives in `_SEED_KING_HASH_SUBPROCESS` (module-level constant).
  Tests mock the global `subprocess.run` (works because of lazy
  import).
* **Suggested:** docstring → "isolated `subprocess.run` of a Python
  child process executing `_SEED_KING_HASH_SUBPROCESS`. Timeout
  configurable via `TEUTONIC_KING_HASH_TIMEOUT_S` (default 1200 s).
  Sets `HF_HUB_DISABLE_XET=1` to avoid the 2026-04-26 xet-abort
  crash."
* **Touched by:** `tests/test_validator_seed_king_hash.py` (M32)

### validator.py:805 — `_seed_king_hash` (HTTP edge cases)

* **Gap:** Docstring lists three skip-fast-path cases — env unset,
  server down, HTTP 404. It does NOT mention:
  * HTTP 200 with empty / missing `sha256` field in body → falls
    through to slow path (warning logged).
  * HTTP status other than 200 or 404 → falls through to slow path
    (warning logged).
* **Observed:** Both fall-through behaviours present in body
  (validator.py:835-848). Tests pin them.
* **Suggested:** docstring → add bullet "Skipped if the server's
  reply is malformed (200 without `sha256`) or any non-200/404 HTTP
  status."
* **Touched by:** `tests/test_validator_seed_king_hash.py` (M32)
