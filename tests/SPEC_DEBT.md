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


### eval_server.py:331 — `_evict_for_challenger`

* **Gap:** Docstring describes intent ("force-evict every cached repo
  that's NOT the king and NOT the challenger we're about to load") and
  the why ("disk-pressure backstop"), but does NOT mention:
  * the function is non-fatal — it wraps everything in
    `try/except Exception: log.warning(...)` so a scan or delete
    failure does not propagate into the caller (`_load_challenger`,
    on the eval dispatch path).
  * `_king_repo=None` and `_king_repo=""` are both treated as "no king
    yet" — `kept_repos.discard()` calls handle both.
* **Observed:** `try/except Exception` wraps the body; explicit
  `kept_repos.discard(None)` and `kept_repos.discard("")` after
  `kept_repos = {_king_repo, target_repo}`.
* **Suggested:** docstring → add "Non-fatal: scan or delete errors
  are logged at WARNING and swallowed so the dispatch loop does not
  stall. `_king_repo=None`/`""` (no king crowned yet) is treated as
  'protect target only'."
* **Touched by:** `tests/test_eval_server_cleanup_hf_cache.py` (M33)


### validator.py:491 — `validate_challenger_config` (post-86ab8dd / 4240f44)

* **Gap:** Docstring says "Check challenger config.json matches king
  architecture before deploying". Since upstream commits 86ab8dd and
  4240f44 (2026-05-08), the function ALSO checks:
  * safetensors **naming layout** — accepts `model.safetensors` alone
    OR `model.safetensors.index.json` + `model-NNNNN-of-NNNNN` shards;
    rejects non-canonical names + sharded-without-index.
  * safetensors **total size** — rejects > 200 GB (default), capped via
    `TEUTONIC_MAX_CHALLENGER_SAFETENSORS_GB` env. Skipped when
    `repo_info(files_metadata=True)` raises (graceful degradation —
    fail-open against transient HF API errors).
  Neither layout nor size are mentioned in the docstring, nor is the
  env-var knob.
* **Observed:** Two new blocks at validator.py:556-602 implementing
  the naming check (with `_SAFETENSORS_SHARD_RE`) and size check
  (with `repo_info(files_metadata=True)` + cap env var).
* **Suggested:** docstring → add "Also rejects challengers whose
  safetensors layout is not loadable by `from_pretrained(use_safetensors=True)`
  (need either single-shard `model.safetensors` or sharded layout
  with `model.safetensors.index.json`) or whose total `.safetensors`
  size exceeds the cap (200 GB default, override via
  `TEUTONIC_MAX_CHALLENGER_SAFETENSORS_GB`). Size check is fail-open
  on `repo_info` exception."
* **Touched by:** `tests/test_validate_challenger_config.py` (M36)


### eval/torch_runner.py:363 — `_lm_head_device`

* **Gap:** Docstring says "Where lm_head's weight lives" — singular —
  but does not specify behaviour when:
  * the model has no `lm_head` attribute (raises AttributeError on
    bare attribute access)
  * `lm_head.parameters()` returns an empty iterator (raises
    StopIteration on `next()`)
  Either case is unreachable for valid HF causal-LM models, but a
  refactor that adds defensive defaults could silently hide bugs in
  upstream test fixtures.
* **Observed:** Both error paths reachable as natural consequences
  of the one-line impl `next(model.lm_head.parameters()).device`.
  Tests pin them so a future refactor must update the docstring
  alongside.
* **Suggested:** docstring → add "Raises AttributeError if the model
  has no `lm_head`; raises StopIteration if `lm_head.parameters()`
  is empty. Both are unreachable for valid HF causal-LM models;
  failure indicates a malformed fake or arch-package import problem."
* **Touched by:** `tests/test_eval_torch_sharded.py` (M34)
