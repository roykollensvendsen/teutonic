# Tests

This directory contains the test suite for the teutonic validator
+ miner. ~440 tests + a small set of `xfail-strict` tests that pin
known bugs.

## Layout

```
tests/
├── README.md                    # this file
├── conftest.py                  # shared fixtures (r2_mock)
├── _harness/                    # in-process doubles for external deps
│   ├── chain.py                 # FakeChain — bittensor.subtensor double
│   ├── eval_server.py           # FakeEvalServer — async ASGI mount
│   └── tick.py                  # run_chain_tick + bind_eval_server_to_validator
├── test_harness_chain.py        # unit tests for FakeChain
├── test_harness_eval_server.py  # unit tests for FakeEvalServer
├── test_harness_tick.py         # unit tests for tick helpers
├── test_harness_integration.py  # cross-harness integration test (M1+M2+M3)
└── test_<module>_<aspect>.py    # one file per validator/eval-server module
```

The `_harness/` package is the foundation that makes bug-reproduction
tests possible without spinning up real subtensors, eval-servers, or
GPUs. See [Harness architecture](#harness-architecture) below.

## How to run tests

| Goal | Command |
|---|---|
| Full suite, parallel | `pytest tests/ -n auto` |
| Fast iteration (skip slow tests) | `pytest tests/ -m "not slow" -n auto` |
| Cross-harness integration only | `pytest tests/ -m integration -v` |
| Single file | `pytest tests/test_state_helpers.py -v` |
| Coverage report (HTML) | `pytest tests/ -m "not slow" -n auto --cov --cov-report=html && open htmlcov/index.html` |

### Markers

- **`slow`** — tests that take >1s individually. Currently the 4
  tests in `test_load_model_public_contract.py` (each ~20s due to
  unmocked `_prefetch_repo` retry-backoff). Skip with
  `-m "not slow"` for fast local iteration.
- **`integration`** — tests that exercise multiple harness modules
  together. Currently `test_harness_integration.py`. Run focused
  with `-m integration`.

## How to add a test

1. **Read the docstring + signature + 1-3 callsites** of the
   function you're testing. Don't read the function body until you
   have a baseline test in hand. The "spec-først" discipline keeps
   tests honest about what's contract vs. impl.

2. **Pick the right fixture from `conftest.py`**. The `r2_mock`
   fixture exposes `_storage`, `_appended`, `_dashboards` for
   assertion. If your test needs more (e.g. HfApi mock,
   `_isolate_evals_dict`), define it inline — `conftest.py` is for
   cross-cutting fixtures only (used by 3+ files).

3. **Document the contract** in your test's name and docstring.
   `test_<function>_<scenario>` reads cleanly in the runner.

4. **Mark slow tests** with `pytestmark = pytest.mark.slow` at
   module level if every test in the file exceeds 1s, or
   `@pytest.mark.slow` per-test for finer control.

5. **xfail-strict for bugs** — see the next section.

## xfail-strict convention

When you find a bug but want to ship a regression test before the
fix lands, mark the test as `@pytest.mark.xfail(strict=True, reason=...)`:

```python
@pytest.mark.xfail(
    strict=True,
    reason="<bug-name>. Fix is a separate plan. "
           "When the fix lands, this xfail flips to pass and "
           "strict=True forces removal of the marker.",
)
def test_<desired_behaviour>():
    # WANT: assert <correct behaviour>
    # GET today: <buggy behaviour>
    assert <correct expectation>
```

What this gets you:
- **Today**: test fails (bug exists) → marked `xfailed` → suite stays green
- **When fix lands**: test passes → `strict=True` forces it to fail
  the suite as `XPASS` until someone removes the marker
- **As regression**: removed marker becomes a permanent green test

Current xfail-strict tests: 4 (2 in `test_uid_mapping_race.py`, 2 in
`test_eval_failed_misclassification.py`).

## Harness architecture

```
                ┌─────────────────────┐
                │  validator.py code  │
                │  (process_challenge,│
                │   refresh_uid_map,  │
                │   ...)              │
                └──────────┬──────────┘
                           │ uses
                ┌──────────┴──────────┐
                │                     │
        subtensor.<x>          httpx.AsyncClient
                │                     │
                ▼                     ▼
        ┌──────────────┐    ┌──────────────────┐
        │  FakeChain   │    │ FakeEvalServer   │
        │              │    │  (ASGITransport) │
        │ register()   │    │                  │
        │ commit_      │    │ queue_event()    │
        │   reveal()   │    │ set_busy()       │
        │ advance_     │    │ posted_evals     │
        │   block()    │    │                  │
        └──────────────┘    └──────────────────┘
                │                     │
                └──────────┬──────────┘
                           │ both wired by
                           ▼
                ┌─────────────────────┐
                │  tests/_harness/    │
                │  tick.py            │
                │                     │
                │ run_chain_tick()    │
                │ bind_eval_server_   │
                │   to_validator()    │
                └─────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   r2_mock fixture   │
                │   (conftest.py)     │
                │                     │
                │  dict-backed R2     │
                │  with spy attrs     │
                └─────────────────────┘
```

### What each harness models (and what it doesn't)

**`FakeChain`** — `tests/_harness/chain.py`
- ✓ `block` property + `metagraph(netuid)` + `get_all_revealed_commitments(netuid)`
- ✓ Mutator API: `register(hotkey, uid, *, coldkey, emission)`, `advance_block(n)`, `commit_reveal(hk, payload, *, block)`, `set_emission(hk, em)`
- ✗ `set_weights()` — write side, mock separately if needed
- ✗ `get_block_hash()` — not modeled, raises AttributeError
- ✗ Real Bittensor RPC failure semantics — never raises spontaneously

**`FakeEvalServer`** — `tests/_harness/eval_server.py`
- ✓ `POST /eval` (returns eval_id or 409 busy) + `GET /eval/{id}/stream` (SSE)
- ✓ Test API: `queue_event()`, `set_busy()`, `posted_evals`
- ✗ Real GPU eval execution
- ✗ `/preload`, `/probe`, `/health` endpoints (add at first use)
- ✗ Threading-lock concurrency (only the 409 surface is modeled)

**`tick.py`** — `tests/_harness/tick.py`
- `run_chain_tick(state, chain, *, netuid)` — chain-driven portion
  of one validator tick (refresh_uid_map + scan_reveals + enqueue)
- `bind_eval_server_to_validator(eval_server, monkeypatch)` —
  redirects validator's `httpx.AsyncClient` to the in-process fake
- ✗ `process_challenge` itself — has HF + R2 + block-hash deps that
  live outside the harness's modeled surface

## Spec-først discipline

Read [the harness contract docstrings](_harness/chain.py) before
writing tests. Each module documents what it models AND what it
doesn't — the "NOT modeled" sections are load-bearing.

When tests go red:
1. **STOP.** Don't self-diagnose silently.
2. Identify three options: (a) test bug, (b) spec ambiguity,
   (c) impl bug.
3. Escalate to the human (commit message, code comment, or chat).

Never write a passing test by adjusting the assertion to match the
impl. Either the docstring should change or the impl should — the
test is the contract.

## CI

`.github/workflows/tests.yml` runs `pytest -n auto --cov` on every
push and pull_request to `tests/foundation`. Coverage HTML + XML
upload as a 14-day artifact named `coverage-<run_id>`.

Trigger scope is intentionally narrow: `main` tracks upstream and
isn't ours to validate.

## See also

- [Coverage configuration](../pyproject.toml) — `[tool.coverage.run]`
  + `[tool.coverage.report]`
- [Ruff configuration](../scripts/teutonic.ruff.toml)
- [CI workflow](../.github/workflows/tests.yml)
