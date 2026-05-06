"""Tick-runner harness — drives subsets of validator's main loop in tests.

The validator's `main()` tick loop (validator.py around lines 2335-2500) does:

1. begin_tick / state-flushes
2. check_king_alive (file/repo health)
3. audit_incumbent_king (occasional)
4. refresh_uid_map(subtensor, NETUID)
5. fetch_tmc_data() — TaoMarketCap external HTTPS call
6. scan_reveals(subtensor, NETUID, state.seen)
7. enqueue(reveal) for each new reveal
8. while state.queue: process_challenge(...) — HF API + R2 + httpx to eval-server
9. maybe_set_weights(subtensor, ...)

This harness exposes the chain-driven, dependency-light parts as
`run_chain_tick(state, chain)` so tests that just need to drive the
metagraph + reveal-enqueue loop can do so without pulling in HF /
torch / heavy HTTP stubs.

For tests that need to drive the eval-streaming portion of the tick
against an in-process FakeEvalServer, `bind_eval_server_to_validator`
monkeypatches `validator.EVAL_SERVER_URL` and `httpx.AsyncClient` so
validator code that creates its own httpx client lands on the fake.

NOT modeled (deliberately):
* Watchdog / `asyncio.wait_for` around `process_challenge`. Tests that
  need the watchdog timeout path call `wait_for` themselves.
* Signal handlers (SIGTERM/SIGINT). The tick-runner is called
  synchronously by tests, no signals involved.
* `maybe_set_weights` weight push. Test the weight-push side via the
  existing `set_weights` unit tests against `_FakeSubtensor` /
  `FakeChain`.
* `audit_incumbent_king`. Out-of-band probe; tests that need it
  monkeypatch the function.
* `check_king_alive`. File-system / repo health check; tests that
  need a specific result monkeypatch it directly.
* `fetch_tmc_data`. Real HTTPS to TaoMarketCap; tests that need
  market state set `state.market = {...}` themselves.
"""
from __future__ import annotations

import httpx

NETUID_DEFAULT = 3


def run_chain_tick(state, chain, *, netuid: int = NETUID_DEFAULT) -> dict:
    """Drive the chain-read portion of one validator tick.

    Performs (in order, matching `validator.main()`):
    * `state.refresh_uid_map(chain, netuid)` — pulls metagraph state
      (hotkeys, emission, coldkeys) into `state.uid_map` and friends.
    * `reveals = scan_reveals(chain, netuid, state.seen)` — chain
      reveal scan; entries already in `state.seen` are filtered out.
    * `state.enqueue(rev, defer_flush=True)` for each new reveal.

    Returns a snapshot dict:
    * `reveals` — what `scan_reveals` returned (list[dict]).
    * `enqueued_count` — how many entries `state.enqueue` accepted.
    * `queue` — copy of `state.queue` after enqueueing.
    * `uid_map` — copy of `state.uid_map` after refresh.

    Side effects on `state`:
    * `uid_map`, `uid_emission_per_block`, `hotkey_coldkey` updated.
    * `state.queue` extended with newly-enqueued entries.
    * `state.seen` extended with the hotkeys whose reveals were new.

    Does NOT:
    * Call `state.flush()` or `state.flush_dashboard()`. The main loop
      does these conditionally; tests can flush themselves if they need
      the side effects.
    * Drive `process_challenge`. That requires HF + httpx mocking that
      lives outside the chain-read scope.
    """
    from validator import scan_reveals

    state.refresh_uid_map(chain, netuid)
    reveals = scan_reveals(chain, netuid, state.seen)
    enqueued_count = 0
    for rev in reveals:
        cid = state.enqueue(rev, defer_flush=True)
        if cid:
            enqueued_count += 1
    return {
        "reveals": reveals,
        "enqueued_count": enqueued_count,
        "queue": list(state.queue),
        "uid_map": dict(state.uid_map),
    }


def bind_eval_server_to_validator(eval_server, monkeypatch, *, validator_module=None) -> None:
    """Wire a FakeEvalServer to the validator's httpx code path.

    Sets:
    * `validator.EVAL_SERVER_URL` = the fake's base_url.
    * `httpx.AsyncClient` = a factory that injects the fake's
      `ASGITransport` and base_url so any client created without
      arguments by validator code lands on the fake.

    Caller must own the `monkeypatch` fixture so the patches unwind at
    test teardown. Example:

        async def test_x(monkeypatch):
            server = FakeEvalServer()
            bind_eval_server_to_validator(server, monkeypatch)
            # validator.process_challenge(...) now hits `server` over
            # the in-process ASGI mount.

    Caveats:
    * The patched `httpx.AsyncClient` ignores any caller-supplied
      `transport=` or `base_url=` kwargs in favour of the fake's. This
      is intentional — the validator hardcodes EVAL_SERVER_URL into
      f-strings, so the only way to redirect is to override transport.
    * Tests that need the original `httpx.AsyncClient` (e.g. for some
      OTHER endpoint) should not call this function.
    """
    if validator_module is None:
        import validator as validator_module

    fake_base = "http://fake-eval-server"
    monkeypatch.setattr(validator_module, "EVAL_SERVER_URL", fake_base)

    real_async_client_cls = httpx.AsyncClient

    def _patched_client(*args, **kwargs):
        kwargs["transport"] = httpx.ASGITransport(app=eval_server.app)
        kwargs["base_url"] = fake_base
        return real_async_client_cls(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _patched_client)
