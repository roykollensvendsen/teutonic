"""FakeEvalServer — in-process ASGI double for eval_server.

Replaces a real eval_server.py FastAPI deployment for tests that drive
the validator's eval-streaming flow without spinning up a network service
or running an actual GPU eval. Mounts a tiny FastAPI app via
`httpx.ASGITransport` so consumer code that uses `httpx.AsyncClient`
talks to the harness over the same wire protocol it would use in
production.

Wire endpoints (matches eval_server.py):
* `POST /eval` — accepts an EvalRequest-shaped JSON body. Returns
  `{"eval_id": "..."}` on success, `409` when the harness has been
  put into busy-mode via `set_busy(True)`.
* `GET /eval/{eval_id}/stream` — returns an SSE stream
  (`text/event-stream`). Each event the test has queued is emitted
  as one `data: <json>\\n\\n` line. The stream terminates when an
  event with `type` in {"verdict", "error"} is emitted (matching the
  real server's terminate-after-final-event semantics).

Test API for event injection:
* `queue_event(eval_id, *, type, data)` — append an event to the
  per-eval-id queue. Events are emitted in the order queued.
  Works for any `eval_id`, including a synthetic value the consumer
  never `POST /eval`-ed against — useful for tests that drive
  `GET /stream` directly with a hand-picked id. `type` and `data` are
  passed through unvalidated; the harness wraps them as
  `{"type": <type>, "data": <data>}` and JSON-encodes for the wire.
* `set_busy(busy)` — toggle the 409-busy mode for `POST /eval`.

Test API for inspection:
* `posted_evals` — list of every EvalRequest payload that hit
  `POST /eval` during the test, in order. Each entry is
  `{"eval_id": ..., "request": <json body as dict>}`.

Test API for the consumer client:
* `client()` — returns a fresh `httpx.AsyncClient` bound to the harness.
  Caller must close it (`async with server.client() as c:` recommended).
* `app` — the bare FastAPI ASGI app. Use this to build your own
  transport (e.g. for `tick.bind_eval_server_to_validator`, which
  monkeypatches `httpx.AsyncClient` to use this app's ASGITransport).

NOT modeled (deliberately):
* Real GPU eval execution. The harness ships no model code; tests that
  want to verify an actual probe path use the existing `_MicroLM`
  fixture against `eval/torch_runner.py` directly.
* `/preload`, `/probe`, `/health` endpoints. Validator code that hits
  these has its own coverage path; if a harness consumer needs them,
  add a stub here at first use.
* Concurrency-control semantics (real eval_server uses a threading
  lock to refuse a second eval). The harness only models the 409
  surface via `set_busy`; if a test needs the lock-acquire-once
  behaviour, build it in the test using `set_busy`.
* Watchdog / max-runtime / self-kill. Tests that need the validator
  to time out should inject silence (no events) and rely on the
  validator's own idle-watchdog under monkeypatched timing.
"""
from __future__ import annotations

import asyncio
import json
from collections import defaultdict

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse


class FakeEvalServer:
    def __init__(self) -> None:
        self._busy: bool = False
        self._next_eval_seq: int = 0
        # Per-eval-id event queue. Created lazily on first POST or first
        # queue_event call so tests can queue events before posting.
        self._events: dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
        self._posted_evals: list[dict] = []
        self.app = self._build_app()

    # ------------------------------------------------------------------
    # Test API — event injection.

    def queue_event(self, eval_id: str, *, type: str, data: dict) -> None:
        # `type` shadows the builtin within this signature on purpose:
        # callers write `queue_event(eid, type="progress", data={...})`
        # which mirrors the SSE event JSON shape the validator parses.
        self._events[eval_id].put_nowait({"type": type, "data": data})

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)

    # ------------------------------------------------------------------
    # Test API — inspection.

    @property
    def posted_evals(self) -> list[dict]:
        return list(self._posted_evals)

    # ------------------------------------------------------------------
    # Test API — consumer-facing httpx client.

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://fake-eval-server",
        )

    # ------------------------------------------------------------------
    # ASGI app construction — internal.

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.post("/eval")
        async def post_eval(request: Request):
            if self._busy:
                raise HTTPException(status_code=409, detail="busy")
            body = await request.json()
            self._next_eval_seq += 1
            eid = f"eid{self._next_eval_seq:03d}"
            self._posted_evals.append({"eval_id": eid, "request": body})
            return {"eval_id": eid}

        @app.get("/eval/{eval_id}/stream")
        async def get_stream(eval_id: str):
            queue = self._events[eval_id]

            async def generate():
                while True:
                    event = await queue.get()
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("verdict", "error"):
                        return

            return StreamingResponse(generate(), media_type="text/event-stream")

        return app
