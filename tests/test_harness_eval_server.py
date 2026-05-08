"""Tests for tests._harness.eval_server.FakeEvalServer.

Pin the wire-protocol contract documented in `eval_server.py`'s
docstring: POST /eval returns eval_id (or 409 when busy), GET stream
emits SSE events the test injected and terminates after verdict/error.

These tests run against the in-process ASGI mount the same way the
validator's real httpx code path does, so the behavior they pin is
the behavior validator code will see when wired against this harness.
"""
import json

from tests._harness.eval_server import FakeEvalServer

_MIN_EVAL_REQUEST = {
    "king_repo": "alice/Teutonic-LXXX-king",
    "challenger_repo": "bob/Teutonic-LXXX-chall",
    "block_hash": "0xabc",
    "hotkey": "hk_bob",
    "shard_key": "shards/0.npy",
}


# ---------------------------------------------------------------------
# POST /eval — eval_id allocation, busy-mode 409, body recorded.

async def test_post_eval_returns_eval_id():
    server = FakeEvalServer()
    async with server.client() as c:
        resp = await c.post("/eval", json=_MIN_EVAL_REQUEST)
    assert resp.status_code == 200
    assert "eval_id" in resp.json()


async def test_post_eval_records_request_body():
    server = FakeEvalServer()
    async with server.client() as c:
        await c.post("/eval", json=_MIN_EVAL_REQUEST)
    assert len(server.posted_evals) == 1
    assert server.posted_evals[0]["request"] == _MIN_EVAL_REQUEST


async def test_post_eval_returns_409_when_busy():
    server = FakeEvalServer()
    server.set_busy(True)
    async with server.client() as c:
        resp = await c.post("/eval", json=_MIN_EVAL_REQUEST)
    assert resp.status_code == 409


async def test_post_eval_recovers_after_busy_cleared():
    # Validator's busy-retry loop relies on transitioning out of 409.
    server = FakeEvalServer()
    server.set_busy(True)
    async with server.client() as c:
        first = await c.post("/eval", json=_MIN_EVAL_REQUEST)
        server.set_busy(False)
        second = await c.post("/eval", json=_MIN_EVAL_REQUEST)
    assert first.status_code == 409
    assert second.status_code == 200


async def test_post_eval_assigns_distinct_ids_per_request():
    server = FakeEvalServer()
    async with server.client() as c:
        a = await c.post("/eval", json=_MIN_EVAL_REQUEST)
        b = await c.post("/eval", json=_MIN_EVAL_REQUEST)
    assert a.json()["eval_id"] != b.json()["eval_id"]


# ---------------------------------------------------------------------
# GET /eval/{id}/stream — SSE-consumes the event queue.

def _parse_sse_line(raw: str) -> dict:
    # SSE wire format the validator parses: "data: {json}\n\n".
    # httpx.aiter_lines() strips trailing newlines but preserves the
    # "data: " prefix.
    assert raw.startswith("data: "), raw
    return json.loads(raw[len("data: "):])


async def _drain_stream(client, eval_id: str) -> list[dict]:
    events: list[dict] = []
    async with client.stream("GET", f"/eval/{eval_id}/stream") as stream:
        async for line in stream.aiter_lines():
            if not line.startswith("data: "):
                continue
            events.append(_parse_sse_line(line))
    return events


async def test_stream_emits_queued_progress_event_then_verdict():
    server = FakeEvalServer()
    async with server.client() as c:
        eid = (await c.post("/eval", json=_MIN_EVAL_REQUEST)).json()["eval_id"]
        server.queue_event(eid, type="progress",
                           data={"done": 10, "total": 100})
        server.queue_event(eid, type="verdict",
                           data={"accepted": True, "mu_hat": 0.05})
        events = await _drain_stream(c, eid)
    assert events == [
        {"type": "progress", "data": {"done": 10, "total": 100}},
        {"type": "verdict", "data": {"accepted": True, "mu_hat": 0.05}},
    ]


async def test_stream_terminates_after_verdict():
    # An event queued AFTER a verdict must not be emitted — verdict
    # is the terminal event for a successful eval.
    server = FakeEvalServer()
    async with server.client() as c:
        eid = (await c.post("/eval", json=_MIN_EVAL_REQUEST)).json()["eval_id"]
        server.queue_event(eid, type="verdict", data={"accepted": True})
        server.queue_event(eid, type="progress", data={"done": 999})
        events = await _drain_stream(c, eid)
    assert len(events) == 1
    assert events[0]["type"] == "verdict"


async def test_stream_terminates_after_error():
    # Same terminal semantics for the error path.
    server = FakeEvalServer()
    async with server.client() as c:
        eid = (await c.post("/eval", json=_MIN_EVAL_REQUEST)).json()["eval_id"]
        server.queue_event(eid, type="error",
                           data="snapshot_download failed")
        server.queue_event(eid, type="progress", data={"done": 1})
        events = await _drain_stream(c, eid)
    assert len(events) == 1
    assert events[0]["type"] == "error"


async def test_stream_emits_multiple_events_in_queue_order():
    server = FakeEvalServer()
    async with server.client() as c:
        eid = (await c.post("/eval", json=_MIN_EVAL_REQUEST)).json()["eval_id"]
        server.queue_event(eid, type="stage", data={"name": "loading_king"})
        server.queue_event(eid, type="stage", data={"name": "loading_challenger"})
        server.queue_event(eid, type="stage", data={"name": "bootstrap"})
        server.queue_event(eid, type="progress", data={"done": 50, "total": 100})
        server.queue_event(eid, type="verdict", data={"accepted": False})
        events = await _drain_stream(c, eid)
    assert [e["type"] for e in events] == [
        "stage", "stage", "stage", "progress", "verdict",
    ]
    # Sanity: stage names preserved.
    stage_names = [e["data"]["name"] for e in events if e["type"] == "stage"]
    assert stage_names == ["loading_king", "loading_challenger", "bootstrap"]


async def test_queue_event_before_post_eval_works():
    # Tests sometimes prepare the event queue against a synthetic eval_id
    # before driving the validator. The harness must accept queue_event
    # for an eval_id the consumer hasn't /eval-posted yet.
    server = FakeEvalServer()
    server.queue_event("eid_synthetic", type="verdict", data={"accepted": True})
    async with server.client() as c:
        events = await _drain_stream(c, "eid_synthetic")
    assert events == [{"type": "verdict", "data": {"accepted": True}}]


async def test_separate_eval_ids_have_independent_queues():
    server = FakeEvalServer()
    async with server.client() as c:
        eid_a = (await c.post("/eval", json=_MIN_EVAL_REQUEST)).json()["eval_id"]
        eid_b = (await c.post("/eval", json=_MIN_EVAL_REQUEST)).json()["eval_id"]
        server.queue_event(eid_a, type="verdict", data={"who": "a"})
        server.queue_event(eid_b, type="verdict", data={"who": "b"})
        events_a = await _drain_stream(c, eid_a)
        events_b = await _drain_stream(c, eid_b)
    assert events_a == [{"type": "verdict", "data": {"who": "a"}}]
    assert events_b == [{"type": "verdict", "data": {"who": "b"}}]


async def test_posted_evals_returns_a_copy_not_internal_list():
    server = FakeEvalServer()
    async with server.client() as c:
        await c.post("/eval", json=_MIN_EVAL_REQUEST)
    snapshot = server.posted_evals
    snapshot.append({"eval_id": "injected", "request": {}})
    assert len(server.posted_evals) == 1


# ---------------------------------------------------------------------
# Integration: stream consumed via the same idiom validator.py uses.

async def test_consumer_can_read_stream_with_aiter_lines():
    # validator.py:2119 does `async with client.stream(...) as stream:
    #     async for line in stream.aiter_lines(): ...`
    # Pin that the harness emits one SSE event per yielded line so the
    # validator's iterate-and-strip pattern works unchanged.
    server = FakeEvalServer()
    async with server.client() as c:
        eid = (await c.post("/eval", json=_MIN_EVAL_REQUEST)).json()["eval_id"]
        server.queue_event(eid, type="verdict", data={"accepted": True})
        data_lines: list[str] = []
        async with c.stream("GET", f"/eval/{eid}/stream") as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    data_lines.append(line)
    assert len(data_lines) == 1
    payload = json.loads(data_lines[0][len("data: "):])
    assert payload == {"type": "verdict", "data": {"accepted": True}}


