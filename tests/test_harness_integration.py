"""Integration test for the M1-M3 harness foundation.

Demonstrates the foundation's primary use case: driving validator-style
eval flows (chain → enqueue → POST /eval → stream verdict → record)
end-to-end in-process without HF, without a real eval-server, without a
real subtensor.

This test exists as proof-of-value for the M1-M3 investment. Each
harness module has its own unit tests pinning the docstring contract:
* `tests/test_harness_chain.py` — FakeChain
* `tests/test_harness_eval_server.py` — FakeEvalServer
* `tests/test_harness_tick.py` — run_chain_tick + bind_eval_server_to_validator

What was missing until now: a single test demonstrating that all three
modules work TOGETHER on a realistic validator flow. If a later refactor
breaks the harness's ability to drive the chain → eval-server interaction
end-to-end, this test surfaces it.

Out of scope (deliberately):
* `validator.process_challenge` itself — pulls in HfApi, R2 writes, and
  block-hash retrieval that live outside the harness's modelled surface.
  This test drives the eval-streaming portion the harness DOES model
  (POST /eval + SSE stream consumption), which is what the foundation
  was built for.
"""
import json

import httpx
import pytest

from tests._harness.chain import FakeChain
from tests._harness.eval_server import FakeEvalServer
from tests._harness.tick import bind_eval_server_to_validator, run_chain_tick
from validator import State

# Cross-harness integration test: exercises M1 + M2 + M3 together.
# Run focused with `pytest -m integration`.
pytestmark = pytest.mark.integration


async def test_chain_reveal_to_eval_verdict_flow_uses_all_three_harnesses(
    r2_mock, monkeypatch,
):
    """End-to-end: chain reveal → tick enqueues → POST /eval → stream verdict.

    Exercises M1 + M2 + M3 in a single scenario:

    1. M1 (FakeChain): chain with a king and a new challenger reveal.
    2. M3 (run_chain_tick): drives `state.refresh_uid_map` +
       `scan_reveals` + `state.enqueue` against the chain.
    3. M2 (FakeEvalServer + bind_eval_server_to_validator): eval-server
       queued with a verdict event; validator-side `httpx.AsyncClient`
       routes to the fake via the bind monkeypatch.
    4. End-to-end: assert the verdict event surfaces with the right
       payload AND that the eval-server recorded the POST body.
    """
    # M1: chain state — king already registered, plus a fresh challenger
    # whose reveal has just hit the chain.
    chain = FakeChain(block=100)
    chain.register("hk_king", uid=0, coldkey="ck_king")
    chain.register("hk_chal", uid=1, coldkey="ck_chal_long_ss58")
    chain.commit_reveal("hk_chal",
                        "kh_seed:bob/Teutonic-LXXX-chall:mh_chal",
                        block=100)

    # State with king already crowned (so enqueue won't reject the
    # challenger as the king's own reveal).
    state = State(r2_mock)
    state.set_king("hk_king", "alice/Teutonic-LXXX-king",
                   "kh_seed", block=90, challenge_id="seed",
                   king_revision="rev_king")

    # M3: drive the chain-read portion of one tick. The challenger
    # reveal should land in state.queue.
    snapshot = run_chain_tick(state, chain)
    assert snapshot["enqueued_count"] == 1, snapshot
    assert state.queue[0]["hotkey"] == "hk_chal"

    # M2: stand up an in-process eval-server and wire validator's
    # httpx code path to it. Any AsyncClient created from now on
    # routes to the fake.
    eval_server = FakeEvalServer()
    bind_eval_server_to_validator(eval_server, monkeypatch)

    # Drive the eval-streaming portion of validator's process_challenge
    # inline (the parts the foundation was built to exercise). We do
    # NOT call process_challenge directly — it has HF / block-hash
    # dependencies that are deliberately out of harness scope.
    entry = state.queue[0]

    async with httpx.AsyncClient() as client:
        eval_payload = {
            "king_repo": state.king["hf_repo"],
            "challenger_repo": entry["hf_repo"],
            "block_hash": "0xdummy",
            "hotkey": entry["hotkey"],
            "shard_key": "shards/0.npy",
        }
        resp = await client.post("/eval", json=eval_payload)
        assert resp.status_code == 200, resp.text
        eval_id = resp.json()["eval_id"]

        # Test injects the eval-server's response stream — first
        # progress, then a final verdict.
        eval_server.queue_event(eval_id, type="progress",
                                data={"done": 50, "total": 100,
                                      "mu_hat": -0.04})
        eval_server.queue_event(eval_id, type="verdict",
                                data={"accepted": False,
                                      "verdict": "king",
                                      "mu_hat": -0.05,
                                      "lcb": -0.08,
                                      "wall_time_s": 312.0})

        # Consume the stream the way validator.py:2119 does.
        events: list[dict] = []
        verdict = None
        async with client.stream("GET", f"/eval/{eval_id}/stream") as stream:
            async for line in stream.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[len("data: "):])
                events.append(event)
                if event["type"] == "verdict":
                    verdict = event["data"]
                    break

    # End-to-end assertions: the verdict surfaced with the test-injected
    # payload, and the eval-server recorded the validator-side POST body.
    assert verdict is not None
    assert verdict["verdict"] == "king"
    assert verdict["accepted"] is False
    assert verdict["mu_hat"] == -0.05
    assert verdict["wall_time_s"] == 312.0

    # The first event was the progress event — pin the order.
    assert [e["type"] for e in events] == ["progress", "verdict"]

    # FakeEvalServer captured the POST body the validator-side client
    # sent. This proves the bind monkeypatch routed the call correctly.
    assert len(eval_server.posted_evals) == 1
    posted = eval_server.posted_evals[0]
    assert posted["eval_id"] == eval_id
    assert posted["request"]["king_repo"] == "alice/Teutonic-LXXX-king"
    assert posted["request"]["challenger_repo"] == "bob/Teutonic-LXXX-chall"
    assert posted["request"]["hotkey"] == "hk_chal"
