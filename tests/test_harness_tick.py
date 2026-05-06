"""Tests for tests._harness.tick.

Pin the contract of `run_chain_tick` (chain-driven portion of one
validator tick) and `bind_eval_server_to_validator` (httpx redirect
to a FakeEvalServer instance).
"""
import httpx
import pytest

from tests._harness.chain import FakeChain
from tests._harness.eval_server import FakeEvalServer
from tests._harness.tick import bind_eval_server_to_validator, run_chain_tick
from validator import State


@pytest.fixture
def r2_mock(mocker):
    """Dict-backed r2 mock — mirrors the pattern in test_state_helpers.py."""
    storage = {}
    appended = {}

    def fake_get(key):
        return storage.get(key)

    def fake_put(key, data):
        storage[key] = data

    def fake_append_jsonl(key, record):
        appended.setdefault(key, []).append(record)

    r2 = mocker.MagicMock()
    r2.get.side_effect = fake_get
    r2.put.side_effect = fake_put
    r2.append_jsonl.side_effect = fake_append_jsonl
    r2._appended = appended
    return r2


def _valid_reveal_payload(repo: str = "alice/Teutonic-XXIV-chall") -> str:
    # scan_reveals expects "king_hash:hf_repo:model_hash" with the repo
    # matching REPO_PATTERN ("Teutonic-XXIV-..." prefix).
    return f"kh:{repo}:mh"


# ---------------------------------------------------------------------
# run_chain_tick — chain-driven part of a tick.

def test_run_chain_tick_enqueues_a_new_reveal(r2_mock):
    chain = FakeChain(block=100)
    chain.register("hk_alpha", uid=0, coldkey="ck_alpha")
    chain.commit_reveal("hk_alpha", _valid_reveal_payload(), block=100)

    state = State(r2_mock)
    snapshot = run_chain_tick(state, chain)

    assert snapshot["enqueued_count"] == 1
    assert len(snapshot["queue"]) == 1
    assert snapshot["queue"][0]["hotkey"] == "hk_alpha"


def test_run_chain_tick_returns_reveals_list(r2_mock):
    chain = FakeChain(block=100)
    chain.register("hk_alpha", uid=0)
    chain.commit_reveal("hk_alpha", _valid_reveal_payload(), block=100)

    state = State(r2_mock)
    snapshot = run_chain_tick(state, chain)

    assert len(snapshot["reveals"]) == 1
    assert snapshot["reveals"][0]["hotkey"] == "hk_alpha"


def test_run_chain_tick_no_reveals_yields_empty_queue(r2_mock):
    # Empty chain → scan_reveals returns []; no enqueues; no side effects
    # that would surprise a downstream test asserting queue==[].
    chain = FakeChain(block=100)
    chain.register("hk_alpha", uid=0)

    state = State(r2_mock)
    snapshot = run_chain_tick(state, chain)

    assert snapshot["enqueued_count"] == 0
    assert snapshot["queue"] == []
    assert snapshot["reveals"] == []


def test_run_chain_tick_populates_uid_map(r2_mock):
    chain = FakeChain(block=100)
    chain.register("hk_alpha", uid=0, coldkey="ck_alpha")
    chain.register("hk_beta", uid=1, coldkey="ck_beta")

    state = State(r2_mock)
    snapshot = run_chain_tick(state, chain)

    assert snapshot["uid_map"] == {"hk_alpha": 0, "hk_beta": 1}
    assert state.uid_map == {"hk_alpha": 0, "hk_beta": 1}
    assert state.hotkey_coldkey == {"hk_alpha": "ck_alpha", "hk_beta": "ck_beta"}


def test_run_chain_tick_dedups_via_state_seen_across_calls(r2_mock):
    # scan_reveals filters by state.seen — calling run_chain_tick twice
    # over an unchanging chain must yield 0 enqueues on the second call.
    chain = FakeChain(block=100)
    chain.register("hk_alpha", uid=0)
    chain.commit_reveal("hk_alpha", _valid_reveal_payload(), block=100)

    state = State(r2_mock)
    first = run_chain_tick(state, chain)
    second = run_chain_tick(state, chain)

    assert first["enqueued_count"] == 1
    assert second["enqueued_count"] == 0


def test_run_chain_tick_picks_up_new_reveal_added_between_ticks(r2_mock):
    # A reveal committed AFTER the first tick must be picked up by the
    # second tick (this is the validator's reveal-polling contract).
    chain = FakeChain(block=100)
    chain.register("hk_alpha", uid=0)
    chain.register("hk_beta", uid=1)
    chain.commit_reveal("hk_alpha",
                        _valid_reveal_payload("alice/Teutonic-XXIV-A"),
                        block=100)

    state = State(r2_mock)
    first = run_chain_tick(state, chain)

    chain.commit_reveal("hk_beta",
                        _valid_reveal_payload("bob/Teutonic-XXIV-B"),
                        block=200)
    second = run_chain_tick(state, chain)

    assert first["enqueued_count"] == 1
    assert second["enqueued_count"] == 1
    # The second-tick queue includes both reveals (first wasn't drained).
    assert {e["hotkey"] for e in state.queue} == {"hk_alpha", "hk_beta"}


def test_run_chain_tick_skips_king_hotkey_reveal(r2_mock):
    # state.enqueue refuses to queue the current king's own reveal.
    chain = FakeChain(block=100)
    chain.register("hk_king", uid=0)
    chain.commit_reveal("hk_king", _valid_reveal_payload(), block=100)

    state = State(r2_mock)
    state.set_king("hk_king", "alice/Teutonic-XXIV-king", "kh", 90,
                   challenge_id="seed", king_revision="rev")
    snapshot = run_chain_tick(state, chain)

    # scan_reveals returns the entry; enqueue rejects it because of king-hotkey rule.
    assert len(snapshot["reveals"]) == 1
    assert snapshot["enqueued_count"] == 0
    assert state.queue == []


def test_run_chain_tick_uses_custom_netuid(r2_mock):
    # The netuid is a passthrough to FakeChain.metagraph(...) /
    # scan_reveals(...). FakeChain's read API doesn't filter by netuid
    # so this is essentially "the kwarg is wired through" rather than
    # a behavioural difference; pin it as a contract.
    chain = FakeChain(block=100)
    chain.register("hk_alpha", uid=0)

    state = State(r2_mock)
    snapshot = run_chain_tick(state, chain, netuid=42)

    assert snapshot["uid_map"] == {"hk_alpha": 0}


# ---------------------------------------------------------------------
# bind_eval_server_to_validator — wire FakeEvalServer to validator's
# httpx.AsyncClient creation.

async def test_bind_routes_validator_module_constant_to_fake(monkeypatch):
    # Verify EVAL_SERVER_URL gets set to the fake's base_url. Validator
    # code that f-strings this into URLs hits the fake on resolution.
    import validator

    fake_server = FakeEvalServer()
    bind_eval_server_to_validator(fake_server, monkeypatch)

    assert validator.EVAL_SERVER_URL == "http://fake-eval-server"


async def test_bind_routes_httpx_post_through_fake(monkeypatch):
    # An httpx.AsyncClient created after binding must hit the fake's
    # POST /eval handler, not the network.
    fake_server = FakeEvalServer()
    bind_eval_server_to_validator(fake_server, monkeypatch)

    payload = {
        "king_repo": "alice/Teutonic-XXIV-king",
        "challenger_repo": "bob/Teutonic-XXIV-chall",
        "block_hash": "0x", "hotkey": "hk", "shard_key": "k",
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post("/eval", json=payload)

    assert resp.status_code == 200
    assert resp.json()["eval_id"].startswith("eid")
    assert len(fake_server.posted_evals) == 1


async def test_bind_routes_httpx_stream_through_fake(monkeypatch):
    # Pin that the SSE stream pulls events from the fake's queue. This
    # is the validator's actual code path: post → read eval_id →
    # async-stream the result events.
    fake_server = FakeEvalServer()
    bind_eval_server_to_validator(fake_server, monkeypatch)

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            "/eval",
            json={"king_repo": "a/Teutonic-XXIV-k",
                  "challenger_repo": "b/Teutonic-XXIV-c",
                  "block_hash": "0x", "hotkey": "h", "shard_key": "s"},
        )
        eval_id = resp.json()["eval_id"]
        fake_server.queue_event(eval_id, type="verdict",
                                data={"accepted": True})

        events = []
        async with client.stream("GET", f"/eval/{eval_id}/stream") as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    events.append(line)

    assert len(events) == 1
    assert "verdict" in events[0]


async def test_bind_replaces_httpx_async_client_class(monkeypatch):
    # The bind function swaps `httpx.AsyncClient` for a factory that
    # forces transport+base_url onto every constructed client. Pin
    # that the swap actually happens (not just that one POST routed
    # successfully).
    import validator

    original_class = httpx.AsyncClient
    fake_server = FakeEvalServer()
    bind_eval_server_to_validator(fake_server, monkeypatch)

    assert httpx.AsyncClient is not original_class
    assert validator.EVAL_SERVER_URL == "http://fake-eval-server"


def test_bind_is_inert_until_called():
    # Sanity: importing the harness module doesn't itself patch
    # anything — if a previous test forgot `monkeypatch`, this assert
    # fires.
    import validator

    assert validator.EVAL_SERVER_URL != "http://fake-eval-server"
