"""Tests for eval_server FastAPI endpoints (health + eval lifecycle).

Four endpoints exercised here:
* `GET /health` — status + GPU info + king + active_evals + disk stats
* `POST /eval` — creates eval record + spawns _run_eval + _watchdog threads
* `GET /eval/{id}` — record snapshot or 404
* `GET /eval/{id}/stream` — SSE stream or 404

These tests run the FastAPI app in-process via httpx.ASGITransport
(same pattern the M2 FakeEvalServer harness uses, but here pointed at
the real `eval_server.app`). Threads spawned by `/eval` are stubbed so
no actual model loads happen.
"""
import contextlib
from queue import Queue

import httpx
import pytest

import eval_server
from tests._chain import repo

# ---------------------------------------------------------------------
# Reset module-level state between tests so each starts clean.

@pytest.fixture(autouse=True)
def _reset_endpoint_state(monkeypatch):
    monkeypatch.setattr(eval_server, "_evals", {})
    eval_server._self_kill_scheduled.clear()
    monkeypatch.setattr(eval_server, "_disk_stats_snapshot",
                        {"disk_total_gb": 1000.0, "disk_used_gb": 200.0})
    monkeypatch.setattr(eval_server, "_disk_stats_thread_started", True)
    monkeypatch.setattr(eval_server, "_gpu_ids", [0, 1])
    monkeypatch.setattr(eval_server, "_king_repo", repo("alice", "king"))
    # Reset _eval_lock so each test starts unlocked.
    if eval_server._eval_lock.locked():
        with contextlib.suppress(RuntimeError):
            eval_server._eval_lock.release()
    yield
    # Clean up any leaked lock acquisitions from a test.
    if eval_server._eval_lock.locked():
        with contextlib.suppress(RuntimeError):
            eval_server._eval_lock.release()


@pytest.fixture
def fake_threads(monkeypatch):
    """Replace `threading.Thread` so /eval doesn't spawn real workers."""
    spawned = []

    class FakeThread:
        def __init__(self, *, target, args=(), daemon=None):
            self.target = target
            self.args = args
            self.daemon = daemon
            spawned.append(self)

        def start(self):
            self.started = True

    monkeypatch.setattr(eval_server.threading, "Thread", FakeThread)
    return spawned


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=eval_server.app),
        base_url="http://eval-server-test",
    ) as c:
        yield c


_MIN_EVAL_REQUEST = {
    "king_repo": repo("alice", "king"),
    "challenger_repo": repo("bob", "chall"),
    "block_hash": "0xabc",
    "hotkey": "hk_bob",
    "shard_key": "shards/0.npy",
}


# ---------------------------------------------------------------------
# GET /health

async def test_health_returns_ok_when_not_self_killing(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["self_kill_scheduled"] is False
    assert body["gpus"] == 2
    assert body["gpu_ids"] == [0, 1]
    assert body["king_loaded"] == repo("alice", "king")


async def test_health_returns_exiting_when_self_kill_scheduled(client):
    eval_server._self_kill_scheduled.set()
    resp = await client.get("/health")
    body = resp.json()
    assert body["status"] == "exiting"
    assert body["self_kill_scheduled"] is True


async def test_health_includes_disk_stats(client):
    # The autouse fixture pre-populated _disk_stats_snapshot with
    # disk_total_gb / disk_used_gb. /health merges them into the body.
    resp = await client.get("/health")
    body = resp.json()
    assert body["disk_total_gb"] == 1000.0
    assert body["disk_used_gb"] == 200.0


async def test_health_reports_active_evals_count(client, monkeypatch):
    monkeypatch.setattr(eval_server, "_evals",
                        {"e1": {"state": "running"},
                         "e2": {"state": "completed"}})
    resp = await client.get("/health")
    body = resp.json()
    assert body["active_evals"] == 2


async def test_health_reports_gpu_busy_state(client, monkeypatch):
    eval_server._gpu_busy.set()
    try:
        resp = await client.get("/health")
        assert resp.json()["gpu_busy"] is True
    finally:
        eval_server._gpu_busy.clear()


# ---------------------------------------------------------------------
# POST /eval — start eval (or 503/409 guards).

async def test_post_eval_returns_503_when_self_kill_scheduled(client):
    eval_server._self_kill_scheduled.set()
    resp = await client.post("/eval", json=_MIN_EVAL_REQUEST)
    assert resp.status_code == 503
    assert "restarting" in resp.json()["detail"]


async def test_post_eval_returns_409_when_lock_busy(client):
    # Pre-acquire the lock to simulate an in-flight eval.
    eval_server._eval_lock.acquire()
    try:
        resp = await client.post("/eval", json=_MIN_EVAL_REQUEST)
        assert resp.status_code == 409
        assert "already running" in resp.json()["detail"]
    finally:
        eval_server._eval_lock.release()


async def test_post_eval_creates_record_with_pending_state(client, fake_threads):
    resp = await client.post("/eval", json=_MIN_EVAL_REQUEST)
    assert resp.status_code == 200
    eval_id = resp.json()["eval_id"]
    assert eval_id in eval_server._evals
    record = eval_server._evals[eval_id]
    assert record["state"] == "pending"
    assert record["verdict"] is None
    assert record["error"] is None
    assert isinstance(record["events"], Queue)
    assert record["request"]["king_repo"] == repo("alice", "king")


async def test_post_eval_spawns_run_eval_and_watchdog_threads(
    client, fake_threads,
):
    resp = await client.post("/eval", json=_MIN_EVAL_REQUEST)
    assert resp.status_code == 200
    # Two threads spawned: _run_eval + _watchdog (both daemon).
    assert len(fake_threads) == 2
    assert all(t.daemon is True for t in fake_threads)
    # Both started.
    assert all(getattr(t, "started", False) for t in fake_threads)


async def test_post_eval_returns_distinct_ids_per_request(client, fake_threads):
    a = await client.post("/eval", json=_MIN_EVAL_REQUEST)
    # Need to release the lock between requests because each spawned
    # thread is mocked away (real _run_eval would release it on exit).
    eval_server._eval_lock.release()
    b = await client.post("/eval", json=_MIN_EVAL_REQUEST)
    assert a.json()["eval_id"] != b.json()["eval_id"]


# ---------------------------------------------------------------------
# GET /eval/{eval_id}

async def test_get_eval_returns_404_for_unknown_id(client):
    resp = await client.get("/eval/unknown_id")
    assert resp.status_code == 404


async def test_get_eval_returns_record_snapshot(client, monkeypatch):
    monkeypatch.setattr(eval_server, "_evals", {
        "eid001": {
            "state": "running",
            "progress": {"done": 50, "total": 100},
            "verdict": None,
            "error": None,
            "request": {},
            "events": Queue(),
            "created_at": 0.0,
        },
    })
    resp = await client.get("/eval/eid001")
    assert resp.status_code == 200
    body = resp.json()
    assert body["eval_id"] == "eid001"
    assert body["state"] == "running"
    assert body["progress"] == {"done": 50, "total": 100}
    assert body["verdict"] is None
    assert body["error"] is None


async def test_get_eval_returns_completed_state_with_verdict(client, monkeypatch):
    verdict = {"accepted": True, "mu_hat": 0.05}
    monkeypatch.setattr(eval_server, "_evals", {
        "eid002": {
            "state": "completed",
            "progress": {"done": 100, "total": 100},
            "verdict": verdict,
            "error": None,
            "request": {},
            "events": Queue(),
            "created_at": 0.0,
        },
    })
    resp = await client.get("/eval/eid002")
    body = resp.json()
    assert body["state"] == "completed"
    assert body["verdict"] == verdict


async def test_get_eval_returns_failed_state_with_error(client, monkeypatch):
    monkeypatch.setattr(eval_server, "_evals", {
        "eid003": {
            "state": "failed",
            "progress": {},
            "verdict": None,
            "error": "snapshot_download timed out",
            "request": {},
            "events": Queue(),
            "created_at": 0.0,
        },
    })
    resp = await client.get("/eval/eid003")
    body = resp.json()
    assert body["state"] == "failed"
    assert body["error"] == "snapshot_download timed out"


# ---------------------------------------------------------------------
# GET /eval/{eval_id}/stream

async def test_stream_eval_returns_404_for_unknown_id(client):
    async with client.stream("GET", "/eval/unknown_id/stream") as resp:
        assert resp.status_code == 404
