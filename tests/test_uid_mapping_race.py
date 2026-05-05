"""Tests for the metagraph-snapshot vs reveal-scan ordering bug.

Discord report from kyle890015 (2026-05-04): newly-registered miners
that post a reveal between two consecutive validator-loop ticks can
show UID=null (`--`) in dashboard.json because the validator's main
loop calls `state.refresh_uid_map(subtensor, NETUID)` (line 2329)
BEFORE `scan_reveals(subtensor, NETUID, state.seen)` (line 2337).

Within a single tick:
1. `refresh_uid_map` snapshots the metagraph at t0 — building
   `state.uid_map` from `meta.hotkeys`.
2. `await fetch_tmc_data()` introduces a network-bound delay
   (~15s timeout possible).
3. `scan_reveals` queries the chain at t1 (fresh state) for all
   revealed commitments.

If a miner registers + reveals between t0 and t1, the reveal is in
`scan_reveals` output but the miner's hotkey is missing from
`state.uid_map` — so `self.uid_map.get(hotkey)` returns None at
flush time. The dashboard renders that as `--`.

These tests pin the basic refresh_uid_map behavior plus a direct
demonstration of the staleness consequence on dashboard output.
A flippable race-path xfail test waits on a fix-strategy decision
(reorder in main loop / second refresh / signature change to
`flush_dashboard`) — see the plan for context.
"""
from types import SimpleNamespace

import pytest

from validator import State


@pytest.fixture
def r2_mock(mocker):
    """Dict-backed mock of validator.R2 with dashboard capture.

    Mirrors the pattern in test_state_dashboard.py so the dashboard
    payload can be inspected via `r2._dashboards["dashboard.json"]`.
    """
    storage: dict = {}
    dashboards: dict = {}
    r2 = mocker.MagicMock()
    r2.get.side_effect = lambda key: storage.get(key)
    r2.put.side_effect = lambda key, data: storage.update({key: data})
    r2.append_jsonl.side_effect = lambda key, rec: None
    r2.put_dashboard.side_effect = (
        lambda key, data: dashboards.update({key: data})
    )
    r2._dashboards = dashboards
    return r2


class _FakeSubtensor:
    """Minimal subtensor stub for refresh_uid_map.

    `metagraph(netuid)` returns a SimpleNamespace with `.hotkeys` and
    `.emission` matching what `State.refresh_uid_map` reads. Callers
    can mutate `self.hotkeys` between calls to simulate on-chain
    registrations between validator-loop ticks.
    """

    def __init__(self, hotkeys):
        self.hotkeys = list(hotkeys)

    def metagraph(self, netuid):
        return SimpleNamespace(
            hotkeys=list(self.hotkeys),
            emission=[0.0] * len(self.hotkeys),
        )

    @property
    def block(self):
        return 100


# ---------------------------------------------------------------------
# refresh_uid_map basic behavior.

def test_refresh_uid_map_populates_from_metagraph(r2_mock):
    s = State(r2_mock)
    sub = _FakeSubtensor(["A", "B", "C"])

    s.refresh_uid_map(sub, netuid=3)

    assert s.uid_map == {"A": 0, "B": 1, "C": 2}


def test_refresh_uid_map_empty_metagraph(r2_mock):
    s = State(r2_mock)
    sub = _FakeSubtensor([])

    s.refresh_uid_map(sub, netuid=3)

    assert s.uid_map == {}


def test_refresh_uid_map_picks_up_newly_registered_hotkey(r2_mock):
    # A second call against a metagraph whose hotkey list grew must
    # extend uid_map. This is the underlying capability the fix would
    # rely on (refresh again after a registration to close the gap).
    s = State(r2_mock)
    sub = _FakeSubtensor(["A", "B", "C"])
    s.refresh_uid_map(sub, netuid=3)

    sub.hotkeys = ["A", "B", "C", "D"]
    s.refresh_uid_map(sub, netuid=3)

    assert s.uid_map == {"A": 0, "B": 1, "C": 2, "D": 3}


# ---------------------------------------------------------------------
# Staleness consequence on dashboard output.

def test_dashboard_shows_uid_none_when_uid_map_is_stale(r2_mock):
    # Pins the bug: if a miner is enqueued after refresh_uid_map
    # captured a metagraph that doesn't include them, the dashboard's
    # queue entry for that miner has uid=None. This is the direct
    # rendering path that produces `--` in the UI.
    s = State(r2_mock)
    sub = _FakeSubtensor(["A", "B", "C"])
    s.refresh_uid_map(sub, netuid=3)

    # Miner D registers + reveals between snapshot and flush.
    s.enqueue({
        "hotkey": "D",
        "hf_repo": "miner/d-repo",
        "challenge_id": "x",
        "reveal_block": 100,
    })

    s.flush_dashboard(force=True)

    payload = r2_mock._dashboards["dashboard.json"]
    d_entries = [e for e in payload["queue"] if e["hotkey"] == "D"]
    assert len(d_entries) == 1
    assert d_entries[0]["uid"] is None


def test_dashboard_shows_correct_uid_when_uid_map_is_fresh(r2_mock):
    # Counter-example: if uid_map is refreshed AFTER the miner
    # registers (so the snapshot includes them), the dashboard renders
    # the correct UID. Demonstrates the fix shape without prescribing
    # which mechanism delivers it (reorder in main loop, second refresh
    # call, or signature change to flush_dashboard).
    s = State(r2_mock)
    sub = _FakeSubtensor(["A", "B", "C"])
    s.refresh_uid_map(sub, netuid=3)

    # D registers; uid_map gets refreshed before flush.
    sub.hotkeys = ["A", "B", "C", "D"]
    s.enqueue({
        "hotkey": "D",
        "hf_repo": "miner/d-repo",
        "challenge_id": "x",
        "reveal_block": 100,
    })
    s.refresh_uid_map(sub, netuid=3)

    s.flush_dashboard(force=True)

    payload = r2_mock._dashboards["dashboard.json"]
    d_entries = [e for e in payload["queue"] if e["hotkey"] == "D"]
    assert len(d_entries) == 1
    assert d_entries[0]["uid"] == 3
