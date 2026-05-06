"""Tests for the metagraph-snapshot vs reveal-scan ordering bug.

Discord report from kyle890015 (2026-05-04): newly-registered miners
that post a reveal between two consecutive validator-loop ticks can
show UID=null (`--`) in dashboard.json because the validator's main
loop calls `state.refresh_uid_map(subtensor, NETUID)` BEFORE
`scan_reveals(subtensor, NETUID, state.seen)` within each tick body
(see `validator.main()`).

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
The fix design (reorder in main loop / second refresh / signature
change to `flush_dashboard`) is a separate plan; this file pins the
behaviour so the fix has a regression test to flip green.

Uses `FakeChain` from `tests/_harness/chain.py` — the standard
in-process subtensor double from the M1 foundation milestone.
"""
import pytest

from tests._harness.chain import FakeChain
from validator import State

# ---------------------------------------------------------------------
# refresh_uid_map basic behavior.

def test_refresh_uid_map_populates_from_metagraph(r2_mock):
    s = State(r2_mock)
    chain = FakeChain()
    chain.register("A", uid=0)
    chain.register("B", uid=1)
    chain.register("C", uid=2)

    s.refresh_uid_map(chain, netuid=3)

    assert s.uid_map == {"A": 0, "B": 1, "C": 2}


def test_refresh_uid_map_empty_metagraph(r2_mock):
    s = State(r2_mock)
    chain = FakeChain()  # no registrations

    s.refresh_uid_map(chain, netuid=3)

    assert s.uid_map == {}


def test_refresh_uid_map_picks_up_newly_registered_hotkey(r2_mock):
    # A second call against a chain whose hotkey list grew must
    # extend uid_map. This is the underlying capability the fix would
    # rely on (refresh again after a registration to close the gap).
    s = State(r2_mock)
    chain = FakeChain()
    for hk, uid in [("A", 0), ("B", 1), ("C", 2)]:
        chain.register(hk, uid=uid)
    s.refresh_uid_map(chain, netuid=3)

    chain.register("D", uid=3)
    s.refresh_uid_map(chain, netuid=3)

    assert s.uid_map == {"A": 0, "B": 1, "C": 2, "D": 3}


# ---------------------------------------------------------------------
# Staleness consequence on dashboard output.

def test_dashboard_shows_uid_none_when_uid_map_is_stale(r2_mock):
    # Pins the bug: if a miner is enqueued after refresh_uid_map
    # captured a metagraph that doesn't include them, the dashboard's
    # queue entry for that miner has uid=None. This is the direct
    # rendering path that produces `--` in the UI.
    s = State(r2_mock)
    chain = FakeChain()
    for hk, uid in [("A", 0), ("B", 1), ("C", 2)]:
        chain.register(hk, uid=uid)
    s.refresh_uid_map(chain, netuid=3)

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
    chain = FakeChain()
    for hk, uid in [("A", 0), ("B", 1), ("C", 2)]:
        chain.register(hk, uid=uid)
    s.refresh_uid_map(chain, netuid=3)

    # D registers; uid_map gets refreshed before flush.
    chain.register("D", uid=3)
    s.enqueue({
        "hotkey": "D",
        "hf_repo": "miner/d-repo",
        "challenge_id": "x",
        "reveal_block": 100,
    })
    s.refresh_uid_map(chain, netuid=3)

    s.flush_dashboard(force=True)

    payload = r2_mock._dashboards["dashboard.json"]
    d_entries = [e for e in payload["queue"] if e["hotkey"] == "D"]
    assert len(d_entries) == 1
    assert d_entries[0]["uid"] == 3


# ---------------------------------------------------------------------
# Staleness consequence on coldkey-prefix gate.

def test_expected_coldkey_prefix_is_none_during_staleness_window(r2_mock):
    # Second consequence of the same bug: process_challenge's
    # coldkey-prefix gate (validator.py around line 1925-1948) calls
    # `state.expected_coldkey_prefix(hotkey)` which ultimately reads
    # `state.hotkey_coldkey` — populated by the same refresh_uid_map
    # call. During the staleness window, a fresh registration that
    # already has a coldkey on chain is invisible to the gate, so
    # the gate's "skip if not in metagraph yet" branch fires
    # (validator.py logs "coldkey for X not in metagraph yet,
    # skipping coldkey check"). The defence-in-depth check therefore
    # does NOT run during this window — a separate visible
    # consequence from the dashboard uid=None render.
    #
    # Requires FakeChain's coldkey kwarg (M1 feature beyond what
    # the original _FakeSubtensor stub modelled).
    s = State(r2_mock)
    chain = FakeChain()
    chain.register("A", uid=0, coldkey="ck_A_long_ss58_prefix")
    s.refresh_uid_map(chain, netuid=3)

    # D registers with a coldkey on chain AFTER the snapshot.
    chain.register("D", uid=1, coldkey="ck_D_long_ss58_prefix")

    # state.hotkey_coldkey doesn't include D yet → expected_coldkey_prefix
    # returns None for D, regardless of D's on-chain coldkey.
    assert s.expected_coldkey_prefix("D") is None
    # Sanity: A is in the snapshot so its prefix resolves.
    assert s.expected_coldkey_prefix("A") is not None


# ---------------------------------------------------------------------
# xfail-pinned bug expectations — these are the assertions we WANT
# the system to satisfy. They fail today (the bug exists) and will
# auto-flip to passing when the staleness fix lands. The
# characterization tests above pin the *current* (buggy) behaviour
# as regression coverage; these xfail tests document the *desired*
# behaviour so a CI reader sees "yes, this is a real bug".

@pytest.mark.xfail(
    strict=True,
    reason="Kyle's UID-mapping-race bug. Fix is a separate plan. "
           "When the fix lands (reorder refresh_uid_map vs scan_reveals, "
           "or refresh again before flush_dashboard, or signature change "
           "to flush_dashboard), this xfail flips to pass.",
)
def test_dashboard_should_show_correct_uid_for_freshly_registered_miner(r2_mock):
    s = State(r2_mock)
    chain = FakeChain()
    for hk, uid in [("A", 0), ("B", 1), ("C", 2)]:
        chain.register(hk, uid=uid)
    s.refresh_uid_map(chain, netuid=3)

    # D registers + reveals AFTER the snapshot — the staleness window.
    chain.register("D", uid=3)
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
    # WANT: dashboard renders D's actual on-chain uid (3).
    # GET today: None (bug — uid_map is stale).
    assert d_entries[0]["uid"] == 3


@pytest.mark.xfail(
    strict=True,
    reason="Kyle's UID-mapping-race bug. Fix is a separate plan. "
           "Same staleness as the dashboard test, second visible "
           "consequence: process_challenge's coldkey-prefix gate skips "
           "the check during the staleness window. When the fix lands, "
           "expected_coldkey_prefix resolves for fresh registrations and "
           "this xfail flips to pass.",
)
def test_expected_coldkey_prefix_should_resolve_for_freshly_registered_hotkey(r2_mock):
    s = State(r2_mock)
    chain = FakeChain()
    chain.register("A", uid=0, coldkey="ck_A_long_ss58_prefix")
    s.refresh_uid_map(chain, netuid=3)

    chain.register("D", uid=1, coldkey="ck_D_long_ss58_prefix")

    # WANT: prefix resolves to the first 8 chars (COLDKEY_PREFIX_LEN
    # default) of D's on-chain coldkey.
    # GET today: None (bug — hotkey_coldkey is stale).
    assert s.expected_coldkey_prefix("D") == "ck_D_lon"
