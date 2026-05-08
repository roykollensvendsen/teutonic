"""Tests for State.replenish_reeval.

`replenish_reeval(subtensor, netuid)` is called when the eval queue
is empty in `--seen` mode (the validator-side `--seen` flag means
"keep evaluating previously-seen hotkeys"). The method:

1. Clears `state.evaluated_repos` (previous-cycle dedup is reset)
2. Calls `scan_reveals` with a throwaway-seen set so the same chain
   reveals come back even if `state.seen` already has them
3. Filters out the king's own reveal + any repo in `failed_repos`
4. Marks each remaining reveal with `reeval=True` and enqueues
5. If anything was enqueued: flushes state + dashboard + emits event
6. Returns the count

We test it by driving `scan_reveals` via a FakeChain (M1 harness)
that has reveals committed for several hotkeys.
"""
from tests._harness.chain import FakeChain
from validator import State


def _payload(repo: str = "alice/Teutonic-LXXX-x") -> str:
    """scan_reveals expects 'king_hash:hf_repo:model_hash'."""
    return f"kh:{repo}:mh"


# ---------------------------------------------------------------------
# Empty-chain / no-reveals fast path.

def test_replenish_reeval_returns_zero_when_chain_empty(r2_mock):
    s = State(r2_mock)
    chain = FakeChain()
    count = s.replenish_reeval(chain, netuid=3)
    assert count == 0
    assert s.queue == []


def test_replenish_reeval_clears_evaluated_repos(r2_mock):
    s = State(r2_mock)
    s.evaluated_repos = {"alice/x", "bob/y", "carol/z"}
    chain = FakeChain()  # no reveals to enqueue
    s.replenish_reeval(chain, netuid=3)
    # Cleared regardless of whether anything was enqueued.
    assert s.evaluated_repos == set()


# ---------------------------------------------------------------------
# Happy path: enqueues all eligible reveals.

def test_replenish_reeval_enqueues_all_reveals(r2_mock):
    s = State(r2_mock)
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a",
                        _payload("alice/Teutonic-LXXX-A"), block=100)
    chain.commit_reveal("hk_b",
                        _payload("bob/Teutonic-LXXX-B"), block=100)
    count = s.replenish_reeval(chain, netuid=3)
    assert count == 2
    assert len(s.queue) == 2
    # Each entry is marked reeval=True for downstream code paths.
    assert all(entry["reeval"] is True for entry in s.queue)


def test_replenish_reeval_returned_count_matches_enqueued(r2_mock):
    s = State(r2_mock)
    chain = FakeChain(block=100)
    for n in range(5):
        chain.commit_reveal(f"hk_{n}",
                            _payload(f"miner_{n}/Teutonic-LXXX-x"),
                            block=100)
    count = s.replenish_reeval(chain, netuid=3)
    assert count == 5
    assert len(s.queue) == 5


# ---------------------------------------------------------------------
# Filter: king's reveal is dropped.

def test_replenish_reeval_drops_king_hotkeys_reveal(r2_mock):
    s = State(r2_mock)
    s.set_king("hk_king", "alice/Teutonic-LXXX-king", "kh", 100)
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_king",
                        _payload("alice/Teutonic-LXXX-king"), block=100)
    chain.commit_reveal("hk_other",
                        _payload("bob/Teutonic-LXXX-x"), block=100)
    count = s.replenish_reeval(chain, netuid=3)
    assert count == 1
    assert s.queue[0]["hotkey"] == "hk_other"


# ---------------------------------------------------------------------
# Filter: repos in failed_repos are dropped.

def test_replenish_reeval_drops_failed_repos(r2_mock):
    s = State(r2_mock)
    s.failed_repos = {"alice/Teutonic-LXXX-bad"}
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a",
                        _payload("alice/Teutonic-LXXX-bad"), block=100)
    chain.commit_reveal("hk_b",
                        _payload("bob/Teutonic-LXXX-good"), block=100)
    count = s.replenish_reeval(chain, netuid=3)
    assert count == 1
    assert s.queue[0]["hf_repo"] == "bob/Teutonic-LXXX-good"


# ---------------------------------------------------------------------
# Side effects: flush + event when anything enqueued.

def test_replenish_reeval_emits_event_when_anything_enqueued(r2_mock):
    s = State(r2_mock)
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a",
                        _payload("alice/Teutonic-LXXX-x"), block=100)
    s.replenish_reeval(chain, netuid=3)
    # The "replenish_reeval" event lands in state/history.jsonl.
    appended = r2_mock._appended.get("state/history.jsonl", [])
    replenish_events = [e for e in appended
                        if e.get("event") == "replenish_reeval"]
    assert len(replenish_events) == 1
    assert replenish_events[0]["count"] == 1


def test_replenish_reeval_does_not_emit_event_when_nothing_enqueued(r2_mock):
    s = State(r2_mock)
    chain = FakeChain()  # empty
    s.replenish_reeval(chain, netuid=3)
    appended = r2_mock._appended.get("state/history.jsonl", [])
    replenish_events = [e for e in appended
                        if e.get("event") == "replenish_reeval"]
    assert replenish_events == []


# ---------------------------------------------------------------------
# Re-eval ignores state.seen — the same reveals come back across calls.

def test_replenish_reeval_returns_revealed_repos_even_when_seen(r2_mock):
    s = State(r2_mock)
    # state.seen already includes hk_a — but replenish_reeval uses a
    # throwaway seen set, so hk_a's reveal still surfaces.
    s.seen = {"hk_a"}
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a",
                        _payload("alice/Teutonic-LXXX-x"), block=100)
    count = s.replenish_reeval(chain, netuid=3)
    assert count == 1


# ---------------------------------------------------------------------
# Multiple calls within same process — failed_repos still suppresses.

def test_replenish_reeval_consistent_across_calls(r2_mock):
    s = State(r2_mock)
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a",
                        _payload("alice/Teutonic-LXXX-A"), block=100)
    first = s.replenish_reeval(chain, netuid=3)
    # First call enqueues 1. Second call: enqueue would dedup against
    # the queue (enqueue checks "repo already queued"), so count=0.
    second = s.replenish_reeval(chain, netuid=3)
    assert first == 1
    assert second == 0
