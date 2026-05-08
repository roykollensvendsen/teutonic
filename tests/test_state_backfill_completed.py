"""Tests for State.backfill_completed_from_chain.

The one-shot startup migration introduced by upstream commit 26bf392
(\"validator: kill re-eval exploit\"). Pre-migration validators tracked
seen hotkeys but not completed repos; on the first restart after the
upstream deploy, the live validator had ~403 hotkeys in `seen` but
only ~33 unique repos in `history`. Without backfill, scan_reveals
would re-pull every previously-seen reveal that didn't make it into
history — at ~10 min/eval, a ~62-hour re-eval stampede.

`backfill_completed_from_chain(subtensor, netuid)` (validator.py:1487):

1. Early-return when `state.seen` is empty.
2. Fetch all reveals (catches Exception broadly so a chain glitch
   doesn't crash startup).
3. Early-return when there are no reveals at all.
4. For each hotkey in `seen`, find the latest commitment, parse
   king_hash:hf_repo:model_hash, and add the hf_repo to
   `completed_repos` if not already present.
5. If anything was added, flush state (also non-fatal).

Idempotent: a second call after a successful first sees every repo
already in completed_repos and no-ops.
"""
from unittest.mock import MagicMock

from tests._chain import repo
from tests._harness.chain import FakeChain
from validator import State


def _payload(hf_repo: str, *, king_hash: str = "kh", model_hash: str = "mh") -> str:
    return f"{king_hash}:{hf_repo}:{model_hash}"


# ---------------------------------------------------------------------
# Happy path: seen hotkeys with valid reveals → backfilled.

def test_backfills_seen_hotkeys_with_chain_reveals(r2_mock):
    s = State(r2_mock)
    s.seen = {"hk_a", "hk_b"}
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a", _payload(repo("alice", "A")), block=100)
    chain.commit_reveal("hk_b", _payload(repo("bob", "B")), block=100)
    s.backfill_completed_from_chain(chain, netuid=3)
    assert repo("alice", "A") in s.completed_repos
    assert repo("bob", "B") in s.completed_repos


def test_backfill_picks_latest_commitment_per_hotkey(r2_mock):
    s = State(r2_mock)
    s.seen = {"hk_a"}
    chain = FakeChain(block=200)
    chain.commit_reveal("hk_a", _payload(repo("alice", "old")), block=100)
    chain.commit_reveal("hk_a", _payload(repo("alice", "new")), block=200)
    s.backfill_completed_from_chain(chain, netuid=3)
    # Latest reveal wins (max-block selection mirrors scan_reveals).
    assert repo("alice", "new") in s.completed_repos
    assert repo("alice", "old") not in s.completed_repos


# ---------------------------------------------------------------------
# Idempotency.

def test_idempotent_on_second_call(r2_mock):
    s = State(r2_mock)
    s.seen = {"hk_a"}
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a", _payload(repo("alice", "A")), block=100)
    s.backfill_completed_from_chain(chain, netuid=3)
    completed_after_first = set(s.completed_repos)
    s.backfill_completed_from_chain(chain, netuid=3)
    # Second call sees the repo already in completed_repos and no-ops.
    assert s.completed_repos == completed_after_first


def test_skips_hotkey_when_repo_already_in_completed(r2_mock):
    s = State(r2_mock)
    s.seen = {"hk_a"}
    s.completed_repos = {repo("alice", "A")}
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a", _payload(repo("alice", "A")), block=100)
    # Capture flush before vs after — backfill should NOT have added
    # anything new, so the flush guard (`if added:`) keeps state file
    # untouched.
    s.backfill_completed_from_chain(chain, netuid=3)
    assert s.completed_repos == {repo("alice", "A")}


# ---------------------------------------------------------------------
# Filters: hotkeys without reveals, malformed payloads.

def test_skips_hotkey_with_no_chain_reveal(r2_mock):
    s = State(r2_mock)
    s.seen = {"hk_a", "hk_ghost"}
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a", _payload(repo("alice", "A")), block=100)
    # hk_ghost is in `seen` but has no on-chain commitment.
    s.backfill_completed_from_chain(chain, netuid=3)
    assert repo("alice", "A") in s.completed_repos
    # No exception, no spurious entry.
    assert len(s.completed_repos) == 1


def test_skips_invalid_payload(r2_mock):
    s = State(r2_mock)
    s.seen = {"hk_bad"}
    chain = FakeChain(block=100)
    # Payload missing the third colon-separated part.
    chain.commit_reveal("hk_bad", "kh:onlytwoparts", block=100)
    s.backfill_completed_from_chain(chain, netuid=3)
    # No KeyError, no garbage — just empty.
    assert s.completed_repos == set()


def test_skips_payload_with_empty_repo(r2_mock):
    s = State(r2_mock)
    s.seen = {"hk_blank"}
    chain = FakeChain(block=100)
    # Three parts but middle is empty.
    chain.commit_reveal("hk_blank", "kh::mh", block=100)
    s.backfill_completed_from_chain(chain, netuid=3)
    assert s.completed_repos == set()


# ---------------------------------------------------------------------
# Early-return paths.

def test_returns_early_when_seen_empty(r2_mock):
    s = State(r2_mock)
    s.seen = set()
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a", _payload(repo("alice", "A")), block=100)
    s.backfill_completed_from_chain(chain, netuid=3)
    # Early return — no reveal pulled in.
    assert s.completed_repos == set()


def test_returns_early_on_subtensor_exception(r2_mock):
    # Broad except — chain glitch must not crash startup.
    s = State(r2_mock)
    s.seen = {"hk_a"}
    sub = MagicMock()
    sub.get_all_revealed_commitments.side_effect = RuntimeError(
        "rpc disconnect")
    s.backfill_completed_from_chain(sub, netuid=3)
    assert s.completed_repos == set()  # no crash, no entries


def test_returns_early_when_no_reveals_at_all(r2_mock):
    s = State(r2_mock)
    s.seen = {"hk_a"}
    chain = FakeChain(block=100)  # empty chain, no commitments
    s.backfill_completed_from_chain(chain, netuid=3)
    assert s.completed_repos == set()


# ---------------------------------------------------------------------
# Side-effect: flush triggered when anything was added.

def test_flushes_state_when_anything_added(r2_mock):
    s = State(r2_mock)
    s.seen = {"hk_a"}
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a", _payload(repo("alice", "A")), block=100)
    s.backfill_completed_from_chain(chain, netuid=3)
    # flush() writes state/completed_repos.json among others.
    persisted = r2_mock._storage.get("state/completed_repos.json")
    assert persisted is not None
    assert repo("alice", "A") in set(persisted["repos"])


def test_does_not_flush_when_nothing_added(r2_mock):
    # If every seen hotkey is already covered, the flush guard
    # (`if added:`) skips the R2 write.
    s = State(r2_mock)
    s.seen = {"hk_a"}
    s.completed_repos = {repo("alice", "A")}
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a", _payload(repo("alice", "A")), block=100)
    s.backfill_completed_from_chain(chain, netuid=3)
    # state/completed_repos.json was NOT written by this call.
    # (We didn't pre-populate the spy storage so it should still be empty.)
    assert "state/completed_repos.json" not in r2_mock._storage
