"""Tests for the completed_repos lifecycle in validator.

Upstream commit 26bf392 (\"validator: kill re-eval exploit\") introduced
`State.completed_repos: set[str]` as the canonical \"every hf_repo gets
exactly one eval, ever\" deduplicator. Three surfaces touch it:

1. **`scan_reveals(subtensor, netuid, completed_repos)`** — the chain-
   read function filters out any reveal whose `hf_repo` is already in
   the set. This is what stops a miner who lost against king K1 from
   getting re-evaluated under a weaker king K2 (the re-eval exploit).
   `validator.py:568-607`.

2. **`State.enqueue(reveal)`** — on a successful enqueue (no king-hotkey
   skip, no dup-queue, no evaluated_repos hit), adds the repo to
   `completed_repos` and the hotkey to `seen`. `validator.py:1196-1198`.

3. **`State.flush()` / `State.load()`** — persist as
   `state/completed_repos.json` on R2; on reload, falls back to a
   one-shot backfill from `history` + `queue` if the file is missing
   (covers the first restart after the upstream deploy).
   `validator.py:951-953, 986-997, 1155-1157`.

Together these stop a re-eval-against-weaker-king exploit. If any of
them silently regress, the protection breaks open. These tests pin
the contracts on every gate.
"""
from unittest.mock import MagicMock

from tests._chain import repo
from tests._harness.chain import FakeChain
from validator import State, scan_reveals


def _payload(hf_repo: str, *, king_hash: str = "kh", model_hash: str = "mh") -> str:
    """scan_reveals expects 'king_hash:hf_repo:model_hash'."""
    return f"{king_hash}:{hf_repo}:{model_hash}"


# ---------------------------------------------------------------------
# Section 1: scan_reveals filter against completed_repos.

def test_scan_reveals_skips_reveals_whose_repo_is_completed():
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a", _payload(repo("alice", "A")), block=100)
    chain.commit_reveal("hk_b", _payload(repo("bob", "B")), block=100)
    completed = {repo("alice", "A")}
    result = scan_reveals(chain, netuid=3, completed_repos=completed)
    repos = {r["hf_repo"] for r in result}
    assert repo("bob", "B") in repos
    assert repo("alice", "A") not in repos


def test_scan_reveals_lets_through_when_completed_is_empty():
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a", _payload(repo("alice", "A")), block=100)
    chain.commit_reveal("hk_b", _payload(repo("bob", "B")), block=100)
    result = scan_reveals(chain, netuid=3, completed_repos=set())
    assert len(result) == 2


def test_scan_reveals_filters_invalid_repos_via_pattern():
    # Repo name not matching REPO_PATTERN is dropped before the
    # completed_repos check.
    chain = FakeChain(block=100)
    chain.commit_reveal("hk_a", _payload("not-a-valid-repo-name"), block=100)
    chain.commit_reveal("hk_b", _payload(repo("bob", "B")), block=100)
    result = scan_reveals(chain, netuid=3, completed_repos=set())
    repos = {r["hf_repo"] for r in result}
    assert "not-a-valid-repo-name" not in repos
    assert repo("bob", "B") in repos


def test_scan_reveals_returns_empty_list_when_chain_empty():
    chain = FakeChain(block=100)
    result = scan_reveals(chain, netuid=3, completed_repos=set())
    assert result == []


def test_scan_reveals_returns_empty_list_when_subtensor_raises():
    # Broad `except Exception` swallows RPC failures and returns []
    # so the eval loop doesn't crash on transient chain glitches.
    sub = MagicMock()
    sub.get_all_revealed_commitments.side_effect = RuntimeError(
        "rpc disconnect")
    result = scan_reveals(sub, netuid=3, completed_repos=set())
    assert result == []


def test_scan_reveals_picks_latest_per_hotkey():
    # When a hotkey has multiple commitments, the highest-block one wins.
    chain = FakeChain(block=200)
    chain.commit_reveal("hk_a", _payload(repo("alice", "old")), block=100)
    chain.commit_reveal("hk_a", _payload(repo("alice", "new")), block=200)
    result = scan_reveals(chain, netuid=3, completed_repos=set())
    assert len(result) == 1
    assert result[0]["hf_repo"] == repo("alice", "new")
    assert result[0]["block"] == 200


# ---------------------------------------------------------------------
# Section 2: State.enqueue side-effect on completed_repos + seen.

def test_enqueue_adds_repo_to_completed_repos(r2_mock):
    s = State(r2_mock)
    reveal = {"hotkey": "hk_chal", "hf_repo": repo("alice", "A"),
              "king_hash": "kh", "block": 100}
    cid = s.enqueue(reveal, defer_flush=True)
    assert cid is not None
    assert repo("alice", "A") in s.completed_repos


def test_enqueue_adds_hotkey_to_seen(r2_mock):
    s = State(r2_mock)
    reveal = {"hotkey": "hk_chal", "hf_repo": repo("alice", "A"),
              "king_hash": "kh", "block": 100}
    s.enqueue(reveal, defer_flush=True)
    assert "hk_chal" in s.seen


def test_enqueue_does_not_add_when_king_hotkey(r2_mock):
    # Gate 1 in enqueue: skip when reveal hotkey is current king.
    s = State(r2_mock)
    s.set_king("hk_king", repo("alice", "king"), "kh", 100)
    reveal = {"hotkey": "hk_king", "hf_repo": repo("alice", "king"),
              "king_hash": "kh", "block": 100}
    cid = s.enqueue(reveal, defer_flush=True)
    assert cid is None
    assert repo("alice", "king") not in s.completed_repos


def test_enqueue_does_not_add_when_repo_already_queued(r2_mock):
    # Gate 2 in enqueue: same hf_repo already in self.queue.
    s = State(r2_mock)
    first = {"hotkey": "hk_a", "hf_repo": repo("alice", "A"),
             "king_hash": "kh", "block": 100}
    s.enqueue(first, defer_flush=True)
    completed_before = set(s.completed_repos)
    second = {"hotkey": "hk_b", "hf_repo": repo("alice", "A"),  # dup repo
              "king_hash": "kh", "block": 101}
    cid = s.enqueue(second, defer_flush=True)
    assert cid is None
    # Set unchanged (already had this repo from first enqueue).
    assert s.completed_repos == completed_before


# ---------------------------------------------------------------------
# Section 3: State.flush persistence.

def test_flush_writes_completed_repos_to_r2(r2_mock):
    s = State(r2_mock)
    s.completed_repos = {repo("alice", "A"), repo("bob", "B")}
    s.flush()
    persisted = r2_mock._storage.get("state/completed_repos.json")
    assert persisted is not None
    assert set(persisted["repos"]) == s.completed_repos
    assert "updated_at" in persisted


def test_flush_writes_repos_in_sorted_order(r2_mock):
    # Sort makes diffs against `r2 get state/completed_repos.json`
    # stable across runs (avoids spurious churn in audits).
    s = State(r2_mock)
    s.completed_repos = {repo("zora", "z"), repo("alice", "a"),
                         repo("mike", "m")}
    s.flush()
    persisted = r2_mock._storage["state/completed_repos.json"]
    assert persisted["repos"] == sorted(s.completed_repos)


# ---------------------------------------------------------------------
# Section 4: State.load — read persisted, fall back to backfill.

def test_load_reads_persisted_completed_repos(r2_mock):
    r2_mock._storage["state/completed_repos.json"] = {
        "repos": [repo("alice", "A"), repo("bob", "B")],
        "updated_at": "2026-05-08T00:00:00Z",
    }
    s = State(r2_mock)
    s.load()
    assert s.completed_repos == {repo("alice", "A"), repo("bob", "B")}


def test_load_backfills_from_history_when_no_persisted(r2_mock):
    # First restart after upstream's 26bf392 deploy: no
    # state/completed_repos.json yet, but we have history with
    # challenger_repo entries. Backfill from there.
    r2_mock._storage["state/dashboard_history.json"] = {
        "history": [
            {"challenger_repo": repo("alice", "A"), "verdict": "king"},
            {"challenger_repo": repo("bob", "B"), "verdict": "challenger"},
        ],
    }
    s = State(r2_mock)
    s.load()
    assert s.completed_repos >= {repo("alice", "A"), repo("bob", "B")}


def test_load_backfills_from_queue_when_no_history(r2_mock):
    r2_mock._storage["state/queue.json"] = {
        "pending": [{"hf_repo": repo("queued", "x"), "hotkey": "hk_q"}],
    }
    s = State(r2_mock)
    s.load()
    assert repo("queued", "x") in s.completed_repos


def test_load_prefers_persisted_over_backfill(r2_mock):
    # If state/completed_repos.json exists, the history+queue backfill
    # is skipped (the `if not self.completed_repos:` gate).
    r2_mock._storage["state/completed_repos.json"] = {
        "repos": [repo("persisted", "p")],
        "updated_at": "2026-05-08T00:00:00Z",
    }
    r2_mock._storage["state/dashboard_history.json"] = {
        "history": [{"challenger_repo": repo("history", "h")}],
    }
    s = State(r2_mock)
    s.load()
    assert s.completed_repos == {repo("persisted", "p")}
    assert repo("history", "h") not in s.completed_repos


def test_load_leaves_completed_repos_empty_when_nothing_anywhere(r2_mock):
    s = State(r2_mock)
    s.load()
    assert s.completed_repos == set()
