"""Tests for eval_server._cleanup_hf_cache.

Cache-cleanup runs after each eval to keep the HF cache under
`CACHE_HIGH_WATERMARK_GB`. Three classes of revisions are protected
from deletion:

1. The current king's revision (matched by `_king_repo` + `_king_revision`)
2. Repos in `_preload_seen` whose timestamp is within the last
   `PRELOAD_KEEP_S` seconds (speculative downloads for the next eval)
3. Anything not in the cache snapshot

Cleanup deletes the oldest non-protected revisions until total cache
size drops below 70% of the watermark (30% headroom buffer).

These tests stub `huggingface_hub.scan_cache_dir` with a synthetic
cache snapshot and verify the deletion-decision logic in isolation.
"""
from types import SimpleNamespace

import pytest

import eval_server

# ---------------------------------------------------------------------
# Reset module-level cleanup-related globals between tests.

@pytest.fixture(autouse=True)
def _reset_cleanup_globals(monkeypatch):
    monkeypatch.setattr(eval_server, "_king_repo", None)
    monkeypatch.setattr(eval_server, "_king_revision", None)
    monkeypatch.setattr(eval_server, "_preload_seen", {})
    yield


def _rev(commit_hash: str, *, size_gb: float, last_modified: float):
    """Build a CachedRevisionInfo-shaped stub.

    Real `huggingface_hub.CachedRevisionInfo` is a frozen dataclass with
    these fields plus a `commit_hash`, `last_modified` (unix ts), and
    `size_on_disk` (bytes).
    """
    return SimpleNamespace(
        commit_hash=commit_hash,
        last_modified=last_modified,
        size_on_disk=int(size_gb * 1e9),
    )


def _repo(repo_id: str, revisions: list):
    return SimpleNamespace(repo_id=repo_id, revisions=revisions)


def _cache_info(repos, *, total_size_gb: float | None = None):
    """Build a HFCacheInfo-shaped stub.

    `delete_revisions(*hashes)` returns a strategy with
    `expected_freed_size` (bytes) + `execute()`.
    """
    if total_size_gb is None:
        total_size_gb = sum(
            rev.size_on_disk / 1e9
            for repo in repos
            for rev in repo.revisions
        )
    deleted: list = []

    def delete_revisions(*hashes):
        return SimpleNamespace(
            expected_freed_size=sum(
                rev.size_on_disk
                for repo in repos
                for rev in repo.revisions
                if rev.commit_hash in hashes
            ),
            execute=lambda: deleted.extend(hashes),
        )

    return SimpleNamespace(
        size_on_disk=int(total_size_gb * 1e9),
        repos=repos,
        delete_revisions=delete_revisions,
        _deleted=deleted,
    )


@pytest.fixture
def fake_scan_cache_dir(mocker):
    """Patch huggingface_hub.scan_cache_dir to return a test-controlled cache.

    Returns a setter the test calls with the cache_info object."""
    holder = {"cache_info": None}

    def _scan():
        return holder["cache_info"]

    mocker.patch("huggingface_hub.scan_cache_dir", side_effect=_scan)

    def setter(cache_info):
        holder["cache_info"] = cache_info

    return setter


# ---------------------------------------------------------------------
# Below-watermark: no-op fast path.

def test_cleanup_below_watermark_does_nothing(fake_scan_cache_dir, monkeypatch):
    monkeypatch.setattr(eval_server, "CACHE_HIGH_WATERMARK_GB", 200.0)
    cache = _cache_info(
        [_repo("alice/king", [_rev("hash_a", size_gb=10, last_modified=100)])],
    )
    fake_scan_cache_dir(cache)
    eval_server._cleanup_hf_cache()
    assert cache._deleted == []


# ---------------------------------------------------------------------
# Above-watermark: deletes oldest until below 70% of watermark.

def test_cleanup_deletes_oldest_until_under_70pct_of_watermark(
    fake_scan_cache_dir, monkeypatch,
):
    # Watermark 200 GB, 70% target = 140 GB. 4 × 60 GB revisions = 240 GB.
    # Need to drop below 140 GB → must delete at least 2 revisions
    # (240 - 60 = 180 still above; 240 - 120 = 120 below).
    monkeypatch.setattr(eval_server, "CACHE_HIGH_WATERMARK_GB", 200.0)
    cache = _cache_info([_repo("alice/m", [
        _rev("h1", size_gb=60, last_modified=100),  # oldest
        _rev("h2", size_gb=60, last_modified=200),
        _rev("h3", size_gb=60, last_modified=300),
        _rev("h4", size_gb=60, last_modified=400),  # newest
    ])])
    fake_scan_cache_dir(cache)
    eval_server._cleanup_hf_cache()
    # Oldest two get deleted; newest two are kept.
    assert "h1" in cache._deleted
    assert "h2" in cache._deleted
    assert "h3" not in cache._deleted
    assert "h4" not in cache._deleted


# ---------------------------------------------------------------------
# King protection: king's revision survives even if oldest.

def test_cleanup_protects_current_king_revision(fake_scan_cache_dir, monkeypatch):
    monkeypatch.setattr(eval_server, "CACHE_HIGH_WATERMARK_GB", 100.0)
    monkeypatch.setattr(eval_server, "_king_repo", "alice/king")
    monkeypatch.setattr(eval_server, "_king_revision", "king_hash_abc")

    # Total 150 GB, watermark 100, 70% = 70 → must drop ≥ 80 GB.
    cache = _cache_info([
        _repo("alice/king", [
            _rev("king_hash_abc", size_gb=80,
                 last_modified=10),  # OLDEST but king-protected
        ]),
        _repo("bob/old", [
            _rev("h_bob", size_gb=70, last_modified=100),
        ]),
    ])
    fake_scan_cache_dir(cache)
    eval_server._cleanup_hf_cache()
    assert "king_hash_abc" not in cache._deleted
    assert "h_bob" in cache._deleted


def test_cleanup_protects_king_when_revision_unset(fake_scan_cache_dir, monkeypatch):
    # `_king_revision` empty → still protect by repo match (the
    # `not keep_rev` branch in the function).
    monkeypatch.setattr(eval_server, "CACHE_HIGH_WATERMARK_GB", 100.0)
    monkeypatch.setattr(eval_server, "_king_repo", "alice/king")
    monkeypatch.setattr(eval_server, "_king_revision", "")

    cache = _cache_info([
        _repo("alice/king", [
            _rev("any_hash", size_gb=80, last_modified=10),
        ]),
        _repo("bob/old", [
            _rev("h_bob", size_gb=70, last_modified=100),
        ]),
    ])
    fake_scan_cache_dir(cache)
    eval_server._cleanup_hf_cache()
    assert "any_hash" not in cache._deleted


# ---------------------------------------------------------------------
# Preload protection: recent preloads survive.

def test_cleanup_protects_recently_preloaded_repos(fake_scan_cache_dir, monkeypatch):
    import time

    monkeypatch.setattr(eval_server, "CACHE_HIGH_WATERMARK_GB", 100.0)
    monkeypatch.setattr(eval_server, "PRELOAD_KEEP_S", 1800.0)
    now = time.time()
    monkeypatch.setattr(eval_server, "_preload_seen", {
        ("bob/preload", ""): now - 100,  # within keep window
        ("zip/old", ""): now - 9999,  # outside keep window — NOT protected
    })

    cache = _cache_info([
        _repo("bob/preload", [_rev("h_pre", size_gb=80, last_modified=10)]),
        _repo("zip/old", [_rev("h_zip", size_gb=70, last_modified=20)]),
    ])
    fake_scan_cache_dir(cache)
    eval_server._cleanup_hf_cache()
    assert "h_pre" not in cache._deleted   # protected
    assert "h_zip" in cache._deleted        # outside keep window


# ---------------------------------------------------------------------
# Edge cases.

def test_cleanup_logs_when_nothing_eligible_to_delete(
    fake_scan_cache_dir, monkeypatch,
):
    # Above watermark but every revision is king-protected → returns
    # without deleting (the function logs "above watermark but nothing
    # eligible").
    monkeypatch.setattr(eval_server, "CACHE_HIGH_WATERMARK_GB", 50.0)
    monkeypatch.setattr(eval_server, "_king_repo", "alice/king")
    monkeypatch.setattr(eval_server, "_king_revision", "")

    cache = _cache_info([
        _repo("alice/king", [
            _rev("h1", size_gb=80, last_modified=100),
        ]),
    ])
    fake_scan_cache_dir(cache)
    eval_server._cleanup_hf_cache()
    assert cache._deleted == []


def test_cleanup_swallows_scan_cache_dir_exception(mocker, monkeypatch):
    # If huggingface_hub.scan_cache_dir raises (rare but possible
    # during HF cache lock), cleanup must NOT raise.
    monkeypatch.setattr(eval_server, "CACHE_HIGH_WATERMARK_GB", 100.0)
    mocker.patch("huggingface_hub.scan_cache_dir",
                 side_effect=OSError("hf cache locked"))
    eval_server._cleanup_hf_cache()  # must not raise


def test_cleanup_swallows_delete_strategy_exception(fake_scan_cache_dir, monkeypatch):
    # If delete_revisions or execute() raises, the function still
    # catches it (outer try/except).
    monkeypatch.setattr(eval_server, "CACHE_HIGH_WATERMARK_GB", 50.0)
    cache = SimpleNamespace(
        size_on_disk=int(100e9),
        repos=[_repo("alice/m", [_rev("h1", size_gb=100, last_modified=10)])],
        delete_revisions=lambda *_h: (_ for _ in ()).throw(OSError("rm failed")),
    )
    fake_scan_cache_dir(cache)
    eval_server._cleanup_hf_cache()  # must not raise


def test_cleanup_stops_deleting_once_under_70pct_threshold(
    fake_scan_cache_dir, monkeypatch,
):
    # Watermark 100, 70% target = 70. 4 × 30 GB = 120 → drop one (oldest)
    # gets us to 90 (still > 70). Drop two → 60 (under 70). Stop after 2.
    monkeypatch.setattr(eval_server, "CACHE_HIGH_WATERMARK_GB", 100.0)
    cache = _cache_info([_repo("alice/m", [
        _rev("h1", size_gb=30, last_modified=100),
        _rev("h2", size_gb=30, last_modified=200),
        _rev("h3", size_gb=30, last_modified=300),
        _rev("h4", size_gb=30, last_modified=400),
    ])])
    fake_scan_cache_dir(cache)
    eval_server._cleanup_hf_cache()
    # Oldest two deleted, newest two kept.
    assert set(cache._deleted) == {"h1", "h2"}


# ---------------------------------------------------------------------
# `_evict_for_challenger(target_repo)` — disk-pressure backstop.
#
# Contract (eval_server.py:331): force-evict every cached repo that's
# NOT the king and NOT the challenger we're about to load. Called
# BEFORE downloading a new challenger to guarantee headroom on disks
# where `_cleanup_hf_cache`'s watermark check leaves a stale just-
# evaluated challenger pinned (the "most recent preload" gets self-
# protected).
#
# Wraps everything in `try/except Exception` and logs warning — must
# never raise into the caller (the dispatch loop). See SPEC_DEBT.md
# for the docstring gap on this last point.

def test_evict_force_evicts_non_king_non_target(
    fake_scan_cache_dir, monkeypatch,
):
    monkeypatch.setattr(eval_server, "_king_repo", "alice/king")
    cache = _cache_info([
        _repo("alice/king", [_rev("h_king", size_gb=80, last_modified=100)]),
        _repo("bob/target", [_rev("h_target", size_gb=80, last_modified=200)]),
        _repo("carol/old", [_rev("h_carol", size_gb=80, last_modified=50)]),
        _repo("dave/older", [_rev("h_dave", size_gb=80, last_modified=10)]),
    ])
    fake_scan_cache_dir(cache)
    eval_server._evict_for_challenger("bob/target")
    # King + target preserved; everyone else evicted.
    assert set(cache._deleted) == {"h_carol", "h_dave"}


def test_evict_preserves_target_when_only_king_else_in_cache(
    fake_scan_cache_dir, monkeypatch,
):
    monkeypatch.setattr(eval_server, "_king_repo", "alice/king")
    cache = _cache_info([
        _repo("alice/king", [_rev("h_king", size_gb=80, last_modified=100)]),
        _repo("bob/target", [_rev("h_target", size_gb=80, last_modified=200)]),
    ])
    fake_scan_cache_dir(cache)
    eval_server._evict_for_challenger("bob/target")
    # No third party — nothing to evict.
    assert cache._deleted == []


def test_evict_no_op_when_cache_empty(fake_scan_cache_dir, monkeypatch):
    monkeypatch.setattr(eval_server, "_king_repo", "alice/king")
    cache = _cache_info([])
    fake_scan_cache_dir(cache)
    eval_server._evict_for_challenger("bob/target")
    assert cache._deleted == []


def test_evict_handles_king_repo_none(fake_scan_cache_dir, monkeypatch):
    # Module starts with _king_repo = None (no king crowned yet).
    # kept_repos must discard None — tested via leaving the autouse
    # fixture's default in place.
    cache = _cache_info([
        _repo("bob/target", [_rev("h_target", size_gb=80, last_modified=200)]),
        _repo("carol/old", [_rev("h_carol", size_gb=80, last_modified=50)]),
    ])
    fake_scan_cache_dir(cache)
    # _king_repo is None — only target is preserved.
    eval_server._evict_for_challenger("bob/target")
    assert set(cache._deleted) == {"h_carol"}


def test_evict_handles_king_repo_empty_string(
    fake_scan_cache_dir, monkeypatch,
):
    # Same logic as None — kept_repos.discard("") catches it.
    monkeypatch.setattr(eval_server, "_king_repo", "")
    cache = _cache_info([
        _repo("bob/target", [_rev("h_target", size_gb=80, last_modified=200)]),
        _repo("carol/old", [_rev("h_carol", size_gb=80, last_modified=50)]),
    ])
    fake_scan_cache_dir(cache)
    eval_server._evict_for_challenger("bob/target")
    assert set(cache._deleted) == {"h_carol"}


def test_evict_evicts_multiple_revisions_of_same_repo(
    fake_scan_cache_dir, monkeypatch,
):
    # A non-protected repo with multiple cached revisions: ALL of its
    # revisions are evicted (not just the oldest — this is the
    # backstop, not the watermark-targeted cleanup).
    monkeypatch.setattr(eval_server, "_king_repo", "alice/king")
    cache = _cache_info([
        _repo("alice/king", [_rev("h_king", size_gb=80, last_modified=100)]),
        _repo("carol/three-revs", [
            _rev("h_c1", size_gb=80, last_modified=10),
            _rev("h_c2", size_gb=80, last_modified=20),
            _rev("h_c3", size_gb=80, last_modified=30),
        ]),
    ])
    fake_scan_cache_dir(cache)
    eval_server._evict_for_challenger("bob/target")
    assert set(cache._deleted) == {"h_c1", "h_c2", "h_c3"}


def test_evict_swallows_scan_cache_exception_non_fatal(
    monkeypatch, mocker,
):
    # If scan_cache_dir raises (e.g. transient HF lib glitch), evict
    # must NOT propagate — caller (_load_challenger) is on the dispatch
    # path and a raise here would stall every eval.
    mocker.patch("huggingface_hub.scan_cache_dir",
                  side_effect=RuntimeError("scan failed"))
    monkeypatch.setattr(eval_server, "_king_repo", "alice/king")
    # No assertion on side-effect — just that no exception escapes.
    eval_server._evict_for_challenger("bob/target")


def test_evict_swallows_delete_revisions_exception(
    fake_scan_cache_dir, monkeypatch,
):
    # If the eviction itself raises, swallow it.
    monkeypatch.setattr(eval_server, "_king_repo", "alice/king")
    from types import SimpleNamespace

    def boom(*hashes):
        raise OSError("ENOSPC during delete")

    cache = SimpleNamespace(
        repos=[_repo("carol/old",
                       [_rev("h_carol", size_gb=80, last_modified=50)])],
        delete_revisions=boom,
    )
    fake_scan_cache_dir(cache)
    eval_server._evict_for_challenger("bob/target")
