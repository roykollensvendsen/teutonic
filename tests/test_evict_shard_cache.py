"""Tests for eval.torch_runner._evict_shard_cache.

Bounds the on-disk shard cache to `SHARD_CACHE_MAX` files. Sorts by
mtime (oldest first), unlinks until the count fits. Silently no-ops
when the cache directory does not yet exist (cold start).
"""
import os

from eval import torch_runner


def _make_npy(path, mtime):
    path.write_bytes(b"\x93NUMPY\x01\x00")
    os.utime(path, (mtime, mtime))


def test_evict_is_noop_when_cache_dir_missing(tmp_path, monkeypatch):
    missing = tmp_path / "does_not_exist"
    monkeypatch.setattr(torch_runner, "SHARD_CACHE_DIR", str(missing))
    # Should not raise — cold-start branch.
    torch_runner._evict_shard_cache()
    assert not missing.exists()


def test_evict_is_noop_when_under_max(tmp_path, monkeypatch):
    monkeypatch.setattr(torch_runner, "SHARD_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(torch_runner, "SHARD_CACHE_MAX", 5)
    for i in range(3):
        _make_npy(tmp_path / f"shard_{i}.npy", mtime=1000 + i)

    torch_runner._evict_shard_cache()

    assert sorted(p.name for p in tmp_path.glob("*.npy")) == [
        "shard_0.npy", "shard_1.npy", "shard_2.npy",
    ]


def test_evict_removes_oldest_first(tmp_path, monkeypatch):
    monkeypatch.setattr(torch_runner, "SHARD_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(torch_runner, "SHARD_CACHE_MAX", 3)
    # Five shards with strictly increasing mtimes; oldest two should go.
    for i in range(5):
        _make_npy(tmp_path / f"shard_{i}.npy", mtime=1000 + i)

    torch_runner._evict_shard_cache()

    surviving = sorted(p.name for p in tmp_path.glob("*.npy"))
    assert surviving == ["shard_2.npy", "shard_3.npy", "shard_4.npy"]


