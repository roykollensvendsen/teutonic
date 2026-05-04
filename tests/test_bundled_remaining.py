"""Tests for the last few surfaces that fit existing mock patterns.

Bundled because each is small and uses only patterns already established
in other test files (HfApi mock from PR 3, tmp_path/monkeypatch stdlib).

Surfaces:
- `eval_torch.parse_gpu_ids(gpu_str)` — pure GPU-id parser
- `eval_torch._evict_shard_cache()` — filesystem housekeeping
- `validator.get_king_config(king_repo, king_revision)` — HF-cached config fetch
"""
import json
import os

import pytest

import eval_torch
import validator
from eval_torch import parse_gpu_ids

# =====================================================================
# parse_gpu_ids — "0,1,2" → [0,1,2]; "auto" → range(device_count).


def test_parse_gpu_ids_single_id():
    assert parse_gpu_ids("0") == [0]


def test_parse_gpu_ids_comma_separated():
    assert parse_gpu_ids("0,1,2") == [0, 1, 2]


def test_parse_gpu_ids_strips_whitespace():
    assert parse_gpu_ids("0, 1 , 2") == [0, 1, 2]


def test_parse_gpu_ids_auto_uses_torch_device_count(mocker):
    # "auto" → list(range(torch.cuda.device_count())).
    mocker.patch("torch.cuda.device_count", return_value=4)
    assert parse_gpu_ids("auto") == [0, 1, 2, 3]


def test_parse_gpu_ids_auto_with_zero_devices(mocker):
    mocker.patch("torch.cuda.device_count", return_value=0)
    assert parse_gpu_ids("auto") == []


# =====================================================================
# _evict_shard_cache — keep newest SHARD_CACHE_MAX *.npy files in
# SHARD_CACHE_DIR; delete the rest. No-op if dir doesn't exist.


@pytest.fixture
def temp_cache_dir(tmp_path, monkeypatch):
    """Point SHARD_CACHE_DIR at an empty tmp_path for the test."""
    monkeypatch.setattr(eval_torch, "SHARD_CACHE_DIR", str(tmp_path))
    return tmp_path


def _make_shard(dir_path, name, mtime):
    """Create a .npy file with the given mtime."""
    p = dir_path / name
    p.write_bytes(b"")
    os.utime(p, (mtime, mtime))
    return p


def test_evict_shard_cache_noop_when_dir_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(eval_torch, "SHARD_CACHE_DIR", str(tmp_path / "nonexistent"))
    # Should not raise.
    eval_torch._evict_shard_cache()


def test_evict_shard_cache_keeps_files_under_cap(temp_cache_dir, monkeypatch):
    monkeypatch.setattr(eval_torch, "SHARD_CACHE_MAX", 5)
    _make_shard(temp_cache_dir, "a.npy", mtime=100)
    _make_shard(temp_cache_dir, "b.npy", mtime=200)
    eval_torch._evict_shard_cache()
    # Both files should still exist — under the cap.
    assert (temp_cache_dir / "a.npy").exists()
    assert (temp_cache_dir / "b.npy").exists()


def test_evict_shard_cache_deletes_oldest_first(temp_cache_dir, monkeypatch):
    monkeypatch.setattr(eval_torch, "SHARD_CACHE_MAX", 2)
    _make_shard(temp_cache_dir, "old.npy", mtime=100)
    _make_shard(temp_cache_dir, "mid.npy", mtime=200)
    _make_shard(temp_cache_dir, "new.npy", mtime=300)
    eval_torch._evict_shard_cache()
    # The oldest should be gone; the two newer ones survive.
    assert not (temp_cache_dir / "old.npy").exists()
    assert (temp_cache_dir / "mid.npy").exists()
    assert (temp_cache_dir / "new.npy").exists()


def test_evict_shard_cache_ignores_non_npy(temp_cache_dir, monkeypatch):
    monkeypatch.setattr(eval_torch, "SHARD_CACHE_MAX", 1)
    _make_shard(temp_cache_dir, "stale.npy", mtime=100)
    _make_shard(temp_cache_dir, "fresh.npy", mtime=200)
    # A non-.npy file should be left untouched regardless of cap.
    other = temp_cache_dir / "not-a-shard.txt"
    other.write_bytes(b"")
    eval_torch._evict_shard_cache()
    assert other.exists()
    # And the eviction still happened on the .npy side.
    assert not (temp_cache_dir / "stale.npy").exists()
    assert (temp_cache_dir / "fresh.npy").exists()


# =====================================================================
# get_king_config — fetch & cache config.json. Cache key = (repo, rev).
# Returns falsy on HF failure (per call-site at validator.py:480).


@pytest.fixture
def reset_king_config_cache():
    """Reset the module-level king-config cache between tests."""
    validator._king_config = None
    validator._king_config_key = None
    yield
    validator._king_config = None
    validator._king_config_key = None


@pytest.fixture
def fake_hf(mocker, tmp_path):
    """Mock validator.HfApi.hf_hub_download to return a tmp config.json path."""
    api_instance = mocker.MagicMock()

    def make_config_file(payload):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(payload))
        return str(cfg_path)

    api_instance._make_config_file = make_config_file
    mocker.patch("validator.HfApi", return_value=api_instance)
    return api_instance


def test_get_king_config_returns_parsed_config(fake_hf, reset_king_config_cache):
    cfg_path = fake_hf._make_config_file({"hidden_size": 768, "num_layers": 12})
    fake_hf.hf_hub_download.return_value = cfg_path
    result = validator.get_king_config("user/king", "rev1")
    assert result == {"hidden_size": 768, "num_layers": 12}


def test_get_king_config_caches_by_repo_revision(fake_hf, reset_king_config_cache):
    cfg_path = fake_hf._make_config_file({"hidden_size": 768})
    fake_hf.hf_hub_download.return_value = cfg_path
    validator.get_king_config("user/king", "rev1")
    validator.get_king_config("user/king", "rev1")
    # Cache hit on the second call → only one HF download.
    assert fake_hf.hf_hub_download.call_count == 1


def test_get_king_config_refetches_on_revision_change(fake_hf, reset_king_config_cache):
    cfg_path = fake_hf._make_config_file({"hidden_size": 768})
    fake_hf.hf_hub_download.return_value = cfg_path
    validator.get_king_config("user/king", "rev1")
    validator.get_king_config("user/king", "rev2")
    # Different cache key → two downloads.
    assert fake_hf.hf_hub_download.call_count == 2


def test_get_king_config_returns_falsy_on_hf_failure(fake_hf, reset_king_config_cache):
    fake_hf.hf_hub_download.side_effect = RuntimeError("HF 404")
    result = validator.get_king_config("user/dead-repo", "rev1")
    # Per call-site (validator.py:480): caller does `if not king_cfg`.
    assert not result
