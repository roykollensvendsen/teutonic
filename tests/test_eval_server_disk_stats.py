"""Tests for eval_server._get_disk_stats.

Returns a dict of disk + HF-cache stats for the /health endpoint.
The contract is intentionally loose — both subqueries are wrapped
in try/except, so any failure inside either branch silently omits
that section rather than 500-ing the health endpoint. This is the
right shape for a health-probe helper.
"""
import eval_server


def test_returns_dict_with_disk_keys_when_shutil_succeeds():
    # `shutil.disk_usage("/")` works on every supported platform; the
    # health endpoint relies on that fact. Pin the three disk-* keys
    # are present and numeric — without that, the dashboard chart
    # would show blank cells when /health is the source of truth.
    stats = eval_server._get_disk_stats()
    assert "disk_total_gb" in stats
    assert "disk_used_gb" in stats
    assert "disk_free_gb" in stats
    assert isinstance(stats["disk_total_gb"], (int, float))
    assert stats["disk_total_gb"] > 0


def test_returns_dict_even_when_inner_calls_fail(mocker):
    # Both the shutil and the HF-cache probes are inside their own
    # try/except; if either raises, the function still returns a
    # dict (possibly empty), not None and not raising.
    mocker.patch("shutil.disk_usage", side_effect=OSError("boom"))
    mocker.patch("huggingface_hub.scan_cache_dir",
                 side_effect=RuntimeError("nope"))

    stats = eval_server._get_disk_stats()

    assert isinstance(stats, dict)
    # Disk keys absent because shutil raised; HF keys absent because
    # scan_cache_dir raised. Empty dict is the documented graceful
    # degradation, not an error.
    assert "disk_total_gb" not in stats
    assert "hf_cache_size_gb" not in stats
