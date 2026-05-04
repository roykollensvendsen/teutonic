"""Tests for validator.get_king_config.

Fetches and process-caches the king's config.json from HuggingFace.
The cache is keyed on `f"{repo}@{revision}"` — once populated, the
same (repo, revision) returns the cached dict without re-hitting HF.
On any HF error, the function caches an empty dict so subsequent
calls don't keep re-trying a known-broken repo within the same key.
"""
import json

import pytest

import validator
from validator import get_king_config


@pytest.fixture(autouse=True)
def _reset_king_config_cache():
    validator._king_config = None
    validator._king_config_key = None
    yield
    validator._king_config = None
    validator._king_config_key = None


@pytest.fixture
def fake_hf(mocker, tmp_path):
    """Mock validator.HfApi for a single (repo, revision) → config dict."""
    repo_state = {}

    def hf_hub_download(repo_id, filename, revision=None, **_kwargs):
        key = (repo_id, revision or None)
        if key not in repo_state:
            raise FileNotFoundError(f"No fake state for {key}")
        path = tmp_path / f"{repo_id.replace('/', '_')}_{revision}_config.json"
        path.write_text(json.dumps(repo_state[key]))
        return str(path)

    api = mocker.MagicMock()
    api.hf_hub_download.side_effect = hf_hub_download
    mocker.patch("validator.HfApi", return_value=api)

    def setup(repo, revision, *, config):
        repo_state[(repo, revision or None)] = config

    return setup


def test_returns_config_on_happy_path(fake_hf):
    fake_hf("unconst/king", "rev1", config={"d_model": 4096})
    cfg = get_king_config("unconst/king", "rev1")
    assert cfg == {"d_model": 4096}


def test_caches_by_repo_and_revision(fake_hf, mocker):
    fake_hf("unconst/king", "rev1", config={"d_model": 4096})
    api_factory = mocker.patch("validator.HfApi", wraps=validator.HfApi)

    get_king_config("unconst/king", "rev1")
    get_king_config("unconst/king", "rev1")

    # Second call hits the cache — HfApi is not constructed again.
    assert api_factory.call_count <= 1


def test_returns_empty_dict_on_fetch_error(fake_hf):
    # No fake state registered — hf_hub_download raises FileNotFoundError.
    cfg = get_king_config("unconst/missing", "rev1")
    assert cfg == {}


def test_cache_invalidates_on_revision_change(fake_hf):
    # Different revision → different cache key → new HF fetch.
    fake_hf("unconst/king", "rev1", config={"d_model": 4096})
    fake_hf("unconst/king", "rev2", config={"d_model": 2048})

    first = get_king_config("unconst/king", "rev1")
    second = get_king_config("unconst/king", "rev2")

    assert first == {"d_model": 4096}
    assert second == {"d_model": 2048}
