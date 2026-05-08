"""Tests for the eval_server `/hash` endpoint.

Upstream commit cf1f2ea (\"LXXX: fast-path coronation hash via eval-server
/hash endpoint\") added a HTTP endpoint that returns sha256 over the
cached safetensors of a given repo+revision. The validator uses it as
fast-path before falling back to the slow `_seed_king_hash` subprocess
that re-downloads 165 GiB of weights — saves ~5-15 min per coronation.

Endpoint contract (eval_server.py:805):

    GET /hash?repo=<str>&revision=<str>

Behaviour:
* Calls `snapshot_download(repo, revision=..., local_files_only=True,
  allow_patterns=[\"*.safetensors\"])` to find the cached snapshot dir.
* On any download failure (cache miss, network, transformers version
  mismatch, ...) → 404 with a one-line detail.
* If the snapshot exists but contains no `.safetensors` → 404.
* Otherwise: streams sha256 over each file in alphabetical filename
  order, returns `{sha256, n_files, repo, revision, elapsed_s}`.

These tests run the FastAPI app in-process via `httpx.ASGITransport`
(same pattern as `test_eval_server_endpoints.py`) and mock
`snapshot_download` to a tmp_path with deterministic file content so
sha256 is independently verifiable.
"""
import hashlib
from pathlib import Path

import httpx
import pytest

import eval_server


@pytest.fixture
async def client():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=eval_server.app),
        base_url="http://eval-server-test",
    ) as c:
        yield c


def _write_safetensors(dir_: Path, name: str, content: bytes) -> Path:
    p = dir_ / name
    p.write_bytes(content)
    return p


def _expected_sha256(files: list[Path]) -> str:
    """Mirror endpoint's stream-over-sorted-files digest computation."""
    h = hashlib.sha256()
    for p in sorted(files):
        h.update(p.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------------
# Happy path: snapshot exists, sha256 streams over its safetensors.

async def test_returns_sha256_for_cached_repo(client, tmp_path, mocker):
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    f1 = _write_safetensors(snapshot_dir, "model-00001.safetensors",
                             b"weight-bytes-shard-1")
    f2 = _write_safetensors(snapshot_dir, "model-00002.safetensors",
                             b"weight-bytes-shard-2")
    mocker.patch("huggingface_hub.snapshot_download",
                 return_value=str(snapshot_dir))
    resp = await client.get("/hash",
                              params={"repo": "alice/model", "revision": "rev1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sha256"] == _expected_sha256([f1, f2])
    assert body["n_files"] == 2
    assert body["repo"] == "alice/model"
    assert body["revision"] == "rev1"
    assert "elapsed_s" in body


async def test_streams_in_alphabetical_filename_order(client, tmp_path, mocker):
    # Two file orders that hash differently if you concatenate in
    # arbitrary order. Endpoint pins sorted order; pin that here too.
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    a = _write_safetensors(snapshot_dir, "a.safetensors", b"AAA")
    b = _write_safetensors(snapshot_dir, "b.safetensors", b"BBB")
    mocker.patch("huggingface_hub.snapshot_download",
                 return_value=str(snapshot_dir))
    resp = await client.get("/hash",
                              params={"repo": "x/y", "revision": ""})
    assert resp.status_code == 200
    expected = hashlib.sha256()
    expected.update(a.read_bytes())   # alphabetical: a < b
    expected.update(b.read_bytes())
    assert resp.json()["sha256"] == expected.hexdigest()


async def test_revision_defaults_to_empty_passed_as_none_to_hf(
    client, tmp_path, mocker,
):
    # When ?revision= is missing or empty, endpoint passes
    # revision=None to snapshot_download (HF's "use HEAD" sentinel).
    snapshot_dir = tmp_path / "snap"
    snapshot_dir.mkdir()
    _write_safetensors(snapshot_dir, "m.safetensors", b"x")
    mock_dl = mocker.patch("huggingface_hub.snapshot_download",
                            return_value=str(snapshot_dir))
    await client.get("/hash", params={"repo": "alice/model"})
    # First positional arg is repo; revision is a kwarg.
    args, kwargs = mock_dl.call_args
    assert kwargs["revision"] is None


# ---------------------------------------------------------------------
# 404 paths: cache miss, empty snapshot.

async def test_returns_404_when_snapshot_download_raises(
    client, tmp_path, mocker,
):
    # Cache miss: snapshot_download raises (e.g.
    # LocalEntryNotFoundError when local_files_only=True and repo
    # is not cached).
    mocker.patch("huggingface_hub.snapshot_download",
                  side_effect=FileNotFoundError("not in cache"))
    resp = await client.get("/hash",
                              params={"repo": "ghost/model", "revision": "x"})
    assert resp.status_code == 404
    assert "not in cache" in resp.json()["detail"].lower() \
        or "FileNotFoundError" in resp.json()["detail"]


async def test_returns_404_when_snapshot_has_no_safetensors(
    client, tmp_path, mocker,
):
    snapshot_dir = tmp_path / "empty-snap"
    snapshot_dir.mkdir()
    # No *.safetensors written — only e.g. config.json.
    (snapshot_dir / "config.json").write_text("{}")
    mocker.patch("huggingface_hub.snapshot_download",
                  return_value=str(snapshot_dir))
    resp = await client.get("/hash",
                              params={"repo": "x/y", "revision": "z"})
    assert resp.status_code == 404
    assert "no safetensors" in resp.json()["detail"]


# ---------------------------------------------------------------------
# Determinism: same content → same digest, byte-identical to
# independent computation.

async def test_sha256_matches_independent_computation(
    client, tmp_path, mocker,
):
    snapshot_dir = tmp_path / "verifiable"
    snapshot_dir.mkdir()
    files = []
    for i in range(3):
        files.append(_write_safetensors(
            snapshot_dir, f"shard-{i:05d}.safetensors",
            f"deterministic-bytes-{i}".encode()))
    mocker.patch("huggingface_hub.snapshot_download",
                  return_value=str(snapshot_dir))
    resp = await client.get("/hash",
                              params={"repo": "x/y", "revision": ""})
    body = resp.json()
    # Reproduce the endpoint's exact algorithm and verify byte-equality.
    h = hashlib.sha256()
    for p in sorted(files):
        with open(p, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
    assert body["sha256"] == h.hexdigest()
    assert body["n_files"] == 3


async def test_two_calls_same_repo_return_identical_hash(
    client, tmp_path, mocker,
):
    snapshot_dir = tmp_path / "cache"
    snapshot_dir.mkdir()
    _write_safetensors(snapshot_dir, "m.safetensors", b"the-only-shard")
    mocker.patch("huggingface_hub.snapshot_download",
                  return_value=str(snapshot_dir))
    a = await client.get("/hash",
                           params={"repo": "x/y", "revision": "r1"})
    b = await client.get("/hash",
                           params={"repo": "x/y", "revision": "r1"})
    assert a.json()["sha256"] == b.json()["sha256"]
