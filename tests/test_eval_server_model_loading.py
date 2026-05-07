"""Tests for eval_server._ensure_king and _load_challenger.

Both functions construct `MultiGPUEvaluator` instances — heavy
torch/HF objects we cannot instantiate in unit tests. We mock
the constructor to return a recording stub and verify the wiring
(GPU split, revision passthrough, king-cache reuse, probe gating).

`_ensure_king` is the more involved one: it caches the king evaluator
keyed on (repo, revision, king_hash), reloads (with shutdown + cuda
empty-cache) when any of those change, and runs `trainability_probe`
on a fresh load — refusing to install a king that fails the probe
(invariant: king got there by winning an eval, which already passed
the probe, so a probe-fail at king-load time is a real bug).
"""
from unittest.mock import MagicMock

import pytest

import eval_server

# ---------------------------------------------------------------------
# Reset module-level king state between tests so each starts clean.

@pytest.fixture(autouse=True)
def _reset_king_globals(monkeypatch):
    monkeypatch.setattr(eval_server, "_king_evaluator", None)
    monkeypatch.setattr(eval_server, "_king_repo", None)
    monkeypatch.setattr(eval_server, "_king_hash", None)
    monkeypatch.setattr(eval_server, "_king_revision", None)
    monkeypatch.setattr(eval_server, "_gpu_ids", [0, 1, 2, 3])
    yield


@pytest.fixture
def fake_evaluator_cls(mocker):
    """Replace `MultiGPUEvaluator` with a recording factory.

    Returns a list `instances` of every (kwargs) tuple recorded.
    Each constructed evaluator has a `.shutdown()` MagicMock and a
    `.models` dict shaped like the real class (gpu_id -> model mock).
    """
    instances = []

    def factory(repo, gpu_ids, *, label, force_download=False,
                revision=None, on_phase=None):
        ev = MagicMock()
        ev.repo = repo
        ev.gpu_ids = gpu_ids
        ev.label = label
        ev.revision = revision
        ev.force_download = force_download
        ev.on_phase = on_phase
        # The probe walks `evaluator.models[gpu_ids[0]]` for the king.
        ev.models = {gpu_ids[0]: MagicMock(name=f"{label}_model")}
        ev.shutdown = MagicMock()
        instances.append(ev)
        return ev

    mocker.patch("eval_server.MultiGPUEvaluator", side_effect=factory)
    return instances


@pytest.fixture
def fake_probe(mocker):
    """Patch `trainability_probe` so probe results are test-controllable."""
    return mocker.patch("eval_server.trainability_probe",
                        return_value={"ok": True, "reason": None,
                                       "max_ratio": 1.0,
                                       "max_grad_norm": 0.5,
                                       "min_loss_before": 1.0,
                                       "max_loss_after": 1.05,
                                       "norm_quantization": 0.4,
                                       "n_seeds": 3,
                                       "n_steps_per_seed": 0,
                                       "warnings": []})


@pytest.fixture
def fake_cuda(mocker):
    """No-op torch.cuda.empty_cache so reload paths don't crash."""
    return mocker.patch("eval_server.torch.cuda.empty_cache")


# ---------------------------------------------------------------------
# _ensure_king — caching by (repo, revision, hash).

def test_ensure_king_first_call_constructs_evaluator(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", True)
    result = eval_server._ensure_king(
        "alice/Teutonic-XXIV-king", king_hash="abc",
        revision="rev1234567890",
    )
    assert len(fake_evaluator_cls) == 1
    assert fake_evaluator_cls[0].repo == "alice/Teutonic-XXIV-king"
    assert fake_evaluator_cls[0].revision == "rev1234567890"
    assert fake_evaluator_cls[0].label == "king"
    assert result is fake_evaluator_cls[0]
    # Module globals are populated.
    assert eval_server._king_repo == "alice/Teutonic-XXIV-king"
    assert eval_server._king_hash == "abc"
    assert eval_server._king_revision == "rev1234567890"


def test_ensure_king_reuses_cached_when_all_keys_match(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", True)
    first = eval_server._ensure_king("repo", king_hash="kh", revision="rev1")
    second = eval_server._ensure_king("repo", king_hash="kh", revision="rev1")
    assert first is second
    assert len(fake_evaluator_cls) == 1  # only one construction


def test_ensure_king_reloads_when_repo_changes(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", True)
    first = eval_server._ensure_king("repo-A", king_hash="kh", revision="rev1")
    second = eval_server._ensure_king("repo-B", king_hash="kh", revision="rev1")
    assert first is not second
    assert len(fake_evaluator_cls) == 2
    # Old evaluator was shut down before the new one was built.
    first.shutdown.assert_called_once()
    fake_cuda.assert_called()


def test_ensure_king_reloads_when_revision_changes(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", True)
    first = eval_server._ensure_king("repo", king_hash="kh", revision="rev_old")
    second = eval_server._ensure_king("repo", king_hash="kh", revision="rev_new")
    assert first is not second


def test_ensure_king_reloads_when_king_hash_changes(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", True)
    first = eval_server._ensure_king("repo", king_hash="hash_old", revision="rev")
    second = eval_server._ensure_king("repo", king_hash="hash_new", revision="rev")
    assert first is not second


def test_ensure_king_reuses_when_revision_omitted_after_first(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    # Empty revision in subsequent calls means "don't compare revision",
    # not "must match empty" — return cached.
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", True)
    first = eval_server._ensure_king("repo", king_hash="kh", revision="rev1")
    second = eval_server._ensure_king("repo", king_hash="kh", revision="")
    assert first is second
    assert len(fake_evaluator_cls) == 1


# ---------------------------------------------------------------------
# Probe gating: PROBE_ENABLED + probe-fail handling.

def test_ensure_king_runs_probe_when_enabled(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", True)
    eval_server._ensure_king("repo")
    fake_probe.assert_called_once()


def test_ensure_king_skips_probe_when_disabled(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", False)
    eval_server._ensure_king("repo")
    fake_probe.assert_not_called()


def test_ensure_king_raises_and_shuts_down_on_probe_fail(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", True)
    fake_probe.return_value = {"ok": False, "reason": "norm_quant_high",
                                "max_ratio": 99.0, "max_grad_norm": 1e9,
                                "min_loss_before": 1.0, "max_loss_after": 99.0,
                                "norm_quantization": 1.0,
                                "n_seeds": 3, "n_steps_per_seed": 0,
                                "warnings": []}
    with pytest.raises(RuntimeError, match="failed trainability probe"):
        eval_server._ensure_king("repo", revision="rev")
    # The newly-constructed evaluator gets shut down before the raise.
    assert len(fake_evaluator_cls) == 1
    fake_evaluator_cls[0].shutdown.assert_called_once()
    fake_cuda.assert_called()
    # Module globals MUST stay None — the failed king was rejected.
    assert eval_server._king_evaluator is None


def test_ensure_king_invokes_on_phase_around_probe(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", True)
    phases = []
    eval_server._ensure_king("repo", on_phase=lambda d: phases.append(d))
    # At minimum the probe-start phase fires; constructor's on_phase is
    # also invoked but goes through MultiGPUEvaluator (mocked).
    phase_names = [p.get("phase") for p in phases]
    assert "king_probe_start" in phase_names


def test_ensure_king_swallows_on_phase_callback_exception(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    # If the validator's on_phase callback raises (e.g. SSE write failed),
    # _ensure_king must NOT propagate — the king load must complete.
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", True)

    def raising(_d):
        raise RuntimeError("sse send failed")

    eval_server._ensure_king("repo", on_phase=raising)


# ---------------------------------------------------------------------
# GPU split: king takes first half, challenger takes second half.

def test_ensure_king_uses_first_half_of_gpus(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", True)
    monkeypatch.setattr(eval_server, "_gpu_ids", [0, 1, 2, 3])
    eval_server._ensure_king("repo")
    assert fake_evaluator_cls[0].gpu_ids == [0, 1]


def test_ensure_king_falls_back_to_first_gpu_when_only_one_available(
    fake_evaluator_cls, fake_probe, fake_cuda, monkeypatch,
):
    # Single-GPU fallback: mid=0, king_gpus = [] or _gpu_ids[:1].
    monkeypatch.setattr(eval_server, "PROBE_ENABLED", True)
    monkeypatch.setattr(eval_server, "_gpu_ids", [7])
    eval_server._ensure_king("repo")
    assert fake_evaluator_cls[0].gpu_ids == [7]


# ---------------------------------------------------------------------
# _load_challenger — uses second half of GPUs.

def test_load_challenger_uses_second_half_of_gpus(fake_evaluator_cls, monkeypatch):
    monkeypatch.setattr(eval_server, "_gpu_ids", [0, 1, 2, 3])
    result = eval_server._load_challenger("bob/repo", revision="rev")
    assert len(fake_evaluator_cls) == 1
    assert fake_evaluator_cls[0].gpu_ids == [2, 3]
    assert fake_evaluator_cls[0].repo == "bob/repo"
    assert fake_evaluator_cls[0].label == "challenger"
    assert fake_evaluator_cls[0].revision == "rev"
    assert result is fake_evaluator_cls[0]


def test_load_challenger_falls_back_to_first_gpu_when_only_one(
    fake_evaluator_cls, monkeypatch,
):
    monkeypatch.setattr(eval_server, "_gpu_ids", [5])
    eval_server._load_challenger("bob/repo")
    assert fake_evaluator_cls[0].gpu_ids == [5]


def test_load_challenger_passes_on_phase_through(fake_evaluator_cls, monkeypatch):
    monkeypatch.setattr(eval_server, "_gpu_ids", [0, 1])
    captured: list = []
    eval_server._load_challenger("repo", on_phase=lambda d: captured.append(d))
    assert fake_evaluator_cls[0].on_phase is not None


def test_load_challenger_omits_revision_when_empty(fake_evaluator_cls, monkeypatch):
    # `revision=""` → the constructor receives `revision=None`
    # (the function does `revision=revision or None`).
    monkeypatch.setattr(eval_server, "_gpu_ids", [0, 1])
    eval_server._load_challenger("repo", revision="")
    assert fake_evaluator_cls[0].revision is None
