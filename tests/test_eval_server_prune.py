"""Tests for eval_server._prune_evals.

Bounds the in-memory eval-record dict to keep the FastAPI server's
working set small. Two pruning rules:

* Age — completed/failed records older than `EVAL_MAX_AGE_S` get
  removed.
* Count cap — if `len(_evals)` after the age sweep exceeds
  `MAX_EVALS_KEPT`, additional finished records (oldest by
  `created_at`) get trimmed. The cap targets *total* in-memory
  state, not finished-only — but only finished records are
  eligible for removal, so if active state alone already exceeds
  the cap the function cannot meet it (it just drains all
  finished).

Active records (state != completed/failed) are never pruned.
"""
import time

import pytest

import eval_server


@pytest.fixture(autouse=True)
def _isolate_evals_dict():
    """Each test gets a fresh `_evals` dict; restore on teardown.

    Without this, a test's seed pollutes the module-level dict for any
    later test that reads `eval_server._evals`.
    """
    saved = eval_server._evals
    eval_server._evals = {}
    yield
    eval_server._evals = saved


def _eval(state, *, age_s=0.0):
    """Build an eval record `_prune_evals` understands.

    `created_at` is back-dated by `age_s` seconds from now.
    """
    return {"state": state, "created_at": time.time() - age_s}


def _seed(records):
    """Replace _evals with the given dict; previous state is dropped."""
    eval_server._evals.clear()
    eval_server._evals.update(records)


def test_prune_removes_completed_records_past_age_threshold(monkeypatch):
    monkeypatch.setattr(eval_server, "EVAL_MAX_AGE_S", 100)
    monkeypatch.setattr(eval_server, "MAX_EVALS_KEPT", 1000)
    _seed({
        "old": _eval("completed", age_s=1000),
        "fresh": _eval("completed", age_s=10),
    })

    eval_server._prune_evals()

    assert "old" not in eval_server._evals
    assert "fresh" in eval_server._evals


def test_prune_keeps_active_records_regardless_of_age(monkeypatch):
    # An eval still in-flight (state != completed/failed) must NEVER
    # be pruned, even if its created_at is ancient — pruning a running
    # record would lose the connection between an in-flight worker and
    # the HTTP client polling for its result.
    monkeypatch.setattr(eval_server, "EVAL_MAX_AGE_S", 100)
    monkeypatch.setattr(eval_server, "MAX_EVALS_KEPT", 1000)
    _seed({
        "running": _eval("running", age_s=99999),
        "queued": _eval("queued", age_s=99999),
    })

    eval_server._prune_evals()

    assert "running" in eval_server._evals
    assert "queued" in eval_server._evals


def test_prune_caps_finished_records_to_max_kept(monkeypatch):
    # 5 finished records, cap is 2 → 3 oldest get removed.
    monkeypatch.setattr(eval_server, "EVAL_MAX_AGE_S", 99999)  # disable age sweep
    monkeypatch.setattr(eval_server, "MAX_EVALS_KEPT", 2)
    _seed({
        "e0": _eval("completed", age_s=50),
        "e1": _eval("completed", age_s=40),
        "e2": _eval("completed", age_s=30),
        "e3": _eval("completed", age_s=20),
        "e4": _eval("completed", age_s=10),
    })

    eval_server._prune_evals()

    assert sorted(eval_server._evals.keys()) == ["e3", "e4"]


def test_prune_cap_targets_total_size_not_finished_only(monkeypatch):
    # MAX_EVALS_KEPT bounds *total* len(_evals). With 3 finished + 2
    # active and cap=3, the function must remove enough finished to
    # bring total to 3 (so 2 oldest finished go, 1 finished remains
    # alongside the 2 actives).
    monkeypatch.setattr(eval_server, "EVAL_MAX_AGE_S", 99999)
    monkeypatch.setattr(eval_server, "MAX_EVALS_KEPT", 3)
    _seed({
        "f_old": _eval("completed", age_s=30),
        "f_mid": _eval("completed", age_s=20),
        "f_new": _eval("completed", age_s=10),
        "running": _eval("running", age_s=15),
        "queued": _eval("queued", age_s=15),
    })

    eval_server._prune_evals()

    assert "running" in eval_server._evals
    assert "queued" in eval_server._evals
    # f_new is the youngest finished and survives; f_old + f_mid go.
    assert "f_new" in eval_server._evals
    assert "f_old" not in eval_server._evals
    assert "f_mid" not in eval_server._evals
    assert len(eval_server._evals) == 3


def test_prune_cap_unreachable_when_active_alone_exceeds_it(monkeypatch):
    # If the cap is smaller than the active count, the function
    # cannot meet it — it removes all finished but cannot touch
    # actives. Records the contract that active state has implicit
    # priority over the cap.
    monkeypatch.setattr(eval_server, "EVAL_MAX_AGE_S", 99999)
    monkeypatch.setattr(eval_server, "MAX_EVALS_KEPT", 1)
    _seed({
        "f0": _eval("completed", age_s=10),
        "f1": _eval("completed", age_s=20),
        "r0": _eval("running", age_s=5),
        "r1": _eval("running", age_s=5),
    })

    eval_server._prune_evals()

    # All finished gone; both actives survive (cap can't kill them).
    assert sorted(eval_server._evals.keys()) == ["r0", "r1"]
