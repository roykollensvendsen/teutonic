"""Tests for module-level check_king_alive(state).

Per docstring: "Verify king repo is still accessible at pinned revision.
Auto-dethrone if not."

Behavior:
- Returns True if king is alive (or unconfigured — nothing to verify).
- Returns False if king repo is unreachable at the pinned revision; in
  that case auto-dethrones to previous_king (when present) and emits
  a `king_dethroned_absent` event.

Call-site (validator.py:2280): False causes the tick loop to skip the
current iteration. So returning False is benign — the next tick sees
the reverted king.
"""
import pytest

from validator import check_king_alive


@pytest.fixture
def r2_mock(mocker):
    storage = {}
    appended = {}
    dashboards = {}
    r2 = mocker.MagicMock()
    r2.get.side_effect = lambda key: storage.get(key)
    r2.put.side_effect = lambda key, data: storage.update({key: data})
    r2.append_jsonl.side_effect = lambda key, rec: appended.setdefault(key, []).append(rec)
    r2.put_dashboard.side_effect = lambda key, data: dashboards.update({key: data})
    r2._storage = storage
    r2._appended = appended
    return r2


@pytest.fixture
def fake_hf(mocker):
    """Mock validator.HfApi.model_info — caller controls success/failure."""
    api_instance = mocker.MagicMock()
    mocker.patch("validator.HfApi", return_value=api_instance)
    return api_instance


@pytest.fixture
def state(r2_mock):
    from validator import State
    return State(r2_mock)


# ---------------------------------------------------------------------
# Unconfigured king — nothing to verify, treat as alive.

def test_returns_true_when_king_is_empty(state, fake_hf):
    # Default state has no king set.
    assert check_king_alive(state) is True


def test_returns_true_when_king_has_no_revision(state, fake_hf):
    # repo set but no pinned revision — nothing to verify against.
    state.king = {"hf_repo": "user/repo"}
    assert check_king_alive(state) is True


def test_unconfigured_king_does_not_call_hf(state, fake_hf):
    state.king = {}
    check_king_alive(state)
    fake_hf.model_info.assert_not_called()


# ---------------------------------------------------------------------
# Live king — HF accessible at pinned revision.

def test_returns_true_when_hf_succeeds(state, fake_hf):
    state.king = {"hf_repo": "user/repo", "king_revision": "abc123"}
    fake_hf.model_info.return_value = object()  # any non-raising return
    assert check_king_alive(state) is True


def test_alive_path_does_not_mutate_king(state, fake_hf):
    state.king = {"hf_repo": "user/repo", "king_revision": "abc123",
                  "hotkey": "hk1", "king_hash": "h"}
    fake_hf.model_info.return_value = object()
    pre = dict(state.king)
    check_king_alive(state)
    assert state.king == pre


# ---------------------------------------------------------------------
# Dead king — HF unreachable, auto-dethrone if previous_king exists.

def test_returns_false_when_hf_raises(state, fake_hf):
    state.king = {"hf_repo": "user/repo", "king_revision": "abc123",
                  "previous_king": {"hf_repo": "user/prev",
                                    "king_revision": "old", "hotkey": "prev"}}
    fake_hf.model_info.side_effect = RuntimeError("HF 404")
    assert check_king_alive(state) is False


def test_dead_king_reverts_to_previous_when_present(state, fake_hf):
    prev = {"hf_repo": "user/prev", "king_revision": "old", "hotkey": "prev"}
    state.king = {"hf_repo": "user/repo", "king_revision": "abc123",
                  "previous_king": prev, "hotkey": "current"}
    fake_hf.model_info.side_effect = RuntimeError("HF 404")
    check_king_alive(state)
    # state.king should now be the previous king.
    assert state.king["hotkey"] == "prev"
    assert state.king["hf_repo"] == "user/prev"


def test_dead_king_emits_dethrone_event(state, fake_hf, r2_mock):
    prev = {"hf_repo": "user/prev", "king_revision": "old", "hotkey": "prev"}
    state.king = {"hf_repo": "user/repo", "king_revision": "abc123",
                  "previous_king": prev, "hotkey": "current"}
    fake_hf.model_info.side_effect = RuntimeError("HF 404")
    check_king_alive(state)
    # event() goes to state/history.jsonl via append_jsonl.
    assert "state/history.jsonl" in r2_mock._appended
    events = r2_mock._appended["state/history.jsonl"]
    assert any(e.get("event") == "king_dethroned_absent" for e in events)


def test_dead_king_with_no_previous_returns_false_and_keeps_king(state, fake_hf):
    # No previous_king to revert to — function returns False but king unchanged.
    state.king = {"hf_repo": "user/repo", "king_revision": "abc123",
                  "hotkey": "current"}
    fake_hf.model_info.side_effect = RuntimeError("HF 404")
    pre = dict(state.king)
    result = check_king_alive(state)
    assert result is False
    # No revert possible — king should be unchanged.
    assert state.king == pre
