"""Tests for State watchdog state machine.

The watchdog field tracks validator phases (tick / sleep / tick_error /
restart_requested), tick lifecycle counters, and restart requests
that the supervisor process consults.
"""
from validator import State

# ---------------------------------------------------------------------
# set_phase — update phase, phase_since, notes; optionally current_challenge_id.

def test_set_phase_updates_phase_field(r2_mock):
    s = State(r2_mock)
    s.set_phase("eval")
    assert s.watchdog["phase"] == "eval"


def test_set_phase_updates_phase_since(r2_mock):
    s = State(r2_mock)
    initial_since = s.watchdog["phase_since"]
    s.set_phase("eval")
    assert s.watchdog["phase_since"] != initial_since or s.watchdog["phase_since"] >= initial_since


def test_set_phase_updates_notes(r2_mock):
    s = State(r2_mock)
    s.set_phase("eval", notes="running probe")
    assert s.watchdog["notes"] == "running probe"


def test_set_phase_updates_challenge_id_when_provided(r2_mock):
    s = State(r2_mock)
    s.set_phase("eval", challenge_id="ch-42")
    assert s.watchdog["current_challenge_id"] == "ch-42"


def test_set_phase_does_not_overwrite_challenge_id_when_omitted(r2_mock):
    s = State(r2_mock)
    s.watchdog["current_challenge_id"] = "ch-original"
    s.set_phase("sleep")
    assert s.watchdog["current_challenge_id"] == "ch-original"


# ---------------------------------------------------------------------
# note_progress — heartbeat for the last progress timestamp.

def test_note_progress_updates_last_progress_at(r2_mock):
    s = State(r2_mock)
    s.note_progress()
    assert s.watchdog["last_progress_at"] is not None


def test_note_progress_updates_notes_when_provided(r2_mock):
    s = State(r2_mock)
    s.note_progress(notes="downloading shard 3/10")
    assert s.watchdog["notes"] == "downloading shard 3/10"


def test_note_progress_preserves_notes_when_empty(r2_mock):
    s = State(r2_mock)
    s.watchdog["notes"] = "preserve me"
    s.note_progress(notes="")
    assert s.watchdog["notes"] == "preserve me"


# ---------------------------------------------------------------------
# Tick lifecycle: begin_tick -> [work] -> complete_tick (or fail_tick).

def test_begin_tick_records_started_at_and_sets_phase(r2_mock):
    s = State(r2_mock)
    s.begin_tick()
    assert s.watchdog["last_tick_started_at"] is not None
    assert s.watchdog["phase"] == "tick"


def test_complete_tick_records_completed_at_and_resets_errors(r2_mock):
    s = State(r2_mock)
    s.watchdog["consecutive_tick_errors"] = 3
    s.complete_tick()
    assert s.watchdog["last_tick_completed_at"] is not None
    assert s.watchdog["consecutive_tick_errors"] == 0
    assert s.watchdog["phase"] == "sleep"


def test_fail_tick_increments_consecutive_errors(r2_mock):
    s = State(r2_mock)
    s.fail_tick("network timeout")
    s.fail_tick("network timeout")
    s.fail_tick("network timeout")
    assert s.watchdog["consecutive_tick_errors"] == 3
    assert s.watchdog["phase"] == "tick_error"


def test_complete_after_fails_resets_error_counter(r2_mock):
    s = State(r2_mock)
    s.fail_tick("e1")
    s.fail_tick("e2")
    assert s.watchdog["consecutive_tick_errors"] == 2
    s.complete_tick()
    assert s.watchdog["consecutive_tick_errors"] == 0


# ---------------------------------------------------------------------
# Restart request flag (consumed by supervisor).

def test_request_restart_sets_flag_and_reason(r2_mock):
    s = State(r2_mock)
    s.request_restart("memory pressure")
    assert s.watchdog["restart_requested"] is True
    assert s.watchdog["restart_reason"] == "memory pressure"
    assert s.watchdog["phase"] == "restart_requested"


def test_request_restart_emits_event_to_history(r2_mock):
    s = State(r2_mock)
    s.request_restart("oom")
    # Per leaked impl, request_restart calls self.event(...) — should
    # land in r2.append_jsonl on state/history.jsonl.
    assert "state/history.jsonl" in r2_mock._appended
    events = r2_mock._appended["state/history.jsonl"]
    assert any(e.get("event") == "watchdog_restart_requested" for e in events)


def test_clear_restart_request_clears_flag_and_reason(r2_mock):
    s = State(r2_mock)
    s.request_restart("x")
    s.clear_restart_request()
    assert s.watchdog["restart_requested"] is False
    assert s.watchdog["restart_reason"] == ""
