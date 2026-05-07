"""Tests for eval_server lifecycle helpers.

Covers four cross-cutting concerns the FastAPI app needs running in
the background:

1. **`_schedule_self_kill`** — idempotent CUDA-fatal exit scheduler.
   Sets a module-level `_self_kill_scheduled` event and spawns a
   daemon thread that delays then calls `os._exit`. Idempotent so a
   second fatal-error report doesn't double-trigger.

2. **`_install_thread_excepthook`** — registers a `threading.excepthook`
   that detects CUDA-fatal text in uncaught thread exceptions and
   triggers `_schedule_self_kill`. Without this hook, a daemon thread
   can crash silently and leave the process running with corrupted
   GPU state.

3. **`_refresh_disk_stats_once` + `_disk_stats_loop` +
   `_ensure_disk_stats_thread` + `_get_disk_stats`** — disk-usage
   monitoring loop. Lifespan kicks off a background refresher so
   `/health` doesn't block on `scan_cache_dir` (a 600 GB cache scan
   takes ~60 s). The refresher writes to a module-level snapshot; the
   getter reads a copy of that snapshot non-blockingly.

These tests stub `os._exit`, `threading.Thread`, `shutil.disk_usage`,
and `huggingface_hub.scan_cache_dir` so no real exits, threads, or
filesystem scans happen.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import eval_server

# ---------------------------------------------------------------------
# Reset module-level state between tests so each starts clean.

@pytest.fixture(autouse=True)
def _reset_lifecycle_state(monkeypatch):
    """Clear `_self_kill_scheduled` and disk-stats globals between tests.

    Module-level threading.Event + dict-snapshot would otherwise carry
    state forward.
    """
    eval_server._self_kill_scheduled.clear()
    monkeypatch.setattr(eval_server, "_disk_stats_snapshot", {})
    monkeypatch.setattr(eval_server, "_disk_stats_thread_started", False)
    yield
    eval_server._self_kill_scheduled.clear()


# ---------------------------------------------------------------------
# _schedule_self_kill — idempotent, spawns daemon thread, calls os._exit.

@pytest.fixture
def fake_thread(monkeypatch):
    """Replace `threading.Thread` so tests can capture spawn-attempts
    without actually starting daemon threads."""
    spawned = []

    class FakeThread:
        def __init__(self, *, target, daemon=None, name=None):
            self.target = target
            self.daemon = daemon
            self.name = name
            spawned.append(self)

        def start(self):
            # Don't actually start — just record it.
            self.started = True

    monkeypatch.setattr(eval_server.threading, "Thread", FakeThread)
    return spawned


def test_schedule_self_kill_sets_event_and_spawns_thread(fake_thread):
    eval_server._schedule_self_kill("test reason")
    assert eval_server._self_kill_scheduled.is_set()
    assert len(fake_thread) == 1
    assert fake_thread[0].daemon is True
    assert fake_thread[0].name == "cuda-fatal-self-kill"
    assert fake_thread[0].started is True


def test_schedule_self_kill_is_idempotent(fake_thread):
    # Second call when event already set → return without spawning.
    eval_server._schedule_self_kill("first reason")
    eval_server._schedule_self_kill("second reason")
    assert len(fake_thread) == 1


def test_schedule_self_kill_die_function_calls_os_exit(fake_thread, monkeypatch):
    # The spawned thread's `target` is a `_die` closure. Run it
    # directly with a stubbed `os._exit` to verify exit-code wiring.
    exits = []
    monkeypatch.setattr(eval_server.os, "_exit", lambda code: exits.append(code))
    monkeypatch.setattr(eval_server.time, "sleep", lambda _s: None)
    eval_server._schedule_self_kill("test", delay_s=0)
    fake_thread[0].target()
    assert exits == [eval_server.CUDA_FATAL_EXIT_CODE]


def test_schedule_self_kill_uses_default_delay_when_none_passed(fake_thread, monkeypatch):
    # When delay_s is None, falls back to CUDA_FATAL_EXIT_DELAY_S.
    sleeps = []
    monkeypatch.setattr(eval_server.os, "_exit", lambda code: None)
    monkeypatch.setattr(eval_server.time, "sleep", lambda s: sleeps.append(s))
    eval_server._schedule_self_kill("test")  # delay_s default
    fake_thread[0].target()
    assert sleeps == [eval_server.CUDA_FATAL_EXIT_DELAY_S]


def test_schedule_self_kill_die_swallows_sleep_exception(fake_thread, monkeypatch):
    # Even if `time.sleep` raises (KeyboardInterrupt etc.), the
    # process-exit must still happen — the function wraps both
    # sleep + log calls in try/except.
    monkeypatch.setattr(eval_server.time, "sleep",
                        lambda _s: (_ for _ in ()).throw(RuntimeError("interrupted")))
    exits = []
    monkeypatch.setattr(eval_server.os, "_exit", lambda code: exits.append(code))
    eval_server._schedule_self_kill("test", delay_s=0)
    fake_thread[0].target()
    assert exits == [eval_server.CUDA_FATAL_EXIT_CODE]


# ---------------------------------------------------------------------
# _install_thread_excepthook — wraps threading.excepthook to detect
# CUDA-fatal errors in uncaught thread exceptions.

def test_install_thread_excepthook_calls_schedule_self_kill_on_cuda_fatal(
    fake_thread, monkeypatch,
):
    # Install hook → simulate a fatal thread exception → verify
    # _schedule_self_kill ran.
    schedule_calls = []
    monkeypatch.setattr(eval_server, "_schedule_self_kill",
                        lambda reason, **_kw: schedule_calls.append(reason))
    # Replace the existing excepthook with a benign no-op so the
    # test's installed hook doesn't chain into a previous test's
    # CUDA-fatal-detection layer (each call to _install_thread_excepthook
    # wraps the current hook).
    monkeypatch.setattr(eval_server.threading, "excepthook",
                        lambda _args: None)

    eval_server._install_thread_excepthook()

    # Build a thread-exception args matching threading.ExceptHookArgs.
    fake_args = SimpleNamespace(
        exc_type=RuntimeError,
        exc_value=RuntimeError("CUDA error: an illegal memory access"),
        exc_traceback=None,
        thread=SimpleNamespace(name="some-worker"),
    )
    eval_server.threading.excepthook(fake_args)

    assert len(schedule_calls) == 1
    assert "CUDA error" in schedule_calls[0]


def test_install_thread_excepthook_ignores_non_cuda_exceptions(monkeypatch):
    schedule_calls = []
    monkeypatch.setattr(eval_server, "_schedule_self_kill",
                        lambda reason, **_kw: schedule_calls.append(reason))

    # Avoid mutating the global excepthook permanently — restore at end.
    original_excepthook = eval_server.threading.excepthook
    try:
        # Stub the prior hook so it doesn't print tracebacks during test.
        eval_server.threading.excepthook = lambda args: None
        eval_server._install_thread_excepthook()

        fake_args = SimpleNamespace(
            exc_type=ValueError,
            exc_value=ValueError("just a regular error"),
            exc_traceback=None,
            thread=SimpleNamespace(name="some-worker"),
        )
        eval_server.threading.excepthook(fake_args)
    finally:
        eval_server.threading.excepthook = original_excepthook

    # Non-CUDA errors must NOT trigger self-kill.
    assert schedule_calls == []


# ---------------------------------------------------------------------
# _refresh_disk_stats_once — single shutil.disk_usage + scan_cache_dir.

def test_refresh_disk_stats_once_returns_disk_fields(monkeypatch):
    monkeypatch.setattr(eval_server.shutil, "disk_usage",
                        lambda _p: SimpleNamespace(total=int(1e12),
                                                    used=int(6e11),
                                                    free=int(4e11)))
    # scan_cache_dir returns a CacheInfo-shaped object.
    cache_info = SimpleNamespace(
        size_on_disk=int(2e10),  # 20 GB
        repos=[SimpleNamespace(revisions=[1, 2]),
               SimpleNamespace(revisions=[3])],
    )
    monkeypatch.setattr("huggingface_hub.scan_cache_dir",
                        lambda: cache_info)

    stats = eval_server._refresh_disk_stats_once()
    assert stats["disk_total_gb"] == 1000.0
    assert stats["disk_used_gb"] == 600.0
    assert stats["disk_free_gb"] == 400.0
    assert stats["hf_cache_size_gb"] == 20.0
    assert stats["hf_cache_repos"] == 2
    assert stats["hf_cache_revisions"] == 3


def test_refresh_disk_stats_once_swallows_disk_usage_error(monkeypatch):
    # disk_usage raises → disk_* fields absent, but cache-side still runs.
    monkeypatch.setattr(eval_server.shutil, "disk_usage",
                        lambda _p: (_ for _ in ()).throw(OSError("readlink failed")))
    monkeypatch.setattr("huggingface_hub.scan_cache_dir",
                        lambda: SimpleNamespace(size_on_disk=0, repos=[]))
    stats = eval_server._refresh_disk_stats_once()
    assert "disk_total_gb" not in stats
    assert stats["hf_cache_size_gb"] == 0.0


def test_refresh_disk_stats_once_swallows_scan_cache_dir_error(monkeypatch):
    monkeypatch.setattr(eval_server.shutil, "disk_usage",
                        lambda _p: SimpleNamespace(total=1, used=1, free=0))
    monkeypatch.setattr("huggingface_hub.scan_cache_dir",
                        lambda: (_ for _ in ()).throw(OSError("hf cache locked")))
    stats = eval_server._refresh_disk_stats_once()
    assert "disk_total_gb" in stats
    assert "hf_cache_size_gb" not in stats


# ---------------------------------------------------------------------
# _ensure_disk_stats_thread — idempotent thread spawner.

def test_ensure_disk_stats_thread_spawns_once(fake_thread):
    eval_server._ensure_disk_stats_thread()
    eval_server._ensure_disk_stats_thread()
    eval_server._ensure_disk_stats_thread()
    assert len(fake_thread) == 1
    assert fake_thread[0].daemon is True
    assert fake_thread[0].name == "disk-stats-refresher"
    assert eval_server._disk_stats_thread_started is True


# ---------------------------------------------------------------------
# _get_disk_stats — returns snapshot copy, ensures thread is running.

def test_get_disk_stats_returns_copy_not_internal_dict(fake_thread, monkeypatch):
    # Pre-populate snapshot via direct assignment (the thread is
    # mocked so it won't actually run the loop).
    monkeypatch.setattr(eval_server, "_disk_stats_snapshot",
                        {"disk_total_gb": 500.0, "disk_used_gb": 200.0})
    snapshot = eval_server._get_disk_stats()
    assert snapshot == {"disk_total_gb": 500.0, "disk_used_gb": 200.0}
    # Caller-side mutation must not bleed back.
    snapshot["injected"] = True
    assert "injected" not in eval_server._disk_stats_snapshot


def test_get_disk_stats_calls_ensure_thread(fake_thread):
    # First call from the test fixture cleared the thread-started flag.
    eval_server._get_disk_stats()
    assert eval_server._disk_stats_thread_started is True


def test_get_disk_stats_returns_empty_dict_before_first_refresh(fake_thread):
    # Before the background refresher has populated the snapshot,
    # the getter returns an empty dict (not None / not raising).
    snapshot = eval_server._get_disk_stats()
    assert snapshot == {}


# ---------------------------------------------------------------------
# _disk_stats_loop — single iteration via injected refresh.

def test_disk_stats_loop_writes_snapshot_each_iteration(monkeypatch):
    # Drive ONE iteration, then break out via raising in time.sleep
    # (mirrors how a daemon thread is normally signalled to stop).
    refresh_calls = []
    monkeypatch.setattr(eval_server, "_refresh_disk_stats_once",
                        lambda: (refresh_calls.append(1),
                                 {"disk_total_gb": 100.0})[1])

    class _Stop(BaseException):
        pass

    monkeypatch.setattr(eval_server.time, "sleep",
                        lambda _s: (_ for _ in ()).throw(_Stop()))

    with pytest.raises(_Stop):
        eval_server._disk_stats_loop()

    assert len(refresh_calls) == 1
    assert eval_server._disk_stats_snapshot == {"disk_total_gb": 100.0}


def test_disk_stats_loop_swallows_refresh_exceptions(monkeypatch):
    # If _refresh_disk_stats_once raises, the loop logs and
    # continues — so the next time.sleep still fires.
    def raising_refresh():
        raise RuntimeError("disk gone")

    monkeypatch.setattr(eval_server, "_refresh_disk_stats_once",
                        raising_refresh)

    sleep_count = [0]

    class _Stop(BaseException):
        pass

    def _fake_sleep(_s):
        sleep_count[0] += 1
        raise _Stop()

    monkeypatch.setattr(eval_server.time, "sleep", _fake_sleep)

    # The first iteration's refresh raises but is swallowed;
    # control reaches time.sleep, which raises _Stop to exit.
    with pytest.raises(_Stop):
        eval_server._disk_stats_loop()
    assert sleep_count[0] == 1


# ---------------------------------------------------------------------
# Reaching _is_cuda_fatal via the excepthook closure (extra coverage).

def test_excepthook_calls_prior_hook_after_check(monkeypatch, fake_thread):
    # Pin the contract that the wrapped excepthook ALWAYS calls the
    # prior hook (regardless of CUDA-fatal classification). This is
    # what keeps default Python tracebacks visible in production logs.
    prior_calls = []
    monkeypatch.setattr(
        eval_server.threading, "excepthook", lambda args: prior_calls.append(args))
    monkeypatch.setattr(eval_server, "_schedule_self_kill", lambda *_a, **_kw: None)

    eval_server._install_thread_excepthook()

    fake_args = SimpleNamespace(
        exc_type=ValueError,
        exc_value=ValueError("benign"),
        exc_traceback=None,
        thread=MagicMock(name="thread"),
    )
    eval_server.threading.excepthook(fake_args)
    assert len(prior_calls) == 1
