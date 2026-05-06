"""Tests for eval_server._is_cuda_fatal.

Pure substring-match against `_CUDA_FATAL_TOKENS` — when a thread
exception or eval error contains one of these tokens, the eval-server
schedules a self-kill so the supervisor can restart it with fresh
CUDA state. Misclassifying a fatal error as recoverable would leave
the process running with corrupted GPU state; misclassifying a
recoverable error as fatal would over-restart and lose useful work.
"""
# ---------------------------------------------------------------------
# Each documented fatal token classifies as fatal.
import pytest

from eval_server import _CUDA_FATAL_TOKENS, _is_cuda_fatal


@pytest.mark.parametrize("token", list(_CUDA_FATAL_TOKENS))
def test_each_fatal_token_classifies_as_fatal(token):
    # Exact-substring presence is the contract — if any token from
    # the tuple appears anywhere in the message, it's fatal.
    assert _is_cuda_fatal(token) is True


def test_token_can_appear_anywhere_in_message():
    # Real CUDA exceptions wrap the marker in surrounding diagnostic
    # text (file, line, traceback). Pin substring matching.
    msg = ("RuntimeError: CUDA error: an illegal memory access was "
           "encountered\nCUDA kernel errors might be asynchronously...")
    assert _is_cuda_fatal(msg) is True


# ---------------------------------------------------------------------
# Non-fatal classifies as not-fatal.

def test_unrelated_error_text_is_not_fatal():
    assert _is_cuda_fatal("ValueError: bad config field") is False


def test_empty_string_is_not_fatal():
    assert _is_cuda_fatal("") is False


def test_none_input_treated_as_empty():
    # `str(None or "")` → "" — pin that None doesn't crash the function.
    assert _is_cuda_fatal(None) is False


# ---------------------------------------------------------------------
# Token list is non-empty (regression guard against an empty tuple
# silently disabling self-kill detection).

def test_fatal_token_list_is_non_empty():
    assert len(_CUDA_FATAL_TOKENS) > 0


def test_fatal_tokens_all_strings():
    for tok in _CUDA_FATAL_TOKENS:
        assert isinstance(tok, str)
        assert tok != ""
