"""Tests for the EVALUATION FAILED-classification bug surface.

Discord triage 2026-05-04: high-volume complaints about models showing
"Evaluation Failed" without retry. Reporters @iamyamal, @rapiiidooo:

  "imo `EVALUATION FAILED` era started"
  "do we have any logic for these eval failed models?"
  "30 minutes per eval average + random failure ... unsustainable"

Root cause investigation:

`process_challenge` raises `RuntimeError("eval stream ended without
verdict")` (validator.py around line 2174) when the eval-server's SSE
stream closes without delivering a final verdict event (eval-server
restart, tunnel blip, k8s pod cycle, etc.).

`_is_transient_eval_error` (validator.py:1863) checks the exception
text against a hardcoded marker list. NONE of those markers match the
substring "eval stream ended without verdict", so the exception is
classified as PERMANENT — `record_failure(..., "eval_error", ...)`
puts it in the duel history and `state.failed_repos.add(hf_repo)`.

But the dashboard renders this exact error_detail with the message
"This is likely transient -- your model will be retried"
(website/index.html:481). Direct mismatch: the dashboard tells miners
their model will retry; the validator never retries it.

These tests pin the classifier's behaviour — current marker coverage
plus an xfail-strict regression test that demonstrates the
misclassification. When the fix lands ("eval stream ended without
verdict" added to transient markers, OR the validator emits a
different exception that does match), the xfail flips to pass and
strict=True forces the marker to be removed.
"""
import asyncio

import pytest

from validator import _is_transient_eval_error

# ---------------------------------------------------------------------
# Each documented transient marker classifies as transient.
#
# Source: `transient_markers` tuple in validator._is_transient_eval_error
# (validator.py around line 1873-1897). The list groups three classes
# of failures: eval-server-side errors, network/connection errors,
# and HTTP-status errors. Pinning each ensures a refactor doesn't
# accidentally drop one.

@pytest.mark.parametrize("marker", [
    # Eval-server-side errors.
    "eval server error",
    "internal error",
    "stream idle",
    "watchdog timeout",
    "timed out",
    "timeout",
    # Network / connection errors.
    "server disconnected",
    "connection reset",
    "connecterror",
    "readerror",
    "remoteprotocolerror",
    "streamconsumed",
    "streamclosed",
    "streamerror",
    "peer closed connection",
    "incomplete chunked",
    "incompleteread",
    # HTTP status codes that always indicate infra problems.
    "503",
    "502",
    "504",
])
def test_marker_classifies_as_transient(marker):
    is_transient, reason = _is_transient_eval_error(Exception(marker))
    assert is_transient is True, f"marker {marker!r} should be transient"
    assert reason == marker


def test_marker_substring_match_is_case_insensitive():
    # The classifier lowercases the exception text before matching, so
    # "Stream Idle" / "STREAMERROR" must classify the same as the
    # canonical lowercase markers.
    is_transient, _reason = _is_transient_eval_error(
        Exception("Connection Reset by peer"))
    assert is_transient is True


def test_marker_can_appear_anywhere_in_exception_text():
    # Real exceptions wrap the marker in surrounding diagnostic text,
    # e.g. "RuntimeError: eval server error: GPU OOM at layer 7".
    # Pin substring matching.
    is_transient, _reason = _is_transient_eval_error(
        RuntimeError("RuntimeError: eval server error: GPU OOM at layer 7"))
    assert is_transient is True


# ---------------------------------------------------------------------
# CancelledError is special-cased by type, not text. str(CancelledError())
# is "" so the marker scan would mis-classify it as fatal.

def test_cancelled_error_is_transient_via_type_not_text():
    is_transient, reason = _is_transient_eval_error(
        asyncio.CancelledError())
    assert is_transient is True
    assert reason == "validator_cancelled"


# ---------------------------------------------------------------------
# Permanent failures — exceptions that don't match any marker stay
# permanent. The dashboard renders these with bespoke messages
# (website/index.html ERROR_MESSAGES["eval_error"]) and they go in
# state.failed_repos so the model isn't re-evaluated.

def test_cuda_oom_is_permanent():
    # Dashboard renders this as a user-error: "Your model ran out of
    # GPU memory ... ensure your model fits within the validator's
    # GPU constraints." Treating as permanent matches that framing.
    is_transient, _reason = _is_transient_eval_error(
        Exception("CUDA out of memory while loading"))
    assert is_transient is False


def test_size_mismatch_is_permanent():
    # Dashboard: "Model weight shapes don't match the expected
    # architecture. Verify your safetensors match your config.json."
    is_transient, _reason = _is_transient_eval_error(
        Exception("size mismatch for layer.0.attn.q_proj"))
    assert is_transient is False


def test_unrelated_exception_text_is_permanent():
    # Catch-all for unmapped exceptions.
    is_transient, _reason = _is_transient_eval_error(
        Exception("totally unrelated diagnostic"))
    assert is_transient is False


# ---------------------------------------------------------------------
# Polymorphic input: classifier accepts Exception OR str.

def test_classifier_accepts_string_input():
    is_transient, _reason = _is_transient_eval_error("stream idle for 200s")
    assert is_transient is True


def test_classifier_accepts_exception_input():
    is_transient, _reason = _is_transient_eval_error(
        Exception("stream idle for 200s"))
    assert is_transient is True


# ---------------------------------------------------------------------
# Bug demonstration — xfail-strict.

@pytest.mark.xfail(
    strict=True,
    reason="EVALUATION FAILED-misclassification bug "
           "(Discord triage 2026-05-04, @iamyamal, @rapiiidooo). "
           "validator.py raises RuntimeError('eval stream ended without "
           "verdict') when the SSE stream closes without a final "
           "verdict (eval-server restart, tunnel blip). The dashboard "
           "renders this as 'This is likely transient -- your model will "
           "be retried' (website/index.html:481), but "
           "_is_transient_eval_error has no marker matching this text — "
           "so the validator records a permanent failure and the model "
           "lands in state.failed_repos. Fix is a separate plan; the "
           "obvious option is to add 'stream ended without verdict' (or "
           "'ended without verdict') to the transient markers tuple. "
           "When the fix lands, this xfail flips to pass and strict=True "
           "forces removal of the marker.",
)
def test_eval_stream_ended_without_verdict_should_be_transient():
    # The exact RuntimeError validator.py raises on a no-verdict stream
    # close (validator.py around line 2174):
    #     raise RuntimeError("eval stream ended without verdict")
    # WANT: classified as transient → re-queue rather than permanent fail.
    # GET today: classified as permanent (bug).
    is_transient, _reason = _is_transient_eval_error(
        RuntimeError("eval stream ended without verdict"))
    assert is_transient is True
