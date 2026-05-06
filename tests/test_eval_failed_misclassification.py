"""Tests for the EVALUATION FAILED-classification bug surface.

Discord triage 2026-05-04: high-volume complaints about models showing
"Evaluation Failed" without retry. Reporters @iamyamal, @rapiiidooo:

  "imo `EVALUATION FAILED` era started"
  "do we have any logic for these eval failed models?"
  "30 minutes per eval average + random failure ... unsustainable"

Production-data analysis (2026-05-06 dashboard.json snapshot from the
live validator at https://us-east-1.hippius.com/teutonic-sn3/):

  * 304 history entries; 94 (31%) have `verdict="error"`.
  * Within `verdict="error"`, `error_code="eval_error"` dominates:
    61/94 (65% of errors, 20% of all duels).
  * Within those 61 `eval_error` entries, the top distinct
    `error_detail` strings:
      27  "Server disconnected without sending a response."  TRANSIENT (retry-cap exhausted)
      17  "eval server error: ... could not load model with any
          attention implementation"                          TRANSIENT (retry-cap exhausted)
       6  "All connection attempts failed"                   PERMANENT — MISCLASSIFIED ❌
       3  "" (empty)                                         PERMANENT
       2  "0"                                                PERMANENT
       2  "eval server error: ... mat1/mat2 dtype"           TRANSIENT (retry-cap exhausted)
       3  "eval server error: ... prefetch ... stuck CDN"    TRANSIENT (retry-cap exhausted)
       1  "eval server error: ... unpack ... 2 bytes"        TRANSIENT (retry-cap exhausted)

  The dominant CLASSIFIER bug in production is "All connection attempts
  failed" — httpx.ConnectError's str() output. The marker list in
  validator._is_transient_eval_error contains "connecterror" but that
  matches the type name, not the str(). Real httpx exceptions never
  contain "connecterror" in their string representation, so 6 entries
  in the snapshot landed in `state.failed_repos` as permanent failures
  for what is unambiguously a transient TCP / DNS issue.

  The bigger PRACTICAL bug — 44+ entries (27+17+others) classified as
  transient but exhausting MAX_TRANSIENT_EVAL_RETRIES (3) — is a
  retry-cap / infra issue, not a classifier issue. Out of scope for
  this PR; tracked separately.

These tests pin the classifier's behaviour — comprehensive marker
coverage plus xfail-strict regression tests for misclassifications
production data confirms (primary) or theory predicts (secondary).
When a fix lands, the relevant xfail flips to pass; strict=True forces
removal of the marker.
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
#
# Primary: production-confirmed misclassification. 6 of 61 eval_error
# entries in the 2026-05-06 dashboard snapshot have this exact detail.

@pytest.mark.xfail(
    strict=True,
    reason="Production-confirmed misclassification: httpx.ConnectError "
           "(when the eval-server's tunnel / DNS / TCP setup fails). "
           "Its str() output is 'All connection attempts failed' — "
           "no surrounding context, no error type name. The transient "
           "markers tuple in validator._is_transient_eval_error includes "
           "'connecterror' but that matches the EXCEPTION TYPE NAME, "
           "never the str(). Production data shows 6/61 eval_error "
           "entries (~10%) misclassified this way; affected models "
           "land permanently in state.failed_repos when the underlying "
           "issue is unambiguously a transient infrastructure failure. "
           "Fix candidates (separate plan): add 'connection attempts "
           "failed' to markers, OR check exc.__class__.__name__ "
           "against a type-name set, OR catch httpx.ConnectError "
           "explicitly in process_challenge.",
)
def test_all_connection_attempts_failed_should_be_transient():
    # The exact str() that httpx.ConnectError produces when no TCP
    # connection succeeds (DNS failure, all RR-records unreachable,
    # tunnel down). Wrapped in RuntimeError because process_challenge's
    # _bounded_eval re-raises whatever it caught.
    # WANT: classified as transient → re-queue rather than permanent fail.
    # GET today: classified as permanent (production bug).
    is_transient, _reason = _is_transient_eval_error(
        RuntimeError("All connection attempts failed"))
    assert is_transient is True


# Secondary: theoretical misclassification. NOT seen in the 2026-05-06
# snapshot but the code path exists in validator.py and the dashboard
# already renders this string with retry-promising text.

@pytest.mark.xfail(
    strict=True,
    reason="Theoretical misclassification (no production occurrences "
           "in the 2026-05-06 dashboard snapshot, but the code path "
           "exists). validator.py around line 2174 raises "
           "RuntimeError('eval stream ended without verdict') when the "
           "SSE stream closes without a final verdict event. The "
           "dashboard renders this exact string as "
           "'This is likely transient -- your model will be retried' "
           "(website/index.html:481), but _is_transient_eval_error has "
           "no marker matching the text — so a model in this scenario "
           "would land permanently. Less urgent than the connection-"
           "attempts case but the same kind of dashboard/validator "
           "promise mismatch. Fix candidate: add 'ended without verdict' "
           "to markers.",
)
def test_eval_stream_ended_without_verdict_should_be_transient():
    is_transient, _reason = _is_transient_eval_error(
        RuntimeError("eval stream ended without verdict"))
    assert is_transient is True
