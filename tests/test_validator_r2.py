"""Tests for validator.R2 — boto3-backed S3 wrapper.

R2 wraps three potential S3-compatible clients (Cloudflare R2 for
state, Hippius for the public dashboard, and an optional dataset
store). All public methods follow the same defensive contract:
S3 errors are caught and logged but never raised — the validator
main loop must keep running through transient storage failures.

These tests mock `validator.boto3.client` at module level so R2()
construction doesn't try to reach a real S3 endpoint. Each test then
inspects the recorded boto3 client mock to verify call args.
"""
import json
from unittest.mock import MagicMock

import pytest

import validator

# ---------------------------------------------------------------------
# fake_boto fixture: mock boto3.client + module-level credential vars
# so R2() construction creates predictable mock clients.

def _make_boto_fixture(mocker, monkeypatch, *, hippius: bool = False, dataset: bool = False):
    """Pre-create mock clients so tests can configure their methods BEFORE
    R2() is constructed. R2.__init__ creates clients in this order:
      [0] R2 (always)
      [1] Hippius (only if HIPPIUS_*_KEY both set)
      [2] dataset store (only if DS_*_KEY all set)
    """
    monkeypatch.setattr(validator, "R2_ACCESS_KEY", "r2-key")
    monkeypatch.setattr(validator, "R2_SECRET_KEY", "r2-secret")
    monkeypatch.setattr(validator, "R2_ENDPOINT", "https://r2.example/")
    monkeypatch.setattr(validator, "R2_BUCKET", "test-bucket")
    monkeypatch.setattr(validator, "HIPPIUS_ACCESS_KEY", "hipp-key" if hippius else "")
    monkeypatch.setattr(validator, "HIPPIUS_SECRET_KEY", "hipp-secret" if hippius else "")
    monkeypatch.setattr(validator, "HIPPIUS_BUCKET", "hipp-bucket")
    monkeypatch.setattr(validator, "DS_ACCESS_KEY", "ds-key" if dataset else "")
    monkeypatch.setattr(validator, "DS_SECRET_KEY", "ds-secret" if dataset else "")
    monkeypatch.setattr(validator, "DS_ENDPOINT", "https://ds.example/" if dataset else "")
    monkeypatch.setattr(validator, "DS_BUCKET", "ds-bucket")

    # Pre-create as many clients as the worst case calls for; the
    # unused ones never get returned by the factory iterator.
    clients = [MagicMock(), MagicMock(), MagicMock()]
    iter_clients = iter(clients)
    mocker.patch("validator.boto3.client",
                 side_effect=lambda *a, **kw: next(iter_clients))
    return clients


@pytest.fixture
def fake_boto(mocker, monkeypatch):
    """Mock boto3.client with no Hippius and no dataset client."""
    return _make_boto_fixture(mocker, monkeypatch)


@pytest.fixture
def fake_boto_with_hippius(mocker, monkeypatch):
    """Mock boto3.client with HIPPIUS_*_KEY set — second client is hippius."""
    return _make_boto_fixture(mocker, monkeypatch, hippius=True)


def _mock_body(payload: bytes) -> MagicMock:
    """Build a MagicMock that mimics the boto3 get_object response body."""
    body = MagicMock()
    body.read.return_value = payload
    return {"Body": body}


# ---------------------------------------------------------------------
# __init__ — wires boto3 clients per env-var presence.

def test_init_creates_only_r2_client_when_hippius_unset(fake_boto):
    r2 = validator.R2()
    assert r2.client is fake_boto[0]
    assert r2._hippius is None
    assert r2._ds_client is None


def test_init_creates_hippius_client_when_creds_set(fake_boto_with_hippius):
    r2 = validator.R2()
    assert r2.client is fake_boto_with_hippius[0]
    assert r2._hippius is fake_boto_with_hippius[1]


# ---------------------------------------------------------------------
# put / get — basic JSON round trip.

def test_put_calls_put_object_with_json_payload(fake_boto):
    r2 = validator.R2()
    r2.put("state/king.json", {"hotkey": "5h", "block": 100})
    fake_boto[0].put_object.assert_called_once()
    kwargs = fake_boto[0].put_object.call_args.kwargs
    assert kwargs["Bucket"] == "test-bucket"
    assert kwargs["Key"] == "state/king.json"
    assert kwargs["ContentType"] == "application/json"
    payload = json.loads(kwargs["Body"].decode())
    assert payload == {"hotkey": "5h", "block": 100}


def test_put_swallows_boto_errors(fake_boto):
    fake_boto[0].put_object.side_effect = Exception("503 Slow Down")
    r2 = validator.R2()
    # Defensive: must NOT raise into validator main loop.
    r2.put("state/x.json", {"a": 1})


def test_get_decodes_json_from_body(fake_boto):
    fake_boto[0].get_object.return_value = _mock_body(b'{"hotkey": "5h"}')
    r2 = validator.R2()
    assert r2.get("state/king.json") == {"hotkey": "5h"}


def test_get_returns_none_on_boto_error(fake_boto):
    fake_boto[0].get_object.side_effect = Exception("NoSuchKey")
    r2 = validator.R2()
    assert r2.get("state/missing.json") is None


# ---------------------------------------------------------------------
# put_raw / range_get — raw bytes paths.

def test_put_raw_calls_put_object_with_passthrough_body(fake_boto):
    r2 = validator.R2()
    r2.put_raw("dashboards/index.html", b"<html/>", "text/html")
    kwargs = fake_boto[0].put_object.call_args.kwargs
    assert kwargs["Body"] == b"<html/>"
    assert kwargs["ContentType"] == "text/html"


def test_put_raw_swallows_boto_errors(fake_boto):
    fake_boto[0].put_object.side_effect = Exception("oops")
    r2 = validator.R2()
    r2.put_raw("k", b"data", "text/plain")


def test_range_get_passes_range_header(fake_boto):
    fake_boto[0].get_object.return_value = _mock_body(b"PARTIAL")
    r2 = validator.R2()
    result = r2.range_get("shards/0.npy", 0, 1023)
    kwargs = fake_boto[0].get_object.call_args.kwargs
    assert kwargs["Range"] == "bytes=0-1023"
    assert result == b"PARTIAL"


# ---------------------------------------------------------------------
# append_jsonl / append_jsonl_batch — read+append+put pattern.

def test_append_jsonl_appends_to_existing(fake_boto):
    fake_boto[0].get_object.return_value = _mock_body(b'{"old": 1}\n')
    r2 = validator.R2()
    r2.append_jsonl("state/history.jsonl", {"new": 2})
    put_kwargs = fake_boto[0].put_object.call_args.kwargs
    body = put_kwargs["Body"].decode()
    # Existing line preserved + new line appended.
    assert body.startswith('{"old": 1}\n')
    assert '{"new": 2}' in body
    assert body.endswith("\n")


def test_append_jsonl_handles_missing_existing_key(fake_boto):
    fake_boto[0].get_object.side_effect = Exception("NoSuchKey")
    r2 = validator.R2()
    r2.append_jsonl("state/history.jsonl", {"first": 1})
    body = fake_boto[0].put_object.call_args.kwargs["Body"].decode()
    assert body == '{"first": 1}\n'


def test_append_jsonl_swallows_put_failure(fake_boto):
    fake_boto[0].get_object.side_effect = Exception("read-fail")
    fake_boto[0].put_object.side_effect = Exception("put-fail")
    r2 = validator.R2()
    # Both read AND write fail — must still not raise.
    r2.append_jsonl("k", {"x": 1})


def test_append_jsonl_batch_writes_all_records_in_order(fake_boto):
    fake_boto[0].get_object.return_value = _mock_body(b"")
    r2 = validator.R2()
    r2.append_jsonl_batch("key", [{"a": 1}, {"b": 2}, {"c": 3}])
    body = fake_boto[0].put_object.call_args.kwargs["Body"].decode()
    lines = [ln for ln in body.split("\n") if ln]
    assert len(lines) == 3
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"b": 2}
    assert json.loads(lines[2]) == {"c": 3}


# ---------------------------------------------------------------------
# Hippius dashboard fall-back path.

def test_put_dashboard_uses_hippius_when_available(fake_boto_with_hippius):
    r2 = validator.R2()
    r2.put_dashboard("dashboard.json", {"hello": "world"})
    fake_boto_with_hippius[1].put_object.assert_called_once()
    fake_boto_with_hippius[0].put_object.assert_not_called()


def test_put_dashboard_falls_back_to_r2_when_hippius_fails(fake_boto_with_hippius):
    fake_boto_with_hippius[1].put_object.side_effect = Exception("hippius down")
    r2 = validator.R2()
    r2.put_dashboard("dashboard.json", {"hello": "world"})
    # Both clients hit: hippius failed, then R2 fallback succeeded.
    fake_boto_with_hippius[1].put_object.assert_called_once()
    fake_boto_with_hippius[0].put_object.assert_called_once()


def test_put_dashboard_uses_r2_directly_when_hippius_unset(fake_boto):
    # No hippius client → straight to R2.
    r2 = validator.R2()
    r2.put_dashboard("dashboard.json", {"hello": "world"})
    fake_boto[0].put_object.assert_called_once()


def test_put_dashboard_raw_passes_cache_control(fake_boto):
    r2 = validator.R2()
    r2.put_dashboard_raw("index.html", b"<html/>", "text/html",
                         cache_control="no-cache")
    kwargs = fake_boto[0].put_object.call_args.kwargs
    assert kwargs["CacheControl"] == "no-cache"


def test_put_dashboard_raw_omits_cache_control_when_none(fake_boto):
    # No cache_control kwarg → CacheControl not in put_object call.
    r2 = validator.R2()
    r2.put_dashboard_raw("index.html", b"<html/>", "text/html")
    kwargs = fake_boto[0].put_object.call_args.kwargs
    assert "CacheControl" not in kwargs


def test_put_dashboard_swallows_full_failure(fake_boto_with_hippius):
    # Both Hippius AND R2 fail — must NOT raise.
    fake_boto_with_hippius[0].put_object.side_effect = Exception("r2 down")
    fake_boto_with_hippius[1].put_object.side_effect = Exception("hippius down")
    r2 = validator.R2()
    r2.put_dashboard("dashboard.json", {"x": 1})


# ---------------------------------------------------------------------
# _hippius_available + cooldown.

def test_hippius_available_returns_false_when_unset(fake_boto):
    r2 = validator.R2()
    assert r2._hippius_available() is False


def test_hippius_available_returns_true_when_set_and_no_cooldown(fake_boto_with_hippius):
    r2 = validator.R2()
    assert r2._hippius_available() is True


def test_hippius_marked_failure_blocks_subsequent_use(fake_boto_with_hippius, mocker):
    # First put fails → cooldown triggered. Second call should skip
    # hippius and go straight to R2 (no second hippius put_object).
    fake_boto_with_hippius[1].put_object.side_effect = Exception("hippius down")
    r2 = validator.R2()
    r2.put_dashboard("dashboard.json", {"x": 1})  # marks failure

    # Second call: hippius should NOT be retried within cooldown window.
    r2.put_dashboard("dashboard.json", {"y": 2})
    assert fake_boto_with_hippius[1].put_object.call_count == 1
    assert fake_boto_with_hippius[0].put_object.call_count == 2


# ---------------------------------------------------------------------
# ds_get — dataset store with R2 fallback.

def test_ds_get_falls_back_to_r2_when_no_ds_client(fake_boto):
    fake_boto[0].get_object.return_value = _mock_body(b'{"k": "v"}')
    r2 = validator.R2()
    assert r2._ds_client is None
    assert r2.ds_get("any-key") == {"k": "v"}


def test_ds_get_falls_back_to_r2_on_ds_client_error(fake_boto, mocker):
    # Stub a ds_client that raises, then verify R2 is consulted.
    fake_boto[0].get_object.return_value = _mock_body(b'{"r2": "fallback"}')
    r2 = validator.R2()
    r2._ds_client = MagicMock()
    r2._ds_bucket = "ds-bucket"
    r2._ds_client.get_object.side_effect = Exception("ds 503")
    assert r2.ds_get("key") == {"r2": "fallback"}
