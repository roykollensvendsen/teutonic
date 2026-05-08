"""Tests for validator's outbound HTTP notification helpers.

Three async helpers POST to external services (TaoMarketCap REST API
and Discord webhooks). All three:
* Are guarded by env-var checks (TMC_API_KEY / DISCORD_BOT_TOKEN +
  DISCORD_CHANNEL_ID) — return early when unset.
* Wrap the actual HTTP call in `try/except` so a failed POST never
  raises into the caller (validator main loop).
* Use `httpx.AsyncClient` with short timeouts.

These tests use a mocked `httpx.AsyncClient` so no real network IO
happens. The mock surface is shared across the three functions: a
single fixture builds an async-context-manager-supporting MagicMock
that records `.get` / `.post` calls.
"""
import pytest

import validator
from tests._chain import repo

# ---------------------------------------------------------------------
# Shared httpx mock: returns a client that supports `async with` and
# whose `.get` / `.post` methods are AsyncMocks the test configures.

@pytest.fixture
def fake_async_client(mocker):
    """Patch `validator.httpx.AsyncClient` to return a controllable mock.

    Returns the inner client (MagicMock) so the test can:
    - Set `.get.side_effect = [resp1, resp2, ...]` for fetch_tmc_data
    - Set `.post.return_value = resp` for notify_*
    - Inspect `.post.call_args` to assert request bodies
    """
    client = mocker.MagicMock()
    client.__aenter__ = mocker.AsyncMock(return_value=client)
    client.__aexit__ = mocker.AsyncMock(return_value=None)
    client.get = mocker.AsyncMock()
    client.post = mocker.AsyncMock()
    factory = mocker.MagicMock(return_value=client)
    mocker.patch("validator.httpx.AsyncClient", factory)
    return client


def _resp(json_payload, *, status_code: int = 200, text: str = ""):
    """Build a MagicMock-shaped httpx response."""
    import unittest.mock
    r = unittest.mock.MagicMock()
    r.status_code = status_code
    r.text = text
    r.json.return_value = json_payload
    return r


# ---------------------------------------------------------------------
# fetch_tmc_data — async fan-out to 3 TMC endpoints, gathered.
#
# Source-of-truth call order:
#   client.get("/market/market-data/")
#   client.get("/subnets/{NETUID}/")
#   client.get("/subnets/burn/{NETUID}/")
# asyncio.gather preserves the order of inputs in the result tuple.

async def test_fetch_tmc_data_returns_none_when_api_key_unset(monkeypatch, fake_async_client):
    monkeypatch.setattr(validator, "TMC_API_KEY", "")
    result = await validator.fetch_tmc_data()
    assert result is None
    fake_async_client.get.assert_not_called()


async def test_fetch_tmc_data_happy_path_returns_dict(monkeypatch, fake_async_client):
    monkeypatch.setattr(validator, "TMC_API_KEY", "fake-key")
    fake_async_client.get.side_effect = [
        # /market/market-data/
        _resp({"current_price": 5.0,
               "usd_quote": {"percent_change_24h": 2.5}}),
        # /subnets/{NETUID}/
        _resp({"latest_snapshot": {
            "alpha_sqrt_price": "0.5",
            "subnet_alpha_out_emission": 1_000_000_000,  # 1 alpha/block gross
            # 26bf392 introduced miner-share scaling: gross_apb is split into
            # server/validator/owner pots, and `sn3_alpha_per_block` exposes
            # only the miner share. With server=1e9 and others=0 the split
            # is 100% miners, so the exposed value matches the gross.
            "pending_server_emission": 1_000_000_000,
            "pending_validator_emission": 0,
            "pending_owner_cut": 0,
        }}),
        # /subnets/burn/{NETUID}/
        _resp([{"burn": 5_000_000_000}]),  # 5 tao
    ]
    result = await validator.fetch_tmc_data()
    assert result["tao_price_usd"] == 5.0
    assert result["tao_change_24h"] == 2.5
    assert result["sn3_alpha_price_tao"] == pytest.approx(0.25)  # 0.5**2
    assert result["sn3_alpha_price_usd"] == pytest.approx(1.25)  # 0.25 * 5.0
    assert result["sn3_reg_burn_tao"] == pytest.approx(5.0)  # 5e9 / 1e9
    assert result["sn3_alpha_per_block"] == pytest.approx(1.0)  # 1e9 * 1.0 share / 1e9
    assert result["sn3_miner_share"] == pytest.approx(1.0)
    assert result["sn3_alpha_per_block_gross"] == pytest.approx(1.0)


async def test_fetch_tmc_data_returns_none_on_http_error(monkeypatch, fake_async_client):
    monkeypatch.setattr(validator, "TMC_API_KEY", "fake-key")
    fake_async_client.get.side_effect = [
        Exception("connection reset"),
        _resp({}),
        _resp([]),
    ]
    result = await validator.fetch_tmc_data()
    # Function catches all exceptions and returns None.
    assert result is None


async def test_fetch_tmc_data_falls_back_to_zero_emission_on_missing_field(
    monkeypatch, fake_async_client,
):
    # The `subnet_alpha_out_emission` field is wrapped in its own
    # try/except; missing → 0.0, not None.
    monkeypatch.setattr(validator, "TMC_API_KEY", "fake-key")
    fake_async_client.get.side_effect = [
        _resp({"current_price": 1.0,
               "usd_quote": {"percent_change_24h": 0.0}}),
        _resp({"latest_snapshot": {"alpha_sqrt_price": "0.1"}}),
        _resp([{"burn": 0}]),
    ]
    result = await validator.fetch_tmc_data()
    assert result is not None
    assert result["sn3_alpha_per_block"] == 0.0


# ---------------------------------------------------------------------
# notify_new_king — Discord webhook for new-king crowning.

async def test_notify_new_king_returns_early_when_token_unset(monkeypatch, fake_async_client):
    monkeypatch.setattr(validator, "DISCORD_BOT_TOKEN", "")
    monkeypatch.setattr(validator, "DISCORD_CHANNEL_ID", "12345")
    await validator.notify_new_king({"hf_repo": "x", "hotkey": "h"})
    fake_async_client.post.assert_not_called()


async def test_notify_new_king_returns_early_when_channel_unset(monkeypatch, fake_async_client):
    monkeypatch.setattr(validator, "DISCORD_BOT_TOKEN", "tok")
    monkeypatch.setattr(validator, "DISCORD_CHANNEL_ID", "")
    await validator.notify_new_king({"hf_repo": "x", "hotkey": "h"})
    fake_async_client.post.assert_not_called()


async def test_notify_new_king_posts_embed_with_repo_and_hotkey(monkeypatch, fake_async_client):
    monkeypatch.setattr(validator, "DISCORD_BOT_TOKEN", "tok")
    monkeypatch.setattr(validator, "DISCORD_CHANNEL_ID", "12345")
    fake_async_client.post.return_value = _resp({}, status_code=200)
    king_repo = repo("alice", "king")
    await validator.notify_new_king({
        "hf_repo": king_repo,
        "hotkey": "5HhKL...",
        "reign_number": 7,
        "king_revision": "abcdef0123456789",
    })
    fake_async_client.post.assert_called_once()
    _args, kwargs = fake_async_client.post.call_args
    body = kwargs["json"]
    embed_text = body["embeds"][0]["description"]
    # The repo, hotkey prefix, reign, and revision-prefix must all
    # appear somewhere in the embed body (formatting may evolve).
    assert king_repo in embed_text
    assert "5HhKL" in embed_text
    assert "7" in embed_text
    assert "abcdef012345" in embed_text


async def test_notify_new_king_includes_eval_metrics_when_verdict_provided(
    monkeypatch, fake_async_client,
):
    monkeypatch.setattr(validator, "DISCORD_BOT_TOKEN", "tok")
    monkeypatch.setattr(validator, "DISCORD_CHANNEL_ID", "12345")
    fake_async_client.post.return_value = _resp({}, status_code=200)
    await validator.notify_new_king(
        {"hf_repo": repo("alice", "king"), "hotkey": "5h"},
        verdict={"mu_hat": 0.0123, "avg_king_loss": 2.5,
                 "avg_challenger_loss": 2.4, "wall_time_s": 312.0},
    )
    body = fake_async_client.post.call_args.kwargs["json"]
    embed_text = body["embeds"][0]["description"]
    # The verdict metrics surface in the embed (formatted).
    assert "0.012" in embed_text  # mu_hat formatted
    assert "2.5" in embed_text or "2.4" in embed_text


async def test_notify_new_king_includes_dethroned_repo_when_previous_king_present(
    monkeypatch, fake_async_client,
):
    monkeypatch.setattr(validator, "DISCORD_BOT_TOKEN", "tok")
    monkeypatch.setattr(validator, "DISCORD_CHANNEL_ID", "12345")
    fake_async_client.post.return_value = _resp({}, status_code=200)
    await validator.notify_new_king({
        "hf_repo": "alice/new-king",
        "hotkey": "5h",
        "previous_king": {"hf_repo": "bob/old-king"},
    })
    body = fake_async_client.post.call_args.kwargs["json"]
    embed_text = body["embeds"][0]["description"]
    assert "bob/old-king" in embed_text


async def test_notify_new_king_swallows_post_failure(monkeypatch, fake_async_client):
    monkeypatch.setattr(validator, "DISCORD_BOT_TOKEN", "tok")
    monkeypatch.setattr(validator, "DISCORD_CHANNEL_ID", "12345")
    fake_async_client.post.side_effect = Exception("network down")
    # Function MUST NOT raise; the validator main loop relies on this.
    await validator.notify_new_king({"hf_repo": "x", "hotkey": "h"})


async def test_notify_new_king_logs_but_does_not_raise_on_4xx(monkeypatch, fake_async_client):
    monkeypatch.setattr(validator, "DISCORD_BOT_TOKEN", "tok")
    monkeypatch.setattr(validator, "DISCORD_CHANNEL_ID", "12345")
    fake_async_client.post.return_value = _resp({}, status_code=403, text="forbidden")
    # 4xx is logged as warning but doesn't raise — same defensive contract.
    await validator.notify_new_king({"hf_repo": "x", "hotkey": "h"})


# ---------------------------------------------------------------------
# notify_king_dethroned_untrainable — Discord webhook on audit-fail.

async def test_notify_dethroned_returns_early_when_token_unset(monkeypatch, fake_async_client):
    monkeypatch.setattr(validator, "DISCORD_BOT_TOKEN", "")
    monkeypatch.setattr(validator, "DISCORD_CHANNEL_ID", "12345")
    await validator.notify_king_dethroned_untrainable(
        "dead/repo", {"hf_repo": "alive/repo"}, {"reason": "x"})
    fake_async_client.post.assert_not_called()


async def test_notify_dethroned_posts_embed_with_dead_repo_and_verdict(
    monkeypatch, fake_async_client,
):
    monkeypatch.setattr(validator, "DISCORD_BOT_TOKEN", "tok")
    monkeypatch.setattr(validator, "DISCORD_CHANNEL_ID", "12345")
    fake_async_client.post.return_value = _resp({}, status_code=200)
    old_repo = repo("alice", "old")
    new_repo = repo("bob", "rev")
    await validator.notify_king_dethroned_untrainable(
        old_repo,
        {"hf_repo": new_repo, "king_revision": "rev123abc"},
        {"reason": "norm_quant_high",
         "max_ratio": 1.5, "max_grad_norm": 999.0,
         "norm_quantization": 0.95},
    )
    body = fake_async_client.post.call_args.kwargs["json"]
    embed_text = body["embeds"][0]["description"]
    assert old_repo in embed_text
    assert new_repo in embed_text
    assert "norm_quant_high" in embed_text


async def test_notify_dethroned_swallows_post_failure(monkeypatch, fake_async_client):
    monkeypatch.setattr(validator, "DISCORD_BOT_TOKEN", "tok")
    monkeypatch.setattr(validator, "DISCORD_CHANNEL_ID", "12345")
    fake_async_client.post.side_effect = Exception("connection reset")
    # Defensive: validator audit path MUST NOT crash on Discord failure.
    await validator.notify_king_dethroned_untrainable(
        "dead/repo", {"hf_repo": "alive/repo"}, {"reason": "x"})
