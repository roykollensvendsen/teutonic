"""Tests for the early-return gates in validator.process_challenge.

`process_challenge` (validator.py:1904) is the orchestrator that
processes one challenger. Its first ~50 lines are a series of
gates that early-return before any heavy work (HfApi, eval-server
POST, R2 verdict write). This file tests THOSE gates in isolation.

Gate order (validator.py:1912-1956):
1. King-hotkey skip — challenger.hotkey == state.king.hotkey
2. failed_repos skip — repo in state.failed_repos
3. evaluated_repos skip — repo in state.evaluated_repos
4. coldkey-prefix gate — repo must contain miner's coldkey prefix
   (or skipped if metagraph hasn't surfaced the coldkey yet)
5. Stale-king check — entry's king_hash doesn't match current king
   (only when check_stale=True)

Each gate either returns silently or records a failure + event.
Gates 1-3 are pure state checks; gate 4 calls
`state.expected_coldkey_prefix`; gate 5 reads `state.king.king_hash`.

We verify gates fire BEFORE the HfApi.model_info call by mocking
that call to raise `_NotReached` (a BaseException subclass that's
NOT caught by the broad `except Exception` inside process_challenge).
A test seeing _NotReached confirms control passed through all gates;
a test that completes silently confirms a gate fired.
"""
import pytest

import validator
from tests._chain import repo


class _NotReached(BaseException):
    """Sentinel exception that is NOT caught by `except Exception`.

    process_challenge wraps the HfApi call in a broad `try/except`,
    so AssertionError / RuntimeError would be swallowed and the
    function would record_failure(hf_metadata_error). A
    BaseException-derived sentinel propagates past the try/except
    so tests can pytest.raises it to assert "control reached HfApi".
    """


@pytest.fixture(autouse=True)
def _fail_fast_hf_api(mocker):
    """Mock HfApi so any code path that reaches it raises _NotReached.

    The early-return gates we test MUST return before this point. If
    a test sees _NotReached, the gate didn't fire (the function
    proceeded past the gates to the HfApi call).
    """
    api = mocker.MagicMock()
    api.model_info.side_effect = _NotReached(
        "process_challenge reached HfApi.model_info — gate should have "
        "returned earlier")
    mocker.patch("validator.HfApi", return_value=api)


@pytest.fixture
def state(r2_mock):
    return validator.State(r2_mock)


def _entry(*, hotkey="hk_chal", hf_repo=None,
           challenge_id="ch1", king_hash=""):
    if hf_repo is None:
        hf_repo = repo("bob", "chall")
    return {
        "challenge_id": challenge_id,
        "hotkey": hotkey,
        "hf_repo": hf_repo,
        "king_hash": king_hash,
        "block": 100,
    }


# ---------------------------------------------------------------------
# Gate 1: King-hotkey skip.

async def test_gate_skips_when_challenger_is_current_king(state):
    king_repo = repo("alice", "king")
    state.set_king("hk_king", king_repo, "kh", 100)
    entry = _entry(hotkey="hk_king", hf_repo=king_repo)
    # If gate 1 fires, returns silently (no recorded failure, no event).
    await validator.process_challenge(state, state.r2, entry,
                                       subtensor=None, wallet=None)
    # Sanity: nothing in failed_repos, no error in history.
    assert king_repo not in state.failed_repos
    error_records = [h for h in state.history if h.get("verdict") == "error"]
    assert error_records == []


async def test_gate_processes_normal_challenger_when_king_set(state):
    # Counter-example: a DIFFERENT hotkey reaches gate 4+ → eventually
    # hits the HfApi mock and _NotReached fires.
    state.set_king("hk_king", repo("alice", "king"), "kh", 100)
    state.hotkey_coldkey = {"hk_chal": "ck_chal"}  # so gate 4 doesn't skip
    entry = _entry(hotkey="hk_chal", hf_repo="bob/ck_chal-XXIV-chall")
    with pytest.raises(_NotReached):
        await validator.process_challenge(state, state.r2, entry,
                                           subtensor=None, wallet=None)


# ---------------------------------------------------------------------
# Gate 2: failed_repos skip.

async def test_gate_skips_when_repo_in_failed_repos(state):
    state.failed_repos.add(repo("bob", "chall"))
    entry = _entry()
    await validator.process_challenge(state, state.r2, entry,
                                       subtensor=None, wallet=None)
    # No new error recorded — the repo was already in failed_repos.
    assert len(state.history) == 0


# ---------------------------------------------------------------------
# Gate 3: evaluated_repos skip.

async def test_gate_skips_when_repo_in_evaluated_repos(state):
    chal_repo = repo("bob", "chall")
    state.evaluated_repos.add(chal_repo)
    entry = _entry()
    await validator.process_challenge(state, state.r2, entry,
                                       subtensor=None, wallet=None)
    # No-op skip — neither failed_repos nor history is touched.
    assert chal_repo not in state.failed_repos


# ---------------------------------------------------------------------
# Gate 4: coldkey-prefix gate.

async def test_gate_rejects_when_repo_missing_coldkey_prefix(state):
    # state knows hotkey's coldkey, repo doesn't contain prefix → reject.
    state.hotkey_coldkey = {"hk_chal": "5HhKLLong_coldkey_ss58_addr"}
    bad_repo = repo("bob", "no-prefix-here")
    entry = _entry(hf_repo=bad_repo)
    await validator.process_challenge(state, state.r2, entry,
                                       subtensor=None, wallet=None)
    # Repo lands in failed_repos with coldkey_required error code.
    assert bad_repo in state.failed_repos
    assert any(h.get("error_code") == "coldkey_required"
               for h in state.history)


async def test_gate_accepts_repo_with_coldkey_prefix_in_owner(state):
    # Coldkey prefix in HF owner part → accepted; reaches HfApi.
    state.hotkey_coldkey = {"hk_chal": "5HhKLLongkey"}
    # COLDKEY_PREFIX_LEN defaults to 8, so first 8 = "5HhKLLon".
    # Lowercased it's "5hhkllon" (NB: two l's). Repo owner contains it.
    entry = _entry(hf_repo=repo("5hhkllon-models"))
    with pytest.raises(_NotReached):
        await validator.process_challenge(state, state.r2, entry,
                                           subtensor=None, wallet=None)


async def test_gate_accepts_repo_with_coldkey_prefix_in_basename(state):
    state.hotkey_coldkey = {"hk_chal": "5HhKLLong_coldkey"}
    entry = _entry(hf_repo=repo("bob", "5hhkllon-v1"))
    with pytest.raises(_NotReached):
        await validator.process_challenge(state, state.r2, entry,
                                           subtensor=None, wallet=None)


async def test_gate_skips_coldkey_check_when_metagraph_stale(state):
    # state has no coldkey for this hotkey (fresh registration) →
    # skip check, fall through to HfApi.
    state.hotkey_coldkey = {}
    entry = _entry(hotkey="hk_unknown", hf_repo="bob/no-prefix-needed")
    with pytest.raises(_NotReached):
        await validator.process_challenge(state, state.r2, entry,
                                           subtensor=None, wallet=None)


# ---------------------------------------------------------------------
# Gate 5 (REMOVED by 26bf392): the stale-king-hash check.
#
# Pre-26bf392 behaviour: an entry whose king_hash didn't match the
# current king's was silently dropped (assumed to be a re-eval against
# a since-dethroned king).
#
# Post-26bf392 behaviour: gate 5 is gone. The bootstrap test compares
# to the *current* king regardless of which king the challenger
# originally targeted — a stale-king-hash entry that beats the new king
# is a valid coronation. The `check_stale` parameter is retained as a
# no-op for backward CLI compat.

async def test_stale_entry_is_processed_after_gate_5_removal(state):
    # 26bf392 removed the stale-king-hash gate; entries are no longer
    # skipped on king_hash mismatch. They now fall through to HfApi.
    state.set_king("hk_king", repo("alice", "king"),
                   "current_kh_long", 100)
    state.hotkey_coldkey = {"hk_chal": "ck_chal"}
    entry = _entry(hf_repo=repo("bob", "ck_chal-x"),
                   king_hash="stale_kh")
    with pytest.raises(_NotReached):
        await validator.process_challenge(state, state.r2, entry,
                                           subtensor=None, wallet=None,
                                           check_stale=True)


async def test_check_stale_false_still_falls_through_to_hf_api(state):
    # check_stale=False has always reached HfApi; preserved as
    # contract test for the no-op parameter.
    state.set_king("hk_king", "alice/king", "current_kh", 100)
    state.hotkey_coldkey = {"hk_chal": "ck_chal"}
    entry = _entry(hf_repo=repo("bob", "ck_chal-x"),
                   king_hash="stale_kh")
    with pytest.raises(_NotReached):
        await validator.process_challenge(state, state.r2, entry,
                                           subtensor=None, wallet=None,
                                           check_stale=False)


async def test_gate_does_not_skip_when_entry_king_hash_matches_prefix(state):
    # The check uses startswith (allowing prefix matches for hash
    # truncation in old payloads). Identical hashes pass through.
    state.set_king("hk_king", "alice/king", "kh_full_abc123", 100)
    state.hotkey_coldkey = {"hk_chal": "ck_chal"}
    entry = _entry(hf_repo=repo("bob", "ck_chal-x"),
                   king_hash="kh_full_abc")  # prefix match
    with pytest.raises(_NotReached):
        await validator.process_challenge(state, state.r2, entry,
                                           subtensor=None, wallet=None)


async def test_gate_does_not_skip_when_either_hash_empty(state):
    # If current king_hash OR entry king_hash is empty, the gate's
    # `if current_hash and entry_king_hash` short-circuits → no skip.
    state.set_king("hk_king", "alice/king", "current_kh", 100)
    state.hotkey_coldkey = {"hk_chal": "ck_chal"}
    entry = _entry(hf_repo=repo("bob", "ck_chal-x"), king_hash="")
    with pytest.raises(_NotReached):
        await validator.process_challenge(state, state.r2, entry,
                                           subtensor=None, wallet=None)


# ---------------------------------------------------------------------
# Gate ordering: king > failed > evaluated > coldkey > stale.

async def test_king_gate_fires_before_failed_repos_gate(state):
    # Even if repo IS in failed_repos, the king-hotkey gate fires
    # first (returns silently without recording anything).
    state.set_king("hk_king", "alice/king", "kh", 100)
    state.failed_repos.add("alice/king")
    entry = _entry(hotkey="hk_king", hf_repo="alice/king")
    await validator.process_challenge(state, state.r2, entry,
                                       subtensor=None, wallet=None)
    # No new error in history (failed_repos was already populated
    # before the call, but no NEW record_failure happened).
    assert state.history == []
