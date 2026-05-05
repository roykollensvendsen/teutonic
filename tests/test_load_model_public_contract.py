"""Tests for eval.torch_runner.load_model public contract.

`load_model(repo, device, label="model", force_download=False,
            revision=None, on_stage=None)` has no docstring; the
public surface tested here is what callers and Discord reports
already rely on:

* On success: returns a loaded model object.
* On consistent failure across attempts: raises
  `RuntimeError("could not load model with any attention "
  "implementation")` — the exact wording Kyle (kyle890015 Discord
  2026-05-03) reported seeing in eval-server output.
* `repo` and `revision` are forwarded to
  `transformers.AutoModelForCausalLM.from_pretrained`.

Internal details (which attention implementations are tried, in
which order, and how many) are intentionally NOT pinned — they are
not part of any documented contract and pinning them here would
prevent harmless internal refactors of the fallback chain.
"""
import pytest

from eval.torch_runner import load_model


@pytest.fixture
def fake_loader(mocker):
    """Patch `from_pretrained` and surface the resulting MagicMock.

    Returns the patched function so tests can configure its
    `side_effect` / `return_value` and inspect call arguments.
    """
    return mocker.patch(
        "eval.torch_runner.AutoModelForCausalLM.from_pretrained"
    )


def _capture_first_call_kwargs(mock):
    """The first call's keyword args, regardless of how many attempts."""
    assert mock.call_count >= 1
    return mock.call_args_list[0].kwargs


# ---------------------------------------------------------------------
# Public contract: success path returns a model.

def test_load_model_returns_model_when_from_pretrained_succeeds(
    fake_loader, mocker,
):
    expected = mocker.MagicMock(name="loaded_model")
    fake_loader.return_value = expected

    result = load_model("miner/repo", "cuda:0")

    assert result is expected


# ---------------------------------------------------------------------
# Public contract: consistent failure raises with Kyle's error string.

def test_load_model_raises_runtime_error_when_from_pretrained_always_fails(
    fake_loader,
):
    fake_loader.side_effect = RuntimeError("anything that's not the public string")

    with pytest.raises(
        RuntimeError,
        match="could not load model with any attention implementation",
    ):
        load_model("miner/repo", "cuda:0")


# ---------------------------------------------------------------------
# Public contract: repo + revision are forwarded to the HF loader.

def test_load_model_forwards_repo_to_from_pretrained(fake_loader, mocker):
    fake_loader.return_value = mocker.MagicMock(name="loaded_model")

    load_model("miner/specific-repo", "cuda:0")

    # `repo` may be passed positionally or as a kwarg — pin the contract
    # that it ends up at `from_pretrained` somehow, not its argument
    # binding style (which is a transformers-API impl detail).
    assert fake_loader.call_count >= 1
    call = fake_loader.call_args_list[0]
    repo_seen = (call.args[0] if call.args else call.kwargs.get("pretrained_model_name_or_path"))
    assert repo_seen == "miner/specific-repo"


def test_load_model_forwards_revision_to_from_pretrained(fake_loader, mocker):
    fake_loader.return_value = mocker.MagicMock(name="loaded_model")

    load_model("miner/repo", "cuda:0", revision="abc123")

    kwargs = _capture_first_call_kwargs(fake_loader)
    assert kwargs.get("revision") == "abc123"
