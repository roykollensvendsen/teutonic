def test_validator_exposes_main_sync():
    import validator

    assert hasattr(validator, "main_sync")


def test_eval_torch_runner_imports():
    from eval import torch_runner  # noqa: F401
