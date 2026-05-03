def test_validator_exposes_main_sync():
    import validator

    assert hasattr(validator, "main_sync")


def test_eval_torch_imports():
    import eval_torch  # noqa: F401
