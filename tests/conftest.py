"""Shared pytest fixtures for the teutonic test suite.

Pytest auto-discovers fixtures defined here for every test under
`tests/`. Test files do NOT need to import these — just declare the
fixture name as a parameter.

What lives here vs. what stays per-file:

* **Cross-cutting fixtures** (used by 3+ test files with the same
  contract) belong in this conftest. The `r2_mock` mock of
  `validator.R2` is the primary example — it was previously copied
  with subtle drift in 12 test files.
* **Test-specific fixtures** (e.g. `_MicroLM` for trainability-probe
  tests, `fake_hf` for HF-dependent tests, `_isolate_evals_dict`
  for eval-server prune tests) stay local to their file. They model
  one corner of the system that other tests don't exercise.

Adding a new shared fixture here:
1. Confirm it's actually used in 3+ files with consistent contract
2. Document the spy surface in the docstring (what attrs are
   exposed for assertions)
3. Default to function-scope so each test gets a fresh instance
"""
import pytest

import chain_config


def pytest_collection_modifyitems(config, items):
    """Auto-skip `arch_specific(<module>)` tests when the active arch differs.

    Test files in `tests/test_archs_<arch>_*.py` apply
    `pytestmark = pytest.mark.arch_specific("archs.<arch>")`. When
    `chain_config.ARCH_MODULE` points at a different package, those tests
    are skipped instead of failing on import / construction errors. This
    keeps the suite green across chain-cutovers (XXIV/Quasar ↔ LXXX/Qwen3
    ↔ ...) without requiring per-cutover test edits.
    """
    active_arch = chain_config.ARCH_MODULE
    skip_reasons: dict[str, pytest.MarkDecorator] = {}
    for item in items:
        for marker in item.iter_markers("arch_specific"):
            required = marker.args[0] if marker.args else None
            if required and required != active_arch:
                key = required
                if key not in skip_reasons:
                    skip_reasons[key] = pytest.mark.skip(
                        reason=f"requires chain_config.ARCH_MODULE={key!r}, "
                               f"active is {active_arch!r}")
                item.add_marker(skip_reasons[key])


@pytest.fixture
def r2_mock(mocker):
    """Dict-backed in-memory mock of `validator.R2`.

    Exposes the full method surface tests rely on:
    * `r2.get(key)` — returns last `r2.put`'d value for `key`, or None.
    * `r2.put(key, data)` — stores `data` under `key`.
    * `r2.append_jsonl(key, record)` — appends `record` to a per-key
      list (tests inspect via `r2._appended[key]`).
    * `r2.put_dashboard(key, data)` — stores `data` under `key` in the
      dashboard slot (tests inspect via `r2._dashboards[key]`).

    Spy attributes for assertion:
    * `r2._storage` — dict of all `put` keys → last value.
    * `r2._appended` — dict of `append_jsonl` keys → list of records.
    * `r2._dashboards` — dict of `put_dashboard` keys → last value.

    Function-scoped (pytest default) so each test gets a fresh mock.
    Tests that only need a subset of the spy surface simply don't
    read the unused attributes.
    """
    storage: dict = {}
    appended: dict = {}
    dashboards: dict = {}

    r2 = mocker.MagicMock()
    r2.get.side_effect = lambda key: storage.get(key)
    r2.put.side_effect = lambda key, data: storage.update({key: data})
    r2.append_jsonl.side_effect = (
        lambda key, record: appended.setdefault(key, []).append(record)
    )
    r2.put_dashboard.side_effect = (
        lambda key, data: dashboards.update({key: data})
    )
    r2._storage = storage
    r2._appended = appended
    r2._dashboards = dashboards
    return r2
