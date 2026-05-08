"""Chain-name helpers — keep tests arch-agnostic.

`validator.REPO_PATTERN` is auto-derived from `chain.toml [chain].name`;
tests that enqueue reveals or set kings need repo strings matching that
pattern. Reading the active chain name from `chain_config` means a
future cutover (e.g. XXIV → LXXX → ...) costs zero test edits.

This module sits next to `tests/_harness/` and is imported via
`from tests._chain import repo` (same pattern as the harness package).
"""
from __future__ import annotations

import chain_config

CHAIN_NAME: str = chain_config.NAME


def repo(owner: str, suffix: str = "x") -> str:
    """Build a chain-name-aware repo string for tests.

    >>> from tests._chain import repo
    >>> repo("alice", "king")           # doctest: +SKIP
    'alice/Teutonic-LXXX-king'

    The suffix defaults to 'x' so most callsites can write `repo("alice")`
    when the suffix is irrelevant to what's being tested. Use an explicit
    suffix when the test asserts on it (e.g. `repo("bob", "king")` for the
    seed-king repo).
    """
    return f"{owner}/{CHAIN_NAME}-{suffix}"
