"""Tests for chain_config.load_arch.

`load_arch()` resolves the active king's architecture module by name
(read from `chain.toml`'s `[arch].module` at chain_config import time).
The arch package's import side effect registers its config + model
classes with HuggingFace AutoConfig / AutoModelForCausalLM, so any
downstream `from_pretrained` resolves the king without
trust_remote_code. A missing `[arch].module` is treated as a hard
configuration error (not silently ignored).
"""
import sys

import pytest

import chain_config


def test_load_arch_returns_imported_module():
    # The chain.toml in the repo points at "archs.quasar"; load_arch
    # should hand back the actual archs.quasar module object.
    mod = chain_config.load_arch()
    assert mod.__name__ == chain_config.ARCH_MODULE
    assert mod is sys.modules[chain_config.ARCH_MODULE]


def test_load_arch_raises_when_arch_module_empty(monkeypatch):
    # An empty / missing [arch].module in chain.toml is a configuration
    # bug, not a fall-through-to-default — refuse to start.
    monkeypatch.setattr(chain_config, "ARCH_MODULE", "")
    with pytest.raises(RuntimeError, match="chain.toml"):
        chain_config.load_arch()


def test_load_arch_propagates_import_errors(monkeypatch):
    # If [arch].module names a non-existent package, the ImportError
    # surfaces directly to the caller rather than being swallowed.
    monkeypatch.setattr(chain_config, "ARCH_MODULE",
                        "archs.does_not_exist_xyz")
    with pytest.raises(ImportError):
        chain_config.load_arch()


def test_load_arch_uses_current_arch_module_value(monkeypatch):
    # Sanity: load_arch reads the module name at call time, not at
    # import time — so a runtime monkeypatch redirects the import.
    monkeypatch.setattr(chain_config, "ARCH_MODULE", "json")
    mod = chain_config.load_arch()
    assert mod.__name__ == "json"
