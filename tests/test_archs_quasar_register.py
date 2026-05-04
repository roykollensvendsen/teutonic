"""Tests for archs.quasar._register.

Registers QuasarConfig + QuasarForCausalLM with HuggingFace's Auto*
APIs at module import time so downstream `from_pretrained` calls
resolve the king without trust_remote_code. The function is
idempotent — re-importing the package (in tests, eval workers,
training scripts) must not raise even though AutoConfig.register
itself raises ValueError on duplicate registration.
"""
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

import archs.quasar
from archs.quasar import QuasarConfig, QuasarForCausalLM, QuasarModel


def test_register_is_idempotent_on_second_call():
    # First call already happened at module import. A second invocation
    # must not raise — the function swallows the duplicate-ValueError
    # from each Auto*.register internally.
    archs.quasar._register()  # should not raise
    archs.quasar._register()  # third invocation, still safe


def test_register_attaches_quasar_to_auto_apis():
    # Sanity check: the side effect of import + _register is that the
    # Auto* registries resolve Quasar classes by config type.
    cfg_type = QuasarConfig
    assert AutoModel._model_mapping[cfg_type] is QuasarModel
    assert AutoModelForCausalLM._model_mapping[cfg_type] is QuasarForCausalLM
    # AutoConfig is registered by model_type string, not config class.
    assert AutoConfig.for_model("quasar").__class__ is QuasarConfig
