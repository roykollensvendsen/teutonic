"""GPT-2 arch shim — used as a tiny "nano-gpt" for local-devnet experiments.

Vanilla `GPT2LMHeadModel` ships in `transformers` and is already self-registered
with `AutoConfig` / `AutoModelForCausalLM`, so the import side-effect of this
package is enough to make `chain_config.load_arch()` resolve a chain.toml that
points here. No vendored modeling — this is a pedagogical / play arch, not a
custom architecture.

Default sizing (`size.py`):
    vocab=50257 (gpt2 BPE), n_embd=128, n_layer=4, n_head=4, n_positions=256
    => ~7M params (mostly the vocab*n_embd embedding matrix)

The arch is small enough to train on Shakespeare in seconds on CPU, which is
the whole point — fast iteration on validator/eval-server changes without
needing GPU access or a real Qwen3-MoE checkpoint. Activated via
chain.toml -> [arch].module = "archs.nano_gpt"; the live LXXX chain stays on
qwen3_moe.
"""
from transformers import GPT2Config, GPT2LMHeadModel  # noqa: F401

__all__ = ["GPT2Config", "GPT2LMHeadModel"]
