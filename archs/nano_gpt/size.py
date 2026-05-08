#!/usr/bin/env python3
"""GPT-2 (nano-gpt) config sizer.

Mirrors `archs/qwen3_moe/size.py`. Builds GPT2Config + GPT2LMHeadModel on the
meta device and reports total params. No experts/MoE bookkeeping — this is a
plain dense model, so total == active.

Usage:
    python -m archs.nano_gpt.size --n-embd 128 --n-layer 4 --n-head 4
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from accelerate import init_empty_weights

from archs.nano_gpt import GPT2Config, GPT2LMHeadModel


def build_config(args) -> GPT2Config:
    return GPT2Config(
        vocab_size=args.vocab_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_positions=args.n_positions,
        n_inner=args.n_inner,
        activation_function=args.activation_function,
        bos_token_id=args.bos_token_id,
        eos_token_id=args.eos_token_id,
        pad_token_id=args.pad_token_id,
        tie_word_embeddings=args.tie_word_embeddings,
    )


def _classify(name: str) -> str:
    n = name.lower()
    if "wte" in n:
        return "embed"
    if "wpe" in n:
        return "pos_embed"
    if "lm_head" in n:
        return "lm_head"
    if any(k in n for k in ("c_attn", "attn.c_proj", "attn.q_attn")):
        return "attn"
    if "mlp" in n:
        return "mlp"
    if "ln" in n or "norm" in n:
        return "norm"
    return "other"


def count_params(model, cfg: GPT2Config):
    """Walk the meta model. Dense, so total == active.

    Tied embeddings: HF GPT2 ties `lm_head.weight` to `transformer.wte.weight`
    by default. With tying enabled, HF's named_parameters() yields wte ONCE
    and skips lm_head — so we don't need extra dedup.
    """
    total = 0
    by_class: dict[str, int] = {}
    for name, p in model.named_parameters():
        n = p.numel()
        bucket = _classify(name)
        by_class[bucket] = by_class.get(bucket, 0) + n
        total += n
    return total, by_class


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-positions", type=int, default=256)
    p.add_argument("--n-inner", type=int, default=None,
                   help="FF inner dim. None => 4*n_embd (HF default).")
    p.add_argument("--activation-function", default="gelu_new")
    p.add_argument("--bos-token-id", type=int, default=50256)
    p.add_argument("--eos-token-id", type=int, default=50256)
    p.add_argument("--pad-token-id", type=int, default=50256)
    p.add_argument("--tie-word-embeddings", action="store_true", default=True)
    p.add_argument("--no-tie", dest="tie_word_embeddings", action="store_false")
    args = p.parse_args()

    cfg = build_config(args)

    print("config:")
    for k in ("vocab_size", "n_embd", "n_layer", "n_head", "n_positions",
              "n_inner", "activation_function", "tie_word_embeddings"):
        print(f"  {k} = {getattr(cfg, k, None)}")

    with init_empty_weights():
        model = GPT2LMHeadModel(cfg)

    total, by_class = count_params(model, cfg)

    print("\nparams (total = active for dense):")
    for bucket in sorted(by_class):
        n = by_class[bucket]
        print(f"  {bucket:12s}  {n/1e6:8.3f}M")
    print(f"  {'-'*12}  -----------------")
    print(f"  {'TOTAL':12s}  {total/1e6:8.3f}M")

    bf16_mib = total * 2 / (1024 ** 2)
    print(f"\nbf16 weight size: {bf16_mib:.1f} MiB")


if __name__ == "__main__":
    main()
