#!/usr/bin/env python3
"""Build a char-level HF-compatible tokenizer from tinyshakespeare.

The vocab is exactly the unique characters in Shakespeare's tinyshakespeare
plus four specials (pad/eos/bos/unk). That keeps embedding-row count tiny
(~70) — most of the budget for a Chinchilla-targeted nano-gpt goes to the
transformer blocks instead of the embedding matrix.

Output is a `PreTrainedTokenizerFast` saved with `save_pretrained()`, so
`AutoTokenizer.from_pretrained(repo_id)` resolves it the same way it
resolves gpt2.

Run:
    python -m playground.dataset.build_shakespeare        # creates the cache
    python -m playground.tokenizers.build_char_tokenizer  # builds + saves
    python -m playground.tokenizers.build_char_tokenizer --push  # + push to HF
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from transformers import PreTrainedTokenizerFast

from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build-char-tokenizer")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEXT = REPO_ROOT / "playground" / "dataset" / "cache" / "input.txt"
DEFAULT_OUT = Path("/tmp/teutonic-char-tokenizer")


def build(text_path: Path) -> PreTrainedTokenizerFast:
    text = text_path.read_text(encoding="utf-8")
    chars = sorted(set(text))
    log.info("source: %s (%d chars, %d unique)",
             text_path, len(text), len(chars))

    # Specials first so their IDs are stable across rebuilds (pad=0, eos=1, etc.)
    specials = ["<pad>", "<eos>", "<bos>", "<unk>"]
    vocab: dict[str, int] = {tok: i for i, tok in enumerate(specials)}
    for c in chars:
        vocab[c] = len(vocab)
    log.info("vocab size: %d (4 specials + %d chars)", len(vocab), len(chars))

    # WordLevel + Split-on-every-char pre-tokenizer = char-level tokenization.
    # The Regex(".") matches each single character; behavior="isolated" emits
    # one piece per match.
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex("."), behavior="isolated")
    # Decoder just concats pieces back to a string. WordPiece decoder with
    # empty prefix does the right thing here: no inserted spaces.
    tk.decoder = decoders.WordPiece(prefix="", cleanup=False)

    hf = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )

    # Round-trip sanity check on a small string.
    sample = "Hello, world!\nKING: To be or not to be."
    enc = hf.encode(sample)
    dec = hf.decode(enc)
    log.info("sanity: %r -> %d ids -> %r", sample, len(enc), dec)
    assert dec == sample, f"char tokenizer round-trip failed: {dec!r} != {sample!r}"
    return hf


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--text", default=str(DEFAULT_TEXT),
                   help="Path to the raw corpus. Default: tinyshakespeare "
                        "cache from build_shakespeare.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT),
                   help="Where to save the PreTrainedTokenizerFast.")
    p.add_argument("--push", action="store_true",
                   help="Upload to HF after saving.")
    p.add_argument("--repo", default="ai-garage/Teutonic-Nano-char-tokenizer",
                   help="HF repo id when --push is set.")
    args = p.parse_args()

    text_path = Path(args.text)
    if not text_path.exists():
        raise SystemExit(
            f"text not found at {text_path}; run "
            f"`python -m playground.dataset.build_shakespeare` first to cache it"
        )

    hf = build(text_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf.save_pretrained(out_dir)
    log.info("saved to %s", out_dir)

    if args.push:
        from huggingface_hub import HfApi
        if not os.environ.get("HF_TOKEN") and not (Path.home() / ".cache/huggingface/token").exists():
            raise SystemExit("HF auth missing — run `hf auth login` first")
        api = HfApi()
        api.create_repo(args.repo, exist_ok=True, private=False, repo_type="model")
        api.upload_folder(folder_path=str(out_dir), repo_id=args.repo)
        log.info("pushed: https://huggingface.co/%s", args.repo)

    return 0


if __name__ == "__main__":
    sys.exit(main())
