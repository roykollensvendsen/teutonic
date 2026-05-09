"""Custom tokenizers for devnet experiments.

Each script here builds a HF-compatible tokenizer (saves to disk in
`PreTrainedTokenizerFast` format) and optionally pushes it to HF so
`AutoTokenizer.from_pretrained(repo_id)` can resolve it during seed-king
push and miner training.
"""
