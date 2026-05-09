"""Experiment registry for the local devnet.

Each subdirectory is one experiment with:
  config.toml        — declarative settings (arch, vocab, training, etc.)
  chain.toml         — the TEUTONIC_CHAIN_OVERRIDE-pointed chain spec
  snapshots/         — periodic dashboard.json saves for comparison
  README.md          — what we're testing (optional)

The CLI lives in `playground.experiments.tracker`. See its --help.
"""
