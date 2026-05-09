# Local devnet env vars.
#
# Source this before launching the validator / eval-server / miner against
# the local minio + subtensor stack:
#
#     source playground/env.devnet.sh
#     python validator.py ...
#
# The TEUTONIC_R2_* names match what eval/torch_runner.R2 reads at
# import time — no code changes; just point the same env names at minio
# instead of Cloudflare. Dataset client (TEUTONIC_DS_*) is intentionally
# unset so R2.__init__ falls through to the primary client + bucket;
# splitting only matters in production where the dataset bucket lives in
# a different storage backend (Hippius vs Cloudflare).

export TEUTONIC_R2_ENDPOINT="http://localhost:9100"
export TEUTONIC_R2_ACCESS_KEY="minioadmin"
export TEUTONIC_R2_SECRET_KEY="minioadmin"
export TEUTONIC_R2_BUCKET="playground"

# Subtensor RPC. validator.py reads TEUTONIC_NETWORK and passes it to
# `bittensor.subtensor(network=...)`; the SDK accepts a full ws:// URL
# here as well as the named "finney"/"test"/"local" presets. Pointed at
# subtensor-one — the deterministic-peer-ID node in docker-compose.yml.
export TEUTONIC_NETWORK="ws://localhost:9944"

# Validator/eval read these too. Keep them aligned with chain.toml — the
# fixture lives at tests/_fixtures/chain.nano_gpt.toml.
export TEUTONIC_CHAIN_OVERRIDE="tests/_fixtures/chain.nano_gpt.toml"
