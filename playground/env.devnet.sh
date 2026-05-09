# Local devnet env vars.
#
# Source this before launching the validator / eval-server / miner against
# the local minio + (forthcoming) subtensor:
#
#     source playground/.env.devnet
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

# Validator/eval read these too. Keep them aligned with chain.toml — the
# fixture lives at tests/_fixtures/chain.nano_gpt.toml.
export TEUTONIC_CHAIN_OVERRIDE="tests/_fixtures/chain.nano_gpt.toml"
