#!/usr/bin/env python3
"""Bootstrap the local subtensor with wallets, balances, a subnet, and
registered hotkeys for the nano-gpt devnet.

Idempotent: re-running skips work already done. Steps:

  1. Generate fresh wallets in `playground/wallets/`:
       validator     (cold + hot)
       miner_alpha   (cold + hot)
       miner_beta    (cold + hot)
  2. Fund each cold key from the chainspec's pre-funded //Alice account.
     (Alice starts with 1,000,000 TAO on a fresh localnet.)
  3. Have the validator wallet create a new subnet (burns 1,000 TAO).
  4. Register every wallet's hotkey on that subnet (burns ~0.09 TAO each).

Safety rails:

  * The wallet path is hardcoded to `playground/wallets/` and the script
    refuses to run if that resolves anywhere under ~/.bittensor or its
    parents — the user explicitly said "ikke mess med mine virkelige
    wallets" and we honor that with a path check, not a comment.
  * Network is hardcoded to TEUTONIC_NETWORK. The script refuses to run
    if that env var is unset or points at a non-localhost endpoint
    (finney/test/anything not starting with `ws://localhost` or
    `ws://127.0.0.1`).
  * //Alice wallet is recreated from the well-known dev URI on every run —
    her keys are public chainspec data, not a secret.

Run:
    docker compose -f playground/docker-compose.yml up -d   # chain up first
    source playground/env.devnet.sh
    python -m playground.chain.bootstrap_devnet
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import bittensor as bt
from bittensor.utils.balance import Balance

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("bootstrap-devnet")


PLAYGROUND_ROOT = Path(__file__).resolve().parents[1]
WALLET_PATH = PLAYGROUND_ROOT / "wallets"

# (cold_name, hotkey_name, fund_amount_tao). Validator gets enough to create
# a subnet (1 000 TAO) plus headroom; miners just need registration cost
# (~0.09 TAO) plus headroom for transfer fees.
WALLETS: list[tuple[str, str, int]] = [
    ("validator", "default", 5_000),
    ("miner_alpha", "default", 100),
    ("miner_beta", "default", 100),
]


def assert_safe_paths() -> None:
    """Refuse to run if the wallet path or network looks like prod."""
    home_bt = (Path.home() / ".bittensor").resolve()
    resolved = WALLET_PATH.resolve()
    # Equality check + parent-of-either check covers
    #   - wallets/ literally is ~/.bittensor/...
    #   - wallets/ contains ~/.bittensor/...
    #   - wallets/ is contained by ~/.bittensor/...
    if resolved == home_bt or home_bt in resolved.parents or resolved in home_bt.parents:
        raise SystemExit(
            f"REFUSING TO RUN: wallet path {resolved} overlaps with "
            f"{home_bt}. This script is devnet-only."
        )

    network = os.environ.get("TEUTONIC_NETWORK", "")
    safe_prefixes = ("ws://localhost", "ws://127.0.0.1", "ws://[::1]")
    if not any(network.startswith(p) for p in safe_prefixes):
        raise SystemExit(
            f"REFUSING TO RUN: TEUTONIC_NETWORK={network!r} doesn't look "
            f"like a localhost endpoint. Expected one of {safe_prefixes}. "
            f"Run `source playground/env.devnet.sh` first."
        )


def get_or_create_wallet(name: str, hotkey: str) -> bt.Wallet:
    """Generate a fresh wallet if not yet present; otherwise reuse it."""
    w = bt.Wallet(name=name, hotkey=hotkey, path=str(WALLET_PATH))
    cold_exists = w.coldkey_file.exists_on_device()
    hot_exists = w.hotkey_file.exists_on_device()
    if not cold_exists:
        log.info("creating new coldkey for %s", name)
        w.create_new_coldkey(n_words=12, use_password=False, overwrite=False)
    if not hot_exists:
        log.info("creating new hotkey for %s/%s", name, hotkey)
        w.create_new_hotkey(n_words=12, use_password=False, overwrite=False)
    return w


def get_alice_wallet() -> bt.Wallet:
    """Reconstruct //Alice from the well-known dev URI. Used for funding only.

    Living under the same `playground/wallets/` path so cleanup is one
    `rm -rf playground/wallets/` away. Alice's mnemonic is a publicly-known
    Substrate dev seed (`//Alice`); recreating it is not a credentials leak.
    """
    w = bt.Wallet(name="alice_funder", hotkey="default", path=str(WALLET_PATH))
    if not w.coldkey_file.exists_on_device():
        log.info("recreating //Alice coldkey for funding (well-known dev URI)")
        w.create_coldkey_from_uri("//Alice", use_password=False, overwrite=False)
    if not w.hotkey_file.exists_on_device():
        w.create_hotkey_from_uri("//Alice", use_password=False, overwrite=False)
    return w


def fund_if_low(sub: bt.Subtensor, alice: bt.Wallet, target: bt.Wallet,
                amount_tao: int) -> None:
    """Top up `target.coldkey` to at least `amount_tao` from Alice.

    Idempotent: skips the transfer if the balance already covers `amount_tao`.
    Catches the common re-run case (wallets exist, chain advanced, balances
    persist).
    """
    addr = target.coldkeypub.ss58_address
    bal = sub.get_balance(addr)
    if bal.tao >= amount_tao:
        log.info("balance %s = %s ≥ target %d τ — skipping transfer",
                 target.name, bal, amount_tao)
        return

    needed = amount_tao - int(bal.tao)
    log.info("transferring %d τ from //Alice to %s (%s)",
             needed, target.name, addr)
    resp = sub.transfer(
        wallet=alice,
        destination_ss58=addr,
        amount=Balance.from_tao(needed),
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    if not resp.success:
        raise SystemExit(
            f"transfer to {target.name} failed: {resp.error_message}"
        )
    log.info("  transfer OK; new balance: %s",
             sub.get_balance(addr))


def find_owned_subnet(sub: bt.Subtensor, validator_hotkey: str) -> int | None:
    """Return the netuid of any subnet whose owner-hotkey is ours, else None.

    Used to skip subnet creation on re-run. Walks all subnets and matches
    on owner_hotkey — there's no direct "subnets owned by X" RPC.
    """
    n = sub.get_total_subnets()
    for nuid in range(n):
        info = sub.subnet(nuid)
        if info is None:
            continue
        # SDK returns either DynamicInfo or a raw struct; try both shapes.
        owner = getattr(info, "owner_hotkey", None) or getattr(info, "owner", None)
        if owner == validator_hotkey:
            return nuid
    return None


def create_subnet_if_missing(sub: bt.Subtensor, validator: bt.Wallet) -> int:
    """Create a subnet owned by validator's hotkey; return the netuid.

    Re-runs are safe: if validator already owns a subnet, return its existing
    netuid without burning more TAO.
    """
    existing = find_owned_subnet(sub, validator.hotkey.ss58_address)
    if existing is not None:
        log.info("validator already owns subnet netuid=%d — skipping creation",
                 existing)
        return existing

    cost = sub.get_subnet_burn_cost() if hasattr(sub, "get_subnet_burn_cost") else None
    log.info("creating new subnet (burn cost: %s)", cost)
    n_before = sub.get_total_subnets()
    resp = sub.register_subnet(
        wallet=validator,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    if not resp.success:
        raise SystemExit(f"register_subnet failed: {resp.error_message}")

    n_after = sub.get_total_subnets()
    if n_after == n_before:
        raise SystemExit(
            f"register_subnet returned success but subnet count didn't grow "
            f"({n_before} -> {n_after}). State out of sync?"
        )
    netuid = n_after - 1
    log.info("  subnet created: netuid=%d", netuid)
    return netuid


def register_hotkey_if_missing(sub: bt.Subtensor, wallet: bt.Wallet,
                               netuid: int) -> None:
    """Burned-register `wallet.hotkey` on `netuid` if not already registered."""
    hk = wallet.hotkey.ss58_address
    if sub.is_hotkey_registered_on_subnet(hk, netuid):
        log.info("%s/%s already registered on netuid=%d — skipping",
                 wallet.name, wallet.hotkey_str, netuid)
        return
    log.info("burned-registering %s/%s on netuid=%d",
             wallet.name, wallet.hotkey_str, netuid)
    resp = sub.burned_register(
        wallet=wallet,
        netuid=netuid,
        wait_for_inclusion=True,
        wait_for_finalization=True,
    )
    if not resp.success:
        raise SystemExit(
            f"burned_register failed for {wallet.name}: {resp.error_message}"
        )
    uid = sub.get_uid_for_hotkey_on_subnet(hk, netuid)
    log.info("  registered: %s/%s -> uid=%d on netuid=%d",
             wallet.name, wallet.hotkey_str, uid, netuid)


def main() -> int:
    assert_safe_paths()

    network = os.environ["TEUTONIC_NETWORK"]
    log.info("connecting to %s", network)
    sub = bt.Subtensor(network=network)
    log.info("chain block: %d", sub.block)

    # 1. Wallets
    log.info("=== step 1/4: wallets ===")
    wallets = {
        name: get_or_create_wallet(name, hotkey)
        for name, hotkey, _ in WALLETS
    }
    alice = get_alice_wallet()

    # 2. Funding
    log.info("=== step 2/4: funding from //Alice ===")
    for name, _, amount in WALLETS:
        fund_if_low(sub, alice, wallets[name], amount)

    # 3. Subnet
    log.info("=== step 3/4: subnet ===")
    netuid = create_subnet_if_missing(sub, wallets["validator"])

    # 4. Hotkey registration
    log.info("=== step 4/4: hotkey registration ===")
    for name, _, _ in WALLETS:
        register_hotkey_if_missing(sub, wallets[name], netuid)

    log.info("=== done ===")
    log.info("netuid:    %d", netuid)
    log.info("validator: %s/%s  cold=%s  hot=%s",
             wallets["validator"].name,
             wallets["validator"].hotkey_str,
             wallets["validator"].coldkeypub.ss58_address,
             wallets["validator"].hotkey.ss58_address)
    for name in ("miner_alpha", "miner_beta"):
        w = wallets[name]
        log.info("%s: cold=%s hot=%s",
                 name, w.coldkeypub.ss58_address, w.hotkey.ss58_address)
    return 0


if __name__ == "__main__":
    sys.exit(main())
