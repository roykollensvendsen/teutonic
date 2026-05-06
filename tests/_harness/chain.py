"""FakeChain — in-process subtensor + metagraph double for tests.

Replaces a real `bittensor.subtensor` for tests that drive the validator
main loop without RPC. Models the read-API surface the validator
actually consumes (see `validator.py` callsites of `subtensor.<x>`):

Read API (matches subtensor):
* `block` property — current block (int). If the chain was constructed
  with `block_raises=<exc>`, accessing `block` raises that exception
  instead — useful for testing RPC-failure error-handling paths
  (`validator._safe_block` and similar).
* `metagraph(netuid)` — returns a SimpleNamespace with attributes
  `hotkeys` (list[str] indexed by uid), `emission` (list[float] indexed
  by uid), and `coldkeys` (list[str | None] indexed by uid). The
  validator's `refresh_uid_map` reads exactly these three.
* `get_all_revealed_commitments(netuid)` — returns
  `{hotkey: [(block, data_str), ...]}` matching the dict shape
  `scan_reveals` consumes. Multiple reveals per hotkey are returned in
  insertion order (`scan_reveals` itself picks the max-block entry).
* `get_subnet_hyperparameters(netuid)` — stub returning a
  SimpleNamespace; populate fields per-test if a consumer needs them.

Mutator API (test setup):
* `register(hotkey, uid, *, coldkey=None, emission=0.0)` — make a
  hotkey appear at `uid`. Re-registering an existing uid REPLACES the
  hotkey at that slot (the chain-side deregister-and-re-register
  scenario). Gap uids are filled with the empty string so list indexing
  by uid stays valid.
* `advance_block(n=1)` — bump the current block by `n`.
* `commit_reveal(hotkey, payload, *, block=None)` — append a reveal for
  `hotkey`. `block` defaults to the current block.
* `set_emission(hotkey, em)` — update a registered hotkey's emission.
  Raises `KeyError` if the hotkey hasn't been registered, so a typo
  surfaces loudly instead of silently no-op'ing.

NOT modeled (deliberately — consult the relevant mock if you need it):
* `set_weights(...)` — write side. Tests that need to verify a weight
  push should mock `subtensor.set_weights` separately.
* `get_block_hash(block)` — not exposed at all; consumers that reach
  for it will get `AttributeError` rather than a misleading stub.
* Real Bittensor websocket reconnect / RPC-error semantics. The chain
  never raises spontaneously EXCEPT on `.block` access when
  `block_raises` was set. Tests that need RPC failures on other
  methods should monkeypatch the relevant method directly.
"""
from __future__ import annotations

from types import SimpleNamespace


class FakeChain:
    def __init__(
        self,
        *,
        block: int = 0,
        block_raises: BaseException | None = None,
    ):
        self._block = int(block)
        # When set, `.block` access raises this exception. Used to drive
        # validator error-handling paths like `_safe_block` that catch
        # any exception during block retrieval.
        self._block_raises = block_raises
        # Indexed by uid. Non-registered slots are "" so list-of-hotkeys
        # access by uid stays a valid-but-non-matching string rather
        # than raising IndexError or returning None.
        self._hotkeys: list[str] = []
        self._emissions: list[float] = []
        self._coldkeys: list[str | None] = []
        # {hotkey: [(block, payload), ...]} in insertion order.
        self._reveals: dict[str, list[tuple[int, str]]] = {}

    # ------------------------------------------------------------------
    # Read API — mirrors bittensor.subtensor.

    @property
    def block(self) -> int:
        if self._block_raises is not None:
            raise self._block_raises
        return self._block

    def metagraph(self, netuid: int) -> SimpleNamespace:  # noqa: ARG002
        return SimpleNamespace(
            hotkeys=list(self._hotkeys),
            emission=list(self._emissions),
            coldkeys=list(self._coldkeys),
        )

    def get_all_revealed_commitments(self, netuid: int) -> dict[str, list[tuple[int, str]]]:  # noqa: ARG002
        return {hk: list(entries) for hk, entries in self._reveals.items()}

    def get_subnet_hyperparameters(self, netuid: int) -> SimpleNamespace:  # noqa: ARG002
        return SimpleNamespace()

    # ------------------------------------------------------------------
    # Mutator API — test setup.

    def register(
        self,
        hotkey: str,
        uid: int,
        *,
        coldkey: str | None = None,
        emission: float = 0.0,
    ) -> None:
        # Pad lists to cover this uid, filling intermediate slots with
        # empty hotkeys / zero emissions / None coldkeys so by-uid
        # indexing stays valid for downstream consumers.
        while len(self._hotkeys) <= uid:
            self._hotkeys.append("")
            self._emissions.append(0.0)
            self._coldkeys.append(None)
        self._hotkeys[uid] = hotkey
        self._emissions[uid] = float(emission)
        self._coldkeys[uid] = coldkey

    def advance_block(self, n: int = 1) -> None:
        self._block += int(n)

    def commit_reveal(
        self,
        hotkey: str,
        payload: str,
        *,
        block: int | None = None,
    ) -> None:
        b = self._block if block is None else int(block)
        self._reveals.setdefault(hotkey, []).append((b, payload))

    def set_emission(self, hotkey: str, em: float) -> None:
        for uid, hk in enumerate(self._hotkeys):
            if hk == hotkey:
                self._emissions[uid] = float(em)
                return
        raise KeyError(f"hotkey {hotkey!r} not registered; call register() first")
