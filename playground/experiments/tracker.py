#!/usr/bin/env python3
"""Experiment tracker for the local devnet.

Subcommands:

  list                 List all experiments under playground/experiments/
  activate NAME        Print the env-var exports needed to run NAME.
                       Pipe to `eval $(...)` or `source <(...)` in a shell.
  snapshot [--name N]  Save the current dashboard.json under
                       experiments/<active>/snapshots/<timestamp>.json.
                       --name overrides; defaults to the active experiment
                       indicated by TEUTONIC_EXPERIMENT.
  compare A B [...]    Render an ASCII loss-curve overlay for the named
                       experiments, reading their accumulated snapshots.
  status               Show which experiment is currently active + counts.

Each experiment is a directory with config.toml, chain.toml, snapshots/.
A current "active" experiment is determined by the TEUTONIC_EXPERIMENT env
var; activate prints the env block that sets it (plus the chain override).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
DASHBOARD_URL = os.environ.get(
    "TEUTONIC_DASHBOARD_URL", "http://localhost:9300/dashboard.json"
)


def list_experiments() -> list[Path]:
    """Return experiment dirs (directly under experiments/, with config.toml)."""
    return sorted(p for p in ROOT.iterdir()
                  if p.is_dir() and (p / "config.toml").exists())


def load_config(exp_dir: Path) -> dict:
    """Read the experiment's config.toml. Returns the parsed dict."""
    with open(exp_dir / "config.toml", "rb") as f:
        return tomllib.load(f)


def cmd_list() -> int:
    exps = list_experiments()
    if not exps:
        print(f"no experiments under {ROOT} (each must have a config.toml)")
        return 1
    active = os.environ.get("TEUTONIC_EXPERIMENT", "")
    print(f"{'name':<24} {'desc':<40} status")
    for e in exps:
        cfg = load_config(e)
        desc = cfg.get("experiment", {}).get("description", "")[:40]
        marker = "  active" if e.name == active else ""
        snaps = len(list((e / "snapshots").glob("*.json"))) \
            if (e / "snapshots").exists() else 0
        print(f"{e.name:<24} {desc:<40} snapshots={snaps}{marker}")
    return 0


def cmd_activate(name: str) -> int:
    exp = ROOT / name
    if not exp.exists():
        print(f"no experiment {name!r} under {ROOT}", file=sys.stderr)
        return 1
    chain = exp / "chain.toml"
    if not chain.exists():
        print(f"experiment {name!r} has no chain.toml", file=sys.stderr)
        return 1
    cfg = load_config(exp)
    print(f"# activate {name}")
    print(f"# {cfg.get('experiment', {}).get('description', '')}")
    print(f"export TEUTONIC_EXPERIMENT={name!r}")
    print(f"export TEUTONIC_CHAIN_OVERRIDE={str(chain)!r}")
    # Optional per-experiment training overrides.
    for k, v in cfg.get("env", {}).items():
        print(f"export {k}={v!r}")
    print("# next:  source playground/env.devnet.sh   # base devnet env")
    print(f"#        source <(python -m playground.experiments.tracker activate {name})")
    print(f"#        python -m playground.launch_validator   # tied to {name}'s chain")
    print("#")
    print("# Use `source <(...)` not `eval $(...)`  — the latter loses newlines")
    print("# without explicit quotes and merges all exports into one line.")
    return 0


def cmd_status() -> int:
    active = os.environ.get("TEUTONIC_EXPERIMENT", "(none)")
    chain_override = os.environ.get("TEUTONIC_CHAIN_OVERRIDE", "(none)")
    print(f"active experiment: {active}")
    print(f"chain override:    {chain_override}")
    if active and active != "(none)":
        exp = ROOT / active
        if exp.exists():
            cfg = load_config(exp)
            print(f"description:       {cfg.get('experiment', {}).get('description', '')}")
            snaps = sorted((exp / "snapshots").glob("*.json")) \
                if (exp / "snapshots").exists() else []
            print(f"snapshots:         {len(snaps)}")
            if snaps:
                print(f"latest snapshot:   {snaps[-1].name}")
    return 0


def cmd_snapshot(name: str | None) -> int:
    if not name:
        name = os.environ.get("TEUTONIC_EXPERIMENT")
    if not name:
        print("no experiment specified; pass --name or set TEUTONIC_EXPERIMENT",
              file=sys.stderr)
        return 1
    exp = ROOT / name
    if not exp.exists():
        print(f"no experiment {name!r} under {ROOT}", file=sys.stderr)
        return 1
    try:
        with urlopen(DASHBOARD_URL, timeout=5) as r:
            body = r.read()
    except Exception as e:
        print(f"could not fetch {DASHBOARD_URL}: {e}", file=sys.stderr)
        return 2
    snap_dir = exp / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = snap_dir / f"{ts}.json"
    out.write_bytes(body)
    d = json.loads(body)
    print(f"saved {out}")
    print(f"  history len: {len(d.get('history', []))}, "
          f"reign: {d.get('king', {}).get('reign_number')}")
    return 0


def cmd_compare(names: list[str]) -> int:
    """ASCII overlay of king-loss curves across experiments."""
    series: dict[str, list[float]] = {}
    for name in names:
        exp = ROOT / name
        if not exp.exists():
            print(f"WARN: no experiment {name!r}; skipping", file=sys.stderr)
            continue
        # Use the latest snapshot's full history as the loss series.
        snaps = sorted((exp / "snapshots").glob("*.json")) \
            if (exp / "snapshots").exists() else []
        if not snaps:
            print(f"WARN: {name} has no snapshots; skipping", file=sys.stderr)
            continue
        latest = json.loads(snaps[-1].read_bytes())
        # dashboard.json stores history newest-first; reverse for chronological
        # left-to-right reading.
        history = list(reversed(latest.get("history", [])))
        losses = [h.get("avg_king_loss", 0.0) for h in history
                  if h.get("avg_king_loss", 0)]
        if losses:
            series[name] = losses
    if not series:
        print("no series to plot", file=sys.stderr)
        return 1

    # Find global min/max for scaling.
    all_vals = [v for s in series.values() for v in s]
    lo, hi = min(all_vals), max(all_vals)
    span = max(hi - lo, 1e-9)
    width = 60
    print(f"king-loss across {len(series)} experiments  "
          f"(min={lo:.3f}, max={hi:.3f})")
    for name, losses in series.items():
        print(f"\n  {name}  (n={len(losses)})")
        for i, v in enumerate(losses):
            n_blocks = int((v - lo) / span * width)
            bar = "█" * n_blocks
            print(f"    {i:3d}  {v:7.4f}  {bar}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Experiment tracker for devnet")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list all experiments")
    sub.add_parser("status", help="show active experiment")

    a = sub.add_parser("activate", help="print env exports for NAME")
    a.add_argument("name")

    s = sub.add_parser("snapshot", help="save current dashboard.json")
    s.add_argument("--name", default=None,
                   help="override TEUTONIC_EXPERIMENT")

    c = sub.add_parser("compare", help="ASCII overlay loss curves")
    c.add_argument("names", nargs="+")

    args = p.parse_args()
    if args.cmd == "list":
        return cmd_list()
    if args.cmd == "status":
        return cmd_status()
    if args.cmd == "activate":
        return cmd_activate(args.name)
    if args.cmd == "snapshot":
        return cmd_snapshot(args.name)
    if args.cmd == "compare":
        return cmd_compare(args.names)
    return 1


if __name__ == "__main__":
    sys.exit(main())
