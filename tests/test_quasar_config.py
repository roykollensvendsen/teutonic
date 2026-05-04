"""Tests for QuasarConfig._build_hybrid_layer_types.

Determines the per-layer architecture topology (quasar vs gla) for a
hybrid Quasar model. Pure deterministic function over four config
fields: n_layers, quasar_layers, gated_layers, use_gla_first.

A regression here silently produces wrong model architectures — the
caller (modeling_quasar) builds layers by reading this list, so an
off-by-one or flipped flag would mis-place attention vs gated blocks.
"""
from quasar.configuration_quasar import QuasarConfig


def _config(*, n_layers, quasar_layers, gated_layers, use_gla_first):
    """Build a minimal QuasarConfig with the four fields that matter."""
    return QuasarConfig(
        n_layers=n_layers,
        quasar_layers=quasar_layers,
        gated_layers=gated_layers,
        use_gla_first=use_gla_first,
    )


# ---------------------------------------------------------------------
# Length invariant.

def test_length_matches_n_layers():
    cfg = _config(n_layers=12, quasar_layers=2, gated_layers=2,
                  use_gla_first=False)
    assert len(cfg._build_hybrid_layer_types()) == 12


def test_partial_cycle_when_n_layers_smaller_than_cycle():
    # n_layers < (quasar_layers + gated_layers) — partial cycle.
    cfg = _config(n_layers=3, quasar_layers=4, gated_layers=2,
                  use_gla_first=False)
    types = cfg._build_hybrid_layer_types()
    # First 3 of cycle [quasar, quasar, quasar, quasar, gla, gla].
    assert types == ["quasar", "quasar", "quasar"]


def test_incomplete_trailing_cycle():
    # n_layers=5, cycle_len=3 (quasar_layers=2 + gated_layers=1) —
    # one full cycle + 2 trailing positions. The trailing piece must
    # follow the cycle, not get truncated to all-quasar or restart.
    cfg = _config(n_layers=5, quasar_layers=2, gated_layers=1,
                  use_gla_first=False)
    types = cfg._build_hybrid_layer_types()
    # cycle = [quasar, quasar, gla] → 5 positions: cycle + first 2 of next
    assert types == ["quasar", "quasar", "gla", "quasar", "quasar"]


# ---------------------------------------------------------------------
# Element domain.

def test_only_quasar_and_gla_strings():
    cfg = _config(n_layers=16, quasar_layers=3, gated_layers=1,
                  use_gla_first=True)
    types = cfg._build_hybrid_layer_types()
    assert set(types).issubset({"quasar", "gla"})


# ---------------------------------------------------------------------
# Cycle pattern — use_gla_first=False puts quasar first.

def test_quasar_first_when_use_gla_first_false():
    cfg = _config(n_layers=8, quasar_layers=2, gated_layers=2,
                  use_gla_first=False)
    types = cfg._build_hybrid_layer_types()
    # cycle = [quasar, quasar, gla, gla], repeated twice
    assert types == ["quasar", "quasar", "gla", "gla",
                     "quasar", "quasar", "gla", "gla"]


def test_gla_first_when_use_gla_first_true():
    cfg = _config(n_layers=8, quasar_layers=2, gated_layers=2,
                  use_gla_first=True)
    types = cfg._build_hybrid_layer_types()
    # cycle = [gla, gla, quasar, quasar], repeated twice
    assert types == ["gla", "gla", "quasar", "quasar",
                     "gla", "gla", "quasar", "quasar"]


# ---------------------------------------------------------------------
# Counts in a full cycle.

def test_counts_per_cycle_match_config():
    # Over one full cycle, count(quasar) == quasar_layers and
    # count(gla) == gated_layers.
    cfg = _config(n_layers=6, quasar_layers=4, gated_layers=2,
                  use_gla_first=False)
    types = cfg._build_hybrid_layer_types()
    assert types.count("quasar") == 4
    assert types.count("gla") == 2


def test_counts_per_cycle_match_config_use_gla_first():
    cfg = _config(n_layers=6, quasar_layers=4, gated_layers=2,
                  use_gla_first=True)
    types = cfg._build_hybrid_layer_types()
    # Same totals, just rearranged.
    assert types.count("quasar") == 4
    assert types.count("gla") == 2


# ---------------------------------------------------------------------
# Default config — sanity check that it produces a non-degenerate result.

def test_default_config_produces_non_degenerate_pattern():
    cfg = QuasarConfig()
    types = cfg._build_hybrid_layer_types()
    assert len(types) == cfg.n_layers
    # Both layer kinds should appear in a default run.
    assert "quasar" in types
    assert "gla" in types
