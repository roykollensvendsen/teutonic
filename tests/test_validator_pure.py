from validator import _rank_sort_key


def _entry(**fields):
    """Helper: build a minimal entry dict with the given fields set."""
    return fields


# _rank_sort_key — sort key returning lower-is-better tuples for ranking
# accepted challenger evaluations.

def test_rank_sort_key_higher_mu_hat_sorts_first():
    better = _entry(mu_hat=0.9)
    worse = _entry(mu_hat=0.5)
    assert _rank_sort_key(better) < _rank_sort_key(worse)


def test_rank_sort_key_higher_lcb_breaks_mu_hat_tie():
    better = _entry(mu_hat=0.5, lcb=0.4)
    worse = _entry(mu_hat=0.5, lcb=0.1)
    assert _rank_sort_key(better) < _rank_sort_key(worse)


def test_rank_sort_key_lower_loss_breaks_lcb_tie():
    better = _entry(mu_hat=0.5, lcb=0.4, avg_challenger_loss=2.0)
    worse = _entry(mu_hat=0.5, lcb=0.4, avg_challenger_loss=3.0)
    assert _rank_sort_key(better) < _rank_sort_key(worse)


def test_rank_sort_key_earlier_timestamp_breaks_loss_tie():
    better = _entry(mu_hat=0.5, lcb=0.4, avg_challenger_loss=2.0, timestamp="2026-01-01T00:00:00")
    worse = _entry(mu_hat=0.5, lcb=0.4, avg_challenger_loss=2.0, timestamp="2026-02-01T00:00:00")
    assert _rank_sort_key(better) < _rank_sort_key(worse)


def test_rank_sort_key_missing_fields_sort_last():
    populated = _entry(mu_hat=0.5, lcb=0.4, avg_challenger_loss=2.0, timestamp="2026-01-01T00:00:00")
    empty = _entry()
    assert _rank_sort_key(populated) < _rank_sort_key(empty)


def test_rank_sort_key_none_fields_sort_last():
    populated = _entry(mu_hat=0.5, lcb=0.4, avg_challenger_loss=2.0, timestamp="2026-01-01T00:00:00")
    nones = _entry(mu_hat=None, lcb=None, avg_challenger_loss=None, timestamp=None)
    assert _rank_sort_key(populated) < _rank_sort_key(nones)


def test_rank_sort_key_used_with_sorted_returns_best_first():
    entries = [
        _entry(mu_hat=0.3),
        _entry(mu_hat=0.9),
        _entry(mu_hat=0.6),
    ]
    ranked = sorted(entries, key=_rank_sort_key)
    assert [e["mu_hat"] for e in ranked] == [0.9, 0.6, 0.3]
