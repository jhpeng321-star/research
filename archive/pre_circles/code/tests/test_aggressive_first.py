"""Tests for aggressive-first CAV selection (autopsy-driven, 2026-08-06)."""

from env.vehicles import CAVSelector


def _sel(pen=0.2, seed=1):
    return CAVSelector(penetration=pen, seed=seed)


def test_aggressive_always_cav():
    s = _sel()
    types = {f"v{i}": "aggressive_car" for i in range(50)}
    cavs = s.add_departed_aggressive_first(list(types), types)
    assert len(cavs) == 50, "all aggressive vehicles must become CAVs"


def test_overall_rate_approx_penetration_and_aggressive_captured():
    import statistics

    rates = []
    all_agg_captured = True
    for rep in range(60):
        s = _sel(seed=rep)
        types = {f"v{i}": ("aggressive_car" if i % 10 == 0 else "normal_car") for i in range(100)}
        cavs = set(s.add_departed_aggressive_first(list(types), types))
        rates.append(len(cavs) / 100)
        # every aggressive vehicle must be a CAV
        if any(f"v{i}" not in cavs for i in range(0, 100, 10)):
            all_agg_captured = False
    mean_rate = statistics.mean(rates)
    assert 0.16 <= mean_rate <= 0.24, f"overall CAV rate {mean_rate:.3f} should be ~0.20"
    assert all_agg_captured, "every aggressive vehicle must be captured as a CAV"


def test_no_dup_cavs():
    s = _sel()
    types = {f"v{i}": "aggressive_car" for i in range(10)}
    first = s.add_departed_aggressive_first(list(types), types)
    assert len(first) == 10
    # re-process the same vehicles (already CAVs) -> no new ones
    again = s.add_departed_aggressive_first(list(types), types)
    assert again == []
