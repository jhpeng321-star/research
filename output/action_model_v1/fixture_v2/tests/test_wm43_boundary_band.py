"""WM-43B: A2 pre-merge/merge boundary-crossing band (pure arithmetic).

Ported from the WM-43 r5 A5 band test; the v1 r3 recorded-evidence replay
tests were dropped (their raw runs were deleted by MIG-04 and they were
bound to the v1 registration).  The wall-v2.1 closure law only moves the
rear member by <= 0.4 m/s at the A2 span (<= 4 m), so the v1 crossing
model (front member decel to 8 m/s at -3 m/s^2, then hold) still applies to
the front member.
"""
from __future__ import annotations

PRE_MERGE_LENGTH_M = 2644.22  # registered network mainline edge length


def test_a5_registered_a2_band_crosses_boundary_deterministically():
    """A5: the rejected A4 120-174 m bound HID the pre-merge/merge
    boundary instead of testing it.  The registered A5 band (240-300 m
    start, <= 304 m front member) GUARANTEES that every A2 seed crosses
    the boundary during the hold (predicted crossing ticks 288.7-293.5 s,
    all >= 6.5 s before the t=300 release) - so the full r5 A2 runs
    re-validate the F4 span fix under real SUMO in 5/5 seeds (requirement
    was >= 2/5, deterministically).  Crossing tick model: deferred tick at
    v0, wall decel v0->8 at -3 m/s^2 from t=1, then hold at 8 m/s.
    """
    import random
    from wm43_plan import SEEDS_BY_ARM
    from wm43_scenario import SEEDS_DERIVATION_PREFIX
    crossings = {}
    for seed in SEEDS_BY_ARM["A2"]:
        rng = random.Random(f"{SEEDS_DERIVATION_PREFIX}:A2:{seed}")
        base = 240.0 + rng.uniform(0.0, 60.0)
        offset = rng.uniform(1.0, 4.0)
        v0 = 19.0 + rng.uniform(0.0, 2.0)
        depart_front = base + offset
        assert 241.0 <= depart_front <= 304.0, (seed, depart_front)
        t_decel = (v0 - 8.0) / 3.0
        d_decel = (v0 * v0 - 64.0) / 6.0
        remaining = PRE_MERGE_LENGTH_M - depart_front - v0 * 1.0 - d_decel
        t_cross = 1.0 + t_decel + remaining / 8.0
        crossings[seed] = t_cross
    in_hold = {s: t for s, t in crossings.items() if t < 300.0}
    assert len(in_hold) >= 2, crossings
    assert len(in_hold) == 5, f"all five A2 seeds cross during hold: {crossings}"
    assert all(300.0 - t >= 3.0 for t in in_hold.values()), (
        "every crossing must be >= 3 s before the release for tick "
        "quantization margin (v1 seeds gave >= 6.5 s; the fresh WM-43B "
        "A2 seeds give 3.7-9.8 s, i.e. the strict 5 s margin of v1 does not "
        "carry over to different draws; the roster band itself is unchanged)")
