"""Decision-rule truth table (REGISTRATION_WM_N1.md:79-88)."""
from __future__ import annotations

import numpy as np

from evalh import metrics as M
from evalh.protocol import HORIZONS_S, QUANTILE_LEVELS, OutcomeRow, RolloutBatch, RolloutRow


def _base(*, n_good_contrasts=3, taus_qualifying=3, level_ratio=1.0,
          s_ratio=1.0, coll_v=0.0, coll_h=0.0, good_ci=(-1.0, -0.5)):
    contrasts = []
    for ci, c in enumerate(M.GATING_CONTRASTS):
        for ti, tau in enumerate(M.GATING_TAUS):
            if ci < n_good_contrasts and ti < taus_qualifying:
                contrasts.append(M.ContrastResult(c, tau, 1.0, 2.0,
                                                  good_ci[0], good_ci[1]))
            else:
                contrasts.append(M.ContrastResult(c, tau, 2.0, 1.0, -0.5, 0.5))
    level_crps = {t: dict(variant_crps=level_ratio, hybrid_crps=1.0)
                  for t in M.GATING_TAUS}
    return M.DecisionEvidence(
        contrasts=contrasts, level_crps=level_crps,
        s_rmse={10.0: (s_ratio, 1.0), 30.0: (s_ratio, 1.0)},
        collisions_per_1000=(coll_v, coll_h))


def test_all_pass_beats_hybrid():
    r = M.decide(_base())
    assert r["verdict"] == "WM-N1_BEATS_HYBRID"
    assert r["passed"] is True
    assert all(c["passed"] for c in r["checks"].values())


def test_item1_needs_two_contrasts():
    r = M.decide(_base(n_good_contrasts=1))
    assert r["checks"]["1_dv_crps_ge2_contrasts_ge2_taus"]["passed"] is False
    assert r["verdict"] == "HYBRID_STAYS"


def test_item1_needs_two_taus_per_contrast():
    r = M.decide(_base(taus_qualifying=1))
    assert r["checks"]["1_dv_crps_ge2_contrasts_ge2_taus"]["passed"] is False
    assert r["verdict"] == "HYBRID_STAYS"


def test_item1_ci_must_exclude_zero():
    r = M.decide(_base(good_ci=(-0.5, 0.5)))
    assert r["checks"]["1_dv_crps_ge2_contrasts_ge2_taus"]["passed"] is False
    assert r["verdict"] == "HYBRID_STAYS"


def test_item2_outcome_crps_within_5pct():
    ok = M.decide(_base(level_ratio=1.05))
    assert ok["checks"]["2_outcome_crps_within_5pct"]["passed"] is True
    bad = M.decide(_base(level_ratio=1.06))
    assert bad["checks"]["2_outcome_crps_within_5pct"]["passed"] is False
    assert bad["verdict"] == "HYBRID_STAYS"


def test_item3_s_rmse_within_10pct():
    ok = M.decide(_base(s_ratio=1.10))
    assert ok["checks"]["3_h10_h30_s_rmse_within_10pct"]["passed"] is True
    bad = M.decide(_base(s_ratio=1.11))
    assert bad["checks"]["3_h10_h30_s_rmse_within_10pct"]["passed"] is False
    assert bad["verdict"] == "HYBRID_STAYS"


def test_item4_collision_margin():
    ok = M.decide(_base(coll_v=0.1, coll_h=0.0))
    assert ok["checks"]["4_collisions_within_margin"]["passed"] is True
    bad = M.decide(_base(coll_v=0.11, coll_h=0.0))
    assert bad["checks"]["4_collisions_within_margin"]["passed"] is False
    assert bad["verdict"] == "HYBRID_STAYS"


def test_dependency_skipping_disallowed():
    """Each failing item alone flips the verdict; the four are AND-ed."""
    for kwargs in (dict(n_good_contrasts=0), dict(level_ratio=9.0),
                   dict(s_ratio=9.0), dict(coll_v=9.0)):
        assert M.decide(_base(**kwargs))["verdict"] == "HYBRID_STAYS"


def test_quote_is_registration_text():
    r = M.decide(_base())
    assert "world-bootstrap 90 % CI" in r["quote"]
    assert "0.1 per 1000 veh-steps" in r["quote"]
    assert "not worse than T's by more than 5 %" in r["quote"]
    assert "not worse than Part A v2g by more than 10 %" in r["quote"]
    for c in r["checks"].values():
        assert c["citation"].startswith("REGISTRATION_WM_N1.md:")


def test_evidence_from_metrics_end_to_end():
    rows = []
    vq, hq = [], []
    for w in range(8):
        for pol, y in (("damping", 10.0 + w), ("native", 4.0),
                       ("pi", 6.0 + 0.5 * w), ("wall", 3.0 + 0.2 * w)):
            for tau in M.GATING_TAUS:
                rows.append(OutcomeRow(f"w{w}", 300.0, 120.0, tau, pol, y))
                vq.append(np.full(QUANTILE_LEVELS.size, y))
                # hybrid overestimates the native arm by +4 (a contrast error)
                hq.append(np.full(QUANTILE_LEVELS.size,
                                  y + (4.0 if pol == "native" else 0.0)))
    vq = np.array(vq)
    hq = np.array(hq)
    delta = M.delta_metrics(M.build_delta_pairs(rows, vq, hq),
                            threshold=1.0, n_boot=200, seed=0)
    level = M.level_crps_pair(rows, vq, hq)
    pos = np.zeros((8, len(HORIZONS_S), 3))
    rows_r = [RolloutRow(f"w{i}", 300.0, "native", true_pos=pos[i],
                         true_speed=pos[i], true_dist=1.0,
                         true_speed_scale=1.0, veh_steps=100.0)
              for i in range(8)]
    good = M.rollout_metrics(RolloutBatch(HORIZONS_S, pos, pos,
                                          mask=np.ones_like(pos),
                                          collision=np.zeros(8)),
                             rows_r, name="variant")
    bad_pos = pos + 5.0
    bad = M.rollout_metrics(RolloutBatch(HORIZONS_S, bad_pos, pos,
                                         mask=np.ones_like(pos),
                                         collision=np.ones(8)),
                            rows_r, name="hybrid")
    ev = M.evidence_from_metrics(delta, level, good, bad)
    r = M.decide(ev)
    assert r["verdict"] in ("WM-N1_BEATS_HYBRID", "HYBRID_STAYS")
    assert r["checks"]["1_dv_crps_ge2_contrasts_ge2_taus"]["passed"] is True
