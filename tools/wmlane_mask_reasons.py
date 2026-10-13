"""Why does the physics feasibility mask veto recorded lane changes? (wall_same, default params).

Usage (workspace root, PYTHONPATH=src):
    python tools/wmlane_mask_reasons.py [out.json] [--gap-rule fixed|speed]

``--gap-rule`` (GAPRULE) selects ``PhysicsConfig.lane_gap_rule`` (default ``fixed`` = the
original diagnostic).  Besides the per-category veto reasons it reports the FALSE-FEASIBLE
side: over every HDV step whose car did NOT change lane, for each legal adjacent target
(no change into the ramp lane, 2 -> 1 only), the fraction the mask allows and the fraction
it allows although the cut-in would be DANGEROUS (``tools/gaprule_fit.dangerous``: bumper
gap < 1 m to the target-lane leader/follower, or > 4.5 m/s2 constant decel needed by the
follower / the ego to avoid a collision, or a time headway < 0.3 s above 5 m/s), judged
on the one-step-anticipated configuration like the speed rule (``recorded_dangerous``
counts recorded SUMO changes that this definition flags).  ``masked_t_and_t1`` counts
recorded changes that are masked at t AND one step later (t+1 state, ego put back into its
origin lane): SUMO moves the cars before it changes lanes, so a car passing side by side at
t is often ahead at t+1.  Diagnostic only - never used to choose constants.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict

sys.path.insert(0, "src")
sys.path.insert(0, "tools")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from gaprule_fit import dangerous  # noqa: E402
from wmlane_event_diag import _kin, _record_tensors, category  # noqa: E402
from wms.eval.normalize import iter_windows  # noqa: E402
from wms.physics.config import PhysicsConfig  # noqa: E402
from wms.physics.idm import idm_accel  # noqa: E402
from wms.physics.mobil import anticipated_kin, mobil_merge_logits, target_overlap  # noqa: E402,E501
from wms.physics.neighbours import lane_neighbour  # noqa: E402

STATES = "E:/research-dense/data_cache/wallc1_states_h60/wall_same.json"


def _speed_tags(p, kin, pres, tl, j, v_ego, gap_l, lv, gap_f, fv):
    """Which part of the GAPRULE speed rule fails for car ``j`` (anticipated state)."""
    tags = []
    req_l = p.gap_s0_lead_m + p.gap_t_lead_s * v_ego \
        + max(v_ego ** 2 - lv ** 2, 0.0) / (2 * p.gap_b_lead_mps2)
    req_f = p.gap_s0_follow_m + p.gap_t_follow_s * fv \
        + max(fv ** 2 - v_ego ** 2, 0.0) / (2 * p.gap_b_follow_mps2)
    if bool(target_overlap(kin, pres, tl, p.gap_overlap_margin_m)[0, j]):
        tags.append("overlap")
    if gap_l < req_l:
        tags.append("lead_gap<req")
    if gap_f < req_f:
        tags.append("follow_gap<req")
    return tags


def _fixed_tags(p, v_ego, gap_l, gap_f, fv):
    tags = []
    # follower braking if the car cuts in (default IDM)
    a_f = float(idm_accel(torch.tensor([fv]), torch.tensor([v_ego]), torch.tensor([gap_f]),
                          p.idm_params(), accel_min=-9.0, accel_max=3.0))
    if gap_l < p.lane_safe_gap_m:
        tags.append("lead_gap<6m")
    if gap_f < p.lane_safe_gap_m:
        tags.append("follow_gap<6m")
    if a_f < -p.b_safe_end_mps2:
        tags.append("follower_brake>6")
    elif a_f < -p.b_safe_base_mps2:
        tags.append("follower_brake>4")
    return tags


def _ff_summary(c):
    k = max(c["nonchange_states"], 1)
    return {**dict(c), "feasible_frac": c["feasible"] / k,
            "feasible_and_dangerous_frac": c["feasible_and_dangerous"] / k,
            "feasible_given_dangerous": c["feasible_and_dangerous"] / max(c["dangerous"], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--gap-rule", default="fixed", choices=("fixed", "speed"))
    ap.add_argument("--states", default=STATES)
    args = ap.parse_args()
    p = PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary",
                      lane_gap_rule=args.gap_rule)
    reasons = defaultdict(Counter)
    gaps = defaultdict(list)
    ff = defaultdict(Counter)       # false-feasible accounting (non-change steps x targets)
    for record in iter_windows(args.states):
        ids, s, v, acc, lane, valid, cav = _record_tensors(record)
        kin = _kin(s, v, acc, lane, valid)
        pres = torch.as_tensor(valid)
        logits = mobil_merge_logits(kin, pres, p)
        feasible = (logits > -8.0 + 1e-6).numpy()
        hdv = valid & ~cav
        t_n, n = lane.shape
        # Danger is judged on the configuration after the step (one-step anticipation).
        kin_a = anticipated_kin(kin, p.gap_anticipation_s)
        v_a = kin_a[..., 4].numpy()
        nb = {}
        for tl in (0, 1):
            q = torch.full(lane.shape, tl)
            gl, lv = lane_neighbour(kin_a, pres, q, p, ahead=True)
            gf, fv = lane_neighbour(kin_a, pres, q, p, ahead=False)
            ov = target_overlap(kin_a, pres, q, p.gap_overlap_margin_m)
            nb[tl] = (gl.numpy(), lv.numpy(), gf.numpy(), fv.numpy(), ov.numpy())
        both = hdv[:-1] & hdv[1:]
        l0, l1 = lane[:-1], lane[1:]
        for d in (-1, 1):
            tgt = l0 + d
            legal = both & (tgt >= 0) & (tgt <= 1) & ((l0 != 2) | (tgt == 1)) & (l1 == l0)
            ti, ki = np.nonzero(legal)
            tg = tgt[ti, ki]
            cols = [np.where(tg == 0, nb[0][c][ti, ki], nb[1][c][ti, ki]) for c in range(5)]
            dang = dangerous(v_a[ti, ki], *cols)
            ok = feasible[ti, ki, tg]
            cat = np.where(l0[ti, ki] == 2, "merge", np.where(tg == 0, "overtake", "return"))
            for name in ("merge", "overtake", "return"):
                m = cat == name
                ff[name]["nonchange_states"] += int(m.sum())
                ff[name]["feasible"] += int(ok[m].sum())
                ff[name]["dangerous"] += int(dang[m].sum())
                ff[name]["feasible_and_dangerous"] += int((ok & dang)[m].sum())
        for t in range(t_n - 1):
            for j in range(n):
                a, b = lane[t, j], lane[t + 1, j]
                if not (hdv[t, j] and hdv[t + 1, j] and a != b and abs(a - b) == 1):
                    continue
                cat = category(int(a), int(b))
                reasons[cat]["recorded"] += 1
                bi = int(b)
                if bool(dangerous(v_a[t, j], *(nb[bi][c][t, j] for c in range(5)))):
                    reasons[cat]["recorded_dangerous"] += 1
                if float(logits[t, j, b]) > -8.0 + 1e-6:
                    continue
                reasons[cat]["masked"] += 1
                # One step later (SUMO moves first, then changes lanes): the same change
                # judged on the t+1 state with the ego put back into its origin lane.
                k2 = kin[t + 1:t + 2].clone()
                k2[0, j, 1:4] = 0.0
                k2[0, j, 1 + int(a)] = 1.0
                if float(mobil_merge_logits(k2, pres[t + 1:t + 2], p)[0, j, b]) <= -8.0 + 1e-6:
                    reasons[cat]["masked_t_and_t1"] += 1
                k1, p1 = kin[t:t + 1], pres[t:t + 1]
                tl = torch.full((1, n), int(b))
                gl, lv = lane_neighbour(k1, p1, tl, p, ahead=True)
                gf, fv = lane_neighbour(k1, p1, tl, p, ahead=False)
                gap_l, gap_f = float(gl[0, j]), float(gf[0, j])
                v_ego = float(v[t, j])
                if args.gap_rule == "speed":
                    k1a = anticipated_kin(k1, p.gap_anticipation_s)
                    gla, lva = lane_neighbour(k1a, p1, tl, p, ahead=True)
                    gfa, fva = lane_neighbour(k1a, p1, tl, p, ahead=False)
                    tags = _speed_tags(p, k1a, p1, tl, j, float(k1a[0, j, 4]), float(gla[0, j]),
                                       float(lva[0, j]), float(gfa[0, j]), float(fva[0, j]))
                else:
                    tags = _fixed_tags(p, v_ego, gap_l, gap_f, float(fv[0, j]))
                if not tags:
                    tags.append("zone_or_other")
                for tag in tags:
                    reasons[cat][tag] += 1
                gaps[cat].append((gap_l, gap_f, v_ego))
    out = {c: dict(r) for c, r in reasons.items()}
    for c, g in gaps.items():
        arr = np.asarray(g)
        out[c]["median_lead_gap_m"] = float(np.median(arr[:, 0]))
        out[c]["median_follow_gap_m"] = float(np.median(arr[:, 1]))
        out[c]["median_speed_mps"] = float(np.median(arr[:, 2]))
    total_rec = sum(r["recorded"] for r in reasons.values())
    total_mask = sum(r.get("masked", 0) for r in reasons.values())
    total_both = sum(r.get("masked_t_and_t1", 0) for r in reasons.values())
    out["total"] = {"recorded": total_rec, "masked": total_mask,
                    "masked_frac": total_mask / max(total_rec, 1),
                    "masked_t_and_t1": total_both,
                    "masked_t_and_t1_frac": total_both / max(total_rec, 1)}
    allc = Counter()
    for c in ff.values():
        allc.update(c)
    out["false_feasible_nonchange"] = {**{k: _ff_summary(c) for k, c in ff.items()},
                                       "all": _ff_summary(allc)}
    out["gap_rule"] = args.gap_rule
    print(json.dumps(out, indent=1))
    if args.out:
        open(args.out, "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
