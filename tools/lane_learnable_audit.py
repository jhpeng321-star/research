"""V10UNBLOCK (2026-10-19) CPU learnability audit of ``physics.lane_learnable``.

Two capacity / blocking checks (no training, no SUMO, no GPU):

``envelope``  Check (a): the physical envelope's retention of recorded SUMO ramp departures.
    Reads the TRAIN-val record pickles (``merge_calib.py records`` output) and replays the
    recorded lane-2 -> lane-1 departure of every ramp HDV.  A departure is ADMITTED when the
    transition is road-legal (in the accel-lane zone) and, one 0.5 s step later, the ego body
    ``[s - L, s]`` does not intersect a body of the target lane, except for a target vehicle
    that is ITSELF changing lane in that step (a simultaneous-change conflict, kept by the
    physical envelope).  Front/rear/centre position conventions are all evaluated to show that
    they are translation-invariant (the convention is NOT the source of the residual overlaps).

``grad``      Check (b): a lane head gradients on ALL at-risk rows, including a row the legacy
    behavioural speed rule vetoed and a forced-window row.  Builds the all-risk rows with
    ``wms.model.lane_v5.recorded_rows`` (``lane_learnable=True``) and prints the gradient norm
    of each of the three heads (overtake / return / merge).

Usage (worktree root, ``PYTHONPATH="src;.;tools"``):

    python tools/lane_learnable_audit.py envelope --records RECS1 [--records RECS2] [--out j]
    python tools/lane_learnable_audit.py grad
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

FORBIDDEN = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1", re.I)
EVAL_SEEDS = range(964000, 964200)
DT = 0.5
ZONE_START_M = 0.0
ZONE_END_M = 75.44
LENGTH_M = 5.0


def _guard(path: str) -> str:
    if FORBIDDEN.search(path):
        raise SystemExit(f"refusing restricted path {path}")
    return path


def _body(s: float, conv: str) -> tuple[float, float]:
    if conv == "center":
        return (s - LENGTH_M / 2.0, s + LENGTH_M / 2.0)
    if conv == "rear":
        return (s, s + LENGTH_M)
    return (s - LENGTH_M, s)          # front bumper convention of the WM (s = front)


def _overlap(a, b, margin: float = 0.0) -> bool:
    return (b[0] < a[1] + margin) and (b[1] > a[0] - margin)


def envelope(records: list[str], margin: float = 0.0) -> dict:
    n_dep = n_legal = 0
    strict = simul = 0
    same_lane_overlap = mid_change_only = 0
    conv_same = {"front": 0, "rear": 0, "center": 0}
    for path in records:
        with open(_guard(path), "rb") as fh:
            recs = pickle.load(fh)
        for rec in recs:
            seed = rec.get("seed")
            if seed is not None and int(seed) in EVAL_SEEDS:
                raise SystemExit(f"refusing evaluation seed {seed}")
            frames = rec["recorded"][rec["arm"]]
            n_frames = len(frames)
            ids = sorted({v["id"] for fr in frames for v in fr["vehicles"]})
            s = {i: [float("nan")] * n_frames for i in ids}
            lane = {i: [-1] * n_frames for i in ids}
            for t, fr in enumerate(frames):
                for v in fr["vehicles"]:
                    s[v["id"]][t] = float(v["s"])
                    lane[v["id"]][t] = int(v["lane"])
            for i in ids:
                for t in range(n_frames - 1):
                    if lane[i][t] != 2 or lane[i][t + 1] == 2 or lane[i][t + 1] < 0:
                        continue
                    if not (s[i][t] == s[i][t]):  # NaN
                        continue
                    if not (s[i][t + 1] == s[i][t + 1]):
                        continue
                    n_dep += 1
                    legal = ZONE_START_M <= s[i][t] <= ZONE_END_M
                    if not legal:
                        continue
                    n_legal += 1
                    front = _body(s[i][t + 1], "front")
                    hit_nochg = hit_mid = False
                    for j in ids:
                        if j == i or lane[j][t + 1] != 1:
                            continue
                        if not (s[j][t + 1] == s[j][t + 1]):
                            continue
                        if _overlap(front, _body(s[j][t + 1], "front"), margin):
                            if lane[j][t] != lane[j][t + 1]:
                                hit_mid = True
                            else:
                                hit_nochg = True
                    strict += int(not (hit_nochg or hit_mid))
                    simul += int(not hit_nochg)
                    same_lane_overlap += int(hit_nochg)
                    mid_change_only += int(hit_mid and not hit_nochg)
                    for conv in conv_same:
                        ego = _body(s[i][t + 1], conv)
                        if not any(_overlap(ego, _body(s[j][t + 1], conv), margin)
                                   for j in ids if j != i and lane[j][t + 1] == 1
                                   and s[j][t + 1] == s[j][t + 1]):
                            conv_same[conv] += 1
    return {
        "records": records, "dt_s": DT, "length_m": LENGTH_M, "margin_m": margin,
        "zone_m": [ZONE_START_M, ZONE_END_M],
        "recorded_departures_all": n_dep,
        "recorded_departures_legal": n_legal,
        "legal_fraction_of_all": (n_legal / n_dep if n_dep else 0.0),
        "admitted_strict_body": strict,
        "admitted_with_simultaneous_change": simul,
        "retention_strict_of_legal": (strict / n_legal if n_legal else 0.0),
        "retention_with_simultaneous_of_legal": (simul / n_legal if n_legal else 0.0),
        "retention_with_simultaneous_of_all": (simul / n_dep if n_dep else 0.0),
        "same_lane_body_overlaps": same_lane_overlap,
        "simultaneous_change_overlaps_only": mid_change_only,
        "clear_by_convention_at_t_plus_1": conv_same,
        "conventions_identical": len(set(conv_same.values())) == 1,
    }


def grad() -> dict:
    import torch

    from wms.model import lane_v5 as lv5
    from wms.physics.config import PhysicsConfig
    from wms.physics.mobil import MASKED_LOGIT, mobil_merge_logits

    p = PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary",
                      lane_gap_rule="speed", merge_end_wait=True,
                      mainline_merge_yield=True, lane_learnable=True)
    # slot 0: ramp merger in the zone with a 1 m lane-1 follower (legacy speed-rule veto, the
    #         bodies do not intersect); slot 1: that lane-1 follower; slot 2: a lane-0 car.
    s1 = p.merge_zone_s1_m
    kin = torch.zeros(1, 4, 7)
    kin[..., 6] = LENGTH_M
    kin[0, 0, 0], kin[0, 0, 1:4], kin[0, 0, 4] = s1 - 15.0, torch.tensor([0.0, 0.0, 1.0]), 8.0
    kin[0, 1, 0], kin[0, 1, 1:4], kin[0, 1, 4] = s1 - 21.0, torch.tensor([0.0, 1.0, 0.0]), 8.0
    kin[0, 2, 0], kin[0, 2, 1:4], kin[0, 2, 4] = 200.0, torch.tensor([1.0, 0.0, 0.0]), 20.0
    kin[0, 3, 0], kin[0, 3, 1:4] = 260.0, torch.tensor([0.0, 1.0, 0.0])       # a second overtake row
    present = torch.ones(1, 4, dtype=torch.bool)
    old = mobil_merge_logits(kin, present, p.__class__(**{**p.__dict__, "lane_learnable": False}))
    new = mobil_merge_logits(kin, present, p)
    formerly_vetoed = int(old[0, 0, 1] <= MASKED_LOGIT + 1e-6 and new[0, 0, 1] > MASKED_LOGIT + 1e-6)

    t = 3
    states = kin.unsqueeze(1).expand(1, t + 1, 4, 7).clone()
    states[:, 2:, 0, 0] += 1.0
    valid = torch.ones(1, t + 1, 4, dtype=torch.bool)
    is_cav = torch.zeros(1, t + 1, 4, dtype=torch.bool)
    age = torch.full((1, t, 4), 999.0)
    acc0 = lv5.initial_acc(valid[:, 0], states[:, 0], p, None, None, 1)
    rows = lv5.recorded_rows(states, valid, is_cav, p, None, age, acc0)
    p_off = p.__class__(**{**p.__dict__, "lane_learnable": False})
    legacy = lv5.recorded_rows(states, valid, is_cav, p_off, None, age, acc0)
    net = lv5.LaneV5Net()
    logits = net(rows["features"]).gather(-1, rows["head"].unsqueeze(-1)).squeeze(-1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, rows["label"])
    loss.backward()
    g = net.heads.weight.grad
    per_head = {name: float(g[h].abs().sum()) for h, name in enumerate(lv5.TRANSITIONS)}
    return {"rows_learnable": int(rows["stats"]["rows"]),
            "rows_legacy": int(legacy["stats"]["rows"]),
            "stats_learnable": rows["stats"], "stats_legacy": legacy["stats"],
            "formerly_vetoed_merge_row_included": bool(formerly_vetoed),
            "grad_norm_per_head": per_head,
            "all_heads_nonzero": all(v > 0.0 for v in per_head.values())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("check", choices=("envelope", "grad"))
    ap.add_argument("--records", action="append", default=[])
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.check == "envelope":
        if not args.records:
            ap.error("envelope needs --records")
        out = envelope(args.records, args.margin)
    else:
        out = grad()
    text = json.dumps(out, indent=1, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
