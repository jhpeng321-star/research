"""V10ENVFIX (2026-10-19) read-only census + runtime envelope replay.

Two read-only measurements on the TRAIN-val recorded departures used by the V10UNBLOCK audit
(``merge_calib/data/recs_native.pkl`` + ``recs_genrand.pkl``):

1. **Envelope departure audit through the RUNTIME code path.**  For every recorded ramp
   (lane 2) departure the census builds the recorded post-step configuration and calls the
   shipped ``wms.physics.mobil.body_overlap`` (not the audit's own re-implementation):
   ``strict`` = no exemption, ``after`` = the V10ENVFIX simultaneous-change exemption
   (target-lane bodies whose lane changed in that step are not intersections).  It also
   replays the *pre-step* ``physical_merge_envelope`` (the path ``lane_v5`` / ``wms.py`` use)
   before and after the fix.

2. **Census per head and distance bin** (Astra follow-up): rows, positives, stays, rows with a
   missing t+1 endpoint (currently labelled 0 in ``lane_v5.recorded_rows``), and excluded
   infeasible positives -- under the strict runtime envelope and under the fixed one.
   ``lane_v5.py`` is NOT touched.

Usage (worktree root, ``PYTHONPATH="src;."``)::

    python tools/lane_envfix_census.py [--records R.pkl ...] [--out census.json]

Read-only: no training, no simulation, no CUDA, no SUMO.  Refuses sealed/holdout paths and
evaluation seeds 964000-964199.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch

from wms.physics.config import PhysicsConfig
from wms.physics.mobil import body_overlap, physical_merge_envelope

FORBIDDEN = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1", re.I)
EVAL_SEEDS = range(964000, 964200)
DT = 0.5
ZONE_START_M = 0.0
ZONE_END_M = 75.44
LENGTH_M = 5.0
BINS = (0.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 80.0)
HEADS = ("overtake", "return", "merge")          # head index of lane 1, 0, 2
HEAD_OF_LANE = (1, 0, 2)
TARGET_OF_LANE = (1, 0, 1)


def guard(path: str) -> str:
    if FORBIDDEN.search(str(path)):
        raise SystemExit(f"refusing restricted path {path}")
    return str(path)


def _bin(d: float) -> int:
    for b in range(len(BINS) - 1):
        if d < BINS[b + 1]:
            return b
    return len(BINS) - 2


def _physics() -> PhysicsConfig:
    return PhysicsConfig(sumo_fidelity=True, lane_rule="learned_discretionary",
                         lane_gap_rule="speed", merge_end_wait=True,
                         mainline_merge_yield=True, lane_learnable=True)


def _kin_frame(s, lane, v, present):
    """``[1, K, 7]`` from a frame; invalid lanes carry no one-hot but keep their state."""
    k = len(s)
    kin = torch.zeros(1, k, 7)
    kin[0, :, 0] = torch.as_tensor(s, dtype=torch.float32)
    onehot = torch.zeros(k, 3)
    ok = (lane >= 0) & (lane <= 2)
    onehot[torch.as_tensor(ok)] = torch.nn.functional.one_hot(
        torch.as_tensor(lane[ok]).long(), 3).float()
    kin[0, :, 1:4] = onehot
    kin[0, :, 4] = torch.as_tensor(v, dtype=torch.float32)
    kin[0, :, 6] = LENGTH_M
    return kin, torch.as_tensor(present, dtype=torch.bool).unsqueeze(0)


def _empty_head_table():
    return {h: [{"rows_strict": 0, "rows_fixed": 0, "positives": 0, "stays": 0,
                 "censored": 0, "infeasible_positives_strict": 0,
                 "infeasible_positives_fixed": 0} for _ in range(len(BINS) - 1)]
            for h in HEADS}


def run(records: list[str]) -> dict:
    p = _physics()
    heads = _empty_head_table()
    totals = {"rows_strict": 0, "rows_fixed": 0, "positives": 0, "stays": 0,
              "censored": 0, "infeasible_positives_strict": 0,
              "infeasible_positives_fixed": 0}
    dep = {"all": 0, "legal": 0, "strict": 0, "after": 0, "after_genuine": 0,
           "newly_admitted": 0, "newly_admitted_overlap_at_t1": 0,
           "newly_admitted_genuine": 0,
           "mc_genuine_only": 0, "mc_absent_at_t_only": 0, "mc_mixed_only": 0,
           "pre_step_strict": 0, "pre_step_fixed": 0, "pre_step_newly_admitted": 0,
           "same_lane_overlaps": 0, "mid_change_only_overlaps": 0}
    n_rec = 0
    for path in records:
        with open(guard(path), "rb") as fh:
            recs = pickle.load(fh)
        for rec in recs:
            seed = rec.get("seed")
            if seed is not None and int(seed) in EVAL_SEEDS:
                raise SystemExit(f"refusing evaluation seed {seed}")
            frames = rec["recorded"][rec["arm"]]
            nf = len(frames)
            ids = sorted({v["id"] for fr in frames for v in fr["vehicles"]})
            index = {i: q for q, i in enumerate(ids)}
            n = len(ids)
            S = np.full((nf, n), np.nan)
            Ln = np.full((nf, n), -1, dtype=np.int64)
            V = np.zeros((nf, n))
            Pr = np.zeros((nf, n), dtype=bool)
            Cav = np.zeros((nf, n), dtype=bool)
            for t, fr in enumerate(frames):
                for v in fr["vehicles"]:
                    q = index[v["id"]]
                    S[t, q] = float(v["s"])
                    Ln[t, q] = int(v["lane"])
                    V[t, q] = float(v["v"])
                    Pr[t, q] = bool(v["present"])
                    Cav[t, q] = bool(v["is_cav"])
            for t in range(nf - 1):
                s_t = np.nan_to_num(S[t])
                lane_t = Ln[t]
                pres_t = Pr[t] & (lane_t >= 0)
                kin, present = _kin_frame(s_t, lane_t, np.nan_to_num(V[t]), pres_t)
                strict = physical_merge_envelope(kin, present, p)["admissible"][0]
                if t >= 1:
                    last = ((Ln[t - 1] != lane_t) & (Ln[t - 1] >= 0) & (lane_t >= 0)
                            & (Ln[t - 1] != Ln[t]))
                else:
                    last = np.zeros(n, dtype=bool)
                fixed = physical_merge_envelope(
                    kin, present, p, mid_change=torch.as_tensor(last).unsqueeze(0)
                )["admissible"][0]
                # ---- census rows (HDV present at t)
                endpoint = Pr[t + 1] & (Ln[t + 1] >= 0)
                for q in range(n):
                    if not (Pr[t, q] and not Cav[t, q] and 0 <= lane_t[q] <= 2):
                        continue
                    if not (S[t, q] == S[t, q]):
                        continue
                    h = HEADS[HEAD_OF_LANE[lane_t[q]]]
                    b = _bin(ZONE_END_M - float(S[t, q]))
                    ep = bool(endpoint[q])
                    pos = ep and (Ln[t + 1, q] == TARGET_OF_LANE[lane_t[q]])
                    row = heads[h][b]
                    fs, ff = bool(strict[q]), bool(fixed[q])
                    for table in (row, totals):
                        table["rows_strict"] += int(fs)
                        table["rows_fixed"] += int(ff)
                        table["positives"] += int(pos)
                        table["stays"] += int(ep and Ln[t + 1, q] == lane_t[q])
                        table["censored"] += int(not ep)
                        table["infeasible_positives_strict"] += int(pos and not fs)
                        table["infeasible_positives_fixed"] += int(pos and not ff)
                # ---- recorded ramp departures
                for q in range(n):
                    if not (Pr[t, q] and 0 <= S[t, q] == S[t, q]):
                        continue
                    if lane_t[q] != 2 or Ln[t + 1, q] == 2 or Ln[t + 1, q] < 0:
                        continue
                    if not (S[t + 1, q] == S[t + 1, q]):
                        continue
                    if not endpoint[q]:
                        continue
                    dep["all"] += 1
                    if not (ZONE_START_M <= S[t, q] <= ZONE_END_M):
                        continue
                    dep["legal"] += 1
                    # audit-config runtime body_overlap at recorded t+1
                    s1 = np.nan_to_num(S[t + 1])
                    kin1, pres1 = _kin_frame(s1, Ln[t + 1], np.nan_to_num(V[t + 1]),
                                             Pr[t + 1] & (Ln[t + 1] >= 0))
                    tq = torch.full((1, n), 1, dtype=torch.long)
                    hit_strict = body_overlap(kin1, pres1, tq, 0.0)
                    # audit-consistent mask: lane index differs between t and t+1 (a body
                    # absent at t carries -1 and is therefore "changed" -- the audit's rule).
                    mc_audit = torch.as_tensor(Ln[t] != Ln[t + 1]).unsqueeze(0)
                    # genuine runtime mask: only bodies that were ON the road at t.
                    mc_gen = torch.as_tensor((Ln[t] != Ln[t + 1]) & (Ln[t] >= 0)).unsqueeze(0)
                    hit_after = body_overlap(kin1, pres1, tq, 0.0, mc_audit)
                    hit_gen = body_overlap(kin1, pres1, tq, 0.0, mc_gen)
                    st = not bool(hit_strict[0, q])
                    af = not bool(hit_after[0, q])
                    ag = not bool(hit_gen[0, q])
                    dep["strict"] += int(st)
                    dep["after"] += int(af)
                    dep["after_genuine"] += int(ag)
                    # overlaps by type at t+1
                    for j in range(n):
                        if j == q or Ln[t + 1, j] != 1 or not (S[t + 1, j] == S[t + 1, j]):
                            continue
                        if (S[t + 1, j] - LENGTH_M) < S[t + 1, q] and S[t + 1, j] > S[t + 1, q] - LENGTH_M:
                            if Ln[t, j] != Ln[t + 1, j]:
                                dep["mid_change_only_overlaps"] += 1
                            else:
                                dep["same_lane_overlaps"] += 1
                    if af and not st:
                        dep["newly_admitted"] += 1
                        if bool(hit_strict[0, q]):
                            dep["newly_admitted_overlap_at_t1"] += 1
                        js = [j for j in range(n)
                              if j != q and Ln[t + 1, j] == 1 and S[t + 1, j] == S[t + 1, j]
                              and (S[t + 1, j] - LENGTH_M) < S[t + 1, q]
                              and S[t + 1, j] > S[t + 1, q] - LENGTH_M]
                        gen, spa = any(Ln[t, j] >= 0 for j in js), any(Ln[t, j] < 0 for j in js)
                        if gen and spa:
                            dep["mc_mixed_only"] += 1
                        elif gen:
                            dep["mc_genuine_only"] += 1
                        else:
                            dep["mc_absent_at_t_only"] += 1
                    if ag and not st:
                        dep["newly_admitted_genuine"] += 1
                    # pre-step runtime envelope (lane_v5 / wms.py path)
                    dep["pre_step_strict"] += int(bool(strict[q]))
                    dep["pre_step_fixed"] += int(bool(fixed[q]))
                    dep["pre_step_newly_admitted"] += int(bool(fixed[q]) and not bool(strict[q]))
            n_rec += 1
    if dep["legal"]:
        dep["retention_strict"] = dep["strict"] / dep["legal"]
        dep["retention_after"] = dep["after"] / dep["legal"]
        dep["retention_after_genuine"] = dep["after_genuine"] / dep["legal"]
        dep["pre_step_retention_strict"] = dep["pre_step_strict"] / dep["legal"]
        dep["pre_step_retention_fixed"] = dep["pre_step_fixed"] / dep["legal"]
    dep["retention_strict_of_all"] = dep["strict"] / dep["all"] if dep["all"] else 0.0
    dep["retention_after_of_all"] = dep["after"] / dep["all"] if dep["all"] else 0.0
    return {"records": records, "n_episodes": n_rec, "dt_s": DT, "length_m": LENGTH_M,
            "zone_m": [ZONE_START_M, ZONE_END_M], "bins_m": list(BINS), "heads": list(HEADS),
            "departures": dep, "census_totals": totals, "census": heads}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", action="append", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    records = args.records or [
        "E:/research-dense/runs/active/C4-V7FT-20261017/merge_calib/data/recs_native.pkl",
        "E:/research-dense/runs/active/C4-V7FT-20261017/merge_calib/data/recs_genrand.pkl",
    ]
    out = run(records)
    text = json.dumps(out, indent=1, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
