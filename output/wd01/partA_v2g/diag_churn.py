"""WD-01-PA2G churn diagnosis (TRAIN-only, zero simulation): reproduce one
logged window's discretionary LC churn and trace the ping-pong.

Usage: python diag_churn.py [world] [w] [steps]
Default: w8538m2600p40t360 870.0 20 (the worst PA2G-A H10 window).
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (str(WD01), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json  # noqa: E402

import partA_v2g.eval_a as EA  # noqa: E402
from partA_v2g.engine import PartAV2GEngine, load_params  # noqa: E402

WORLD = sys.argv[1] if len(sys.argv) > 1 else "w8538m2600p40t360"
W = float(sys.argv[2]) if len(sys.argv) > 2 else 870.0
NSTEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 20

params = load_params()
EA.MAPPER = EA.ModelMapper()
worlds = EA.load_train_worlds()
assert EA.TRAIN_WORLD_NAMES, "no TRAIN worlds loaded"
entry = next(e for e in worlds if e["world"] == WORLD)
log = EA.NativeLog(entry)
truth_w = log.truth_at(W)
vehicles, free_true = EA.tracked_state(truth_w, params)
print(f"world {WORLD} w={W} t0={log.t0} n_tracked={len(vehicles)} "
      f"n_cav_in_zone={sum(1 for v in vehicles if v in log.cav_ids)}")

eng = PartAV2GEngine(params, t0=W, vehicles=vehicles)

# per-step LC trace
lc_trace = []          # (step, t, vid, from_lane, to_lane)
prev_lane = dict(eng.snapshots[0])
for j in range(1, NSTEPS + 1):
    eng.step()
    for vid, st in eng.snapshots[j].items():
        if vid == "__t__":
            continue
        pl = prev_lane.get(vid)
        if pl is not None and pl[0] != st[0]:
            lc_trace.append((j, round(W + (j - 1) * 0.5, 1), vid,
                             pl[0], st[0]))
    prev_lane = {k: v for k, v in eng.snapshots[j].items() if k != "__t__"}

per_vid = {}
for row in lc_trace:
    per_vid.setdefault(row[2], []).append(row)
order = sorted(per_vid.items(), key=lambda kv: -len(kv[1]))
print(f"total LC events in {NSTEPS} steps: {len(lc_trace)} "
      f"(engine counts {eng.lc_counts})")
print("worst vehicles (vid: n_changes, sequence):")
for vid, seq in order[:12]:
    seqs = " ".join(f"{a[3]}->{a[4]}@{a[1]}" for a in seq)
    print(f"  {vid}: {len(seq)}  {seqs}")
print("interval stats (steps between consecutive changes, per vid):")
gaps = []
for vid, seq in per_vid.items():
    for k in range(1, len(seq)):
        gaps.append(seq[k][0] - seq[k - 1][0])
if gaps:
    import collections
    print("  ", dict(sorted(collections.Counter(gaps).items())))

# deep-dive the single worst ping-ponger: instrument its disc decisions
if order:
    vid = order[0][0]
    eng2 = PartAV2GEngine(params, t0=W, vehicles=vehicles)
    from partA_v2g.engine import _idm1, _seg_limit1, GAP_MAX
    calls = []

    def spy(self, r, tgt, to_right, d_sorted_t, rows_t):
        li = int(self.lane[r])
        d_i = float(self.off[li] - self.s[r])
        lrow, frow = self._d_neighbors(d_sorted_t, rows_t, d_i)
        s_ego_t = float(self.s[r]) + float(self.off[tgt] - self.off[li])
        v0_t = _seg_limit1(tgt, s_ego_t, self.p)
        if lrow >= 0:
            gap_lead = float(self.s[lrow] - self.length[lrow] - s_ego_t)
            v_lead = float(self.v[lrow])
        else:
            gap_lead, v_lead = GAP_MAX, 0.0
        if frow >= 0:
            gap_foll = float(s_ego_t - self.length[r] - self.s[frow])
            v_foll = float(self.v[frow])
        else:
            gap_foll, v_foll = GAP_MAX, 0.0
        rec = dict(step=self.step_i, vid=self.names[r],
                   direction=f"{li}->{tgt}", gap_lead=round(gap_lead, 2),
                   gap_foll=round(gap_foll, 2))
        if gap_lead < float(self.merge_p["gap_floor_m"]):
            rec["why"] = "gap_floor"
            calls.append(rec)
            return False, d_i
        a_ego_new = _idm1(float(self.v[r]), gap_lead,
                          float(self.v[r]) - v_lead, v0_t, self.idm_p)
        a_ego_old = self._a_stay(r)
        rec.update(a_new=round(a_ego_new, 3), a_stay=round(a_ego_old, 3))
        if (a_ego_new - a_ego_old) < float(
                self.mobil_p.get("ego_gain_floor_mps2", 0.0)):
            rec["why"] = "ego_gain_floor"
            calls.append(rec)
            return False, d_i
        a_new_after = a_new_before = 0.0
        blocked = None
        if frow >= 0:
            v0_f = _seg_limit1(tgt, float(self.s[frow]), self.p)
            a_new_after = _idm1(v_foll, gap_foll, v_foll - float(self.v[r]),
                                v0_f, self.idm_p)
            if a_new_after < -float(self.mobil_p["b_safe_mps2"]):
                blocked = "b_safe"
            lo, _ = self._own_leader_follower(frow)
            if lo >= 0:
                cg = float(self.s[lo] - self.length[lo] - self.s[frow])
                cv = float(self.v[lo])
            else:
                cg, cv = GAP_MAX, 0.0
            a_new_before = _idm1(v_foll, cg, v_foll - cv, v0_f, self.idm_p)
        lo_s, fo_s = self._own_leader_follower(r)
        a_old_after = a_old_before = 0.0
        if fo_s >= 0:
            v_of = float(self.v[fo_s])
            gap_of = float(self.s[r] - self.length[r] - self.s[fo_s])
            v0_o = _seg_limit1(li, float(self.s[fo_s]), self.p)
            a_old_before = _idm1(v_of, gap_of, v_of - float(self.v[r]),
                                 v0_o, self.idm_p)
            if lo_s >= 0:
                gap_ol = float(self.s[lo_s] - self.length[lo_s] - self.s[fo_s])
                v_ol = float(self.v[lo_s])
            else:
                gap_ol, v_ol = GAP_MAX, 0.0
            a_old_after = _idm1(v_of, gap_ol, v_of - v_ol, v0_o, self.idm_p)
        pol = float(self.mobil_p["politeness_p"])
        gain = (a_ego_new - a_ego_old) + pol * (
            (a_new_after - a_new_before) + (a_old_after - a_old_before))
        th = float(self.mobil_p["accel_threshold_mps2"])
        bias = float(self.mobil_p["keep_right_bias_mps2"])
        eff = th - bias if to_right else th + bias
        rec.update(d_new=round(a_new_after - a_new_before, 3),
                   d_old=round(a_old_after - a_old_before, 3),
                   gain=round(gain, 3), eff=round(eff, 3))
        if blocked:
            rec["why"] = blocked
        elif gain >= eff:
            rec["why"] = "ACCEPT"
        else:
            rec["why"] = "gain<eff"
        calls.append(rec)
        return (not blocked and gain >= eff), d_i

    PartAV2GEngine._disc_decision = spy
    for _ in range(NSTEPS):
        eng2.step()
    PartAV2GEngine._disc_decision = spy.__wrapped__ if hasattr(
        spy, "__wrapped__") else PartAV2GEngine._disc_decision
    mine = [c for c in calls if c["vid"] == vid]
    print(f"\nall disc decisions for {vid}:")
    for c in mine:
        print("  ", c)

print("\nopened run files:", sorted(EA.OPENED_RUN_FILES))
EA.assert_train_only()
print("assert_train_only: passed")
