"""WD-01-PA2G-LC1 pred-side instrumentation (zero SUMO).

Re-runs the v2g engine on the SAME 32 PA2G-A2 TRAIN worlds and windows
with the WD-01-PA2G-LC1 class-(ii) tracked-state fix applied, and dumps
BEFORE aggregation:

- pred_events.jsonl      every executed predicted lane change with local
                         context (same-lane neighbours, regime);
- decisions_accepted.jsonl  full MOBIL/merge term vectors of accepted
                         decisions (trace engine; re-derived terms are
                         asserted equal to the engine's decision);
- decisions_rejected.jsonl  compact rejection tallies (reason codes);
- window_rows.jsonl      per-window predicted LC counts per H (tracked
                         cohort, counting identical to truth);
- states.jsonl           tracked initial state per window;
- merge_outcomes_pred.jsonl  per tracked accel-lane vehicle: merge time /
                         stop-at-end / never-merged / dwell.

TRAIN only; opened run files recorded and asserted at close-out.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (str(WD01), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wdpaths  # noqa: E402
DATA_ROOT = Path(wdpaths.RESEARCH_DENSE_ROOT)

from partA_v2g import metrics as MX  # noqa: E402
from partA_v2g import eval_a as EA  # noqa: E402
from partA_v2g import eval_a2 as EA2  # noqa: E402
from partA_v2g.engine import (PartAV2GEngine, _idm1, _seg_limit1,  # noqa: E402
                              load_params)

ANALYSIS_ID = "WD-01-PA2G-LC1-pred"
DT = 0.5
HS = (10, 30, 60, 120)
H_MAX = 120
GAP_MAX = 250.0
RESULTS = HERE / "results" / "lc1"


class TraceEngine(PartAV2GEngine):
    """Instrumented engine: re-derives the decision term vectors (the
    parent's math, asserted consistent) and logs every decision."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.accepted: List[dict] = []
        self.rejected: List[dict] = []
        self._lc_seen = 0

    # -- term re-derivations (mirror engine.py; asserted vs the parent's
    #    accept boolean; diagnostic-only code path)
    def _disc_terms(self, r: int, tgt: int, to_right: bool,
                    d_sorted_t: List[float], rows_t: List[int]) -> dict:
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
        terms = dict(gap_lead=gap_lead, gap_foll=gap_foll,
                     v_ego=float(self.v[r]), v_lead=v_lead,
                     v_foll=v_foll, reason="")
        if gap_lead < float(self.merge_p["gap_floor_m"]):
            terms["reason"] = "gap_floor"
            return terms
        a_ego_new = self._aclip1(_idm1(
            float(self.v[r]), gap_lead, float(self.v[r]) - v_lead, v0_t,
            self.idm_p))
        a_ego_old = self._a_stay(r)
        a_new_after = a_new_before = 0.0
        if frow >= 0:
            v0_f = _seg_limit1(tgt, float(self.s[frow]), self.p)
            a_new_after = self._aclip1(_idm1(
                v_foll, gap_foll, v_foll - float(self.v[r]), v0_f,
                self.idm_p))
            if a_new_after < -float(self.mobil_p["b_safe_mps2"]):
                terms.update(a_ego_new=a_ego_new, a_ego_old=a_ego_old,
                             reason="b_safe")
                return terms
            lo, _ = self._own_leader_follower(frow)
            if lo >= 0:
                cg = float(self.s[lo] - self.length[lo] - self.s[frow])
                cv = float(self.v[lo])
            else:
                cg, cv = GAP_MAX, 0.0
            a_new_before = self._aclip1(_idm1(
                v_foll, cg, v_foll - cv, v0_f, self.idm_p))
        lo_s, fo_s = self._own_leader_follower(r)
        a_old_after = a_old_before = 0.0
        if fo_s >= 0:
            v_of = float(self.v[fo_s])
            gap_of = float(self.s[r] - self.length[r] - self.s[fo_s])
            v0_o = _seg_limit1(li, float(self.s[fo_s]), self.p)
            a_old_before = self._aclip1(_idm1(
                v_of, gap_of, v_of - float(self.v[r]), v0_o, self.idm_p))
            if lo_s >= 0:
                gap_ol = float(self.s[lo_s] - self.length[lo_s]
                               - self.s[fo_s])
                v_ol = float(self.v[lo_s])
            else:
                gap_ol, v_ol = GAP_MAX, 0.0
            a_old_after = self._aclip1(_idm1(
                v_of, gap_ol, v_of - v_ol, v0_o, self.idm_p))
        pol = float(self.mobil_p["politeness_p"])
        gain = (a_ego_new - a_ego_old) + pol * (
            (a_new_after - a_new_before) + (a_old_after - a_old_before))
        th = float(self.mobil_p["accel_threshold_mps2"])
        bias = float(self.mobil_p["keep_right_bias_mps2"])
        eff = th - bias if to_right else th + bias
        terms.update(a_ego_new=a_ego_new, a_ego_old=a_ego_old,
                     a_new_after=a_new_after, a_new_before=a_new_before,
                     a_old_after=a_old_after, a_old_before=a_old_before,
                     gain=gain, threshold=th, bias=bias, eff=eff,
                     politeness=pol)
        return terms

    def _disc_decision(self, r, tgt, to_right, d_sorted_t, rows_t):
        accept, d_i = super()._disc_decision(r, tgt, to_right,
                                             d_sorted_t, rows_t)
        terms = self._disc_terms(r, tgt, to_right, d_sorted_t, rows_t)
        re_acc = (terms["reason"] == "" and
                  terms["gain"] >= terms["eff"])
        assert re_acc == accept, \
            f"trace mismatch disc: {re_acc} vs {accept} terms={terms}"
        self._log_decision(accept, r, self.lane[r], tgt, "disc", terms)
        return accept, d_i

    def _merge_terms(self, r: int, d_sorted1: List[float],
                     rows1: List[int]) -> dict:
        mp = self.merge_p
        s1z = float(mp["zone_s1_m"])
        Lz = float(mp["zone_len_m"])
        u = min(max(1.0 - (s1z - float(self.s[r])) / Lz, 0.0), 1.0)
        bsafe = float(mp["b_safe_base_mps2"]) + u * (
            float(mp["b_safe_end_mps2"]) - float(mp["b_safe_base_mps2"]))
        d_i = float(self.off[2] - self.s[r])
        lrow, frow = self._d_neighbors(d_sorted1, rows1, d_i)
        s_ego_t = float(self.s[r]) + float(self.off[1] - self.off[2])
        if lrow >= 0:
            d_gap_lead = float(self.s[lrow] - s_ego_t)
            v_lead = float(self.v[lrow])
        else:
            d_gap_lead, v_lead = GAP_MAX, 0.0
        if frow >= 0:
            d_gap_foll = float(s_ego_t - self.s[frow])
            v_foll = float(self.v[frow])
        else:
            d_gap_foll, v_foll = GAP_MAX, 0.0
        terms = dict(u=u, bsafe=bsafe, d_gap_lead=d_gap_lead,
                     d_gap_foll=d_gap_foll, v_ego=float(self.v[r]),
                     v_foll=v_foll, reason="")
        s0i = float(self.idm_p["s0_m"])
        Ti = float(self.idm_p["T_s"])
        floor = float(mp["gap_floor_m"])
        if lrow >= 0 and d_gap_lead < s0i + max(0.0, float(self.v[r])
                                                * Ti):
            terms["reason"] = "lead_headway"
            return terms
        if frow >= 0 and d_gap_foll < s0i:
            terms["reason"] = "foll_s0"
            return terms
        if lrow >= 0:
            b_req_lead = (float(self.v[r]) ** 2 - v_lead ** 2) / \
                (2.0 * max(d_gap_lead - s0i, floor))
            if b_req_lead > bsafe:
                terms["reason"] = "lead_bsafe"
                terms["b_req_lead"] = b_req_lead
                return terms
        if frow >= 0:
            b_req_foll = (v_foll ** 2 - float(self.v[r]) ** 2) / \
                (2.0 * max(d_gap_foll - s0i, floor))
            if b_req_foll > bsafe:
                terms["reason"] = "foll_bsafe"
                terms["b_req_foll"] = b_req_foll
                return terms
        v0_t = _seg_limit1(1, s_ego_t, self.p)
        gap_lead = (GAP_MAX if lrow < 0 else
                    float(self.s[lrow] - self.length[lrow] - s_ego_t))
        a_ego_new = self._aclip1(_idm1(
            float(self.v[r]), gap_lead, float(self.v[r]) - v_lead, v0_t,
            self.idm_p))
        d_new_foll = 0.0
        if frow >= 0:
            v0_f = _seg_limit1(1, float(self.s[frow]), self.p)
            gap_foll = float(s_ego_t - self.length[r] - self.s[frow])
            a_f_after = self._aclip1(_idm1(
                v_foll, gap_foll, v_foll - float(self.v[r]), v0_f,
                self.idm_p))
            lo, _ = self._own_leader_follower(frow)
            if lo >= 0:
                cg = float(self.s[lo] - self.length[lo] - self.s[frow])
                cv = float(self.v[lo])
            else:
                cg, cv = GAP_MAX, 0.0
            a_f_before = self._aclip1(_idm1(
                v_foll, cg, v_foll - cv, v0_f, self.idm_p))
            d_new_foll = a_f_after - a_f_before
        d_old_foll = 0.0
        lo2, fo2 = self._own_leader_follower(r)
        if fo2 >= 0:
            v_of = float(self.v[fo2])
            gap_of = float(self.s[r] - self.length[r] - self.s[fo2])
            v0_o = _seg_limit1(2, float(self.s[fo2]), self.p)
            a_of_before = self._aclip1(_idm1(
                v_of, gap_of, v_of - float(self.v[r]), v0_o, self.idm_p))
            if lo2 >= 0:
                gap_ol = float(self.s[lo2] - self.length[lo2]
                               - self.s[fo2])
                v_ol = float(self.v[lo2])
            else:
                gap_ol, v_ol = GAP_MAX, 0.0
            a_of_after = self._aclip1(_idm1(
                v_of, gap_ol, v_of - v_ol, v0_o, self.idm_p))
            d_old_foll = a_of_after - a_of_before
        gain = (a_ego_new - self._a_stay(r)) + \
            float(self.mobil_p["politeness_p"]) * (d_new_foll + d_old_foll)
        thr = float(self.mobil_p["accel_threshold_mps2"]) * (1.0 - u)
        terms.update(a_ego_new=a_ego_new, d_new_foll=d_new_foll,
                     d_old_foll=d_old_foll, gain=gain, thr=thr,
                     a_stay=self._a_stay(r))
        return terms

    def _merge_decision(self, r, d_sorted1, rows1):
        accept, d_i = super()._merge_decision(r, d_sorted1, rows1)
        terms = self._merge_terms(r, d_sorted1, rows1)
        re_acc = terms["reason"] == "" and terms["gain"] >= terms["thr"]
        assert re_acc == accept, \
            f"trace mismatch merge: {re_acc} vs {accept} terms={terms}"
        self._log_decision(accept, r, 2, 1, "merge", terms)
        return accept, d_i

    def _log_decision(self, accept: bool, r: int, src: int, tgt: int,
                      kind: str, terms: dict) -> None:
        base = dict(world=self._world, w=self._w, step=self.step_i,
                    t=round(self.t0 + (self.step_i - 1) * self.dt, 3),
                    vid=self.names[r], src=int(src), tgt=int(tgt),
                    kind=kind,
                    s=round(float(self.s[r]), 2), is_cav=bool(self.cav[r]))
        if accept:
            self.accepted.append({**base, **terms})
        else:
            self.rejected.append({**base,
                                  "reason": terms.get("reason", "")
                                  or "threshold"})


def extract_events(eng: TraceEngine, snaps, vids) -> List[dict]:
    """Executed LC events from consecutive snapshots + local context."""
    out = []
    for j in range(1, len(snaps)):
        prev, cur = snaps[j - 1], snaps[j]
        for vid in vids:
            p, c = prev.get(vid), cur.get(vid)
            if p is None or c is None or p[0] < 0 or c[0] < 0 or \
                    p[0] == c[0]:
                continue
            # local context at the previous snapshot: same-lane count
            # within +-50 m, their mean speed, own-lane leader gap
            same = [(o[1] - p[1], o[2]) for o in prev.values()
                    if o is not p and o[0] == p[0]]
            near = [x for x in same if -50.0 <= x[0] <= 50.0]
            ahead = [x for x in same if x[0] > 0]
            out.append(dict(
                vid=vid, step=j, t=round(eng.t0 + j * DT, 3),
                f=p[0], t_=c[0], s=round(p[1], 2), v=round(p[2], 2),
                n_near=len(near),
                near_mean_v=round(float(np.mean([x[1] for x in near]))
                                  if near else float("nan"), 2),
                gap_ahead=round(min(ahead)[0] if ahead else float("nan"),
                                2),
                is_cav=vid in eng.cav0))
    return out


def merge_outcomes(eng: TraceEngine, snaps, vids, hs) -> List[dict]:
    """Per tracked vehicle on lane 2 at window start: merge time,
    stop-at-end, never-merged, dwell (entry->merge when entry visible)."""
    out = []
    s1z = float(eng.merge_p["zone_s1_m"])
    for vid in vids:
        p0 = snaps[0].get(vid)
        if p0 is None or p0[0] != 2:
            continue
        # find the global lane-2 entry step (first step on lane 2)
        entry_step = None
        for j in range(len(snaps)):
            st = snaps[j].get(vid)
            if st is not None and st[0] == 2:
                entry_step = j
                break
        merge_step = None
        stopped_step = None
        for j in range(1, len(snaps)):
            st = snaps[j].get(vid)
            if st is None:
                continue
            if st[0] == 1 and merge_step is None:
                merge_step = j
                break
            if st[0] == 2 and stopped_step is None and \
                    st[1] >= s1z - 0.51 and st[2] < 0.1:
                stopped_step = j
        out.append(dict(
            vid=vid, is_cav=vid in eng.cav0,
            s0=round(p0[1], 2), v0=round(p0[2], 2),
            entry_t=round(eng.t0 + entry_step * DT, 3)
            if entry_step is not None else None,
            merge_t=round(eng.t0 + merge_step * DT, 3)
            if merge_step is not None else None,
            stop_t=round(eng.t0 + stopped_step * DT, 3)
            if stopped_step is not None else None,
            merged_in={f"H{H}": bool(merge_step is not None and
                                     merge_step * DT <= H) for H in hs}))
    return out


def _worker(args):
    rank, entries, params = args
    EA.MAPPER = EA.ModelMapper()
    opened = set()
    EA.OPENED_RUN_FILES.clear()
    shard = RESULTS / f"pred_shard_{rank}"
    shard.mkdir(parents=True, exist_ok=True)
    paths = {k: open(shard / f"pred_{name}.jsonl", "w",
                     encoding="utf-8")
             for k, name in ((0, "events"), (1, "decisions_accepted"),
                             (2, "rejections"), (3, "window_rows"),
                             (4, "states"), (5, "merge_outcomes"))}
    n_roll = int(round(H_MAX / DT))
    stats = []
    for wi, entry in enumerate(entries):
        t_a = time.perf_counter()
        log = EA.NativeLog(entry)
        opened.update(EA.OPENED_RUN_FILES)
        EA.OPENED_RUN_FILES.clear()
        names = sorted(log.rows[0]) if 0 in log.rows else []
        n_w = 0
        for k in range(1, 19):
            w = log.t0 + 30.0 * k
            if w > log.t_end + 1e-9:
                break
            j0 = int(round((w - log.t0) / 0.5))
            truth_w = log.truth_at(w)
            vehicles, _free = EA.tracked_state(truth_w, params,
                                               inj_at=log.inj.get(j0))
            if not vehicles:
                continue
            vids = sorted(vehicles)
            cav_ids = set(vids) & log.cav_ids
            hook = EA2.native_control_hook(w, params, cav_ids)
            eng = TraceEngine(params, t0=w, vehicles=vehicles,
                              control_hook=hook)
            eng._world = entry["world"]
            eng._w = w
            eng.cav0 = cav_ids
            eng.run(n_roll)
            snaps = MX.snaps_list(eng)
            for e in extract_events(eng, snaps, vids):
                e["world"] = entry["world"]
                e["w"] = w
                paths[0].write(json.dumps(e) + "\n")
            for d in eng.accepted:
                paths[1].write(json.dumps(d) + "\n")
            tally: Dict[str, int] = {}
            for d in eng.rejected:
                key = f"{d['kind']}|{d['reason']}"
                tally[key] = tally.get(key, 0) + 1
            paths[2].write(json.dumps(dict(world=entry["world"], w=w,
                                           rejected_tally=tally)) + "\n")
            # per-H window LC counts (identical counting to truth)
            counts_by_H = {}
            for H in HS:
                if w + H > log.t_end + 1e-9:
                    continue
                counts = {"merge": 0, "disc_left": 0, "disc_right": 0}
                for j in range(1, int(round(H / DT)) + 1):
                    prev, cur = snaps[j - 1], snaps[j]
                    for vid in vids:
                        p, c = prev.get(vid), cur.get(vid)
                        if p is None or c is None or p[0] < 0 or \
                                c[0] < 0 or p[0] == c[0]:
                            continue
                        key = {(2, 1): "merge", (1, 0): "disc_left",
                               (0, 1): "disc_right"}.get((p[0], c[0]))
                        if key:
                            counts[key] += 1
                counts_by_H[f"H{H}"] = counts
            paths[3].write(json.dumps(dict(
                world=entry["world"], w=w, n_tracked=len(vids),
                n_cav=len(cav_ids), lc=counts_by_H)) + "\n")
            paths[4].write(json.dumps(dict(
                world=entry["world"], w=w,
                vids={v: [int(vehicles[v]["lane"]),
                          round(float(vehicles[v]["s"]), 2),
                          round(float(vehicles[v]["v"]), 2),
                          v in cav_ids] for v in vids})) + "\n")
            for mo in merge_outcomes(eng, snaps, vids, HS):
                mo["world"] = entry["world"]
                mo["w"] = w
                mo["at_w"] = True
                paths[5].write(json.dumps(mo) + "\n")
            n_w += 1
        stats.append(dict(world=entry["world"], windows=n_w,
                          wall_s=round(time.perf_counter() - t_a, 1)))
        print(f"  [w{rank}] [{wi + 1}/{len(entries)}] "
              f"{entry['world']}: {n_w} windows, "
              f"{time.perf_counter() - t_a:.1f}s", flush=True)
    for fh in paths.values():
        fh.close()
    return str(shard), sorted(opened), stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    n_workers = max(1, min(4, args.workers))
    t_start = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    worlds = json.loads(
        (HERE / "results" / "lc1_a2_worlds32.json").read_text())
    EA.OPENED_RUN_FILES.clear()
    EA.TRAIN_WORLD_NAMES.clear()
    plan = json.loads(EA2.PLAN.read_text(encoding="utf-8"))
    meta = {w["world"]: w for w in plan["worlds"]}
    entries = []
    for name in worlds:
        e = dict(meta[name])
        assert e.get("split") == "TRAIN"
        e["chunk"] = 4
        e["native_run"] = DATA_ROOT / "runs" / "campaign_stage1_chunk4" / \
            f"{name}__native_base" / "run" / "physics_steps.jsonl"
        entries.append(e)
    EA.TRAIN_WORLD_NAMES.update(worlds)
    print(f"{ANALYSIS_ID}: {len(entries)} TRAIN worlds, workers "
          f"{n_workers}", flush=True)
    params = load_params()
    if n_workers == 1:
        results = [_worker((0, entries, params))]
    else:
        ctx = mp.get_context("spawn")
        chunks = [entries[i::n_workers] for i in range(n_workers)]
        with ctx.Pool(n_workers) as pool:
            results = pool.map(_worker, [(i, chunks[i], params)
                                         for i in range(n_workers)])
    # merge shards in the assigned world order (rank -> worlds)
    world_rank = {}
    for rank, (_sh, _o, stats) in enumerate(results):
        for st in stats:
            world_rank[st["world"]] = rank
    names = ("events", "decisions_accepted", "rejections", "window_rows",
             "states", "merge_outcomes")
    shard_rows: Dict[str, Dict[str, list]] = {}
    for (sh, _o, _s) in results:
        d = {}
        for name in names:
            with open(Path(sh) / f"pred_{name}.jsonl",
                      encoding="utf-8") as fh:
                d[name] = fh.readlines()
        shard_rows[sh] = d
    outs = {name: open(RESULTS / f"pred_{name}.jsonl", "w",
                       encoding="utf-8") for name in names}
    n_lines = Counter()
    for entry in entries:
        rank = world_rank.get(entry["world"])
        if rank is None:
            continue
        sh = results[rank][0]
        for name in names:
            for line in shard_rows[sh][name]:
                r = json.loads(line)
                if r.get("world") == entry["world"]:
                    outs[name].write(line if line.endswith("\n")
                                     else line + "\n")
                    n_lines[name] += 1
    for fh in outs.values():
        fh.close()
    for sh, _o, _s in results:
        for name in names:
            (Path(sh) / f"pred_{name}.jsonl").unlink(missing_ok=True)
        Path(sh).rmdir()
    print("merged rows:", dict(n_lines), flush=True)
    opened_all = sorted(set().union(*[set(o) for _s, o, _st in results]))
    EA.OPENED_RUN_FILES.update(opened_all)
    EA.assert_train_only()
    (RESULTS / "pred_opened_run_files.json").write_text(
        json.dumps(opened_all, indent=1), encoding="utf-8")
    wall = time.perf_counter() - t_start
    (RESULTS / "pred_run_meta.json").write_text(json.dumps(dict(
        analysis_id=ANALYSIS_ID, worlds=len(entries),
        workers=n_workers, wall_clock_s=wall,
        tracking="WD-01-PA2G-LC1 class-(ii) fix applied "
                 "(tracked_state inj_at)",
        engine="TraceEngine(PartAV2GEngine) terms asserted vs engine",
        merged_rows=dict(n_lines),
        stats=[s for _sh, _o, ss in results for s in ss])),
        encoding="utf-8")
    print("DONE", ANALYSIS_ID, f"wall {wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
