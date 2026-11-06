"""PLANGEN analysis (derived from newmap2 nm2_analyze.py sha b0074755; verdict/admission logic unchanged):
per-variant funnel verdicts, per-cell table, B/R example rows and negatives, plus RSU-range tags.

    python pg_analyze.py OUT_DIR

Writes OUT_DIR/{PG_SUMMARY.json, PLANGEN_MOVES_B.jsonl, PLANGEN_MOVES_R.jsonl, PLANGEN_NEGATIVES.jsonl}.
Admission rule: see REGISTRATION.md PLANGEN block (= NEWMAP2 rule incl. amendments 3-4).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import pg_plan as P

PHASE_ID = "PLANGEN-20261106"
X_B = P.X_B
# RSU zones of the registered wz3to2 map descriptor (c4/nm-abstract 65d5f619 tools/coopmwm/map_descriptor.py:
# frame s = x - (X_B - 75.44); RSU block zone s in [-100, 75.44] closing lane / [-100, 175.44] through lanes;
# frozen MAPPO RSU field cells s in [-378.66, 75.44] closing lane / [-800, 225.44] through lanes).
RSU_RULE = ("wz3to2 RSU (source cav_reports): block zone x in [X_B-175.44, X_B] (closing lane 0) / "
            "[X_B-175.44, X_B+100] (through); RSU field cells x in [X_B-454.10, X_B] (closing) / "
            "[X_B-875.44, X_B+150] (through); X_B=2696 m; member in range = its position inside the zone")
_RSU = dict(block=((X_B - 175.44, X_B), (X_B - 175.44, X_B + 100.0)),
            field=((X_B - 454.10, X_B), (X_B - 875.44, X_B + 150.0)))


def _in_rsu(kind, edge, lane, x):
    closing = edge == "up" and int(lane) == 0
    lo, hi = _RSU[kind][0 if closing else 1]
    return lo <= float(x) <= hi

LANE_W = 3.2
STAT = ("per-replicate one-sided 95% circular-block (30 s) paired bootstrap B=2000 RNG 20261103 of the "
        "per-second difference on [t0,t0+300); S = best solo (re-chosen per replicate) minus group, "
        "T = base minus group; replicate 1 = undithered, 2/3 = common CRN dither on a non-member CAV")


def dist(a, b):
    return math.hypot(a[0] - b[0], (a[1] - b[1]) * LANE_W)


def v2v(m, vid):
    v = m["variants"][vid]
    pts = {mm["id"]: (mm["x"], mm["lane"]) for mm in v["members"]}
    ids = list(pts)
    d0 = max((dist(pts[a], pts[b]) for i, a in enumerate(ids) for b in ids[i + 1:]), default=0.0)
    # 4-slot observability at t0: some member (ego) has every other member among its K=4 nearest CAVs <=100 m
    allc = [(c["id"], (c["x"], c["lane"])) for c in m["all_cavs_t0"] if c["edge"] == "up"]
    obs_egos = []
    for ego in ids:
        near = sorted(((dist(pts[ego], p), cid) for cid, p in allc if cid != ego))
        slots = {cid for dd, cid in near[:4] if dd <= 100.0}
        if all(o in slots for o in ids if o != ego):
            obs_egos.append(ego)
    # max pairwise distance while held (group r0 tracks)
    g = P.load(P.mdir(m) / "r0" / f"g__{vid}")
    dh = None
    if g is not None:
        by = defaultdict(dict)                       # distances among members still held at second el
        for mid in ids:
            rel_el = g["release_el"].get(mid)
            for row in g["tracks"].get(mid, []):
                el, edge, lane, x = row[0], row[1], row[2], row[3]
                if rel_el is None or el < rel_el:
                    by[el][mid] = (x, lane)
        dh = 0.0
        for el, d in by.items():
            if len(d) >= 2:
                k = list(d)
                dh = max(dh, max(dist(d[a], d[b]) for i, a in enumerate(k) for b in k[i + 1:]))
    # RSU range: members inside the RSU block zone / field cells at t0 and while all members are held
    in_f0 = {mm["id"]: _in_rsu("field", "up", mm["lane"], mm["x"]) for mm in v["members"]}
    in_b0 = {mm["id"]: _in_rsu("block", "up", mm["lane"], mm["x"]) for mm in v["members"]}
    held_s = held_all_f = 0
    any_f_held, first_el = False, None
    if g is not None:
        rows_by = defaultdict(dict)
        for mid in ids:
            rel_el = g["release_el"].get(mid)
            for row in g["tracks"].get(mid, []):
                if rel_el is None or row[0] < rel_el:
                    rows_by[row[0]][mid] = _in_rsu("field", row[1], row[2], row[3])
        for el in sorted(rows_by):
            d = rows_by[el]
            if any(d.values()):
                any_f_held = True
                first_el = el if first_el is None else first_el
            if len(d) == len(ids):
                held_s += 1
                held_all_f += all(d.values())
    rsu = dict(rule=RSU_RULE, in_field_t0=in_f0, in_block_t0=in_b0, all_in_field_t0=all(in_f0.values()),
               any_in_field_t0=any(in_f0.values()), any_in_block_t0=any(in_b0.values()),
               any_in_field_while_held=any_f_held, first_el_any_in_field_held=first_el,
               n_s_all_held=held_s, n_s_all_held_all_in_field=held_all_f)
    return dict(max_dist_t0_m=round(d0, 1), within_100_t0=d0 <= 100, within_200_t0=d0 <= 200,
                within_300_t0=d0 <= 300, max_dist_hold_m=None if dh is None else round(dh, 1),
                within_100_hold=None if dh is None else dh <= 100,
                within_200_hold=None if dh is None else dh <= 200,
                within_300_hold=None if dh is None else dh <= 300,
                observable_4slot_t0=bool(obs_egos), observable_egos=obs_egos, rsu_range=rsu,
                v2v100_or_rsu_field_t0=bool(d0 <= 100 or rsu["all_in_field_t0"]))


def verdict(m, vid):
    """Funnel verdict of one (moment, variant)."""
    v = m["variants"][vid]
    out = dict(sid=m["sid"], cell=m["cell"], demand=m["demand"], penetration=m["penetration"], seed=m["seed"],
               t0=m["t0"], window=m["window"], variant=vid, family=v["family"],
               arm="R" if v["family"] == "R_random" else "B", members=[mm["id"] for mm in v["members"]],
               member_lanes=[mm["lane"] for mm in v["members"]], params=v["params"],
               approach_speed_t0=m["approach_speed"], closing_lane_slow_t0=m.get("closing_lane_slow"))
    e0 = P.rep_eval(m, vid, 0)
    if e0 is None:
        out.update(stage="screen_missing", admitted=False, reason="screen arm missing/failed")
        return out
    out.update(T_r0=e0["T"], T_lb_r0=e0["T_lb"], J_base_r0=e0["J_base"], J_group_r0=e0["J_group"],
               safety_r0=e0["group"], base_safety_r0=e0["base_safety"])
    if "S" not in e0:
        out.update(stage="screen", admitted=False,
                   reason=("group not better than base on r0 (LB95(T)<=0 or T<1)" if not (e0["T_lb"] > 0 and e0["T"] >= 1)
                           else ("not among the top-2 T random proposals at this moment (solos not run; "
                                 "count-matched to B's <=2 solo sets)" if v["family"] == "R_random" else
                                 "not the best-T variant of its family at this moment (solos not run)")))
        return out
    out.update(S_r0=e0["S"], S_lb_r0=e0["S_lb"], best_solo_r0=e0["best_solo"], solo_gain_r0=e0["solo_gain"])
    if not P.r0_candidate(m, vid):
        out.update(stage="solos", admitted=False,
                   reason="r0 synergy not admitted (LB95(S)<=0 or S<1)" if e0["safe"] else "r0 group unsafe")
        return out
    if vid not in P.rep1_picks(m):
        out.update(stage="r0_not_picked", admitted=False,
                   reason="r0 synergy candidate, but another variant of the same arm had a larger LB95(S) at this "
                          "moment (amendment 3: one replicated candidate per moment and arm)")
        return out
    reps = {0: e0}
    for rep in (1, 2):
        e = P.rep_eval(m, vid, rep)
        if e is None or "S" not in e:
            out.update(stage=f"r{rep}_missing", admitted=False,
                       reason=f"replicate {rep} not run (earlier replicate failed or budget)", reps=_reps(reps))
            return out
        reps[rep] = e
        if not (e["T_lb"] > 0 and e["S_lb"] > 0 and e["S"] >= 1 and e["safe"]):
            out.update(stage=f"r{rep}", admitted=False, reason=f"replicate {rep} main-window test failed",
                       reps=_reps(reps))
            return out
    el0 = P.rep_eval(m, vid, 0, n=P.H_LONG, need_solos=False)      # amendment 4: later harm on T (advantage)
    e0["T_ext"], e0["S_ext"] = (el0["T"], None) if el0 else (None, None)
    out["reps"] = _reps(reps)
    if el0 is None:
        out.update(stage="r0ext_missing", admitted=False, reason="r0 later-window extension missing")
        return out
    if not el0["T"] > 0:
        out.update(stage="r0ext", admitted=False, reason="r0 later-harm check failed (T over [t0,t0+420) <= 0)")
        return out
    out["move_dither"] = None                                          # amendment 4: not run (budget)
    out.update(stage="admitted", admitted=True, reason="")
    return out


def _reps(reps):
    return {str(k + 1): dict(T=e["T"], T_lb=e["T_lb"], S=e.get("S"), S_lb=e.get("S_lb"), best_solo=e.get("best_solo"),
                             T_ext=e.get("T_ext"), S_ext=e.get("S_ext"), safe=e["safe"], group=e["group"],
                             base_safety=e["base_safety"], J_base=e["J_base"], J_group=e["J_group"],
                             solo_gain=e.get("solo_gain"))
            for k, e in sorted(reps.items())}


def accel_series(track, n):
    """Per-second measured acceleration a[k] = v(k+1) - v(k) from a whole-second track."""
    v = {row[0]: row[4] for row in track}
    return [float(v[k + 1] - v[k]) if (k in v and k + 1 in v) else 0.0 for k in range(n)]


def profiles(m, vid, rep):
    root = P.mdir(m) / f"r{rep}"
    g, b = P.load(root / f"g__{vid}"), P.load(root / "base")
    ids = [mm["id"] for mm in m["variants"][vid]["members"]]
    rel = [x for x in g["release_el"].values() if x is not None]
    dur = int(max(rel)) if rel else int(max(len(g["issued"].get(i, [])) for i in ids))
    dur = max(dur, 1)
    prof, pol = [], []
    for i in ids:
        iss = {el: a for el, a in g["issued"].get(i, [])}
        meas = accel_series(g["tracks"].get(i, []), dur)
        prof.append([float(iss.get(k, meas[k])) for k in range(dur)])
        pol.append(accel_series(b["tracks"].get(i, []), dur))
    pert = [[round(p - q, 6) for p, q in zip(a, c)] for a, c in zip(prof, pol)]
    return dur, prof, pol, pert, g["release_el"], g["arm_events"]


def example_rows(m, vd):
    vid = vd["variant"]
    v = m["variants"][vid]
    plan = dict(map="wz3to2", variant=vid, family=v["family"],
                members=[dict(id=mm["id"], lane=mm["lane"], params=mm["params"]) for mm in v["members"]])
    psha = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    reps = vd["reps"]
    Ss = [reps[k]["S"] for k in sorted(reps)]
    mean = float(np.mean(Ss))
    sd = float(np.std(Ss, ddof=1)) if len(Ss) > 1 else float("nan")
    ci = [mean - 4.303 * sd / math.sqrt(len(Ss)), mean + 4.303 * sd / math.sqrt(len(Ss))]
    vv = v2v(m, vid)
    rows = []
    for seed in sorted(reps, key=int):
        rep = int(seed) - 1
        dur, prof, pol, pert, rel, ev = profiles(m, vid, rep)
        r = reps[seed]
        rows.append(dict(
            schema="coopmwm-mined-move-v1", arm=vd["arm"], run_root=str(P.mdir(m)),
            verification_id=f"pgv_{m['sid']}_{vid}_{psha[:8]}", plan_sha256=psha, replay_sources={},
            seed=int(seed), situation_id=m["sid"],
            world=dict(map="wz3to2", cell=m["cell"], demand=m["demand"], penetration=m["penetration"],
                       seed=m["seed"], repeat=0, split="TRAIN 989203-989349 (plangen)"),
            moment=dict(situation_id=m["sid"], world_seed=m["seed"], t0=m["t0"], window=m["window"],
                        window_s=list(P.WINDOWS[m["window"]]), cell=m["cell"]),
            t0=m["t0"], family=v["family"], variant=vid, members=[mm["id"] for mm in v["members"]],
            member_lanes=[mm["lane"] for mm in v["members"]], duration_s=dur, profile=prof, policy_profile=pol,
            perturbation_profile=pert,
            synergy_veh_s={seed: r["S"]}, synergy_lower95={seed: r["S_lb"]},
            advantage_veh_s={seed: r["T"]}, adv_lower95={seed: r["T_lb"]}, best_solo=r["best_solo"],
            solo_gain_veh_s=r["solo_gain"], n_reps=len(reps), statistic=STAT,
            base="native CAVs (cav_typical, speedMode 27, SUMO LC2013; no MAPPO)",
            source="plangen 2026-11-06 (owner 15:14 phase plangen; N3 newmap2 pipeline, R count-matched; "
                   "SUMO-only, matched CRN branches from saved native state)",
            host="cpuV", observable=vv["observable_4slot_t0"], v2v_range=vv, rsu_range=vv["rsu_range"],
            v2v100_or_rsu_field_t0=vv["v2v100_or_rsu_field_t0"],
            observability=dict(rule="ego + K=4 nearest CAVs within 100 m at t0 (V2V_OBS_READY layout)",
                               egos_t0=vv["observable_egos"], observable_t0=vv["observable_4slot_t0"]),
            move=plan, release_el=rel, arm_events=ev,
            dither=P.dither_spec(m, rep), later_harm=dict(window="[t0,t0+420) on replicate 1 (undithered)",
                                                         T_ext_420=reps["1"]["T_ext"], S_ext_420=reps["1"]["S_ext"]),
            move_dither=vd["move_dither"], safety=dict(group=r["group"], base=r["base_safety"]),
            replicate_summary=dict(S=Ss, mean=mean, t95=ci, T=[reps[k]["T"] for k in sorted(reps)]),
            situation=dict(approach_speed_t0=m["approach_speed"], closing_lane_slow_t0=m.get("closing_lane_slow"),
                           n_zone_cavs=m["n_zone_cavs"]),
            restart=dict(native_state=str(P.native_dir(m["demand"], m["penetration"], m["seed"]) / "states" /
                                          f"t{m['t0']}.xml.gz"), t0=m["t0"], inserted_t0=m["inserted_t0"],
                         note="seed 1 = load state, act; seeds 2/3 = same plus the listed common dither"),
            cell=m["cell"], new_situation=True, near_duplicate_of=None, newmap2_id=PHASE_ID, phase="plangen",
            binding_note="group_lane_bank.binding() world-seed range check (986000-986071) must be extended "
                         "for wz3to2 TRAIN 989203-989349 before ingestion (N4); N4 re-verifies on the final base"))
    return rows


def ci_t(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    if len(v) == 0:
        return dict(n=0)
    m = float(v.mean())
    if len(v) < 2:
        return dict(n=1, mean=m)
    se = float(v.std(ddof=1) / math.sqrt(len(v)))
    return dict(n=len(v), mean=m, t95=[m - 2.0 * se, m + 2.0 * se] if len(v) > 30 else
                [m - {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31}.get(len(v), 2.2) * se,
                 m + {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36, 9: 2.31}.get(len(v), 2.2) * se],
                median=float(np.median(v)), n_pos=int((v > 0).sum()))


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    allm = json.loads((P.BASE / "MOMENTS.json").read_text())
    nworlds = {P.cell_key(d, p): sum(1 for w in P.worlds() if w[:2] == (d, p)) for d, p in P.ALLOWED_CELLS}
    verdicts, B, R, NEG = [], [], [], []
    for m in allm:
        if not m["valid"]:
            continue
        for vid in m["variants"]:
            vd = verdict(m, vid)
            vd["v2v"] = v2v(m, vid)
            verdicts.append(vd)
            if vd["admitted"]:
                (B if vd["arm"] == "B" else R).extend(example_rows(m, vd))
            else:
                NEG.append(dict(schema="newmap2-negative-v1", arm=vd["arm"], situation_id=m["sid"],
                                world=dict(map="wz3to2", cell=m["cell"], demand=m["demand"],
                                           penetration=m["penetration"], seed=m["seed"]),
                                t0=m["t0"], family=vd["family"], variant=vid, members=vd["members"],
                                member_lanes=vd["member_lanes"], params=vd["params"], admitted=False,
                                status="rejected", stage=vd["stage"], reason=vd["reason"],
                                advantage_veh_s=vd.get("T_r0"), adv_lower95=vd.get("T_lb_r0"),
                                synergy_veh_s=vd.get("S_r0"), synergy_lower95=vd.get("S_lb_r0"),
                                reps=vd.get("reps"), move_dither=vd.get("move_dither"),
                                safety=vd.get("safety_r0"), observable=vd["v2v"]["observable_4slot_t0"],
                                v2v_range=vd["v2v"], rsu_range=vd["v2v"]["rsu_range"], host="cpuV",
                                newmap2_id=PHASE_ID, phase="plangen"))
    cells = []
    for d, p in sorted(P.ALLOWED_CELLS, key=lambda c: -c[1]):
        if True:
            ck = P.cell_key(d, p)
            ms = [m for m in allm if m["cell"] == ck]
            vds = [v for v in verdicts if v["cell"] == ck]
            row = dict(cell=ck, demand=d, penetration=p, n_worlds=nworlds[ck], n_window_moments=len(ms),
                       n_valid=sum(m["valid"] for m in ms),
                       n_pair=sum(1 for m in ms if m["valid"] and "pair_g" in m["variants"]),
                       n_wall=sum(1 for m in ms if m["valid"] and "wall_g" in m["variants"]),
                       approach_speed_t0=[m.get("approach_speed") for m in ms])
            for arm in ("B", "R"):
                adm = {v["sid"] for v in vds if v["arm"] == arm and v["admitted"]}
                row[f"{arm}_moments_admitted"] = len(adm)
                row[f"{arm}_frac_window"] = len(adm) / max(1, len(ms))
                row[f"{arm}_frac_valid"] = len(adm) / max(1, row["n_valid"])
            for fam in ("closing_pair", "abreast_wall", "R_random"):
                fv = [v for v in vds if v["family"] == fam]
                row[fam] = dict(
                    n=len(fv), n_T_pass=sum(1 for v in fv if v.get("T_lb_r0", -1) > 0 and v.get("T_r0", 0) >= 1),
                    n_solos=sum(1 for v in fv if "S_r0" in v),
                    n_r0_syn=sum(1 for v in fv if v["stage"] not in ("screen", "solos", "screen_missing")),
                    n_admitted=sum(v["admitted"] for v in fv),
                    T_r0=ci_t([v.get("T_r0") for v in fv]),
                    S_r0=ci_t([v.get("S_r0") for v in fv]),
                    S_admitted_mean=ci_t([np.mean([r["S"] for r in v["reps"].values()]) for v in fv if v["admitted"]]),
                    stages={s: sum(1 for v in fv if v["stage"] == s) for s in sorted({v["stage"] for v in fv})})
            vs = [v for v in vds]
            row["safety"] = dict(
                group_collisions=sum(v.get("safety_r0", {}).get("collisions", 0) for v in vs),
                group_teleports=sum(v.get("safety_r0", {}).get("teleports", 0) for v in vs),
                group_hard_brake_vs_mean=ci_t([v.get("safety_r0", {}).get("hard_brake_vs") for v in vs]).get("mean"),
                base_hard_brake_vs_mean=ci_t([v.get("base_safety_r0", {}).get("hard_brake_vs") for v in vs]).get("mean"),
                group_min_accel=min([v.get("safety_r0", {}).get("min_accel", 0) for v in vs] or [0]),
                group_emergency_steps=sum(v.get("safety_r0", {}).get("emergency_steps", 0) for v in vs))
            adm = [v for v in vds if v["admitted"]]
            row["admitted_v2v"] = dict(n=len(adm), within_100_t0=sum(v["v2v"]["within_100_t0"] for v in adm),
                                       within_200_t0=sum(v["v2v"]["within_200_t0"] for v in adm),
                                       within_300_t0=sum(v["v2v"]["within_300_t0"] for v in adm),
                                       within_100_hold=sum(bool(v["v2v"]["within_100_hold"]) for v in adm),
                                       within_200_hold=sum(bool(v["v2v"]["within_200_hold"]) for v in adm),
                                       within_300_hold=sum(bool(v["v2v"]["within_300_hold"]) for v in adm),
                                       observable_4slot=sum(v["v2v"]["observable_4slot_t0"] for v in adm),
                                       rsu_all_in_field_t0=sum(v["v2v"]["rsu_range"]["all_in_field_t0"] for v in adm),
                                       rsu_any_in_field_held=sum(v["v2v"]["rsu_range"]["any_in_field_while_held"]
                                                                 for v in adm),
                                       v2v100_or_rsu_field_t0=sum(v["v2v"]["v2v100_or_rsu_field_t0"] for v in adm))
            cells.append(row)
    (out / "PG_SUMMARY.json").write_text(json.dumps(dict(cells=cells, verdicts=verdicts), indent=1, default=str))
    for name, rows in (("PLANGEN_MOVES_B.jsonl", B), ("PLANGEN_MOVES_R.jsonl", R), ("PLANGEN_NEGATIVES.jsonl", NEG)):
        (out / name).write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    for c in cells:
        print(c["cell"], "win", c["n_window_moments"], "valid", c["n_valid"], "pair", c["n_pair"], "wall", c["n_wall"],
              "B_adm", c["B_moments_admitted"], "R_adm", c["R_moments_admitted"],
              {f: (c[f]["n"], c[f]["n_T_pass"], c[f]["n_solos"], c[f]["n_r0_syn"], c[f]["n_admitted"])
               for f in ("closing_pair", "abreast_wall", "R_random")})
    print("B rows", len(B), "R rows", len(R), "negatives", len(NEG))


if __name__ == "__main__":
    main(sys.argv[1])
