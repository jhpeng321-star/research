"""PLANGEN staged job planner + shared statistics (wz3to2, phase plangen).  Derived from newmap2
nm2_plan.py (sha 5ef8eb9d); decision rules unchanged except the registered PLANGEN differences:
  * worlds/cells come from BASE/WORLDS.json (3200 x {0.6, 0.4}, TRAIN 989203-989349 only);
  * R is count-matched to B per moment: N_R = 5 random proposals (B screens 5 variants), solos for at
    most 2 R proposals (top-2 T; B: best-T per family, i.e. at most 2), 1 replicated candidate per arm;
  * --fit RESERVE: whole moment units are planned only if they fit the remaining cap (no partial funnels).

    python pg_plan.py natives  OUT.jsonl
    python pg_plan.py screen   OUT.jsonl          (moments + B variants + R proposals, r0, H=300)
    python pg_plan.py solos    OUT.jsonl          (best-T B pair / B wall, top-2 R per moment with LB95(T)>0)
    python pg_plan.py rep1     OUT.jsonl          (r0 synergy candidates -> replicate 1, H=300)
    python pg_plan.py rep2     OUT.jsonl          (r1 survivors -> replicate 2 + r0 extension)

Registered rules (REGISTRATION.md NEWMAP2 block, before physics): geometry-only moment/member selection;
admission = every replicate LB95(T)>0 & LB95(S)>0 & S>=1 veh-s on [t0,t0+300); later harm: T and S
over [t0,t0+420) > 0 in every replicate; move-dither robustness LB95(T_md)>0 & LB95(S_md)>0; group arms
collision/teleport-free.  S = min_i J(solo_i) - J(group) (no zero floor), T = J(base) - J(group).
"""
from __future__ import annotations

import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
MAPSPEC = BASE / "maps" / "wz3to2" / "MAP_SPEC.json"
RUNS = BASE / "runs"
X_B = 2696.0
WORLDS_FILE = BASE / "WORLDS.json"           # [{demand, penetration, seed, wave}], appended per wave
ALLOWED_CELLS = {(3200, 0.6), (3200, 0.4)}
SEED_RANGE = (989203, 989349)                 # MAIN: 989350-989399 reserved for the low-CAV-share probe
LEDGER = BASE / "ledger" / "CHARGES.jsonl"
CAP = 308400.0                                # amendment 1 (MAIN top-up 8,400 for stranded r2)
WINDOWS = [(450, 550), (800, 900)]
SAVE_TIMES = [t for a, b in WINDOWS for t in range(a, b + 1, 10)]
T_END = 900
ZONE = (X_B - 1800.0, X_B - 400.0)
PAIR_DX = 100.0
WALL_SPREAD = 200.0
R_SPREAD = 300.0
H_MAIN, H_LONG, H_EXT = 300, 420, 120
H_REP = 300                                    # amendment 3: replicates r1/r2 main window only
REL = dict(release_x=X_B, amax=1.0)
VARIANTS = {
    "pair_g": ("closing_pair", dict(v_w=22.0, hold_s=120, dmax=1.0)),
    "pair_h": ("closing_pair", dict(v_w=15.0, hold_s=90, dmax=1.5)),
    "pair_t": ("closing_pair", dict(v_w=22.0, hold_s=150, dmax=1.0, anchor_release_x=X_B - 150.0)),
    "wall_g": ("abreast_wall", dict(v_w=22.0, hold_s=120, dmax=1.0)),
    "wall_h": ("abreast_wall", dict(v_w=15.0, hold_s=90, dmax=1.5)),
    "wall_l": ("abreast_wall", dict(v_w=18.0, hold_s=180, dmax=1.0)),
}
R_GRID = dict(v_w=[15.0, 18.0, 22.0], hold_s=[90, 120, 180], dmax=[1.0, 1.5])
N_R = 5                                        # PLANGEN: count-matched to the 5 B variants
N_SOLO_R = 2                                   # PLANGEN: B runs <=2 solo sets (one per family)
DITHER = {1: dict(a=-1.0, start_s=1, dur_s=2), 2: dict(a=-1.0, start_s=3, dur_s=2)}
MOVE_DITHER = dict(dv=-1.0, start_s=2)
B_BOOT, L_BLOCK, RNG, ALPHA = 2000, 30, 20261103, 0.05
MATERIAL = 1.0


def cell_key(d, p):
    return f"D{d}_p{int(round(p * 100)):02d}"


def native_dir(d, p, s):
    return RUNS / cell_key(d, p) / f"s{s}" / "native"


def worlds():
    out = []
    for w in json.loads(WORLDS_FILE.read_text()):
        d, p, s = int(w["demand"]), float(w["penetration"]), int(w["seed"])
        assert (d, p) in ALLOWED_CELLS and SEED_RANGE[0] <= s <= SEED_RANGE[1], w
        out.append((d, p, s))
    assert len(set(out)) == len(out)
    return out


def load(path: Path):
    f = path / "RESULT.json"
    if not f.exists():
        return None
    r = json.loads(f.read_text())
    return r if r.get("status") == "ok" else None


# ----------------------------------------------------------------------------- statistics
def boot_lb(x):
    """30 s circular-block paired bootstrap of the summed per-second difference, one-sided 95% LB
    (bank_statistics.paired_bootstrap per_seed_v1 convention, as in newmap_probe/analyze.py)."""
    x = np.asarray(x, float)
    n = len(x)
    rng = np.random.default_rng(RNG)
    lb = int(max(1, min(L_BLOCK, n // 2)))
    nb = -(-n // lb)
    starts = rng.integers(0, n, (B_BOOT, nb))
    idx = ((starts[:, :, None] + np.arange(lb)[None, None, :]) % n).reshape(B_BOOT, nb * lb)[:, :n]
    tot = x[idx].sum(1)
    return float(np.quantile(tot, ALPHA))


def series(r, n=None):
    s = np.asarray(r["J_road"], float) + np.asarray(r["J_src"], float)
    return s if n is None else s[:n]


# ----------------------------------------------------------------------------- selection (geometry only)
def zone_cavs(snap):
    return [c for c in snap["cavs"] if c["edge"] == "up" and ZONE[0] <= c["x"] <= ZONE[1]]


def pick_pair(snap):
    z = zone_cavs(snap)
    best = None
    for c0 in (c for c in z if c["lane"] == 0):
        for c1 in (c for c in z if c["lane"] == 1):
            dx = abs(c0["x"] - c1["x"])
            if dx <= PAIR_DX:
                key = (dx, -c0["x"])
                if best is None or key < best[0]:
                    best = (key, [c0, c1])
    return None if best is None else best[1]


def pick_wall(snap):
    z = zone_cavs(snap)
    lanes = [[c for c in z if c["lane"] == ln] for ln in (0, 1, 2)]
    if any(not x for x in lanes):
        return None
    best = None
    for combo in itertools.product(*lanes):
        xs = [c["x"] for c in combo]
        sp = max(xs) - min(xs)
        if sp <= WALL_SPREAD:
            key = (sp, -max(xs))
            if best is None or key < best[0]:
                best = (key, list(combo))
    return None if best is None else best[1]


def r_proposals(snap, sid, exclude_sets):
    z = zone_cavs(snap)
    out = []
    for i in range(N_R):
        rng = random.Random(f"nm2_R_v1:{sid}:{i}")
        k = rng.choice([2, 3])
        prop = None
        for kk in (k, 2):
            if len(z) < kk:
                continue
            for _ in range(400):
                mem = rng.sample(z, kk)
                xs = [c["x"] for c in mem]
                ids = frozenset(c["id"] for c in mem)
                if max(xs) - min(xs) <= R_SPREAD and ids not in exclude_sets:
                    prop = mem
                    break
            if prop:
                break
        if not prop:
            continue
        exclude_sets.add(frozenset(c["id"] for c in prop))
        params = dict(v_w=rng.choice(R_GRID["v_w"]), hold_s=rng.choice(R_GRID["hold_s"]),
                      dmax=rng.choice(R_GRID["dmax"]))
        out.append((f"R{i}", prop, params))
    return out


def dither_targets(snap, members_union, sid):
    nm = [c for c in snap["cavs"] if c["edge"] == "up" and c["id"] not in members_union and c["x"] < X_B]
    if not nm:
        return {}
    d1 = max(nm, key=lambda c: c["x"])
    near = [c for c in nm if X_B - 1000 <= c["x"] and c["id"] != d1["id"]]
    pool = near or [c for c in nm if c["id"] != d1["id"]] or [d1]
    d2 = random.Random(f"nm2_dither_v1:{sid}:2").choice(sorted(pool, key=lambda c: c["id"]))
    return {1: d1["id"], 2: d2["id"]}


def member_spec(c, params, anchor=None):
    p = dict(REL)
    p.update(params)
    if "anchor_release_x" in p:
        p["anchor"] = anchor
    return dict(id=c["id"], role="hold", params=p, lane=c["lane"], x=c["x"], v=c["v"])


def moments():
    """All window-moments (geometry only) with their B variants, R proposals and dither targets."""
    out = []
    if True:
        for (d, p, s) in worlds():
            if True:
                nat = load(native_dir(d, p, s))
                if nat is None:
                    continue
                for wi, (a, b) in enumerate(WINDOWS):
                    times = list(range(a, b + 1, 10))
                    info = {t: (pick_pair(nat["snapshots"][f"t{t}"]), pick_wall(nat["snapshots"][f"t{t}"]))
                            for t in times}
                    t0 = next((t for t in times if info[t][0] and info[t][1]), None)
                    t0 = t0 or next((t for t in times if info[t][0]), None)
                    t0 = t0 or next((t for t in times if info[t][1]), None)
                    row = dict(cell=cell_key(d, p), demand=d, penetration=p, seed=s, window=wi,
                               t0=t0, valid=t0 is not None)
                    if t0 is None:
                        snap = nat["snapshots"][f"t{a}"]
                        row.update(n_zone_cavs=len(zone_cavs(snap)), approach_speed=snap["approach_speed"])
                        out.append(row)
                        continue
                    snap = nat["snapshots"][f"t{t0}"]
                    sid = f"pg_{cell_key(d, p)}_s{s}_t{t0}"
                    pair, wall = info[t0]
                    variants = {}
                    if pair:
                        for vid in ("pair_g", "pair_h", "pair_t"):
                            fam, prm = VARIANTS[vid]
                            variants[vid] = dict(family=fam, params=prm,
                                                 members=[member_spec(c, prm, anchor=pair[0]["id"]) for c in pair])
                    if wall:
                        for vid in ("wall_g", "wall_h"):            # wall_l dropped at amendment 1 (budget)
                            fam, prm = VARIANTS[vid]
                            variants[vid] = dict(family=fam, params=prm, members=[member_spec(c, prm) for c in wall])
                    excl = {frozenset(c["id"] for c in x) for x in (pair, wall) if x}
                    for vid, mem, prm in r_proposals(snap, sid, excl):
                        variants[vid] = dict(family="R_random", params=prm, members=[member_spec(c, prm) for c in mem])
                    union = sorted({m["id"] for v in variants.values() for m in v["members"]})
                    dith = dither_targets(snap, set(union), sid)
                    row.update(sid=sid, inserted_t0=snap["inserted"], approach_speed=snap["approach_speed"],
                               closing_lane_slow=snap.get("closing_lane_slow"), n_zone_cavs=len(zone_cavs(snap)),
                               variants=variants, union=union, dither_ids=dith,
                               all_cavs_t0=snap["cavs"])
                    out.append(row)
    return out


# ----------------------------------------------------------------------------- jobs
def mdir(m):
    return RUNS / m["cell"] / f"s{m['seed']}" / f"m_t{m['t0']}"


def bjob(m, rep, arm, members, H, save_at=(), dither=None, extra=None):
    d, p, s, t0 = m["demand"], m["penetration"], m["seed"], m["t0"]
    nd = native_dir(d, p, s)
    j = dict(kind="branch", job_id=f"{m['sid']}_r{rep}_{arm}", map="wz3to2", map_spec=str(MAPSPEC), seed=s,
             demand={"main": d}, penetration=p, t0=t0, t_load=t0, state_file=str(nd / "states" / f"t{t0}.xml.gz"),
             inserted_load=m["inserted_t0"], native_dir=str(nd), H=H, members=members,
             dither=dither or [], save_at=list(save_at), track_ids=m["union"] if arm == "base" else [],
             track_s=300 if arm == "base" else 200, sid=m["sid"], rep=rep, arm=arm,
             out_dir=str(mdir(m) / f"r{rep}" / arm))
    if extra:
        j.update(extra)
    return j


def dither_spec(m, rep):
    if rep == 0:
        return []
    vid = m["dither_ids"].get(rep) if m["dither_ids"] else None
    if vid is None:
        vid = m["dither_ids"].get(str(rep)) if m["dither_ids"] else None
    if vid is None:
        return []
    return [dict(id=vid, role="pulse", params=dict(DITHER[rep]))]


def charged():
    if not LEDGER.exists():
        return 0.0
    return sum(float(json.loads(x)["planned_sim_s"]) for x in LEDGER.read_text().splitlines() if x.strip())


def jcost(r):
    return float(r["t_end"] if r["kind"] == "native" else r["H"])


def fit(rows, reserve=0.0):
    """Keep rows not yet run; then whole units (one moment sid, or one native) in planned order while the
    unit fits under CAP - charged - reserve.  A unit that does not fit is skipped and receipted."""
    todo = [r for r in rows if not (Path(r["out_dir"]) / "RESULT.json").exists()]
    left = CAP - charged() - reserve
    units, order = {}, []
    for r in todo:
        k = r.get("sid", r["job_id"])
        if k not in units:
            units[k] = []
            order.append(k)
        units[k].append(r)
    keep, skipped = [], []
    for k in order:
        c = sum(jcost(r) for r in units[k])
        if c <= left + 1e-6:
            keep += units[k]
            left -= c
        else:
            skipped.append((k, c))
    if skipped:
        with (BASE / "BUDGET_SKIPS.jsonl").open("a") as fh:
            for k, c in skipped:
                fh.write(json.dumps(dict(unit=k, planned=c, left=left, reserve=reserve)) + "\n")
    return keep


def write(out, rows):
    with Path(out).open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(len(rows), "jobs", sum(float(r["t_end"] if r["kind"] == "native" else r["H"]) for r in rows), "sim-s ->", out)


def plan_natives():
    rows = []
    for (d, p, s) in worlds():
        rows.append(dict(kind="native", job_id=f"nat_{cell_key(d, p)}_{s}", map="wz3to2",
                         map_spec=str(MAPSPEC), seed=s, demand={"main": d}, penetration=p,
                         t_end=T_END, save_times=SAVE_TIMES, out_dir=str(native_dir(d, p, s))))
    return rows


def plan_screen():
    ms = moments()
    (BASE / "MOMENTS.json").write_text(json.dumps(ms))
    rows = []
    for m in ms:
        if not m["valid"]:
            continue
        rows.append(bjob(m, 0, "base", [], H_MAIN, save_at=[H_MAIN]))
        for vid, v in m["variants"].items():
            rows.append(bjob(m, 0, f"g__{vid}", v["members"], H_MAIN, save_at=[H_MAIN]))
    return rows


def moments_loaded():
    return [m for m in json.loads((BASE / "MOMENTS.json").read_text()) if m["valid"]]


def arm_series(m, rep, arm, n):
    """(series of the first n s, RESULT) of one arm; rep 0 beyond H_MAIN concatenates the r0 extension."""
    r = load(mdir(m) / f"r{rep}" / arm)
    if r is None:
        return None, None
    s = series(r)
    if rep == 0 and n > len(s):
        e = load(mdir(m) / "r0ext" / arm)
        if e is None:
            return None, r
        s = np.concatenate([s, series(e)])
    if len(s) < n:
        return None, r
    return s[:n], r


def rep_eval(m, vid, rep, n=H_MAIN, need_solos=True, group_arm=None):
    """Statistics of one (moment, variant, replicate) on the first n seconds; None if arms missing."""
    v = m["variants"][vid]
    jb, base = arm_series(m, rep, "base", n)
    jg, grp = arm_series(m, rep, group_arm or f"g__{vid}", n)
    if jb is None or jg is None:
        return None
    out = dict(T=float((jb - jg).sum()), T_lb=boot_lb(jb - jg),
               safe=grp["collisions"] == 0 and grp["teleports"] == 0,
               group=dict(collisions=grp["collisions"], teleports=grp["teleports"], hard_brake_vs=grp["hard_brake_vs"],
                          min_accel=grp["min_accel"], emergency_steps=grp["emergency_steps"]),
               base_safety=dict(collisions=base["collisions"], teleports=base["teleports"],
                                hard_brake_vs=base["hard_brake_vs"], min_accel=base["min_accel"]),
               J_base=float(jb.sum()), J_group=float(jg.sum()))
    if not need_solos:
        return out
    solos = {}
    for mm in v["members"]:
        x, _ = arm_series(m, rep, f"s__{vid}__{mm['id']}", n)
        if x is None:
            return out
        solos[mm["id"]] = x
    best = min(solos, key=lambda k: solos[k].sum())
    out.update(S=float((solos[best] - jg).sum()), S_lb=boot_lb(solos[best] - jg), best_solo=best,
               J_solo={k: float(x.sum()) for k, x in solos.items()},
               solo_gain={k: float((jb - x).sum()) for k, x in solos.items()})
    return out


def best_per_family(m):
    """Solo picks on r0 among variants with LB95(T)>0, T>=1, safe: B = best-T per family (closing_pair,
    abreast_wall; <=2 sets); R = the N_SOLO_R best-T random proposals (count-matched, <=2 sets)."""
    picks, rpool = {}, []
    for vid, v in m["variants"].items():
        e = rep_eval(m, vid, 0, need_solos=False)
        if e is None or not (e["T_lb"] > 0 and e["T"] >= MATERIAL and e["safe"]):
            continue
        fam = v["family"]
        if fam == "R_random":
            rpool.append((e["T"], vid))
            continue
        if fam not in picks or e["T"] > picks[fam][1]:
            picks[fam] = (vid, e["T"])
    rpool.sort(key=lambda x: (-x[0], x[1]))
    return [vid for vid, _ in picks.values()] + [vid for _, vid in rpool[:N_SOLO_R]]


def solo_jobs(m, vid, rep, H, save_at=()):
    v = m["variants"][vid]
    return [bjob(m, rep, f"s__{vid}__{mm['id']}", [mm], H, save_at=save_at, dither=dither_spec(m, rep))
            for mm in v["members"]]


def r0_candidate(m, vid):
    e = rep_eval(m, vid, 0)
    return e is not None and "S" in e and e["T_lb"] > 0 and e["S_lb"] > 0 and e["S"] >= MATERIAL and e["safe"]


def rep_pass(m, vid, rep):
    """Main-window replicate test (amendment 3: later harm is checked on the r0 extension only)."""
    e = rep_eval(m, vid, rep)
    return e is not None and "S" in e and e["T_lb"] > 0 and e["S_lb"] > 0 and e["S"] >= MATERIAL and e["safe"]


def rep1_picks(m):
    """Amendment 3 (budget): per moment, the r0 candidate with the largest LB95(S) per arm (B, R)."""
    best = {}
    for vid, v in m["variants"].items():
        if not r0_candidate(m, vid):
            continue
        arm = "R" if v["family"] == "R_random" else "B"
        lb = rep_eval(m, vid, 0)["S_lb"]
        if arm not in best or lb > best[arm][1]:
            best[arm] = (vid, lb)
    return [vid for vid, _ in best.values()]


def plan_solos():
    rows = []
    for m in moments_loaded():
        for vid in best_per_family(m):
            rows += solo_jobs(m, vid, 0, H_MAIN, save_at=[H_MAIN])
    return rows


def plan_rep1():
    rows = []
    for m in moments_loaded():
        cands = rep1_picks(m)
        if not cands or not dither_spec(m, 1):
            continue
        rows.append(bjob(m, 1, "base", [], H_REP, dither=dither_spec(m, 1)))
        for vid in cands:
            rows.append(bjob(m, 1, f"g__{vid}", m["variants"][vid]["members"], H_REP, dither=dither_spec(m, 1)))
            rows += solo_jobs(m, vid, 1, H_REP)
    return rows


def ext_job(m, arm):
    src = mdir(m) / "r0" / arm
    r = load(src)
    sv = r["saved"][f"e{H_MAIN}"]
    j = bjob(m, 0, arm, [], H_EXT)
    j.update(kind="extend", job_id=f"{m['sid']}_r0ext_{arm}", t_load=m["t0"] + H_MAIN,
             state_file=str(src / "states" / f"e{H_MAIN}.xml.gz"), inserted_load=sv["inserted"],
             track_ids=[], out_dir=str(mdir(m) / "r0ext" / arm))
    return j


def plan_rep2():
    rows = []
    for m in moments_loaded():
        cands = [vid for vid in rep1_picks(m) if rep_pass(m, vid, 1)]
        if not cands or not dither_spec(m, 2):
            continue
        rows.append(bjob(m, 2, "base", [], H_REP, dither=dither_spec(m, 2)))
        rows.append(ext_job(m, "base"))
        for vid in cands:
            v = m["variants"][vid]
            rows.append(bjob(m, 2, f"g__{vid}", v["members"], H_REP, dither=dither_spec(m, 2)))
            rows += solo_jobs(m, vid, 2, H_REP)
            rows.append(ext_job(m, f"g__{vid}"))       # amendment 4: no solo extensions, no move-dither arm
    return rows


if __name__ == "__main__":
    mode, out = sys.argv[1], sys.argv[2]
    fn = dict(natives=plan_natives, screen=plan_screen, solos=plan_solos, rep1=plan_rep1, rep2=plan_rep2)[mode]
    rows = fn()
    if "--fit" in sys.argv:
        rows = fit(rows, reserve=float(sys.argv[sys.argv.index("--fit") + 1]))
    else:
        rows = [r for r in rows if not (Path(r["out_dir"]) / "RESULT.json").exists()]
    write(out, rows)
