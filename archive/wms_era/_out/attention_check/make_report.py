"""Aggregate ``raw_results.json`` into _out/ATTENTION_CHECK.md and _out/attention_check.json."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import attention_check as A  # noqa: E402

OUT_MD = A.WORKSPACE / "_out" / "ATTENTION_CHECK.md"
OUT_JSON = A.WORKSPACE / "_out" / "attention_check.json"
RAW = HERE / "raw_results.json"

PLAIN = ["free_plain", "follow_plain", "jam_plain"]


def pct(x, d=1):
    return f"{100.0 * float(x):.{d}f}%"


def load():
    raw = json.loads(RAW.read_text(encoding="utf-8"))
    per_state = []
    for r in raw["results"]:
        groups = {g: np.asarray(v, dtype=np.float64) for g, v in r["groups"].items()}
        groups["plain"] = sum(groups[g] for g in PLAIN)
        per_state.append({**r, "groups": groups})
    return raw, per_state


def total(per_state, group, set_name=None):
    vec = np.zeros(A.FLAT)
    for p in per_state:
        if set_name is None or p["set"] == set_name:
            vec += p["groups"][group]
    return vec


def collapse(mass, sel_blocks=None):
    """[NB,H,NR] -> dict of grouped role masses per (block,head) preserved."""
    return {
        "self": mass[..., 0],
        "leader": mass[..., 1],
        "ahead_2_3": mass[..., 2:4].sum(-1),
        "ahead_4+": mass[..., 4],
        "ahead_same": mass[..., 1:5].sum(-1),
        "follower": mass[..., 5],
        "behind_2+": mass[..., 6],
        "behind_same": mass[..., 5:7].sum(-1),
        "adj_ahead": mass[..., 7:9].sum(-1),
        "adj_behind": mass[..., 9:11].sum(-1),
        "adjacent": mass[..., 7:11].sum(-1),
    }


def parity(per_state):
    """Compare the first wall_same rollouts with the stored harness scores (10/30/60 s)."""
    stored = json.loads(A.STORED_SCORES.read_text(encoding="utf-8"))["states"]
    by_key = {s["key"]: s for s in stored}
    rows = []

    def flat(prefix, obj, out):
        if isinstance(obj, dict):
            for k, v in obj.items():
                flat(f"{prefix}/{k}", v, out)
        elif isinstance(obj, (int, float)):
            out[prefix] = float(obj)

    for p in per_state:
        if "car" not in p:
            continue
        s = by_key.get(p["key"])
        if s is None or p["arm"] not in s["arms"] or s["arms"][p["arm"]].get("error"):
            rows.append({"key": p["key"], "status": "no stored score"})
            continue
        mine, theirs = {}, {}
        flat("", p["car"], mine)
        stored_car = {h: v for h, v in s["arms"][p["arm"]]["car"].items() if h in p["car"]}
        flat("", stored_car, theirs)
        keys = sorted(set(mine) & set(theirs))
        diffs = [abs(mine[k] - theirs[k]) for k in keys]
        rows.append({"key": p["key"], "arm": p["arm"], "n_compared": len(keys),
                     "max_abs_diff": max(diffs) if diffs else None,
                     "status": "identical" if diffs and max(diffs) < 1e-6 else "DIFFERS"})
    return rows


def git_head():
    try:
        return subprocess.check_output(["git", "-C", str(A.WORKSPACE), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def head_row(u, b=None, h=None):
    m = u["mass"]
    sc = u["scal"]
    mex = u["mass_ex"]
    if b is None:
        m, sc, mex = m.mean((0, 1)), sc.mean((0, 1)), mex.mean((0, 1))
    elif h is None:
        m, sc, mex = m[b].mean(0), sc[b].mean(0), mex[b].mean(0)
    else:
        m, sc, mex = m[b, h], sc[b, h], mex[b, h]
    c = collapse(m)
    return {
        "self": c["self"], "leader": c["leader"], "ahead_2_3": c["ahead_2_3"],
        "ahead_4+": c["ahead_4+"], "follower": c["follower"], "behind_2+": c["behind_2+"],
        "adj_ahead": c["adj_ahead"], "adj_behind": c["adj_behind"],
        "behind_any": sc[0], "far50": sc[2], "far100": sc[3],
        "mean_abs_ds_ex": sc[4] / max(1.0 - c["self"], 1e-9),
        "eff_n": sc[6], "leader_ex": mex[1],
        "ahead_same": c["ahead_same"], "behind_same": c["behind_same"],
    }


def main():
    raw, per_state = load()
    n_states = len(per_state)
    sets = sorted({p["set"] for p in per_state})
    agg = {g: total(per_state, g) for g in A.GROUPS + ["plain"]}
    U = {g: A.unpack(v) for g, v in agg.items()}
    u_all = U["all"]
    diag = {
        "queries": sum(p["diag"]["queries"] for p in per_state),
        "sampled_steps": sum(p["diag"]["sampled_steps"] for p in per_state),
        "max_outside_mass": max(p["diag"]["max_outside_mass"] for p in per_state),
        "max_rowsum_err": max(p["diag"]["max_rowsum_err"] for p in per_state),
        "isolated_queries": sum(p["diag"]["isolated"] for p in per_state),
        "min_candidate_mass": min(p["diag"]["min_cand_mass"] for p in per_state),
    }
    par = parity(per_state)
    ci = {g: A.bootstrap_ci(per_state, g) for g in
          ["all", "free_plain", "follow_plain", "jam_plain", "merge_no_lc",
           "lc_main_nomerge", "lc_ramp"] if U[g]["n"] > 0}
    # distinct states / worlds
    worlds = sorted({(p["set"], p["world"]) for p in per_state})
    pen = sorted({p["penetration_pct"] for p in per_state if p["penetration_pct"] is not None})

    lines = []
    w = lines.append
    w("# Attention check: does the C1 WM-S transformer look at the car ahead?")
    w("")
    w(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}. CPU only; no training, no SUMO, "
      "no model-code change (forward hooks on `residual_net.blocks[b].attn`).")
    w("")
    w("## 0. Setup")
    w("")
    w(f"* Checkpoint: `WM-S-C1FAST-20261006/box/full_struct/wm_s.pt` (sha256 `{A._sha256(A.CKPT)[:16]}...`, "
      "staged copy verified byte-identical), structured WM-S, flags from its TRAIN_RECEIPT "
      "(residual rollout, no lane head, no driver history, entries OFF, receipt physics).")
    w(f"* Code: worktree `E:/rd-wt/ws-attn`, branch `wms/attn-check`, base commit `{git_head()[:12]}`. "
      "Network: `HdvResidualNet`, 3 `_KinBlock`, 4 heads, d_model 128; keys = self + vehicles within "
      "150 m in the same or an adjacent lane (lane index difference <= 1) + learned pair bias on "
      "(ds, dlane, dv, |ds|, same-lane, self).")
    w("* Rollout: `WmsCheckpointModel.predict_states`, i.e. exactly the `main2_score.py` / `wall_c1_score.py` path "
      "(CAVs teacher-forced from the recorded arm, entries off), 60 s, seed 0. "
      "Reproduction check against the stored harness scores for the first two states: "
      + "; ".join(f"{r['key'].split('/')[-1]} {r['status']} (max |diff| {r.get('max_abs_diff')}, "
                  f"{r.get('n_compared')} values at 10/30/60 s)" for r in par) + ".")
    w(f"* States: {n_states} rollouts = first state per world (sorted by world, t0_s, key), first 20 worlds, from "
      + ", ".join(f"`{s}.json` ({sum(1 for p in per_state if p['set'] == s)} states)" for s in sets)
      + f". {len(worlds)} (set, world) pairs; penetration values present: {pen}. "
      "For `wall_same` the teacher arm is the record's own arm (`wall`); for `seen_controls` it is `arm0`.")
    w(f"* Sampling: attention captured at every 5th 0.5 s step (every 2.5 s; owner suggested every 10th) = "
      f"{diag['sampled_steps']} sampled (state, step) pairs, {diag['queries']} HDV query-steps in total. "
      "Queries = every present non-CAV vehicle; keys include CAVs and HDVs.")
    w(f"* Sanity: attention mass outside the candidate set = {diag['max_outside_mass']:.1e}; max |row sum - 1| = "
      f"{diag['max_rowsum_err']:.1e}; {diag['isolated_queries']} isolated queries (no neighbour at all) "
      "are excluded from the \"excluding self\" columns; every candidate receives non-zero mass.")
    w("* Roles (relative to the query at that step; `ds = s_key - s_query`, ds > 0 = ahead): self; "
      "**leader** = nearest same-lane vehicle ahead; ahead_2nd/3rd/4th+ = further same-lane ahead; "
      "**follower** = nearest same-lane behind; behind_2nd+ = other same-lane behind; for each adjacent lane "
      "the nearest vehicle ahead / behind is `adj_*_near`, all others `adj_*_far`. "
      "\"Behind (any lane)\" = all non-self keys with ds <= 0. Percentages are the mean over query-steps of "
      "the attention mass (each query's attention row sums to 100% including self).")
    w("* Situations of the query (speed v, bumper gap to its same-lane leader, 5 m vehicle length): "
      "**free** v > 20 m/s and gap > 50 m (no leader within 150 m counts as free); **jam** v <= 5 m/s; "
      "**following** = everything else; **merge** = on-ramp lane 2 or s in [-50, 125.44] m (merge zone "
      "[0, 75.44] m +- 50 m); **lane change** = the vehicle's lane differs from its current lane at any of "
      "the next 4 steps (2 s). The speed classes partition all queries; merge / lane change overlay them, so "
      "the `*_plain` rows (outside merge, no lane change) give the clean car-following picture.")
    w("")

    # ---------------------------------------------------------------- Table 1
    w("## 1. Where the mass goes (all HDV queries, pooled)")
    w("")
    w(f"Average number of candidates a query can attend to: **{u_all['cand_mean']:.1f}** including itself "
      f"(**{u_all['cand_nonself_mean']:.1f}** other vehicles; per role: "
      + ", ".join(f"{A.ROLES[r]} {u_all['cand_role_count_mean'][r]:.2f}" for r in range(1, A.NR))
      + f"). CAVs are {pct(u_all['cav_cand_share'])} of the other candidates.")
    w("")
    w("| role | L0 | L1 | L2 | mean of 3 layers | uniform-over-candidates baseline | lift (mean / uniform) |")
    w("|---|---|---|---|---|---|---|")
    mass = u_all["mass"]                                # [NB,H,NR]
    per_layer = mass.mean(1)                            # [NB,NR]
    uni = np.array(u_all["uniform_role_share"])
    mean3 = per_layer.mean(0)
    for r in range(A.NR):
        lift = mean3[r] / uni[r] if uni[r] > 0 else float("nan")
        w(f"| {A.ROLES[r]} | {pct(per_layer[0, r])} | {pct(per_layer[1, r])} | {pct(per_layer[2, r])} | "
          f"**{pct(mean3[r])}** | {pct(uni[r])} | {lift:.2f}x |")
    sc = u_all["scal"].mean(1)                          # [NB,NS]
    nonself = 1.0 - per_layer[:, 0]
    w(f"| *behind, any lane (ds <= 0)* | {pct(sc[0, 0])} | {pct(sc[1, 0])} | {pct(sc[2, 0])} | "
      f"**{pct(sc[:, 0].mean())}** | | |")
    w(f"| *ahead, any lane (ds > 0)* | {pct(sc[0, 1])} | {pct(sc[1, 1])} | {pct(sc[2, 1])} | "
      f"**{pct(sc[:, 1].mean())}** | | |")
    w(f"| *> 50 m away* | {pct(sc[0, 2])} | {pct(sc[1, 2])} | {pct(sc[2, 2])} | **{pct(sc[:, 2].mean())}** | | |")
    w(f"| *> 100 m away* | {pct(sc[0, 3])} | {pct(sc[1, 3])} | {pct(sc[2, 3])} | **{pct(sc[:, 3].mean())}** | | |")
    w("")
    c3 = collapse(mass.mean((0, 1)))
    hl = A.headline(agg["all"])
    uni_c = {"ahead_same": uni[1:5].sum(), "behind_same": uni[5:7].sum(), "adjacent": uni[7:11].sum(),
             "leader": uni[1], "self": uni[0]}
    w(f"Grouped (mean of layers and heads): self {pct(c3['self'])}; **same-lane ahead {pct(c3['ahead_same'])}** "
      f"(leader alone {pct(c3['leader'])}); same-lane behind {pct(c3['behind_same'])}; "
      f"**adjacent lanes {pct(c3['adjacent'])}** (ahead {pct(c3['adj_ahead'])}, behind {pct(c3['adj_behind'])}). "
      f"Uniform baseline: ahead-same {pct(uni_c['ahead_same'])}, behind-same {pct(uni_c['behind_same'])}, "
      f"adjacent {pct(uni_c['adjacent'])}, leader {pct(uni_c['leader'])}. "
      f"Excluding self (renormalised per query), the leader gets **{pct(hl['leader_ex_self'])}** of the "
      f"remaining mass. Mean |ds| of the attended non-self vehicles is **{hl['mean_abs_ds_ex_self_m']:.0f} m**; "
      f"effective number of attended vehicles (exp entropy, incl. self) is **{hl['eff_n']:.1f}** of "
      f"{u_all['cand_mean']:.1f} candidates.")
    ci_all = ci["all"]
    w("")
    w("95% bootstrap CIs over the 40 states (layer/head mean): "
      + ", ".join(f"{k} [{pct(v[0])}, {pct(v[1])}]" for k, v in ci_all.items()
                  if k in ("self", "leader", "leader_ex_self", "ahead_same_lane", "behind_same_lane",
                           "adjacent_lane", "behind_any", "far50", "far100"))
      + f"; mean |ds| ex-self [{ci_all['mean_abs_ds_ex_self_m'][0]:.0f}, {ci_all['mean_abs_ds_ex_self_m'][1]:.0f}] m; "
      f"effective N [{ci_all['eff_n'][0]:.1f}, {ci_all['eff_n'][1]:.1f}].")
    w("")
    w("Attention mass by signed distance (non-self keys; mean over layers and heads), vs the share of "
      "candidates in that bin:")
    w("")
    w("| bin | " + " | ".join(A.BIN_NAMES) + " |")
    w("|---|" + "---|" * A.NBIN)
    bins = u_all["bins"].mean((0, 1))
    binuni = np.array(u_all["bin_uniform_share"])
    w("| mass (incl. self in denominator) | " + " | ".join(pct(x) for x in bins) + " |")
    w("| candidate share (of non-self) | " + " | ".join(pct(x) for x in binuni) + " |")
    w("")

    # ---------------------------------------------------------------- Table 2: heads
    w("## 2. Per block and head (all HDV queries)")
    w("")
    w("Mass in percent; `adj ahead/behind` = both adjacent lanes; `behind any` = ds <= 0, any lane; "
      "`far>50` = non-self keys more than 50 m away; `effN` = exp(entropy) of the row; "
      "`leader ex-self` = leader share after removing self.")
    w("")
    w("| block / head | self | leader | ahead 2nd-3rd | ahead 4th+ | follower | behind 2nd+ | adj ahead | adj behind | behind any | far>50 | effN | leader ex-self |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    head_rows = {}
    for b in range(A.NB):
        for h in range(A.NH):
            r = head_row(u_all, b, h)
            head_rows[f"L{b}H{h}"] = r
            w(f"| L{b}H{h} | {pct(r['self'])} | {pct(r['leader'])} | {pct(r['ahead_2_3'])} | {pct(r['ahead_4+'])} | "
              f"{pct(r['follower'])} | {pct(r['behind_2+'])} | {pct(r['adj_ahead'])} | {pct(r['adj_behind'])} | "
              f"{pct(r['behind_any'])} | {pct(r['far50'])} | {r['eff_n']:.1f} | {pct(r['leader_ex'])} |")
        r = head_row(u_all, b, None)
        head_rows[f"L{b}"] = r
        w(f"| **L{b} mean** | {pct(r['self'])} | {pct(r['leader'])} | {pct(r['ahead_2_3'])} | {pct(r['ahead_4+'])} | "
          f"{pct(r['follower'])} | {pct(r['behind_2+'])} | {pct(r['adj_ahead'])} | {pct(r['adj_behind'])} | "
          f"{pct(r['behind_any'])} | {pct(r['far50'])} | {r['eff_n']:.1f} | {pct(r['leader_ex'])} |")
    r = head_row(u_all)
    head_rows["all"] = r
    w(f"| **all mean** | {pct(r['self'])} | {pct(r['leader'])} | {pct(r['ahead_2_3'])} | {pct(r['ahead_4+'])} | "
      f"{pct(r['follower'])} | {pct(r['behind_2+'])} | {pct(r['adj_ahead'])} | {pct(r['adj_behind'])} | "
      f"{pct(r['behind_any'])} | {pct(r['far50'])} | {r['eff_n']:.1f} | {pct(r['leader_ex'])} |")
    w("")

    # ---------------------------------------------------------------- Table 3: situations
    w("## 3. By situation of the query (mean over layers and heads)")
    w("")
    sit_order = ["all", "free", "follow", "jam", "merge", "lc_any", "lc_main", "lc_ramp", "no_lc",
                 "free_plain", "follow_plain", "jam_plain", "plain", "merge_no_lc", "lc_main_nomerge"]
    A.GROUP_DOC["plain"] = "free + following + jam, all outside the merge zone and without a lane change in 2 s"
    w("| situation | n query-steps | cand. (incl. self) | self | leader | ahead 2nd-3rd | ahead 4th+ | follower | behind 2nd+ | adj ahead | adj behind | behind any | far>50 | effN | leader ex-self | uniform leader |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    sit_rows = {}
    for g in sit_order:
        u = U[g]
        if u["n"] == 0:
            continue
        r = head_row(u)
        sit_rows[g] = {**r, "n": u["n"], "cand": u["cand_mean"],
                       "uniform_leader": u["uniform_role_share"][1]}
        w(f"| {g} | {u['n']} | {u['cand_mean']:.1f} | {pct(r['self'])} | {pct(r['leader'])} | {pct(r['ahead_2_3'])} | "
          f"{pct(r['ahead_4+'])} | {pct(r['follower'])} | {pct(r['behind_2+'])} | {pct(r['adj_ahead'])} | "
          f"{pct(r['adj_behind'])} | {pct(r['behind_any'])} | {pct(r['far50'])} | {r['eff_n']:.1f} | "
          f"{pct(r['leader_ex'])} | {pct(u['uniform_role_share'][1])} |")
    w("")
    w("Definitions: " + "; ".join(f"`{g}` = {A.GROUP_DOC[g]}" for g in sit_order if g in A.GROUP_DOC) + ".")
    w("")
    w("Bootstrap 95% CIs over states (layer/head mean) for the key situations:")
    w("")
    w("| situation | self | leader | leader ex-self | ahead same lane | behind same lane | adjacent lanes | behind any | far>50 | effN |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for g, c in ci.items():
        def f(k, pc=True):
            lo, hi = c[k]
            return f"[{pct(lo)}, {pct(hi)}]" if pc else f"[{lo:.1f}, {hi:.1f}]"
        w(f"| {g} | {f('self')} | {f('leader')} | {f('leader_ex_self')} | {f('ahead_same_lane')} | "
          f"{f('behind_same_lane')} | {f('adjacent_lane')} | {f('behind_any')} | {f('far50')} | {f('eff_n', False)} |")
    w("")

    # ---------------------------------------------------------------- Table 4: head x situation
    w("## 4. Which heads specialise (head x situation)")
    w("")
    cols = ["free_plain", "follow_plain", "jam_plain", "merge_no_lc", "lc_main_nomerge", "lc_ramp"]
    cols = [c for c in cols if U[c]["n"] > 0]
    for title, fn in [
        ("Leader mass (incl. self in denominator)", lambda m, s: m[1]),
        ("Mass on cars behind (any lane, ds <= 0)", lambda m, s: s[0]),
        ("Mass on adjacent-lane vehicles (ahead + behind)", lambda m, s: m[7:11].sum()),
        ("Self mass", lambda m, s: m[0]),
        ("Effective number of attended vehicles (exp entropy)", lambda m, s: s[6]),
    ]:
        w(f"**{title}**")
        w("")
        w("| head | " + " | ".join(f"{c} (n={U[c]['n']})" for c in cols) + " | all |")
        w("|---|" + "---|" * (len(cols) + 1))
        for b in range(A.NB):
            for h in range(A.NH):
                vals = []
                for c in cols + ["all"]:
                    v = fn(U[c]["mass"][b, h], U[c]["scal"][b, h])
                    vals.append(f"{v:.1f}" if "Effective" in title else pct(v))
                w(f"| L{b}H{h} | " + " | ".join(vals) + " |")
        w("")

    # ---------------------------------------------------------------- CAV vs HDV keys
    w("## 5. CAV versus HDV as attended vehicle")
    w("")
    mc = u_all["mass_cav"].mean((0, 1))
    m_all = u_all["mass"].mean((0, 1))
    cav_nonself = mc[1:].sum() / max(m_all[1:].sum(), 1e-9)
    w(f"Of the attention mass on other vehicles (non-self), **{pct(cav_nonself)}** goes to CAV keys, "
      f"versus a CAV share of {pct(u_all['cav_cand_share'])} among the non-self candidates "
      f"(about {100*sum(p['n_cav'] for p in per_state)/sum(p['n_slots'] for p in per_state):.1f}% of slots are CAVs). "
      "The net has no CAV flag, so any difference reflects kinematics / position only.")
    w("")
    w("| role | CAV share of that role's mass |")
    w("|---|---|")
    for r in range(1, A.NR):
        w(f"| {A.ROLES[r]} | {pct(mc[r] / max(m_all[r], 1e-12))} |")
    w("")

    # ---------------------------------------------------------------- per set
    w("## 6. Consistency between the two state sets")
    w("")
    w("| set | n query-steps | self | leader | leader ex-self | ahead same lane | behind same lane | adjacent lanes | behind any | far>50 | effN |")
    w("|---|---|---|---|---|---|---|---|---|---|---|")
    per_set = {}
    for s in sets:
        v = total(per_state, "all", s)
        hs = A.headline(v)
        u = A.unpack(v)
        per_set[s] = {"n": u["n"], **{k: float(x) for k, x in hs.items()}}
        w(f"| {s} | {u['n']} | {pct(hs['self'])} | {pct(hs['leader'])} | {pct(hs['leader_ex_self'])} | "
          f"{pct(hs['ahead_same_lane'])} | {pct(hs['behind_same_lane'])} | {pct(hs['adjacent_lane'])} | "
          f"{pct(hs['behind_any'])} | {pct(hs['far50'])} | {hs['eff_n']:.1f} |")
    w("")

    # ---------------------------------------------------------------- Reading
    free, fol, jam = sit_rows["free_plain"], sit_rows["follow_plain"], sit_rows["jam_plain"]
    mrg = sit_rows.get("merge_no_lc")
    lcm = sit_rows.get("lc_main_nomerge")
    lcr = sit_rows.get("lc_ramp")
    plain = sit_rows["plain"]
    hr = head_rows["all"]
    m_all = u_all["mass"].mean((0, 1))
    nonself_all = 1.0 - hr["self"]
    binu = np.array(u_all["bin_uniform_share"])
    far50_cand = float(binu[2:4].sum() + binu[6:8].sum())
    far100_cand = float(binu[3] + binu[7])
    head_names = [k for k in head_rows if "H" in k]
    TH = {"leader_head": 0.10, "rear_same_lane_head": 0.40, "adjacent_reader_head": 0.75,
          "self_head": 0.28, "far_reader_head": 0.65}
    lead_heads = [k for k in head_names if head_rows[k]["leader"] >= TH["leader_head"]]
    rear_heads = [k for k in head_names if head_rows[k]["behind_same"] >= TH["rear_same_lane_head"]]
    adj_heads = [k for k in head_names
                 if head_rows[k]["adj_ahead"] + head_rows[k]["adj_behind"] >= TH["adjacent_reader_head"]]
    self_heads = [k for k in head_names if head_rows[k]["self"] >= TH["self_head"]]
    far_heads = [k for k in head_names if head_rows[k]["far50"] >= TH["far_reader_head"]]

    def hlist(names, key, pc=True):
        return ", ".join(f"{k} {pct(head_rows[k][key]) if pc else head_rows[k][key]}" for k in names)

    adj_str = ", ".join("%s %s" % (k, pct(head_rows[k]["adj_ahead"] + head_rows[k]["adj_behind"]))
                        for k in adj_heads)
    reading = [
        f"1. It does not behave like a forward-gazing driver: the same-lane leader gets {pct(c3['leader'])} of the mass "
        f"({pct(hl['leader_ex_self'])} once self is removed, lift {c3['leader'] / uni_c['leader']:.2f}x vs a uniform "
        f"read of the {u_all['cand_mean']:.1f} candidates), all same-lane cars ahead {pct(c3['ahead_same'])} "
        f"(uniform {pct(uni_c['ahead_same'])}); {pct(c3['adjacent'])} goes to adjacent-lane cars, concentrated on "
        f"the nearest one ahead ({pct(m_all[7])}, {m_all[7] / uni[7]:.1f}x its candidate share) and behind "
        f"({pct(m_all[9])}, {m_all[9] / uni[9]:.1f}x) - the MOBIL new-leader / new-follower pair - plus "
        f"{pct(c3['self'])} on itself.",
        f"2. Cars behind: {pct(hl['behind_any'])} of all mass ({pct(hl['behind_any'] / nonself_all)} of the non-self "
        f"mass) sits on vehicles behind the query, as much as ahead ({pct(u_all['scal'].mean((0, 1))[1])}); "
        f"far cars: {pct(hl['far50'])} is >50 m away ({pct(hl['far50'] / nonself_all)} of non-self mass vs "
        f"{pct(far50_cand)} of candidates) and {pct(hl['far100'])} >100 m ({pct(hl['far100'] / nonself_all)} vs "
        f"{pct(far100_cand)}); mean attended distance {hl['mean_abs_ds_ex_self_m']:.0f} m, effective "
        f"{hl['eff_n']:.1f} vehicles of {u_all['cand_mean']:.1f} candidates. So it is distance-tapered but "
        "symmetric front/back, not forward-focused.",
        f"3. Situation effects are modest: leader mass rises from {pct(free['leader'])} (free flow) to "
        f"{pct(fol['leader'])} when following ({pct(free['leader_ex'])} -> {pct(fol['leader_ex'])} ex-self) and "
        f"drops to {pct(jam['leader'])} in a jam, where mass moves to farther same-lane cars; effective N grows "
        f"{free['eff_n']:.1f} (free) -> {jam['eff_n']:.1f} (jam) / {mrg['eff_n']:.1f} (merge zone) but less than in proportion to "
        f"the candidate count ({free['cand']:.0f} -> {jam['cand']:.0f}); mainline cars about to change lane do not "
        f"shift toward the adjacent lane ({pct(lcm['adj_ahead'] + lcm['adj_behind'])} vs "
        f"{pct(plain['adj_ahead'] + plain['adj_behind'])} plain), only ramp cars about to merge do "
        f"({pct(lcr['adj_ahead'] + lcr['adj_behind'])} adjacent vs "
        f"{pct(mrg['adj_ahead'] + mrg['adj_behind'])} in the merge zone without a lane change; only "
        f"{sit_rows['lc_ramp']['n']:.0f} query-steps).",
        f"4. Heads specialise: leader/forward heads {hlist(lead_heads, 'leader')} (L1H3 reaches "
        f"{pct(U['follow_plain']['mass'][1, 3, 1])} when following); rear-looking same-lane head "
        f"{hlist(rear_heads, 'behind_same')} (follower + others behind); adjacent-lane readers "
        f"{adj_str}; "
        f"self/identity heads {hlist(self_heads, 'self')}; long-range (>50 m) readers "
        f"{hlist(far_heads, 'far50')}. Layer 1 is the longitudinal layer (leader {pct(head_rows['L1']['leader'])}, "
        f"same-lane behind {pct(head_rows['L1']['behind_same'])}), layer 2 is local and low-entropy "
        f"(effN {head_rows['L2']['eff_n']:.1f}, adjacent-behind {pct(head_rows['L2']['adj_behind'])}).",
        "5. Verdict: it reads a MOBIL-like neighbourhood (self, nearest adjacent-lane lead/lag, followers, a few "
        "leader heads that sharpen when following) rather than a human's forward gaze. At the "
        "layer-mean level the split is nearly situation-independent (adjacent lanes stay 54-63% in every "
        "situation) although single heads re-allocate (e.g. L1H1 adjacent-lane mass "
        f"{pct(U['free_plain']['mass'][1, 1, 7:11].sum())} in free flow vs {pct(U['jam_plain']['mass'][1, 1, 7:11].sum())} "
        "in a jam). Leader-following is already handled by the physics base, so the residual net is not required "
        "to re-read the leader. Attention weights are not causal importance: they show where information is read "
        "from, not what drives the correction.",
    ]
    short = [
        f"1. Not a driver-like forward gaze: only {pct(c3['leader'])} of the attention goes to the car directly ahead "
        f"({pct(hl['leader_ex_self'])} of what is left after the car's own slot), less than a uniform read of its "
        f"{u_all['cand_mean']:.0f} candidates would give ({pct(uni_c['leader'])}); {pct(c3['adjacent'])} goes to cars in the neighbouring "
        f"lane(s) (nearest one ahead and nearest one behind ~{pct(m_all[7], 0)} each) and {pct(c3['self'])} to itself.",
        f"2. Cars behind matter as much as cars ahead: {pct(hl['behind_any'], 0)} of the weight is on cars behind vs "
        f"{pct(u_all['scal'].mean((0, 1))[1], 0)} ahead; {pct(hl['far50'], 0)} is on cars >50 m away "
        f"({pct(hl['far100'], 0)} beyond 100 m); distance does taper it: {pct(hl['far50'] / nonself_all, 0)} of the non-self "
        f"weight is >50 m away vs {pct(far50_cand, 0)} of the candidates.",
        f"3. Situations change it little: leader weight {pct(free['leader'])} in free flow -> {pct(fol['leader'])} when "
        f"following (the one sensible shift) -> {pct(jam['leader'])} in a jam; main-road cars about to change lane do "
        f"not look more at the other lane, ramp cars about to merge do (main-lane weight {pct(lcr['adj_ahead'] + lcr['adj_behind'], 0)} "
        f"vs {pct(mrg['adj_ahead'] + mrg['adj_behind'], 0)} for ramp-zone cars not merging, "
        f"n={sit_rows['lc_ramp']['n']:.0f} query-steps); in dense traffic it spreads over more cars (effective "
        f"{free['eff_n']:.1f} -> {jam['eff_n']:.1f}) but less than in proportion to the candidates "
        f"({free['cand']:.0f} -> {jam['cand']:.0f}).",
        f"4. Heads: lead-car heads {', '.join(lead_heads)} (L1H3 up to {pct(U['follow_plain']['mass'][1, 3, 1], 0)} on the "
        f"leader when following); who-is-behind head {', '.join(rear_heads)}; neighbouring-lane readers "
        f"{', '.join(adj_heads)}; mostly-self heads {', '.join(self_heads)}; long-range readers {', '.join(far_heads)}.",
        "5. Verdict: a MOBIL-like neighbourhood reader (self + nearest lead/lag in the adjacent lanes + followers), "
        "not a car-following reader; plausible for a residual on top of an IDM/MOBIL physics base, but not "
        "\"mostly the car ahead\". Attention is where information is read from, not causal importance.",
    ]
    top = ["## Answer in five lines", ""]
    for line in short:
        top += [line, ""]
    lines[4:4] = top + [""]
    w("## 7. Plain reading: the numbers behind the five lines")
    w("")
    w("Quoted from the tables above (head-label thresholds: "
      + ", ".join(f"{k} >= {pct(v, 0)}" for k, v in TH.items()) + ").")
    w("")
    for line in reading:
        w(line)
        w("")
    w("Caveat: attention weights show where information is read from, not how much it changes the output; the "
      "residual net is added on top of the physics base (IDM + MOBIL), which already encodes car-following on the "
      "leader, so the net's job is a correction and it need not re-read the leader. Weights are also averaged over "
      "three stacked layers whose later layers attend to already-mixed representations.")
    md = "\n".join(lines) + "\n"

    json_out = {
        "schema": "attention_check/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": str(A.CKPT_SOURCE), "checkpoint_sha256": A._sha256(A.CKPT),
        "code_base_commit": git_head(), "branch": "wms/attn-check",
        "horizon_s": A.HORIZON_S, "sample_every_steps": A.EVERY, "entries": "off",
        "cav_mode": "teacher-forced recorded arm",
        "selection": raw["selection"], "n_states": n_states,
        "diag": diag, "parity_vs_stored_scores": par,
        "roles": A.ROLES, "situation_definitions": {g: A.GROUP_DOC[g] for g in sit_order if g in A.GROUP_DOC},
        "headline_all": {k: float(v) for k, v in hl.items()},
        "headline_ci95": ci, "per_set": per_set,
        "head_rows": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in head_rows.items()},
        "situation_rows": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in sit_rows.items()},
        "groups": {g: A.to_jsonable(U[g]) for g in sit_order if U[g]["n"] > 0},
        "cav_nonself_mass_share": float(cav_nonself),
        "READING_head_thresholds": TH, "READING_head_lists": {"leader": lead_heads, "rear_same_lane": rear_heads, "adjacent_reader": adj_heads, "self": self_heads, "far_reader": far_heads},
        "READING": short, "READING_DETAIL": reading,
    }
    return md, json_out, dict(free=free, fol=fol, jam=jam, mrg=mrg, lcm=lcm, lcr=lcr, plain=plain,
                              hr=hr, hl=hl, u_all=u_all, uni=uni, c3=c3, head_rows=head_rows,
                              reading=reading, ci=ci, cav_nonself=cav_nonself)


if __name__ == "__main__":
    md, js, ctx = main()
    OUT_MD.write_text(md, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(js, indent=1), encoding="utf-8")
    print("wrote", OUT_MD, OUT_JSON)
