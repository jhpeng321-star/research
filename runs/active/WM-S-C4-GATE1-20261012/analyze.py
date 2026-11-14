"""Gate 1 analysis: does the WM rank CAV controllers the way SUMO does?

Pair-level (paired differences arm A - arm B, WM vs SUMO): sign agreement,
Pearson, Spearman.  Group-level: Kendall tau of the arm ranking and the regret of
the WM-chosen arm (SUMO outcome of the pick minus the SUMO best).  All with
world-cluster bootstrap 95% CIs, split by arms seen vs unseen in WM training
(seen = {native, base}; PI and the wall arms are unseen for both models).

References: a physics-only row (R5b network with the learned residual and lane
correction gated off, population-default driver parameters) and a constant
predictor row (always pick native).

Usage:  python analyze.py            # writes results_gate1.json + REPORT.md
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

RUN = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

sys.path.insert(0, str(RUN.parents[2] / "src"))
from wms.eval.ranking import spearman  # noqa: E402

from outcomes import HORIZONS_S, METRICS, orient  # noqa: E402

MODELS = ("r5b", "c1", "physics")
MODEL_LABELS = {"r5b": "WM-S R5b best", "c1": "WM-S C1", "physics": "physics-only",
                "constant": "constant (native)"}
SEEN_ARMS = {"native", "base"}
PRIMARY_METRIC = "hdv_zone_delay_s"
N_BOOT, SEED = 1000, 0
EPS = 1e-9


# ------------------------------------------------------------------ statistics
def sign(v: float) -> int:
    return 1 if v > EPS else (-1 if v < -EPS else 0)


def pearson(x, y):
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx <= 1e-12 or sy <= 1e-12:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def kendall_tau_b(x, y):
    n = len(x)
    if n < 2:
        return None
    c = d = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = sign(x[i] - x[j]), sign(y[i] - y[j])
            if dx * dy > 0:
                c += 1
            elif dx * dy < 0:
                d += 1
            if dx == 0 and dy != 0:
                tx += 1
            if dy == 0 and dx != 0:
                ty += 1
    n0 = n * (n - 1) / 2.0
    denom = math.sqrt(max(0.0, (n0 - tx) * (n0 - ty)))
    if denom <= 0:
        return None
    return (c - d) / denom


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def boot_ci(worlds, values_of_world, stat_of_worlds, n_boot=N_BOOT, seed=SEED):
    """World-cluster bootstrap: resample worlds with replacement."""
    import random

    rng = random.Random(seed)
    world_list = sorted(worlds)
    points = []
    for _ in range(n_boot):
        sample = [rng.choice(world_list) for _ in world_list]
        vals = []
        for w in sample:
            vals.extend(values_of_world[w])
        if not vals:
            continue
        points.append(stat_of_worlds(vals, sample))
    if not points:
        return {"lo": None, "hi": None}
    points = sorted(p for p in points if p is not None)
    if not points:
        return {"lo": None, "hi": None}

    def q(p):
        idx = min(len(points) - 1, max(0, int(round(p * (len(points) - 1)))))
        return points[idx]
    return {"lo": q(0.025), "hi": q(0.975)}


# ------------------------------------------------------------------ data
def group_key(world, t0):
    return f"{world}|{round(float(t0), 3)}"


def load():
    groups = json.loads((RUN / "groups_gate1.json").read_text(encoding="utf-8"))
    cells = {}
    for tag in ("recorded",) + MODELS:
        cells[tag] = json.loads((RUN / f"outcomes_{tag}.json")
                                .read_text(encoding="utf-8"))["cells"]
    return groups, cells


def build_tables(groups, cells):
    """(groups: [{world, arms}], tables[model][world][arm][h][metric] = oriented)."""
    glist = []
    for meta in groups["groups"]:
        arms = meta.get("matched_arms") or []
        if len(arms) >= 2:
            glist.append({"world": meta["world"], "t0": meta["t0_s"], "arms": arms})
    tables = {}
    for model in ("recorded",) + MODELS:
        tab = {}
        for cell in cells[model].values():
            if cell.get("error"):
                continue
            world = str(cell["world"])
            tab.setdefault(world, {})[cell["arm"]] = {
                h: {m: orient(m, cell["outcomes"][h][m])
                    if cell["outcomes"][h][m] is not None else None
                    for m in METRICS}
                for h in ("10", "30", "60")}
        tables[model] = tab
    return glist, tables


def arms_differ(su_a: dict, su_b: dict, horizon: str) -> bool:
    """True when the two arms' recorded SUMO outcomes differ at ``horizon``.

    Within the 60 s window many wall branches are still trajectory-identical to
    native (the wall assembles later), so their paired differences are vacuous.
    """
    return any(su_a[horizon][m] != su_b[horizon][m] for m in METRICS)


# ------------------------------------------------------------------ metrics
def pair_rows(glist, tables, model):
    """Rows: (world, horizon, metric, pair, dWM, dSU) for matched arms.

    A pair enters horizon h only if it is *informative* there: the two arms'
    recorded SUMO outcomes differ at h (identical recorded trajectories, e.g. a
    wall that has not assembled inside the window, cannot test ranking).
    """
    rows = []
    for g in glist:
        world = g["world"]
        wm, su = tables[model].get(world, {}), tables["recorded"].get(world, {})
        for a, b in combinations(sorted(g["arms"]), 2):
            if a not in wm or a not in su or b not in wm or b not in su:
                continue
            for h in ("10", "30", "60"):
                if not arms_differ(su[a], su[b], h):
                    continue
                for m in METRICS:
                    va, vb = wm[a][h][m], wm[b][h][m]
                    ra, rb = su[a][h][m], su[b][h][m]
                    if None in (va, vb, ra, rb):
                        continue
                    rows.append({"world": world, "h": h, "m": m, "pair": (a, b),
                                 "dwm": va - vb, "dsu": ra - rb})
    return rows


def pair_stats(rows):
    if not rows:
        return None
    dwm = [r["dwm"] for r in rows]
    dsu = [r["dsu"] for r in rows]
    agree = sum(1 for r in rows if sign(r["dwm"]) == sign(r["dsu"])) / len(rows)
    return {"n_pairs": len(rows),
            "n_worlds": len({r["world"] for r in rows}),
            "sign_agreement": agree,
            "pearson": pearson(dwm, dsu),
            "spearman": spearman(dwm, dsu)}


def group_ranking_stats(glist, tables, model, metric, horizon):
    """Mean Kendall tau and regret stats over groups (worlds).

    Arms whose recorded outcomes at ``horizon`` equal native's (e.g. a wall that
    has not assembled inside the window) are collapsed away: they are the native
    arm for every purpose at that horizon.
    """
    taus, regrets, const_regrets, picks = [], [], [], []
    for g in glist:
        world = g["world"]
        wm, su = tables[model].get(world, {}), tables["recorded"].get(world, {})
        arms = [a for a in g["arms"]
                if a in wm and a in su and wm[a][horizon][metric] is not None
                and su[a][horizon][metric] is not None]
        if "native" in arms and "native" in su:
            arms = [a for a in arms
                    if a == "native" or arms_differ(su[a], su["native"], horizon)]
        if len(arms) < 2:
            continue
        wmv = [wm[a][horizon][metric] for a in arms]
        suv = [su[a][horizon][metric] for a in arms]
        tau = kendall_tau_b(wmv, suv)
        if tau is not None:
            taus.append({"world": world, "v": tau})
        # WM pick: min oriented prediction; ties prefer native then alphabetical
        order = sorted(range(len(arms)),
                       key=lambda i: (wmv[i], arms[i] != "native", arms[i]))
        pick = arms[order[0]]
        best = min(suv)
        regrets.append({"world": world, "v": suv[order[0]] - best})
        picks.append(pick)
        if "native" in arms:
            const_regrets.append(
                {"world": world, "v": suv[arms.index("native")] - best})
    return {"taus": taus, "regrets": regrets, "const_regrets": const_regrets,
            "picks": picks}


def analyse():
    groups, cells = load()
    glist, tables = build_tables(groups, cells)
    results = {"schema": "gate1_results/1",
               "n_groups": len(glist),
               "n_worlds": len({g["world"] for g in glist}),
               "models": {}, "horizons": list(HORIZONS_S),
               "metrics": list(METRICS),
               "primary_metric": PRIMARY_METRIC,
               "bootstrap": {"n": N_BOOT, "seed": SEED, "cluster": "world"},
               "seen_arms": sorted(SEEN_ARMS),
               "pair_counts": {}}

    # pair-type coverage (informative pairs per horizon)
    pcount = defaultdict(int)
    for g in glist:
        su = tables["recorded"].get(g["world"], {})
        for h in ("10", "30", "60"):
            for a, b in combinations(sorted(g["arms"]), 2):
                if a in su and b in su and arms_differ(su[a], su[b], h):
                    pcount[f"{h}|{a}-{b}"] += 1
    results["pair_counts"] = dict(sorted(pcount.items()))

    for model in MODELS + ("constant",):
        blk = {"pairs": {}, "ranking": {}}
        if model != "constant":
            rows_all = pair_rows(glist, tables, model)
            for h in ("10", "30", "60"):
                for m in METRICS:
                    for scope, keep in (
                            ("all", lambda r: True),
                            ("seen", lambda r: r["pair"][0] in SEEN_ARMS
                             and r["pair"][1] in SEEN_ARMS),
                            ("unseen", lambda r: r["pair"][0] not in SEEN_ARMS
                             or r["pair"][1] not in SEEN_ARMS)):
                        rows = [r for r in rows_all
                                if r["h"] == h and r["m"] == m and keep(r)]
                        if not rows:
                            blk["pairs"][f"{h}|{m}|{scope}"] = None
                            continue
                        st = pair_stats(rows)
                        by_world = defaultdict(list)
                        for r in rows:
                            by_world[r["world"]].append(r)

                        def stat(vals, sample):
                            agree = sum(1 for r in vals
                                        if sign(r["dwm"]) == sign(r["dsu"]))
                            return agree / len(vals)
                        st["sign_agreement_ci"] = boot_ci(
                            by_world, by_world, stat)
                        xs, ys = [r["dwm"] for r in rows], [r["dsu"] for r in rows]

                        def stat_pearson(vals, sample):
                            return pearson([v["dwm"] for v in vals],
                                           [v["dsu"] for v in vals])
                        def stat_spearman(vals, sample):
                            return spearman([v["dwm"] for v in vals],
                                            [v["dsu"] for v in vals])
                        st["pearson_ci"] = boot_ci(by_world, by_world, stat_pearson)
                        st["spearman_ci"] = boot_ci(by_world, by_world, stat_spearman)
                        blk["pairs"][f"{h}|{m}|{scope}"] = st
            blk["n_pair_rows_total"] = len(rows_all)
        for h in ("10", "30", "60"):
            for m in METRICS:
                if model == "constant":
                    rs = group_ranking_stats(glist, tables, "r5b", m, h)
                    reg = rs["const_regrets"]
                    blk["ranking"][f"{h}|{m}"] = {
                        "n_groups": len(reg),
                        "kendall_tau": None, "kendall_tau_ci": None,
                        "regret_mean": mean([r["v"] for r in reg]),
                        "regret_ci": boot_ci(
                            {r["world"]: [r["v"]] for r in reg},
                            {r["world"]: [r["v"]] for r in reg},
                            lambda vals, sample: sum(vals) / len(vals)),
                        "pick_rule": "always native"}
                    continue
                rs = group_ranking_stats(glist, tables, model, m, h)
                taus = rs["taus"]
                reg = rs["regrets"]
                by_world_t = {t["world"]: [t["v"]] for t in taus}
                by_world_r = {r["world"]: [r["v"]] for r in reg}
                blk["ranking"][f"{h}|{m}"] = {
                    "n_groups": len(reg),
                    "kendall_tau": mean([t["v"] for t in taus]),
                    "kendall_tau_ci": boot_ci(
                        by_world_t, by_world_t,
                        lambda vals, sample: sum(vals) / len(vals)),
                    "regret_mean": mean([r["v"] for r in reg]),
                    "regret_ci": boot_ci(
                        by_world_r, by_world_r,
                        lambda vals, sample: sum(vals) / len(vals)),
                    "always_native_regret_mean": mean(
                        [r["v"] for r in rs["const_regrets"]]),
                    "pick_counts": {p: rs["picks"].count(p)
                                    for p in sorted(set(rs["picks"]))}}
        results["models"][model] = blk

    # per-arm SUMO + WM outcome means (matched groups only)
    arm_summary = {}
    for arm in sorted({a for g in glist for a in g["arms"]}):
        n = sum(1 for g in glist if arm in g["arms"])
        entry = {"n_groups": n}
        for model in ("recorded",) + MODELS:
            vals = defaultdict(list)
            for g in glist:
                cell = tables[model].get(g["world"], {}).get(arm)
                if not cell:
                    continue
                for h in ("10", "30", "60"):
                    for m in METRICS:
                        v = cell[h][m]
                        if v is not None:
                            vals[f"{h}|{m}"].append(v)
            entry[model] = {k: (round(sum(v) / len(v), 3) if v else None)
                            for k, v in sorted(vals.items())}
        arm_summary[arm] = entry
    results["arm_summary_oriented"] = arm_summary

    # verdict: R5b, primary metric, 10 and 30 s, pooled pairs
    verdict = {"rule": ("supported iff at BOTH 10 s and 30 s (pooled pairs, "
                        f"{PRIMARY_METRIC}): sign-agreement CI lo > 0.5 AND "
                        "Kendall tau CI lo > 0; not supported iff at BOTH "
                        "horizons sign-agreement CI hi <= 0.5 OR tau CI hi <= 0; "
                        "else mixed"),
               "per_horizon": {}}
    flags = []
    for h in ("10", "30"):
        pairs = results["models"]["r5b"]["pairs"][f"{h}|{PRIMARY_METRIC}|all"]
        rank = results["models"]["r5b"]["ranking"][f"{h}|{PRIMARY_METRIC}"]
        sa, tau = pairs["sign_agreement_ci"], rank["kendall_tau_ci"]
        good = (sa and sa["lo"] is not None and sa["lo"] > 0.5
                and tau and tau["lo"] is not None and tau["lo"] > 0)
        bad = ((sa and sa["hi"] is not None and sa["hi"] <= 0.5)
               or (tau and tau["hi"] is not None and tau["hi"] <= 0))
        verdict["per_horizon"][h] = {"sign_agreement_ci": sa,
                                     "kendall_tau_ci": tau,
                                     "good": good, "bad": bad}
        flags.append((good, bad))
    if all(g for g, _ in flags):
        verdict["verdict"] = "supported"
    elif all(b for _, b in flags):
        verdict["verdict"] = "not supported"
    else:
        verdict["verdict"] = "mixed"
    results["verdict"] = verdict
    return results, glist, tables


def fmt(v, digits=2):
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def fmt_ci(ci, digits=2):
    if not ci or ci.get("lo") is None:
        return "n/a"
    return f"[{fmt(ci['lo'], digits)}, {fmt(ci['hi'], digits)}]"


def write_report(results, glist, groups):
    lines = []
    a = lines.append
    a("# WM-S C4 Gate 1 — action fidelity (CPU, existing data)")
    a("")
    a("## Question")
    a("On paired controller arms that start from the **same saved SUMO state**, does "
      "WM-S rank controllers (native / damping / PI / wall) the way SUMO does at "
      "**10 / 30 / 60 s**? Verdict wording target: *WM-S ranks controllers like SUMO "
      "at 10/30 s* — **supported / not supported / mixed**.")
    a("")
    a("## Data")
    pen = defaultdict(int)
    for g in glist:
        pen[g["world"].split("p")[1].split("t")[0]] += 1
    a(f"- **{results['n_groups']} paired groups** (world @ branch time t0), TRAIN "
      f"split only, from `pack_v3` + the re-aligned `pack_damping` snap "
      f"(damping arm); penetrations (%% CAV): "
      + ", ".join(f"{k}% x{v}" for k, v in sorted(pen.items()))
      + "; demands 2400-2700 veh/h.")
    a("- **Arms** per group where paired: `native`, `base` (CIRCLES damping), `pi`, "
      "`w-lo-early/late`, `w-mid-early/late` (wall v2 laws).")
    a("- **Pairing rule**: an arm is kept only if its t0 frame equals the native "
      "t0 frame within tolerance (same vehicle ids and lanes, max |ds| <= 1 m, "
      "max |dv| <= 1 m/s); arms branched from a different physical state "
      "(e.g. an already-engaged early wall) are dropped for that group "
      "(`groups_gate1.json`). Matched arm-group counts: native 39, w-*-late 39+39, "
      "base 17, pi 16, w-*-early 12+12.")
    a("- **Informative-pair rule (data finding)**: inside the 60 s window most "
      "wall branches are still trajectory-identical to native (the wall "
      "assembles later than the window), so their paired differences are "
      "vacuous. A pair (or a group-ranking arm) enters horizon h only if the "
      "two arms' recorded SUMO outcomes differ at h. Informative pair counts "
      "per horizon and pair type: " + json.dumps(results["pair_counts"]) + ".")
    a("- **SUMO outcome** per arm: computed from the recorded arm trajectory "
      "(60 s); **WM outcome** per arm: one WM-S rollout from the identical t0 "
      "state with CAVs teacher-forced to that arm's recorded CAV trajectory "
      "(HDVs predicted, entries off, 0.5 s steps).")
    a("- **Models**: WM-S R5b best "
      "(`WM-S-R5B-20261012/box/gpuA/r5b_main_a1/wm_s_best.pt`, trained WITHOUT PI "
      "and wall arms), WM-S C1 (`data_cache/ckpt_c1fast/wm_s.pt`; PI/wall unseen "
      "too); references: **physics-only** (R5b network with the learned residual "
      "and lane correction gated to zero, population-default driver parameters = "
      "IDM/MOBIL default settings) and a **constant predictor** (always native).")
    a("- **Seen vs unseen**: both arms in {native, base} = seen in WM training; "
      "any arm in {pi, wall x4} = unseen (excluded from both models' training "
      "corpora, `lpwo_v3`).")
    a("- **Outcomes at 10/30/60 s** (HDVs of the t0 roster): cumulative delay over "
      "the merge zone 0-75.44 m (vehicle-seconds, free speed 27.78 m/s), mean "
      "HDV speed in the zone, HDV hard-brake events (runs of a < -3 m/s^2), "
      "throughput past the merge (t0 vehicles crossing s = 75.44). Metrics are "
      "oriented lower-is-better (speed/throughput negated) before differencing.")
    a("- **Statistics**: pairwise sign agreement, Pearson and Spearman of paired "
      "differences (A-B, WM vs SUMO); Kendall tau-b of the arm ranking per group; "
      "regret of the WM-chosen arm (SUMO outcome of the WM pick minus SUMO best, "
      "oriented units). World-cluster bootstrap 95% CIs "
      f"(n={results['bootstrap']['n']}, seed {results['bootstrap']['seed']}).")
    a("")
    a("## Verdict")
    a(f"**{results['verdict']['verdict']}** for *WM-S ranks controllers like SUMO "
      "at 10/30 s* (primary metric HDV zone delay, R5b, pooled pairs).")
    a("")
    v = results["verdict"]["per_horizon"]
    a("| horizon | sign agreement [95% CI] | Kendall tau [95% CI] | passes |")
    a("|---|---|---|---|")
    for h in ("10", "30"):
        row = v[h]
        ok = "yes" if row["good"] else ("no" if row["bad"] else "inconclusive")
        a(f"| {h} s | {fmt(row['sign_agreement_ci']['lo'],3)}-"
          f"{fmt(row['sign_agreement_ci']['hi'],3)} (pt "
          f"{fmt(results['models']['r5b']['pairs'][f'{h}|{PRIMARY_METRIC}|all']['sign_agreement'],3)}) | "
          f"{fmt(row['kendall_tau_ci']['lo'],3)}-{fmt(row['kendall_tau_ci']['hi'],3)} "
          f"(pt {fmt(results['models']['r5b']['ranking'][f'{h}|{PRIMARY_METRIC}']['kendall_tau'],3)}) | {ok} |")
    a("")
    a("## Main table — HDV zone delay (paired differences, pooled pairs)")
    for h in ("10", "30", "60"):
        a(f"### {h} s")
        a("| model | n pairs (worlds) | sign agreement [CI] | Pearson [CI] | "
          "Spearman [CI] | Kendall tau [CI] | regret [CI] | always-native regret |")
        a("|---|---|---|---|---|---|---|---|")
        for model in MODELS + ("constant",):
            blk = results["models"][model]
            if model == "constant":
                rk = blk["ranking"][f"{h}|{PRIMARY_METRIC}"]
                a(f"| {MODEL_LABELS[model]} | - | - | - | - | - | "
                  f"{fmt(rk['regret_mean'],1)} {fmt_ci(rk['regret_ci'],1)} | "
                  f"{fmt(rk['regret_mean'],1)} |")
                continue
            pr = blk["pairs"][f"{h}|{PRIMARY_METRIC}|all"]
            rk = blk["ranking"][f"{h}|{PRIMARY_METRIC}"]
            a(f"| {MODEL_LABELS[model]} | {pr['n_pairs']} ({pr['n_worlds']}) | "
              f"{fmt(pr['sign_agreement'],3)} {fmt_ci(pr['sign_agreement_ci'],3)} | "
              f"{fmt(pr['pearson'],2)} {fmt_ci(pr['pearson_ci'],2)} | "
              f"{fmt(pr['spearman'],2)} {fmt_ci(pr['spearman_ci'],2)} | "
              f"{fmt(rk['kendall_tau'],3)} {fmt_ci(rk['kendall_tau_ci'],3)} | "
              f"{fmt(rk['regret_mean'],1)} {fmt_ci(rk['regret_ci'],1)} | "
              f"{fmt(rk['always_native_regret_mean'],1)} |")
        a("")
    a("## Seen vs unseen arms (HDV zone delay, R5b)")
    a("| horizon | scope | n pairs | sign agreement [CI] | Spearman [CI] |")
    a("|---|---|---|---|---|")
    for h in ("10", "30", "60"):
        for scope in ("seen", "unseen"):
            pr = results["models"]["r5b"]["pairs"][f"{h}|{PRIMARY_METRIC}|{scope}"]
            if pr is None:
                a(f"| {h} s | {scope} | 0 | n/a | n/a |")
                continue
            a(f"| {h} s | {scope} | {pr['n_pairs']} | "
              f"{fmt(pr['sign_agreement'],3)} {fmt_ci(pr['sign_agreement_ci'],3)} | "
              f"{fmt(pr['spearman'],2)} {fmt_ci(pr['spearman_ci'],2)} |")
    a("")
    a("## All metrics — sign agreement (pooled pairs) and regret, R5b vs references")
    a("| horizon | metric | R5b sign [CI] | C1 sign [CI] | physics sign [CI] | "
      "R5b regret | C1 regret | physics regret | const regret |")
    a("|---|---|---|---|---|---|---|---|---|")
    for h in ("10", "30", "60"):
        for m in METRICS:
            cells_ = []
            for model in MODELS:
                pr = results["models"][model]["pairs"][f"{h}|{m}|all"]
                cells_.append(f"{fmt(pr['sign_agreement'],3)} "
                              f"{fmt_ci(pr['sign_agreement_ci'],3)}")
            regrets = []
            for model in MODELS + ("constant",):
                rk = results["models"][model]["ranking"][f"{h}|{m}"]
                regrets.append(fmt(rk["regret_mean"], 1))
            a(f"| {h} s | `{m}` | " + " | ".join(cells_) + " | "
              + " | ".join(regrets) + " |")
    a("")
    a("Regret is in oriented units (delay: vehicle-seconds; brakes: events; "
      "speed/throughput: negated m/s / vehicles). Constant predictor regret "
      "= regret of always picking native.")
    a("")
    a("## SUMO vs WM per-arm means (HDV zone delay 30 s, vehicle-seconds, "
      "matched groups)")
    a("| arm | n | SUMO | R5b | C1 | physics |")
    a("|---|---|---|---|---|---|")
    for arm, entry in results["arm_summary_oriented"].items():
        def g(model):
            v = entry[model].get(f"30|{PRIMARY_METRIC}")
            return fmt(v, 0) if v is not None else "n/a"
        a(f"| {arm} | {entry['n_groups']} | {g('recorded')} | {g('r5b')} | "
          f"{g('c1')} | {g('physics')} |")
    a("")
    a("(Vehicle-seconds of HDV zone delay; lower = less delay = better for every "
      "column. The WM columns are the model's own predicted delay.)")
    a("")
    a("## Caveats")
    a("- **Wall evidence inside the window is thin**: the saved wall branches "
      "assemble after the 60 s window in most groups (w-lo/mid-late are "
      "trajectory-identical to native throughout; w-*-early diverge only near "
      "60 s in a few groups), so Gate 1 mainly tests native vs damping vs PI. "
      "Wall-action fidelity needs branches saved at assembly-minus-10..30 s or "
      "a longer window (next action).")
    a("- Small n after the informative-pair filter (100/135/156 pairs on "
      "15/19/21 worlds at 10/30/60 s); CIs are wide.")
    a("- Exact ties (WM delta 0 and SUMO delta 0, e.g. equal integer "
      "throughput or equal hard-brake counts) count as agreements once the "
      "pair is informative.")
    a("- The physics-only reference is as good as or better than both learned "
      "models on sign agreement and Kendall tau at 10-30 s: on these pairs the "
      "learned residual does not yet add ranking fidelity over its own physics "
      "base.")
    a("- WM rollouts see no post-t0 entrants (entries off) and under-predict "
      "the delay level (~8% lower than SUMO); only paired differences carry "
      "the Gate-1 signal.")
    a("")
    a("## Reproduce")
    a("```")
    a("cd E:/agent-work/c4-gate1")
    a("set PYTHONPATH=src")
    a("C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe "
      "runs/active/WM-S-C4-GATE1-20261012/build_states.py")
    a("C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe "
      "runs/active/WM-S-C4-GATE1-20261012/run_rollouts.py --workers 4")
    a("C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe "
      "runs/active/WM-S-C4-GATE1-20261012/analyze.py")
    a("```")
    a("")
    a("Inputs (read-only): `E:/research-dense/data_cache/pack_v3`, "
      "`pack_damping`, `sidecars_packv3_v2/identities`, `identities_damping`, "
      "`ckpt_c1fast/wm_s.pt`, "
      "`runs/active/WM-S-R5B-20261012/box/gpuA/r5b_main_a1/wm_s_best.pt`. "
      "TRAIN split only; no sealed/CAL/holdout/chunk5/J4B/INC1 path is read; "
      "no SUMO, no GPU, no training. `states_gate1.json` (148 MB) is "
      "regenerable and not committed.")
    a("")
    (RUN / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (RUN / "results_gate1.json").write_text(json.dumps(results, indent=1),
                                            encoding="utf-8")


def main() -> int:
    results, glist, tables = analyse()
    write_report(results, glist, tables)
    sizes = {p.name: p.stat().st_size
             for p in RUN.glob("*.json") if p.name != "states_gate1.json"}
    print("wrote REPORT.md and results_gate1.json; verdict:",
          results["verdict"]["verdict"])
    print("json sizes (bytes):", json.dumps(sizes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
