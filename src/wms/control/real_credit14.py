"""S-SUMOCREDIT (2): real-SUMO cooperation credit from the loop's policy-pair branches.

Per decision point of ``collect --policy-pairs --credit-horizon H`` (traffic_shepherd.c4.policy_pairs)
SUMO runs, from ONE saved state with the same RNG, the arms ``B`` (all native), ``ALL`` (the
policy on every CAV; = the POLICY arm), ``ONLY:i`` (only CAV i of the close group G on the policy),
``GROUP`` (only G) and optionally ``FLIP:i`` (G minus i).  With ``J_X`` = the authoritative SUMO
ledger over the first H seconds (on-road speed-deficit + source-waiting vehicle-seconds), the
credit uses the sign of ``traffic_shepherd.c4.branching`` / ``wms.control.coop_imagine``
(``team_delay_vs`` / ``road_delay_vs`` are costs, SIGN -1; D POSITIVE = BETTER):

    D_X = J_B - J_X ;  C_G = D_G - sum_{i in G} D_i ;  M_i = D_G - D_{G\\i}

The same arms are imagined in WM-S from the same (oracle) state (``confident_gate.ArmImagination``,
reference = learned native head, common random numbers), giving imagined D_X.

Correction fed to the trainer (``Policy14Trainer.credit_multipliers``, flag-gated): for every
credit type X the OLS slope ``b_X`` of real on imagined D_X across points estimates how much of
an imagined credit gradient is real (E[D_real | D_imag] ~ a + b D_imag => grad E[D_real] ~
b grad D_imag).  Its weight is set by the measured sampling variance only: the multiplier is the
one-sided 97.5 % UPPER confidence bound of the slope, ``m_X = clip(b_X + 1.96 SE_X, 0, 1)`` (1.96 =
the deployment check's confidence_z).  So the WM credit is only scaled down where SUMO
significantly contradicts its scale (an imprecise slope leaves m = 1), it is never amplified and
never sign-flipped (a significantly negative slope gives 0).  No estimate (n < 3 or no imagined
spread) leaves the weight unchanged (multiplier 1).
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

POINT_SCHEMA = "c4_real_credit_point/1"
SUMMARY_SCHEMA = "c4_real_credit/1"
TYPES = ("all", "individual", "group", "coop", "team")
#: credit types whose actor weight the correction may scale (``all`` = the main objective, reported)
CORRECTED = ("individual", "group", "coop", "team")
EVAL_SEEDS = (964000, 964199)
#: two-sided 95 % normal quantile (wms.control.deploy --confidence-z default)
CONFIDENCE_Z = 1.96


def credit_terms(J: Mapping[str, float], group: Sequence[str], flips: Sequence[str] = ()) -> dict:
    """D/C/M terms (positive = better) from arm costs ``J`` (``B``, ``ALL``, ``ONLY:i``, ``GROUP``,
    ``FLIP:i``).  Missing arms give None."""
    if "B" not in J:
        raise ValueError("the all-native reference arm B is required")
    b = float(J["B"])

    def d(name):
        return None if J.get(name) is None else b - float(J[name])
    out = {"D_all": d("ALL"), "D_i": {v: d(f"ONLY:{v}") for v in group}, "D_G": d("GROUP"),
           "D_G_minus_i": {v: d(f"FLIP:{v}") for v in flips}, "sum_D_i": None, "C_G": None, "M_i": {}}
    if group and out["D_G"] is not None and all(out["D_i"][v] is not None for v in group):
        out["sum_D_i"] = sum(out["D_i"][v] for v in group)
        out["C_G"] = out["D_G"] - out["sum_D_i"]
    for v in flips:
        if out["D_G"] is not None and out["D_G_minus_i"][v] is not None:
            out["M_i"][v] = out["D_G"] - out["D_G_minus_i"][v]
    return out


def pairs_by_type(points: Sequence[Mapping]) -> dict:
    """``{type: [(imagined, real, seed), ...]}`` over points with both SUMO and imagined arms."""
    out = {t: [] for t in TYPES}
    for p in points:
        if p.get("imagined") is None or p.get("sumo") is None:
            continue
        group, flips = list(p.get("group") or []), list(p.get("flips") or [])
        real = credit_terms(p["sumo"], group, flips)
        imag = credit_terms(p["imagined"], group, flips)
        seed = int(p["seed"])

        def add(t, x, y):
            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                out[t].append((float(x), float(y), seed))
        add("all", imag["D_all"], real["D_all"])
        add("group", imag["D_G"], real["D_G"])
        add("coop", imag["C_G"], real["C_G"])
        for v in group:
            add("individual", imag["D_i"].get(v), real["D_i"].get(v))
        for v in flips:
            add("team", imag["M_i"].get(v), real["M_i"].get(v))
    return out


def slope_multiplier(pairs: Sequence[tuple]) -> dict:
    """OLS slope of real on imagined credit; multiplier = clip(upper 97.5 % bound, 0, 1)."""
    n = len(pairs)
    out = {"n": n, "slope": None, "slope_se": None, "intercept": None, "corr": None,
           "multiplier": 1.0, "estimated": False}
    if n < 3:
        out["reason"] = "n < 3"
        return out
    x = np.array([p[0] for p in pairs], np.float64)
    y = np.array([p[1] for p in pairs], np.float64)
    sxx = float(((x - x.mean()) ** 2).sum())
    if sxx <= 0:
        out["reason"] = "no imagined spread"
        return out
    slope = float(((x - x.mean()) * (y - y.mean())).sum() / sxx)
    intercept = float(y.mean() - slope * x.mean())
    resid = y - intercept - slope * x
    se = float(math.sqrt(float((resid ** 2).sum()) / (n - 2) / sxx))
    upper = slope + CONFIDENCE_Z * se
    out.update(slope=slope, slope_se=se, intercept=intercept, slope_upper=upper,
               corr=(float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None),
               multiplier=float(min(1.0, max(0.0, upper))), estimated=True)
    return out


def variance_report(points: Sequence[Mapping], *, cycle_budget_sim_s: float = 29_776.0) -> dict:
    """Spread of the REAL credit and the points / sim-s needed for useful precision.

    ``n_for_mean``: points so that |mean| >= 2 SE (sign resolved); ``n_for_slope``: points so that
    the slope SE <= |slope| / 2 (given the measured residual spread)."""
    real = {t: [] for t in ("D_all", "D_i", "D_G", "C_G", "M_i")}
    cost = []
    for p in points:
        if p.get("sumo") is None:
            continue
        group, flips = list(p.get("group") or []), list(p.get("flips") or [])
        terms = credit_terms(p["sumo"], group, flips)
        for key in ("D_all", "D_G", "C_G"):
            if terms[key] is not None:
                real[key].append(terms[key])
        real["D_i"] += [v for v in terms["D_i"].values() if v is not None]
        real["M_i"] += [v for v in terms["M_i"].values() if v is not None]
        cost.append(float(p.get("extra_sim_s", 0.0)))
    per_point = float(np.mean(cost)) if cost else None
    out = {"points": len(cost), "extra_sim_s_per_point": per_point, "terms": {}}
    for key, vals in real.items():
        x = np.asarray(vals, np.float64)
        row = {"n": int(x.size)}
        if x.size >= 2:
            mean, sd = float(x.mean()), float(x.std(ddof=1))
            row.update(mean=mean, sd=sd, se=sd / math.sqrt(x.size))
            if mean != 0:
                need = int(math.ceil(4.0 * sd * sd / (mean * mean)))
                row["n_for_mean"] = need
                if per_point:
                    row["sim_s_for_mean"] = need * per_point
                    row["share_of_cycle_budget"] = need * per_point / cycle_budget_sim_s
        out["terms"][key] = row
    pairs = pairs_by_type(points)
    for t, pr in pairs.items():
        fit = slope_multiplier(pr)
        if fit["estimated"] and fit["slope"]:
            need = int(math.ceil(fit["n"] * (fit["slope_se"] / (abs(fit["slope"]) / 2.0)) ** 2))
            fit["n_for_slope"] = max(3, need)
            if per_point:
                fit["sim_s_for_slope"] = fit["n_for_slope"] * per_point
        out.setdefault("slopes", {})[t] = fit
    return out


def read_points(paths: Sequence) -> list[dict]:
    from .start_states import checked_path
    points = []
    for path in paths:
        for line in checked_path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            if p.get("schema") != POINT_SCHEMA:
                raise ValueError(f"{path}: not a real-credit point")
            seed = int(p["seed"])
            if EVAL_SEEDS[0] <= seed <= EVAL_SEEDS[1]:
                raise ValueError(f"evaluation seed {seed} in real-credit data")
            if p.get("split") != "TRAIN":
                raise ValueError("real-credit points must be TRAIN")
            if p.get("collisions", 0) or p.get("teleports", 0):
                continue                  # a SUMO incident makes the ledger non-authoritative
            points.append(p)
    return points


def fit_credit(paths: Sequence, *, cycle_budget_sim_s: float = 29_776.0) -> dict:
    points = read_points(paths)
    pairs = pairs_by_type(points)
    fits = {t: slope_multiplier(pairs[t]) for t in TYPES}
    return {"schema": SUMMARY_SCHEMA, "split": "TRAIN", "sources": [str(p) for p in paths],
            "points": len(points), "sign": "D positive = better (J_B - J_X, veh-s)",
            "fits": fits, "multipliers": {t: fits[t]["multiplier"] for t in CORRECTED},
            "variance": variance_report(points, cycle_budget_sim_s=cycle_budget_sim_s)}


def load_multipliers(path) -> dict:
    """Trainer-side reader: ``{individual, group, coop, team}`` multipliers in [0, 1]."""
    from .start_states import checked_path
    doc = json.loads(checked_path(path).read_text(encoding="utf-8"))
    if doc.get("schema") != SUMMARY_SCHEMA or doc.get("split") != "TRAIN":
        raise ValueError("not a TRAIN real-credit summary")
    out = {}
    for t in CORRECTED:
        m = float(doc["multipliers"][t])
        if not (math.isfinite(m) and 0.0 <= m <= 1.0):
            raise ValueError("real-credit multipliers must lie in [0, 1]")
        out[t] = m
    return out


def gather(collect_root, out_credit, out_gate) -> dict:
    """Concatenate the per-job SUMOCREDIT_*.jsonl files of a collection (never overwrites)."""
    root = Path(collect_root)
    counts = {}
    for name, target in (("SUMOCREDIT_POINTS.jsonl", out_credit), ("GATE_CALIBRATION.jsonl", out_gate)):
        if target is None:
            continue
        target = Path(target)
        if target.exists():
            raise ValueError(f"{target} exists (never overwritten)")
        lines = []
        for path in sorted(root.rglob(name)):
            lines += [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
        counts[name] = len(lines)
    return counts


def main(argv=None) -> int:
    import argparse

    from .start_states import checked_path
    ap = argparse.ArgumentParser(description="Real-SUMO cooperation credit (S-SUMOCREDIT)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gather")
    g.add_argument("--collect-root", required=True)
    g.add_argument("--out-credit", default=None)
    g.add_argument("--out-gate", default=None)
    f = sub.add_parser("fit")
    f.add_argument("--points", nargs="+", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--cycle-budget-sim-s", type=float, default=29_776.0)
    args = ap.parse_args(argv)
    if args.cmd == "gather":
        print(json.dumps(gather(checked_path(args.collect_root), args.out_credit, args.out_gate)))
        return 0
    out = checked_path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists (never overwritten)")
    doc = fit_credit(args.points, cycle_budget_sim_s=args.cycle_budget_sim_s)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, allow_nan=False), encoding="utf-8")
    print(json.dumps({"points": doc["points"], "multipliers": doc["multipliers"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
