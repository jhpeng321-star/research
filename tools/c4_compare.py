#!/usr/bin/env python
"""Reproducible paired SUMO comparison of C4 controllers.

    python tools/c4_compare.py run --controllers native damping pi mappo=path/policy.pt \\
        --penetrations 0.05,0.2 --demands 2400,2700 --seeds 4107-4114 --warmup 300 --horizon 600 \\
        --workers 6 --out runs/active/C4-COMPARE-x [--stochastic]
    python tools/c4_compare.py summarize --out runs/active/C4-COMPARE-x

``run`` calls ``python -m traffic_shepherd.c4.collect`` once per controller into ``<out>/<label>``
(``--no-physics-log``; resumable because collect skips complete episodes).  A controller is a name
(native|damping|pi|wall|random|zero) or ``label=path/to/policy.pt``; labels must be unique.
``summarize`` pairs episodes by (seed, demand, penetration) across all controllers and writes
``COMPARE.md`` + ``compare.json``: per-controller means and paired differences (treatment - native)
with 95 % bootstrap CIs (resampling paired episodes, 2000 draws, seed 0).
Total wait = window speed_deficit_vehicle_seconds + window source_wait_vehicle_seconds (primary;
lower is better).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

FORBIDDEN = ("holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "sealed")
NAMED = ("native", "damping", "pi", "wall", "random", "zero")
BASELINE = "native"
N_BOOT = 2000
BOOT_SEED = 0
# metric -> (label, direction): "lower" means lower is better, "higher" the opposite
METRICS = {
    "total_wait": ("total wait (veh-s)", "lower"),
    "delay": ("on-road delay (veh-s)", "lower"),
    "source_wait": ("entry waiting (veh-s)", "lower"),
    "arrivals": ("arrivals", "higher"),
    "braking": ("braking (veh-s)", "lower"),
    "ttc3": ("TTC<3 s (veh-s)", "lower"),
    "collisions": ("collisions", "lower"),
    "teleports": ("teleports", "lower"),
}


def _check_path(p) -> None:
    low = str(p).lower()
    if any(tok in low for tok in FORBIDDEN):
        raise SystemExit(f"refusing a path with a restricted-split token: {p}")


def parse_controllers(items: list[str]) -> list[tuple[str, str]]:
    """-> [(label, spec)]; spec is the collect --policy value.  Labels must be unique."""
    out, seen = [], set()
    for item in items:
        if "=" in item:
            label, spec = item.split("=", 1)
            if not label or not spec:
                raise SystemExit(f"bad controller {item!r}; expected label=path")
            _check_path(spec)
        elif item in NAMED:
            label, spec = item, item
        else:
            raise SystemExit(f"controller {item!r} is not one of {NAMED}; use label=path/to/policy.pt")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise SystemExit(f"label {label!r} must match [A-Za-z0-9_.-]+")
        if label in seen:
            raise SystemExit(f"duplicate controller label {label!r}")
        seen.add(label)
        out.append((label, spec))
    return out


# ---------------------------------------------------------------- run
def build_command(spec: str, out_dir: Path, a, python: str) -> list[str]:
    return [python, "-m", "traffic_shepherd.c4.collect", "--policy", spec,
            "--penetrations", a.penetrations, "--demands", a.demands, "--seeds", a.seeds,
            "--warmup", str(a.warmup), "--horizon", str(a.horizon), "--workers", str(a.workers),
            "--out", str(out_dir), "--no-physics-log"]


def cmd_run(a) -> int:
    ctrls = parse_controllers(a.controllers)
    out = Path(a.out).resolve()
    _check_path(out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = "1"
    if a.stochastic:
        env["C4_POLICY_STOCHASTIC"] = "1"
    else:
        env.pop("C4_POLICY_STOCHASTIC", None)
    manifest = dict(controllers={lab: spec for lab, spec in ctrls}, penetrations=a.penetrations,
                    demands=a.demands, seeds=a.seeds, warmup=a.warmup, horizon=a.horizon,
                    workers=a.workers, stochastic=bool(a.stochastic))
    (out / "compare_run.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    rc = 0
    for label, spec in ctrls:
        cmd = build_command(spec, out / label, a, a.python)
        print(f"[compare] {label}: {' '.join(cmd)}", flush=True)
        if a.dry_run:
            continue
        r = subprocess.run(cmd, env=env)
        if r.returncode != 0:
            print(f"[compare] {label} exited {r.returncode}", file=sys.stderr, flush=True)
            rc = r.returncode
    return rc


# ---------------------------------------------------------------- summarize
def episode_row(d: dict) -> dict:
    w = d["window"]
    delay = float(w["speed_deficit_vehicle_seconds"])
    sw = float(w["source_wait_vehicle_seconds"])
    return dict(total_wait=delay + sw, delay=delay, source_wait=sw, arrivals=float(w["arrivals"]),
                braking=float(w["braking_vehicle_seconds"]),
                ttc3=float(w["ttc_below_3_vehicle_seconds"]),
                collisions=float(d.get("collisions", 0)), teleports=float(d.get("teleports", 0)))


def load_controller(root: Path) -> dict:
    data = {}
    for f in sorted(root.glob("episodes/*/*/summary.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        key = (int(d["seed"]), int(d["demand_vph"]), round(float(d["penetration"]), 4))
        data[key] = episode_row(d)
    return data


def load_all(out: Path, labels: list[str] | None = None) -> dict:
    if labels is None:
        labels = sorted(p.name for p in out.iterdir() if (p / "episodes").is_dir())
    return {lab: load_controller(out / lab) for lab in labels}


def paired_keys(data: dict) -> list[tuple]:
    sets = [set(v) for v in data.values()]
    if not sets or any(not s for s in sets):
        return []
    return sorted(set.intersection(*sets))


def bootstrap_ci(diffs, n_boot: int = N_BOOT, seed: int = BOOT_SEED, level: float = 0.95) -> dict:
    """Mean of paired differences with a percentile bootstrap CI (resample pairs)."""
    x = np.asarray(diffs, dtype=float)
    n = len(x)
    if n == 0:
        return dict(n=0, mean=None, lo=None, hi=None)
    mean = float(x.mean())
    if n == 1:
        return dict(n=1, mean=mean, lo=None, hi=None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    a = (1 - level) / 2
    lo, hi = np.quantile(means, [a, 1 - a])
    return dict(n=n, mean=mean, lo=float(lo), hi=float(hi))


def paired_diff(data: dict, label: str, keys: list, metric: str, base: str = BASELINE) -> dict:
    return bootstrap_ci([data[label][k][metric] - data[base][k][metric] for k in keys])


def build_report(data: dict, base: str = BASELINE) -> dict:
    keys = paired_keys(data)
    rep = dict(baseline=base, controllers=list(data), n_paired=len(keys),
               episodes_per_controller={lab: len(v) for lab, v in data.items()},
               bootstrap=dict(draws=N_BOOT, seed=BOOT_SEED, level=0.95),
               sign="difference = controller - native; negative is better except arrivals",
               means={}, diffs={}, by_penetration={}, by_demand={})
    if not keys:
        return rep
    for lab in data:
        rep["means"][lab] = {m: float(np.mean([data[lab][k][m] for k in keys])) for m in METRICS}
    if base not in data:
        return rep
    others = [lab for lab in data if lab != base]

    def block(ks):
        return {lab: {m: paired_diff(data, lab, ks, m, base) for m in METRICS} for lab in others}
    rep["diffs"] = block(keys)
    for pen in sorted({k[2] for k in keys}):
        ks = [k for k in keys if k[2] == pen]
        rep["by_penetration"][str(pen)] = dict(n=len(ks), diffs=block(ks))
    for dem in sorted({k[1] for k in keys}):
        ks = [k for k in keys if k[1] == dem]
        rep["by_demand"][str(dem)] = dict(n=len(ks), diffs=block(ks))
    base_tw = rep["means"][base]["total_wait"]
    rep["total_wait_percent_vs_native"] = {
        lab: (100.0 * rep["diffs"][lab]["total_wait"]["mean"] / base_tw if base_tw else None)
        for lab in others}
    return rep


def _fmt_ci(d: dict, nd: int = 1) -> str:
    if d["mean"] is None:
        return "n/a"
    if d["lo"] is None:
        return f"{d['mean']:+.{nd}f}"
    return f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f}, {d['hi']:+.{nd}f}]"


def _diff_table(diffs: dict, metrics: list[str]) -> list[str]:
    head = "| controller | " + " | ".join(METRICS[m][0] for m in metrics) + " |"
    lines = [head, "|" + "---|" * (len(metrics) + 1)]
    for lab, row in diffs.items():
        cells = [_fmt_ci(row[m], 2 if m in ("collisions", "teleports") else 1) for m in metrics]
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    return lines


def render_md(rep: dict) -> str:
    lines = ["# C4 paired comparison", "",
             f"Paired episodes (seed, demand, penetration) common to all controllers: "
             f"**{rep['n_paired']}**. Episodes per controller: {rep['episodes_per_controller']}.", "",
             "Total wait = window speed_deficit_vehicle_seconds + window source_wait_vehicle_seconds "
             "(primary; lower is better). Differences = controller - native, mean with 95 % bootstrap "
             f"CI ({rep['bootstrap']['draws']} draws over paired episodes, seed "
             f"{rep['bootstrap']['seed']}). Negative is better except for arrivals (positive is "
             "better).", ""]
    if not rep["means"]:
        return "\n".join(lines + ["No common paired episodes."]) + "\n"
    ms = list(METRICS)
    lines += ["## Means over paired episodes", "",
              "| controller | " + " | ".join(METRICS[m][0] for m in ms) + " |",
              "|" + "---|" * (len(ms) + 1)]
    for lab, row in rep["means"].items():
        cells = [f"{row[m]:.2f}" if m in ("collisions", "teleports") else f"{row[m]:.1f}" for m in ms]
        lines.append(f"| {lab} | " + " | ".join(cells) + " |")
    if rep["diffs"]:
        lines += ["", "## Paired difference vs native (all episodes)", ""] + _diff_table(rep["diffs"], ms)
        pct = rep.get("total_wait_percent_vs_native", {})
        if pct:
            lines += ["", "Total wait vs native mean: "
                      + ", ".join(f"{lab} {v:+.1f} %" for lab, v in pct.items() if v is not None)]
        for title, key in (("penetration", "by_penetration"), ("demand (veh/h)", "by_demand")):
            lines += ["", f"## Split by {title}"]
            for val, blk in rep[key].items():
                lines += ["", f"### {title} = {val} (n = {blk['n']})", ""] + _diff_table(blk["diffs"], ms)
    return "\n".join(lines) + "\n"


def cmd_summarize(a) -> int:
    out = Path(a.out).resolve()
    _check_path(out)
    data = load_all(out, a.controllers or None)
    if not data:
        raise SystemExit(f"no controller directories with episodes under {out}")
    rep = build_report(data, a.baseline)
    (out / "compare.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    (out / "COMPARE.md").write_text(render_md(rep), encoding="utf-8")
    print(render_md(rep))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--controllers", nargs="+", required=True)
    r.add_argument("--penetrations", default="0.05,0.1,0.2,0.4")
    r.add_argument("--demands", default="2400,2600,2700")
    r.add_argument("--seeds", required=True, help="e.g. 4107-4112 (exposed 4107-4148)")
    r.add_argument("--warmup", type=int, default=300)
    r.add_argument("--horizon", type=int, default=600)
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--out", required=True)
    r.add_argument("--stochastic", action="store_true", help="set C4_POLICY_STOCHASTIC=1 (default off)")
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--dry-run", action="store_true", help="print commands only")
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    s.add_argument("--controllers", nargs="*", help="labels to include (default: every dir with episodes)")
    s.add_argument("--baseline", default=BASELINE)
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_summarize(a)


if __name__ == "__main__":
    raise SystemExit(main())
