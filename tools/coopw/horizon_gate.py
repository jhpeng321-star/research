#!/usr/bin/env python3
"""S-COOPW horizon gate: WM paired action-difference accuracy at imagination horizons (CPU, no SUMO).

On a fixed panel of TRAIN-role paired branch groups (``pairs300``-style source: one native arm
``B`` and >= 1 action arms ``SEED`` branched from the same SUMO saveState), every arm is rolled out
by the WM from the branch point with its recorded CAV commands and common seeds (the LOOP14
acceptance ``free_rollout``), and for each horizon h the t0-HDV delay difference
dJ = J(arm) - J(B) (``wm_eval_common.delay_proxy``, the loop's paired-difference measure) is
compared with SUMO's: sign agreement (pairs with dJ_SUMO != 0) and Pearson r, with 95 % group
bootstrap intervals.

Gate rule (registered before use; no hand-picked number): the threshold is the lower 95 % bound
of the sign agreement at the BASE horizon (the 10 s horizon the policy already trains on) of the
WM the curriculum starts from (``--register``).  A WM passes horizon h when its sign agreement at
h is >= that threshold, i.e. its long-horizon credit is at least as reliable as the base-horizon
credit was judged to be.  Later WMs are gated with the REGISTERED threshold (``--threshold``).

    python tools/coopw/horizon_gate.py --wm-ckpt CK --wm-config CFG --pairs ROOT \\
        --groups 48 --horizons 10 30 60 --register --out GATE.json          # registration run
    python tools/coopw/horizon_gate.py ... --threshold REGISTERED_GATE.json --out GATE.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("COOPW_ROOT", Path(__file__).resolve().parents[2]))
for _p in (ROOT, ROOT / "src", ROOT / "tools", ROOT / "tools" / "r10b_checks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SCHEMA = "coopw_horizon_gate/1"
DT_S = 0.5
SEED0 = 140014


def _accept():
    spec = importlib.util.spec_from_file_location("loop14_accept", ROOT / "tools/loop14_checks/accept.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def panel_groups(ds, n, seed):
    """Branch-time windows grouped by pair key: {key: {"B": idx, "SEED": [idx, ...]}}."""
    entries = ds._loader.entries
    groups = {}
    for i, (ep, tick) in enumerate(ds.positions):
        e = entries[ep]
        branch = e.get("branch") or {}
        if not e.get("pair_key"):
            continue
        if abs(float(ds._loader.episode(ep).array("time_s")[tick]) - float(e["warmup_s"])) > 1e-6:
            continue
        kind = branch.get("kind")
        group = groups.setdefault(e["pair_key"], {"B": None, "SEED": []})
        if kind == "B":
            group["B"] = i
        elif kind is not None:
            group["SEED"].append(i)
    keys = sorted(k for k, g in groups.items() if g["B"] is not None and g["SEED"])
    order = np.random.default_rng(seed).permutation(len(keys))
    return [(keys[i], groups[keys[i]]) for i in sorted(order[:n])]


def bootstrap(values_by_group, stat, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    groups = list(values_by_group)
    point = stat([v for g in groups for v in g])
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        sample = [v for i in pick for v in groups[i]]
        value = stat(sample)
        if value is not None and np.isfinite(value):
            draws.append(value)
    lo, hi = (np.quantile(draws, [0.025, 0.975]).tolist() if draws else [None, None])
    return point, [lo, hi]


def sign_agreement(pairs):
    kept = [(w, s) for w, s in pairs if s != 0]
    return float(np.mean([np.sign(w) == np.sign(s) for w, s in kept])) if kept else None


def pearson(pairs):
    if len(pairs) < 3:
        return None
    a = np.asarray(pairs, float)
    if a[:, 0].std() == 0 or a[:, 1].std() == 0:
        return None
    return float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wm-ckpt", required=True)
    ap.add_argument("--wm-config", required=True)
    ap.add_argument("--pairs", required=True, help="converted paired-branch TRAIN source")
    ap.add_argument("--groups", type=int, default=48)
    ap.add_argument("--horizons", type=float, nargs="+", default=[10.0, 30.0, 60.0])
    ap.add_argument("--panel-seed", type=int, default=20261026)
    ap.add_argument("--register", action="store_true",
                    help="derive the threshold from THIS WM's base-horizon lower 95 %% bound")
    ap.add_argument("--threshold", default=None, help="registered gate JSON (threshold source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=1, help="fixed K CRN draws (legacy: 1)")
    ap.add_argument("--deterministic-crn", action="store_true")
    ap.add_argument("--entrant-control", action="store_true")
    ap.add_argument("--entrant-controllers", help="JSON arm -> hash-bound feedback specification")
    ap.add_argument("--ramp-speed-adapt", choices=("on", "off"))
    args = ap.parse_args(argv)
    if args.register == (args.threshold is not None):
        ap.error("exactly one of --register / --threshold")
    if args.samples < 1:
        ap.error("--samples must be positive")
    import torch
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", "2")))
    acc = _accept()
    acc.check_roots([args.pairs])
    cfg, settings, model = acc.load_arm(args.wm_config, args.wm_ckpt)
    from dataclasses import replace
    if args.entrant_control:
        cfg = replace(cfg, data=replace(cfg.data, entrant_control=True, entries_v2=True))
        settings = replace(settings, entrant_control=True, entries_v2=True)
    if args.entrant_controllers:
        settings = replace(settings, entrant_controllers=json.loads(acc.guard(args.entrant_controllers).read_text()))
    if args.ramp_speed_adapt is not None:
        model.ramp_speed_adapt_learnable = args.ramp_speed_adapt == "on"
    horizons = sorted(args.horizons)
    steps = int(round(horizons[-1] / DT_S))
    ds = acc.build_sets(cfg, [args.pairs], steps)[0]
    panel = panel_groups(ds, args.groups, args.panel_seed)
    if not panel:
        raise SystemExit("no complete TRAIN B/SEED pair groups")
    per_h = {h: [] for h in horizons}
    rows = []
    for g, (key, group) in enumerate(panel):
        runs = {}
        for name, index in [("B", group["B"])] + [(f"SEED{j}", i) for j, i in enumerate(group["SEED"])]:
            draws = []
            for draw in range(args.samples):
                r = acc.free_rollout(model, settings, ds[index], steps,
                    SEED0 + 7919 * g + 1000003 * draw, deterministic_crn=args.deterministic_crn)
                sel = r["t0_present"] & ~r["is_cav"]
                draws.append({h: (acc.hdv_delay(r["speed"], r["mask"], sel, int(round(h / DT_S))),
                                  acc.hdv_delay(r["rec_speed"], r["rec_mask"], sel, int(round(h / DT_S))))
                              for h in horizons})
            runs[name] = {h: np.asarray([d[h] for d in draws]).mean(0).tolist() for h in horizons}
        for h in horizons:
            per_h[h].append([(runs[a][h][0] - runs["B"][h][0], runs[a][h][1] - runs["B"][h][1])
                             for a in runs if a != "B"])
        rows.append({"pair_key": key, "arms": len(runs) - 1,
                     "dJ": {str(int(h)): per_h[h][-1] for h in horizons}})
        print(json.dumps({"group": g, "pair_key": key}), flush=True)
    results = {}
    for h in horizons:
        sign, sign_ci = bootstrap(per_h[h], sign_agreement, seed=int(h))
        corr, corr_ci = bootstrap(per_h[h], pearson, seed=int(h) + 1)
        results[str(int(h))] = {"sign_agreement": sign, "sign_ci95": sign_ci, "pearson": corr,
                                "pearson_ci95": corr_ci, "n_pairs": sum(len(p) for p in per_h[h]),
                                "n_groups": len(per_h[h])}
    base = str(int(horizons[0]))
    if args.register:
        threshold = results[base]["sign_ci95"][0]
        source = {"kind": "registered_here", "base_horizon_s": horizons[0],
                  "wm_sha256": sha256(args.wm_ckpt)}
    else:
        reg = json.loads(Path(args.threshold).read_text(encoding="utf-8"))
        threshold = reg["threshold"]
        source = {"kind": "registered_file", "file": str(args.threshold), "sha256": sha256(args.threshold),
                  **reg.get("threshold_source", {})}
    for h in horizons[1:]:
        entry = results[str(int(h))]
        entry["verdict"] = ("PASS" if entry["sign_agreement"] is not None and threshold is not None
                            and entry["sign_agreement"] >= threshold else "FAIL")
    results[base]["verdict"] = "BASE"
    doc = {"schema": SCHEMA, "wm_sha256": sha256(args.wm_ckpt), "wm_config_sha256": sha256(args.wm_config),
           "samples": args.samples, "entrant_control": settings.entrant_control,
           "deterministic_crn": args.deterministic_crn or settings.entrant_control,
           "ramp_speed_adapt": model.ramp_speed_adapt_learnable,
           "pairs_root": str(args.pairs), "panel_seed": args.panel_seed, "groups": len(panel),
           "rule": ("horizon h PASS iff WM sign agreement of dJ(arm - B) t0-HDV delay at h >= "
                    "threshold; threshold = lower 95% group-bootstrap bound of the sign agreement "
                    "at the base horizon of the registering WM"),
           "threshold": threshold, "threshold_source": source, "horizons": results, "rows": rows}
    Path(args.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in doc.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
