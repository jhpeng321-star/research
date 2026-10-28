"""S-BACKLOG step 4b: fit + validate the long-credit CQR from WM2 rows and stage the artifacts.

Variants (the rule and tolerances are the unchanged registered ``wms.control.long_credit`` code):
  loop       : cycle 0-2 real pairs, hashed world roles (``world_role``)
  loop_e3cal : loop rows + S-E3DATA-1 'calibration' worlds as calibration and 'validation' worlds
               as validation (explicit roles; E3 'fit' worlds are never present); fit worlds are
               loop worlds only.
Tail horizons 120/180/300 s are the beyond-trusted (>90 s) windows exactly as ``fit_from_loop``
builds them.  Also writes ``probe_cqr`` (c4_probe_cqr/1) artifacts from the full-window rows at
60/120/300 s with ``traffic_shepherd.c4.probe.fit_cqr``.

Usage: python cqr_fit.py --rows DIR_OR_FILES... --out STAGE_DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

WM_SHA = "b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436"
TAIL_HORIZONS = (120, 180, 300)
PROBE_HORIZONS = (60, 120, 300)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_rows(paths):
    rows, tasks = [], set()
    for p in paths:
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            doc = json.loads(line)
            if doc["task"] in tasks:
                raise SystemExit(f"REFUSED: duplicate task {doc['task']}")
            tasks.add(doc["task"])
            rows += doc["rows"]
    return rows


def subset_validation(model, rows, role_of):
    """Coverage and within-world ranking of an already fitted model on a subset of validation rows."""
    from wms.control.long_credit import sample_location_width, validate_model
    if not rows:
        return None
    worlds = np.asarray([str(r["world"]) for r in rows])
    mean, width = sample_location_width([r["samples"] for r in rows])
    real = np.asarray([r["real_benefit"] for r in rows], float)
    out = validate_model(worlds, mean, width, real, model)
    return {k: out.get(k) for k in ("qualified", "world_coverage", "ranking_lower95", "ranking_worlds")} | {
        "n_worlds": len(set(worlds.tolist())), "n_rows": len(rows)}


def ranking_mean(model, rows):
    """Point estimate of the within-world paired-benefit ordering agreement used by validate_model
    (same centre and tie rule); None without an informative world."""
    from wms.control.long_credit import conditional_scales, sample_location_width
    if not rows or model.get("correction_scale") is None:
        return None
    worlds = np.asarray([str(r["world"]) for r in rows])
    mean, width = sample_location_width([r["samples"] for r in rows])
    real = np.asarray([r["real_benefit"] for r in rows], float)
    lower, upper = conditional_scales(mean, width, model)
    centre = ((mean + (lower - model["correction_scale"]) * width)
              + (mean + (upper + model["correction_scale"]) * width)) / 2
    out = []
    for w in sorted(set(worlds.tolist())):
        m = worlds == w
        dr = real[m][:, None] - real[m][None, :]
        dp = centre[m][:, None] - centre[m][None, :]
        info = np.triu(np.ones(dr.shape, dtype=bool), 1) & (dr != 0)
        if info.any():
            out.append(float(((np.sign(dp[info]) == np.sign(dr[info])) * 1. + (dp[info] == 0) * .5).mean()))
    return {"mean_agreement": float(np.mean(out)) if out else None, "worlds": len(out)}


def fit_variant(name, rows, role_override):
    from wms.control.long_credit import fit_cqr, world_role
    sel = []
    for r in rows:
        if r["horizon_s"] not in TAIL_HORIZONS:
            continue
        r = dict(r)
        if role_override(r) is not None:
            r["role"] = role_override(r)
        sel.append(r)
    doc = fit_cqr(sel, {"wm_sha256": WM_SHA, "variant": name, "objective": "source_integral_plus_road_delay",
                        "safe_horizon_s": 90, "tail": "beyond-trusted (>90 s) native-minus-policy cost"})
    summary = {}
    for h, model in doc["horizons"].items():
        rows_h = [r for r in sel if str(int(r["horizon_s"])) == h]
        val_rows = [r for r in rows_h if (r.get("role") or world_role(r["world"])) == "validation"]
        loop_val = [r for r in val_rows if r.get("kind") == "loop"]
        e3_val = [r for r in val_rows if r.get("kind") == "e3"]
        v = model.get("validation") or {}
        summary[h] = {"qualified": bool(model.get("qualified")), "correction_scale": model.get("correction_scale"),
                      "rank": model.get("rank"), "fit_worlds": len(model.get("fit_worlds", [])),
                      "calibration_worlds": len(model.get("calibration_worlds", [])),
                      "validation_worlds": len(v.get("worlds", [])) if v else 0,
                      "world_coverage": v.get("world_coverage"), "ranking_lower95": v.get("ranking_lower95"),
                      "ranking_worlds": v.get("ranking_worlds"),
                      "ranking_mean": ranking_mean(model, val_rows),
                      "validation_loop_subset": subset_validation(model, loop_val, None),
                      "validation_e3_subset": subset_validation(model, e3_val, None),
                      "reason": model.get("reason")}
    return doc, summary


def probe_rows(rows, include_e3):
    """c4_probe_cqr/1 rows: K=8 full-window native-minus-action samples vs real full-window benefit."""
    by_task = {}
    for r in rows:
        if r["horizon_s"] in PROBE_HORIZONS and (include_e3 or r["kind"] == "loop"):
            by_task.setdefault(r["task"], {})[r["horizon_s"]] = r
    out = []
    for task, per in by_task.items():
        if set(per) != set(PROBE_HORIZONS):
            continue
        first = per[300]
        out.append({"split": "TRAIN", "wm_sha256": WM_SHA, "seed": int(first["world"]),
                    "world": f"{first['kind']}:{first['world']}", "pair_key": first["pair_key"],
                    "kind": first["kind"], "arm": first.get("arm"), "cycle": first.get("cycle"),
                    "paired_benefit": {str(h): per[h]["samples_full"] for h in PROBE_HORIZONS},
                    "realized_benefit": {str(h): per[h]["real_full"] for h in PROBE_HORIZONS}})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    files = []
    for p in a.rows:
        p = Path(p)
        files += sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    rows = load_rows(files)
    kinds = {k: sum(1 for r in rows if r["kind"] == k and r["horizon_s"] == 300) for k in ("loop", "e3")}
    result = {"schema": "backlog_cqr_qualification/1", "wm_sha256": WM_SHA, "rows_files":
              {str(f.name): sha256(f) for f in files}, "pairs": kinds, "variants": {}}
    from wms.control.long_credit import fit_cqr
    # E3 roles come from the registered S-E3DATA-1 split carried by the rows themselves.
    variants = {
        "loop": ([r for r in rows if r["kind"] == "loop"], lambda r: None),
        "loop_e3cal": (rows, lambda r: r.get("role") if r["kind"] == "e3" else None),
    }
    for name, (subset, override) in variants.items():
        doc, summary = fit_variant(name, subset, override)
        path = out / f"CQR_{WM_SHA[:16]}_{name}.json"
        path.write_text(json.dumps(doc, indent=1, allow_nan=False), encoding="utf-8")
        result["variants"][name] = {"artifact": path.name, "artifact_sha256": sha256(path), "horizons": summary}
    # probe_cqr-ready artifacts (full-window rows, 60/120/300 s)
    from traffic_shepherd.c4.probe import fit_cqr as probe_fit
    for name, include_e3 in (("loop", False), ("loop_e3cal", True)):
        prow = probe_rows(rows, include_e3)
        jsonl = out / f"PROBE_ROWS_{name}.jsonl"
        jsonl.write_text("".join(json.dumps(r) + "\n" for r in prow), encoding="utf-8")
        doc = probe_fit(prow, WM_SHA)
        doc["sources_sha256"] = {jsonl.name: sha256(jsonl)}
        path = out / f"probe_cqr_{name}.json"
        path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        result.setdefault("probe_cqr", {})[name] = {"artifact": path.name, "artifact_sha256": sha256(path),
            "ready": doc["ready"], "fit_worlds": len(doc["fit_worlds"]),
            "calibration_worlds": len(doc["calibration_worlds"]), "rows": len(prow), "scale": doc["scale"],
            "correction": doc["correction"]}
    (out / "QUALIFICATION.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    sums = {p.name: sha256(p) for p in sorted(out.iterdir()) if p.name != "SHA256SUMS.json"}
    (out / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({v: {h: (s["qualified"], s["world_coverage"], s["ranking_lower95"]) for h, s in d["horizons"].items()}
                      for v, d in result["variants"].items()}, indent=1))
    print(json.dumps(result["probe_cqr"], indent=1))


if __name__ == "__main__":
    main()
