"""Stage 1b Part 0: within-cell, cell-demeaned and rank-correlation observability.

Reads the prep-5 artifacts only. Targets use the stored W-D Delta_delay.
0a: within-cell AUC at 2400 and 2600 (t = 250 D-prefix vector).
0b: pooled across all six cells with features cell-demeaned.
0c: pooled continuous-target Spearman rank correlations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from experiments import REPO_ROOT
from experiments.io import atomic_write_json as _atomic_json

PROBE = REPO_ROOT / "runs" / "probe"
OUTPUT_DIR = PROBE / "stage1b_generation"
CELLS = (2000, 2100, 2200, 2400, 2600, 2700)


def _load(cell: int) -> list[Mapping[str, Any]]:
    dirs = sorted(PROBE.glob(f"sustained_wall_v1_stage1b_prep5_D_m{cell}_p20_s*/seeds"))
    if not dirs:
        return []
    rows = []
    for path in sorted(dirs[0].glob("seed*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def _w_payload(cell: int) -> Dict[int, Mapping[str, Any]]:
    dirs = sorted(PROBE.glob(f"sustained_wall_v1_stage1b_prep5_W_m{cell}_p20_s*/seeds"))
    if not dirs:
        return {}
    return {
        int(path.stem[4:]): json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(dirs[0].glob("seed*.json"))
    }


def _row_features(features: Mapping[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in features.items():
        try:
            out[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def _auc(pos: Sequence[float], neg: Sequence[float]) -> float:
    combined = list(pos) + list(neg)
    ranks = rankdata(combined)
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def analyze() -> Dict[str, Any]:
    result: Dict[str, Any] = {"schema": "stage1b_part0_summary/1"}
    pooled_rows: list[Dict[str, Any]] = []
    for cell in CELLS:
        d_payloads = _load(cell)
        w_map = _w_payload(cell)
        for p in d_payloads:
            seed = int(p["seed"])
            w = w_map.get(seed)
            if w is None:
                continue
            delta = (
                float(w["w_branch"]["delay"]["mean_per_vehicle_delay_s"])
                - float(w["d_branch"]["delay"]["mean_per_vehicle_delay_s"])
            )
            features = p["step_features"].get("250")
            if features is None:
                continue
            row = _row_features(features)
            row["h"] = float(p["h_series"][250])
            pooled_rows.append({"cell": cell, "seed": seed, "delta": delta, **row})

    # 0a within-cell at 2400 and 2600
    within: Dict[str, Any] = {}
    for cell in (2400, 2600):
        rows = [r for r in pooled_rows if r["cell"] == cell]
        variables = [k for k in rows[0] if k not in ("cell", "seed", "delta")]
        aucs = {}
        for variable in variables:
            pos = [r[variable] for r in rows if r["delta"] < 0.0]
            neg = [r[variable] for r in rows if r["delta"] >= 0.0]
            aucs[variable] = _auc(pos, neg)
        top = sorted(aucs.items(), key=lambda kv: (-(kv[1] if kv[1] == kv[1] else -1), kv[0]))
        within[str(cell)] = {
            "n": len(rows),
            "n_positive": sum(1 for r in rows if r["delta"] < 0.0),
            "top_ten": [{"variable": k, "auc": v} for k, v in top[:10]],
        }

    # 0b pooled with cell-demeaned features
    variable_names = [
        k for k in pooled_rows[0] if k not in ("cell", "seed", "delta")
    ]
    cell_means: Dict[int, Dict[str, float]] = {}
    for cell in CELLS:
        rows = [r for r in pooled_rows if r["cell"] == cell]
        if not rows:
            continue
        cell_means[cell] = {
            v: float(np.mean([r[v] for r in rows])) for v in variable_names
        }
    demeaned_rows = []
    for r in pooled_rows:
        drift = {v: r[v] - cell_means[r["cell"]][v] for v in variable_names}
        demeaned_rows.append({"cell": r["cell"], "delta": r["delta"], **drift})
    aucs_demeaned = {}
    for variable in variable_names:
        pos = [r[variable] for r in demeaned_rows if r["delta"] < 0.0]
        neg = [r[variable] for r in demeaned_rows if r["delta"] >= 0.0]
        aucs_demeaned[variable] = _auc(pos, neg)
    top_d = sorted(aucs_demeaned.items(), key=lambda kv: (-(kv[1] if kv[1] == kv[1] else -1), kv[0]))
    demeaned = {
        "n": len(demeaned_rows),
        "n_positive": sum(1 for r in demeaned_rows if r["delta"] < 0.0),
        "top_ten": [{"variable": k, "auc": v} for k, v in top_d[:10]],
    }

    # 0c pooled continuous Spearman
    spearman = {}
    for variable in variable_names:
        x = [r[variable] for r in pooled_rows]
        y = [r["delta"] for r in pooled_rows]
        spearman[variable] = _spearman(x, y)
    top_s = sorted(spearman.items(), key=lambda kv: (-abs(kv[1]), kv[0]))
    spearman_block = {
        "n": len(pooled_rows),
        "top_ten": [{"variable": k, "rho": v} for k, v in top_s[:10]],
    }

    result.update({"within_cell": within, "pooled_demeaned": demeaned,
                   "pooled_spearman": spearman_block})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=str(OUTPUT_DIR))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = analyze()
    _atomic_json(output / "part0_summary.json", result)
    print(json.dumps(result, indent=1)[:2500])


if __name__ == "__main__":
    main()
