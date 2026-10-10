"""Stage 1b-prep Part B analysis: per-cell incidence and onset structure.

Reads the three band-locator cell directories plus the 2200/p20 Stage 1a cell
(no rerun) and reports per cell: health and physical label incidence with
Clopper-Pearson 95% CI, onset-time median/IQR/min/max among collapsing seeds,
the fraction of seeds with h(300) > 0.45, and the median delay s/veh.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import beta

from experiments import REPO_ROOT
from experiments.idm import sustained_wall_v1_contract as C
from experiments.io import atomic_write_json as _atomic_json


def _clopper_pearson(k: int, n: int) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    if k <= 0:
        return (0.0, float(beta.ppf(0.975, 1, n)))
    if k >= n:
        return (float(beta.ppf(0.025, n, 1)), 1.0)
    return (
        float(beta.ppf(0.025, k, n - k + 1)),
        float(beta.ppf(0.975, k + 1, n - k)),
    )


def _load_seeds(seeds_dir: Path) -> List[Mapping[str, Any]]:
    payloads = []
    for path in sorted(seeds_dir.glob("seed*.json")):
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
    return payloads


def _cell_summary(payloads: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    rows = []
    for payload in payloads:
        branch = payload["branch"]
        rows.append(_row(int(payload["seed"]), branch))
    return _rows_summary(rows)


def _row(seed: int, branch: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "seed": seed,
        "health_incidence": bool(branch["health_collapse"]["incidence"]),
        "health_onset": branch["health_collapse"]["first_onset_step"],
        "health_total_duration_s": float(branch["health_collapse"]["total_duration_s"]),
        "health_longest_run_s": float(branch["health_collapse"]["longest_run_s"]),
        "physical_incidence": bool(branch["physical_collapse"]["incidence"]),
        "physical_onset": branch["physical_collapse"]["first_onset_step"],
        "physical_total_duration_s": float(branch["physical_collapse"]["total_duration_s"]),
        "h300": float(branch["series"]["h"][300]),
        "delay_s_per_veh": float(branch["delay"]["mean_per_vehicle_delay_s"]),
        "hard_brake_steps": int(branch["outcomes"]["hard_brake_vehicle_steps"]),
        "low_ttc_steps": int(branch["outcomes"]["low_ttc_exposure_steps"]),
        "terminal": branch["termination"]["classification"],
    }


def _rows_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    k_health = sum(1 for row in rows if row["health_incidence"])
    k_physical = sum(1 for row in rows if row["physical_incidence"])
    onsets = sorted(rows, key=lambda r: (r["health_onset"] is None, r["health_onset"] or 0))
    collapsing = [row["health_onset"] for row in rows if row["health_onset"] is not None]
    onset_array = np.asarray([float(v) for v in collapsing], dtype=float)
    h300 = np.asarray([row["h300"] for row in rows], dtype=float)
    delays = np.asarray([row["delay_s_per_veh"] for row in rows], dtype=float)
    ph_onsets = [
        row["physical_onset"] for row in rows if row["physical_onset"] is not None
    ]
    return {
        "n": n,
        "health_incidence": k_health,
        "health_incidence_ci95_clopper_pearson": _clopper_pearson(k_health, n),
        "physical_incidence": k_physical,
        "physical_incidence_ci95_clopper_pearson": _clopper_pearson(k_physical, n),
        "health_onset_among_collapsing": {
            "n_collapsing": len(collapsing),
            "min": float(np.min(onset_array)) if onset_array.size else None,
            "median": float(np.median(onset_array)) if onset_array.size else None,
            "q25": float(np.percentile(onset_array, 25)) if onset_array.size else None,
            "q75": float(np.percentile(onset_array, 75)) if onset_array.size else None,
            "max": float(np.max(onset_array)) if onset_array.size else None,
            "values": sorted(int(v) for v in collapsing),
        },
        "physical_onset_among_collapsing": {
            "n_collapsing": len(ph_onsets),
            "values": sorted(int(v) for v in ph_onsets),
        },
        "fraction_h300_above_0_45": float(np.mean(h300 > 0.45)),
        "h300_values": [float(v) for v in h300],
        "median_delay_s_per_veh": float(np.median(delays)),
        "p10_delay_s_per_veh": float(np.percentile(delays, 10)),
        "p90_delay_s_per_veh": float(np.percentile(delays, 90)),
        "median_health_total_duration_s": float(np.median([r["health_total_duration_s"] for r in rows])),
        "median_health_longest_run_s": float(np.median([r["health_longest_run_s"] for r in rows])),
        "median_hard_brake_steps": float(np.median([r["hard_brake_steps"] for r in rows])),
        "median_low_ttc_steps": float(np.median([r["low_ttc_steps"] for r in rows])),
        "terminal_counts": {t: sum(1 for r in rows if r["terminal"] == t) for t in ("horizon", "natural_clearance", "early_incomplete")},
        "rows": rows,
    }


def analyze(cell_dirs: Mapping[str, Path], output: Path) -> Dict[str, Any]:
    result: Dict[str, Any] = {"schema": "sustained_wall_v1_stage1b_band_locator_summary/1"}
    for label, seeds_dir in cell_dirs.items():
        result[label] = _cell_summary(_load_seeds(seeds_dir))
    return result


def _stage1a_2200_rows() -> List[Mapping[str, Any]]:
    seeds_dir = (
        REPO_ROOT
        / "runs"
        / "probe"
        / "sustained_wall_v1_stage1a_m2200_p20_s6001_6030"
        / "seeds"
    )
    rows = []
    for seed in range(6001, 6031):
        payload = json.loads((seeds_dir / f"seed{seed}.json").read_text(encoding="utf-8"))
        rows.append(_row(seed, payload["arms"]["C"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=str(REPO_ROOT / "runs" / "probe" / "sustained_wall_v1_stage1b_prep"),
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    cell_dirs: Dict[str, Path] = {}
    for demand in (1900, 2000, 2100):
        matches = list(
            (REPO_ROOT / "runs" / "probe").glob(
                f"sustained_wall_v1_stage1b_band_locator_m{demand}_s*/seeds"
            )
        )
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one m{demand} band-locator dir, got {matches}")
        cell_dirs[f"m{demand}"] = matches[0]
    result = analyze(cell_dirs, output)
    result["m2200_stage1a_reused"] = _rows_summary(_stage1a_2200_rows())
    output.mkdir(parents=True, exist_ok=True)
    _atomic_json(output / "band_locator_summary.json", result)
    for label in ("m1900", "m2000", "m2100", "m2200_stage1a_reused"):
        cell = result[label]
        ci = cell["health_incidence_ci95_clopper_pearson"]
        print(
            f"{label}: n={cell['n']} health={cell['health_incidence']} (CP95 [{ci[0]:.3f}, {ci[1]:.3f}]) "
            f"physical={cell['physical_incidence']} onset_med={cell['health_onset_among_collapsing']['median']} "
            f"h300>0.45={cell['fraction_h300_above_0_45']:.3f} delay_med={cell['median_delay_s_per_veh']:.2f}"
        )


if __name__ == "__main__":
    main()
