"""Collection-only S-PROBE contracts. Benefits are native J minus action J (veh-s).

No simulator imports. The fixed 50% allocation and 90% one-sided CQR coverage
are registration constants, not behavioural thresholds or outcome-tuned knobs.
"""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import numpy as np

FRACTION = 0.5
FUTURES = 8
HORIZONS = (60, 120, 300)
ALPHA = 0.1
SCHEMA = "c4_probe_cqr/1"


def split_world(world):
    return "calibration" if int(hashlib.sha256(str(world).encode()).hexdigest(), 16) % 2 else "fit"


def distribution(values):
    x = np.asarray(values, dtype=float)
    if x.shape != (FUTURES,) or not np.isfinite(x).all():
        raise ValueError("probe needs exactly eight finite CRN paired benefits")
    return float(x.mean()), float(np.quantile(x, 1 - ALPHA)), float(x.std())


def fit_cqr(rows, wm_sha256):
    """Split by entire world; fit a quantile multiplier of WM paired dispersion.

    Fit worlds learn a nonnegative uncertainty multiplier, never the conformal correction.
    World-max nonconformity prevents pseudo-replication by states/actions. Fewer
    than nine calibration worlds cannot give a finite 90% bound: remain unavailable.
    """
    fitting, calibration_rows, fit, calibration = defaultdict(list), [], set(), set()
    for row in rows:
        if row.get("split") != "TRAIN" or row["wm_sha256"] != wm_sha256:
            raise ValueError("CQR requires TRAIN real pairs rescored by the accepted WM")
        if 964000 <= int(row.get("seed", -1)) <= 964199:
            raise ValueError("evaluation seeds cannot calibrate probes")
        world = str(row["world"])
        if (row.get("complete") is False or row.get("valid") is False or row.get("collisions", 0)
                or row.get("teleports", 0) or row.get("dropped_vehicle_steps", 0)):
            continue
        if split_world(world) == "fit":
            fit.add(world)
            for h in HORIZONS:
                mean, _, std = distribution(row["paired_benefit"][str(h)])
                realized = float(row["realized_benefit"][str(h)])
                if not math.isfinite(realized):
                    raise ValueError("nonfinite real pair")
                if std > 0:
                    fitting[h, world].append((realized - mean) / std)
            continue
        calibration.add(world)
        calibration_rows.append(row)
    scale = {}
    for h in HORIZONS:
        maxima = [max(v) for (horizon, _), v in fitting.items() if horizon == h]
        if maxima:
            scale[str(h)] = max(0., float(np.quantile(maxima, 1 - ALPHA)))
    scores = defaultdict(list)
    for row in calibration_rows:
        for h in HORIZONS:
            mean, _, std = distribution(row["paired_benefit"][str(h)])
            realized = float(row["realized_benefit"][str(h)])
            if not math.isfinite(realized):
                raise ValueError("nonfinite real pair")
            scores[h, str(row["world"])].append(realized - (mean + scale.get(str(h), 0.) * std))
    correction = {}
    n = len(calibration)
    rank = math.ceil((n + 1) * (1 - ALPHA))
    if len(scale) == len(HORIZONS) and rank <= n:
        for h in HORIZONS:
            maxima = sorted(max(scores[h, w]) for w in calibration)
            correction[str(h)] = max(0., maxima[rank - 1])
    return {"schema": SCHEMA, "wm_sha256": wm_sha256, "alpha": ALPHA,
            "fit_worlds": sorted(fit), "calibration_worlds": sorted(calibration),
            "scale": scale, "correction": correction, "ready": bool(correction),
            "units": "native_minus_action_J_vehicle_seconds"}


def select(candidates, calibration, wm_sha256, world):
    """Highest calibrated optimistic bound among uncertain negative-mean actions.

    No positive-bound cutoff: the question is which pessimistic action to measure,
    not whether to deploy it. Fail closed on missing/stale/leaking calibration.
    """
    if not calibration or not calibration.get("ready"):
        return None
    if (calibration.get("schema") != SCHEMA or calibration.get("wm_sha256") != wm_sha256
            or calibration.get("alpha") != ALPHA
            or set(calibration["fit_worlds"]) & set(calibration["calibration_worlds"])
            or str(world) in calibration["fit_worlds"] + calibration["calibration_worlds"]):
        raise ValueError("stale or overlapping probe CQR worlds/WM")
    correction = calibration["correction"]
    ranked = []
    for candidate in candidates:
        stats = {}
        for h in HORIZONS:
            mean, _, std = distribution(candidate["paired_benefit"][str(h)])
            upper = mean + float(calibration["scale"][str(h)]) * std + float(correction[str(h)])
            stats[str(h)] = {"mean": mean, "upper": upper, "std": std}
        if stats["300"]["mean"] < 0 and stats["300"]["std"] > 0:
            ranked.append({**candidate, "prediction": stats})
    return min(ranked, key=lambda r: (-r["prediction"]["300"]["upper"], r["id"])) if ranked else None


def world_options(collect, *, index, cycle, guard=False):
    """One of each two existing long slots, deterministic across retries and twins."""
    cfg = collect.get("probe", {})
    if type(cfg.get("enabled", False)) is not bool:
        raise ValueError("probe.enabled must be boolean")
    if not cfg.get("enabled") or guard or cycle < int(cfg.get("from_cycle", 3)):
        return {}
    if not all(collect.get(k, {}).get("enabled") for k in ("long_group", "long_single_hold")):
        raise ValueError("S-PROBE requires both existing long slots; cannot add budget")
    if int(collect["branch_horizon"]) < 300 or not cfg.get("calibration"):
        raise ValueError("S-PROBE requires 300s native prefixes and a CQR artifact")
    return {"probe_slot": ("single", "group")[(index + cycle) % 2],
            "probe_calibration": cfg["calibration"], "probe_role": "collection"}


def correction_report(rows):
    grouped = defaultdict(list)
    for row in rows:
        for h, realized in row.get("realized_benefit", {}).items():
            grouped[row["action_type"], h].append(float(realized) - row["prediction"][h]["mean"])
    return {"schema": "c4_probe_report/1", "question": "WM pessimism corrected?",
            "units": "realized_minus_predicted_benefit_vehicle_seconds",
            "by_action_type": [{"action_type": a, "horizon_s": int(h), "n": len(v),
                                "mean_realized_minus_predicted": float(np.mean(v))}
                               for (a, h), v in sorted(grouped.items())]}
