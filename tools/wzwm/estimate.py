"""Time and GPU-memory estimate of the wz3to2 fine-tune (a priori, then re-based on the smoke).

Reference (measured, gpuA RTX 4090 D): WM2's loop fine-tune c02 (same code, same model, merge map
K = 192 slots, 120 s unroll, truncation 120, batch 4): 40 steps, mean 198 s/step (median 216),
peak CUDA 1,993 MiB, stage-end multi-horizon validation (16 windows) ~60 min
(train_log.jsonl / RUN_SUMMARY.json timestamps of restart20261026/arm/loop/wm/c02).

Scaling to wz3to2 (K = 512 slots, WM_PORT_API):
* time per step: the rollout is launch-latency bound (2 GB peak on a 48 GB card), so slot count
  enters between not at all and linearly: factor in [1, 512/192]; the central value is the
  geometric mean.  A 300 s step (600 physics steps, truncation 60, batch 2) costs 2.5x a 120 s step.
* memory: linear in slots x batch x truncation chunk: 120 s stage 1,993 MiB x 2.67 = ~5.3 GiB;
  the 300 s stage (batch 2, truncation 60) is below that.
After the 2-step smoke the measured 120 s step time and peak memory replace the a priori ones.
"""
from __future__ import annotations

import math
from typing import Mapping, Optional

REF_S_PER_STEP_120 = 198.25
REF_PEAK_MB = 1993.2
REF_VAL_S_16 = 3600.0
REF_K, WZ_K = 192, 512
STEP_300_FACTOR = 2.5


def phase_hours(steps_by_stage: Mapping[float, int], s120: float, val_windows: int, k_factor: float,
                n_stage_val: int) -> float:
    train_s = sum(n * s120 * (STEP_300_FACTOR if float(u) >= 300.0 else 1.0) * (float(u) / 120.0 if float(u) < 120 else 1.0)
                  for u, n in steps_by_stage.items())
    val_s = REF_VAL_S_16 * (val_windows / 16.0) * k_factor * n_stage_val
    return (train_s + val_s) / 3600.0


def estimate(plan: Mapping, *, measured_s_per_step_120: Optional[float] = None,
             measured_peak_mb: Optional[float] = None) -> dict:
    k_hi = WZ_K / REF_K
    out = {"reference": {"s_per_step_120": REF_S_PER_STEP_120, "peak_mb": REF_PEAK_MB,
                         "val_s_16_windows": REF_VAL_S_16, "k_ref": REF_K, "k_wz": WZ_K},
           "scenarios": {}}
    scen = {"low": 1.0, "central": math.sqrt(k_hi), "high": k_hi}
    if measured_s_per_step_120:
        scen = {"measured": float(measured_s_per_step_120) / REF_S_PER_STEP_120}
    for name, f in scen.items():
        s120 = REF_S_PER_STEP_120 * f
        total, per = 0.0, {}
        for ph, spec in plan["phases"].items():
            cur = spec["curriculum"]
            steps = {}
            for u, n in zip(cur["unroll_s"], cur["stage_steps"]):
                steps[float(u)] = steps.get(float(u), 0) + int(n)
            vw = 16 if ph == "B" else 4
            for o in spec.get("overrides", []):
                if o.startswith("selection.val_max_windows="):
                    vw = int(o.split("=", 1)[1])
            h = phase_hours(steps, s120, vw, f, len(cur["stage_steps"]))
            per[ph] = round(h, 2)
            total += h
        smoke_h = (2 * s120 + REF_VAL_S_16 * (2 / 16) * f) / 3600.0
        out["scenarios"][name] = {"k_time_factor": round(f, 3), "s_per_step_120": round(s120, 1),
                                  "s_per_step_300": round(s120 * STEP_300_FACTOR, 1),
                                  "phase_h": per, "smoke_h": round(smoke_h, 2),
                                  "total_h": round(total + smoke_h, 2)}
    peak = measured_peak_mb or REF_PEAK_MB * k_hi
    out["gpu_peak_mb"] = round(float(peak), 1)
    out["gpu_peak_basis"] = "measured smoke (120 s, batch 4)" if measured_peak_mb else "a priori linear in K"
    out["disk"] = {"checkpoint_mb": 3.5, "per_phase_mb": "<= 40 (stage/latest/best/final checkpoints + logs)",
                   "total_mb": "< 200"}
    key = "measured" if measured_s_per_step_120 else "central"
    out["total_h"] = out["scenarios"][key]["total_h"]
    out["basis"] = key
    return out
