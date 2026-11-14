"""v0.4b design matrix (registration s.3): per PAIR at t0.

Columns, in order:
  * state: the deployable block of the SHARED base trace at t0 --
    deployable_strict (HEADLINE): demand[6] designated-CAV entrance
    counts + context[8] with the entrance-queue columns zeroed;
    deployable_detector (SENSITIVITY): demand[12] + context[8] with the
    all-vehicle counts and queues (partB_v0data.deployable_features
    SourceObs semantics; the build stored both variants);
  * v0.4a per-CAV action counts at t with the unknown mask
    (cav_n_native/damping/pi/wall/unknown -- the v0.4a CAV_RAW 20 token
    counts; wall counts stay empty in v0.4a and are populated here);
  * the canonical policy one-hot (POLICIES_V04A; wall arms carry no
    v0.4a policy class and zero the block -- the arm-family one-hot
    identifies them);
  * the planner-known schedule summary pi_native/pi_damping/pi_pi;
  * the arm-family one-hot (wall / damping / pi / fixed / switching) so
    contrast families are separable (s.3);
  * the PLANNED-SCHEDULE descriptor of the ARM (83 cols: 19 x 30 s
    segment shares + the wall hold/speed/cap block) -- the s.12 action
    input, NOT the arm name.

Prohibited inputs never appear: penetration, pre_fix, speed_mode, world
strings; nothing with timestamp > t; t0 recoverability stays fixed.

One model per (tau x sensing variant) consumes rows of ONE tau; tau is
not a feature column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401
from . import reg

VARIANTS = ("deployable_strict", "deployable_detector")

POLICY_CLASSES = ("switching_mix", "damping_fleet", "native_fleet",
                  "pi_fleet", "fixed_mix")          # partB_v0.model


def _state_cols(variant: str) -> list[str]:
    if variant == "deployable_strict":
        return [f"dem_strict_{i}" for i in range(6)] + \
               [f"ctx_{i}" for i in range(8)]
    return [f"dem_det_{i}" for i in range(12)] + \
           [f"ctx_{i}" for i in range(8)]


def feature_names(variant: str) -> list[str]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown sensing variant {variant!r} "
                         f"({' | '.join(VARIANTS)})")
    state = _state_cols(variant)
    sched = [f"seg{s}_{a}" for s in range(reg.N_SEG)
             for a in reg.SCHED_ACTIONS] + \
            ["wall_hold", "wall_speed", "wall_speed_factor",
             "wall_timing_late", "wall_offset", "wall_has_cap", "wall_cap"]
    return (state + list(ACT_NAMES)
            + [f"pol_{p}" for p in POLICY_CLASSES]
            + list(PI_NAMES)
            + [f"fam_{f}" for f in reg.FAMILIES] + sched)


ACT_NAMES = ("cav_n_native", "cav_n_damping", "cav_n_pi", "cav_n_wall",
             "cav_n_unknown")
PI_NAMES = ("pi_native", "pi_damping", "pi_pi")


def state_matrix(pairs: pd.DataFrame, variant: str,
                 arm_rows: pd.DataFrame) -> np.ndarray:
    """[n, D] state block: the shared t0 state of the pair's ARM row
    (identical to the sibling's by the saved-state assertion)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown sensing variant {variant!r}")
    key = pairs.set_index(["world", "arm_source"]).index
    idx = arm_rows.set_index(["world", "source"]).index
    pos = pd.Series(np.arange(len(arm_rows)), index=idx)
    sel = pos.reindex(key).to_numpy()
    if np.isnan(sel).any():
        raise ValueError("pair without an arm row in arm_rows")
    sel = sel.astype(int)
    dem = (np.stack(arm_rows["dem_strict"].to_numpy()[sel])
           if variant == "deployable_strict"
           else np.stack(arm_rows["dem_det"].to_numpy()[sel]))
    ctx = np.stack(arm_rows["ctx_strict"].to_numpy()[sel]) \
        if variant == "deployable_strict" \
        else np.stack(arm_rows["ctx_det"].to_numpy()[sel])
    return np.concatenate([dem, ctx], axis=1).astype(np.float64)


def pair_matrix(pairs: pd.DataFrame, arm_rows: pd.DataFrame,
                variant: str) -> np.ndarray:
    """[n, F] full v0.4b design for the pairs frame (any tau mix)."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown sensing variant {variant!r}")
    S = state_matrix(pairs, variant, arm_rows)
    key = pairs.set_index(["world", "arm_source"]).index
    idx = arm_rows.set_index(["world", "source"]).index
    pos = pd.Series(np.arange(len(arm_rows)), index=idx)
    sel = pos.reindex(key).to_numpy().astype(int)

    act = np.stack([np.asarray(a, np.float64)
                    for a in arm_rows[list(ACT_NAMES)].to_numpy()[sel]])
    pol = np.zeros((len(pairs), len(POLICY_CLASSES)))
    pi = np.stack([np.asarray(a, np.float64) for a in
                   arm_rows[list(PI_NAMES)].to_numpy()[sel]])
    fam = np.zeros((len(pairs), len(reg.FAMILIES)))
    for i, f in enumerate(pairs["family"]):
        fam[i, reg.FAMILIES.index(f)] = 1.0
    sched = np.stack([np.asarray(a, np.float32) for a in pairs["arm_desc"]])
    return np.column_stack([S, act, pol, pi, fam,
                            sched.astype(np.float64)])
