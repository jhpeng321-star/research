"""Pair builder for v0.4b (registration s.2/s.4): one pair per (world,
t0, arm) matched to its native sibling at the SAME world + t0.

A pair row carries:
  * identity: contrast, chunk, world, key (chunk/world), tau, arm_source,
    split, demand, penetration;
  * the shared saved_state_id (both members restored the same state; a
    mismatch REFUSES the world -- loud, never silently dropped);
  * dY per (tau, head): Y(arm window) - Y(native window) over the same
    (t, t+tau]; negative dY = the arm lowers the quantity.  Complete
    horizons only (tau <= tau_available; on the 600 s J4 branches every
    registered tau is complete at t0);
  * the PLANNED-SCHEDULE descriptors of BOTH sides (arm and native
    reference) -- the arm side feeds the model, both sides define the
    duration/onset strata of s.12.1 item 3 and the first-60-only
    representation count of item 4.

The native reference row of a world is unique; a world without exactly
one native t0 row is refused.  Pairs only form at equal (world, t0):
the merge is ON those keys and a cross-world or cross-t0 merge is
impossible by construction (unit-tested).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _paths  # noqa: F401
from . import reg, schedule_desc as SD


class PairContractError(ValueError):
    """A pair contract violation; the world is refused loudly."""


# row schema (one row per (world, source) at t0; produced by the build)
ID_COLS = ("world", "source", "chunk_id", "split", "t0_s", "demand_veh_h",
           "penetration", "state_id", "policy")
STATE_COLS = ("dem_strict", "ctx_strict", "dem_det", "ctx_det")
ACT_COLS = ("cav_n_native", "cav_n_damping", "cav_n_pi", "cav_n_wall",
            "cav_n_unknown")
PI_COLS = ("pi_native", "pi_damping", "pi_pi")
DESC_COL = ("sched_desc", "sched_wall")     # arm-side [83] + [7] block
LABEL_COLS = tuple(f"y_{h}_{int(t)}" for t in reg.TAUS
                   for h in reg.HEADS)
AVAIL_COLS = tuple(f"tau_avail_{int(t)}" for t in reg.TAUS)
ROW_COLS = (list(ID_COLS) + list(STATE_COLS) + list(ACT_COLS)
            + list(PI_COLS) + ["sched_desc", "sched_wall"]
            + list(LABEL_COLS) + list(AVAIL_COLS))


def _dy_cols(tau: float) -> list[str]:
    return [f"dy_{h}_{int(tau)}" for h in reg.HEADS]


def build_pairs(rows: pd.DataFrame,
                contrasts=tuple(reg.CONTRAST_SOURCES),
                state_assert: bool = True) -> pd.DataFrame:
    """Wall (+ aux) pairs vs the native sibling at the same world + t0.

    rows: the t0 row frame (ROW_COLS).  The native reference must be
    unique per world; arm rows merge ON (world, t0) only.  Every pair
    row records both schedule descriptors and the s.12 strata.
    """
    r = rows.copy()
    # one t0 per WORLD (the plan's branch point); the merge below is ON
    # (world, t0) so a cross-world or cross-t0 pair is impossible
    t0s = r.groupby("world")["t0_s"].nunique()
    if (t0s != 1).any():
        bad = sorted(t0s[t0s != 1].index.tolist())
        raise PairContractError(
            f"world(s) with >1 t0 among their rows: {bad[:5]}")
    t0 = r["t0_s"]
    nat = r[r["source"] == reg.NATIVE_SOURCE]
    dup = nat["world"][nat["world"].duplicated()].tolist()
    if dup:
        raise PairContractError(f"world(s) with >1 native t0 row: {dup[:5]}")
    out = []
    for c in contrasts:
        srcs, ref = reg.CONTRAST_SOURCES[c]
        arm = r[r["source"].isin(srcs)]
        m = arm.merge(nat, on=["world", "t0_s"], how="inner",
                      suffixes=("_a", "_n"), validate="many_to_one")
        if state_assert:
            bad = m["state_id_a"] != m["state_id_n"]
            if bad.any():
                ws = sorted(m.loc[bad, "world"].tolist())
                raise PairContractError(
                    f"contrast {c}: saved_state_id mismatch in world(s) "
                    f"{ws[:5]} -- the pair does not share the restored "
                    "state; refusing the world")
        rec = dict(
            contrast=c, chunk=m["chunk_id_a"], world=m["world"],
            key=m["chunk_id_a"] + "/" + m["world"], t0=m["t0_s"],
            arm_source=m["source_a"], split=m["split_a"],
            demand=m["demand_veh_h_a"].astype(int),
            penetration=m["penetration_a"].astype(float),
            state_id=m["state_id_a"],
            family=[reg.ARM_FAMILY.get(s, "native") for s in m["source_a"]],
            arm_desc=list(m["sched_desc_a"]),
            ref_desc=list(m["sched_desc_n"]),
            arm_wall=list(m["sched_wall_a"]),
        )
        for tau in reg.TAUS:
            av = f"tau_avail_{int(tau)}"
            complete = (m[av + "_a"] >= tau - 1e-9) & \
                       (m[av + "_n"] >= tau - 1e-9)
            for h in reg.HEADS:
                dy = m[f"y_{h}_{int(tau)}_a"] - m[f"y_{h}_{int(tau)}_n"]
                rec[f"dy_{h}_{int(tau)}"] = dy.where(complete)
        pf = pd.DataFrame(rec)
        # s.12 strata from the planned-schedule descriptors
        strat = [SD.duration_stratum(a, b)
                 for a, b in zip(pf["arm_desc"], pf["ref_desc"])]
        onb = [SD.onset_bin(a, b)
               for a, b in zip(pf["arm_desc"], pf["ref_desc"])]
        f60 = [SD.first60_only(a, b)
               for a, b in zip(pf["arm_desc"], pf["ref_desc"])]
        pf["dur_stratum"] = strat
        pf["onset_bin"] = onb
        pf["first60_only"] = f60
        pf["tau"] = [reg.TAUS] * len(pf)
        out.append(pf.explode("tau", ignore_index=True))
    p = pd.concat(out, ignore_index=True)
    p["tau"] = p["tau"].astype(float)
    for tau in reg.TAUS:
        for h in reg.HEADS:
            p[f"dy_{h}_{int(tau)}"] = pd.to_numeric(
                p[f"dy_{h}_{int(tau)}"], errors="coerce")
            m = np.isclose(p["tau"], tau)
            p.loc[~m, f"dy_{h}_{int(tau)}"] = np.nan
    return p


def complete_pairs(pairs: pd.DataFrame, tau: float,
                   head: str = reg.GATING_HEAD) -> pd.DataFrame:
    """Pairs of one tau with a complete horizon for that tau (truncated
    rows are dropped; tau_available was recorded at build)."""
    return pairs[np.isclose(pairs["tau"], tau)
                 & pairs[f"dy_{head}_{int(tau)}"].notna()].copy()


ARRAY_COLS = ("dem_strict", "ctx_strict", "dem_det", "ctx_det",
              "sched_desc", "sched_wall", "arm_desc", "ref_desc",
              "arm_wall")


def to_parquet_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Array-valued cells -> float lists so the frame survives a parquet
    round trip (every consumer re-materialises with np.asarray)."""
    out = df.copy()
    for c in ARRAY_COLS:
        if c in out.columns:
            out[c] = [np.asarray(x, np.float32).tolist() for x in out[c]]
    return out


def representation_counts(pairs: pd.DataFrame) -> dict:
    """The s.12.1 item 4 a-priori counts over DISTINCT worlds x arms
    (one pair per (world, arm) per tau; counted once per arm)."""
    one = pairs.drop_duplicates(subset=["contrast", "world", "arm_source"])
    pairs_desc = list(zip(one["arm_desc"], one["ref_desc"]))
    return SD.count_by_stratum(pairs_desc)
