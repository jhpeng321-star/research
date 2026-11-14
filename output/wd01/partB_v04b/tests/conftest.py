"""Shared fixtures for the partB_v04b tests.

Everything here is SYNTHETIC: no J4 run, no v0.4a file and no CAL
world is opened by the unit tests (the real-data dry-run is a separate
command, see README.md).  WD_FORBID_D=1 (installed by output/wd01/
conftest.py) stays active for the whole suite.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
PART = HERE.parent                       # output/wd01/partB_v04b
WD01 = PART.parent                       # output/wd01
for p in (str(WD01), str(PART)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("WD_DEVICE", "cpu")

from partB_v04b import reg               # noqa: E402
from partB_v04b import schedule_desc as SD   # noqa: E402

WORLDS = ("wTEST1m2700p10t300", "wTEST2m2400p40t300",
          "wTEST3m2700p20t300")
SOURCES = ("native", "base", "pi", "fixed1", "fixed2",
           "switching1", "switching2", "switching3",
           "W-lo-early", "W-lo-late", "W-mid-early", "W-mid-late")
WALL_HOLD = {"W-lo-early": 60.0, "W-lo-late": 120.0,
             "W-mid-early": 240.0, "W-mid-late": 60.0}
WALL_REQ_REL = {"W-lo-early": 0.0, "W-lo-late": 144.0,
                "W-mid-early": 60.0, "W-mid-late": 300.0}
POLICY = {"native": "native_fleet", "base": "damping_fleet",
          "pi": "pi_fleet", "fixed1": "fixed_mix",
          "fixed2": "fixed_mix", "switching1": "switching_mix",
          "switching2": "switching_mix", "switching3": "switching_mix"}


def _fleet_desc(action: str) -> np.ndarray:
    return SD.describe(SD.fleet_schedule(action, n_roster=1))


def _switch_sched(k: int) -> dict:
    # k of the roster switches damping->native after 0/k*30 s: start/stop
    # variation inside the aux families
    events = [(f"cav{i}", 0.0, "damping") for i in range(4)] + \
             [(f"cav{i}", k * 30.0, "native") for i in range(k % 4)]
    return SD.schedule_from_events(events)


def _wall_desc(arm: str) -> tuple[dict, np.ndarray]:
    members = ("mainline.1", "mainline.2")
    speed = 30.0 / 3.6 * reg.ARM_SPEED_FACTORS[
        "W-lo" if arm.startswith("W-lo") else "W-mid"]
    wall7 = SD.wall_block(WALL_HOLD[arm], speed,
                          0.6 if arm.startswith("W-lo") else 0.85,
                          "early" if arm.endswith("early") else "late",
                          reg.TIMING_OFFSET_S[
                              "early" if arm.endswith("early")
                              else "late"], None)
    sched = SD.wall_member_schedule(members, WALL_HOLD[arm], t0=0.0,
                                    request_s=WALL_REQ_REL[arm])
    return sched, wall7


def make_rows(n_worlds: int = 3, seed: int = 0) -> pd.DataFrame:
    """Synthetic t0 rows frame (ROW_COLS), 12 sources per world, with a
    DETERMINISTIC label structure: dY_delay(wall arm) = -hold_s * 2.0
    (negative: the wall lowers delay), dY_delay(damping arm) =
    -30, dY_delay(pi/fixed/switching) = +10 * k, plus per-tau scaling."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in WORLDS[:n_worlds]:
        loading = 2700 if "m2700" in w else 2400
        pen = 0.1 if "p10" in w else (0.4 if "p40" in w else 0.2)
        dem_s = rng.random(6).astype(np.float32)
        ctx_s = np.zeros(8, np.float32)
        ctx_s[0], ctx_s[3] = 0.3, loading / 2700.0
        dem_d = np.concatenate([rng.random(6), dem_s]).astype(np.float32)
        ctx_d = ctx_s.copy()
        ctx_d[6] = ctx_d[7] = 0.5
        state_id = "state" + w[1:6]
        y_native = {t: 100.0 + t for t in reg.TAUS}
        for src in SOURCES:
            wall7 = np.zeros(7, np.float32)
            if src == "native":
                sched = SD.fleet_schedule("native", n_roster=1)
            elif src in POLICY and POLICY[src] == "damping_fleet":
                sched = SD.fleet_schedule("damping", n_roster=1)
            elif src in POLICY and POLICY[src] == "pi_fleet":
                sched = SD.fleet_schedule("pi", n_roster=1)
            elif src.startswith("fixed"):
                sched = SD.fleet_schedule("damping", n_roster=1)
            elif src.startswith("switching"):
                sched = _switch_sched(int(src[-1]))
            else:
                sched, wall7 = _wall_desc(src)
            desc = SD.describe(sched, wall=wall7)
            dy_base = 0.0
            if src in reg.WALL_ARMS:
                dy_base = -2.0 * WALL_HOLD[src]
            elif src == "base":
                dy_base = -30.0
            elif src.startswith("switching"):
                dy_base = 10.0 * int(src[-1])
            elif src.startswith("fixed"):
                dy_base = 10.0
            elif src == "pi":
                dy_base = 15.0
            acts = {"native": 8.0, "damping": 1.0, "pi": 0.0,
                    "wall": 0.0, "unknown": 0.0}
            if src in reg.WALL_ARMS:
                acts = {"native": 6.0, "damping": 0.0, "pi": 0.0,
                        "wall": 2.0, "unknown": 0.0}
            row = dict(
                world=w, source=src, chunk_id="chunk4", split="TRAIN",
                t0_s=300.0, demand_veh_h=loading, penetration=pen,
                state_id=state_id,
                policy=POLICY.get(src, "wall_v04b"),
                dem_strict=dem_s, ctx_strict=ctx_s,
                dem_det=dem_d, ctx_det=ctx_d,
                cav_n_native=acts["native"], cav_n_damping=acts["damping"],
                cav_n_pi=acts["pi"], cav_n_wall=acts["wall"],
                cav_n_unknown=acts["unknown"],
                pi_native=SD.pi_shares(sched)["native"],
                pi_damping=SD.pi_shares(sched)["damping"],
                pi_pi=SD.pi_shares(sched)["pi"],
                sched_desc=desc,
                sched_wall=np.asarray(wall7, np.float32),
            )
            for tau in reg.TAUS:
                for h, scale in (("delay", 1.0), ("ttc", 0.05),
                                 ("brake", 0.2)):
                    row[f"y_{h}_{int(tau)}"] = \
                        float(y_native[tau] + dy_base * scale)
                row[f"tau_avail_{int(tau)}"] = 600.0
            rows.append(row)
    from partB_v04b.pairs_v04b import ROW_COLS
    return pd.DataFrame(rows, columns=list(ROW_COLS))


@pytest.fixture(scope="session")
def rows3() -> pd.DataFrame:
    return make_rows(3)


@pytest.fixture(scope="session")
def pairs3(rows3):
    from partB_v04b import pairs_v04b as PB
    return PB.build_pairs(rows3)

