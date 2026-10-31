"""Synthetic EXAM v1 dataset generator matching the executor export format.

Produces exactly what tools/coopbandit/export.py writes (states.jsonl, labels.csv,
group_credit.csv, worlds.json, provenance.json, README.md, MANIFEST.json), with
planted option effects so tests can verify learning:

  O1 metering helps at high mainline (lane 0) density;
  O2 wall helps when congested and members span lanes 0/1;
  O3 yield helps when a ramp vehicle is present and lane 1 is slow;
  O4 gap hold helps when group front gaps are small and congestion is high.

Availability mirrors the registered geometric rules (2-member group; O2 spans
lanes 0/1; O3 both in lane 1 with ramp observed in [-100, 75.44]); unavailable
options carry exact no-op labels (D=0), as in the real export. World random
effects make world-grouped CV the only leakage-safe evaluation. No SUMO.
"""
import csv
from datetime import datetime, timezone
import hashlib
import json

import numpy as np

from .schema import (BOOTSTRAP, BOOTSTRAP_SEED, CREDIT, HORIZONS, LABELS, MANIFEST, OPTIONS,
                     POLICY_HORIZON, PROVENANCE, README, STATES, WORLDS, canonical, checked, sha)

CELLS = tuple((p, d) for p in (0.05, 0.1, 0.2, 0.4) for d in (1900, 2400, 2700))
TIMES = (300, 360, 420, 480, 540)
GENERATOR = "tools.coopbandit_learn.fixture v2"
README_TEXT = ("# COOPBANDIT world-model EXAM v1 (synthetic fixture)\n\n"
               "states.jsonl contains numeric observable features and the observed vehicle roster.\n"
               "labels.csv is long format: state_id, option O0..O4, horizon_s 120/300/600;\n"
               "D=J(O0)-J(option), positive saves fleet plus source delay (vehicle-seconds).\n"
               "Unavailable options are explicit exact no-ops. world_id is grouping metadata,\n"
               "never a predictor; cell is descriptive only. Group every split by world_id.\n")


def _observable(rng, congestion, t, setting, lanes, ramp, gaps):
    f = {"time_s": float(t), "setting": float(setting)}
    speeds = {0: 22.0 - 9.0 * congestion, 1: 21.0 - 9.5 * congestion, 2: 24.0 - 8.0 * congestion}
    for lane in range(3):
        n = int(rng.poisson(2.0 + 10.0 * congestion * (0.6 if lane == 2 else 1.0)))
        f[f"l{lane}_count"] = float(n)
        f[f"l{lane}_cavs"] = float(min(n, int(rng.binomial(n, 0.1 + 0.5 * congestion))))
        f[f"l{lane}_speed"] = max(0.0, speeds[lane] + rng.normal(0, 1.5))
        f[f"l{lane}_queue"] = float(sum(rng.random() < congestion * 0.7 for _ in range(n)))
        f[f"l{lane}_density"] = n / 0.17544
    for i in range(2):
        lane = lanes[i]
        f[f"g{i}_s"] = float(rng.uniform(-100.0, 0.0))
        f[f"g{i}_v"] = max(0.0, speeds[lane] + rng.normal(0, 1.0))
        f[f"g{i}_lane"] = float(lane)
        f[f"g{i}_gap"] = float(gaps[i])
        f[f"g{i}_backgap"] = float(rng.uniform(8.0, 60.0))
    f["group_size"] = 2.0
    f["ramp_present"] = float(ramp)
    return f


def _d600(option, f, world_effect, rng):
    noise = rng.normal(0.0, 8.0)
    if option == "O1":
        return 90.0 * (f["l0_density"] / 68.0 - 0.55) + noise + world_effect
    if option == "O2":
        return 55.0 * (f["l0_density"] / 68.0 - 0.5) * (f["l1_density"] / 68.0 + 0.3) - 12.0 + noise + world_effect
    if option == "O3":
        return 75.0 * f["ramp_present"] * (1.0 - min(f["l1_speed"], 25.0) / 25.0) - 10.0 + noise + world_effect
    if option == "O4":
        small = 1.0 - min(f["g0_gap"], f["g1_gap"]) / 45.0
        return 50.0 * max(0.0, small) * (0.4 + 0.6 * f["l0_density"] / 68.0) - 12.0 + noise + world_effect
    return 0.0


def _available(option, lanes, ramp):
    if option in ("O1", "O4"):
        return True
    if option == "O2":
        return set(lanes) == {0, 1}
    if option == "O3":
        return lanes[0] == 1 and lanes[1] == 1 and bool(ramp)
    return False


def _roster(rng, f):
    rows, vid = [], 0
    for lane in range(3):
        for _ in range(int(f[f"l{lane}_count"])):
            rows.append(dict(vid=f"v{vid}", lane=lane, s_m=round(float(rng.uniform(-100.0, 75.44)), 2),
                             v_mps=round(float(max(0.0, f[f"l{lane}_speed"] + rng.normal(0, 2.0))), 2),
                             is_cav=bool(rng.random() < f[f"l{lane}_cavs"] / max(1, f[f"l{lane}_count"]))))
            vid += 1
    return rows


def build(out_dir, n_worlds=25, seed=7, partial=False):
    """Write a complete EXAM v1 export-format dataset; returns the MANIFEST dict."""
    out = checked(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    worlds_info, states, labels, credits = [], [], [], []
    for wi in range(n_worlds):
        seed_i = 981000 + wi
        cell = wi % len(CELLS)
        penetration, demand = CELLS[cell]
        worlds_info.append(dict(world=dict(seed=seed_i, cell=cell, repeat=wi // len(CELLS),
                                           penetration=penetration, demand=demand),
                                selected_by="ordinary" if cell not in (7, 8, 10, 11) else "rule"))
        congestion = float(np.clip(0.25 + 0.2 * penetration / 0.4 + 0.25 * (demand - 1900) / 800
                                   + rng.normal(0, 0.08), 0.0, 1.0))
        effects = {o: float(rng.normal(0, 5.0)) for o in OPTIONS}
        for t in TIMES:
            state_id = f"s{seed_i}_t{t}"
            setting = (wi + TIMES.index(t)) % 4
            lanes = (1, 1) if rng.random() < 0.35 else tuple(rng.choice([0, 1], size=2, replace=False).tolist())
            ramp = float(rng.random() < 0.3 + 0.45 * congestion)
            gaps = (float(rng.uniform(4.0, 80.0)), float(rng.uniform(4.0, 80.0)))
            f = _observable(rng, congestion, t, setting, lanes, ramp, gaps)
            avail = {o: _available(o, lanes, ramp) for o in OPTIONS}
            states.append(dict(state_id=state_id, world_id=seed_i, t0=t, features=f,
                               observed_vehicles=_roster(rng, f),
                               prefix_sha256=hashlib.sha256(state_id.encode()).hexdigest()))
            for horizon_i, horizon in enumerate(HORIZONS):
                frac = (0.25, 0.6, 1.0)[horizon_i]
                base_j = float(rng.uniform(3000.0, 8000.0))
                for option in ("O0",) + OPTIONS:
                    d600 = _d600(option, f, effects[option], rng) if option != "O0" else 0.0
                    if option == "O0":
                        d, road, source = 0.0, 0.0, 0.0
                    elif avail[option]:
                        d = d600 * frac + float(rng.normal(0, 2.0))
                        road, source = 0.7 * d, 0.3 * d
                    else:
                        d = road = source = 0.0
                    labels.append(dict(
                        state_id=state_id, world_id=seed_i, cell=cell, t0=t, setting=setting,
                        option=option, horizon_s=horizon,
                        available="True" if (option == "O0" or avail[option]) else "False",
                        D=round(d, 3), D_road=round(road, 3), D_source=round(source, 3),
                        J=round(base_j - d, 3), collisions=0,
                        teleports=int(rng.random() < 0.01), ttc_below_3_vs=int(rng.random() < 0.02),
                        hard_brake_vs=int(rng.random() < 0.05), override_share=round(rng.random() * 0.1, 4)))
                for option in OPTIONS:
                    if not avail[option]:
                        continue
                    d600 = _d600(option, f, effects[option], rng)
                    d_g = d600 * frac + float(rng.normal(0, 2.0))
                    solos = [float(rng.normal(0.4 * d600 * frac, 6.0)) for _ in lanes]
                    credits.append(dict(state_id=state_id, world_id=seed_i, option=option,
                                        horizon_s=horizon, available="True",
                                        D_G=round(d_g, 3), D_0=round(solos[0], 3), D_1=round(solos[1], 3),
                                        sum_D_i=round(sum(solos), 3), C_G=round(d_g - sum(solos), 3)))
    (out / STATES).write_text("".join(canonical(s) + "\n" for s in states), encoding="utf-8")
    _write_csv(out / LABELS, labels)
    _write_csv(out / CREDIT, credits)
    checked(out / WORLDS).write_text(canonical(dict(worlds=worlds_info, solo_worlds=[])), encoding="utf-8")
    checked(out / PROVENANCE).write_text(canonical(dict(
        design_sha256="0" * 64, sources=[], charged_sim_s=0, actual_sim_s=0.0,
        partial=partial, states=len(states))), encoding="utf-8")
    checked(out / README).write_text(README_TEXT, encoding="utf-8")
    manifest = {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != MANIFEST}
    checked(out / MANIFEST).write_text(canonical(manifest), encoding="utf-8")
    return dict(manifest=manifest, n_states=len(states), policy_horizon=POLICY_HORIZON,
                bootstrap=BOOTSTRAP, bootstrap_seed=BOOTSTRAP_SEED, partial=partial)


def _write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, keys)
        writer.writeheader()
        writer.writerows(rows)
