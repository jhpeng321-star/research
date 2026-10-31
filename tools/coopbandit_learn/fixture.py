"""Synthetic EXAM v1 dataset generator: planted option effects, zero SUMO.

Planted D600 structure (positive = saves delay), so tests can verify learning:
  O1 metering helps at high mainline density;
  O2 wall helps when congested and members span lanes 0/1;
  O3 yield helps when a ramp vehicle is present and lane 1 is slow;
  O4 gap hold helps when the group front gaps are small and congestion is high.
Availability mirrors the registered geometric rules (2-member group; O2 spans
lanes; O3 both in lane 1 plus ramp observed). World random effects make
world-grouped CV the only leakage-safe evaluation.
"""
import csv
from datetime import datetime, timezone
import json

import numpy as np

from .schema import (BOOTSTRAP, CREDIT, CREDIT_REQUIRED, FOLDS, HORIZONS, LABEL_REQUIRED, LABELS,
                     OPTIONS, GROUP_OPTIONS, POLICY_HORIZON, PROVENANCE, SCHEMA_DOC, SPLITS, STATES,
                     canonical, checked, sha)

CELLS = tuple((p, d) for p in (0.05, 0.1, 0.2, 0.4) for d in (1900, 2400, 2700))
TIMES = (300, 360, 420, 480, 540)
GENERATOR = "tools.coopbandit_learn.fixture v1"


def _features(rng, congestion, t, setting, lanes, ramp, gaps):
    out = {"time_s": float(t), "setting": float(setting)}
    speeds = {0: 22.0 - 9.0 * congestion, 1: 21.0 - 9.5 * congestion, 2: 24.0 - 8.0 * congestion}
    for lane in range(3):
        n = int(rng.poisson(2.0 + 10.0 * congestion * (0.6 if lane == 2 else 1.0)))
        out[f"l{lane}_count"] = float(n)
        out[f"l{lane}_cavs"] = float(min(n, int(rng.binomial(n, 0.1 + 0.5 * congestion))))
        out[f"l{lane}_speed"] = max(0.0, speeds[lane] + rng.normal(0, 1.5))
        out[f"l{lane}_queue"] = float(sum(rng.random() < congestion * 0.7 for _ in range(n)))
        out[f"l{lane}_density"] = n / 0.17544
    for i in range(2):
        lane, s, v = lanes[i], float(rng.uniform(-100.0, 0.0)), max(0.0, speeds[lanes[i]] + rng.normal(0, 1.0))
        front = float(gaps[i]) if lane != lanes[1 - i] or s < float(gaps[1 - i]) else 6.0
        out[f"g{i}_s"] = s
        out[f"g{i}_v"] = v
        out[f"g{i}_lane"] = float(lane)
        out[f"g{i}_gap"] = front
        out[f"g{i}_backgap"] = float(rng.uniform(8.0, 60.0))
    out["group_size"] = 2.0
    out["ramp_present"] = float(ramp)
    return out


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
        return {lanes[0], lanes[1]} == {0, 1}
    if option == "O3":
        return lanes[0] == 1 and lanes[1] == 1 and bool(ramp)
    return False


def _roster(rng, f):
    rows, vid = [], 0
    for lane in range(3):
        for _ in range(int(f[f"l{lane}_count"])):
            rows.append(dict(vid=f"v{vid}", lane=lane, s_m=float(rng.uniform(-100.0, 75.44)),
                             v_mps=float(max(0.0, f[f"l{lane}_speed"] + rng.normal(0, 2.0))),
                             is_cav=bool(rng.random() < f[f"l{lane}_cavs"] / max(1, f[f"l{lane}_count"]))))
            vid += 1
    return rows


def build(out_dir, n_worlds=24, seed=7, folds=FOLDS):
    """Write a complete EXAM v1 dataset (states, labels, credit, splits, provenance)."""
    out = checked(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    states, label_rows, credit_rows = [], [], []
    world_names = [f"w{i:03d}" for i in range(n_worlds)]
    for wi, world in enumerate(world_names):
        cell = wi % len(CELLS)
        penetration, demand = CELLS[cell]
        congestion = float(np.clip(0.25 + 0.2 * penetration / 0.4 + 0.25 * (demand - 1900) / 800
                                   + rng.normal(0, 0.08), 0.0, 1.0))
        effects = {o: float(rng.normal(0, 5.0)) for o in GROUP_OPTIONS}
        for ti, t in enumerate(TIMES):
            setting = (wi + ti) % 4
            lanes = (1, 1) if rng.random() < 0.35 else tuple(rng.choice([0, 1], size=2, replace=False).tolist())
            ramp = float(rng.random() < 0.3 + 0.45 * congestion)
            gaps = (float(rng.uniform(4.0, 80.0)), float(rng.uniform(4.0, 80.0)))
            f = _features(rng, congestion, t, setting, lanes, ramp, gaps)
            state_id = f"{world}_t{t}"
            avail = {o: _available(o, lanes, ramp) for o in GROUP_OPTIONS}
            states.append(dict(state_id=state_id, world=world, cell=cell, penetration=penetration,
                               demand=demand, t_decision=t, setting=setting, available=avail,
                               group=["g0", "g1"], features=f, roster=_roster(rng, f)))
            for option in GROUP_OPTIONS:
                d600 = _d600(option, f, effects[option], rng)
                if not avail[option]:
                    continue
                for horizon, frac in zip(HORIZONS, (0.25, 0.6, 1.0)):
                    label_rows.append(dict(
                        state_id=state_id, option=option, horizon=horizon,
                        D=round(d600 * frac + rng.normal(0, 2.0), 3),
                        D_fleet=round(0.7 * d600 * frac, 3), D_source=round(0.3 * d600 * frac, 3),
                        collisions=0, ttc_violations=int(rng.random() < 0.02),
                        hard_brakes=int(rng.random() < 0.05), override_share=round(rng.random() * 0.1, 4)))
                solos = [round(float(rng.normal(0.4 * d600, 6.0)), 3) for _ in lanes]
                credit_rows.append(dict(state_id=state_id, option=option, C_G=round(d600 - sum(solos), 3),
                                        D_solo_member0=solos[0], D_solo_member1=solos[1],
                                        members="g0|g1"))
    _write_states(out / STATES, states)
    _write_labels(out / LABELS, label_rows)
    _write_credit(out / CREDIT, credit_rows)
    fold_of = {w: i % folds for i, w in enumerate(sorted(world_names))}
    checked(out / SPLITS).write_text(canonical({"fold_of": fold_of, "folds": folds}), encoding="utf-8")
    _write_schema_doc(out / SCHEMA_DOC)
    files = {p.name: sha(p) for p in sorted(out.iterdir()) if p.name != PROVENANCE}
    manifest = dict(files=files, generator=GENERATOR, created_utc=datetime.now(timezone.utc).isoformat(),
                    n_states=len(states), policy_horizon=POLICY_HORIZON, options=list(OPTIONS))
    checked(out / PROVENANCE).write_text(canonical(manifest), encoding="utf-8")
    return manifest


def _write_states(path, states):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for s in states:
            fh.write(json.dumps(s, sort_keys=True, allow_nan=False) + "\n")


def _write_labels(path, rows):
    fields = ["state_id", "option", "horizon", "D", "D_fleet", "D_source", "collisions",
              "ttc_violations", "hard_brakes", "override_share"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _write_credit(path, rows):
    fields = list(CREDIT_REQUIRED) + ["D_solo_member0", "D_solo_member1", "members"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


SCHEMA_TEXT = """# EXAM v1 dataset schema (synthetic fixture identical to the planned export)

- states.jsonl: one JSON object per decision state.
  Required: state_id (unique str), world (str, grouping key), cell (int 0..11),
  t_decision (number), setting (int 0..3), available ({O1..O4: bool}),
  features (dict of finite floats; MUST NOT contain world/seed/cell/penetration/
  demand/identifiers; follows tools/coopbandit protocol.observable() keys),
  roster (sensed rows; optional for modelling, kept for audits).
- labels.csv: long format with required columns state_id, option, horizon, D
  (units veh-s of delay saved, positive = option better than O0). Optional
  components/safety columns are passed through: D_fleet, D_source, collisions,
  ttc_violations, hard_brakes, override_share. Every AVAILABLE option must have
  rows at horizons 120/300/600; unavailable options must have none.
- group_credit.csv: state_id, option, C_G (+ optional solo components).
- splits.json: {"fold_of": {world: 0..4}, "folds": 5} world-grouped CV folds.
- provenance.json: {"files": {name: sha256}, ...} verified on load.
Metadata (world/cell/penetration/demand) never enter the feature matrix.
"""


def _write_schema_doc(path):
    checked(path).write_text(SCHEMA_TEXT, encoding="utf-8")
