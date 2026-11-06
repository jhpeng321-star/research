#!/usr/bin/env python3
"""LOOP14 SUMO parity probe of policy lane commands (mechanics only, no scientific use).

    python tools/loop14_checks/lane_parity_probe.py --policy <policy_latest.pt> --wm <wm_s.pt> \\
        --wm-config configs/c4_wm_v10round13e.toml --registration configs/c4_loop14_registration.json \\
        --seeds 968100,968101,968102 --out runs/loop14_lane_probe/<new dir>

Drives the REAL collector path (``traffic_shepherd.c4.policy_pairs.run_policy_pair_job`` with the
real :class:`PairPolicy` / :class:`PolicyExecutor` / :class:`WmSettings`, SUMO ``saveState`` branches,
``CavSumoEnv`` ``changeLane`` path).  The policy is sampled (C4_POLICY_STOCHASTIC=1) so it issues many
lane changes; each is mask-feasible by construction (``lane_feasibility`` + ``apply_lane_choices``
inside the executor).  Per issued command the probe records, from SUMO's own lane (``traci``):
the requested target, the tick at which the target lane was reached (1 s control ticks), whether
the vehicle left the road first, and the WM mask verdict.  ``--inject`` additionally requests, for a
few CAVs per tick, an adjacent lane the WM mask REJECTS (class ``rejected``) to see whether SUMO
would have executed it (a conservative mask is not a mismatch; an accepted-but-refused change is).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--wm-config", required=True)
    ap.add_argument("--registration", required=True)
    ap.add_argument("--seeds", default="968100,968101,968102")
    ap.add_argument("--demands", default="2400,2700,2400")
    ap.add_argument("--penetrations", default="0.4,0.4,0.2")
    ap.add_argument("--inject-seeds", default="968102",
                    help="seeds whose POLICY arm also requests mask-REJECTED adjacent lanes")
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--branch-first", type=int, default=180)
    ap.add_argument("--branch-horizon", type=int, default=240)
    ap.add_argument("--max-inject", type=int, default=2)
    ap.add_argument("--track-s", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.environ["C4_POLICY_STOCHASTIC"] = "1"
    import numpy as np
    import torch

    from traffic_shepherd.c4 import policy_pairs as pp
    from traffic_shepherd.c4.branching import SAVE_STATE_CLI
    from traffic_shepherd.c4.env import CavSumoEnv
    from traffic_shepherd.c4.geometry import LANE_TABLE, is_junction_lane, lane_and_s
    from wms.control.lanes import lane_feasibility

    seeds = [int(x) for x in a.seeds.split(",")]
    demands = [int(x) for x in a.demands.split(",")]
    pens = [float(x) for x in a.penetrations.split(",")]
    inject_seeds = {int(x) for x in a.inject_seeds.split(",") if x}
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    # The probe CONSUMES its seeds: the template lists them under "exclude" for the loop, so the
    # probe checks against a copy without that exclusion (still the reserved block, never 964xxx).
    reg = json.loads(Path(a.registration).read_text(encoding="utf-8"))
    reg["exclude"] = [e for e in reg.get("exclude", []) if not any(e[0] <= s <= e[1] for s in seeds)]
    (out / "PROBE_REGISTRATION.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    pp.check_registered_seeds(seeds, str(out / "PROBE_REGISTRATION.json"))
    factory = pp.load_pair_policy_factory(a.policy, a.wm, a.wm_config)
    times = pp.decision_times(a.warmup, a.branch_first, 300, 1)
    env = CavSumoEnv(k_slots=192, check_parity=False, ttc=True, physics_log=False,
                     extra_sumo_cli=SAVE_STATE_CLI, cav_vtype="normal_car", population="textbook")

    class Probe:
        def __init__(self):
            self.rows, self.pending = [], {}
            self.arm_counts = Counter()
            self.current = None

        def sumo_lane(self, vid):
            api = env.api
            try:
                lid = api.vehicle.getLaneID(vid)
            except Exception:
                return None                      # vehicle left the network
            if is_junction_lane(lid):
                return -1
            return lane_and_s(lid, api.vehicle.getLanePosition(vid))[0]

        def poll(self, now, final=False):
            for vid, rec in list(self.pending.items()):
                lane = self.sumo_lane(vid)
                dt = int(round(now - rec["t_s"]))
                if lane is None:
                    rec.update(outcome="left_network", dt_s=dt)
                elif lane == rec["target"]:
                    rec.update(outcome="reached", dt_s=dt)
                elif dt >= a.track_s or final:
                    rec.update(outcome="not_reached", dt_s=dt, end_lane=lane)
                else:
                    continue
                self.rows.append(self.pending.pop(vid))

    probe = Probe()
    orig_issue = env._issue_lane_commands

    def issue(targets, policy_cavs):
        issued, counts = orig_issue(targets, policy_cavs)
        probe.arm_counts.update(counts)
        t_now = env.now
        for vid, (target, index) in issued.items():
            meta = probe.current["req"].get(vid, {})
            lane_now = probe.sumo_lane(vid)
            probe.pending[vid] = {"vid": vid, "t_s": t_now, "arm": probe.current["arm"],
                                  "seed": probe.current["seed"], "target": target,
                                  "sumo_index": index, "from_lane": lane_now,
                                  "wm_class": meta.get("cls", "policy"),
                                  "wm_mask_ok": meta.get("mask_ok"), "speed": meta.get("v")}
        for vid in targets:
            if vid not in issued:
                probe.rows.append({"vid": vid, "t_s": t_now, "arm": probe.current["arm"],
                                   "seed": probe.current["seed"], "target": targets[vid],
                                   "outcome": "not_issued",
                                   "wm_class": probe.current["req"].get(vid, {}).get("cls", "policy"),
                                   "wm_mask_ok": probe.current["req"].get(vid, {}).get("mask_ok")})
        return issued, counts

    env._issue_lane_commands = issue

    class ProbePolicy(pp.PairPolicy):
        def __init__(self, base, seed, inject):
            super().__init__(base.executor, base.settings)
            self.seed, self.inject, self.rng = seed, inject, random.Random(seed)

        def __call__(self, env_, state):
            probe.poll(env_.now)
            aug = self.settings.augment(env_, state) if self.settings is not None else state
            accel, lanes = pp.PairPolicy.__call__(self, env_, state)
            lanes = np.array(lanes, copy=True)
            req = {}
            slot_ids = env_.slot_vehicle_ids()
            ages = self.executor.dwell.peek()
            with torch.no_grad():
                mask = lane_feasibility({k: (v.to(self.executor.device) if torch.is_tensor(v) else v)
                                         for k, v in aug.items()}, ages, self.executor.lane_rules)[0]
            lane_now = aug["lane"][0].long().numpy()
            speed = aug["v_mps"][0].numpy()
            for slot in np.flatnonzero(lanes[0] >= 0).tolist():
                vid = slot_ids[slot] if slot < len(slot_ids) else None
                if vid is None:
                    continue
                delta = int(lanes[0][slot]) - int(lane_now[slot])
                choice = 1 if delta < 0 else 2            # LANE_DELTA: 1 = left (-1), 2 = right (+1)
                req[vid] = {"cls": "policy", "mask_ok": bool(mask[slot, choice]), "v": float(speed[slot])}
            if self.inject:
                cav = np.flatnonzero(np.isfinite(accel[0]) & (lanes[0] < 0)).tolist()
                self.rng.shuffle(cav)
                n = 0
                for slot in cav:
                    cur = int(lane_now[slot])
                    for choice, tgt in ((1, cur - 1), (2, cur + 1)):
                        if tgt in (0, 1) and not bool(mask[slot, choice]) and slot < len(slot_ids):
                            vid = slot_ids[slot]
                            lanes[0][slot] = tgt
                            req[vid] = {"cls": "rejected", "mask_ok": False, "v": float(speed[slot])}
                            n += 1
                            break
                    if n >= a.max_inject:
                        break
            probe.current["req"] = req
            return accel, lanes

    results = []
    t_all = time.monotonic()
    for i, seed in enumerate(seeds):
        inject = seed in inject_seeds
        probe.current = {"seed": seed, "arm": "setup", "req": {}}

        def make(seed=seed, inject=inject):
            base = factory()
            probe.current = {"seed": seed, "arm": "POLICY", "req": {}}
            return ProbePolicy(base, seed, inject)

        # the collector builds a fresh PairPolicy per POLICY arm; B arms never call the policy
        res = pp.run_policy_pair_job(
            env, policy_factory=make, seed=seed, demand_vph=demands[i], penetration=pens[i],
            warmup_s=a.warmup, times=times, branch_h=a.branch_horizon,
            episode_dir=out / f"base_{seed}", branch_root=out / f"branches_{seed}",
            meta={"policy": "native", "probe": "lane_parity"}, policy_ckpt=a.policy)
        probe.poll(env.now if env.api is not None else 0, final=True)
        b = res["branches"]
        results.append({"seed": seed, "demand": demands[i], "penetration": pens[i], "inject": inject,
                        "sim_s": b["sim_s"], "collisions": b["collisions"], "teleports": b["teleports"],
                        "branches": [{k: x[k] for k in ("arm", "t_s", "collisions", "teleports",
                                                        "lane_issued", "lane_completed", "J")}
                                     for x in b["branches"]]})
        print(json.dumps(results[-1]), flush=True)
    env_counts = dict(env.lane_command_counts)
    rows = probe.rows
    summary = {"schema": "loop14_lane_parity_probe/1", "scientific_use": False,
               "policy": a.policy, "wm": a.wm, "seeds": seeds, "worlds": results,
               "sim_s_total": sum(r["sim_s"] for r in results),
               "wall_s": time.monotonic() - t_all,
               "env_lane_command_counts_all_arms": env_counts,
               "policy_arm_issue_counts": dict(probe.arm_counts)}
    by = defaultdict(Counter)
    lags = defaultdict(Counter)
    for r in rows:
        key = f"{r['wm_class']}|mask_ok={r['wm_mask_ok']}"
        by[key][r["outcome"]] += 1
        if r["outcome"] == "reached":
            lags[key][r["dt_s"]] += 1
    summary["outcome_by_class"] = {k: dict(v) for k, v in sorted(by.items())}
    summary["reach_lag_s_by_class"] = {k: dict(sorted(v.items())) for k, v in sorted(lags.items())}
    summary["n_rows"] = len(rows)
    (out / "PROBE_ROWS.json").write_text(json.dumps(rows, indent=0), encoding="utf-8")
    (out / "PROBE_SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("outcome_by_class", "reach_lag_s_by_class",
                                               "policy_arm_issue_counts", "sim_s_total")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
