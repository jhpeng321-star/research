"""Cost quote for the wz3to2 speed-only MAPPO comparator (N7).  Zero SUMO; pure arithmetic.

    python tools/nm_mappo/quote.py [--json] [--wz-factor-low 1.0 --wz-factor-high 1.3]

Units: ``sim_s`` = simulated SUMO seconds incl. the 300 s native warmup (the project's budget
unit, as charged for S-MAPPOSPEED); ``controlled_s`` = the trainer's charged seconds
(``max_sumo_seconds``, warmup uncharged).  Wall-time model, calibrated on the ORIGINAL
speed-only run (S-MAPPOSPEED production2, cpuQ 32 cores, 30 workers, 46 episodes/iteration,
16 update threads; merge map):

    iteration wall = waves * WAVE_S * contention + UPDATE_S          (merge map)
    waves          = ceil(46 / workers)
    wz3to2 wall    = merge-map wall * [factor_low, factor_high]      (default 1.0 .. 1.3)

* ``WAVE_S`` = 31.86 s: mean collect wall of iterations 1-2 (64.09 / 63.34 s) over 2 waves.
* ``UPDATE_S`` = 103.1 s: whole-run mean iteration (169 it <= 7.83 h, 08:19:05 UTC start ->
  completion logged 00:09 UTC+8) minus 2 waves; iterations 1-2 measured 103.9 / 117.0 s.
  Update time does not shrink with more cores (no gain above 16 threads, MAPPO_NOTES.md).
* contention: 1.0 at 30 workers on 32 cores; 1.1 assumed at 46 workers on 48 cores.
* wz3to2 factor: more lanes + the forced lane-drop lane changes slow SUMO stepping (probe native
  rate ~34 sim-s per wall-s per worker on cpuV vs 45-60 on the merge map) while the policy
  forward / PPO update scale with present CAVs (similar vehicle counts at 2400-2800 veh/h);
  the first two production iterations (82,800 sim-s) re-quote the ETA, as for S-MAPPOSPEED.
* Host speed: cpuV per-core speed assumed equal to cpuQ (+-20% not modelled).
"""

from __future__ import annotations

import argparse
import json
import math

EPISODES = 46
WARMUP_S = 300.0
HORIZON_S = 600.0
EPISODE_SIM_S = WARMUP_S + HORIZON_S
SMOKE_SIM_S = 2 * (300.0 + 30.0)          # 1 iteration x 2 episodes x (300 + 30) s, as S-MAPPOSPEED
WAVE_S = (64.08635367301758 + 63.34149609802989) / 2.0 / 2.0
FULL_RUN_WALL_H = 7.83                    # upper bound: 169 iterations, cpuQ, 30 workers
UPDATE_S = FULL_RUN_WALL_H * 3600.0 / 169.0 - 2.0 * WAVE_S

#: (label, cores, workers, update threads, contention, note)
HOSTS = {
    "cpuV": dict(cores=48, workers=46, contention=1.1,
                 note="dedicated: 46 workers = one wave of 46 episodes"),
    "cpuQ": dict(cores=32, workers=30, contention=1.0,
                 note="second machine (cpuQ-class, 32 cores): the original speed-only host/settings"),
    "cpuV_shared12": dict(cores=12, workers=12, contention=1.0, update_penalty=1.1,
                          note="cpuV while N3/N4 hold the rest (12-core share as for newmap_probe)"),
}

#: option -> (iterations, init, purpose)
OPTIONS = {
    "full169_u0": dict(iterations=169, init="u0",
                       purpose="RECOMMENDED full fair run: original speed-only recipe (warm start from the "
                               "previous MAPPO actor + 169 x 46 episodes); snapshots 24/42/84 nested"),
    "full169_scratch": dict(iterations=169, init="scratch",
                            purpose="same budget from a fresh actor (weaker: the original actor had ~14M sim-s "
                                    "of merge-map MAPPO history before its 169 speed-only iterations)"),
    "ft84_u0": dict(iterations=84, init="u0", purpose="reduced: 1/2 of the original updates"),
    "ft42_u0": dict(iterations=42, init="u0",
                    purpose="REDUCED-COST: 1/4 of the original updates (>= 3.5x the per-arm on-map "
                            "training experience of the clean3-style method by its 57,600 sim-s/update A+B+R basis)"),
    "ft24_u0": dict(iterations=24, init="u0", purpose="minimum: 24 updates, matches the method's 24-update schedule"),
}


def iteration_wall_s(host: str, factor: float = 1.0) -> float:
    h = HOSTS[host]
    waves = math.ceil(EPISODES / h["workers"])
    update = UPDATE_S * h.get("update_penalty", 1.0)
    return (waves * WAVE_S * h["contention"] + update) * factor


def wall_h_for_workers(iterations: int, workers: int, *, factor_low: float = 1.0,
                       factor_high: float = 1.3) -> list[float]:
    """Wall-hour band for an arbitrary worker count (same model: contention 1.1 above 30
    workers, update penalty 1.1 below 16 cores' worth of workers)."""
    waves = math.ceil(EPISODES / int(workers))
    contention = 1.1 if workers > 30 else 1.0
    update = UPDATE_S * (1.1 if workers < 16 else 1.0)
    per_it = waves * WAVE_S * contention + update
    return [round(iterations * per_it * f / 3600.0, 2) for f in (factor_low, factor_high)]


def option_quote(name: str, *, factor_low: float = 1.0, factor_high: float = 1.3) -> dict:
    opt = OPTIONS[name]
    it = int(opt["iterations"])
    controlled = it * EPISODES * HORIZON_S
    planned = it * EPISODES * EPISODE_SIM_S
    walls = {host: [round(it * iteration_wall_s(host, f) / 3600.0, 2) for f in (factor_low, factor_high)]
             for host in HOSTS}
    return {"option": name, "init": opt["init"], "iterations": it, "episodes": it * EPISODES,
            "controlled_s": controlled, "max_sumo_seconds_flag": controlled,
            "planned_sim_s": planned, "smoke_sim_s": SMOKE_SIM_S,
            "planned_total_sim_s": planned + SMOKE_SIM_S,
            # S-MAPPOSPEED rule: one failed-attempt retry per episode reserved (cap, not expected spend)
            "hard_cap_sim_s": 2 * planned + 2 * SMOKE_SIM_S,
            "wall_h": walls, "purpose": opt["purpose"]}


def sanity_eval_quote(worlds: int = 12, arms: int = 3) -> dict:
    sim = worlds * arms * EPISODE_SIM_S
    return {"worlds": worlds, "arms": arms, "sim_s": sim,
            "wall_min_12_workers": round(math.ceil(worlds * arms / 12) * 45.0 / 60.0, 1)}


def quote(factor_low: float = 1.0, factor_high: float = 1.3) -> dict:
    return {"model": {"wave_s": round(WAVE_S, 2), "update_s": round(UPDATE_S, 2),
                      "episodes_per_iteration": EPISODES, "episode_sim_s": EPISODE_SIM_S,
                      "wz_factor": [factor_low, factor_high],
                      "iteration_wall_s_merge": {h: round(iteration_wall_s(h), 1) for h in HOSTS},
                      "hosts": HOSTS},
            "options": {n: option_quote(n, factor_low=factor_low, factor_high=factor_high) for n in OPTIONS},
            "sanity_eval": sanity_eval_quote()}


def markdown(q: dict) -> str:
    rows = ["| option | init | it | planned sim-s (excl. 660 smoke) | hard cap | cpuV 46w h | cpuQ 30w h | cpuV 12-core h |",
            "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, o in q["options"].items():
        w = o["wall_h"]
        rows.append(f"| {name} | {o['init']} | {o['iterations']} | {o['planned_sim_s']:,.0f} | "
                    f"{o['hard_cap_sim_s']:,.0f} | {w['cpuV'][0]}-{w['cpuV'][1]} | "
                    f"{w['cpuQ'][0]}-{w['cpuQ'][1]} | {w['cpuV_shared12'][0]}-{w['cpuV_shared12'][1]} |")
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--wz-factor-low", type=float, default=1.0)
    ap.add_argument("--wz-factor-high", type=float, default=1.3)
    a = ap.parse_args(argv)
    q = quote(a.wz_factor_low, a.wz_factor_high)
    print(json.dumps(q, indent=1) if a.json else markdown(q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
