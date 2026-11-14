"""HORIZON-CHECK part B add-on: imagine extra WM tags on the decision points of an existing
branch_horizon.py episode (no new SUMO arms).

Replays the episode's native base exactly as branch_horizon.py did (base to t_k, saveState,
fresh-process loadState before continuing), checks the decision point matches the recorded
one (CAV count, group, cohort size), re-imagines every controller's arms with the requested
WM tags, and replays the all-native B arm once (fresh process + loadState, 120 s) for the
per-vehicle HDV |ds| of the imagined B.  Output: ``<episode>/reimagine_<suffix>.jsonl``.

    python scripts/horizon_check/reimagine_B.py --episode <dir> --suffix pe \
        --wm C4=<ckpt>,mean,1 --wm C4R=<ckpt>,mean,1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np

import branch_horizon as BH
from branch_horizon import BR, HORIZONS


def main() -> int:
    from traffic_shepherd.c4.env import CavSumoEnv
    from traffic_shepherd.c4.seeds import check_case

    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--wm", action="append", required=True)
    args = ap.parse_args()
    ep = BR.checked(args.episode)
    man = json.loads((ep / "MANIFEST.json").read_text(encoding="utf-8"))["args"]
    points = [json.loads(x) for x in (ep / "points.jsonl").read_text(encoding="utf-8").splitlines()
              if x.strip()]
    out_path = ep / f"reimagine_{args.suffix}.jsonl"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists (never overwrite)")
    check_case(man["seed"], man["demand"])
    wms = BH.parse_wms(args.wm)
    times = [int(t) for t in man["times"].split(",")]
    work = ep / f"reimagine_{args.suffix}_work"
    work.mkdir(exist_ok=False)
    env = CavSumoEnv(k_slots=192, controller="direct", extra_sumo_cli=BR.SAVE_STATE_CLI,
                     scenario_root=work / "scenarios", physics_log=False)
    native = BR.BranchController("native")
    ctrls = [BR.BranchController(c, seed=man["rng_seed"] + man["seed"], name=lab)
             for c, lab in zip(man["controllers"], man["labels"] or man["controllers"])]
    end = times[-1] + max(HORIZONS) + 1
    t_start = time.monotonic()
    env.reset(seed=man["seed"], demand_vph=man["demand"], penetration=man["penetration"],
              warmup_s=man["warmup"], horizon_s=end - man["warmup"], controller="direct")
    fo = open(out_path, "w", encoding="utf-8")
    try:
        for k, t_k in enumerate(times):
            while env.now < t_k - 1e-9:
                BR.base_step(env, native, None)
            rec = next(p for p in points if int(p["t_s"]) == t_k)
            path = work / f"s{man['seed']}_t{t_k}.xml"
            BR.save_state(env, path)
            snap = BR.snapshot_env(env)
            cavs = BR.cav_positions(env)
            group = BR.select_group(cavs)
            cohort = BH.present_vids(env)
            match = {"n_cavs": len(cavs) == rec["n_cavs"], "group": group == rec["group"],
                     "n_cohort": len(cohort) == rec["n_cohort"]}
            row = {"seed": man["seed"], "penetration": man["penetration"], "t_s": t_k,
                   "match": match, "controllers": {}, "B_hdv_abs_ds_median_m": {}}
            if not all(match.values()):
                row["error"] = "decision point does not match the recorded episode"
                fo.write(json.dumps(row) + "\n")
                fo.flush()
                break
            imag = {}
            for c in ctrls:
                arms = [{"name": a, "members": (frozenset() if a == "B" else None if a == "ALL"
                                                else frozenset(rec["group"]) if a == "GROUP"
                                                else frozenset([a[5:]]))}
                        for a in rec["controllers"][c.name]["arms"]]
                for tag, (ckpt, mode, n_s) in wms.items():
                    im = BH.imagine_multi(ckpt, env, arms, c, c.snapshot(), n_samples=n_s, seed=0,
                                          driver_mode=mode)
                    imag[(c.name, tag)] = im
                    row["controllers"].setdefault(c.name, {})[tag] = {
                        "wall_s": im["wall_s"], "slots": im["slots"],
                        "h": {h: {"metrics": im["h"][h]["metrics"], "D": im["h"][h]["D"],
                                  "zone_by_arm": im["h"][h]["zone_by_arm"],
                                  "field_B": im["h"][h]["field_B"]} for h in HORIZONS}}
                    print(f"seed {man['seed']} t {t_k} {c.name} {tag}: {im['wall_s']:.0f} s",
                          flush=True)
            # replay B once for per-vehicle |ds|
            BR.load_state(env, path, snap)
            BR.prepare_branch_mode(env)
            snaps_b, traj_b = BH.run_arm(env, native, frozenset(), None, cohort, record=True)
            traj = dict(traj_b)
            row["B_replay_road_delay_vs"] = {h: snaps_b[h]["all"]["road_delay_vs"] for h in HORIZONS}
            row["B_recorded_road_delay_vs"] = {h: rec["sumo_B"][str(h)]["all"]["road_delay_vs"]
                                               for h in HORIZONS}
            for (cname, tag), im in imag.items():
                for h in HORIZONS:
                    r = traj.get(round(t_k + h, 3), {})
                    d = [abs(im["h"][h]["B_s"][v] - r[v][0]) for v in im["h"][h]["B_s"]
                         if v in r and v not in cavs]
                    row["B_hdv_abs_ds_median_m"].setdefault(tag, {}).setdefault(cname, {})[h] = (
                        float(np.median(d)) if d else None)
            fo.write(json.dumps(BH.jsonable(row), allow_nan=False) + "\n")
            fo.flush()
            if k + 1 < len(times):
                BR.load_state(env, path, snap)
    finally:
        fo.close()
        env.close()
    print(f"done in {time.monotonic() - t_start:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
