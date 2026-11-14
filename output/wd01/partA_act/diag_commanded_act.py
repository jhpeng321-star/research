"""A2 diagnostic: COMMANDED-CAV speed tracking, act vs naive.

The per-arm pooled CAV RMSE mixes commanded and uncommanded CAV steps,
diluting the action-conditioning signal (only CAVs inside the
dispatcher's control zone are commanded).  This diagnostic restricts
the id-matched speed error to CAV steps that were actively commanded
(the engine issued a target at the governing control tick) on the
high-penetration TRAIN worlds, act vs the CAVs-as-native naive roll.

ZERO simulation; TRAIN only.  Supplementary to validate_act.py (does
not modify the raw rows).
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("WD_DEVICE", "cpu")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import act_lib as L  # noqa: E402
import act_engine as AE  # noqa: E402

T_START = time.time()


def main():
    try:
        torch.set_num_threads(2)
    except Exception:
        pass
    worlds = L.train_worlds()
    names = sorted(w for w in worlds
                   if worlds[w].get("penetration", 0) >= 0.20)
    print("=== A2 diagnostic: commanded-CAV speed tracking (act vs "
          "naive), penetration >= 0.20 ===")
    print(f"{len(names)} worlds x 7 arms x {{act, naive}}")
    acc = defaultdict(lambda: dict(n=0, se=0.0))
    for world in names:
        t0w = float(worlds[world]["t0_s"])
        w = L.ActWorld(world, t0w)
        for arm in w.arms:
            truth = w.truth_rows(arm)
            cav_names = {n for n in w.cav_ids}
            for variant in ("act", "naive"):
                r = AE.ActionRollout(w, arm, member_seed=0,
                                     force_native=(variant == "naive"))
                cmd_at = {}
                orig = r._control

                def spy(tick, _r=r, _o=orig):
                    _o(tick)
                    cmd_at[round(float(tick), 6)] = set(_r.cmd)
                r._control = spy
                r.run(L.H30_TICKS)
                snaps = r.snap_dicts()
                a = acc[f"{arm}|{variant}"]
                for j in range(1, L.H30_TICKS + 1):
                    tick = round(r.t0 + (j - 1) * AE.DT, 6)
                    ctick = round(tick - (tick % 1.0), 6)
                    commanded = cmd_at.get(ctick, set())
                    for name, st in snaps[j].items():
                        if name not in cav_names or name not in truth[j]:
                            continue
                        if name not in commanded:
                            continue
                        tv = truth[j][name][2]
                        a["se"] += (st[2] - tv) ** 2
                        a["n"] += 1
    print("\n--- commanded-CAV-step speed RMSE (m/s) ---")
    result = dict(experiment="WD-01 partA_act commanded-CAV tracking",
                  worlds=names, per_arm={})
    for arm in L.BRANCH_ARMS:
        row = {}
        for variant in ("act", "naive"):
            a = acc.get(f"{arm}|{variant}")
            if a and a["n"]:
                row[variant] = dict(n_steps=a["n"],
                                    rmse=float(np.sqrt(a["se"] / a["n"])))
        if row:
            print(f"{arm:>11s}: " + "  ".join(
                f"{v} rmse {row[v]['rmse']:.3f} (n {row[v]['n_steps']})"
                for v in row))
            result["per_arm"][arm] = row
    out = HERE / "results" / "commanded_tracking_act.json"
    out.write_text(json.dumps(result, indent=1, default=str),
                   encoding="utf-8")
    print(f"\nwrote {out}  ({time.time()-T_START:.0f}s)")


if __name__ == "__main__":
    main()
