"""A2 command parity: the emulator INSIDE the engine (teacher-forced)
vs the logged command sequence - the E4 contract per arm.

For a declared deterministic subset of TRAIN worlds, run the kind
'idm_action' rollout teacher-forced over the 30 s window (the state at
every control tick equals the logged world state - the E4 frame
conditions) and compare the engine's issued targets and fallback
classification with the logged suffix_commands rows:

  matched  = |issued - emulated| <= 0.01 m/s with identical fallback
             classification (issued <-> finite target, fallback <->
             no target)
  mismatch = anything else (listed, first 10)

Rows for CAVs not live in the teacher-forced state are skipped (E4
liveness convention).  ZERO simulation; TRAIN only.
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

WORLD_STRIDE = 8                    # declared deterministic subset
T_START = time.time()


def main():
    try:
        torch.set_num_threads(2)
    except Exception:
        pass
    worlds = L.train_worlds()
    names = sorted(worlds)[::WORLD_STRIDE]
    print("=== A2 command parity: engine-in-the-loop emulator vs logged "
          "commands (teacher-forced) ===")
    print(f"declared deterministic subset: every {WORLD_STRIDE}th sorted "
          f"TRAIN world -> {len(names)} worlds x 7 arms")
    stats = defaultdict(lambda: dict(rows=0, matched=0, skipped_dead=0,
                                     bad=[]))
    for world in names:
        t0w = float(worlds[world]["t0_s"])
        w = L.ActWorld(world, t0w)
        for arm in w.arms:
            # teacher truth keeps junction rows with lane -1: a CAV
            # inside a junction is absent from the model-world state
            # (zero-length boundary convention), so its logged command
            # row cannot be reproduced by ANY model-world engine and is
            # skipped (E4 checked those rows through the raw internal
            # lane; that part of the sensor has no model-world
            # counterpart)
            truth = w.truth_rows_teacher(arm)
            r = AE.ActionRollout(w, arm, member_seed=0)
            issued_hist = {}
            live_at = {}
            orig = r._control

            def spy(tick, _r=r, _o=orig):
                tick6 = round(float(tick), 6)
                live_at[tick6] = {str(x) for x in _r.names}
                _o(tick)
                for k, v in _r.cmd.items():
                    issued_hist[(tick6, k)] = v
            r._control = spy
            r.run_teacher(truth, n_ticks=L.H30_TICKS)
            cmds = w.commands(arm)
            st = stats[arm]
            for (tick, vid), (owner, issued, fallback) in \
                    sorted(cmds.items()):
                tick6 = round(tick, 6)
                if vid not in live_at.get(tick6, set()):
                    st["skipped_dead"] += 1
                    continue
                emu = issued_hist.get((tick6, vid))
                st["rows"] += 1
                if issued is None:
                    ok = emu is None or emu != emu
                else:
                    ok = emu is not None and emu == emu and \
                        abs(emu - issued) <= 0.01
                if ok:
                    st["matched"] += 1
                elif len(st["bad"]) < 10:
                    st["bad"].append(f"{world}__{arm} {vid}@{tick}: "
                                     f"log {issued} (owner {owner}, "
                                     f"fb {fallback}) vs emu {emu}")
    print(f"\n--- per arm: matched/rows (rate); skipped rows for CAVs "
          f"already exited: liveness gate ---")
    result = dict(experiment="WD-01 partA_act command parity",
                  worlds=names, per_arm={}, match_tol_mps=0.01)
    for arm in L.BRANCH_ARMS:
        st = stats.get(arm)
        if not st or st["rows"] == 0:
            continue
        rate = st["matched"] / st["rows"]
        print(f"{arm:>11s}: {st['matched']}/{st['rows']} ({rate:.6f})  "
              f"skipped_dead {st['skipped_dead']}  "
              f"mismatches {len(st['bad']) and st['rows'] - st['matched']}")
        result["per_arm"][arm] = dict(
            rows=st["rows"], matched=st["matched"], rate=rate,
            skipped_dead=st["skipped_dead"],
            mismatch_examples=st["bad"])
    out = HERE / "results" / "command_parity_act.json"
    out.write_text(json.dumps(result, indent=1, default=str),
                   encoding="utf-8")
    print(f"\nwrote {out}  ({time.time()-T_START:.0f}s)")


if __name__ == "__main__":
    main()
