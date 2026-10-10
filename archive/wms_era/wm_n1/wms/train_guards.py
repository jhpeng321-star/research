"""Read-only per-window controller exclusions, independent of list filenames."""
from __future__ import annotations

import numpy as np

from .train import TrainDataRefusal


class TrainingWindowGuard:
    def __init__(self, *, horizon_steps, heldout=None, leave_pi_wall_out=False):
        self.horizon_steps = int(horizon_steps)
        self.heldout = heldout
        self.leave_pi_wall_out = leave_pi_wall_out

    def __call__(self, corpus, positions):
        kept, dropped = [], 0
        for i in positions:
            meta = corpus._items[corpus._order[i]]
            if self.heldout is not None:
                self.heldout.refuse_worlds([meta["world"]], where="original TRAIN windows")
            if self.leave_pi_wall_out:
                tick = int(meta["tick"])
                end = min(corpus._run_end(tick), tick + self.horizon_steps + 1)
                bad = False
                for t in range(tick, end):
                    row = np.asarray(corpus._array("veh_step")[t])
                    cav = np.asarray(corpus._array("veh_mask")[t], bool) & (row[:, 7] > .5)
                    if not np.isfinite(row[cav, 8:14]).all():
                        raise TrainDataRefusal("nonfinite controller ownership channels")
                    if (cav & ((row[:, 10] > .5) | (row[:, 11] > .5) | (row[:, 13] > .5))).any():
                        bad = True
                        break
                if bad:
                    dropped += 1
                    continue
            kept.append(i)
        return kept, {"leave_pi_wall_out": self.leave_pi_wall_out,
                      "controller_windows_dropped": dropped, "n_kept": len(kept),
                      "held_world_guard": self.heldout is not None}
