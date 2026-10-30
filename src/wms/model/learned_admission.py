"""Physical admission from the same observable inferred driver draw as entrant motion."""
from __future__ import annotations

import hashlib
import math

from .entries import NOMINAL_LENGTH_M, SUMO_POSITION_EPS_M, _NO_GAP, sumo_safe_insert_speed


class LearnedAdmission:
    """No trained parameters, class tables, assignments, future service or behaviour constants.

    Coupled candidate ordinals advance only on insertion when conservation is ON.
    `LearnedMix` uses exactly this key when the candidate becomes an actual newborn.
    """
    def __init__(self, mix):
        self.mix = mix
        if not {"v0", "time_headway", "decel", "min_gap"} <= set(mix.names):
            raise ValueError("inferred admission settings missing")

    def candidate(self, b, entrance, ordinal):
        key = f"errsrc-mix:{self.mix.seed}:{self.mix.row_keys[b]}:{entrance}:{ordinal}"
        index = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") % len(self.mix.pools[b])
        values = self.mix.pools[b][index].detach().cpu().tolist()
        result = dict(zip(self.mix.names, values))
        if any(not math.isfinite(result[n]) or result[n] <= 0 for n in ("v0", "time_headway", "decel", "min_gap")):
            raise ValueError("nonphysical inferred admission settings")
        return result

    def place(self, process, work, b, entrance, lanes, cav, ordinal):
        if not process.cfg.conserve_queue or process.cfg.learned_entry is not None:
            raise ValueError("learned admission needs conservation and no learned-entry hazard bundle")
        params = self.candidate(b, entrance, ordinal)
        length = float(NOMINAL_LENGTH_M)  # incumbent observable geometric nominal, not a driver setting
        if len(lanes) > 1:
            if process.cfg.sumo_lane_mode == "best_free":
                lanes = [min(lanes, key=lambda ln: process._lane_occupancy(work, b, ln))]
            else:
                lanes = sorted(lanes, key=lambda ln: -process._entrance_gap(work, b, ln, length)[0])
        for lane in lanes:
            gap, leader_speed = process._entrance_gap(work, b, lane, length)
            v_cap = min(params["v0"], float(process.cfg.sumo_lane_speed_mps[lane]))
            if gap >= _NO_GAP:
                speed = v_cap
            else:
                # End-of-insertion-step geometry; all behavioural settings are inferred.
                if not (process.cfg.fresh_no_credit and process._rear_is_fresh(work, b, lane)):
                    gap += max(0., leader_speed) * process.dt
                speed = min(v_cap, sumo_safe_insert_speed(
                    gap, 0., params["time_headway"], params["decel"], params["min_gap"]))
                if gap < params["min_gap"]:
                    continue
            if process._place(work, b, entrance, lane, cav, speed, length):
                slot = work.inserted[-1][1]
                work.s[b, slot] = process.entry_s[lane] + length + SUMO_POSITION_EPS_M
                return True
        return False
