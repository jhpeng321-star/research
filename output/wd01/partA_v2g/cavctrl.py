"""WD-01 partA_v2g CAV action path: plan actions -> commanded speeds via
the VERIFIED partA_ctrl emulators (damping ONNX-port, PI external-U,
wall-v2 pacing law) + the direct_target bridge, on ideal-sensor frames of
the PREDICTED state (the act_engine E4-verified segment-local sensing
convention), executed by the engine's measured execution rule.

This module is the ONLY torch-dependent part of partA_v2g (lazy import:
native-only rollouts never touch it).  ZERO SIMULATION.

Action semantics (predict contract §3.3 + partA_ctrl ACTION_CODES):
- native: no command (the generic rules drive the CAV);
- damping: CIRCLES damping decide() on (ego, leader, bumper gap);
- pi-external-u: the PI recurrence (registered U = 31.29 m/s), state
  advanced EVERY control tick for the whole CAV roster (adapter
  invalidate semantics, act_engine convention);
- wall-v2 (law wall-v2.1-closure, owner decision 2026-09-30 Option B):
  during ASSEMBLY the front member (largest engine.s) is paced to the wall
  speed and each rear member to v_wall + min(2.0, 0.1 * gap_to_front)
  (wall_pace with gap_to_front_m); assembly completes when every member is
  within assemble_tol_mps of the wall speed AND span <= 6 m; HOLD paces all
  members to v_wall; assembly not complete 60 s after t0 -> the wall aborts
  (assembly_timeout) and members go native.  Released at release_at_s /
  after hold_s from t0; phase 'release' at t0 means no commands.  Positions
  are engine.s (the shared mainline coordinate, downstream-increasing).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_WD01 = _HERE.parent
# NOTE: never put the partA_v2g dir itself on sys.path -- its predict.py
# would shadow the shared output/wd01/predict kit.
for _p in (str(_WD01), str(_WD01 / "partA_ctrl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _emulators():
    import emulators  # partA_ctrl (torch)
    import refload
    return emulators, refload


class WallBook:
    """Wall member bookkeeping: pacing + assembly tracking."""

    def __init__(self, walls: Sequence, t0: float, params: Mapping):
        self.t0 = float(t0)
        self.tol = float(params["wall"]["assemble_tol_mps"])
        self.specs = {w.wall_id: w for w in walls}
        self.members: Dict[str, List[str]] = {}
        for w in walls:
            for m in w.members:
                self.members.setdefault(m, []).append(w.wall_id)
        self.assembled_at: Dict[str, Optional[float]] = {
            wid: None for wid in self.specs}
        self.safety_abort: Dict[str, bool] = {wid: False for wid in self.specs}
        self.timed_out: Dict[str, bool] = {wid: False for wid in self.specs}
        wp = params["wall"]
        self.span_tol = float(wp.get("assemble_span_tol_m", 6.0))
        self.timeout_s = float(wp.get("assembly_timeout_s", 60.0))

    def active_speed(self, vid: str, t_now: float) -> Optional[float]:
        """The commanded wall speed for this member at t_now, or None
        when not wall-paced (not a member / released)."""
        for wid in self.members.get(vid, ()):
            w = self.specs[wid]
            if w.phase == "release":
                return None
            if self.timed_out[wid]:
                return None
            if self.assembled_at[wid] is None and                     t_now - self.t0 >= self.timeout_s - 1e-9:
                # assembly_timeout safety abort: members handed back native
                self.timed_out[wid] = True
                return None
            if w.release_at_s is not None and t_now - self.t0 >= \
                    float(w.release_at_s):
                return None
            if w.hold_s is not None and t_now - self.t0 >= float(w.hold_s):
                return None
            return float(w.speed_mps)
        return None

    def assembling(self, vid: str) -> Optional[str]:
        """The wall id of this member's still-ASSEMBLING wall (closure law
        applies), else None."""
        for wid in self.members.get(vid, ()):
            if self.assembled_at[wid] is None and not self.timed_out[wid]:
                return wid
        return None

    def observe(self, t_now: float, speeds: Mapping[str, float],
                positions: Optional[Mapping[str, float]] = None) -> None:
        """Assembly check: all members within tol of the wall speed AND
        (when positions are supplied) longitudinal span <= span_tol."""
        for wid, w in self.specs.items():
            if self.assembled_at[wid] is not None or w.phase == "release"                     or self.timed_out[wid]:
                continue
            if not all(abs(speeds.get(m, float("nan")) - float(w.speed_mps))
                       <= self.tol for m in w.members):
                continue
            if positions is not None:
                ps = [positions.get(m, float("nan")) for m in w.members]
                if not all(p == p for p in ps) or max(ps) - min(ps) > self.span_tol:
                    continue
            self.assembled_at[wid] = t_now - self.t0

    def note_fallback(self, vid: str) -> None:
        for wid in self.members.get(vid, ()):
            self.safety_abort[wid] = True


class CavControllerLoop:
    """control_hook(t_now, engine) -> {vid: commanded target speed}."""

    def __init__(self, *, t0: float, params: Mapping,
                 per_cav_actions: Mapping[str, str],
                 walls: Sequence, cav_state_hint: Mapping[str, Mapping],
                 switch_native_at: Optional[float] = None):
        self.t0 = float(t0)
        self.params = params
        self.actions = dict(per_cav_actions)
        self.walls = WallBook(list(walls), t0, params)
        self.hint = dict(cav_state_hint or {})
        self.switch_native_at = switch_native_at
        self._emu = None
        self._pi_store: Dict[str, List] = {}
        self.n_commands = 0
        self.n_fallbacks = 0
        self.called_ticks = 0

    def _ensure_emu(self):
        if self._emu is None:
            emulators, refload = _emulators()
            self._emu = emulators.ControllerEmulators(
                refload.MODEL_ONNX, device=__import__("torch").device("cpu"))
        return self._emu

    def _segments(self, engine):
        """(seg_lo, seg_hi, seg_lim) arrays for the live rows."""
        p = engine.p["geometry"]
        starts = p["lane_segment_starts_m"]
        lims = p["lane_segment_limits_mps"]
        n = len(engine.names)
        lo = np.zeros(n)
        hi = np.zeros(n)
        lim = np.zeros(n)
        lane = engine.lane
        s = engine.s
        for li in (0, 1, 2):
            m = np.where(lane == li)[0]
            if not len(m):
                continue
            st = np.asarray(starts[li], dtype=np.float64)
            lm = np.asarray(lims[li], dtype=np.float64)
            idx = np.searchsorted(st, s[m], side="right") - 1
            idx = np.clip(idx, 0, len(lm) - 1)
            lo[m] = st[idx]
            hi_idx = np.minimum(idx + 1, len(st) - 1)
            hi[m] = np.where(idx + 1 < len(st), st[hi_idx],
                             float(engine.llen[li]))
            lim[m] = lm[idx]
        return lo, hi, lim

    def __call__(self, t_now: float, engine) -> Dict[str, float]:
        self.called_ticks += 1
        emulators, _ = _emulators()
        cav_rows = [r for r in range(len(engine.names)) if engine.cav[r]]
        if not cav_rows:
            return {}
        # v2 continuation 'native': every CAV runs native after the
        # commit window (the plan's actions apply to [t0, t0+Delta] only)
        if self.switch_native_at is not None and \
                t_now >= self.switch_native_at - 1e-9:
            self._tick_pi_only(t_now, engine, cav_rows)
            self.walls.observe(t_now, {engine.names[r]: engine.v[r]
                                       for r in cav_rows},
                               {engine.names[r]: float(engine.s[r])
                                for r in cav_rows})
            return {}
        kinds = []
        wall_speeds = []
        wall_group = []         # per-row assembling wall index (-1 = none)
        wall_ids = list(self.walls.specs)
        for r in cav_rows:
            vid = engine.names[r]
            ws = self.walls.active_speed(vid, t_now)
            if ws is not None:
                kinds.append(emulators.ACTION_CODES["wall-v2"])
                wall_speeds.append(ws)
                aw = self.walls.assembling(vid)
                wall_group.append(-1 if aw is None else wall_ids.index(aw))
            else:
                act = self.actions.get(vid, "native")
                if act == "pi-external-u":
                    kinds.append(emulators.ACTION_CODES["pi-external-u"])
                elif act == "damping":
                    kinds.append(emulators.ACTION_CODES["damping"])
                else:
                    kinds.append(emulators.ACTION_CODES["native"])
                wall_speeds.append(0.0)
                wall_group.append(-1)
        if all(k == emulators.ACTION_CODES["native"] for k in kinds):
            # PI state still advances every tick (invalidate semantics)
            self._tick_pi_only(t_now, engine, cav_rows)
            self.walls.observe(t_now, {engine.names[r]: engine.v[r]
                                       for r in cav_rows},
                               {engine.names[r]: float(engine.s[r])
                                for r in cav_rows})
            return {}
        emu = self._ensure_emu()
        lo, hi, lim = self._segments(engine)
        frames = []
        for r in cav_rows:
            lane_i = int(engine.lane[r])
            pos = float(engine.s[r])
            speed = float(engine.v[r])
            front_cov = min(100.0, max(0.0, float(hi[r]) - pos))
            rear_cov = min(100.0, max(0.0, pos - float(lo[r])))
            front = rear = None
            fb = rb = None
            seg_lo, seg_hi = float(lo[r]), float(hi[r])
            for q in range(len(engine.names)):
                if q == r or int(engine.lane[q]) != lane_i:
                    continue
                sq = float(engine.s[q])
                if sq < seg_lo - 1e-9 or sq > seg_hi + 1e-9:
                    continue          # cross-segment neighbours invisible
                if sq >= pos:
                    gap = sq - 5.0 - pos
                    if gap <= front_cov and (fb is None or gap < fb[0]):
                        fb = (gap, float(engine.v[q]) - speed)
                else:
                    gap = pos - 5.0 - sq
                    if gap <= rear_cov and (rb is None or gap < rb[0]):
                        rb = (gap, float(engine.v[q]) - speed)
            if fb is not None:
                front = {"gap_m": fb[0], "relative_speed_mps": fb[1]}
            if rb is not None:
                rear = {"gap_m": rb[0], "relative_speed_mps": rb[1]}
            frames.append(dict(
                now_s=t_now, ts_s=t_now, speed_mps=speed,
                allowed_speed_mps=float(lim[r]),
                front_coverage_m=front_cov, rear_coverage_m=rear_cov,
                front=front, rear=rear))
        batch = emulators.FrameBatch.stack(frames)
        defaults = [0.0, float("nan"), False]
        pi_state = emulators.ControllerState(pi=emulators.PIState(
            raw_prev=torch_tensor([self._pi_store.get(
                engine.names[r], defaults)[0] for r in cav_rows]),
            last_ts=torch_tensor([self._pi_store.get(
                engine.names[r], defaults)[1] for r in cav_rows]),
            active=torch_bool([self._pi_store.get(
                engine.names[r], defaults)[2] for r in cav_rows])))
        # closure law: gap to the front (max engine.s) of the member's own
        # still-ASSEMBLING wall; members of held/other walls get gap 0.  The
        # front is taken over ALL members of that wall present in the roster.
        grp = torch_tensor(wall_group, dtype="int64")
        wall_pos = [float(engine.s[r]) for r in cav_rows]
        gap = emulators.wall_gap_to_front(torch_tensor(wall_pos), grp)
        out = emu.emulate(torch_tensor(kinds, dtype="int64"), pi_state, batch,
                          wall_speed_mps=torch_tensor(wall_speeds),
                          wall_gap_to_front_m=gap)
        new_pi = out["new_state"].pi
        cmd: Dict[str, float] = {}
        for i, r in enumerate(cav_rows):
            vid = engine.names[r]
            self._pi_store[vid] = [float(new_pi.raw_prev[i]),
                                   float(new_pi.last_ts[i]),
                                   bool(new_pi.active[i])]
            tgt = float(out["target_speed_mps"][i])
            self.n_commands += 1
            if tgt == tgt:            # finite: a command was issued
                cmd[vid] = tgt
            else:
                self.n_fallbacks += 1
                if kinds[i] == emulators.ACTION_CODES["wall-v2"]:
                    self.walls.note_fallback(vid)
        self.walls.observe(t_now, {engine.names[r]: float(engine.v[r])
                                   for r in cav_rows},
                           {engine.names[r]: float(engine.s[r])
                            for r in cav_rows})
        return cmd

    def _tick_pi_only(self, t_now: float, engine, cav_rows) -> None:
        """All-native tick: advance (invalidate) the PI recurrence state
        without loading the ONNX stack beyond emulators import."""
        emulators, _ = _emulators()
        defaults = [0.0, float("nan"), False]
        pi_state = emulators.PIState(
            raw_prev=torch_tensor([self._pi_store.get(
                engine.names[r], defaults)[0] for r in cav_rows]),
            last_ts=torch_tensor([self._pi_store.get(
                engine.names[r], defaults)[1] for r in cav_rows]),
            active=torch_bool([self._pi_store.get(
                engine.names[r], defaults)[2] for r in cav_rows]))
        engaged = torch_bool([False] * len(cav_rows))
        pi = emulators.PIEmulator()
        _, new_state = pi.raw_command(pi_state, engaged,
                                      torch_tensor([float(t_now)]
                                                   * len(cav_rows)),
                                      torch_tensor([0.0] * len(cav_rows)),
                                      torch_tensor([0.0] * len(cav_rows)),
                                      torch_tensor([0.0] * len(cav_rows)))
        for i, r in enumerate(cav_rows):
            self._pi_store[engine.names[r]] = [
                float(new_state.raw_prev[i]), float(new_state.last_ts[i]),
                bool(new_state.active[i])]


def torch_tensor(x, dtype=None):
    import torch
    if dtype == "int64":
        return torch.tensor(x, dtype=torch.int64)
    if dtype == "bool":
        return torch.tensor(x, dtype=torch.bool)
    return torch.tensor(x, dtype=torch.float64)


def torch_bool(x):
    import torch
    return torch.tensor(x, dtype=torch.bool)
