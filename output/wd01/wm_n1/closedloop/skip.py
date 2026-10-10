"""Exact identical-rollout detection for single-CAV candidates.

A candidate that differs from an already-rolled REFERENCE plan only in CAV
``i`` (native in the reference, damping/PI in the candidate) has a rollout
bitwise identical to the reference as long as CAV ``i``'s commanded
execution never changes its speed: before any divergence the candidate's
world equals the reference world, so its control-tick inputs are the
reference snapshots, and the only place the action enters
``rollout2.step.step_once`` is the commanded execution

    v_i' = execution_speed(held_target_i, v_i, idm_raw_i)   (if finite)

versus the native ``v_i' = max(0, v_i + baseline_idm_i * dt)`` (CAV slots
carry no HDV residual). Everything after that line (arrivals off, gap
repair, exits) is a deterministic function of the identical state.

``identical_slots`` replays the controller (``bundle.ctrl``, the same
CavMover call and inputs as step_once's control stage) on the reference
snapshots with action ``a`` on every reference-native CAV slot at once --
per-slot controller outputs depend only on the world and the slot's own
PI state, never on other slots' codes -- and compares the commanded
execution speed with the reference's native speed at every physics step,
using the reference's exact ``idm_raw`` / ``baseline_idm`` captured during
its rollout. A slot whose candidate never diverges over the whole horizon
can reuse the reference rollout (parity-tested against full rollouts).
"""
from __future__ import annotations

import numpy as np
import torch

_TRACE = None          # list | None: active capture of V2GCore.accelerations


def install_trace_hook() -> None:
    """Wrap ``V2GCore.accelerations`` once to record ``(slots, idm_raw)``."""
    from rollout2 import baseline

    if getattr(baseline.V2GCore.accelerations, "_cl_traced", False):
        return
    orig = baseline.V2GCore.accelerations

    def accelerations(self, w, *args, **kwargs):
        out = orig(self, w, *args, **kwargs)
        if _TRACE is not None:
            _TRACE.append((np.asarray(w.slots, np.int64).copy(),
                           np.asarray(out[1], np.float64).copy()))
        return out

    accelerations._cl_traced = True
    baseline.V2GCore.accelerations = accelerations


class Trace:
    """Context manager capturing per-step per-b idm_raw during a rollout."""

    def __enter__(self):
        global _TRACE
        install_trace_hook()
        self.calls = []
        _TRACE = self.calls
        return self

    def __exit__(self, *exc):
        global _TRACE
        _TRACE = None
        return False

    def idm_raw(self, n_steps: int, b_n: int, k_n: int) -> np.ndarray:
        """[T, B, K] idm_raw (NaN where the slot is absent)."""
        if len(self.calls) != n_steps * b_n:
            raise RuntimeError(f"trace has {len(self.calls)} calls, expected {n_steps * b_n}")
        out = np.full((n_steps, b_n, k_n), np.nan)
        for c, (slots, raw) in enumerate(self.calls):
            j, b = divmod(c, b_n)
            out[j, b, slots] = raw
        return out


def identical_slots(bundle, roll, b: int, ref_codes: np.ndarray, action_code: int,
                    is_cav: np.ndarray, t0_s: float, n_steps: int) -> np.ndarray:
    """bool [K]: reference-native CAV slots whose candidate (slot -> action)
    rollout is bitwise identical to the reference over ``n_steps``."""
    from ctrl.config import CtrlConfig
    from ctrl.mover import CavStepInput, execution_speed
    from rollout2 import baseline
    from rollout2.step import _CTRL_TICK_EPS, _segment_view

    params = baseline.Params(None)
    dt = float(params.dt)
    ctrl = bundle.ctrl
    k_n = roll.pos.shape[2]
    cand = np.asarray(is_cav, bool) & (np.asarray(ref_codes) == 0)
    codes = np.asarray(ref_codes, np.int64).copy()
    codes[cand] = int(action_code)
    acts = torch.as_tensor(codes).reshape(1, -1)
    ws = torch.full((1, k_n), float("nan"), dtype=torch.float64)
    ctrl_cfg = CtrlConfig(dt_s=dt, accel_min_mps2=-float(params.exec_p["b_kin_max_mps2"]),
                          accel_max_mps2=float(params.exec_p["a_kin_max_mps2"]))
    diverged = np.zeros(k_n, bool)
    held = np.full(k_n, np.nan)
    state = None
    length = torch.as_tensor(roll.length[b:b + 1])
    is_cav_t = torch.as_tensor(np.asarray(is_cav, bool)).reshape(1, -1)
    wall_group = torch.full((1, k_n), -1, dtype=torch.int64)
    for j in range(int(n_steps)):
        t = float(t0_s) + j * dt
        s = roll.pos[j, b]
        v = roll.spd[j, b]
        lane = roll.lns[j, b]
        pres = roll.msk[j, b]
        if abs(t - round(t)) < _CTRL_TICK_EPS:
            if state is None:
                state = ctrl.init_state(1, k_n)
            lo, hi, lim = _segment_view(params, lane[None], s[None])
            inp = CavStepInput(
                s_m=torch.as_tensor(s[None].copy()), v_mps=torch.as_tensor(v[None].copy()),
                lane=torch.as_tensor(lane[None].copy()), length_m=length,
                present=torch.as_tensor(pres[None].copy()), is_cav=is_cav_t,
                wall_group=wall_group, segment_lo_m=torch.from_numpy(lo),
                segment_hi_m=torch.from_numpy(hi), segment_limit_mps=torch.from_numpy(lim),
                t_s=t, control_t_s=round(t, 6))
            with torch.no_grad():
                out = ctrl.step(state, inp, acts, wall_speed_mps=ws)
            held = out.target_speed_mps.detach().to(torch.float64).numpy().reshape(-1).copy()
        live = cand & pres & ~diverged & np.isfinite(held)
        if live.any():
            idx = np.flatnonzero(live)
            v_t = torch.as_tensor(v[idx])
            v_exec = execution_speed(torch.as_tensor(held[idx]), v_t,
                                     torch.as_tensor(roll.idm_raw[j, b, idx]),
                                     config=ctrl_cfg).numpy()
            v_nat = np.maximum(v[idx] + roll.baseline_idm[j, b, idx] * dt, 0.0)
            # torch clamp(min=0) on the native path: v_raw.clamp(min=0.0)
            diff = v_exec != v_nat
            diverged[idx[diff]] = True
        if (diverged | ~cand).all():
            break
    return cand & ~diverged
