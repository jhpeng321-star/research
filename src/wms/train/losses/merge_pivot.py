"""Persistent-logistic strict and F1 hybrid supplied-path likelihoods.

Recorded geometry is a frozen causal longitudinal replay. Only the boundary
parameters are differentiated on that path. Chunk contributions are conditional
log-survival increments (or the conditional event mass), and telescope exactly.
An unobserved risk-entry age is enumerated and masked, never assigned a prior.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Mapping, Optional

import torch
from torch import Tensor

from wms.model.merge_pivot import (advance_longitudinal, geometry, hybrid_log_masses,
                                  log_event_mass, log_survival, ramp_lane_end_stop)

from .base import LossContext, LossTerm, TermValue, UnrollTargets

PARAM_NAMES = ("s_f", "T_f", "b_f", "s_l", "T_l", "b_l", "tau")


@dataclass(frozen=True)
class MergePivotConfig:
    weight: float = 0.0
    tf_weight: float = 0.0
    metrics: bool = False
    z_tf: float = 1.0
    z_on: float = 1.0
    tau_reg_weight: float = 0.0
    w_reg_weight: float = 0.0
    passage_law: str = "strict"
    quadrature_order: int = 32
    pos_tol_m: float = 25.0
    speed_tol_mps: float = 5.0
    gap_tol_m: float = 20.0
    gap_speed_tol_mps: float = 5.0
    merge_lane: int = 2
    target_lane: int = 1
    zone_s0_m: float = 0.0
    zone_s1_m: float = 75.44

    @property
    def enabled(self) -> bool:
        return bool(self.weight or self.tf_weight or self.metrics or self.tau_reg_weight or self.w_reg_weight)


def score_chunk(c: Tensor, tau: Tensor, risk: Tensor, event: Tensor, *,
                known_entry: Tensor, reset: Tensor, eligible: Optional[Tensor] = None,
                support: Optional[Tensor] = None, stop: Optional[Tensor] = None,
                 carry: Optional[Mapping[str, Tensor]] = None,
                 initial_maximum: Optional[Tensor] = None,
                 w: Optional[Tensor] = None, quadrature_order: int = 32,
                 initial_history: Optional[Tensor] = None) -> tuple[Tensor, dict, dict]:
    """Exact sequence NLL on [B,T,K], with absorbing exits/support/event lifecycle.

    ``known_entry`` marks births whose causal risk history is complete. An explicit
    ``initial_maximum`` supplies a survival-conditioned entry posterior instead.
    ``reset`` is identity lifecycle metadata, never a boundary feature. A temporary
    physical veto (eligible=False) preserves the posterior; leaving risk does not.
    Returned carry is detached for TBPTT. Exact zero mass remains infinite NLL.
    """
    shape = c[:, 0].shape
    carry = {} if carry is None else carry
    if w is not None and initial_maximum is not None:
        raise ValueError("hybrid survivor initialization needs causal history, not a strict maximum")
    history = carry.get("history", c.new_empty((*shape, 0)))
    if initial_history is not None and not carry:
        history = torch.where(torch.isnan(initial_history),
                              torch.full_like(initial_history, -torch.inf), initial_history).to(c)
    maximum = carry.get("maximum", torch.full_like(c[:, 0], -torch.inf))
    if initial_maximum is not None and not carry:
        maximum = torch.where(torch.isnan(initial_maximum), torch.full_like(initial_maximum, -torch.inf),
                              initial_maximum).to(c)
    started = carry.get("started", torch.zeros(shape, dtype=torch.bool, device=c.device))
    terminal = carry.get("terminal", torch.zeros_like(started))
    known = carry.get("known", torch.zeros_like(started))
    doomed = carry.get("doomed", torch.zeros_like(started))
    # The maximum is sufficient for this truncated logistic posterior. Recompute
    # logS with the CURRENT tau graph, rather than losing its denominator gradient
    # when a TBPTT carry was detached. The saved logS is also exposed to callers.
    previous_logS = log_survival(maximum, tau)
    integrated = None
    if w is not None:
        previous_logS = (hybrid_log_masses(history, tau, w, quadrature_order)[1][..., -1]
                         if history.shape[-1] else torch.zeros_like(c[:, 0], dtype=torch.float64))
        # The ordinary no-identity-reset chunk needs ONE integration of its whole
        # fixed path, not a growing quadratic grid rebuilt at every tick. A reset
        # inside a chunk uses the general streaming fallback below.
        if not bool(reset[:, 1:].any()):
            prefix = torch.where(reset[:, 0].bool().unsqueeze(-1),
                                 torch.full_like(history, -torch.inf), history)
            seen = started & ~reset[:, 0].bool()
            ended = terminal & ~reset[:, 0].bool()
            rows = []
            for j in range(c.shape[1]):
                seen = seen | (risk[:, j].bool() & ~ended)
                ended = ended | (seen & ~risk[:, j].bool())
                if support is not None:
                    ended = ended | (seen & ~support[:, j].bool())
                active = seen & ~ended & risk[:, j].bool()
                feasible = eligible[:, j].bool() if eligible is not None else c[:, j] != -torch.inf
                current = torch.where(active & feasible, c[:, j], torch.full_like(c[:, j], -torch.inf))
                rows.append(current)
                ended = ended | (active & (event[:, j].bool() | (current == torch.inf)))
                if stop is not None:
                    ended = ended | (active & stop[:, j].bool())
            path = torch.cat((prefix, torch.stack(rows, -1)), -1)
            integrated = hybrid_log_masses(path, tau, w, quadrature_order)
    nll = tau.sum() * 0.0
    stats = dict(events=0, positive_mass_events=0, zero_mass_events=0, zero_survival=0, physical_fail_events=0,
                 unknown_age=0, unknown_age_events=0, births=0, exits=0,
                 support_failures=0, stopped=0, scored_intervals=0, eligible_intervals=0)
    for i in range(c.shape[1]):
        new = reset[:, i].bool()
        maximum = torch.where(new, torch.full_like(maximum, -torch.inf), maximum)
        previous_logS = torch.where(new, torch.zeros_like(previous_logS), previous_logS)
        if w is not None:
            history = torch.where(new.unsqueeze(-1), torch.full_like(history, -torch.inf), history)
        started, terminal, known, doomed = (x & ~new for x in (started, terminal, known, doomed))
        born = risk[:, i].bool() & ~started & ~terminal
        is_known = known_entry[:, i].bool()
        if i == 0 and initial_maximum is not None and not carry:
            is_known = is_known | (~torch.isnan(initial_maximum) & ~new)
        if i == 0 and initial_history is not None and not carry:
            is_known = is_known | (~torch.isnan(initial_history).any(-1) & ~new)
        known = torch.where(born, is_known, known)
        stats["births"] += int(born.sum())
        stats["unknown_age"] += int((born & ~known).sum())
        started = started | born
        exited = started & ~terminal & ~risk[:, i].bool()
        stats["exits"] += int(exited.sum())
        failed = (started & ~terminal & risk[:, i].bool() & ~support[:, i].bool()
                  if support is not None else torch.zeros_like(started))
        stats["support_failures"] += int(failed.sum())
        terminal = terminal | exited | failed
        active = started & ~terminal & risk[:, i].bool()
        ev = active & event[:, i].bool()
        doomed_event = doomed & event[:, i].bool() & risk[:, i].bool() & known
        stats["events"] += int(doomed_event.sum())
        stats["zero_mass_events"] += int(doomed_event.sum())
        stats["unknown_age_events"] += int((ev & ~known).sum())
        feasible = eligible[:, i].bool() if eligible is not None else c[:, i] != -torch.inf
        stats["physical_fail_events"] += int((ev & ~feasible).sum())
        used = active & known
        current = torch.where(active & feasible, c[:, i], torch.full_like(maximum, -torch.inf))
        updated = torch.maximum(maximum, current)
        next_logS = log_survival(updated, tau)
        if w is not None:
            history = torch.cat((history, current.unsqueeze(-1)), -1)
            if integrated is None:
                le, ls = hybrid_log_masses(history, tau, w, quadrature_order)
                hybrid_event, next_logS = le[..., -1], ls[..., -1]
            else:
                index = path.shape[-1] - c.shape[1] + i
                hybrid_event, next_logS = integrated[0][..., index], integrated[1][..., index]
        # No subtraction of -inf - -inf on impossible paths, even in masked branches.
        safe_den = torch.where(torch.isfinite(previous_logS), previous_logS,
                               torch.zeros_like(previous_logS))
        logp_event = log_event_mass(maximum, updated, tau) - safe_den
        if w is not None:
            logp_event = hybrid_event - safe_den
        logp_wait = next_logS - safe_den
        logp = torch.where(ev, logp_event, logp_wait)
        impossible_entry = used & torch.isneginf(previous_logS)
        logp = torch.where(impossible_entry, torch.full_like(logp, -torch.inf), logp)
        nll = nll - torch.where(used, logp, torch.zeros_like(logp)).sum()
        stats["events"] += int((ev & known).sum())
        stats["positive_mass_events"] += int((ev & known & torch.isfinite(logp_event)).sum())
        stats["zero_mass_events"] += int((ev & known & torch.isneginf(logp_event)).sum())
        impossible_wait = used & ~ev & torch.isneginf(next_logS)
        stats["zero_survival"] += int(impossible_wait.sum())
        stats["zero_survival"] += int((impossible_entry & ev).sum())
        stats["scored_intervals"] += int(used.sum())
        stats["eligible_intervals"] += int((used & feasible).sum())
        maximum = updated
        previous_logS = torch.where(used, next_logS, previous_logS)
        terminal = terminal | ev | impossible_wait
        doomed = (doomed | impossible_wait) & ~event[:, i].bool() & risk[:, i].bool()
        if stop is not None:
            stats["stopped"] += int((active & ~ev & stop[:, i].bool()).sum())
            terminal = terminal | (active & stop[:, i].bool())
    posterior = {"maximum": maximum.detach(), "logS": previous_logS.detach(),
                 "started": started.detach(), "terminal": terminal.detach(), "known": known.detach(),
                 "doomed": doomed.detach()}
    if w is not None:
        posterior["history"] = history.detach()
    return nll, posterior, stats


def _successor_labels(inp, shifts: Tensor, junction: Tensor, i: int) -> dict[str, Tensor]:
    """``label_*``: the RECORDED successor row (route frame), for label-side diagnosis only.

    SUMO moves every vehicle and then changes lanes, so row ``i + 1`` carries the true
    decision positions.  The model never reads these: they classify which vetoed
    recorded events are P1-unrepresentable (``merge_pivot_support``) and feed audits.
    """
    states, present = inp.states, inp.present.bool()
    kin, now = states[:, i], present[:, i]
    out = {"label_pre_junction": junction[:, i] & now}
    if i + 1 >= states.shape[1]:
        zero = torch.zeros_like(kin[..., 0])
        return {**out, "label_succ_x": zero, "label_succ_length": zero,
                "label_succ_lane": torch.zeros_like(kin[..., 0], dtype=torch.long),
                "label_linked": torch.zeros_like(now)}
    nxt, there = states[:, i + 1], present[:, i + 1]
    continuity = getattr(inp, "continuity", None)
    linked = now & there if continuity is None else now & there & continuity[:, i].bool()
    out.update(label_succ_x=nxt[..., 0] + shifts[:, i + 1].to(nxt.dtype),
               label_succ_length=nxt[..., 6],
               label_succ_lane=nxt[..., 1:4].argmax(-1), label_linked=linked)
    return out


@torch.no_grad()
def build_recorded_geometry(model, inp, targets: UnrollTargets) -> dict[str, Tensor]:
    """Frozen forward from each RECORDED PRESTATE, followed by shared causal motion.

    No next-row position/speed/acceleration or teacher trajectory is consumed. A
    separate controller instance replays available current commands, leaving the
    on-policy controller untouched. All present scene vehicles, including CAVs,
    remain neighbors; supervision masks are applied only by the likelihood.
    """
    scenes = getattr(inp, "merge_pivot_scene", None)
    if scenes is not None:
        rows = []
        for b, scene in enumerate(scenes):
            values = vars(inp).copy()
            for key in ("states", "present", "is_cav", "hist", "hist_mask",
                        "observed_hist", "observed_hist_mask", "controller_id"):
                val = values.get(key)
                values[key] = None if val is None else val[b:b + 1]
            values["merge_pivot_scene"] = None
            if scene is not None:
                for key in ("states", "present", "is_cav", "continuity"):
                    val = torch.as_tensor(scene[key], device=inp.states.device).unsqueeze(0)
                    pad = inp.states.shape[2] - val.shape[2]
                    if pad:
                        val = torch.nn.functional.pad(val, (0, 0, 0, pad) if key == "states"
                                                      else (0, pad))
                    values[key] = val
                values["action_timeline"] = {
                    key: (float(val) if key == "control_dt_s" else
                          torch.as_tensor(val, device=inp.states.device).unsqueeze(0))
                    for key, val in scene["action_timeline"].items()}
                for key, val in values["action_timeline"].items():
                    if torch.is_tensor(val) and val.shape[-1] < inp.states.shape[2]:
                        values["action_timeline"][key] = torch.nn.functional.pad(
                            val, (0, inp.states.shape[2] - val.shape[-1]))
            elif inp.action_timeline is not None:
                values["action_timeline"] = {
                    key: val if key == "control_dt_s" else val[b:b + 1]
                    for key, val in inp.action_timeline.items()}
            rows.append(build_recorded_geometry(model, SimpleNamespace(**values), targets))
        return {key: torch.cat([row[key] for row in rows]) for key in rows[0]}

    from wms.model import cav as cav_motion
    from wms.model.controllers import DIRECT_ID
    from wms.model.history import history_step_features, initial_history, push_history
    from wms.physics.neighbours import lead_neighbour
    from ..cav_drive import TimelineControllers
    from ..inputs import state_from_kin

    controller = (TimelineControllers(model, inp.action_timeline)
                  if inp.action_timeline is not None else None)
    # Lossless route coordinate (junction lengths + exact dead reckoning of junction
    # rows) from the causal scene alone: every consumer of the recorded geometry
    # (fitter, support audit, likelihood) goes through this single construction.
    from wms.data.route_frame import route_shift
    shifts, junction = route_shift(inp.states, inp.present, getattr(inp, "continuity", None),
                                   dt_s=float(model.physics.dt_s),
                                   zone_start_m=float(model.physics.merge_zone_s0_m),
                                   zone_end_m=float(model.physics.merge_zone_s1_m),
                                   return_junction=True)
    hist, hist_mask = inp.hist, inp.hist_mask
    params = embedding = None
    recurrent = {}
    lane_age = None
    rows = []
    for i in range(targets.n_steps):
        progress = getattr(inp, "replay_progress", None)
        if progress is not None and i % 500 == 0:
            progress(i, targets.n_steps)
        kin, present = inp.states[:, i], inp.present[:, i]
        continuity = getattr(inp, "continuity", None)
        fresh = None if i == 0 or continuity is None else ~continuity[:, i - 1].bool()
        if fresh is not None:
            recurrent = {key: torch.where(fresh.reshape(*fresh.shape, *([1] * (val.ndim - 2))),
                                          torch.zeros_like(val), val)
                         for key, val in recurrent.items()}
            hist_mask = hist_mask & ~fresh.unsqueeze(-1)
            gap, _ = lead_neighbour(kin, present, model.physics)
            new_hist, new_mask = initial_history(history_step_features(kin, gap), present,
                                                 model.history_len)
            hist = torch.where(fresh[..., None, None], new_hist, hist)
            hist_mask = torch.where(fresh[..., None], new_mask, hist_mask)
            if lane_age is not None:
                lane_age = torch.where(fresh, torch.zeros_like(lane_age), lane_age)
        state = {**recurrent, **state_from_kin(kin, present), "is_cav": inp.is_cav[:, i]}
        if hist is None:
            gap, _ = lead_neighbour(kin, present, model.physics)
            hist, hist_mask = initial_history(history_step_features(kin, gap), present,
                                              model.history_len)
        if lane_age is None and hasattr(model, "_initial_lane_age"):
            lane_age = model._initial_lane_age(state, present, kin, inp.observed_hist,
                                               inp.observed_hist_mask)
        if fresh is not None and bool((fresh & present).any()) and hasattr(model, "encode_drivers"):
            observed, observed_mask = hist, hist_mask
            if getattr(model, "long_history", False):
                from wms.data.history import long_history_features

                features = [long_history_features(
                    inp.states[b, :i + 1], inp.present[b, :i + 1], continuity[b, :i], present[b],
                    history_s=model.history_s, history_stride_s=model.history_stride_s,
                    dt_s=model.physics.dt_s) for b in range(len(kin))]
                observed = torch.stack([x[0] for x in features])
                observed_mask = torch.stack([x[1] for x in features])
            _, new_embedding, new_params = model.encode_drivers(
                state, history=hist, history_mask=hist_mask,
                driver_history=observed, driver_history_mask=observed_mask)
            if params is not None and new_params is not None:
                params = {key: torch.where(fresh, new_params[key], val) for key, val in params.items()}
            if embedding is not None and new_embedding is not None:
                embedding = torch.where(fresh.unsqueeze(-1), new_embedding, embedding)
            if controller is not None:
                for b, k in torch.nonzero(fresh).tolist():
                    vid = f"{b}:{k}"
                    controller._invalidate(vid)
                    controller.active.pop(vid, None)
                    controller.damping.pop(vid, None)
        action_id = action_command = None
        if inp.action_timeline is not None:
            action_id, action_command = cav_motion.timeline_controls(
                inp.action_timeline, i, model.physics.dt_s,
                cav_motion.timeline_period(inp.action_timeline), present, kin.dtype)
        out = model.forward(
            state, k_steps=i, history=hist, history_mask=hist_mask,
            driver_history=inp.observed_hist if i == 0 else None,
            driver_history_mask=inp.observed_hist_mask if i == 0 else None,
            driver_params=params, driver_embedding=embedding,
            lane_age=lane_age, lane_v5_acc=recurrent.get("lane_v5_acc"),
            lane_blocked=recurrent.get("lane_blocked"),
            controller_id=inp.controller_id if action_id is None else action_id,
            action_command=action_command)
        if params is None and "driver_parameters" in out:
            from wms.model.driver_encoder import PARAMETER_NAMES
            params = {name: out["driver_parameters"][..., j] for j, name in enumerate(PARAMETER_NAMES)}
        if embedding is None:
            embedding = out.get("driver_embedding")
        accel = out["accel_mps2"]
        moved = advance_longitudinal(kin, accel, float(model.physics.dt_s),
                                     bool(model.physics.stop_time_integration))
        cav = present & inp.is_cav[:, i].bool()
        direct = torch.zeros_like(cav)
        exec_cap = (model._exec_cap(out, state, params)
                    if bool(cav.any()) and getattr(model, "cav_exec_cap", "off") != "off" else None)
        if action_id is not None and getattr(model, "apply_direct_commands", False):
            direct = (action_id == DIRECT_ID) & cav
            if bool(direct.any()):
                direct, moved[..., 0], moved[..., 4] = model._apply_direct_commands(
                    out, direct, kin[..., 0], kin[..., 4], moved[..., 0], moved[..., 4],
                    action_command, exec_cap=exec_cap)
        fixed_next_lane = state["lane"]
        if controller is not None:
            ns, nv, fixed_next_lane = model._apply_cav_overrides(
                out, state, cav, direct, present, i, action_id, action_command, None,
                controller, moved[..., 0], moved[..., 4], state["lane"], exec_cap=exec_cap)
            moved[..., 0], moved[..., 4] = ns, nv
        # Without commands, use the frozen native longitudinal branch for CAVs.
        # This fallback is explicitly counted, never replaced by recorded future motion.
        moved[..., 5] = out["accel_mps2"]
        if getattr(model.physics, "ramp_lane_end_stop", False):
            # Same physical lane-end stop as WmSModel.step / pivot_decision_geometry; applying
            # it here lets the replay observers and the fitter see the pinned motion.
            moved[..., 0], moved[..., 4], _ = ramp_lane_end_stop(
                kin, moved[..., 0], moved[..., 4], present, model.physics, shifts[:, i])
        if hasattr(model, "pivot_decision_geometry"):
            geo = model.pivot_decision_geometry(state, out["accel_mps2"],
                                                 next_s=moved[..., 0], next_v=moved[..., 4],
                                                 fixed_next_lane=fixed_next_lane,
                                                 route_shift=shifts[:, i])
        else:
            geo = geometry(moved, present, model.physics, route_shift=shifts[:, i])
        observer = getattr(inp, "replay_observer", None)
        if observer is not None:
            observer(i, state, out, moved, geo,
                     dict(driver_params=params, lane_age=lane_age,
                          fixed_next_lane=fixed_next_lane,
                          lane_v5_acc=recurrent.get("lane_v5_acc"),
                          driver_history=inp.observed_hist if i == 0 else None,
                          driver_history_mask=inp.observed_hist_mask if i == 0 else None))
        geo = dict(geo)
        geo.update(_successor_labels(inp, shifts, junction, i))
        rows.append({key: value.detach() for key, value in geo.items()})
        for key in ("temporal_residual_mps2", "temporal_residual_present", "lane_v5_acc", "lane_blocked"):
            if key in out:
                recurrent[key] = out[key].detach()
        if i + 1 < targets.n_steps:
            next_kin, next_present = inp.states[:, i + 1], inp.present[:, i + 1]
            gap, _ = lead_neighbour(next_kin, next_present, model.physics)
            hist, hist_mask = push_history(hist, hist_mask,
                                           history_step_features(next_kin, gap), next_present)
            if lane_age is not None:
                same = ((next_kin[..., 1:4].argmax(-1) == state["lane"])
                        & present & next_present)
                lane_age = torch.where(same, lane_age + 1, torch.zeros_like(lane_age))
    return {key: torch.stack([row[key] for row in rows], 1) for key in rows[0]}


@torch.no_grad()
def build_pre_window_history(model, inp, cfg: MergePivotConfig) -> list[Optional[dict]]:
    """Frozen full-scene replay strictly BEFORE t0, with certified entry lifecycles.

    Each ragged prefix includes t0 only as a continuity/lane anchor. Its motion,
    teacher and future labels never enter prefix decision geometry. The returned
    geometry is detached; evaluate its boundary afresh for each backward chunk.
    """
    from wms.data.history import long_history_features

    prefixes = getattr(inp, "merge_pivot_prefix", None)
    if prefixes is None:
        return [None] * len(inp.states)
    if len(prefixes) != len(inp.states):
        raise ValueError("merge_pivot_prefix needs one raw prefix per batch row")
    result = []
    for b, prefix in enumerate(prefixes):
        if prefix is None:
            result.append(None)
            continue
        device = inp.states.device
        raw = {key: torch.as_tensor(prefix[key], device=device).unsqueeze(0)
               for key in ("states", "present", "is_cav", "continue")}
        rec = raw["states"].float()
        present, cav, cont = (raw[key].bool() for key in ("present", "is_cav", "continue"))
        h, k = rec.shape[1] - 1, rec.shape[2]
        if (rec.shape[-1] != 7 or present.shape != rec.shape[:-1]
                or cav.shape != present.shape or cont.shape != (1, h, k)
                or k > inp.states.shape[2]):
            raise ValueError("merge_pivot_prefix raw state/continuity shape mismatch")
        shared = present[0, -1] & inp.present[b, 0, :k]
        if not torch.equal(rec[0, -1, shared], inp.states[b, 0, :k][shared]):
            # Junction filling can change the training t0 scene. Such a prefix
            # cannot certify that scene; absent-slot packing differences are inert.
            result.append(None)
            continue
        if not h:
            result.append(None)
            continue
        timeline = prefix.get("action_timeline")
        if timeline is None and bool((present[:, :-1] & cav[:, :-1]).any()):
            # Native fallback is counted in window replay, but cannot certify
            # survival under unobserved historical CAV commands.
            result.append(None)
            continue
        if timeline is not None:
            timeline = {key: (float(val) if key == "control_dt_s" else
                              torch.as_tensor(val, device=device).unsqueeze(0))
                        for key, val in timeline.items()}
            count = (h + 1) // 2  # registered 0.5 s physics / 1 s controller clock
            fields = ("controller", "acceleration_mps2", "target_speed_mps")
            if (timeline.get("control_dt_s", 1.) != 1.
                    or any(key not in timeline or timeline[key].ndim != 3
                           or timeline[key].shape[0] != 1 or timeline[key].shape[1] < count
                           or timeline[key].shape[2] != k for key in fields)):
                result.append(None)
                continue
            live = torch.zeros_like(timeline["controller"], dtype=torch.bool)
            for i in range(h):
                live[:, i // 2] |= present[:, i] & cav[:, i]
            direct = (timeline["controller"] == 3) & live
            known_command = (torch.isfinite(timeline["acceleration_mps2"])
                             & torch.isfinite(timeline["target_speed_mps"]))
            if "command_known" in timeline:
                known_command &= timeline["command_known"].bool()
            if bool((direct & ~known_command).any()):
                result.append(None)
                continue
        observed = observed_mask = None
        if getattr(model, "long_history", False):
            observed, observed_mask = long_history_features(
                rec[0, :1], present[0, :1], cont[0, :0], present[0, 0],
                history_s=model.history_s, history_stride_s=model.history_stride_s,
                dt_s=model.physics.dt_s)
            observed, observed_mask = observed.unsqueeze(0), observed_mask.unsqueeze(0)
        source = SimpleNamespace(states=rec, present=present, is_cav=cav, continuity=cont,
                                 hist=None, hist_mask=None, observed_hist=observed,
                                 observed_hist_mask=observed_mask, action_timeline=timeline,
                                 controller_id=None)
        geo = build_recorded_geometry(model, source, SimpleNamespace(n_steps=h))
        result.append(certify_pre_window(rec, present, cont, geo, cfg))
    return result


def certify_pre_window(rec, present, cont, geo, cfg):
    """Certify a causal prefix; also usable with a once-replayed audit episode.

    The last recorded row is the t0 anchor. Geometry ends strictly before it.
    """
    h = rec.shape[1] - 1
    lane = rec[..., 1:4].argmax(-1)
    lane_valid = rec[..., 1:4].sum(-1) > 0
    pre_risk = (present & lane_valid & (lane == cfg.merge_lane)
                & (rec[..., 0] >= cfg.zone_s0_m) & (rec[..., 0] <= cfg.zone_s1_m))
    risk = geo["ramp"] | pre_risk[:, :-1]
    event = (present[:, :-1] & present[:, 1:] & cont
             & (lane[:, :-1] == cfg.merge_lane) & (lane[:, 1:] == cfg.target_lane))
    scene_ok = ((~present | (torch.isfinite(rec).all(-1) & lane_valid)).all(-1))
    seen = torch.zeros_like(present[:, 0])
    known, terminal = seen.clone(), seen.clone()
    masks = []
    for i in range(h):
        same = torch.zeros_like(seen) if i == 0 else (
            cont[:, i - 1] & present[:, i - 1] & present[:, i])
        seen, known, terminal = (val & same for val in (seen, known, terminal))
        born = risk[:, i] & ~seen & ~terminal
        outside = ~pre_risk[:, i]
        if i:
            outside = outside | (same & ~pre_risk[:, i - 1] & lane_valid[:, i - 1])
        known = torch.where(born, outside, known)
        seen = seen | born
        terminal = terminal | (seen & ~risk[:, i])
        known = known & (~risk[:, i] | scene_ok[:, i, None])
        masks.append(seen & ~terminal & risk[:, i] & geo["eligible"][:, i])
        terminal = terminal | event[:, i]
    # Entry can occur in the recorded t0 row even when the final frozen
    # pre-window substep did not yet reach risk. Its prior max is then -inf.
    fresh_entry = (~seen & ~terminal & ~pre_risk[:, -2] & lane_valid[:, -2]
                   & scene_ok[:, -2, None])
    certified = (((seen & known & ~terminal) | fresh_entry) & cont[:, -1] & present[:, -1]
                 & present[:, -2] & pre_risk[:, -1] & scene_ok[:, -1, None])
    # Earlier occupants of a reused slot must never supply this identity's max.
    same_to_t0 = present[:, -1].clone()
    for i in range(h - 1, -1, -1):
        same_to_t0 = same_to_t0 & cont[:, i] & present[:, i]
        masks[i] = masks[i] & same_to_t0
    return dict(geometry=geo, history_mask=torch.stack(masks, 1),
                certified=certified, n_steps=h)


def pre_window_maximum(model, histories: list, like: Tensor) -> Tensor:
    """Differentiable exact eligible maximum; NaN means uncertified/unknown age."""
    rows = []
    for b, history in enumerate(histories):
        row = torch.full_like(like[b:b + 1], torch.nan)
        if history is not None and bool(history["certified"].any()):
            slots = torch.nonzero(history["certified"][0]).flatten()
            geo = {key: val.index_select(-1, slots) for key, val in history["geometry"].items()}
            c = model.merge_pivot(geo)
            maximum = torch.where(history["history_mask"].index_select(-1, slots), c,
                                  torch.full_like(c, -torch.inf)).amax(1)
            row = row.index_copy(1, slots, maximum)
        rows.append(row)
    return torch.cat(rows)


def pre_window_hybrid_history(model, histories: list, like: Tensor) -> Tensor:
    """Rebuild the full certified prefix graph; a maximum is insufficient for F1.

    NaN marks uncertified identities; -inf pads only non-decisions. This keeps
    gradients through the posterior denominator for all boundary/scale parameters.
    """
    size = max(1, max((h["n_steps"] for h in histories if h is not None), default=0))
    rows = []
    for b, history in enumerate(histories):
        row = like.new_full((*like[b:b + 1].shape, size), torch.nan)
        if history is not None:
            c = model.merge_pivot(history["geometry"])
            path = torch.where(history["history_mask"], c, torch.full_like(c, -torch.inf))
            path = torch.nn.functional.pad(path.transpose(1, 2), (0, size - c.shape[1]),
                                           value=-torch.inf)
            width = path.shape[1]
            row[:, :width] = torch.where(history["certified"].unsqueeze(-1), path,
                                         row[:, :width])
        rows.append(row)
    return torch.cat(rows)


class MergePivotTerm(LossTerm):
    name = "merge_pivot"

    def __init__(self, cfg: MergePivotConfig):
        self.cfg = cfg
        self.physics = SimpleNamespace(merge_lane=cfg.merge_lane, merge_target_lane=cfg.target_lane,
                                       merge_zone_s0_m=cfg.zone_s0_m, merge_zone_s1_m=cfg.zone_s1_m)

    def prepare(self, targets: UnrollTargets) -> None:
        source = targets.prepared.get("merge_pivot_source")
        if source is None:
            raise RuntimeError("merge_pivot needs model/inputs injected before composer.prepare")
        model, inp = source
        devices = [targets.states.device.index or 0] if targets.states.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            geo = build_recorded_geometry(model, inp, targets)
        # Prefix RNG cannot depend on the number of in-window forward calls.
        with torch.random.fork_rng(devices=devices):
            pre_window = build_pre_window_history(model, inp, self.cfg)
        n = targets.n_steps
        rec, present = targets.states[:, :n + 1], inp.present[:, :n + 1].bool()
        lane = rec[..., 1:4].argmax(-1)
        pre_risk = (present & (rec[..., 1:4].sum(-1) > 0) & (lane == self.cfg.merge_lane)
                    & (rec[..., 0] >= self.cfg.zone_s0_m) & (rec[..., 0] <= self.cfg.zone_s1_m))
        event = (present[:, :-1] & present[:, 1:] & (lane[:, :-1] == self.cfg.merge_lane)
                 & (lane[:, 1:] == self.cfg.target_lane))
        own = targets.loss_valid[:, :n].bool() & ~targets.is_cav[:, :n].bool()
        risk = (geo["ramp"] | pre_risk[:, :n] | event) & own
        reset = torch.cat((torch.zeros_like(present[:, :1]),
                           present[:, 1:n] & ~present[:, :n - 1]), 1)
        # Recorded merges that are vetoed by P1 although SUMO executed them legally
        # (same-step sequential lane change of a leader, internal-junction occupant) are
        # excluded WITH their whole risk sequence; every other veto stays a zero-mass
        # failure.  The identical classification feeds the fitter and the support audit.
        from .merge_pivot_support import classify_vetoed_events, sequence_rows
        verdict = classify_vetoed_events(geo, event & own, target_lane=self.cfg.target_lane)
        excluded = verdict["explained"]
        support_counts = dict(vetoed_events=int(verdict["vetoed"].sum()),
                              excluded_events=int(excluded.sum()),
                              unexplained_vetoed_events=int((verdict["vetoed"] & ~excluded).sum()),
                              wm_motion_vetoed_events=int(verdict["recorded_clear"].sum()))
        if support_counts["excluded_events"]:
            dropped = sequence_rows(excluded, risk, reset)
            risk, event = risk & ~dropped, event & ~excluded
        # Complete entry only if observed from before zone entry; a presence birth
        # directly into risk is unknown age. IDs are neither needed nor consumed.
        prior_outside = torch.cat((~pre_risk[:, :1],
                                   present[:, :n - 1] & ~pre_risk[:, :n - 1]), 1)
        known = ~pre_risk[:, :n] | prior_outside
        seen = torch.zeros_like(risk[:, 0])
        terminal = torch.zeros_like(seen)
        birth_rows = []
        history_rows = []
        for i in range(n):
            seen = seen & ~reset[:, i]
            terminal = terminal & ~reset[:, i]
            terminal = terminal | (seen & ~risk[:, i])
            birth_rows.append(risk[:, i] & ~seen)
            history_rows.append(risk[:, i] & ~terminal)
            seen = seen | risk[:, i]
            terminal = terminal | event[:, i]
        births = torch.stack(birth_rows, 1)
        history_mask = torch.stack(history_rows, 1)
        steps = torch.arange(n, device=rec.device).view(1, n, 1).expand_as(reset)
        last_reset = torch.cummax(torch.where(reset, steps, torch.full_like(steps, -1)), 1).values
        unknown = births & ~known
        unknown_before = unknown.clone()
        recovered = torch.zeros_like(unknown)
        for b, history in enumerate(pre_window):
            if history is not None:
                k = history["certified"].shape[-1]
                recovered[b, 0, :k] = unknown[b, 0, :k] & history["certified"][0]
        unknown = unknown & ~recovered
        known_before = known
        known = known | recovered
        with torch.no_grad():
            entry_snapshot = pre_window_maximum(model, pre_window, rec[:, 0, :, 0])
        competing = risk & ~event & (~present[:, 1:] | (lane[:, 1:] != self.cfg.merge_lane))
        targets.prepared[self.name] = dict(
            geometry=geo, risk=risk, event=event & own, reset=reset, known=known,
            support_counts=support_counts,
            known_before=known_before, pre_window_initial_maximum=entry_snapshot,
            n_cars=int(births.sum()), model=model, tf_carry=None, on_carry=None,
            stats={}, causal_cav_fallback=int((present[:, :n] & inp.is_cav[:, :n]).sum())
            if inp.action_timeline is None else 0,
            unknown_age_sequences=torch.nonzero(unknown).cpu().tolist(),
            unknown_age_sequences_before=torch.nonzero(unknown_before).cpu().tolist(),
            pre_window_recovered_sequences=torch.nonzero(recovered).cpu().tolist(),
            pre_window_history=pre_window,
            competing_exit=competing & present[:, 1:], vanished=competing & ~present[:, 1:],
            history_mask=history_mask, last_reset=last_reset)

    def __call__(self, ctx: LossContext) -> TermValue:
        cfg, prep = self.cfg, ctx.targets.prepared[self.name]
        model, c0, c1 = prep["model"], ctx.c0, ctx.c1
        params = model.merge_pivot.params()
        tau = params["tau"]
        width = params.get("w")
        value = tau * 0.0
        stats = {}
        if c0 == 0:
            stats.update({"support_" + key: val for key, val in prep["support_counts"].items()})

        def cut(x):
            return x[:, c0:c1]
        entry_max = pre_window_maximum(model, prep["pre_window_history"], prep["risk"][:, 0].to(tau))
        explicit = ctx.targets.prepared.get("merge_pivot_initial_maximum")
        if explicit is not None:
            entry_max = torch.where(~torch.isnan(explicit), explicit.to(entry_max), entry_max)
        initial = entry_max if c0 == 0 and width is None else None
        entry_history = None
        if width is not None:
            entry_history = ctx.targets.prepared.get("merge_pivot_initial_history")
            if entry_history is None:
                entry_history = pre_window_hybrid_history(
                    model, prep["pre_window_history"], prep["risk"][:, 0].to(tau))
        initial_history = entry_history if c0 == 0 else None
        stats["recorded_events"] = int(cut(prep["event"]).sum())
        stats["recorded_physical_fail_events"] = int((cut(prep["event"])
                                                        & ~cut(prep["geometry"]["eligible"])).sum())
        stats["recorded_geometric_fail_events"] = int((cut(prep["event"])
            & ~cut(prep["geometry"]["physical_eligible"])).sum())
        stats["recorded_reservation_fail_events"] = int((cut(prep["event"])
            & cut(prep["geometry"]["reservation_veto"])).sum())
        stats["recorded_competing_exits"] = int(cut(prep["competing_exit"]).sum())
        stats["recorded_vanished_exits"] = int(cut(prep["vanished"]).sum())
        if cfg.tf_weight or cfg.metrics:
            geo = {key: cut(x) for key, x in prep["geometry"].items()}
            c = model.merge_pivot(geo)
            tf_carry = prep["tf_carry"]
            if c0 and tf_carry is not None:
                # Frozen geometry makes prefix boundary replay cheap: rebuild its
                # exact max graph instead of retaining/reusing a graph already
                # freed by the previous chunk's backward. This preserves analytic
                # gradients of ALL seven parameters, not just forward NLL parity.
                prefix_geo = {key: x[:, :c0] for key, x in prep["geometry"].items()}
                prefix_c = model.merge_pivot(prefix_geo)
                step = torch.arange(c0, device=c.device).view(1, c0, 1)
                keep = (prep["history_mask"][:, :c0] & prefix_geo["eligible"]
                        & (step >= prep["last_reset"][:, c0 - 1].unsqueeze(1)))
                maximum = torch.where(keep, prefix_c, torch.full_like(prefix_c, -torch.inf)).amax(1)
                valid_entry = ~torch.isnan(entry_max) & (prep["last_reset"][:, c0 - 1] < 0)
                maximum = torch.maximum(maximum, torch.where(
                    valid_entry, entry_max.to(maximum), torch.full_like(maximum, -torch.inf)))
                tf_carry = {**tf_carry, "maximum": maximum}
                if width is not None:
                    history = torch.where(keep, prefix_c, torch.full_like(prefix_c, -torch.inf)).transpose(1, 2)
                    if entry_history is not None:
                        valid_entry = prep["last_reset"][:, c0 - 1] < 0
                        entry_history = torch.where(valid_entry.unsqueeze(-1) & ~torch.isnan(entry_history),
                                                    entry_history.to(history),
                                                    torch.full_like(entry_history.to(history), -torch.inf))
                        history = torch.cat((entry_history, history), -1)
                    tf_carry = {**tf_carry, "history": history}
            nll, prep["tf_carry"], counts = score_chunk(
                c, tau, cut(prep["risk"]), cut(prep["event"]),
                known_entry=cut(prep["known"]), reset=cut(prep["reset"]),
                eligible=geo["eligible"], carry=tf_carry, initial_maximum=initial,
                w=width, quadrature_order=cfg.quadrature_order, initial_history=initial_history)
            stats.update({"tf_" + key: val for key, val in counts.items()}, tf_nll=float(nll.detach()))
            if cfg.tf_weight:
                value = value + cfg.tf_weight * nll / max(prep["n_cars"], 1) / cfg.z_tf
        if cfg.weight or cfg.metrics:
            for key in ("merge_pivot_c", "merge_pivot_eligible",
                         *("merge_pivot_" + name for name in params)):
                if key not in ctx.pred:
                    raise RuntimeError(f"first_passage rollout must emit {key}")
            c = ctx.pred["merge_pivot_c"]
            k = prep["risk"].shape[-1]
            c = c[..., :k]
            support, early = self._comparison(ctx, prep)
            on_carry = prep["on_carry"]
            if c0 and width is not None and entry_history is not None and on_carry is not None:
                # Reattach only the frozen pre-window graph. Rollout history is
                # a TBPTT carry and remains detached; slot resets erase its prior.
                valid = prep["last_reset"][:, c0 - 1] < 0
                prefix = torch.where(valid.unsqueeze(-1) & ~torch.isnan(entry_history),
                                     entry_history, torch.full_like(entry_history, -torch.inf))
                on_carry = {**on_carry, "history": torch.cat(
                    (prefix, on_carry["history"][..., entry_history.shape[-1]:]), -1)}
            nll, prep["on_carry"], counts = score_chunk(
                c, ctx.pred["merge_pivot_tau"].mean(), cut(prep["risk"]), cut(prep["event"]),
                known_entry=cut(prep["known"]), reset=cut(prep["reset"]),
                eligible=ctx.pred["merge_pivot_eligible"][..., :k].bool(), support=support,
                stop=early, carry=on_carry, initial_maximum=initial,
                w=ctx.pred["merge_pivot_w"].mean() if width is not None else None,
                quadrature_order=cfg.quadrature_order, initial_history=initial_history)
            stats.update({"on_" + key: val for key, val in counts.items()}, on_nll=float(nll.detach()))
            if cfg.weight:
                value = value + cfg.weight * nll / max(prep["n_cars"], 1) / cfg.z_on
        if cfg.tau_reg_weight:
            value = value + cfg.tau_reg_weight * tau.square() * (c1 - c0) / ctx.targets.n_steps
        if cfg.w_reg_weight:
            if width is None:
                raise ValueError("w regularization requires the hybrid law")
            value = value + cfg.w_reg_weight * width.square() * (c1 - c0) / ctx.targets.n_steps
        for name in params:
            emitted = ctx.pred.get("merge_pivot_" + name)
            stats[name] = float((emitted.mean() if torch.is_tensor(emitted) else params[name]).detach())
        prep["stats"][c0] = stats
        return TermValue(value)

    def _comparison(self, ctx, prep):
        """Detached full-scene comparison at the causal decision geometry."""
        from .merge_first_passage import _kin

        with torch.no_grad():
            k = prep["risk"].shape[-1]
            pred = ctx.pred
            def pre(key, current):
                return torch.cat((ctx.current[current][..., :k].unsqueeze(1), pred[key][:, :-1, :k]), 1)
            lane, present = pre("lanes", "lane"), pre("masks", "present").bool()
            kin = _kin(pre("positions", "s_m"), lane, pre("speeds", "v_mps"),
                       pre("accel", "a_mps2"), ctx.targets.states[:, ctx.c0:ctx.c1, :, 6])
            moved = advance_longitudinal(kin, pred["accel"][..., :k], ctx.targets.dt_s,
                                         bool(getattr(prep["model"].physics, "stop_time_integration", False)))
            # CAV outputs contain executed controller motion. HDV geometry uses
            # the longitudinal substep BEFORE later projection/lane-end clamps.
            # In both cases retain PRE-change lane membership for neighbors.
            cav = ctx.targets.is_cav[:, ctx.c0:ctx.c1, :k].bool()
            moved[..., 0] = torch.where(cav, pred["positions"][..., :k], moved[..., 0])
            moved[..., 4] = torch.where(cav, pred["speeds"][..., :k], moved[..., 4])
            if "merge_pivot_geometry_gap_f" in pred:
                # Captured before commits, using actual rollout lengths and ALL
                # present neighbours (including entrant padding), not target metadata.
                p = {key: pred["merge_pivot_geometry_" + key][..., :k]
                     for key in ("gap_f", "gap_l", "v_f", "v_l", "has_f", "has_l")}
            elif hasattr(prep["model"], "pivot_decision_geometry"):
                from ..inputs import state_from_kin

                b, t = kin.shape[:2]
                state = state_from_kin(kin.reshape(b * t, k, 7), present.reshape(b * t, k))
                state["is_cav"] = cav.reshape(b * t, k)
                flat = prep["model"].pivot_decision_geometry(
                    state, pred["accel"][..., :k].reshape(b * t, k),
                    next_s=moved[..., 0].reshape(b * t, k), next_v=moved[..., 4].reshape(b * t, k),
                    fixed_next_lane=pred["lanes"][..., :k].reshape(b * t, k))
                p = {key: val.reshape(b, t, k) for key, val in flat.items()}
            else:
                p = geometry(moved, present, self.physics)
            r = {key: val[:, ctx.c0:ctx.c1] for key, val in prep["geometry"].items()}
            ok = ((kin[..., 0] - ctx.targets.states[:, ctx.c0:ctx.c1, :, 0]).abs()
                  <= self.cfg.pos_tol_m)
            ok &= (kin[..., 4] - ctx.targets.states[:, ctx.c0:ctx.c1, :, 4]).abs() <= self.cfg.speed_tol_mps
            ok &= present & (lane == self.cfg.merge_lane)
            entrant = pred.get("entrant_seq")
            if torch.is_tensor(entrant):
                ok &= ~entrant[..., :k].bool()
            for side in ("f", "l"):
                ok &= p[f"has_{side}"] == r[f"has_{side}"]
                both = p[f"has_{side}"] & r[f"has_{side}"]
                ok &= ~both | ((p[f"gap_{side}"] - r[f"gap_{side}"]).abs() <= self.cfg.gap_tol_m)
                ok &= ~both | ((p[f"v_{side}"] - r[f"v_{side}"]).abs() <= self.cfg.gap_speed_tol_mps)
            early = ((pred["lanes"][..., :k] != self.cfg.merge_lane)
                     | ~pred["masks"][..., :k].bool()) & ~prep["event"][:, ctx.c0:ctx.c1]
            return ok, early


def merge_pivot_metrics(prep: Optional[dict], dt_s: float) -> dict:
    if not prep:
        return {}
    rows = list(prep["stats"].values())
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    out = {"merge_pivot_" + key: sum(row.get(key, 0) for row in rows)
           for key in keys if key not in (*PARAM_NAMES, "w")}
    for name in (*PARAM_NAMES, *(("w",) if "w" in keys else ())):
        out["merge_pivot_" + name] = rows[-1][name] if rows else None
    out["merge_pivot_n_cars"] = prep["n_cars"]
    out["merge_pivot_causal_cav_fallback_intervals"] = prep["causal_cav_fallback"]
    out["merge_pivot_unknown_age_sequences"] = prep["unknown_age_sequences"]
    out["merge_pivot_unknown_age_sequences_before"] = prep["unknown_age_sequences_before"]
    out["merge_pivot_pre_window_recovered_sequences"] = prep["pre_window_recovered_sequences"]
    out["merge_pivot_unknown_age_before"] = len(prep["unknown_age_sequences_before"])
    out["merge_pivot_unknown_age_after"] = len(prep["unknown_age_sequences"])
    out["merge_pivot_pre_window_recovered"] = len(prep["pre_window_recovered_sequences"])
    out["merge_pivot_pre_window_replayed_intervals"] = sum(
        h["n_steps"] for h in prep["pre_window_history"] if h is not None)
    for path in ("tf", "on"):
        out[f"merge_pivot_{path}_scored_seconds"] = out.get(f"merge_pivot_{path}_scored_intervals", 0) * dt_s
    return out


def merge_pivot_terms(cfg: MergePivotConfig) -> list[MergePivotTerm]:
    return [MergePivotTerm(cfg)] if cfg.enabled else []
