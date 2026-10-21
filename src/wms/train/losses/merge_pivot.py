"""P1--P3 exact persistent-logistic likelihood; no hazard or quadrature path.

Recorded geometry is a frozen causal longitudinal replay. Only the seven boundary
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
                                  log_event_mass, log_survival)

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
        history = initial_history.to(c)
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
    if w is not None:
        previous_logS = (hybrid_log_masses(history, tau, w, quadrature_order)[1][..., -1]
                         if history.shape[-1] else torch.zeros_like(c[:, 0], dtype=torch.float64))
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
            is_known = is_known | ~torch.isnan(initial_maximum)
        if i == 0 and initial_history is not None and not carry:
            is_known = is_known | ~torch.isnan(initial_history).any(-1)
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
            le, ls = hybrid_log_masses(history, tau, w, quadrature_order)
            next_logS = ls[..., -1]
        # No subtraction of -inf - -inf on impossible paths, even in masked branches.
        safe_den = torch.where(torch.isfinite(previous_logS), previous_logS,
                               torch.zeros_like(previous_logS))
        logp_event = log_event_mass(maximum, updated, tau) - safe_den
        if w is not None:
            logp_event = le[..., -1] - safe_den
        logp_wait = next_logS - safe_den
        logp = torch.where(ev, logp_event, logp_wait)
        nll = nll - torch.where(used, logp, torch.zeros_like(logp)).sum()
        stats["events"] += int((ev & known).sum())
        stats["positive_mass_events"] += int((ev & known & torch.isfinite(logp_event)).sum())
        stats["zero_mass_events"] += int((ev & known & torch.isneginf(logp_event)).sum())
        impossible_wait = used & ~ev & torch.isneginf(next_logS)
        stats["zero_survival"] += int(impossible_wait.sum())
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


@torch.no_grad()
def build_recorded_geometry(model, inp, targets: UnrollTargets) -> dict[str, Tensor]:
    """Frozen forward from each RECORDED PRESTATE, followed by shared causal motion.

    No next-row position/speed/acceleration or teacher trajectory is consumed. A
    separate controller instance replays available current commands, leaving the
    on-policy controller untouched. All present scene vehicles, including CAVs,
    remain neighbors; supervision masks are applied only by the likelihood.
    """
    from wms.model import cav as cav_motion
    from wms.model.controllers import DIRECT_ID
    from wms.model.history import history_step_features, initial_history, push_history
    from wms.physics.neighbours import lead_neighbour
    from ..cav_drive import TimelineControllers
    from ..inputs import state_from_kin

    controller = (TimelineControllers(model, inp.action_timeline)
                  if inp.action_timeline is not None else None)
    hist, hist_mask = inp.hist, inp.hist_mask
    params = embedding = None
    recurrent = {}
    lane_age = None
    rows = []
    for i in range(targets.n_steps):
        kin, present = inp.states[:, i], inp.present[:, i]
        state = {**recurrent, **state_from_kin(kin, present), "is_cav": inp.is_cav[:, i]}
        if hist is None:
            gap, _ = lead_neighbour(kin, present, model.physics)
            hist, hist_mask = initial_history(history_step_features(kin, gap), present,
                                              model.history_len)
        if lane_age is None and hasattr(model, "_initial_lane_age"):
            lane_age = model._initial_lane_age(state, present, kin, inp.observed_hist,
                                               inp.observed_hist_mask)
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
        if hasattr(model, "pivot_decision_geometry"):
            geo = model.pivot_decision_geometry(state, out["accel_mps2"],
                                                 next_s=moved[..., 0], next_v=moved[..., 4],
                                                 fixed_next_lane=fixed_next_lane)
        else:
            geo = geometry(moved, present, model.physics)
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
        competing = risk & ~event & (~present[:, 1:] | (lane[:, 1:] != self.cfg.merge_lane))
        targets.prepared[self.name] = dict(
            geometry=geo, risk=risk, event=event & own, reset=reset, known=known,
            n_cars=int(births.sum()), model=model, tf_carry=None, on_carry=None,
            stats={}, causal_cav_fallback=int((present[:, :n] & inp.is_cav[:, :n]).sum())
            if inp.action_timeline is None else 0,
            unknown_age_sequences=torch.nonzero(unknown).cpu().tolist(),
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
        def cut(x):
            return x[:, c0:c1]
        initial = ctx.targets.prepared.get("merge_pivot_initial_maximum") if c0 == 0 else None
        initial_history = ctx.targets.prepared.get("merge_pivot_initial_history") if c0 == 0 else None
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
                entry_max = ctx.targets.prepared.get("merge_pivot_initial_maximum")
                if entry_max is not None:
                    valid_entry = ~torch.isnan(entry_max) & (prep["last_reset"][:, c0 - 1] < 0)
                    maximum = torch.maximum(maximum, torch.where(
                        valid_entry, entry_max.to(maximum), torch.full_like(maximum, -torch.inf)))
                tf_carry = {**tf_carry, "maximum": maximum}
                if width is not None:
                    history = torch.where(keep, prefix_c, torch.full_like(prefix_c, -torch.inf)).transpose(1, 2)
                    entry_history = ctx.targets.prepared.get("merge_pivot_initial_history")
                    if entry_history is not None:
                        valid_entry = prep["last_reset"][:, c0 - 1] < 0
                        entry_history = torch.where(valid_entry.unsqueeze(-1), entry_history.to(history),
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
            nll, prep["on_carry"], counts = score_chunk(
                c, ctx.pred["merge_pivot_tau"].mean(), cut(prep["risk"]), cut(prep["event"]),
                known_entry=cut(prep["known"]), reset=cut(prep["reset"]),
                eligible=ctx.pred["merge_pivot_eligible"][..., :k].bool(), support=support,
                stop=early, carry=prep["on_carry"], initial_maximum=initial,
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
            if hasattr(prep["model"], "pivot_decision_geometry"):
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
    for path in ("tf", "on"):
        out[f"merge_pivot_{path}_scored_seconds"] = out.get(f"merge_pivot_{path}_scored_intervals", 0) * dt_s
    return out


def merge_pivot_terms(cfg: MergePivotConfig) -> list[MergePivotTerm]:
    return [MergePivotTerm(cfg)] if cfg.enabled else []
