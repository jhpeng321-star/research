"""Controller continuity for synthetic births; no simulator or truth-file inputs.

Callbacks are episode-local and see the complete scene, including all neighbours.
Only their entrant outputs replace recorded commands. Original slots retain the
recorded command contract used by the paired-loss/horizon estimand.
"""
from __future__ import annotations

import torch


class EntrantControllers:
    def __init__(self, recorded, feedback):
        self.recorded, self.feedback = recorded, feedback
        self.last_recorded = None
        self.last_intent = None
        self.step_index = 0

    def __call__(self, state):
        from wms.model.cav import WM_MERGE_KEY
        old = self.recorded(state)
        old_merge = state.get(WM_MERGE_KEY, torch.zeros_like(state["present"]))
        # Ordinary policy observations of t0 neighbours must report executed, not
        # discarded counterfactual feedback commands. Entrants retain bridge memory.
        mask = state.get("entrant", torch.zeros_like(state["present"]))
        observed = dict(state)
        observed["external_command_mask"] = ~mask.bool()
        observed["cmd_accel_mps2"] = (torch.zeros_like(state["s_m"])
                                     if self.last_recorded is None else self.last_recorded)
        observed["lane_intent"] = (torch.zeros_like(state["s_m"])
                                   if self.last_intent is None else self.last_intent)
        new = self.feedback(observed)
        # Match the ordinary bridge's one-second communication clock. Discarded
        # feedback lane requests must never become reports by recorded t0 cars.
        if self.step_index % 2 == 0:
            self.last_recorded = old[..., 3].detach().clone()
            self.last_intent = (old[..., 1]-state["lane"]).detach().clone() if "lane" in state else torch.zeros_like(state["s_m"])
        self.step_index += 1
        mask = mask.bool() & state["present"].bool() & state["is_cav"].bool()
        new_merge = observed.get(WM_MERGE_KEY, torch.zeros_like(mask))
        state[WM_MERGE_KEY] = (old_merge & ~mask) | (new_merge & mask)
        return torch.where(mask[..., None], new, old)

    def native_lane_mask(self, state):
        mask = state.get("entrant", torch.zeros_like(state["present"])).bool()
        fn = getattr(self.feedback, "native_lane_mask", None)
        return mask & (fn(state) if fn else torch.zeros_like(mask))


class RowControllers:
    """Per-arm callbacks with stable [row, slot] identity and full-scene observations."""
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def __call__(self, state):
        # Each callback sees its own row; no controller observes a counterfactual twin.
        batch = state["s_m"].shape[0]
        def row(i):
            return {k: (v[i:i+1] if torch.is_tensor(v) and v.ndim and v.shape[0] == batch else v)
                    for k, v in state.items()}
        outputs = []
        delegated = []
        from wms.model.cav import WM_MERGE_KEY
        for i, callback in enumerate(self.callbacks):
            st = row(i)
            outputs.append(callback(st))
            delegated.append(st.get(WM_MERGE_KEY, torch.zeros_like(st["present"])))
        state[WM_MERGE_KEY] = torch.cat(delegated)
        return torch.cat(outputs)

    def native_lane_mask(self, state):
        batch = state["s_m"].shape[0]
        masks = []
        for i, callback in enumerate(self.callbacks):
            st = {k: (v[i:i+1] if torch.is_tensor(v) and v.ndim and v.shape[0] == batch else v)
                  for k, v in state.items()}
            fn = getattr(callback, "native_lane_mask", None)
            masks.append(fn(st) if fn else torch.zeros_like(st["present"]))
        return torch.cat(masks)


def policy_feedback(checkpoint, *, device="cpu", native_cav="head",
                    native_accel_max=1.2, native_ramp_merge="wm_law",
                    native_lane_change=True, symmetric_exec_cap=True):
    """Frozen policy/MAPPO adapter using the exact ordinary CAV observation builder.

    ``load_policy_components`` supports the published MAPPO checkpoint interface.
    The caller binds checkpoint hashes and plant settings in its scoring receipt.
    """
    from .policy import load_policy_components
    from .bridge import make_cav_controller
    from .counterfactual import native_fn
    from .safety import safe_accel
    policy, encoder, rules, _ = load_policy_components(str(checkpoint), device=device)
    policy.eval(); encoder.eval()
    def deterministic(obs, signal, **kwargs):
        return policy(obs, signal, deterministic=True, **kwargs)
    deterministic.accel_max = policy.accel_max
    deterministic.residual_on_native = bool(getattr(policy, "residual_on_native", False))
    deterministic.residual_reference = getattr(policy, "residual_reference", "base")
    return make_cav_controller(
        deterministic, encoder, safe_accel, lane_rules=rules,
        native=native_fn(native_cav, a_max=native_accel_max),
        native_accel_max=native_accel_max, native_ramp_merge=native_ramp_merge,
        native_lane_change=native_lane_change, symmetric_exec_cap=symmetric_exec_cap,
        mask_forced_merge=True)


def inferred_timeline(timeline, original_slots):
    """Infer only unambiguous algorithm plans; direct actions require a callback.

    A direct command is not a controller algorithm and cannot be transferred to a
    new identity. In particular, never copy an arbitrary t0 CAV's acceleration.
    """
    from wms.data.action_timeline import CONTROLLERS
    ids = timeline["controller"][..., :original_slots]
    native = CONTROLLERS.index("native")
    plan = torch.full((*ids.shape[:2], 1), native, device=ids.device, dtype=ids.dtype)
    for b in range(ids.shape[0]):
        for t in range(ids.shape[1]):
            active = ids[b, t][ids[b, t] != native].unique()
            if active.numel() > 1 or (active.numel() and int(active[0]) == CONTROLLERS.index("direct")):
                raise ValueError("entrant_control needs an explicit feedback controller for direct/mixed arms")
            if active.numel():
                plan[b, t] = active[0]
    return plan


def feedback_from_specs(specs, arm_names, model, timeline, original_slots, device, *, branches=None):
    """Explicit algorithm binding carried by scoring/training configuration."""
    if arm_names is None:
        raise ValueError("entrant controller specs require observable arm metadata")
    from wms.train.cav_drive import TimelineControllers
    callbacks = []
    for i, arm in enumerate(arm_names):
        branch = branches[i] if branches else None
        if branch and (branch.get("policy_params") or {}).get("policy") == "generic_random":
            callbacks.append(generic_random_feedback(branch, device=device))
        elif arm in specs:
            spec = dict(specs[arm])
            checkpoint = spec.pop("checkpoint")
            expected = spec.pop("sha256")
            import hashlib
            from pathlib import Path
            path = Path(checkpoint)
            bad = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth")
            if any(s in str(path.resolve()).lower() for s in bad):
                raise ValueError("prohibited entrant controller checkpoint path")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("entrant controller checkpoint hash mismatch")
            callbacks.append(policy_feedback(path, device=device, **spec))
        else:
            row = {k: (v[i:i+1] if torch.is_tensor(v) else v) for k, v in timeline.items()}
            plan = inferred_timeline(row, original_slots)
            row["controller"] = plan.expand_as(row["controller"])
            callbacks.append(TimelineControllers(model, row))
    return RowControllers(callbacks)


def generic_random_feedback(branch, *, device="cpu"):
    """Re-use the recorded undirected algorithm and its observable branch parameters.

    No simulator is imported or run. Branch reseeding resets latent/segment state;
    imagined identities have episode-local slot tracks, never reused by entries-v2.
    """
    from traffic_shepherd.c4.genrand import GenericRandomPolicy
    from .bridge import make_cav_controller
    from .lanes import LaneRules, LANE_DELTA
    from .policy import PolicyOutput
    from .safety import safe_accel
    p = branch["policy_params"]
    algorithm = GenericRandomPolicy(p["seed"], mode=p["mode"],
        low=p["accel_range_mps2"][0], high=p["accel_range_mps2"][1],
        hold_s=p["hold_s"], jerk=p["jerk_mps3"], lane_hold_s=p["lane_hold_s"],
        rho=p["rho"], n_latent=p["n_latent"], lane_rate_per_s=p["lane_rate_per_s"])
    if algorithm.params() != p:
        raise ValueError("recorded generic-random parameters disagree with source algorithm")
    algorithm.reseed(branch["arm_seed"])
    scene = [None]
    def encoder(obs, state):
        scene[0] = state
        return state["s_m"].new_zeros((*state["s_m"].shape, 8))
    encoder.requires_state = True
    def actor(obs, signal, *, lane_mask):
        state = scene[0]
        accel, lanes = algorithm(state, with_lanes=True)
        accel = torch.as_tensor(accel, dtype=state["s_m"].dtype, device=device)
        target = torch.as_tensor(lanes, device=device)
        delta = target - state["lane"]
        choices = torch.zeros_like(state["lane"])
        for index, change in enumerate(LANE_DELTA):
            if change:
                choices = torch.where((target >= 0) & (delta == change), index, choices)
        zeros = torch.zeros_like(accel)
        return PolicyOutput(accel, choices, zeros, zeros, zeros, zeros, lane_mask.sum(-1))
    actor.accel_max = algorithm.high
    return make_cav_controller(actor, encoder, safe_accel, lane_rules=LaneRules(),
                               native_ramp_merge="wm_law", mask_forced_merge=True)


def evaluation_feedback(controllers, arm, model, timeline, original_slots, steps, total_slots):
    """Resolve an evaluation arm from an explicit callback/spec or native metadata."""
    feedback = controllers.get("entrant_controller")
    if callable(feedback):
        return feedback
    if timeline is None:
        if arm not in ("native", "NATIVE", "B") and "entrant_controller_spec" not in controllers:
            raise ValueError("entrant_control requires the same-arm entrant controller algorithm")
        timeline = dict(controller=torch.zeros(1, (steps+1)//2, total_slots, dtype=torch.long),
            acceleration_mps2=torch.zeros(1, (steps+1)//2, total_slots),
            target_speed_mps=torch.zeros(1, (steps+1)//2, total_slots))
    specs = ({arm: controllers["entrant_controller_spec"]}
             if "entrant_controller_spec" in controllers else {})
    return feedback_from_specs(specs, [arm], model, timeline, original_slots, "cpu",
                               branches=[controllers.get("controller_branch")])
