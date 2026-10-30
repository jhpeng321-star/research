"""Shared observable-only feedback and frozen WM rollouts for all distillation tools."""
from __future__ import annotations

import hashlib
import json
import re

import torch

from wms.control.contract import MERGE_S0_M, MERGE_S1_M, Observation
from wms.control.obs import build_observations
from wms.control.rsu_field import build_rsu_field
from wms.control.start_states import checked_path

from .law import absolute_executed, execute

WM_SHA = "b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436"
MAPPO_SHA = "aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564"
STATE_KEYS = ("s_m", "v_mps", "a_mps2", "lane", "present", "is_cav", "length_m")
FORBIDDEN = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype", re.I)


def safe_path(path):
    path = checked_path(path)
    if FORBIDDEN.search(str(path)):
        raise ValueError("prohibited distillation path")
    return path


def sha256(path):
    return hashlib.sha256(safe_path(path).read_bytes()).hexdigest()


def configure_execution(driver):
    import os
    from wms.control.execution import contract
    os.environ.update(C4_EXECUTION_PARITY="2", C4_SUMO_LEADER_RULE="1",
                      C4_EXECUTION_DRIVER=str(safe_path(driver)))
    return contract()


def check_seed(seed):
    seed = int(seed)
    if seed < 0 or 964000 <= seed <= 964199 or 977000 <= seed <= 977999:
        raise ValueError(f"excluded seed: {seed}")
    return seed


def check_metadata(metadata):
    for meta in metadata:
        if FORBIDDEN.search(str(meta)):
            raise ValueError("restricted corpus provenance")
        world = str(meta.get("world", meta.get("world_id", "")))
        match = re.search(r"w(\d+)m", world)
        if match:
            check_seed(match[1])
        if "seed" in meta:
            check_seed(meta["seed"])


def observable_state(state):
    # An allowlist is important: callback state also carries inferred driver labels.
    return {k: state[k].detach().clone() for k in STATE_KEYS}


def student_inputs(state):
    public = observable_state(state)
    zeros = torch.zeros_like(public["s_m"])
    obs = build_observations({**public, "cmd_accel_mps2": zeros, "lane_intent": zeros})
    return obs, public["s_m"], build_rsu_field(public)


def detach_inputs(inputs, device="cpu"):
    obs, positions, field = inputs
    values = (obs.own, obs.reports, obs.report_mask, obs.cav_mask) if isinstance(obs, Observation) else obs
    # Plain tensors/tuples keep replay compatible with weights_only=True.
    return (tuple(v.detach().to(device).clone() for v in values),
            positions.detach().to(device).clone(), field.detach().to(device).clone())


def residual(policy, encoder, inputs):
    obs, positions, field = inputs
    if not isinstance(obs, Observation):
        obs = Observation(*obs)
    return policy(obs, encoder(obs, positions, field), deterministic=True)[0]


def zones(state):
    merge = (state["s_m"] >= MERGE_S0_M) & (state["s_m"] <= MERGE_S1_M)
    ramp = (state["lane"] == 2) & ~merge
    return {"merge": merge, "ramp": ramp, "mainline": ~merge & ~ramp}


class Feedback:
    """One-second actor clock, half-second shared execution, live entrant masking.

    MAPPO memory is private to the teacher. The student's cached decision contains
    only its own deployable Observation + RSU field, never teacher memory/settings.
    """

    def __init__(self, teacher, policy=None, encoder=None, *, drive="teacher", law="current",
                 record=None):
        from wms.control.mappo.agent import ActorStep
        self.teacher = ActorStep(teacher.policy, teacher.group_encoder, teacher.lane_rules)
        self.policy, self.encoder, self.drive, self.law = policy, encoder, drive, law
        self.record = record
        self.tick = 0
        self.previous = self.held_teacher = self.held_student = self.inputs = None

    def native_lane_mask(self, state):
        mask = state["present"].bool() & state["is_cav"].bool() & (state["lane"] >= 0)
        return torch.zeros_like(mask) if self.drive == "teacher" else mask

    @torch.no_grad()
    def __call__(self, state):
        from wms.model.cav import WM_LANE_TARGET_KEY, WM_MERGE_KEY
        public = observable_state(state)
        mask = public["present"].bool() & public["is_cav"].bool() & (public["lane"] >= 0)
        decision = self.tick % 2 == 0
        teacher_lane = public["lane"]
        lane_changed = torch.zeros_like(mask)
        if decision:
            action = self.teacher(public, deterministic=True)
            self.held_teacher = torch.where(mask, action["safe"], 0.)
            teacher_lane = torch.where(action["lanes"] >= 0, action["lanes"], teacher_lane)
            lane_changed = action["lane_changed"]
            self.inputs = detach_inputs(student_inputs(public), public["s_m"].device)
            self.held_student = (torch.zeros_like(self.held_teacher) if self.policy is None
                                 else residual(self.policy, self.encoder, self.inputs))
        else:
            continuing = mask & self.previous
            self.held_teacher = torch.where(continuing, self.held_teacher, 0.)
            self.held_student = torch.where(continuing, self.held_student, 0.)
        # A newborn between decisions has no inherited command. It uses CTRL0
        # until the next decision, and is still counted in every physics-tick mask.
        ready = mask if decision else mask & self.previous
        teacher_accel = absolute_executed(public, self.held_teacher)
        ctrl0 = execute(public, torch.zeros_like(self.held_teacher))
        teacher_accel = torch.where(ready, teacher_accel, ctrl0)
        student_accel = execute(public, self.held_student, self.law)
        if self.record is not None:
            self.record(dict(state=public, inputs=detach_inputs(self.inputs), mask=mask,
                ready=ready, teacher=teacher_accel, student=student_accel,
                teacher_lane=teacher_lane, lane_changed=lane_changed, decision=decision,
                entrant=state.get("entrant", torch.zeros_like(mask)).detach().clone(), tick=self.tick,
                native_lane=state.get(WM_LANE_TARGET_KEY, public["lane"]).detach().clone(),
                native_lane_available=self.drive != "teacher"))
        if self.drive == "teacher":
            accel, lane = teacher_accel, teacher_lane
        else:
            accel = ctrl0 if self.drive == "ctrl0" else student_accel
            if WM_LANE_TARGET_KEY not in state:
                raise ValueError("student/CTRL0 require WM-computed native lane targets")
            lane = torch.where(public["lane"] != 2, state[WM_LANE_TARGET_KEY], public["lane"])
            state[WM_MERGE_KEY] = mask & (public["lane"] == 2)
        speed = (public["v_mps"] + .5 * accel).clamp_min(0.)
        position = public["s_m"] + .5 * speed
        self.previous = mask
        self.tick += 1
        return torch.stack((position, lane.to(speed), speed, accel), -1)


def load_assets(wm, wm_config, mappo, device):
    from wms.control.mappo.eval_bridge import load_policy_fn
    from wms.control.policy14_cli import load_members
    if sha256(wm) != WM_SHA or sha256(mappo) != MAPPO_SHA:
        raise ValueError("distillation requires the registered WM2 and MAPPO checkpoints")
    models, receipts = load_members([safe_path(wm)], [safe_path(wm_config)], device)
    teacher = load_policy_fn(safe_path(mappo), deterministic=True, device=device)
    teacher.policy.requires_grad_(False)
    teacher.group_encoder.requires_grad_(False)
    return models[0], teacher, receipts


def load_training_starts(raw, model, seed):
    from wms.control.train_cli import _read_config, _load_starts
    from wms.config.schema import from_mapping
    from wms.config.train import TrainConfig
    from wms.data.registry import resolve
    # Metadata preflight precedes dataset construction/array loading.
    source = from_mapping(TrainConfig, _read_config(safe_path(raw["data"]["wm_config"])))
    provenance = {}
    for spec in source.data.c4_sources:
        root = safe_path(resolve(spec.path))
        manifest = json.loads(safe_path(root / "C4_EPISODES_MANIFEST.json").read_text())
        check_metadata(manifest["episodes"])
        provenance[spec.name] = {"root": str(root),
            "manifest_sha256": sha256(root / "C4_EPISODES_MANIFEST.json")}
        for entry in manifest["episodes"]:
            safe_path(root / entry["dir"])
    train, val = _load_starts(raw["data"], [model], seed, entries=True)
    check_metadata(train.metadata.values())
    check_metadata(val.metadata.values())
    train.distill_provenance = provenance
    return train, val


@torch.no_grad()
def rollout(model, batch, feedback, *, horizon_s, seed, pad_max=288, draw=0,
            paired_batch=None, group_keys=None):
    from dataclasses import replace
    from wms.control.dreamer import with_entry_slots, _entry_process, _EntryRecorder
    from wms.control.futures import lane_uniform_draws, HISTORY_KEYS
    from wms.control.nativeinit import pair_merge_streams
    from tools.errsrc.inference import LearnedMix
    from wms.train.scoring_random import scoring_rng, sample_seed
    check_metadata(batch.metadata)
    disallowed = set(batch.rollout_kwargs) - set(HISTORY_KEYS)
    if disallowed:
        raise ValueError(f"only observed pre-t0 history is allowed: {sorted(disallowed)}")
    if batch.entry_feat is None or batch.entry_queue is None:
        raise ValueError("WM distillation requires observed entry features and queue state")
    if batch.entry_feat_control is not None:
        batch = replace(batch, entry_feat=batch.entry_feat_control)
    batch = with_entry_slots(batch, horizon_s, pad_max,
                             torch.Generator().manual_seed(seed), queue_state=True)
    initial = {k: v.clone() for k, v in batch.state.items()}
    pr = None
    if paired_batch is not None:
        # The exact identity-keyed streams used by wmcombined.e3_score / W-verdict.
        import numpy as np
        from wms.train.paired_random import PairedRandom
        pr = PairedRandom.from_batch(paired_batch, seed=sample_seed(seed, draw),
            device=initial["s_m"].device, pad=batch.entry_spec["pad"])
        initial["merge_pivot_rng_seed"] = torch.as_tensor(
            (pr._keys & np.uint64((1 << 52) - 1)).astype(np.int64), device=initial["s_m"].device)
        batch.entry_spec["seeds"] = pr.row_seeds
    else:
        pair_merge_streams(initial, None, len(batch.metadata), sample_seed(seed, draw))
    steps = int(round(horizon_s / .5))
    history = {k: v for k, v in batch.rollout_kwargs.items() if k in HISTORY_KEYS}
    _, embedding, params = model.encode_drivers(initial, **history)
    observed = history.get("driver_history_mask")
    if params is None or observed is None:
        raise ValueError("learned mix requires pre-t0 inferred settings and observed history")
    mix = LearnedMix(params, observed, sample_seed(seed, draw),
                     list(range(len(batch.metadata))) if group_keys is None else group_keys)
    learned = None
    if model.physics.entry_learned:
        from wms.model.entry_learned import load_learned_entry
        learned = load_learned_entry()
    entries = _EntryRecorder(_entry_process(batch.entry_spec, list(range(len(batch.metadata))),
        initial["s_m"].shape[1], "sumo_like_v2", learned_entry=learned,
        couple_draws=True, sample_index=draw if pr is None else 0))
    lane = (pr.uniform("lane", range(steps)) if pr is not None and model._samples_lanes else
            lane_uniform_draws([model], len(batch.metadata), steps, initial["s_m"].shape[1],
                              sample_seed(seed, draw), device=initial["s_m"].device))
    kwargs = {} if lane is None else {"lane_uniform": lane}
    with scoring_rng(sample_seed(seed, draw), initial["s_m"].device):
        roll = model.rollout(initial, n_steps=steps, cav_teacher=None, cav_controller=feedback,
            entries=entries, driver_params=params, driver_embedding=embedding,
            entrant_driver_sampler=mix, **history, **kwargs)
    return roll, entries, mix
