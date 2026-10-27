"""Learned-action proposal/replay and K=8 paired WM scoring, collection only."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from .probe import FUTURES, HORIZONS, select


class FrozenAction:
    """Persist a registered actor output, rechecking existing safety/lane masks.

    Commands are keyed by vehicle ID, never by a recycled collection slot. There
    are no gap, speed, headway or driver-setting thresholds in this wrapper.
    """
    def __init__(self, policy, commands, ids, absolute_hold=False):
        self.policy, self.commands, self.ids = policy, commands, ids
        self.residual_on_native = bool(getattr(policy, "residual_on_native", False)) and not absolute_hold

    def __getattr__(self, key):
        return getattr(self.policy, key)

    def __call__(self, obs, signal, **kwargs):
        from wms.control.policy import PolicyOutput
        ids = self.ids() if callable(self.ids) else self.ids
        shape = obs.cav_mask.shape
        accel = torch.tensor([self.commands.get(v, (0., 0))[0] for v in ids],
                             device=obs.own.device).expand(shape)
        zero = torch.zeros_like(accel)
        if "lane_mask" not in kwargs:
            return accel, zero
        choices = torch.tensor([self.commands.get(v, (0., 0))[1] for v in ids],
                               device=accel.device, dtype=torch.long).expand(shape)
        mask = kwargs["lane_mask"]
        allowed = mask.gather(-1, choices[..., None]).squeeze(-1)
        choices = torch.where(allowed, choices, torch.zeros_like(choices))
        return PolicyOutput(accel, choices, zero, zero, zero, zero, mask.sum(-1))


def slot_ids(env):
    from wms.control.confident_gate import env_vehicle_id
    return [env_vehicle_id(env, i) for i in range(env.k)]


@torch.no_grad()
def proposals(factory, state, ids, slot, seed):
    """Eight registered policy samples plus two registered zero holds.

    Durations alternate 30/60s. GROUP shares its latent noise; SINGLE uses
    individual units. Raw values/lanes come only from the trained action space.
    """
    from wms.control.confident_gate import IdNoiseStream, gate_units
    from wms.control.lanes import LaneDwell, LaneRules, lane_feasibility
    from wms.control.obs import build_observations
    from wms.control.policy import PolicyNoise
    from wms.control.rsu_field import build_rsu_field
    policy, encoder, rules = factory.components
    obs = build_observations(state)
    signal = (encoder(obs, state) if getattr(encoder, "requires_state", False)
              else encoder(obs, state["s_m"], build_rsu_field(state)))
    groups = gate_units(state, "cav" if slot == "single" else "group")
    groups = [g for g in groups if slot == "single" or len(g) >= 2]
    if not groups:
        return []
    lane_rules = rules or LaneRules()
    ages = LaneDwell(lane_rules.dwell_s, 1.).step(state)
    lane_mask = lane_feasibility(state, ages, lane_rules)
    result = []
    for i in range(10):
        group = groups[i % len(groups)]
        noise = IdNoiseStream(seed, i + 1, correlated=(slot == "group")).draw(0, ids)
        noise = PolicyNoise(noise.accel.to(obs.own.device), noise.lane_gumbel.to(obs.own.device))
        kwargs = {"lane_mask": lane_mask} if getattr(policy, "has_lane_head", False) else {}
        out = policy(obs, signal, deterministic=False, noise=noise, **kwargs)
        a = out.accel if hasattr(out, "accel") else out[0]
        lanes = out.lane_choice if hasattr(out, "lane_choice") else torch.zeros_like(a, dtype=torch.long)
        commands = {ids[j]: (float(a[0, j]) if i < 8 else 0., int(lanes[0, j]) if i < 8 else 0)
                    for j in group}
        # Labels describe sampled commands; they do not trigger hand-designed behaviour.
        kind = ("lane_vacate" if any(v[1] for v in commands.values()) else
                "speed_up" if sum(v[0] for v in commands.values()) > 0 else
                "hold" if i >= 8 else "merge_yield_meter")
        result.append({"id": f"{slot}:{i:02d}", "action_type": f"{slot}:{kind}",
                       "duration_s": (30, 60)[i % 2], "commands": commands, "absolute_hold": i >= 8})
    return result


@torch.no_grad()
def score(factory, state, inputs, ids, candidates, seed):
    """Eight stochastic futures of the accepted WM; identical randomness per arm.

    Native controller, inferred drivers, lane uniforms and arrivals use the
    existing paired-imagination path. No driver labels or invented noise scales.
    """
    from wms.control.confident_gate import ArmImagination
    from wms.control.bridge import make_cav_controller
    from wms.control.safety import safe_accel

    class Imagination(ArmImagination):
        def _controller(self, member, rows, stream=None, ids=None):
            candidate = lookup.get(id(member))
            if candidate is None:
                return super()._controller(member, rows, stream, ids)
            actor = FrozenAction(self.policy, candidate["commands"], ids, candidate["absolute_hold"])
            # The bridge's index is physics-local and restarts for every arm/future.
            tick = [0]
            def released(current):
                index = current.get("physics_step_index", tick[0])
                tick[0] = int(index) + 1
                return torch.full_like(current["present"], int(index) * .5 >= candidate["duration_s"])
            return make_cav_controller(actor, self.group_encoder, safe_accel,
                                       lane_rules=self.lane_rules, act_mask=member[None].expand(rows, -1),
                                       zone=released, native=self.native, **self.controller_kwargs)

    arms = [torch.zeros(len(ids), dtype=torch.bool)]
    lookup = {}
    for candidate in candidates:
        mask = torch.tensor([v in candidate["commands"] for v in ids], dtype=torch.bool)
        arms.append(mask)
        lookup[id(mask)] = candidate
        candidate["paired_benefit"] = {}
    policy, encoder, rules = factory.components
    if len(factory.models) != 1:
        raise ValueError("probe scoring requires one SHA-bound accepted WM")
    from wms.control.contract import RewardWeights
    from wms.control.reward import team_reward
    prefixes = {}

    class PrefixModel:
        def __getattr__(self, key):
            return getattr(factory.models[0], key)

        def rollout(self, initial, **kwargs):
            rollout = factory.models[0].rollout(initial, **kwargs)
            current = dict(initial)
            total = torch.zeros(current["s_m"].shape[0], dtype=torch.float64)
            dt = float(self.physics.dt_s)
            for t in range(kwargs["n_steps"]):
                accel = rollout["accel"][:, t]
                nxt = {**current, "s_m": rollout["positions"][:, t], "v_mps": rollout["speeds"][:, t],
                       "lane": rollout["lanes"][:, t], "present": rollout["masks"][:, t], "a_mps2": accel}
                nxt.pop("veh_kin", None)
                total += team_reward(current, nxt, accel, RewardWeights())["delay"].double().cpu()
                current = nxt
                h = (t + 1) * dt
                if h in HORIZONS:
                    prefixes[str(int(h))] = total.clone().reshape(len(arms), FUTURES)
            return rollout

    imag = Imagination([PrefixModel()], policy, encoder, lane_rules=rules, reference="native",
                       horizon_s=300, seed=seed)
    # Eight *lane/merge futures*, shared learned driver encodings. No artificial
    # driver/embedding noise is introduced just to manufacture uncertainty.
    imag.n_samples = FUTURES
    imag.arms_J(state, arms, ids=ids, key=f"probe:{seed}", **inputs)
    for h, J in prefixes.items():
        for j, candidate in enumerate(candidates):
            candidate["paired_benefit"][h] = (J[0] - J[j + 1]).tolist()
    return candidates


def choose(extras, env, point, factory, seed, world):
    from wms.control.deploy import OracleSource, compact_inputs
    try:
        calibration = json.loads(Path(extras.probe_calibration).read_text())
    except (OSError, ValueError):
        extras._probe_skip_reason = "missing or malformed real-pair CQR"
        return None
    sha = factory.wm_sha256
    # Validate calibration before spending CPU on long rollouts.
    try:
        select([], calibration, sha, world)
    except (ValueError, KeyError, TypeError):
        extras._probe_skip_reason = "stale or overlapping real-pair CQR"
        return None
    if not calibration.get("ready"):
        extras._probe_skip_reason = "insufficient calibration worlds"
        return None
    extras._load(env, point, 300)
    raw, _ = OracleSource(factory.models[0]).inputs(env)
    if factory.settings is not None:
        raw["state"] = factory.settings.augment(env, raw["state"])
    inputs, keep = compact_inputs(raw)
    state = inputs.pop("state")
    ids = [slot_ids(env)[i] for i in keep.tolist()]
    candidates = proposals(factory, state, ids, extras.probe_slot, seed)
    if not candidates:
        return None
    selected = select(score(factory, state, inputs, ids, candidates, seed), calibration, sha, world)
    if selected is not None:
        import hashlib
        selected["calibration_sha256"] = hashlib.sha256(Path(extras.probe_calibration).read_bytes()).hexdigest()
    else:
        extras._probe_skip_reason = "no uncertain negative-mean candidate"
    return selected


def execute(extras, env, point, prefix, factory, candidate, seed, world):
    from wms.control.policy import PolicyExecutor
    from .policy_pairs import PairPolicy
    from .sumocredit import prefix_record
    from .genrand import MemoryRecorder, PREHISTORY_S, write_branch_episode

    extras._load(env, point, 300)
    policy, encoder, rules = factory.components
    frozen = FrozenAction(policy, candidate["commands"], lambda: slot_ids(env), candidate.get("absolute_hold", False))
    exe = PolicyExecutor(frozen, encoder, lane_rules=rules, device=getattr(factory, "device", "cpu"))
    fn = PairPolicy(exe, factory.settings)
    members = frozenset(candidate["commands"])
    step_fn = extras._members_step(fn, members)
    recorder = MemoryRecorder(env.k)
    env.recorder = recorder
    totals = {"collisions": 0, "teleports": 0, "dropped_vehicle_steps": 0}
    ledger = {"speed_deficit_vehicle_seconds": 0., "source_wait_vehicle_seconds": 0.}
    st = env.state()
    realized, marks = {}, {}
    n = 0
    for n in range(1, 301):
        if n <= candidate["duration_s"]:
            accel, kw = step_fn(env, st)
            st, info = env.step(accel, **kw)
        else:
            st, info = env.step(None)
        iv = info["interval"]
        totals["collisions"] += int(iv.get("collision_involvements", 0))
        totals["teleports"] += int(iv.get("starting_teleports", 0))
        totals["dropped_vehicle_steps"] += int(iv.get("dropped_vehicle_steps", 0))
        for key in ledger:
            ledger[key] += iv.get(key, 0)
        if n in HORIZONS:
            marks[str(n)] = prefix_record(ledger, totals)
            realized[str(n)] = prefix[("B", n)]["J"] - marks[str(n)]["J"]
        if totals["collisions"]:
            extras._probe_stopped = True
            break
    env.recorder = None
    record = {**candidate, "schema": "c4_probe_point/1", "split": "TRAIN", "world": world,
              "seed": seed, "t_s": point["t_s"], "wm_sha256": factory.wm_sha256,
              "pair_key": f"{world}@t{point['t_s']}", "role": "collection_data_only",
              "horizon_s": 300, "realized_benefit": realized, "prefixes": marks,
              "extra_sim_s": n, **totals, "extra_collisions": totals["collisions"],
              "extra_teleports": totals["teleports"], "complete": n == 300 and not totals["collisions"]}
    record["native_prefixes"] = {str(h): prefix[("B", h)] for h in HORIZONS if ("B", h) in prefix}
    record["valid"] = bool(record["complete"] and not any(
        r.get(k, 0) for r in [totals, *record["native_prefixes"].values()]
        for k in ("collisions", "teleports", "dropped_vehicle_steps")))
    # Full trajectories remain separate from policy evaluation/return manifests.
    if record["complete"]:
        root = Path(point["probe_root"])
        meta = copy.deepcopy(point["native_meta"])
        meta.update(policy="probe", arm="probe", branch={**meta["branch"], "kind": "PROBE",
                    "arm": "PROBE", "policy_checkpoint": None, "probe": True,
                    "prehistory_s": PREHISTORY_S})
        summary = {"schema": "c4_probe_episode/1", **totals, "ledger": {**ledger,
                   "J_vehicle_seconds": sum(ledger.values())}, "sim_s": 300,
                   "horizon_s": 300, "warmup_s": point["t_s"], "controller": "direct"}
        write_branch_episode(root / f"t{point['t_s']:04d}_probe", base_recorder=point["base_recorder"],
                             i_k=point["i_k"], branch=recorder, env=env, meta=meta, summary=summary)
        # Reuse B on disk, with its original CRN pair key; zero simulated seconds.
        import shutil
        shutil.copytree(point["native_episode"], root / f"t{point['t_s']:04d}_native")
    return record
