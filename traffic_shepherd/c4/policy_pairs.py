"""LOOP14 paired collection: the learned policy and the native reference from ONE SUMO state.

    python -m traffic_shepherd.c4.collect --policy-pairs --policy <policy_latest.pt> \\
        --settings-wm-ckpt <wm_s.pt> --settings-wm-config <cfg> --registration <REG.json> \\
        --seeds 968000-968023 --case-plan weighted-cycle --population textbook \\
        --warmup 300 --branch-first 360 --branch-cadence 300 --branch-points 2 \\
        --branch-horizon 300 --workers 46 --out <new dir>

Per world (one TRAIN seed of a registered block):

1. **Base** episode, every CAV native (no policy), from SUMO insertion to the last decision
   time.  At each decision time ``t_k`` the full SUMO state is saved
   (``branching.save_state``; ``--save-state.rng true``) together with the env's Python state.
   The base episode is written like every collector episode (native TRAIN data).
2. **Arms** from every saved state, each in a fresh SUMO process with the identical state/RNG:

   * ``POLICY`` - the current learned policy drives every CAV (accel + lane head), through
     :class:`wms.control.policy.PolicyExecutor` (deterministic unless
     ``C4_POLICY_STOCHASTIC=1``).  Its lane consequence mask needs the WM's LEARNED per-slot
     driver settings: :class:`WmSettings` runs the accepted WM's driver encoder on the env's
     kinematic history (no driver labels, no vType/type files, no truth tables);
   * ``B`` - all CAVs native (the reference arm of :mod:`wms.data.c4_pairs`).

   A fresh executor per arm: the base is native, so the policy has no pre-branch memory.
3. Each arm is written as a ``c4_sumo_episode/1`` branch episode (60 s base pre-history +
   branch; ``warmup_s = t_k``; ``branch.pair_key = <world>@t<t_k>``), the format consumed by
   ``wms.data.c4_episodes`` and :class:`wms.data.c4_pairs.PairedBranchSource`.  Its
   ``summary.json`` carries the authoritative J ledger of the branch (on-road speed-deficit
   vehicle-seconds + source-waiting vehicle-seconds, summed over physics steps, exactly the
   env's ``interval`` accounting) and the exogenous requested population up to the branch end,
   so ``wms.control.loop14_returns`` can export real critic returns without reconstruction.

Nothing here is a designed controller, a timer, a forced merge or a threshold: the base and
the reference arm are SUMO native; the policy arm is the learned actor.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

POLICY_TAG = "loop14_pairs"
PAIR_SCHEMA = "c4_loop14_pair_summary/1"
BRANCHES_SCHEMA = "c4_loop14_pairs/1"
ARMS = (("POLICY", "policy"), ("B", "native"))


def decision_times(warmup_s: int, first_s: int, cadence_s: int, points: int) -> list[int]:
    """``first + k * cadence`` for ``k < points``; the 60 s pre-history must follow the warmup."""
    from .genrand import PREHISTORY_S
    if points < 1 or cadence_s < 1:
        raise ValueError("policy pairs need >= 1 decision point and a positive cadence")
    if first_s < max(warmup_s, PREHISTORY_S):
        raise ValueError("first decision time must be at or after the warmup and the pre-history")
    return [int(first_s + k * cadence_s) for k in range(int(points))]


class WmSettings:
    """Learned per-slot driver settings for the policy's lane consequence mask in SUMO.

    The accepted WM's driver encoder is applied to the env's rolling kinematic history (the
    same featuriser the deployment ``OracleSource`` uses).  Only trajectories enter; the
    output is the WM's inferred IDM settings, never a label."""

    def __init__(self, model, *, secure_gap_tier: bool = False):
        self.model = model
        self.secure_gap_tier = bool(secure_gap_tier)

    def augment(self, env, state: dict) -> dict:
        import torch
        from wms.model.driver_encoder import PARAMETER_NAMES

        m = self.model
        hist, hmask = env.driver_history(None, history_len=int(m.history_len))
        kwargs = {"history": hist, "history_mask": hmask}
        if getattr(m, "long_history", False):
            kwargs["driver_history"], kwargs["driver_history_mask"] = env.driver_history(
                m.history_s, m.history_stride_s)
        with torch.no_grad():
            _, _, params = m.encode_drivers(state, **kwargs)
        if params is None or any(name not in params for name in PARAMETER_NAMES):
            raise ValueError("WM returned no learned driver parameters for the lane mask")
        out = dict(state)
        out["inferred_driver_parameters"] = torch.stack(
            [params[name].float() for name in PARAMETER_NAMES], -1)
        out["inferred_idm_delta"] = float(m.physics.idm()[4])
        if self.secure_gap_tier:
            out["secure_gap_T_s"] = params["time_headway"].float()
            out["secure_gap_b_mps2"] = params["decel"].float()
        return out


class PairPolicy:
    """``(env, state) -> (accel [1,K], lanes [1,K] | None)`` for the POLICY arm."""

    def __init__(self, executor, settings: Optional[WmSettings]):
        self.executor = executor
        self.settings = settings

    def __call__(self, env, state):
        if self.settings is not None:
            state = self.settings.augment(env, state)
        if getattr(self.executor, "has_lane_head", False):
            return self.executor(state, with_lanes=True)
        return self.executor(state), None


def load_pair_policy_factory(policy_ckpt: str, wm_ckpt: Optional[str], wm_config: Optional[str],
                             *, device: str = "cpu", need_wm: bool = False) -> Callable[[], PairPolicy]:
    """Factory of fresh :class:`PairPolicy` objects (one per arm; networks/WM shared).

    ``need_wm`` (S-SUMOCREDIT extras: gate / credit imagination) loads the accepted WM even when
    the lane mask needs no learned settings; ``make.models`` / ``make.components`` expose the WM
    members and ``(policy, group_encoder, lane_rules)``."""
    from wms.control.policy import PolicyExecutor, load_policy_components
    import os

    policy, encoder, rules, _ = load_policy_components(policy_ckpt, device=device)
    needs_settings = bool(rules is not None and (rules.consequence_mask or rules.secure_gap_hook))
    settings = None
    models = None
    if needs_settings or need_wm:
        if not (wm_ckpt and wm_config):
            raise ValueError("the policy's lane mask / the S-SUMOCREDIT extras need the accepted "
                             "WM (--settings-wm-ckpt/--settings-wm-config)")
        from wms.control.policy14_cli import load_members
        models, _ = load_members([wm_ckpt], [wm_config], device)
    if needs_settings:
        settings = WmSettings(models[0], secure_gap_tier=bool(rules.secure_gap_hook))
    deterministic = os.environ.get("C4_POLICY_STOCHASTIC") != "1"

    def make() -> PairPolicy:
        exe = PolicyExecutor(policy, encoder, lane_rules=rules, deterministic=deterministic,
                             device=device)
        return PairPolicy(exe, settings)

    make.needs_settings = needs_settings
    make.models = models
    make.components = (policy, encoder, rules)
    make.settings = settings
    make.device = device
    return make


def requested_by(env, t_end: float) -> int:
    """Exogenous requested population with a scheduled departure at or before ``t_end``."""
    return int(sum(1 for r in env.requests if float(r.depart_s) <= float(t_end) + 1e-6))


def entry_backlog_at(env, now: float) -> list[int]:
    """Requested-but-not-inserted vehicles ``[mainline, ramp]`` at ``now`` (the converter's
    ``entry_backlog`` convention: entrance 1 = a route id containing "ramp").  Uses only the
    request schedule's departure times/routes and the ledger's inserted ids."""
    inserted = env.ledger.inserted
    out = [0, 0]
    for r in env.requests:
        if float(r.depart_s) <= float(now) + 1e-6 and r.vehicle_id not in inserted:
            out[1 if "ramp" in str(r.route).lower() else 0] += 1
    return out


def run_policy_pair_job(env, *, policy_factory: Callable[[], PairPolicy], seed: int,
                        demand_vph: int, penetration: float, warmup_s: int, times: list[int],
                        branch_h: int, episode_dir: Path, branch_root: Path,
                        meta: Optional[dict] = None, policy_ckpt: str = "",
                        extras=None) -> dict:
    """Native base + POLICY/B arms from every saved state (module docstring).

    ``extras`` (S-SUMOCREDIT, :class:`traffic_shepherd.c4.sumocredit.PairExtras`; default None =
    unchanged): gate calibration, the gated policy arm and real-credit arms run after the POLICY/B
    arms of every point; their ledgers go to separate JSONL files (no branch episodes)."""
    from .branching import load_state, prepare_branch_mode, save_state
    from .genrand import PREHISTORY_S, MemoryRecorder, _jsonable, _snapshot, _world, write_branch_episode
    from .recorder import write_json

    if not times or branch_h < 2:
        raise ValueError("policy pairs need decision times and a branch horizon >= 2 s")
    episode_dir, branch_root = Path(episode_dir), Path(branch_root)
    base_h = int(times[-1] - warmup_s + 1)       # the base only has to reach the last t_k
    meta = dict(meta or {}, policy="native", loop14={"policy_checkpoint": policy_ckpt})
    t0 = time.monotonic()
    state = env.reset(seed=seed, demand_vph=demand_vph, penetration=penetration,
                      warmup_s=warmup_s, horizon_s=base_h, controller="direct",
                      episode_dir=episode_dir, meta=meta)
    base_rec = env.recorder
    state_dir = branch_root / "_states"
    saved = []
    belief = extras.base_belief(policy_factory) if extras is not None else None
    while True:
        now = int(round(env.now))
        if now in times:
            state_dir.mkdir(parents=True, exist_ok=True)
            path = state_dir / f"t{now:04d}.xml"
            save_state(env, path)
            saved.append({"t_s": now, "path": path, "snap": _snapshot(env),
                          "i_k": len(base_rec.frames) - 1,
                          "entry_backlog": entry_backlog_at(env, env.now)})
            if belief is not None:
                from wms.control.confident_gate import clone_belief_source
                saved[-1]["belief"] = clone_belief_source(belief)
        state, info = env.step(None)
        if info["done"]:
            base_summary = info["summary"]
            break
        if belief is not None:
            belief.observe(env)
    if [p["t_s"] for p in saved] != list(times):
        raise RuntimeError(f"base episode did not reach every decision time {times}")
    base_wall = time.monotonic() - t0
    if env.api is None:
        import traci
        env.api = traci
    world = _world(seed, demand_vph, penetration, warmup_s)
    world_meta = {k: v for k, v in meta.items() if k != "loop14"}
    branches, t_branch = [], time.monotonic()
    marks = extras.marks(branch_h) if extras is not None else ()
    extra_out = []
    for point in saved:
        t_k = point["t_s"]
        prefix = {}
        for kind, tag in ARMS:
            load_state(env, point["path"], point["snap"])
            prepare_branch_mode(env)
            env.end_s = max(env.end_s, t_k + branch_h + 1)   # a branch never closes the episode
            rec = MemoryRecorder(env.k)
            env.recorder = rec
            fn = policy_factory() if kind == "POLICY" else None
            st = env.state()
            totals = {"collisions": 0, "teleports": 0, "lane_issued": 0, "lane_completed": 0,
                      "dropped_vehicle_steps": 0}
            ledger = {"speed_deficit_vehicle_seconds": 0.0, "source_wait_vehicle_seconds": 0.0,
                      "ttc_below_3_vehicle_seconds": 0.0, "arrivals": 0}
            w0 = time.monotonic()
            for step in range(int(branch_h)):
                if fn is None:
                    st, info = env.step(None)
                else:
                    accel, lanes = fn(env, st)
                    st, info = env.step(accel, lanes=lanes)
                iv = info["interval"]
                totals["collisions"] += int(iv.get("collision_involvements", 0))
                totals["teleports"] += int(iv.get("starting_teleports", 0))
                totals["dropped_vehicle_steps"] += int(iv.get("dropped_vehicle_steps", 0))
                totals["lane_issued"] += int(info["lane_commands"].get("issued", 0))
                totals["lane_completed"] += int(info["lane_commands"].get("completed", 0))
                for key in ledger:
                    ledger[key] += iv.get(key, 0)
                if step + 1 in marks:              # S-SUMOCREDIT: in-memory prefix ledgers only
                    from .sumocredit import prefix_record
                    prefix[(kind, step + 1)] = prefix_record(ledger, totals)
            env.recorder = None
            ledger["J_vehicle_seconds"] = (ledger["speed_deficit_vehicle_seconds"]
                                           + ledger["source_wait_vehicle_seconds"])
            arm_tag = f"{POLICY_TAG}_b{t_k:04d}_{tag}"
            pair_key = f"{world}@t{t_k}"
            bmeta = dict(world_meta, seed=seed, demand_vph=demand_vph, penetration=penetration,
                         warmup_s=float(t_k), horizon_s=float(branch_h), world_t0_s=float(warmup_s),
                         controller="direct", k_slots=env.k, dt_s=0.5, control_dt_s=1.0,
                         arm=arm_tag, policy=arm_tag,
                         branch={"pair_key": pair_key, "base_world": world,
                                 "base_episode": str(episode_dir), "branch_t_s": t_k,
                                 "arm": kind, "kind": kind, "arm_seed": None,
                                 "horizon_s": branch_h, "prehistory_s": PREHISTORY_S,
                                 "policy_checkpoint": policy_ckpt if kind == "POLICY" else None,
                                 "settings_source": ("wm_driver_encoder_on_kinematic_history"
                                                     if kind == "POLICY" and fn.settings else None)})
            bsum = {"schema": PAIR_SCHEMA, **totals, "ledger": ledger,
                    "seed": seed, "controller": "native" if kind == "B" else "current_policy",
                    "warmup_s": float(t_k), "horizon_s": float(branch_h),
                    "requested_population": requested_by(env, t_k + branch_h),
                    "start_entry_backlog": point["entry_backlog"],
                    "wall_s": time.monotonic() - w0, "sim_s": branch_h}
            out = write_branch_episode(branch_root / f"t{t_k:04d}_{tag}", base_recorder=base_rec,
                                       i_k=point["i_k"], branch=rec, env=env, meta=bmeta,
                                       summary=bsum)
            branches.append({"t_s": t_k, "arm": kind, "pair_key": pair_key, **totals,
                             "J": ledger["J_vehicle_seconds"], "wall_s": bsum["wall_s"], **out})
        if extras is not None:
            extra_out.append(extras.run_point(env, point, prefix, policy_factory=policy_factory,
                                              seed=seed, world=world, branch_h=branch_h))
        point["path"].unlink(missing_ok=True)      # own scratch state file
    env.close()
    if state_dir.exists() and not any(state_dir.iterdir()):
        state_dir.rmdir()
    pairs = {}
    for b in branches:
        pairs.setdefault(b["pair_key"], {})[b["arm"]] = b["J"]
    block = {"schema": BRANCHES_SCHEMA, "n_points": len(saved), "decision_times_s": list(times),
             "n_branches": len(branches), "branch_horizon_s": branch_h,
             "collisions": sum(b["collisions"] for b in branches),
             "teleports": sum(b["teleports"] for b in branches),
             "base_wall_s": base_wall, "branch_wall_s": time.monotonic() - t_branch,
             "branches": branches, "sim_s": int(base_summary["sim_seconds"]) + branch_h * len(branches),
             # SUMO policy-minus-native J per pair (negative = the policy is better)
             "pair_dJ": {k: v["POLICY"] - v["B"] for k, v in pairs.items() if len(v) == 2}}
    branch_root.mkdir(parents=True, exist_ok=True)
    if extras is not None:
        block["sumocredit"] = extras.write(branch_root, extra_out)
        block["sim_s"] += block["sumocredit"]["sim_s"]
        if extras.long_group_horizon:
            for p in extra_out:
                rec = p.get("credit_long", {})
                block["collisions"] += rec.get("extra_collisions", 0)
                block["teleports"] += rec.get("extra_teleports", 0)
    write_json(branch_root / "BRANCHES_COMPLETE.json", _jsonable(block))
    return {"summary": base_summary, "branches": block}


def check_registered_seeds(seeds, registration: Optional[str]) -> dict:
    """Fail closed unless every seed lies in the registration's fresh TRAIN block.

    The registration (LOOP14_DESIGN.md template) is a JSON file the integrator writes BEFORE any
    run: ``{"id", "status": "registered", "split": "TRAIN", "seed_block": [lo, hi],
    "exclude": [[lo, hi], ...]}``.  Evaluation seeds 964000-964199 are refused regardless."""
    import json
    if not registration:
        raise ValueError("--registration REG.json is required for policy-pair collection")
    doc = json.loads(Path(registration).read_text(encoding="utf-8"))
    if doc.get("status") != "registered" or doc.get("split") != "TRAIN":
        raise ValueError("registration must have status 'registered' and split 'TRAIN'")
    lo, hi = (int(x) for x in doc["seed_block"])
    excluded = [(int(a), int(b)) for a, b in doc.get("exclude", [])]
    for seed in seeds:
        if 964000 <= seed <= 964199:
            raise ValueError(f"evaluation seed {seed} refused")
        if not lo <= seed <= hi or any(a <= seed <= b for a, b in excluded):
            raise ValueError(f"seed {seed} is outside the registered fresh TRAIN block {lo}-{hi}")
    return doc


def pair_deltas(branch_root: Path) -> dict:
    """``{pair_key: J_policy - J_native}`` from a finished job's BRANCHES_COMPLETE.json."""
    import json
    doc = json.loads((Path(branch_root) / "BRANCHES_COMPLETE.json").read_text(encoding="utf-8"))
    return {k: float(v) for k, v in doc.get("pair_dJ", {}).items() if np.isfinite(v)}
