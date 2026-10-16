"""C4 deployment runner: one SUMO episode with a frozen or learned CAV controller.

    python -m wms.control.deploy --controller policy_check --state oracle \\
        --policy runs/active/C4-X/policy_best.pt --wm-ckpt <wm_s_best.pt> \\
        --seed 4107 --demand 2700 --penetration 0.2 --warmup 300 --horizon 600 --out runs/active/C4-EVAL-x/ep

Controllers:

* ``native | damping | pi | wall`` - the frozen baselines of :class:`traffic_shepherd.c4.env.CavSumoEnv`
  (identical to ``python -m traffic_shepherd.c4.collect --policy <name>``).
* ``policy`` - ``wms.control.policy.load_policy_fn(ckpt)`` every 1 s (``collect --policy <ckpt>`` path):
  per-CAV direct accelerations AND lane commands (``CavSumoEnv.step(accel, lanes=...)``).
* ``policy_check`` - design 2.6 Option B with DIRECT-OUTPUT candidates only (no
  designed controller is a check candidate or executed in this mode): every ``decision_period_s`` the RSU
  imagines each candidate ``{policy, policy_group_0, policy_group_1, policy_sample_0, policy_sample_1,
  hold_speed}`` in the frozen WM-S ensemble (:class:`wms.control.rsu_check.RsuChecker`, paired driver
  samples) from either the env's true WM-frame state + rolling history (``--state oracle``, privileged upper
  bound) or an RSU belief (``--state belief``: :class:`wms.control.belief.BeliefTracker` fed every control
  step with the CAV self reports + their front/rear sensor detections and the RSU zone detections, WM-S
  step propagator), then executes the chosen candidate in SUMO until the next decision.  ``policy`` =
  deterministic accel + lane; ``policy_group_0`` / ``policy_group_1`` = zero / perturbed group signal;
  ``policy_sample_k`` = the stochastic policy with a reproducible noise stream (the same draws in every WM
  member, driver sample and in SUMO execution); ``hold_speed`` = direct command a = 0, stay in lane
  (safety-filtered fallback).  ALL are executed as direct accelerations (+ lane commands) through the
  policy path; native / damping / pi remain only as whole-episode baselines (``--controller``).

Outputs in ``--out`` (a new directory): ``episode/`` (the collector's episode format, incl. ``summary.json``),
``decisions.jsonl`` (one record per decision: time, candidates, imagined returns mean/std/paired diffs/LCB,
choice, executed controller, state source, belief coverage, wall-clock), ``segments.jsonl`` (realised SUMO
interval totals per decision segment) and ``DEPLOY_SUMMARY.json``.  The campaign driver
(``scripts/c4/eval_campaign.py``) calls this module; its argument names are accepted as aliases.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import torch
from torch import Tensor

_RESTRICTED = re.compile(r"sealed|holdout|heldout|cal_split|chunk5|j4b|inc1", re.I)
BASELINES = ("native", "damping", "pi", "wall")
CONTROLLERS = (*BASELINES, "policy", "policy_check")
ALIASES = {"c4_policy": "policy", "c4_policy_rsu_check": "policy_check"}
#: Check candidates: direct-output only, all executed through the policy path.
CANDIDATES = ("policy", "policy_group_0", "policy_group_1", "policy_sample_0", "policy_sample_1",
              "hold_speed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def checked(path) -> Path:
    """Refuse restricted-split tokens (before and after resolution)."""
    value = Path(path)
    if _RESTRICTED.search(str(value)) or _RESTRICTED.search(str(value.resolve())):
        raise ValueError(f"restricted path refused: {value}")
    return value


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().tolist())
    return value


def _dump(stream, record) -> None:
    stream.write(json.dumps(_jsonable(record), allow_nan=False) + "\n")
    stream.flush()


# ------------------------------------------------------------------ imagination inputs
def compact_inputs(inputs: Mapping[str, Any]) -> tuple[dict, Tensor]:
    """Keep only present slots (no entries are imagined, so free padding slots are inert).

    ``inputs``: ``state`` ([1,K] tensors) and optional ``history``/``driver_history`` [1,K,...]
    tensors + masks.  Returns the gathered copy and the kept slot indices.
    """
    state = inputs["state"]
    k = state["s_m"].shape[1]
    keep = torch.nonzero(state["present"][0].bool()).flatten()
    if keep.numel() == 0:
        keep = torch.zeros(1, dtype=torch.long, device=state["s_m"].device)

    def gather(value):
        if torch.is_tensor(value) and value.ndim >= 2 and value.shape[0] == 1 and value.shape[1] == k:
            return value.index_select(1, keep.to(value.device))
        return value

    out = {key: gather(value) for key, value in inputs.items() if key != "state"}
    out["state"] = {key: gather(value) for key, value in state.items()}
    return out, keep


def long_history(raw: Mapping[str, Any], present_now, *, history_s, history_stride_s, dt_s):
    """``(driver_history [1,K,H,9], mask)`` from a slot-indexed raw history (trainer featuriser)."""
    from wms.data.history import long_history_features

    kin = torch.as_tensor(np.asarray(raw["kin"]), dtype=torch.float32)
    mask = torch.as_tensor(np.asarray(raw["mask"]), dtype=torch.bool)
    target = torch.as_tensor(np.asarray(raw["target_mask"]), dtype=torch.bool)
    feat, fmask = long_history_features(kin, mask, target, torch.as_tensor(present_now).bool(),
                                        history_s=history_s, history_stride_s=history_stride_s,
                                        dt_s=dt_s)
    return feat[None], fmask[None]


# ------------------------------------------------------------------ candidates
def build_candidates(model, policy, encoder, *, horizon_s: float = None, lane_rules=None,
                     sample_streams=(), perturbation: float = 0.1, log: Optional[list] = None) -> dict:
    """Direct-output check candidates (``CANDIDATES`` order with two sample streams)."""
    del horizon_s, log
    from .bridge import make_cav_controller
    from .rsu_check import default_candidates
    from .safety import safe_accel

    if lane_rules is None and getattr(policy, "has_lane_head", False):
        from .lanes import LaneRules
        lane_rules = LaneRules()
    return default_candidates(model, policy=policy, group_encoder=encoder, safety=safe_accel,
                              bridge=make_cav_controller, lane_rules=lane_rules,
                              perturbation=perturbation, sample_streams=sample_streams)


def make_exec_fn(policy, encoder_fn, *, device="cpu", lane_rules=None, deterministic=True,
                 noise_fn=None, hold=False, dwell=None, memory=None):
    """SUMO execution :class:`wms.control.policy.PolicyExecutor`: ``exe(state)`` -> accel [1,K]
    (NaN on non-CAV slots); ``exe(state, with_lanes=True, dwell_age=...)`` -> (accel, lanes)."""
    from .policy import PolicyExecutor

    return PolicyExecutor(policy, encoder_fn, lane_rules=lane_rules, deterministic=deterministic,
                          noise_fn=noise_fn, hold=hold, dwell=dwell, memory=memory, device=device)


def zero_signal(encoder):
    from .rsu_check import zero_group
    return zero_group(encoder)


class StreamNoise:
    """SUMO-side replay of a check sample stream: control tick j after the decision."""

    def __init__(self, stream):
        self.stream, self.step = stream, 0

    def begin(self) -> None:
        self.step = 0

    def __call__(self, shape):
        out = self.stream.noise(self.step, shape, slot_ids=torch.arange(shape[1]))
        self.step += 1
        return out


# ------------------------------------------------------------------ RSU state sources
def cav_reports_from_env(env) -> list:
    """CAV self reports (WM frame) + the CAV's own front/rear sensor detections (SUMO sensor)."""
    from .belief import CavReport, Detection

    frame, now = env.frame, env.now
    env.sensor.bind(env._table)
    reports = []
    for slot in np.flatnonzero(frame.present & frame.is_cav).tolist():
        if int(frame.lane[slot]) < 0:          # junction transit: no WM-frame lane this tick
            continue
        vid = env.builder.vehicle_ids[int(frame.vid_index[slot])]
        try:
            sensed = env.sensor.capture(vid, now_s=now)
        except (KeyError, ValueError):
            sensed = None
        front = rear = None
        if sensed is not None and sensed.front is not None:
            front = Detection(float(sensed.front.gap_m), float(sensed.front.relative_speed_mps))
        if sensed is not None and sensed.rear is not None:
            rear = Detection(float(sensed.rear.gap_m), float(sensed.rear.relative_speed_mps))
        reports.append(CavReport(cav_id=str(vid), s_m=float(frame.s[slot]), lane=int(frame.lane[slot]),
                                 v_mps=float(frame.v[slot]), a_mps2=float(frame.a[slot]),
                                 front=front, rear=rear))
    return reports


def rsu_detections_from_env(env) -> list:
    """Anonymous RSU-zone detections (the env's RSU sensor; slot kept as evaluation-only id)."""
    from traffic_shepherd.c4.state import rsu_detections

    from .belief import RsuDetection

    return [RsuDetection(d["s_m"], d["lane"], d["v_mps"], d["a_mps2"], debug_id=d["slot"])
            for d in rsu_detections(env.frame, env.builder.vehicle_ids) if d["lane"] >= 0]


def wm_propagator(model):
    """WM-S one-step belief propagator; CAVs hold speed, drivers at the encoder's population
    default (empty history: track slots are re-packed every tick, so no per-track history)."""
    from wms.core.constants import LONG_HISTORY_F

    from .belief import hold_speed_cav_controller

    controller = hold_speed_cav_controller()

    def propagate(state):
        k = state["s_m"].shape[1]
        kwargs = {}
        if getattr(model, "long_history", False):
            h = int(model.driver_history_len)
            kwargs = dict(driver_history=torch.zeros(1, k, h, LONG_HISTORY_F),
                          driver_history_mask=torch.zeros(1, k, h, dtype=torch.bool))
        with torch.no_grad():
            out = model.step(dict(state), cav_controller=controller, **kwargs)
        nxt = dict(state)
        nxt.update({"s_m": out["next_s_m"], "v_mps": out["next_v_mps"], "a_mps2": out["accel_mps2"],
                    "lane": out["next_lane"].long(), "present": out["next_present"].bool()})
        return nxt

    return propagate


class OracleSource:
    """Privileged: the env's true WM-frame state and its rolling history buffer."""

    name = "oracle"

    def __init__(self, model):
        self.model = model

    def observe(self, env) -> None:
        pass

    def inputs(self, env) -> tuple[dict, dict]:
        m = self.model
        state = {k: v for k, v in env.state().items()}
        hist, hmask = env.driver_history(None, history_len=int(m.history_len))
        out = {"state": state, "history": hist, "history_mask": hmask}
        if getattr(m, "long_history", False):
            out["driver_history"], out["driver_history_mask"] = env.driver_history(
                m.history_s, m.history_stride_s)
        present = state["present"][0]
        info = {"n_present": int(present.sum()),
                "n_cav": int((present & state["is_cav"][0]).sum()), "coverage": 1.0}
        return out, info


class BeliefSource:
    """RSU belief: CAV reports + detections + RSU zone sensors, WM-S propagation."""

    name = "belief"

    def __init__(self, model, *, propagator: str = "wms"):
        from .belief import BeliefConfig, BeliefTracker

        self.model = model
        self.tracker = BeliefTracker(BeliefConfig(), propagate=(
            wm_propagator(model) if propagator == "wms" else None))
        self.update_seconds: list[float] = []

    def observe(self, env) -> None:
        started = time.perf_counter()
        self.tracker.update(env.now, cav_reports_from_env(env), rsu_detections_from_env(env))
        self.update_seconds.append(time.perf_counter() - started)

    def inputs(self, env) -> tuple[dict, dict]:
        from .contract import RSU_MAINLINE_S_M

        m, tr = self.model, self.tracker
        state = tr.state()
        hist, hmask = tr.driver_history(int(m.history_len))
        out = {"state": state, "history": hist, "history_mask": hmask}
        if getattr(m, "long_history", False):
            raw = tr.history(history_s=m.history_s, history_stride_s=m.history_stride_s)
            out["driver_history"], out["driver_history_mask"] = long_history(
                raw, state["present"][0], history_s=m.history_s,
                history_stride_s=m.history_stride_s, dt_s=float(m.physics.dt_s))
        truth = env.state()
        true_present = truth["present"][0]
        cov = tr.coverage()
        zone = tr.coverage(zone=RSU_MAINLINE_S_M)
        info = {"n_present": int(state["present"][0].sum()),
                "n_cav": int((state["present"][0] & state["is_cav"][0]).sum()),
                "n_true_present": int(true_present.sum()),
                "n_true_cav": int((true_present & truth["is_cav"][0]).sum()),
                "coverage": cov["fraction_observed"], "coverage_by_lane": cov["fraction_by_lane"],
                "coverage_rsu_mainline": zone["fraction_observed"],
                "n_tracks": cov["n_tracks"], "mean_hdv_age_s": cov["mean_hdv_age_s"],
                "tracker_stats": dict(tr.stats)}
        return out, info


# ------------------------------------------------------------------ episode
def _load_world_models(paths, device, cav_exec_cap=None):
    from wms.eval.backend import load_backend

    return [load_backend(checked(p), device=device, cav_exec_cap=cav_exec_cap).model
            for p in paths]


def run(args) -> dict:
    """One episode; returns the deploy summary (also written to ``<out>/DEPLOY_SUMMARY.json``)."""
    from traffic_shepherd.c4.env import CavSumoEnv

    controller = ALIASES.get(args.controller, args.controller)
    out = checked(args.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output directory {out} exists and is not empty (use a new directory)")
    out.mkdir(parents=True, exist_ok=True)
    if args.threads is not None:          # default: torch's (OMP_NUM_THREADS; campaign sets 1)
        torch.set_num_threads(max(1, int(args.threads)))
    device = args.device
    notes: list[str] = []
    started = time.monotonic()
    policy_fn = models = checker = source = dwell = None
    exec_fns: dict[str, Callable] = {}
    streams, stream_noise = [], {}
    if controller in ("policy", "policy_check"):
        if not args.policy:
            raise SystemExit("--policy <checkpoint> is required for policy controllers")
        from .contract import CONTROL_DT_S
        from .lanes import LaneDwell
        from .policy import load_policy_fn
        from .rsu_check import NoiseStream, perturbed_group, zero_group

        policy_fn = load_policy_fn(str(checked(args.policy)), device=device)
        rules = policy_fn.lane_rules
        # One dwell clock and one command memory (last commanded accel / lane intent relayed in
        # the CAV reports) per episode, shared by every executed candidate (same CAVs).
        dwell = LaneDwell(rules.dwell_s, CONTROL_DT_S)
        memory = policy_fn.memory
        exec_fns["policy"] = policy_fn
        if controller == "policy_check":
            enc = policy_fn.group_encoder
            exec_fns["policy_group_0"] = make_exec_fn(policy_fn.policy, zero_group(enc), device=device,
                                                      lane_rules=rules, memory=memory)
            exec_fns["policy_group_1"] = make_exec_fn(policy_fn.policy,
                                                      perturbed_group(enc, args.group_perturbation),
                                                      device=device, lane_rules=rules, memory=memory)
            streams = [NoiseStream(args.checker_seed, k, k_full=args.k)
                       for k in range(args.n_policy_samples)]
            for k, stream in enumerate(streams):
                stream_noise[f"policy_sample_{k}"] = StreamNoise(stream)
                exec_fns[f"policy_sample_{k}"] = make_exec_fn(
                    policy_fn.policy, enc, device=device, lane_rules=rules, deterministic=False,
                    noise_fn=stream_noise[f"policy_sample_{k}"], memory=memory)
            # hold: a = 0 (never calls the policy); the policy is passed only for its stored accel
            # bound (safety ceiling, same as the imagined hold candidate; ACCELCAP_NOTES.md).
            exec_fns["hold_speed"] = make_exec_fn(policy_fn.policy, None, device=device, lane_rules=rules,
                                                  hold=True, memory=memory)
    if controller == "policy_check":
        from .contract import RewardWeights
        from .reward import team_reward
        from .rsu_check import RsuChecker

        if not args.wm_ckpt:
            raise SystemExit("--wm-ckpt is required for policy_check")
        models = _load_world_models(args.wm_ckpt, device, getattr(args, "cav_exec_cap", None))
        candidates = build_candidates(models[0], policy_fn.policy, policy_fn.group_encoder,
                                      lane_rules=policy_fn.lane_rules, sample_streams=streams,
                                      perturbation=args.group_perturbation)
        checker = RsuChecker(models, candidates, team_reward, RewardWeights(),
                             horizon_s=args.imagine_horizon, decision_period_s=args.decision_period,
                             n_driver_samples=args.n_driver_samples, margin=args.margin,
                             confidence_z=args.confidence_z, seed=args.checker_seed,
                             driver_noise_scale=args.driver_noise_scale,
                             embedding_noise_std=args.embedding_noise_std)
        source = (OracleSource(models[0]) if args.state == "oracle"
                  else BeliefSource(models[0], propagator=args.belief_propagator))

    env = CavSumoEnv(k_slots=args.k, ttc=not args.no_ttc, physics_log=not args.no_physics_log)
    env_mode = controller if controller in BASELINES else "direct"
    hook = None
    if isinstance(source, BeliefSource):
        lead_from = args.warmup - args.belief_lead

        def hook(e):
            if e.now >= lead_from - 1e-9:
                source.observe(e)

    meta = dict(policy=controller, deploy_state=args.state if controller == "policy_check" else None,
                policy_ckpt=args.policy, wm_ckpt=list(args.wm_ckpt or []), deploy="wms.control.deploy")
    decisions = (out / "decisions.jsonl").open("w", encoding="utf-8")
    segments = (out / "segments.jsonl").open("w", encoding="utf-8")
    choice_counts, executed_ticks, lane_commands = Counter(), Counter(), Counter()
    decision_wall, decision_ms, rollout_ms = [], [], []
    summary = None
    status, error = "failed", None
    try:
        state = env.reset(seed=args.seed, demand_vph=args.demand, penetration=args.penetration,
                          warmup_s=args.warmup, horizon_s=args.horizon, controller=env_mode,
                          episode_dir=out / "episode", meta=meta, warmup_hook=hook)
        choice, next_decision, segment = "policy", -math.inf, None
        while True:
            now = env.now
            ages = dwell.step(state) if dwell is not None else None
            if checker is not None and now >= next_decision - 1e-9:
                if segment is not None:
                    _dump(segments, segment)
                t0 = time.perf_counter()
                raw, src_info = source.inputs(env)
                if isinstance(source, OracleSource) and ages is not None:
                    # Same env slots: the true dwell clocks and the last commands / lane intents.
                    raw["state"]["lane_dwell_s"] = ages.clone()
                    reported = policy_fn.memory.augment(raw["state"])
                    raw["state"]["cmd_accel_mps2"] = reported["cmd_accel_mps2"]
                    raw["state"]["lane_intent"] = reported["lane_intent"]
                built = time.perf_counter() - t0
                inputs, keep = compact_inputs(raw) if not args.no_compact else (raw, None)
                inputs = {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict)
                              else v.to(device) if torch.is_tensor(v) else v)
                          for k, v in inputs.items()}
                st = inputs.pop("state")
                for stream in streams:
                    stream.begin(len(checker.decision_log), slot_ids=keep)
                choice, evidence = checker.decide(st, now_s=now, **inputs)
                if choice in stream_noise:
                    stream_noise[choice].begin()
                wall = time.perf_counter() - t0
                record = dict(checker.decision_log[-1])
                executed = choice
                record.update(oracle_state=args.state == "oracle", state_source=source.name,
                              executed=executed,
                              slots_imagined=int(st["s_m"].shape[1]), source=src_info,
                              wall_clock_s=wall, state_build_s=built,
                              sumo_time_s=now, window_time_s=now - args.warmup)
                _dump(decisions, record)
                choice_counts[choice] += 1
                decision_wall.append(wall)
                decision_ms.append(checker.last_timing["total_ms"])
                rollout_ms.append(checker.last_timing["rollout_ms"])
                print(f"[deploy] t={now:.0f} choice={choice} wall={wall:.1f}s "
                      f"slots={st['s_m'].shape[1]} source={source.name} "
                      f"coverage={src_info.get('coverage')}", flush=True)
                segment = {"decision": record["decision"], "start_s": now, "choice": choice,
                           "executed": executed, "ticks": 0, "interval": Counter(),
                           "sumo_cav_actions": Counter(), "lane_commands": Counter()}
                next_decision = now + args.decision_period
            accel = lanes = None
            if controller == "policy":
                executed = "policy"
            elif checker is not None:
                executed = choice
            else:
                executed = controller
            if executed in exec_fns:
                # Every policy-path candidate: direct accelerations + lane commands.
                accel, lanes = exec_fns[executed](state, with_lanes=True, dwell_age=ages)
            state, info = env.step(accel, lanes=lanes)
            executed_ticks[executed] += 1
            lane_commands.update(info.get("lane_commands", {}))
            if segment is not None:
                segment["ticks"] += 1
                segment["interval"].update(info["interval"])
                segment["sumo_cav_actions"].update(info["actions"].values())
                segment["lane_commands"].update(info.get("lane_commands", {}))
                segment["end_s"] = info["time_s"]
            if info["done"]:
                summary = info["summary"]
                break
            if source is not None:
                source.observe(env)
        if segment is not None:
            _dump(segments, segment)
        status = "complete"
    except Exception as exc:  # noqa: BLE001 - recorded in the summary, exit code 1
        error = f"{type(exc).__name__}: {exc}"
        notes.append(traceback.format_exc())
        try:
            env.close()
        except Exception:  # noqa: BLE001
            pass
    finally:
        decisions.close()
        segments.close()

    def stats(xs):
        return ({"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
                 "max": max(xs), "min": min(xs)} if xs else None)

    result = {
        "schema": "c4_deploy_summary/1", "status": status, "error": error,
        "controller": controller, "state_source": source.name if source is not None else None,
        "seed": args.seed, "demand_vph": args.demand, "penetration": args.penetration,
        "warmup_s": args.warmup, "horizon_s": args.horizon, "device": device,
        "threads": torch.get_num_threads(), "policy_ckpt": args.policy, "wm_ckpt": list(args.wm_ckpt or []),
        "cav_exec_cap": (getattr(args, "cav_exec_cap", None)
                         or __import__("os").environ.get("WMS_CAV_EXEC_CAP", "off")),
        "belief_ckpt_ignored": args.belief_ckpt,
        "check": None if checker is None else {
            "candidates": list(checker.candidates), "decision_period_s": args.decision_period,
            "imagine_horizon_s": args.imagine_horizon, "n_driver_samples": args.n_driver_samples,
            "driver_noise_scale": args.driver_noise_scale, "embedding_noise_std": args.embedding_noise_std,
            "ensemble_members": len(models), "margin": args.margin, "confidence_z": args.confidence_z,
            "compact_slots": not args.no_compact, "belief_lead_s": args.belief_lead,
            "belief_propagator": args.belief_propagator if args.state == "belief" else None,
            "candidate_kind": "direct_output_only", "group_perturbation": args.group_perturbation,
            "n_policy_samples": args.n_policy_samples},
        "decisions": sum(choice_counts.values()), "choice_counts": dict(choice_counts),
        "executed_control_ticks": dict(executed_ticks), "lane_commands": dict(lane_commands),
        "decision_wall_clock_s": stats(decision_wall), "decision_checker_ms": stats(decision_ms),
        "decision_rollout_ms": stats(rollout_ms),
        "belief_update_s": (stats(source.update_seconds) if isinstance(source, BeliefSource) else None),
        "belief_tracker_stats": (dict(source.tracker.stats) if isinstance(source, BeliefSource) else None),
        "episode_summary": summary, "collisions": None if summary is None else summary["collisions"],
        "teleports": None if summary is None else summary["teleports"],
        "wall_seconds": time.monotonic() - started, "finished_utc": _now(), "notes": notes}
    (out / "DEPLOY_SUMMARY.json").write_text(json.dumps(_jsonable(result), indent=1, allow_nan=False),
                                             encoding="utf-8")
    return result


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m wms.control.deploy",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--controller", required=True, choices=(*CONTROLLERS, *ALIASES))
    ap.add_argument("--seed", type=int, required=True, help="exposed C4 development seed (4107-4148)")
    ap.add_argument("--demand", "--loading", dest="demand", type=int, default=2700)
    ap.add_argument("--penetration", type=float, default=0.2)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--out", "--output", dest="out", required=True)
    ap.add_argument("--role", default="exposed", choices=("exposed",),
                    help="campaign compatibility; the env's seed gate is the authority")
    ap.add_argument("--policy", "--policy-ckpt", dest="policy")
    ap.add_argument("--wm-ckpt", nargs="+", help="frozen WM-S checkpoint(s) = imagination ensemble")
    ap.add_argument("--state", "--rsu-state", dest="state", choices=("oracle", "belief"), default="oracle")
    ap.add_argument("--belief-ckpt", help="accepted for campaign compatibility; unused (belief = tracker)")
    ap.add_argument("--belief-propagator", choices=("wms", "idm"), default="wms")
    ap.add_argument("--belief-lead", type=int, default=60,
                    help="seconds of warmup fed to the belief tracker (driver history needs 60 s)")
    ap.add_argument("--decision-period", type=float, default=5.0)
    ap.add_argument("--imagine-horizon", type=float, default=20.0)
    ap.add_argument("--n-driver-samples", type=int, default=1,
                    help="driver-setting samples per WM; >1 needs --driver-noise-scale > 0 "
                         "(uncalibrated encoder uncertainty, opt-in). Default: point estimate; "
                         "pass >=2 --wm-ckpt so the check has an ensemble spread")
    ap.add_argument("--driver-noise-scale", type=float, default=0.0)
    ap.add_argument("--embedding-noise-std", type=float, default=0.0)
    ap.add_argument("--margin", type=float, default=0.0, help="summed team-return units")
    ap.add_argument("--confidence-z", type=float, default=1.96)
    ap.add_argument("--checker-seed", type=int, default=0)
    ap.add_argument("--no-compact", action="store_true", help="imagine all K slots (slower)")
    ap.add_argument("--group-perturbation", type=float, default=0.1,
                    help="offset scale of the policy_group_1 candidate's group signal")
    ap.add_argument("--n-policy-samples", type=int, default=2,
                    help="stochastic policy_sample_k candidates (common random numbers)")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--k", type=int, default=192)
    ap.add_argument("--no-ttc", action="store_true")
    ap.add_argument("--no-physics-log", action="store_true")
    ap.add_argument("--cav-exec-cap", choices=("off", "idm", "idm_normal", "idm_normal:textbook",
                             "idm_normal:legacy"), default=None,
                    help="WM imagination execution cap of commanded CAVs (EXECCAP_NOTES.md); "
                         "default: env WMS_CAV_EXEC_CAP, else off")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.belief_lead < 0 or args.belief_lead > args.warmup:
        raise SystemExit("--belief-lead must lie in [0, warmup]")
    result = run(args)
    brief = {k: result[k] for k in ("status", "error", "controller", "state_source", "decisions",
                                    "choice_counts", "collisions", "teleports", "lane_commands",
                                    "wall_seconds")}
    brief["decision_wall_clock_s"] = result["decision_wall_clock_s"]
    print(json.dumps(_jsonable(brief)), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
