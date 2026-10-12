"""C4 deployment runner: one SUMO episode with a frozen or learned CAV controller.

    python -m wms.control.deploy --controller policy_check --state oracle \\
        --policy runs/active/C4-X/policy_best.pt --wm-ckpt <wm_s_best.pt> \\
        --seed 4107 --demand 2700 --penetration 0.2 --warmup 300 --horizon 600 --out runs/active/C4-EVAL-x/ep

Controllers:

* ``native | damping | pi | wall`` - the frozen baselines of :class:`traffic_shepherd.c4.env.CavSumoEnv`
  (identical to ``python -m traffic_shepherd.c4.collect --policy <name>``).
* ``policy`` - ``wms.control.policy.load_policy_fn(ckpt)`` every 1 s (``collect --policy <ckpt>`` path).
* ``policy_check`` - design 2.6 Option B: every ``decision_period_s`` the RSU imagines each candidate
  ``{policy, policy_group_0, native, damping, pi}`` in the frozen WM-S ensemble
  (:class:`wms.control.rsu_check.RsuChecker`, paired driver samples) from either the env's true WM-frame
  state + rolling history (``--state oracle``, privileged upper bound) or an RSU belief
  (``--state belief``: :class:`wms.control.belief.BeliefTracker` fed every control step with the CAV self
  reports + their front/rear sensor detections and the RSU zone detections, WM-S step propagator), then
  executes the chosen candidate in SUMO until the next decision.  Execution: ``policy*`` -> the learned
  policy's direct accelerations (zero group signal for ``policy_group_0``); ``native`` -> CAVs released to
  SUMO native; ``damping`` / ``pi`` -> the env's per-tick ``fallback="damping"`` (frozen CIRCLES) /
  ``fallback="pi"`` (a fresh frozen PI external-U session each time PI is selected; PI state does not
  survive a switch away).  In imagination ``native`` is the WM-S hold-speed teacher (RsuChecker default);
  damping/PI run the published controller code through ``wms.train.cav_drive.TimelineControllers``.

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
CANDIDATES = ("policy", "policy_group_0", "native", "damping", "pi")


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
class TimelineCandidate:
    """Fresh-per-imagination damping/PI callback: a constant action timeline replayed through
    ``wms.train.cav_drive.TimelineControllers`` (built lazily at the first call's [B,K])."""

    def __init__(self, model, controller: str, n_control: int, dependencies: tuple):
        self.model, self.controller = model, controller
        self.n_control, self.dependencies = int(n_control), dependencies
        self.inner = None

    def __call__(self, state):
        if self.inner is None:
            from wms.data.action_timeline import CONTROLLERS as TIMELINE
            from wms.train.cav_drive import TimelineControllers

            b, k = state["s_m"].shape
            ids = torch.full((b, self.n_control, k), TIMELINE.index(self.controller), dtype=torch.long)
            zeros = torch.zeros(b, self.n_control, k)
            self.inner = TimelineControllers(
                self.model, {"controller": ids, "acceleration_mps2": zeros,
                             "target_speed_mps": zeros}, dependencies=self.dependencies)
        return self.inner(state)


def classic_dependencies() -> tuple:
    """Frozen controller sources with ONE shared (stateless) CIRCLES ONNX session."""
    from wms.train.cav_drive import load_controller_dependencies

    deps = list(load_controller_dependencies(np.array([1, 2])))
    shared = deps[1](deps[0])
    deps[1] = lambda _config, _shared=shared: _shared
    return tuple(deps)


def build_candidates(model, policy, encoder, *, horizon_s: float, classic: bool = True,
                     log: Optional[list] = None) -> dict:
    """``{policy, policy_group_0, native[, damping, pi]}`` zero-argument factories."""
    from .bridge import make_cav_controller
    from .rsu_check import default_candidates
    from .safety import safe_accel

    n_control = int(math.ceil(horizon_s)) + 2
    damping = pi = None
    if classic:
        try:
            deps = classic_dependencies()

            def damping():
                return TimelineCandidate(model, "damping", n_control, deps)

            def pi():
                return TimelineCandidate(model, "pi-external-u", n_control, deps)
        except Exception as exc:  # noqa: BLE001 - recorded; candidates dropped, not faked
            if log is not None:
                log.append(f"classic imagination unavailable: {exc!r}")
    cands = default_candidates(model, policy=policy, group_encoder=encoder, safety=safe_accel,
                               bridge=make_cav_controller, damping=damping or (lambda: None),
                               pi=pi or (lambda: None), perturbation=0.0)
    if damping is None:
        cands.pop("damping")
        cands.pop("pi")
    return cands


def make_exec_fn(policy, encoder_fn, *, device="cpu") -> Callable:
    """SUMO execution callable ``state -> accel [1,K]`` (NaN on non-CAV slots), as load_policy_fn."""
    from .obs import build_observations
    from .rsu_field import build_rsu_field
    from .safety import safe_accel

    def act(state):
        with torch.no_grad():
            state = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in state.items()}
            obs = build_observations(state)
            signal = encoder_fn(obs, state["s_m"], build_rsu_field(state))
            accel, _ = policy(obs, signal, deterministic=True)
            accel = safe_accel(state, accel)
        out = accel.detach().cpu().numpy().astype(np.float64)
        out[~obs.cav_mask.cpu().numpy()] = np.nan
        return out

    return act


def zero_signal(encoder):
    def encode(*args, **kwargs):
        return torch.zeros_like(encoder(*args, **kwargs))
    return encode


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
def _load_world_models(paths, device):
    from wms.eval.backend import load_backend

    return [load_backend(checked(p), device=device).model for p in paths]


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
    policy_fn = models = checker = source = None
    exec_fns: dict[str, Callable] = {}
    if controller in ("policy", "policy_check"):
        if not args.policy:
            raise SystemExit("--policy <checkpoint> is required for policy controllers")
        from .policy import load_policy_fn

        policy_fn = load_policy_fn(str(checked(args.policy)), device=device)
        exec_fns["policy"] = policy_fn
        exec_fns["policy_group_0"] = make_exec_fn(policy_fn.policy, zero_signal(policy_fn.group_encoder),
                                                  device=device)
    if controller == "policy_check":
        from .contract import RewardWeights
        from .reward import team_reward
        from .rsu_check import RsuChecker

        if not args.wm_ckpt:
            raise SystemExit("--wm-ckpt is required for policy_check")
        models = _load_world_models(args.wm_ckpt, device)
        candidates = build_candidates(models[0], policy_fn.policy, policy_fn.group_encoder,
                                      horizon_s=args.imagine_horizon,
                                      classic=not args.no_classic_imagination, log=notes)
        checker = RsuChecker(models, candidates, team_reward, RewardWeights(),
                             horizon_s=args.imagine_horizon, decision_period_s=args.decision_period,
                             n_driver_samples=args.n_driver_samples, margin=args.margin,
                             confidence_z=args.confidence_z, seed=args.checker_seed)
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
    choice_counts, executed_ticks = Counter(), Counter()
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
            if checker is not None and now >= next_decision - 1e-9:
                if segment is not None:
                    _dump(segments, segment)
                t0 = time.perf_counter()
                raw, src_info = source.inputs(env)
                built = time.perf_counter() - t0
                inputs, keep = compact_inputs(raw) if not args.no_compact else (raw, None)
                inputs = {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict)
                              else v.to(device) if torch.is_tensor(v) else v)
                          for k, v in inputs.items()}
                st = inputs.pop("state")
                choice, evidence = checker.decide(st, now_s=now, **inputs)
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
                           "sumo_cav_actions": Counter()}
                next_decision = now + args.decision_period
            fallback, accel = "native", None
            if controller == "policy":
                accel, executed = policy_fn(state), "policy"
            elif checker is not None:
                executed = choice
                if executed in exec_fns:
                    accel = exec_fns[executed](state)
                elif executed in ("damping", "pi"):
                    fallback = executed
            else:
                executed = controller
            state, info = env.step(accel, fallback=fallback)
            executed_ticks[executed] += 1
            if segment is not None:
                segment["ticks"] += 1
                segment["interval"].update(info["interval"])
                segment["sumo_cav_actions"].update(info["actions"].values())
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
        "belief_ckpt_ignored": args.belief_ckpt,
        "check": None if checker is None else {
            "candidates": list(checker.candidates), "decision_period_s": args.decision_period,
            "imagine_horizon_s": args.imagine_horizon, "n_driver_samples": args.n_driver_samples,
            "ensemble_members": len(models), "margin": args.margin, "confidence_z": args.confidence_z,
            "compact_slots": not args.no_compact, "belief_lead_s": args.belief_lead,
            "belief_propagator": args.belief_propagator if args.state == "belief" else None,
            "native_imagination": "wm_hold_teacher"},
        "decisions": sum(choice_counts.values()), "choice_counts": dict(choice_counts),
        "executed_control_ticks": dict(executed_ticks),
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
    ap.add_argument("--n-driver-samples", type=int, default=4)
    ap.add_argument("--margin", type=float, default=0.0, help="summed team-return units")
    ap.add_argument("--confidence-z", type=float, default=1.96)
    ap.add_argument("--checker-seed", type=int, default=0)
    ap.add_argument("--no-compact", action="store_true", help="imagine all K slots (slower)")
    ap.add_argument("--no-classic-imagination", action="store_true",
                    help="drop the damping/pi candidates")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--k", type=int, default=192)
    ap.add_argument("--no-ttc", action="store_true")
    ap.add_argument("--no-physics-log", action="store_true")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.belief_lead < 0 or args.belief_lead > args.warmup:
        raise SystemExit("--belief-lead must lie in [0, warmup]")
    result = run(args)
    brief = {k: result[k] for k in ("status", "error", "controller", "state_source", "decisions",
                                    "choice_counts", "collisions", "teleports",
                                    "wall_seconds")}
    brief["decision_wall_clock_s"] = result["decision_wall_clock_s"]
    print(json.dumps(_jsonable(brief)), flush=True)
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
