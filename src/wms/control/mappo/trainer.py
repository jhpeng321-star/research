"""MAPPO trainer: iterations of universal SUMO episode collection + PPO updates.

Universal training: every episode samples its penetration (uniform over the configured set,
or continuous U(min, max) with ``penetration_mode = "uniform"``), its demand (uniform over
``demands_vph``), its warmup (``warmup_choices_s`` if given) and its seed (uniform over the
exposed C4 development block ``seeds``, guarded again by ``CavSumoEnv.reset``).

Budget accounting is dual-unit for the equal-budget comparison: ``controlled`` SUMO-seconds
(decision steps x 1 s), ``simulated`` SUMO-seconds (incl. warmup) and env-steps, each
cumulative and per iteration in ``log.jsonl``; ``charge_warmup`` selects which seconds count
against ``max_sumo_seconds``.  Failed episodes are retried once with a new seed; their
simulated seconds are logged as ``wasted_sim_seconds`` (not charged).

Spot-instance safe (loses at most the in-flight iteration): at the end of EVERY iteration a
full resumable checkpoint ``checkpoints/ckpt_XXXXX.pt`` (models, optimisers, Lagrange
multiplier, return scaler, budget counters, episode/seed cursor, numpy/torch/python RNG,
``log.jsonl`` byte offset, cumulative wall time) is written atomically (tmp + fsync + rename),
then the pointer ``checkpoints/LATEST`` (json: file, sha256, iteration) and ``status.json``.
The newest ``keep_last`` checkpoints are kept (plus every ``keep_every``-th).  Resume
(automatic when the output directory has checkpoints) takes the pointer target if its sha256
matches, else the newest checkpoint that loads (truncated files are skipped); log rows written
after that checkpoint are moved to ``log.orphan_*.jsonl`` and the log is truncated to the
recorded offset.  A ``STOP_REQUESTED`` file in the output directory (the Windows spot watcher
writes it) is checked between episodes, while waiting for workers and between PPO minibatches:
the in-flight iteration is abandoned, the workers are stopped and the run exits with status
``stopped_by_request`` (resume continues from the last checkpoint).  A ``STOP_REQUESTED`` found
at start-up is from a previous session and is renamed (consumed).

Deployable actor: ``policy_latest.pt`` (and ``policy_XXXXX.pt`` every ``keep_every``) in the
``c4_dreamer/3`` payload schema, loaded by ``wms.control.policy.load_policy_fn`` unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from wms.control.contract import CONTROL_DT_S, RewardWeights
from wms.control.lanes import LaneRules

from .agent import export_payload
from .buffer import EpisodeData, ReturnScaler, episode_gae, flatten_episodes
from .ppo import PPOConfig, PPOUpdater, update_lagrange
from .rollout import StopRequested, WorkerPool, run_job

DEFAULT_PENETRATIONS = (0.05, 0.1, 0.2, 0.4)
DEFAULT_DEMANDS_VPH = (2400, 2600, 2700)
DEFAULT_SEEDS = "4107-4148"
CHECKPOINT_FORMAT = "c4_mappo_trainer/2"
STOP_FILE = "STOP_REQUESTED"
_CKPT_RE = re.compile(r"^ckpt_(\d{5,})\.pt$")


def parse_seed_spec(text) -> tuple[int, ...]:
    if isinstance(text, (list, tuple)):
        return tuple(int(x) for x in text)
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    return tuple(out)


@dataclass
class RolloutConfig:
    seed: int = 1
    iterations: int = 1000
    episodes_per_iteration: int = 8
    horizon_s: float = 300.0
    warmup_s: float = 300.0
    warmup_choices_s: tuple[float, ...] = ()
    penetrations: tuple[float, ...] = DEFAULT_PENETRATIONS
    penetration_mode: str = "discrete"     # "discrete" (uniform over the set) | "uniform" (U(min, max))
    demands_vph: tuple[int, ...] = DEFAULT_DEMANDS_VPH
    seeds: str = DEFAULT_SEEDS
    max_sumo_seconds: float = 1.0e6
    max_env_steps: int | None = None
    charge_warmup: bool = False
    retries: int = 1                        # re-runs of a failed episode (new seed each)
    episode_timeout_s: float = 3600.0       # per-episode wall budget (deadline per worker wave)
    keep_last: int = 3                      # rolling per-iteration checkpoints kept
    keep_every: int = 10                    # permanently kept checkpoints / policy_XXXXX.pt exports
    update_threads: int = 0                 # torch threads for the update (0 = torch default)

    def __post_init__(self):
        if self.penetration_mode not in ("discrete", "uniform"):
            raise ValueError("penetration_mode must be 'discrete' or 'uniform'")
        if not self.penetrations or not self.demands_vph:
            raise ValueError("penetrations and demands_vph must be non-empty")
        if not parse_seed_spec(self.seeds):
            raise ValueError("seeds must name at least one seed")


@dataclass
class MappoConfig:
    ppo: PPOConfig = field(default_factory=PPOConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    lane_rules: LaneRules = field(default_factory=LaneRules)

    def to_dict(self) -> dict:
        return {"ppo": asdict(self.ppo), "rollout": asdict(self.rollout),
                "lane_rules": asdict(self.lane_rules)}


@dataclass
class Budget:
    max_sumo_seconds: float
    max_env_steps: int | None
    used_sumo_seconds: float = 0.0       # charged (controlled, + warmup if charge_warmup)
    used_env_steps: int = 0
    controlled_seconds: float = 0.0
    simulated_seconds: float = 0.0       # incl. warmup, successful episodes
    wasted_sim_seconds: float = 0.0      # failed episodes (not charged)

    def can_start(self, horizon_s: float, warmup_s: float, charge_warmup: bool) -> bool:
        seconds = horizon_s + (warmup_s if charge_warmup else 0.0)
        steps = int(round(horizon_s / CONTROL_DT_S))
        if self.used_sumo_seconds + seconds > self.max_sumo_seconds + 1e-9:
            return False
        if self.max_env_steps is not None and self.used_env_steps + steps > self.max_env_steps:
            return False
        return True

    def charge(self, episode: EpisodeData, warmup_s: float, charge_warmup: bool) -> None:
        self.used_sumo_seconds += episode.sumo_seconds + (warmup_s if charge_warmup else 0.0)
        self.used_env_steps += episode.n_steps
        self.controlled_seconds += episode.sumo_seconds
        self.simulated_seconds += episode.sumo_seconds + warmup_s

    def to_dict(self) -> dict:
        return asdict(self)


def _warn_accel_override(path, configured, loaded) -> None:
    """The checkpoint's stored actor bound wins (legacy 1.5 when it has none; ACCELCAP_NOTES.md)."""
    if configured is not None and loaded is not None and abs(configured - loaded) > 1e-9:
        print(f"WARNING: {path}: actor accel_max {loaded} from the checkpoint overrides the configured "
              f"{configured} (re-decoding needs an explicit policy.set_accel_bounds)",
              file=sys.stderr, flush=True)


def _torch_load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _atomic_save(payload, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        torch.save(payload, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _atomic_text(text: str, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _as_pytuple(state):
    return tuple((tuple(inner) if isinstance(inner, list) else inner) for inner in state)


class MappoTrainer:
    """Trains the C4 actor (CavPolicy + RsuGroupEncoder) with MAPPO in SUMO (or the fake env)."""

    def __init__(self, *, out_dir: Path | str, config: MappoConfig, policy, group_encoder, critic,
                 env_spec: Mapping, weights: RewardWeights | None = None,
                 meta: Mapping | None = None, workers: int = 1):
        self.out_dir = Path(out_dir)
        self.cfg = config
        self.policy, self.group_encoder, self.critic = policy, group_encoder, critic
        self.weights = weights or RewardWeights()
        self.meta = dict(meta or {})
        self.workers = int(workers)
        self.env_spec = dict(env_spec)
        self.ckpt_dir = self.out_dir / "checkpoints"
        self.log_path = self.out_dir / "log.jsonl"
        self.summary_path = self.out_dir / "RUN_SUMMARY.json"
        self.stop_path = self.out_dir / STOP_FILE
        self.updater = PPOUpdater(policy=policy, group_encoder=group_encoder, critic=critic,
                                  config=config.ppo)
        self.scaler = ReturnScaler()
        self.budget = Budget(config.rollout.max_sumo_seconds, config.rollout.max_env_steps)
        self.iteration = 0
        self.episode_index = 0
        self.episodes_done = 0
        self.failures = 0
        self.wall_s_total = 0.0
        self.rng = np.random.default_rng(config.rollout.seed)
        self.seeds = parse_seed_spec(config.rollout.seeds)
        self.pool: WorkerPool | None = None
        self.status = "initialized"
        self.last_diag: dict = {}
        self.last_checkpoint: dict = {}
        self.resume_info: dict = {}

    # ---------------------------------------------------------------- persistence
    def _spec(self) -> dict:
        return {**self.env_spec, "lane_rules": asdict(self.cfg.lane_rules),
                "reward_weights": asdict(self.weights)}

    def _checkpoint(self) -> dict:
        return {
            "format": CHECKPOINT_FORMAT,
            "iteration": self.iteration, "episode_index": self.episode_index,
            "episodes_done": self.episodes_done, "failures": self.failures,
            "policy": self.policy.state_dict(), "group_encoder": self.group_encoder.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_opt": self.updater.actor_opt.state_dict(),
            "critic_opt": self.updater.critic_opt.state_dict(),
            "lagrange_alpha": self.updater.alpha, "return_scaler": self.scaler.state_dict(),
            "budget": self.budget.to_dict(),
            "rng": {"numpy": self.rng.bit_generator.state, "torch": torch.get_rng_state(),
                    "python": random.getstate()},
            "log_offset": self.log_path.stat().st_size if self.log_path.is_file() else 0,
            "wall_s_total": self.wall_s_total,
            "meta": self.meta, "config": self.cfg.to_dict(),
        }

    def export_actor(self, path: Path) -> None:
        _atomic_save(export_payload(self.policy, self.group_encoder, self.cfg.lane_rules,
                                    reward_weights=asdict(self.weights),
                                    config=self.cfg.to_dict(), iteration=self.iteration,
                                    budget=self.budget.to_dict()), path)

    def _save(self) -> None:
        """Checkpoint of the finished iteration -> LATEST pointer -> prune -> actor -> status."""
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        name = f"ckpt_{self.iteration:05d}.pt"
        path = self.ckpt_dir / name
        _atomic_save(self._checkpoint(), path)
        self.last_checkpoint = {"file": f"checkpoints/{name}", "sha256": sha256_file(path),
                                "iteration": self.iteration, "bytes": path.stat().st_size}
        _atomic_text(json.dumps(self.last_checkpoint), self.ckpt_dir / "LATEST")
        self._prune()
        self.export_actor(self.out_dir / "policy_latest.pt")
        if self.iteration % max(1, int(self.cfg.rollout.keep_every)) == 0:
            self.export_actor(self.out_dir / f"policy_{self.iteration:05d}.pt")
        self._write_status()

    def _checkpoints(self) -> list[tuple[int, Path]]:
        """Own per-iteration checkpoints, newest first."""
        if not self.ckpt_dir.is_dir():
            return []
        found = []
        for path in self.ckpt_dir.iterdir():
            match = _CKPT_RE.match(path.name)
            if match and path.is_file():
                found.append((int(match.group(1)), path))
        return sorted(found, reverse=True)

    def _prune(self) -> None:
        """Keep the newest ``keep_last`` checkpoints and every ``keep_every``-th."""
        keep_last = max(1, int(self.cfg.rollout.keep_last))
        keep_every = max(1, int(self.cfg.rollout.keep_every))
        for rank, (it, path) in enumerate(self._checkpoints()):
            if rank >= keep_last and it % keep_every != 0:
                path.unlink(missing_ok=True)

    def _write_status(self, extra: dict | None = None) -> None:
        """Small ``status.json`` for a laptop-side puller (fetch only new files)."""
        files = {}
        for rel in ("log.jsonl", "policy_latest.pt", "RUN_SUMMARY.json"):
            path = self.out_dir / rel
            if path.is_file():
                st = path.stat()
                files[rel] = {"bytes": st.st_size, "mtime": st.st_mtime}
        row = {"status": self.status, "iteration": self.iteration,
               "episodes": self.episodes_done, "failed_episodes": self.failures,
               "sumo_seconds_charged": self.budget.used_sumo_seconds,
               "sumo_seconds_simulated": self.budget.simulated_seconds,
               "env_steps": self.budget.used_env_steps, "wall_s_total": self.wall_s_total,
               "last_checkpoint": self.last_checkpoint, "files": files,
               "init_from": self.meta.get("init_from"),
               "mean_episode_return": self.last_diag.get("mean_episode_return"),
               "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        row.update(extra or {})
        self.out_dir.mkdir(parents=True, exist_ok=True)
        _atomic_text(json.dumps(row, indent=1, default=float), self.out_dir / "status.json")

    def _load_valid_checkpoint(self, skipped: list) -> tuple[dict, Path] | None:
        """LATEST pointer target if its sha256 matches, else the newest loadable checkpoint."""
        candidates: list[tuple[Path, str | None]] = []
        try:
            info = json.loads((self.ckpt_dir / "LATEST").read_text(encoding="utf-8"))
            candidates.append((self.out_dir / info["file"], info.get("sha256")))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        seen = {p.resolve() for p, _ in candidates}
        candidates += [(p, None) for _, p in self._checkpoints() if p.resolve() not in seen]
        for path, want_sha in candidates:
            if not path.is_file():
                continue
            if want_sha and sha256_file(path) != want_sha:
                skipped.append({"file": path.name, "reason": "sha256 mismatch"})
                continue
            try:
                state = _torch_load(path)
            except Exception as exc:  # noqa: BLE001 - truncated / corrupt: try the next one
                skipped.append({"file": path.name, "reason": type(exc).__name__})
                continue
            if isinstance(state, dict) and state.get("format") == CHECKPOINT_FORMAT:
                return state, path
            skipped.append({"file": path.name, "reason": "not a trainer checkpoint"})
        return None

    def _truncate_log(self, offset: int) -> dict | None:
        """Move log rows written after the checkpoint to an orphan file; truncate the log."""
        if not self.log_path.is_file() or self.log_path.stat().st_size <= offset:
            return None
        with self.log_path.open("rb") as fh:
            fh.seek(offset)
            tail = fh.read()
        orphan = self.out_dir / f"log.orphan_{self.iteration:05d}_{int(time.time())}.jsonl"
        orphan.write_bytes(tail)
        with self.log_path.open("r+b") as fh:
            fh.truncate(offset)
        return {"orphan_file": orphan.name, "orphan_bytes": len(tail)}

    def _try_resume(self) -> bool:
        skipped: list[dict] = []
        found = self._load_valid_checkpoint(skipped)
        if found is None:
            self.resume_info = {"skipped": skipped} if skipped else {}
            return False
        state, path = found
        self.iteration = state["iteration"]
        self.episode_index = state["episode_index"]
        self.episodes_done = state["episodes_done"]
        self.failures = state.get("failures", 0)
        configured = getattr(self.policy, "accel_max", None)
        self.policy.load_state_dict(state["policy"])
        _warn_accel_override(path, configured, getattr(self.policy, "accel_max", None))
        self.group_encoder.load_state_dict(state["group_encoder"])
        self.critic.load_state_dict(state["critic"])
        self.updater.actor_opt.load_state_dict(state["actor_opt"])
        self.updater.critic_opt.load_state_dict(state["critic_opt"])
        self.updater.alpha = state["lagrange_alpha"]
        self.scaler = ReturnScaler.from_state(state["return_scaler"])
        self.budget = Budget(**state["budget"])
        self.rng.bit_generator.state = state["rng"]["numpy"]
        torch.set_rng_state(state["rng"]["torch"].to(torch.uint8))
        random.setstate(_as_pytuple(state["rng"]["python"]))
        self.wall_s_total = float(state.get("wall_s_total", 0.0))
        self.last_checkpoint = {"file": f"checkpoints/{path.name}", "sha256": sha256_file(path),
                                "iteration": self.iteration, "bytes": path.stat().st_size}
        self.resume_info = {"checkpoint": path.name, "skipped": skipped,
                            "log": self._truncate_log(int(state.get("log_offset", 0)))}
        return True

    def init_from_checkpoint(self, path: Path | str, lagrange: str = "carry") -> dict:
        """Warm start a FRESH run from existing weights (``--init-from``).

        ``path`` is a trainer checkpoint (``checkpoints/ckpt_XXXXX.pt``: policy, group_encoder
        and critic are loaded) or an exported actor (``policy_*.pt``: policy / group_encoder
        only, the critic stays freshly initialised).  Weights are loaded strictly.  Optimisers,
        iteration / episode counters, budget, RNG and the return scaler stay fresh.  The Lagrange
        multiplier: ``carry`` takes the checkpoint's ``lagrange_alpha`` (an exported actor has
        none -> fresh), ``reset`` keeps the freshly initialised one.  Refuses a run directory
        that already has checkpoints.  The record goes into ``self.meta["init_from"]`` (config
        log row, RUN_SUMMARY, checkpoints) and is returned."""
        if lagrange not in ("carry", "reset"):
            raise ValueError("lagrange must be 'carry' or 'reset'")
        if self._checkpoints() or (self.ckpt_dir / "LATEST").exists():
            raise RuntimeError(f"--init-from needs a fresh run directory; {self.ckpt_dir} "
                               "already has checkpoints")
        path = Path(path)
        state = _torch_load(path)
        if not isinstance(state, dict) or "policy" not in state or "group_encoder" not in state:
            raise ValueError(f"{path}: neither a trainer checkpoint nor an exported actor")
        is_trainer = state.get("format") == CHECKPOINT_FORMAT and "critic" in state
        configured = getattr(self.policy, "accel_max", None)
        self.policy.load_state_dict(state["policy"], strict=True)
        _warn_accel_override(path, configured, getattr(self.policy, "accel_max", None))
        self.group_encoder.load_state_dict(state["group_encoder"], strict=True)
        if is_trainer:
            self.critic.load_state_dict(state["critic"], strict=True)
        alpha_before = float(self.updater.alpha)
        if lagrange == "carry" and is_trainer and "lagrange_alpha" in state:
            self.updater.alpha = float(state["lagrange_alpha"])
            alpha_note = "carried from checkpoint"
        elif lagrange == "carry":
            alpha_note = "exported actor has no multiplier: fresh"
        else:
            alpha_note = "reset to fresh"
        record = {"path": str(path), "sha256": sha256_file(path),
                  "kind": "trainer_checkpoint" if is_trainer else "exported_actor",
                  "source_iteration": state.get("iteration"),
                  "critic": "loaded" if is_trainer else "fresh",
                  "optimizers": "fresh", "return_scaler": "fresh", "iteration_counter": 0,
                  "lagrange": lagrange, "lagrange_alpha_fresh": alpha_before,
                  "lagrange_alpha_source": (float(state["lagrange_alpha"])
                                            if is_trainer and "lagrange_alpha" in state else None),
                  "lagrange_alpha_used": float(self.updater.alpha), "lagrange_note": alpha_note}
        self.meta["init_from"] = record
        return record

    def _log_line(self, row: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=float) + "\n")

    def _write_summary(self) -> None:
        row = {"status": self.status, "iterations_completed": self.iteration,
               "episodes": self.episodes_done, "failed_episodes": self.failures,
               "budget": self.budget.to_dict(), "checkpoint": self.last_checkpoint,
               "actor": "policy_latest.pt",
               "actor_load": "wms.control.policy.load_policy_fn (c4_dreamer/3 payload, trainer=mappo)",
               "workers": self.workers, "meta": self.meta, "config": self.cfg.to_dict(),
               "last": self.last_diag, "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _atomic_text(json.dumps(row, indent=1, default=float), self.summary_path)

    # ---------------------------------------------------------------- sampling
    def _sample_case(self) -> dict:
        r = self.cfg.rollout
        if r.penetration_mode == "uniform":
            pen = float(self.rng.uniform(min(r.penetrations), max(r.penetrations)))
        else:
            pen = float(self.rng.choice(np.asarray(r.penetrations, dtype=float)))
        warmup = (float(self.rng.choice(np.asarray(r.warmup_choices_s, dtype=float)))
                  if r.warmup_choices_s else float(r.warmup_s))
        return {"seed": int(self.rng.choice(self.seeds)),
                "demand_vph": int(self.rng.choice(np.asarray(r.demands_vph, dtype=int))),
                "penetration": pen, "warmup_s": warmup, "horizon_s": float(r.horizon_s)}

    def _new_job(self, attempt: int = 0, avoid_seed: int | None = None) -> dict:
        case = self._sample_case()
        if avoid_seed is not None and len(self.seeds) > 1:
            while case["seed"] == avoid_seed:
                case["seed"] = int(self.rng.choice(self.seeds))
        job = {**case, "episode_index": self.episode_index, "attempt": attempt,
               "noise_seed": self.cfg.rollout.seed * 1_000_003 + self.episode_index}
        self.episode_index += 1
        return job

    def _plan_jobs(self) -> list[dict]:
        r = self.cfg.rollout
        jobs = []
        warm = max(r.warmup_choices_s) if r.warmup_choices_s else r.warmup_s
        for _ in range(r.episodes_per_iteration):
            planned = r.horizon_s * len(jobs) + (warm * len(jobs) if r.charge_warmup else 0.0)
            probe = Budget(self.budget.max_sumo_seconds, self.budget.max_env_steps,
                           self.budget.used_sumo_seconds + planned,
                           self.budget.used_env_steps + int(round(r.horizon_s)) * len(jobs))
            if not probe.can_start(r.horizon_s, warm, r.charge_warmup):
                break
            jobs.append(self._new_job())
        return jobs

    # ---------------------------------------------------------------- rollout
    def stop_requested(self) -> bool:
        return self.stop_path.exists()

    def _check_stop(self) -> None:
        if self.stop_requested():
            raise StopRequested(str(self.stop_path))

    def _weights_payload(self) -> dict:
        return {"policy_state": {k: v.detach().cpu().clone() for k, v in self.policy.state_dict().items()},
                "encoder_state": {k: v.detach().cpu().clone()
                                  for k, v in self.group_encoder.state_dict().items()}}

    def _execute(self, jobs: list[dict]) -> list[dict]:
        weights = self._weights_payload()
        payloads = [{**job, **weights} for job in jobs]
        if self.workers > 1:
            if self.pool is None:
                self.pool = WorkerPool(self.workers, self._spec())
            waves = -(-len(jobs) // self.workers)
            return self.pool.run(payloads, deadline_s=self.cfg.rollout.episode_timeout_s * waves,
                                 stop_check=self.stop_requested)
        spec = self._spec()
        results = []
        for payload in payloads:
            self._check_stop()
            results.append(run_job({**payload, "spec": spec}))
        return results

    def _collect(self) -> tuple[list[EpisodeData], list[dict]]:
        jobs = self._plan_jobs()
        if not jobs:
            return [], []
        episodes, failures = [], []
        attempts = {job["episode_index"]: 0 for job in jobs}
        while jobs:
            results = self._execute(jobs)
            retry = []
            for job, res in zip(jobs, results):
                if res.get("ok"):
                    episodes.append(res["episode"])
                    continue
                self.failures += 1
                record = {"type": "failure", "iteration": self.iteration + 1,
                          "error": res.get("error"), "job": res.get("job", {}),
                          "traceback": res.get("traceback")}
                failures.append(record)
                self.budget.wasted_sim_seconds += float(job["warmup_s"]) + float(job["horizon_s"])
                n = attempts.get(job["episode_index"], 0)
                if n < self.cfg.rollout.retries:
                    new = self._new_job(attempt=n + 1, avoid_seed=job["seed"])
                    new["retry_of"] = job["episode_index"]
                    attempts[new["episode_index"]] = n + 1
                    record["retry"] = {k: new[k] for k in ("episode_index", "seed", "demand_vph",
                                                            "penetration")}
                    retry.append(new)
                self._log_line(record)
            jobs = retry
        return episodes, failures

    # ---------------------------------------------------------------- main loop
    def close(self, kill: bool = False) -> None:
        if self.pool is not None:
            if kill:
                self.pool.kill()
            else:
                self.pool.close()
            self.pool = None

    def run(self, max_iterations: int | None = None) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        consumed = None
        if self.stop_path.exists():         # left by a previous (reclaimed) session
            consumed = self.stop_path.with_name(f"{STOP_FILE}.consumed_{int(time.time())}")
            os.replace(self.stop_path, consumed)
        resumed = self._try_resume()
        r = self.cfg.rollout
        if r.update_threads:
            torch.set_num_threads(int(r.update_threads))
        if resumed:
            self._log_line({"type": "resume", "iteration": self.iteration,
                            "budget": self.budget.to_dict(), "workers": self.workers,
                            **self.resume_info,
                            "consumed_stop_file": consumed.name if consumed else None})
        else:
            random.seed(r.seed)
            np.random.seed(r.seed % (2**32))
            torch.manual_seed(r.seed)
            self._log_line({"type": "config", **self.cfg.to_dict(), "meta": self.meta,
                            "workers": self.workers, **self.resume_info})
        target = self.iteration + (max_iterations or r.iterations)
        self.status = "running"
        try:
            while self.iteration < target:
                self._check_stop()
                self._iterate_once(time.monotonic())
                if self.status != "running":
                    break
        except StopRequested:
            self.status = "stopped_by_request"
            self.close(kill=True)
            self._log_line({"type": "stopped", "iteration": self.iteration,
                            "resume_from": self.last_checkpoint, "reason": STOP_FILE})
        finally:
            if self.status == "running":
                self.status = "completed"
            self._write_summary()
            self._write_status({"stopped": self.status == "stopped_by_request"})
            self.close()
        return {"status": self.status, "iterations_completed": self.iteration,
                "budget": self.budget.to_dict()}

    def _iterate_once(self, t0: float) -> None:
        r, p = self.cfg.rollout, self.cfg.ppo
        episodes, failures = self._collect()
        if not episodes:
            self.status = "budget_exhausted" if not failures else "all_episodes_failed"
            return
        t_collect = time.monotonic() - t0
        self._check_stop()
        for ep in episodes:
            self.budget.charge(ep, float(ep.meta.get("warmup_s", r.warmup_s)), r.charge_warmup)
            self.episodes_done += 1
        batch = flatten_episodes(episodes)
        v_norm, vf_norm = self.updater.values(batch)
        values_raw = self.scaler.unscale(v_norm) if p.return_norm else v_norm
        finals_raw = self.scaler.unscale(vf_norm) if p.return_norm else vf_norm
        adv, ret, cost_ret = episode_gae(batch, values_raw, finals_raw, gamma=p.gamma,
                                         lam=p.gae_lambda)
        self.policy.train()
        self.group_encoder.train()
        self.critic.train()
        noise = bool(p.grad_noise_every) and (self.iteration % int(p.grad_noise_every) == 0)
        t1 = time.monotonic()
        diag = self.updater.update(batch, advantages=adv, returns=ret, cost_returns=cost_ret,
                                   scaler=self.scaler, grad_noise=noise,
                                   stop_check=self.stop_requested)
        t_update = time.monotonic() - t1
        if p.return_norm:
            self.scaler.update(ret)
        steps = sum(ep.n_steps for ep in episodes)
        mean_cost = float(batch.costs.sum()) / max(1, steps)
        self.updater.alpha = update_lagrange(self.updater.alpha, mean_cost, p.cost_limit_per_step,
                                             p.lagrange_lr, p.lagrange_max)
        self.iteration += 1
        for ep in episodes:
            self._log_line({"type": "episode", "iteration": self.iteration,
                            "return": float(ep.rewards.sum()), "steps": ep.n_steps,
                            "mean_cost": float(ep.costs.mean()) if ep.n_steps else 0.0,
                            **{k: float(v.sum()) for k, v in ep.components.items()},
                            **{k: v for k, v in ep.meta.items()}})
        rates = [ep.meta["sim_s_per_wall_s"] for ep in episodes if ep.meta.get("sim_s_per_wall_s")]
        decisions = sum(ep.meta.get("cav_decisions", 0) for ep in episodes)
        lane_issued = sum(ep.meta.get("lane_commands", {}).get("issued", 0) for ep in episodes)
        lane_done = sum(ep.meta.get("lane_commands", {}).get("completed", 0) for ep in episodes)
        self.wall_s_total += time.monotonic() - t0
        self.last_diag = {
            "type": "iter", "iteration": self.iteration, "episodes": len(episodes),
            "failed_episodes": len(failures), "env_steps": steps,
            "controlled_s": steps * CONTROL_DT_S,
            "simulated_s": sum(ep.meta.get("sim_s", 0.0) for ep in episodes),
            "budget": self.budget.to_dict(),
            "sumo_seconds_used": self.budget.used_sumo_seconds,
            "mean_episode_return": float(np.mean([float(ep.rewards.sum()) for ep in episodes])),
            "mean_reward_per_step": float(batch.rewards.mean()),
            "mean_cost_per_step": mean_cost,
            **{f"mean_{k}_per_step": float(torch.cat([ep.components[k] for ep in episodes]).mean())
               for k in episodes[0].components},
            "collisions": sum(int(ep.meta.get("collisions") or 0) for ep in episodes),
            "teleports": sum(int(ep.meta.get("teleports") or 0) for ep in episodes),
            "policy_lane_change_rate": (sum(ep.meta.get("policy_lane_changes", 0) for ep in episodes)
                                        / max(1, decisions)),
            "lane_commands_issued": lane_issued, "lane_commands_completed": lane_done,
            "penetrations": [ep.meta.get("penetration") for ep in episodes],
            "demands": [ep.meta.get("demand_vph") for ep in episodes],
            "collect_wall_s": t_collect, "update_wall_s": t_update,
            "iteration_wall_s": time.monotonic() - t0, "wall_s_total": self.wall_s_total,
            "sim_s_per_wall_s_per_worker": float(np.mean(rates)) if rates else None,
            "pool_restarts": self.pool.restarts if self.pool is not None else 0,
            **diag,
        }
        self._log_line(self.last_diag)
        self._save()
        self._write_summary()
