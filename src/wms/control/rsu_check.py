"""Oracle-state deployment check: batched, paired WM-S counterfactuals.

Controller factories take no arguments and return fresh contract CavControllers.
Factories must isolate mutable controller state and be deterministic; the learned
bridge can be supplied through :func:`default_candidates`. No simulation imports,
checkpoint loading, or training are performed here.
"""
from __future__ import annotations

import math
from time import perf_counter
from typing import Callable, Mapping, Optional

import torch
from torch import Tensor

from .contract import CONTROL_DT_S, PHYSICS_DT_S, CavController, RewardWeights

ControllerFactory = Callable[[], CavController]


def default_candidates(model, *, policy, group_encoder, safety, bridge,
                       damping: ControllerFactory, pi: ControllerFactory,
                       alternative_group_encoders=(), perturbation: float = 0.1):
    """Build policy, policy_group_k, native, damping and pi factories.

    ``bridge(policy, group_encoder, safety)`` is injected (make_cav_controller).
    The learned actor is forced deterministic. Group alternatives include zero,
    a fixed perturbation (unless perturbation=0), and injected encoders from other
    zones. Frozen classic controller adapters are injected as fresh factories.
    """
    if not math.isfinite(perturbation) or perturbation < 0:
        raise ValueError("perturbation must be finite and nonnegative")

    def actor(obs, signal, **kwargs):
        return policy(obs, signal, **{**kwargs, "deterministic": True})

    def factory(encoder):
        return lambda: bridge(actor, encoder, safety)

    def zero(*args, **kwargs):
        return torch.zeros_like(group_encoder(*args, **kwargs))

    def perturbed(*args, **kwargs):
        signal = group_encoder(*args, **kwargs)
        offset = torch.arange(signal.shape[-1], device=signal.device,
                              dtype=signal.dtype).cos() * perturbation
        return signal + offset

    encoders = [zero]
    if perturbation:
        encoders.append(perturbed)
    encoders.extend(alternative_group_encoders)
    result = {"policy": factory(group_encoder)}
    result.update({f"policy_group_{k}": factory(encoder)
                   for k, encoder in enumerate(encoders)})
    result.update(native=lambda: model.hold_teacher, damping=damping, pi=pi)
    return result


def _repeat(value: Optional[Tensor], count: int) -> Optional[Tensor]:
    if value is None:
        return None
    if value.ndim == 0 or value.shape[0] != 1:
        raise ValueError("state/history tensors must have batch size 1")
    return value.detach().repeat(count, *([1] * (value.ndim - 1)))


def _slice_state(state, start, stop, batch_size):
    return {key: (value[start:stop] if torch.is_tensor(value) and value.ndim
                  and value.shape[0] == batch_size else value)
            for key, value in state.items()}


def _number(value):
    value = float(value)
    return value if math.isfinite(value) else None


class RsuChecker:
    """Keep policy unless a paired improvement clears margin and positive LCB.

    Each ensemble member receives ONE [candidates*samples,K] rollout. Rows are
    candidate-major, sample-minor. The same standard-normal driver draws are
    reused for all candidates and ensemble members; member-specific encodings
    and uncertainty scales remain distinct. The LCB is mean - confidence_z*SEM
    over the ensemble x driver-sample paired differences (not unpaired returns).
    """

    def __init__(self, world_models: list, candidates: Mapping[str, ControllerFactory],
                 reward_fn, weights: RewardWeights, horizon_s: float = 20,
                 decision_period_s: float = 5, n_driver_samples: int = 4,
                 margin: float = 0.0, *, confidence_z: float = 1.96,
                 driver_noise_scale: float = 1.0, embedding_noise_std: float = 0.05,
                 seed: int = 0, log_fn: Optional[Callable] = None,
                 synchronize: Optional[Callable[[], None]] = None):
        if not world_models:
            raise ValueError("world_models must not be empty")
        if "policy" not in candidates or not all(callable(f) for f in candidates.values()):
            raise ValueError("candidates must include a policy and callable factories")
        if (not math.isfinite(horizon_s) or not 10 <= horizon_s <= 30
                or not math.isfinite(decision_period_s) or decision_period_s <= 0):
            raise ValueError("horizon must be 10..30 s and decision period positive")
        if isinstance(n_driver_samples, bool) or not isinstance(n_driver_samples, int) \
                or n_driver_samples < 1:
            raise ValueError("n_driver_samples must be a positive integer")
        for name, value in (("margin", margin), ("confidence_z", confidence_z),
                            ("driver_noise_scale", driver_noise_scale),
                            ("embedding_noise_std", embedding_noise_std)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self.world_models = list(world_models)
        self.candidates = dict(candidates)
        self.reward_fn, self.weights = reward_fn, weights
        self.horizon_s, self.decision_period_s = horizon_s, decision_period_s
        self.n_driver_samples, self.margin = n_driver_samples, margin
        self.confidence_z = confidence_z
        self.driver_noise_scale, self.embedding_noise_std = driver_noise_scale, embedding_noise_std
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.log_fn, self.synchronize = log_fn, synchronize
        self.decision_log: list[dict] = []
        self._chosen = None
        self._controller = None
        self._next_decision_s = None
        self._clock_s = 0.0
        self._last_time_s = None
        self._execution_step = 0
        self._execution_start_s = 0.0
        self.last_timing: dict = {}

    def _normal(self, shape, reference, cache, key):
        if key not in cache:
            cache[key] = torch.randn(shape, generator=self.generator)
        if tuple(cache[key].shape) != tuple(shape):
            raise ValueError("ensemble driver encodings must have matching shapes/keys")
        return cache[key].to(reference)

    def _sample_drivers(self, model, state, history_args, count, noise):
        encoded, embedding, params = model.encode_drivers(dict(state), **history_args)
        sampled = {}
        if params is not None:
            names = tuple(params)
            values = torch.stack([params[name] for name in names], -1)
            draws = self._normal((self.n_driver_samples, *values.shape[1:]),
                                 values, noise, ("parameters", names))
            # Physical-unit encoder uncertainty; cap excursions to +/-50% of the
            # valid encoded value so positive IDM parameters stay positive.
            scale = encoded["uncertainty"] if encoded is not None else values.abs() * 0.1
            samples = values + draws * scale * self.driver_noise_scale
            samples = torch.maximum(torch.minimum(samples, values * 1.5), values * 0.5)
            samples = samples.repeat(count, 1, 1)
            sampled["driver_params"] = {name: samples[..., k] for k, name in enumerate(names)}
        if embedding is not None:
            # Perturb the embedding as well: hybrid residuals and history lane
            # heads condition on it, whereas residual-only WMs ignore parameters.
            draws = self._normal((self.n_driver_samples, *embedding.shape[1:]),
                                 embedding, noise, "embedding")
            samples = embedding + draws * self.embedding_noise_std * self.driver_noise_scale
            sampled["driver_embedding"] = samples.repeat(count, 1, 1)
        if not sampled:
            raise ValueError("world model must expose driver parameters or embedding")
        return sampled

    def _returns(self, state, rollout, n_steps):
        current = dict(state)
        total = torch.zeros(state["s_m"].shape[0], device=state["s_m"].device, dtype=torch.float64)
        for step in range(n_steps):
            accel = rollout["accel"][:, step]
            nxt = {**current, "s_m": rollout["positions"][:, step],
                   "v_mps": rollout["speeds"][:, step],
                   "lane": rollout["lanes"][:, step],
                   "present": rollout["masks"][:, step], "a_mps2": accel}
            nxt.pop("veh_kin", None)
            if "entrant_seq" in rollout:
                nxt["entrant"] = rollout["entrant_seq"][:, step]
            reward = self.reward_fn(current, nxt, accel, self.weights)["reward"]
            if reward.shape != total.shape:
                raise ValueError("reward_fn must return reward [batch]")
            total = total + reward.double()
            current = nxt
        return total

    @torch.no_grad()
    def decide(self, state, history=None, *, history_mask=None, driver_history=None,
               driver_history_mask=None, now_s=None):
        """Return (chosen name, JSON-safe per-candidate mean/std/paired evidence)."""
        if state["s_m"].ndim != 2 or state["s_m"].shape[0] != 1:
            raise ValueError("decide requires a single oracle state [1,K]")
        if now_s is not None and not math.isfinite(now_s):
            raise ValueError("now_s must be finite")
        if self.synchronize:
            self.synchronize()
        started = perf_counter()
        names = list(self.candidates)
        count, samples = len(names), self.n_driver_samples
        batch_size = count * samples
        batched_state = {key: (_repeat(value, batch_size) if torch.is_tensor(value)
                               and value.ndim else value)
                         for key, value in state.items()}
        history_args = dict(history=history, history_mask=history_mask,
                            driver_history=driver_history, driver_history_mask=driver_history_mask)
        # An explicit history without a mask is all observed; do not silently
        # discard it in WM-S's encoder (which otherwise treats it as missing).
        for hist, mask in (("history", "history_mask"), ("driver_history", "driver_history_mask")):
            if history_args[hist] is not None and history_args[mask] is None:
                history_args[mask] = torch.ones_like(history_args[hist][..., 0], dtype=torch.bool)
        batched_history = {key: _repeat(value, batch_size) for key, value in history_args.items()}
        noise, member_returns = {}, []
        rollout_seconds = 0.0
        for model in self.world_models:
            dt = float(getattr(getattr(model, "physics", None), "dt_s", PHYSICS_DT_S))
            steps = round(self.horizon_s / dt) if dt > 0 and math.isfinite(dt) else 0
            if steps < 1 or not math.isclose(steps * dt, self.horizon_s, abs_tol=1e-6):
                raise ValueError("horizon_s must be an integer number of WM physics steps")
            controllers = [self.candidates[name]() for name in names]

            def dispatch(predicted):
                motions = []
                for index, controller in enumerate(controllers):
                    subset = _slice_state(predicted, index * samples, (index + 1) * samples,
                                          batch_size)
                    motion = controller(subset)
                    if motion.shape != (*subset["s_m"].shape, 4):
                        raise ValueError("CavController must return [batch,K,4]")
                    motions.append(motion)
                return torch.cat(motions, dim=0)

            # Preserve the caller's train/eval lifecycle, including mixed submodule modes.
            modes = [(module, module.training) for module in model.modules()] \
                if isinstance(model, torch.nn.Module) else []
            try:
                if modes:
                    model.eval()
                drivers = self._sample_drivers(model, state, history_args, count, noise)
                if self.synchronize:
                    self.synchronize()
                rollout_start = perf_counter()
                rollout = model.rollout(dict(batched_state), n_steps=steps, cav_teacher=None,
                                        cav_controller=dispatch, **batched_history, **drivers)
                if self.synchronize:
                    self.synchronize()
                rollout_seconds += perf_counter() - rollout_start
                member_returns.append(self._returns(batched_state, rollout, steps).reshape(count, samples))
            finally:
                for module, training in modes:
                    module.training = training
        returns = torch.stack(member_returns).double().cpu()  # [ensemble,candidate,sample]
        paired = returns - returns[:, names.index("policy"):names.index("policy") + 1]
        evidence = {}
        chosen, best = "policy", -math.inf
        for index, name in enumerate(names):
            values, diffs = returns[:, index].flatten(), paired[:, index].flatten()
            finite = bool(torch.isfinite(values).all() and torch.isfinite(diffs).all())
            mean, std = values.mean(), values.std(unbiased=values.numel() > 1)
            paired_mean, paired_std = diffs.mean(), diffs.std(unbiased=diffs.numel() > 1)
            lcb = (float(paired_mean - self.confidence_z * paired_std / math.sqrt(diffs.numel()))
                   if diffs.numel() > 1 and finite else -math.inf)
            eligible = name != "policy" and finite and float(paired_mean) > self.margin and lcb > 0
            evidence[name] = {"mean": _number(mean), "std": _number(std),
                              "paired_mean": _number(paired_mean), "paired_std": _number(paired_std),
                              "paired_lcb": _number(lcb), "eligible": eligible,
                              "returns": [[_number(x) for x in row] for row in returns[:, index]],
                              "paired_diffs": [[_number(x) for x in row] for row in paired[:, index]]}
            if eligible and float(mean) > best:
                chosen, best = name, float(mean)
        self.last_timing = {"total_ms": (perf_counter() - started) * 1000,
                            "rollout_ms": rollout_seconds * 1000, "batch_size": batch_size,
                            "slots": state["s_m"].shape[1], "ensemble_members": len(self.world_models),
                            "steps_per_member": steps, "device": str(state["s_m"].device)}
        record = {"decision": len(self.decision_log), "time_s": now_s,
                  "candidates": names, "imagined_returns": evidence, "choice": chosen,
                  "margin": self.margin, "confidence_z": self.confidence_z,
                  "horizon_s": self.horizon_s, "timing": dict(self.last_timing), "oracle_state": True}
        self.decision_log.append(record)
        if self.log_fn:
            self.log_fn(record)
        return chosen, evidence

    @torch.no_grad()
    def rsu_policy_fn(self, state, history=None, *, now_s=None, **history_args):
        """Runner callback: accel [1,K]; recheck on simulation-time boundaries.

        Omit now_s for a runner called every CONTROL_DT_S, or supply monotonic
        simulation timestamps. Closed-loop actions update each call, while the
        selected plan stays fixed for decision_period_s. Imagination controllers
        never share lifecycle state with the execution controller.
        """
        if state["s_m"].ndim != 2 or state["s_m"].shape[0] != 1:
            raise ValueError("rsu_policy_fn requires a single state [1,K]")
        time_s = self._clock_s if now_s is None else float(now_s)
        if not math.isfinite(time_s) or (self._last_time_s is not None and time_s <= self._last_time_s):
            raise ValueError("runner timestamps must be finite and strictly increasing; reset per episode")
        if self._next_decision_s is None or time_s >= self._next_decision_s - 1e-9:
            chosen, _ = self.decide(state, history, now_s=time_s, **history_args)
            if chosen != self._chosen:
                self._controller = self.candidates[chosen]()
                self._execution_step = 0
                self._execution_start_s = time_s
            self._chosen = chosen
            self._next_decision_s = time_s + self.decision_period_s
        self._execution_step = round((time_s - self._execution_start_s) / PHYSICS_DT_S)
        execution = {**state, "history": history,
                     "physics_step_index": self._execution_step, **history_args}
        motion = self._controller(execution)
        if motion.shape != (*state["s_m"].shape, 4) or not torch.isfinite(motion[..., 3]).all():
            raise ValueError("execution controller must return finite [1,K,4] motion")
        self._last_time_s, self._clock_s = time_s, time_s + CONTROL_DT_S
        mask = state["present"].bool() & state["is_cav"].bool()
        return torch.where(mask, motion[..., 3], torch.zeros_like(motion[..., 3]))

    def reset(self):
        """Reset execution at an episode boundary; retain the decision audit log."""
        self._chosen = self._controller = self._next_decision_s = self._last_time_s = None
        self._clock_s, self._execution_step, self._execution_start_s = 0.0, 0, 0.0
