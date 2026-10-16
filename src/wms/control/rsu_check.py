"""Oracle-state deployment check: batched, paired WM-S counterfactuals.

Controller factories take no arguments and return fresh contract CavControllers.
Factories must isolate mutable controller state and be deterministic; the learned
bridge can be supplied through :func:`default_candidates`, which builds direct-output
candidates only. No simulation imports,
checkpoint loading, or training are performed here.
"""
from __future__ import annotations

import hashlib
import math
from time import perf_counter
from typing import Callable, Mapping, Optional

import torch
from torch import Tensor

from .contract import CONTROL_DT_S, PHYSICS_DT_S, CavController, RewardWeights
from .futures import DRIVER_SAMPLE_MAX_REL_DEV, arm_major_lane_uniform, sample_driver_encodings

ControllerFactory = Callable[[], CavController]


def _seed_of(*parts) -> int:
    digest = hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()
    return int(digest[:15], 16)


class NoiseStream:
    """Reproducible policy noise of one stochastic candidate ``policy_sample_<index>``.

    The noise of control step ``j`` after decision ``d`` is a function of
    ``(seed, index, d, j, full slot id)`` only, so every WM member and every driver sample
    of the check sees the SAME draws (common random numbers), and SUMO execution of the
    chosen sample replays the same per-slot stream.  ``begin(d, slot_ids)`` is called by
    the runner before each decision; ``slot_ids`` maps compacted imagination slots back to
    the full ``k_full`` slot pool (None = identity)."""

    def __init__(self, seed: int, index: int, *, k_full: Optional[int] = None):
        self.seed, self.index, self.k_full = int(seed), int(index), k_full
        self.decision, self.slot_ids = 0, None

    def begin(self, decision: int, slot_ids=None) -> None:
        self.decision = int(decision)
        self.slot_ids = None if slot_ids is None else torch.as_tensor(slot_ids).long().cpu()

    def noise(self, step: int, shape, *, slot_ids=None, device=None):
        from .policy import PolicyNoise, draw_policy_noise

        rows, k = shape
        ids = self.slot_ids if slot_ids is None else torch.as_tensor(slot_ids).long().cpu()
        ids = torch.arange(k) if ids is None else ids
        if ids.numel() != k:
            raise ValueError("slot ids do not match the imagined slot count")
        full = max(int(self.k_full or 0), int(ids.max()) + 1 if ids.numel() else 0, k)
        generator = torch.Generator().manual_seed(_seed_of(self.seed, self.index, self.decision, step))
        draw = draw_policy_noise((1, full), generator=generator)
        accel = draw.accel[:, ids].expand(rows, k)
        gumbel = draw.lane_gumbel[:, ids].expand(rows, k, draw.lane_gumbel.shape[-1])
        return PolicyNoise(accel.to(device) if device is not None else accel,
                           gumbel.to(device) if device is not None else gumbel)


def zero_group(group_encoder):
    def encode(*args, **kwargs):
        return torch.zeros_like(group_encoder(*args, **kwargs))
    return encode


def perturbed_group(group_encoder, perturbation: float):
    """Fixed deterministic offset of the group signal (cos ramp x perturbation)."""
    def encode(*args, **kwargs):
        signal = group_encoder(*args, **kwargs)
        offset = torch.arange(signal.shape[-1], device=signal.device,
                              dtype=signal.dtype).cos() * perturbation
        return signal + offset
    return encode


def hold_actor(obs, signal, *, lane_mask=None, **kwargs):
    """Direct-output safe fallback: a = 0 (then safety-filtered), stay in lane."""
    del signal, kwargs
    zero = torch.zeros(obs.cav_mask.shape, dtype=obs.own.dtype, device=obs.own.device)
    if lane_mask is None:
        return zero, zero
    from .policy import PolicyOutput
    return PolicyOutput(zero, torch.zeros_like(zero, dtype=torch.long), zero, zero, zero, zero,
                        torch.ones_like(zero, dtype=torch.long))


def default_candidates(model=None, *, policy, group_encoder, safety, bridge, lane_rules=None,
                       perturbation: float = 0.1, sample_streams=(),
                       alternative_group_encoders=()):
    """Direct-output candidates ONLY (no designed controllers):

    ``policy`` (deterministic accel + lane), ``policy_group_0`` (zero group signal),
    ``policy_group_1`` (perturbed group signal), ``policy_group_<2+>`` (injected
    alternative encoders), ``policy_sample_<k>`` (stochastic policy with the reproducible
    noise of ``sample_streams[k]``, common random numbers across WM members / driver
    samples) and ``hold_speed`` (direct command a = 0, stay in lane, safety-filtered).
    ``bridge(actor, group_encoder, safety[, lane_rules=...])`` is injected
    (make_cav_controller); with ``lane_rules`` the lane head is used.  ``model`` is unused
    (kept for call compatibility).
    """
    del model
    if not math.isfinite(perturbation) or perturbation <= 0:
        raise ValueError("perturbation must be finite and positive")
    lane_kwargs = {} if lane_rules is None else {"lane_rules": lane_rules}
    # The bridge bounds commands by ``actor.accel_max`` (ACCELCAP_NOTES.md): every candidate wrapper
    # exposes the policy's stored bound, so a 2.6 actor is not clipped and a legacy one stays at 1.5.
    from .policy import policy_accel_bounds
    bound = policy_accel_bounds(policy)[1]

    def actor(obs, signal, **kwargs):
        return policy(obs, signal, **{**kwargs, "deterministic": True})
    actor.accel_max = bound

    def hold(obs, signal, **kwargs):
        return hold_actor(obs, signal, **kwargs)
    hold.accel_max = bound

    def factory(encoder, act=actor):
        return lambda: bridge(act, encoder, safety, **lane_kwargs)

    def sampler(stream):
        def make():
            counter = [0]

            def act(obs, signal, **kwargs):
                noise = stream.noise(counter[0], tuple(obs.cav_mask.shape), device=obs.own.device)
                counter[0] += 1
                return policy(obs, signal, **{**kwargs, "deterministic": False, "noise": noise})
            act.accel_max = bound
            return bridge(act, group_encoder, safety, **lane_kwargs)
        return make

    encoders = [zero_group(group_encoder), perturbed_group(group_encoder, perturbation)]
    encoders.extend(alternative_group_encoders)
    result = {"policy": factory(group_encoder)}
    result.update({f"policy_group_{k}": factory(encoder) for k, encoder in enumerate(encoders)})
    result.update({f"policy_sample_{k}": sampler(stream) for k, stream in enumerate(sample_streams)})
    result["hold_speed"] = factory(group_encoder, hold)
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
    Default: point-estimate drivers (one sample), so the spread is WM-ensemble disagreement;
    with a single WM and one sample the LCB is undefined and the policy is always kept.
    Driver sampling (``driver_noise_scale`` > 0) uses the UNCALIBRATED encoder uncertainty
    (see :data:`wms.control.futures.DRIVER_SAMPLE_MAX_REL_DEV`) and is opt-in only.
    """

    def __init__(self, world_models: list, candidates: Mapping[str, ControllerFactory],
                 reward_fn, weights: RewardWeights, horizon_s: float = 20,
                 decision_period_s: float = 5, n_driver_samples: int = 1,
                 margin: float = 0.0, *, confidence_z: float = 1.96,
                 driver_noise_scale: float = 0.0, embedding_noise_std: float = 0.0,
                 driver_max_rel_dev: float = DRIVER_SAMPLE_MAX_REL_DEV,
                 seed: int = 0, log_fn: Optional[Callable] = None,
                 synchronize: Optional[Callable[[], None]] = None,
                 cav_exec_cap: Optional[str] = None):
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
        if n_driver_samples > 1 and driver_noise_scale == 0 and embedding_noise_std == 0:
            raise ValueError("driver samples are identical without driver noise; use n_driver_samples=1 "
                             "(the paired LCB then runs over the WM ensemble members)")
        self.world_models = list(world_models)
        # EXECCAP_NOTES.md: None keeps each model's own runtime switch (load_backend: env
        # WMS_CAV_EXEC_CAP); "off"/"idm"/"idm_normal" forces it for this checker's rollouts.
        if cav_exec_cap is not None:
            from wms.model.cav import check_exec_cap_mode
            cav_exec_cap = check_exec_cap_mode(cav_exec_cap)
        self.cav_exec_cap = cav_exec_cap
        self.candidates = dict(candidates)
        self.reward_fn, self.weights = reward_fn, weights
        self.horizon_s, self.decision_period_s = horizon_s, decision_period_s
        self.n_driver_samples, self.margin = n_driver_samples, margin
        self.confidence_z = confidence_z
        self.driver_noise_scale, self.embedding_noise_std = driver_noise_scale, embedding_noise_std
        self.driver_max_rel_dev = driver_max_rel_dev
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
        """Driver samples via the shared :func:`wms.control.futures.sample_driver_encodings`
        (batch size 1), tiled candidate-major / sample-minor over ``count`` candidates."""
        sampled = sample_driver_encodings(
            model, state, history_args, self.n_driver_samples,
            lambda shape, reference, key: self._normal(shape, reference, noise, key),
            driver_noise_scale=self.driver_noise_scale,
            embedding_noise_std=self.embedding_noise_std, max_rel_dev=self.driver_max_rel_dev)
        out = {}
        if "driver_params" in sampled:
            out["driver_params"] = {name: value[:, 0].repeat(count, 1)
                                    for name, value in sampled["driver_params"].items()}
        if "driver_embedding" in sampled:
            out["driver_embedding"] = sampled["driver_embedding"][:, 0].repeat(count, 1, 1)
        return out

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
            previous_cap = getattr(model, "cav_exec_cap", None)
            try:
                if self.cav_exec_cap is not None:
                    if previous_cap is None and self.cav_exec_cap != "off":
                        raise ValueError("cav_exec_cap requires world models with the switch")
                    if previous_cap is not None:
                        model.cav_exec_cap = self.cav_exec_cap
                if modes:
                    model.eval()
                drivers = self._sample_drivers(model, state, history_args, count, noise)
                if self.synchronize:
                    self.synchronize()
                rollout_start = perf_counter()
                # LANEFIX: sampled lane decisions share fixed draws across candidates.
                lanes = arm_major_lane_uniform([model], count, samples, steps,
                                               batched_state["s_m"].shape[1],
                                               torch.Generator().manual_seed(0))
                rollout = model.rollout(dict(batched_state), n_steps=steps, cav_teacher=None,
                                        cav_controller=dispatch, **batched_history, **drivers,
                                        **lanes)
                if self.synchronize:
                    self.synchronize()
                rollout_seconds += perf_counter() - rollout_start
                member_returns.append(self._returns(batched_state, rollout, steps).reshape(count, samples))
            finally:
                for module, training in modes:
                    module.training = training
                if previous_cap is not None:
                    model.cav_exec_cap = previous_cap
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
