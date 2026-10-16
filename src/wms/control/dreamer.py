"""Analytic, continuous-action Dreamer training over frozen WM-S members.

Control steps follow ``contract.STEPS_PER_CONTROL``; ``horizon=60`` therefore
means 60 seconds under the current shared contract (120 physics steps).
All control/reward dependencies are injected; importing this module needs no
parallel control modules, simulator, dataset, or checkpoint.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .contract import ACCEL_MAX_MPS2, PHYSICS_DT_S, STEPS_PER_CONTROL, RewardWeights
from .counterfactual import (KIND_ALL, CounterfactualConfig, baseline_index, build_variants,
                             decompose, expand_rows, group_rich, mean_or_none, native_fn,
                             row_coefficients)
from .futures import FuturesConfig, SampledFutures, lane_uniform_draws
from .start_states import StartBatch, checked_path

# /2 (2026-10-12 late): lane head, lane rules, counterfactuals; /3: reports carry the other CAVs'
# last commanded acceleration and lane intent (older actors load with zero weights on them).
CHECKPOINT_FORMAT = "c4_dreamer/3"
READABLE_FORMATS = ("c4_dreamer/1", "c4_dreamer/2", CHECKPOINT_FORMAT)


@dataclass
class DreamerConfig:
    batch_size: int = 16
    horizon: int = 60
    gamma: float = 0.99
    lambda_: float = 0.95
    actor_lr: float = 5e-5
    critic_lr: float = 1e-4
    entropy_weight: float = 3e-4
    uncertainty_weight: float = 0.1
    ttc_budget: float = 3.6
    dual_lr: float = 0.01
    lagrange_init: float = 0.0
    grad_clip: float = 100.0
    target_tau: float = 0.02
    return_ema: float = 0.99
    checkpoint_every: int = 100
    validation_every: int = 100
    lane_pg_weight: float = 1.0          # REINFORCE term of the categorical lane head
    lane_entropy_weight: float = 3e-4    # DreamerV3 discrete-action entropy bonus (lane head)
    # Entropy temperature (V6EXPLORE, 2026-10-15; default None = off = bit-identical): with
    # ``entropy_target`` set (nats per action dim) the fixed ``entropy_weight`` bonus is
    # replaced by ``alpha * entropy`` with a learned SAC-style dual ``log_alpha`` (Adam,
    # ``alpha_lr``) minimising ``-(log_alpha * (entropy_target - entropy))``: alpha rises while
    # the actor's mean entropy is below the target and falls above it.  Evidence: the v5 run
    # (fixed 3e-3 weight) let the per-action entropy fall 1.39 -> -0.35 nats over 517 updates
    # while D_all plateaued and C_G decayed to ~0.  log_alpha starts at log(entropy_weight)
    # (alpha = the fixed weight the run would otherwise use; 0 when the weight is 0), and the
    # same machinery applies to the categorical lane head via ``lane_entropy_target`` (requires
    # the lane rules + a policy lane head).
    entropy_target: Optional[float] = None
    lane_entropy_target: Optional[float] = None
    alpha_lr: float = 3e-4
    # Memory (CFMEM_NOTES.md): recompute the pure per-step pieces of imagination (WM forward,
    # policy / group encoder / safety / native / reward / target-critic calls) during backward
    # instead of storing their activations.  Values and gradients are unchanged; ~1 extra forward.
    checkpoint_imagination: bool = True
    # Entry queue (SRCWAIT_NOTES.md, 2026-10-13): vehicles enter the imagined rollouts from the
    # observed pre-t0 entry history of each start window (TorchEntryProcessV2); arrivals that find
    # no safe gap at the entrance queue (uncapped backlog) and are charged as source waiting by
    # team_reward.  Needs start windows with ``entry_feat`` (train_cli builds them); without them
    # imagination runs without entries, exactly as before.
    imagine_entries: bool = True
    entry_pad_max: int = 256            # cap on the empty entrant slots appended per batch
    # Sampled futures (FUTURES_NOTES.md; default 1 = the single deterministic rollout, bit-identical):
    # every start is imagined under ``future_samples`` stochastic worlds -- HDV driver parameters
    # perturbed within the encoder uncertainty (``future_driver_scale`` x uncertainty; 0 = off,
    # embedding std ``future_embedding_std``) and, with ``future_accel_noise``, per-step HDV
    # acceleration noise N(0, (scale * min(exp(log_sigma), sigma_max))^2).  Returns are averaged
    # over members x samples and the disagreement penalty is the std over members x samples;
    # counterfactual rows, members and (via ``future_seed`` + update) candidates share the draws.
    future_samples: int = 1
    future_driver_scale: float = 1.0
    future_embedding_std: float = 0.05
    future_accel_noise: bool = False
    future_accel_scale: float = 1.0
    future_accel_sigma_max: float = 1.0
    future_seed: int = 0
    # Entry-queue gradient (SRCGRAD_NOTES.md, 2026-10-13; default off = bit-identical): the reward
    # keeps the exact imagined backlog as its value but gets the gradient of a differentiable
    # insertion-blocking surrogate (wms.control.srcgrad), so blocking an entrance is no longer free
    # for the actor.  ``critic_entry_backlog`` feeds the per-entrance backlog to the TeamCritic
    # (which must be built with ``entry_backlog=True``) so the bootstrap prices queues beyond H.
    source_wait_grad: bool = False
    critic_entry_backlog: bool = False
    # SUMO-consistent execution of commanded CAVs (EXECCAP_NOTES.md, default "off" = bit-identical):
    # "idm" -> each 0.5 s step a policy CAV realises min(safe command, its own physics-core IDM accel
    # with the WM-inferred driver settings, (v0-v)/dt, accel setting), as SUMO speedMode 27 does.
    cav_exec_cap: str = "off"
    # ENTRYFIX (ENTRYFIX_NOTES.md, default "v2" = bit-identical): insertion rule of the imagined
    # entry process; "sumo_like" = SUMO departSpeed="max"/departLane="best" insertion (the v2
    # sampled required gap over-blocks: imagined source wait far above SUMO's);
    # "sumo_like_v2" = V6ENTRY saturated-entrance semantics (end-of-step gap test, stop-speed
    # departure; ENTRY_DIAG.md) -- the rule that reproduces SUMO's queued mainline service.
    entry_insert_rule: str = "v2"
    # V6: None preserves the WM's training-time damping, including the exact default graph.
    # Overrides apply only while imagining; the frozen member's setting is always restored.
    imagination_grad_decay: float | None = None
    # Reward-window credit to the first continuous policy action (one start/member/future).
    # Zero disables all diagnostic autograd work; otherwise sample every N updates.
    credit_profile_every: int = 0

    def __post_init__(self):
        for name in ("batch_size", "horizon"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        from wms.model.cav import check_exec_cap_mode
        self.cav_exec_cap = check_exec_cap_mode(self.cav_exec_cap)
        decay = self.imagination_grad_decay
        if decay is not None and (isinstance(decay, bool) or not isinstance(decay, (int, float))
                                  or not math.isfinite(decay) or not 0 < decay <= 1):
            raise ValueError("imagination_grad_decay must be None or in (0, 1]")
        if (isinstance(self.credit_profile_every, bool)
                or not isinstance(self.credit_profile_every, int) or self.credit_profile_every < 0):
            raise ValueError("credit_profile_every must be a nonnegative integer")
        for name, value in asdict(self).items():
            if name in ("cav_exec_cap", "entry_insert_rule", "imagination_grad_decay",
                        "entropy_target", "lane_entropy_target") or value is None:
                continue   # the string fields; the optional decay / entropy targets
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("entropy_target", "lane_entropy_target"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value)):
                raise ValueError(f"{name} must be None or a finite number")
        for name in ("gamma", "lambda_", "target_tau", "return_ema"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if min(self.actor_lr, self.critic_lr, self.alpha_lr, self.grad_clip) <= 0:
            raise ValueError("learning rates and grad_clip must be positive")
        for name in ("checkpoint_every", "validation_every"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer")
        for name in ("checkpoint_imagination", "imagine_entries", "future_accel_noise",
                     "source_wait_grad", "critic_entry_backlog"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.entry_insert_rule not in ("v2", "sumo_like", "sumo_like_v2"):
            raise ValueError("entry_insert_rule must be 'v2', 'sumo_like' or 'sumo_like_v2'")
        if isinstance(self.entry_pad_max, bool) or not isinstance(self.entry_pad_max, int) \
                or self.entry_pad_max < 1:
            raise ValueError("entry_pad_max must be a positive integer")
        for name in ("future_samples", "future_seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        FuturesConfig.from_trainer(self)           # validates the sampling fields


def lambda_returns(rewards: Tensor, values: Tensor, gamma=0.99, lambda_=0.95) -> Tensor:
    """Time-first TD(lambda), including a value bootstrap at the final horizon."""
    if rewards.ndim != 2 or values.shape != (rewards.shape[0] + 1, rewards.shape[1]):
        raise ValueError("expected rewards [H,B] and values [H+1,B]")
    if not rewards.shape[0] or not 0 <= gamma <= 1 or not 0 <= lambda_ <= 1:
        raise ValueError("nonempty horizon and gamma/lambda in [0,1] required")
    result, carry = [], values[-1]
    for t in reversed(range(rewards.shape[0])):
        carry = rewards[t] + gamma * ((1 - lambda_) * values[t + 1] + lambda_ * carry)
        result.append(carry)
    return torch.stack(result[::-1])


def ensemble_penalty(rewards: Tensor, weight: float) -> Tensor:
    """Population disagreement in per-control-step reward, zero for one member."""
    if rewards.ndim != 3 or rewards.shape[0] < 1 or weight < 0:
        raise ValueError("expected nonempty [E,H,B] rewards and nonnegative weight")
    if rewards.shape[0] == 1 or weight == 0:
        return torch.zeros_like(rewards[0])
    return weight * rewards.std(dim=0, unbiased=False)


def _rng_state(device):
    numpy_state = np.random.get_state()
    # Plain types/tensors keep checkpoints loadable with weights_only=True.
    numpy_state = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
    out = {"python": random.getstate(), "numpy": numpy_state,
           "torch": torch.get_rng_state()}
    if device.type == "cuda":
        out["cuda"] = torch.cuda.get_rng_state(device)
    return out


def _restore_rng(state, device):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         *numpy_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if "cuda" in state and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def _rows(state, stop, total):
    """First ``stop`` rows of every [total, ...] tensor in a state dict."""
    return {key: (value[:stop] if torch.is_tensor(value) and value.ndim and value.shape[0] == total
                  else value) for key, value in state.items()}


def _masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def _detach(state):
    return {key: value.detach() if torch.is_tensor(value) else value
            for key, value in state.items()}


def _accepts(fn, name: str) -> bool:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in params)


class _Recompute:
    """Call ``fn`` under non-reentrant activation checkpointing while gradients are enabled.

    Only for PURE calls (output a function of the arguments; the stateful bridge controller is
    never wrapped, so nothing with side effects is replayed).  ``preserve_rng`` restores the
    RNG state at recomputation for calls that sample internally."""

    def __init__(self, fn, enabled, preserve_rng=False):
        self.fn, self.enabled, self.preserve_rng = fn, bool(enabled), bool(preserve_rng)

    def __call__(self, *args, **kwargs):
        if self.enabled and torch.is_grad_enabled():
            return activation_checkpoint(self.fn, *args, use_reentrant=False,
                                         preserve_rng_state=self.preserve_rng, **kwargs)
        return self.fn(*args, **kwargs)


def _model_identity(model, imagination_grad_decay=None):
    digest = hashlib.sha256(type(model).__qualname__.encode())
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    digest.update(str(getattr(model, "grad_decay_per_step", 1.0)).encode())
    if imagination_grad_decay is not None:
        digest.update(f":imagination_grad_decay={float(imagination_grad_decay)}".encode())
    return digest.hexdigest()


def _credit_profile(rewards, action, mask):
    """L2 norm of d R[k-1, row=0] / d a[0, row=0] for one member/future.

    R[k-1] sums the physics rewards in the kth (one-based) control window, NOT the
    cumulative prefix or a bootstrapped return. Action is the original policy acceleration
    tensor, before safety/execution caps (not the post-rollout stacked logging tensor).
    Include the terminal window for tiny horizons; disconnected/absent CAVs have zero credit.
    autograd.grad leaves parameter .grad buffers alone and retains the training graph.
    """
    horizon = rewards.shape[0]
    steps = sorted({k for k in (1, 5, 10, 20, 30, horizon) if k <= horizon})
    norms = {}
    for k in steps:
        reward = rewards[k - 1, 0]
        grad = None
        if action.requires_grad and reward.requires_grad:
            grad, = torch.autograd.grad(reward, action, retain_graph=True, allow_unused=True)
        norms[str(k)] = (float((grad[0] * mask[0]).norm().detach()) if grad is not None else 0.0)
    return {"norms": norms, "sample_size": 1, "start_row": 0, "member": 0, "future": 0,
            "window_control_steps": 1, "action": "policy_acceleration",
            "active_actions": int(mask[0].sum())}


_ENTRY_PAD_SAFETY = 1.5
_ENTRY_PAD_MARGIN = 16
_PAD_FILL = {"lane": -1, "length_m": 5.0}


def _pad_slots(value, k, pad, fill=0):
    """Append ``pad`` slots along dim 1 of a [B, K, ...] tensor (other values unchanged)."""
    if not (torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == k):
        return value
    shape = (value.shape[0], pad, *value.shape[2:])
    return torch.cat((value, torch.full(shape, fill, dtype=value.dtype, device=value.device)), 1)


def with_entry_slots(batch: StartBatch, horizon_s: float, pad_max: int, generator=None):
    """Start batch with empty entrant slots appended and its entry spec (``batch.entry_spec``).

    The pad holds 1.5x the expected arrivals (observed entry-history rate x horizon) + 16, the
    batch maximum, capped at ``pad_max``; one process seed per start (rows of a start share it).
    Returns ``batch`` unchanged when it has no ``entry_feat``."""
    if batch.entry_feat is None:
        return batch
    from wms.model.entries import EntranceHistory

    feats = batch.entry_feat.detach().cpu().numpy()
    histories = [EntranceHistory.from_features(row) for row in feats]
    k = batch.state["s_m"].shape[1]
    pad = max(int(math.ceil(float(np.sum(h.rate)) * horizon_s * _ENTRY_PAD_SAFETY))
              + _ENTRY_PAD_MARGIN for h in histories)
    pad = max(1, min(int(pad_max), pad))
    seeds = torch.randint(0, 2 ** 31 - 1, (len(histories),), generator=generator).tolist()
    state = {key: _pad_slots(value, k, pad, _PAD_FILL.get(key, 0))
             for key, value in batch.state.items()}
    kwargs = {key: _pad_slots(value, k, pad) for key, value in batch.rollout_kwargs.items()}
    return StartBatch(state, kwargs, batch.metadata, batch.entry_feat,
                      {"histories": histories, "seeds": seeds, "k0": k, "pad": pad})


class _EntryRecorder:
    """Entry process wrapper: records the post-insertion state and the backlog of every step,
    plus (for the source-wait gradient) the per-entrance pending counts before / after the
    insertion and the mask of the slots inserted on that step."""

    def __init__(self, process):
        self.process, self.states, self.backlog = process, [], []
        self.pending_before, self.pending_after, self.new_masks = [], [], []

    def insert(self, state):
        before = self.process.pending.copy()
        mask = self.process.insert(state)
        self.states.append(state)
        self.backlog.append(self.process.backlog())
        self.pending_before.append(before)
        self.pending_after.append(self.process.pending.copy())
        self.new_masks.append(mask)
        return mask


def _queue_surrogate_fn(state, new_mask, pending_before, hist, entry_s, entry_cfg):
    """Straight-through-ready surrogate queue [B] from the PRE-insertion state of a step (the
    post-insertion state with this step's entrants masked out; insertion only writes free slots)."""
    from .srcgrad import queue_surrogate

    present = state["present"].bool() & ~new_mask.bool()
    return queue_surrogate(state, present, pending_before, hist, entry_s, entry_cfg, PHYSICS_DT_S)[0]


def _entry_process(spec, rows, n_slots, insert_rule: str = "v2"):
    from wms.model.entries import EntryConfig, TorchEntryProcessV2

    pool = np.zeros((len(rows), n_slots), dtype=bool)
    pool[:, spec["k0"]:] = True
    # Uncapped backlog: every due arrival that cannot enter keeps waiting (as in SUMO).
    return TorchEntryProcessV2(histories=[spec["histories"][r] for r in rows],
                               dt_s=PHYSICS_DT_S, seed=0, batch=len(rows),
                               cfg=EntryConfig(max_pending=10 ** 9, insert_rule=insert_rule),
                               pool=pool,
                               row_seeds=[spec["seeds"][r] for r in rows])


def _same_groups(sampler, state):
    """True when a saved sampler ``state`` matches ``sampler``'s window groups (resume check)."""
    if sampler is None or not hasattr(sampler, "groups") or state is None:
        return False
    return (state.get("groups") == sampler.groups and state.get("relabel") == sampler.relabel
            and state.get("target_penetration") == sampler.target_penetration)


class DreamerTrainer:
    """Owns actor/group encoder, critic, EMA target, dual and sampler state.

    The internal controller samples once per control step, holds acceleration,
    reapplies safety and integrates motion on every physics callback. A single
    WM rollout preserves driver encoding, history, lane ages and gradient damping.
    """

    def __init__(self, world_models, policy, group_encoder, critic, start_states, *,
                 observation_builder, reward_fn, safety, controller_factory=None, config=None,
                 reward_weights=None, device="cpu", output_dir=None, lane_rules=None,
                 counterfactual=None):
        self.config = config or DreamerConfig()
        # Lane head: active when lane rules are supplied AND the
        # policy has a categorical lane head; trained by REINFORCE (accelerations stay analytic).
        self.lane_rules = lane_rules
        self.use_lanes = lane_rules is not None and bool(getattr(policy, "has_lane_head", False))
        self.counterfactual = counterfactual or CounterfactualConfig()
        self.device = torch.device(device)
        if not world_models:
            raise ValueError("at least one world model is required")
        self.world_models = [model.to(self.device).eval() for model in world_models]
        for model in self.world_models:
            model.requires_grad_(False)
            model.zero_grad(set_to_none=True)
            dt = getattr(getattr(model, "physics", None), "dt_s", PHYSICS_DT_S)
            if not math.isclose(dt, PHYSICS_DT_S):
                raise ValueError("WM physics dt does not match the shared control contract")
        self.world_model_ids = [_model_identity(model, self.config.imagination_grad_decay)
                                for model in self.world_models]
        self.policy = policy.to(self.device)
        self.group_encoder = group_encoder.to(self.device)
        self.critic = critic.to(self.device)
        if self.config.critic_entry_backlog != bool(getattr(critic, "entry_backlog", False)):
            raise ValueError("critic_entry_backlog requires (and only works with) a TeamCritic "
                             "built with entry_backlog=True")
        self.target_critic = copy.deepcopy(critic).to(self.device).eval().requires_grad_(False)
        self.start_states = start_states
        self.observation_builder, self.reward_fn, self.safety = observation_builder, reward_fn, safety
        self.controller_factory = controller_factory
        self.reward_weights = reward_weights or RewardWeights()
        self.actor_parameters = list(dict.fromkeys(
            list(self.policy.parameters()) + list(self.group_encoder.parameters())))
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)
        # V6EXPLORE entropy temperatures (default off: None targets keep the fixed weights).
        self.log_alpha = self.alpha_optimizer = None
        self.log_lane_alpha = self.lane_alpha_optimizer = None
        self._init_entropy_temperature()
        self.lagrange_multiplier = self.config.lagrange_init
        self.return_low = self.return_high = None
        self.update_count = 0
        self.best_validation = -math.inf
        self.best_coop_validation = -math.inf
        self.validation_states = None
        self._pending_validation_state = None
        self.output_dir = checked_path(output_dir) if output_dir is not None else None

    def _init_entropy_temperature(self):
        """(Re)create the learned entropy-temperature parameters for the configured targets.

        ``log_alpha`` starts at ``log(entropy_weight)`` (0 when the weight is 0), so enabling a
        target continues from the fixed bonus the run would otherwise use.  Called again after
        a checkpoint replaces ``self.config``, so resumed temperatures match the saved config.
        """
        if self.config.entropy_target is not None and self.log_alpha is None:
            start = math.log(self.config.entropy_weight) if self.config.entropy_weight > 0 else 0.0
            self.log_alpha = torch.tensor(start, device=self.device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        if self.config.lane_entropy_target is not None:
            if not self.use_lanes:
                raise ValueError("lane_entropy_target requires lane rules and a policy lane head")
            if self.log_lane_alpha is None:
                start = (math.log(self.config.lane_entropy_weight)
                         if self.config.lane_entropy_weight > 0 else 0.0)
                self.log_lane_alpha = torch.tensor(start, device=self.device, requires_grad=True)
                self.lane_alpha_optimizer = torch.optim.Adam([self.log_lane_alpha],
                                                             lr=self.config.alpha_lr)

    def _alpha_update(self, imagined, entropy):
        """V6EXPLORE dual steps on the entropy temperatures; extra metrics for ``update``.

        ``-(log_alpha * (target - entropy))``: gradient descent raises log_alpha while the mean
        entropy is below the target (more exploration) and lowers it above."""
        out = {}
        if self.log_alpha is not None:
            alpha_loss = -(self.log_alpha * (self.config.entropy_target - entropy.detach()))
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            out.update({"alpha": float(self.log_alpha.detach().exp()),
                        "entropy_target": float(self.config.entropy_target)})
        lanes = imagined.get("lanes")
        if self.log_lane_alpha is not None and lanes is not None:
            lane_entropy = _masked_mean(lanes["lane_entropy"].detach(), lanes["lane_active"])
            lane_alpha_loss = -(self.log_lane_alpha *
                                (self.config.lane_entropy_target - lane_entropy))
            self.lane_alpha_optimizer.zero_grad(set_to_none=True)
            lane_alpha_loss.backward()
            self.lane_alpha_optimizer.step()
            out.update({"lane_alpha": float(self.log_lane_alpha.detach().exp()),
                        "lane_entropy_target": float(self.config.lane_entropy_target)})
        return out

    def imagine(self, batch: StartBatch, deterministic=False, variants=None, future_seed=None,
                credit_profile=False):
        """Imagined rollouts of ``batch`` (and, with ``variants``, of its counterfactual rows).

        With ``variants`` (:func:`wms.control.counterfactual.build_variants`) every row r
        replays start ``variants.row_start[r]``; only the CAVs in ``variants.act_mask[r]``
        follow the policy, the others the native CAV behaviour.  Rows 0..B-1 are the normal
        all-CAV rollouts.  Policy noise is drawn per (start, control step, slot) and shared by
        all rows of a start (common random numbers) when the policy accepts ``noise``.

        Sampled futures (``config.future_samples`` > 1 or ``future_accel_noise``): every ensemble
        member is rolled out once per sampled future, and the member x sample pairs act as the
        "ensemble" axis ``E' = E * S`` of all outputs (member-major: index ``e * S + s``).  A future
        is one draw of the HDV driver parameters (within the encoder uncertainty) and, optionally,
        of the per-step HDV acceleration noise; both are drawn once per START and shared by all
        members and all counterfactual rows of that start.  ``future_seed`` fixes the draws
        (default: from the config seed and the update count).  Off: nothing changes.
        """
        cfg = self.config
        forbidden = {"cav_teacher", "lane_teacher", "action_timeline", "present_seq", "is_cav_seq",
                     "junction_seq", "truncation_steps", "cav_controller"}
        if forbidden.intersection(batch.rollout_kwargs):
            raise ValueError("imagination inputs must not contain teachers or recorded controls")
        n_base = batch.state["s_m"].shape[0]
        if variants is not None:
            index = variants.row_start.to(self.device)
            start_state = {key: expand_rows(value, index, n_base) for key, value in batch.state.items()}
            rollout_kwargs = {key: expand_rows(value, index, n_base)
                              for key, value in batch.rollout_kwargs.items()}
            act_mask = variants.act_mask.to(self.device)
        else:
            index, start_state, rollout_kwargs, act_mask = None, batch.state, batch.rollout_kwargs, None
        entry_spec = getattr(batch, "entry_spec", None) if cfg.imagine_entries else None
        entry_rows = None
        if entry_spec is not None:
            entry_rows = index.tolist() if index is not None else list(range(n_base))
            if act_mask is not None:
                # Entrant CAVs follow the policy on all-CAV rows, the native behaviour elsewhere.
                act_mask = act_mask.clone()
                act_mask[variants.row_kind.to(self.device) == KIND_ALL, entry_spec["k0"]:] = True
        if self.use_lanes and self.controller_factory is None:
            raise ValueError("lane decisions require the bridge controller factory")
        need_bridge = self.controller_factory is not None and (self.use_lanes or act_mask is not None)
        crn = act_mask is not None and bool(getattr(self.policy, "supports_noise", False))
        recompute = cfg.checkpoint_imagination and torch.is_grad_enabled()
        from wms.model.cav import apply_exec_cap
        native = _Recompute(native_fn(self.counterfactual.native_cav,
                                      a_max=self.counterfactual.native_accel_max_mps2), recompute)
        # Action bounds (ACCELCAP_NOTES.md): the policy's stored cap bounds its commands; native
        # counterfactual CAVs use the native (SUMO CAV) accel; safety layers without an
        # ``accel_max`` keyword are called unchanged.
        policy_max = float(getattr(self.policy, "accel_max", ACCEL_MAX_MPS2))
        native_max = float(self.counterfactual.native_accel_max_mps2)
        # Policy samples internally only without explicit (CRN) noise and when stochastic.
        policy_call = _Recompute(self.policy, recompute, preserve_rng=not (crn or deterministic))
        group_call = _Recompute(self.group_encoder, recompute)
        from .safety import with_accel_max
        safety_call = _Recompute(with_accel_max(self.safety), recompute)
        reward_call = _Recompute(self.reward_fn, recompute)
        observe_call = _Recompute(self.observation_builder, recompute)
        value_call = _Recompute(self.target_critic, recompute)
        surrogate_call = _Recompute(_queue_surrogate_fn, recompute)
        futures_cfg = FuturesConfig.from_trainer(cfg)
        future_extras = None
        if futures_cfg.active:
            seed = self.update_count if future_seed is None else future_seed
            future_extras = SampledFutures(futures_cfg, seed).plan(
                self.world_models, batch.state, batch.rollout_kwargs,
                cfg.horizon * STEPS_PER_CONTROL)
            if index is not None:       # rows of one start share its sample
                future_extras = [[expand_rows(extra, index, n_base) for extra in per_sample]
                                 for per_sample in future_extras]
        n_futures = futures_cfg.n_samples if futures_cfg.active else 1
        # LANEFIX: WMs with sampled lane decisions get per-start draws shared by every member
        # and every counterfactual row of the start (None for argmax WMs: nothing changes).
        lane_uniform = lane_uniform_draws(
            self.world_models, n_base, cfg.horizon * STEPS_PER_CONTROL,
            batch.state["s_m"].shape[1],
            self.update_count if future_seed is None else future_seed, device=self.device)
        if lane_uniform is not None and index is not None:
            lane_uniform = expand_rows(lane_uniform, index, n_base)
        member_states, member_rewards, member_costs, member_values = [], [], [], []
        member_entropy, member_actions, member_components, member_lanes = [], [], [], []
        profile = None
        rng = _rng_state(self.device)
        advanced_rng = None
        # Common random numbers: disagreement must measure WM differences, not action noise.
        for model_index, sample_index in [(e, s) for e in range(len(self.world_models))
                                          for s in range(n_futures)]:
            model = self.world_models[model_index]
            future = {} if future_extras is None else future_extras[model_index][sample_index]
            _restore_rng(rng, self.device)
            records = []
            step = 0
            held_out = None
            last_decision_step = -1

            def policy_proxy(obs, signal, **kwargs):
                nonlocal held_out, last_decision_step
                if step % STEPS_PER_CONTROL == 0 and last_decision_step != step:
                    if crn:
                        from .policy import PolicyNoise, draw_policy_noise
                        noise = draw_policy_noise((n_base, obs.cav_mask.shape[-1]), device=self.device)
                        kwargs["noise"] = PolicyNoise(noise.accel.index_select(0, index),
                                                      noise.lane_gumbel.index_select(0, index))
                    held_out = policy_call(obs, signal, deterministic=deterministic, **kwargs)
                    acting = obs.cav_mask if act_mask is None else obs.cav_mask & act_mask
                    log_prob = getattr(held_out, "log_prob_accel", None)
                    log_prob = held_out[1] if log_prob is None else log_prob
                    record = records[-1]
                    if credit_profile and model_index == 0 and sample_index == 0 and step == 0:
                        record["credit_action"] = held_out[0]
                        record["credit_mask"] = acting
                    record["entropy"] = -(log_prob * acting).sum(-1) / acting.sum(-1).clamp_min(1)
                    if hasattr(held_out, "log_prob_lane"):
                        record["lane_log_prob"] = held_out.log_prob_lane
                        record["lane_entropy"] = held_out.entropy_lane
                        record["lane_active"] = acting & (held_out.lane_options > 1)
                        record["lane_acting"] = acting
                    last_decision_step = step
                if held_out is None:
                    raise RuntimeError("controller factory did not sample an initial policy command")
                return held_out
            policy_proxy.accel_max = policy_max

            def lane_record(entry):
                records[-1]["lane_changed"] = entry["lane_changed"].detach()

            external_callback = None
            if need_bridge:
                extra = {}
                if self.use_lanes:
                    extra["lane_rules"] = self.lane_rules
                if act_mask is not None:
                    extra.update(act_mask=act_mask, native=native)
                    if _accepts(self.controller_factory, "native_accel_max"):
                        extra["native_accel_max"] = native_max
                external_callback = self.controller_factory(
                    policy_proxy, group_call, safety_call,
                    lane_record if self.use_lanes else None, **extra)
            elif self.controller_factory is not None:
                external_callback = self.controller_factory(policy_proxy, group_call,
                                                            safety_call, None)

            def controller(state):
                nonlocal step
                mask = state["present"].bool() & state["is_cav"].bool()
                if step % STEPS_PER_CONTROL == 0:
                    records.append({"mask": mask})
                if external_callback is not None:
                    command = external_callback(state)
                    if step % STEPS_PER_CONTROL == 0:
                        if "entropy" not in records[-1]:
                            raise RuntimeError("controller factory must call the supplied policy at each decision")
                        records[-1]["action"] = command[..., 3]
                    step += 1
                    return command
                if step % STEPS_PER_CONTROL == 0:
                    obs = observe_call(state)
                    signal = group_call(obs, state)
                    policy_proxy(obs, signal)
                command = held_out[0]
                ceiling = policy_max
                if act_mask is not None:       # counterfactual rows: non-acting CAVs native
                    command = torch.where(act_mask, command, native(state))
                    ceiling = torch.where(act_mask, torch.full_like(command, policy_max),
                                          torch.full_like(command, native_max))
                accel = safety_call(state, command, accel_max=ceiling)
                cav = state["present"].bool() & state["is_cav"].bool()
                accel = torch.where(cav, accel, torch.zeros_like(accel))
                # EXECCAP_NOTES.md: policy CAVs only (identity unless the WM's cav_exec_cap is on).
                accel = apply_exec_cap(state, accel, cav if act_mask is None else cav & act_mask)
                if step % STEPS_PER_CONTROL == 0:
                    records[-1]["action"] = accel
                # SUMO's ballistic position integration, including a stop inside this step.
                speed = (state["v_mps"] + PHYSICS_DT_S * accel).clamp_min(0)
                moving_dt = torch.where(
                    accel < 0, (state["v_mps"] / (-accel).clamp_min(1e-8)).clamp(0, PHYSICS_DT_S),
                    torch.full_like(accel, PHYSICS_DT_S))
                position = state["s_m"] + state["v_mps"] * moving_dt + 0.5 * accel * moving_dt.square()
                step += 1
                return torch.stack((position, state["lane"].to(speed.dtype), speed, accel), -1)

            initial = {key: value.clone() if torch.is_tensor(value) else value
                       for key, value in start_state.items()}
            recorder = None
            member_kwargs = {**rollout_kwargs, **future}
            if lane_uniform is not None:
                member_kwargs["lane_uniform"] = lane_uniform
            if entry_spec is not None:
                recorder = _EntryRecorder(_entry_process(entry_spec, entry_rows,
                                                         initial["s_m"].shape[1],
                                                         cfg.entry_insert_rule))
                member_kwargs = {**member_kwargs, "entries": recorder}
            switch = hasattr(model, "checkpoint_forward")
            previous_switch = getattr(model, "checkpoint_forward", False)
            cap_switch = hasattr(model, "cav_exec_cap")
            previous_cap = getattr(model, "cav_exec_cap", "off")
            if not cap_switch and cfg.cav_exec_cap != "off":
                raise ValueError("cav_exec_cap requires world models with the execution-cap switch")
            decay_switch = cfg.imagination_grad_decay is not None
            if decay_switch and not hasattr(model, "grad_decay_per_step"):
                raise ValueError("imagination_grad_decay requires world models with grad_decay_per_step")
            previous_decay = getattr(model, "grad_decay_per_step", None)
            try:
                if switch:
                    model.checkpoint_forward = recompute
                if cap_switch:
                    model.cav_exec_cap = cfg.cav_exec_cap
                if decay_switch:
                    model.grad_decay_per_step = cfg.imagination_grad_decay
                roll = model.rollout(initial, n_steps=cfg.horizon * STEPS_PER_CONTROL,
                                     cav_controller=controller, cav_teacher=None,
                                     truncation_steps=None, **member_kwargs)
            finally:
                if switch:
                    model.checkpoint_forward = previous_switch
                if cap_switch:
                    model.cav_exec_cap = previous_cap
                if decay_switch:
                    model.grad_decay_per_step = previous_decay
            if step != cfg.horizon * STEPS_PER_CONTROL:
                raise RuntimeError("WM must invoke the CAV callback once per physics step")
            if advanced_rng is None:
                advanced_rng = _rng_state(self.device)
            if recorder is not None:
                # State of transition t = the WM input after this step's insertions (entrants
                # are charged from their first physics step); entry_queue = the backlog waiting
                # during transition t (team_reward: source_wait = dt * entry_queue).
                if len(recorder.states) != step:
                    raise RuntimeError("entry process must be called once per physics step")
                physics_states = []
                dtype = initial["s_m"].dtype
                hist = None
                if cfg.source_wait_grad:
                    from .srcgrad import history_tensors
                    hist = history_tensors(recorder.process, initial["s_m"].shape[0],
                                           self.device, dtype)
                for t, current in enumerate(recorder.states):
                    queue = torch.as_tensor(recorder.backlog[t], dtype=dtype, device=self.device)
                    if hist is not None:
                        # SRCGRAD: exact backlog value, insertion-blocking surrogate gradient.
                        surrogate = surrogate_call(
                            {key: current[key] for key in ("s_m", "v_mps", "lane", "present",
                                                           "length_m") if key in current},
                            recorder.new_masks[t],
                            torch.as_tensor(recorder.pending_before[t], dtype=dtype,
                                            device=self.device),
                            hist, recorder.process.entry_s, recorder.process.cfg)
                        queue = queue + (surrogate - surrogate.detach())
                    physics_states.append({
                        **initial, **{key: current[key] for key in (
                            "s_m", "v_mps", "lane", "present", "a_mps2", "is_cav", "length_m",
                            "entrant") if key in current},
                        "entry_queue": queue})
                    if cfg.critic_entry_backlog:
                        physics_states[-1]["entry_backlog"] = torch.as_tensor(
                            recorder.pending_after[t], dtype=dtype, device=self.device)
                physics_states.append(None)
            else:
                physics_states = [initial]
                for t in range(step):
                    nxt = {**initial, "s_m": roll["positions"][:, t], "v_mps": roll["speeds"][:, t],
                           "lane": roll["lanes"][:, t], "present": roll["masks"][:, t],
                           "a_mps2": roll["accel"][:, t]}
                    if "entrant_seq" in roll:
                        nxt["entrant"] = roll["entrant_seq"][:, t]
                    physics_states.append(nxt)
            # Preserve WM-specific state fields at the terminal boundary.
            physics_states[-1] = roll["state"]
            if recorder is not None and cfg.critic_entry_backlog:
                physics_states[-1] = {**roll["state"], "entry_backlog": torch.as_tensor(
                    recorder.pending_after[-1], dtype=initial["s_m"].dtype, device=self.device)}
            parts = []
            for t in range(step):
                parts.append(reward_call(physics_states[t], physics_states[t + 1],
                                         roll["accel"][:, t], self.reward_weights))
            keys = set(parts[0])
            if not {"reward", "cost_ttc"}.issubset(keys):
                raise ValueError("team_reward must provide reward and cost_ttc")
            components = {key: torch.stack([part[key] for part in parts]).reshape(
                cfg.horizon, STEPS_PER_CONTROL, -1).sum(1) for key in keys}
            if credit_profile and model_index == 0 and sample_index == 0:
                profile = _credit_profile(components["reward"], records[0]["credit_action"],
                                          records[0]["credit_mask"])
            member_components.append(components)
            member_rewards.append(components["reward"])
            member_costs.append(components["cost_ttc"])
            member_states.append(physics_states[::STEPS_PER_CONTROL])
            member_entropy.append(torch.stack([record["entropy"] for record in records]))
            member_actions.append(torch.stack([record["action"] for record in records]))
            if self.use_lanes:
                member_lanes.append({name: torch.stack([record[name] for record in records])
                                     for name in ("lane_log_prob", "lane_entropy", "lane_active",
                                                  "lane_acting", "lane_changed")})
        _restore_rng(advanced_rng, self.device)
        rewards, costs = torch.stack(member_rewards), torch.stack(member_costs)
        disagreement = ensemble_penalty(rewards, 1.0)
        adjusted = rewards - cfg.uncertainty_weight * disagreement.unsqueeze(0)
        returns = []
        for e, states in enumerate(member_states):
            # Freeze critic weights, not state derivatives: the terminal bootstrap
            # must provide analytic actor credit through the imagined WM trajectory.
            values = torch.stack([value_call(state) for state in states])
            member_values.append(values)
            returns.append(lambda_returns(adjusted[e], values, cfg.gamma, cfg.lambda_))
        masks = torch.stack([torch.stack([state["present"] & state["is_cav"]
                                         for state in states[:-1]]) for states in member_states])
        if act_mask is not None:
            masks = masks & act_mask
        lanes = None
        if member_lanes:
            lanes = {name: torch.stack([m[name] for m in member_lanes]) for name in member_lanes[0]}
        return {"rewards": rewards, "costs": costs, "returns": torch.stack(returns),
                **({"credit_profile": profile} if profile is not None else {}),
                "values": torch.stack(member_values),
                "disagreement": disagreement, "states": member_states,
                "entropy": torch.stack(member_entropy), "actions": torch.stack(member_actions),
                "action_mask": masks, "lanes": lanes, "variants": variants, "n_base": n_base,
                "n_members": len(self.world_models), "n_futures": n_futures,
                "components": {key: torch.stack([c[key] for c in member_components])
                               for key in member_components[0]}}

    def update_lagrange(self, mean_cost):
        value = float(mean_cost)
        if not math.isfinite(value):
            raise ValueError("nonfinite safety cost")
        self.lagrange_multiplier = max(0.0, self.lagrange_multiplier + self.config.dual_lr *
                                      (value - self.config.ttc_budget))
        return self.lagrange_multiplier

    def _return_scale(self, returns):
        quantiles = torch.quantile(returns.detach().float().flatten(),
                                   torch.tensor([0.05, 0.95], device=self.device))
        low, high = map(float, quantiles)
        if self.return_low is None:
            self.return_low, self.return_high = low, high
        else:
            decay = self.config.return_ema
            self.return_low = decay * self.return_low + (1 - decay) * low
            self.return_high = decay * self.return_high + (1 - decay) * high
        return max(1.0, self.return_high - self.return_low)

    def actor_objective(self, imagined, scale, weights):
        """Actor objective to MINIMISE (without the accel entropy / Lagrangian terms).

        The maximised objective is ``sum_r c_r * mean_{E,H}(R_r - B_r) / scale`` with the row
        coefficients ``c`` of :func:`wms.control.counterfactual.row_coefficients` (``w_all *
        D_all + w_group * D_G + w_ind * D_i + w_coop * C_G + w_team * M_i`` averaged over starts;
        ``B_r`` = the base return of row r's start, constant w.r.t. the actor).  Accelerations get
        analytic gradients through the WM.  Lane head: REINFORCE with the same coefficients,
        ``-sum_r c_r * mean_{E,H}[A_r(t) * sum_k log pi(lane_k)]`` over acting CAV decisions with
        > 1 feasible choice, ``A = (R - V)/scale`` on all-CAV rows (critic baseline) and ``(R -
        B)/scale`` on counterfactual rows (common random numbers), plus ``lane_entropy_weight`` x
        the masked categorical entropy.  Without lanes / counterfactual rows this reduces exactly
        to the original ``-mean(R)/scale``.
        """
        n_base = imagined["n_base"]
        returns = imagined["returns"]                     # [E,H,N]
        n_rows = returns.shape[-1]
        variants = imagined.get("variants")
        coefficients = row_coefficients(variants, weights, n_rows, n_base, device=returns.device,
                                        dtype=returns.dtype)
        base = baseline_index(variants, n_rows, device=returns.device)
        has_base = base != torch.arange(n_rows, device=returns.device)
        # R_r - B_r for rows whose start has a base row (B constant w.r.t. the actor), else R_r.
        difference = returns - torch.where(has_base, returns[..., base].detach(), torch.zeros_like(returns))
        loss = -(difference.mean((0, 1)) * coefficients).sum() / scale
        extra = {}
        if variants is not None and variants.base_row:
            extra = DreamerTrainer._cooperation_metrics(imagined, variants, weights)
        lanes = imagined.get("lanes")
        if lanes is not None:
            advantage = difference.detach() / scale
            values = imagined.get("values")
            if values is not None:
                advantage[..., :n_base] = (returns[..., :n_base] - values[:, :-1, :n_base]).detach() / scale
            active = lanes["lane_active"]                 # [E,H,N,K]
            joint = (lanes["lane_log_prob"] * active.to(returns.dtype)).sum(-1)   # [E,H,N]
            pg_loss = -((joint * advantage).mean((0, 1)) * coefficients).sum()
            lane_entropy = _masked_mean(lanes["lane_entropy"], active)
            # V6EXPLORE: a learned lane temperature replaces the fixed weight when a target is set.
            lane_weight = (self.log_lane_alpha.detach().exp()
                           if getattr(self, "log_lane_alpha", None) is not None
                           else self.config.lane_entropy_weight)
            loss = loss + self.config.lane_pg_weight * pg_loss - lane_weight * lane_entropy
            acting = lanes["lane_acting"][..., :n_base, :]
            changed = lanes["lane_changed"][..., :n_base, :] & acting
            extra.update({"lane_pg_loss": float(pg_loss.detach()),
                          "lane_entropy": float(lane_entropy.detach()),
                          "lane_change_rate": float(changed.sum() / acting.sum().clamp_min(1)),
                          "lane_decisions_with_options": int(active[..., :n_base, :].sum())})
        return loss, extra

    @staticmethod
    def _cooperation_metrics(imagined, variants, weights):
        """B, D_i, D_G, D_all, C_G, M_i, I_i (start-of-rollout lambda-returns, ensemble mean), the
        fraction of groups with C_G > 0, by group size and by penetration bucket."""
        parts = decompose(imagined["returns"][:, 0].detach().mean(0), variants)
        out = {"cf_rows": variants.n_rows - variants.n_starts, "cf_weights": dict(weights)}
        for name in ("B", "D_all", "D_i", "D_G", "C_G", "M_i", "I_i"):
            out[name] = mean_or_none(parts[name])
        out["coop_positive_fraction"] = mean_or_none(float(c > 0) for c in parts["C_G"])
        out["n_groups"] = len(parts["D_G"])
        out["n_C_G"] = len(parts["C_G"])
        out["mean_group_size"] = mean_or_none(g["size"] for g in parts["groups"])
        buckets = imagined.get("penetration_buckets") or {}
        for key, label in (("by_group_size", lambda g: str(g["size"])),
                           ("by_penetration", lambda g: buckets.get(g["start"], "unknown"))):
            table = {}
            for group in parts["groups"]:
                table.setdefault(label(group), []).append(group)
            out[key] = {name: {"n": len(rows), "D_G": mean_or_none(g["D_G"] for g in rows),
                               "C_G": mean_or_none(g["C_G"] for g in rows if g["C_G"] is not None),
                               "coop_positive_fraction": mean_or_none(
                                   float(g["C_G"] > 0) for g in rows if g["C_G"] is not None)}
                        for name, rows in sorted(table.items())}
        return out

    def _sample_starts(self, sampler):
        """Group-rich oversampling (``group_state_fraction`` of the batch satisfies
        :func:`wms.control.counterfactual.group_rich`) when the sampler supports it."""
        cf = self.counterfactual
        if cf.enabled and cf.group_state_fraction > 0 and hasattr(sampler, "sample_where"):
            return sampler.sample_where(self.config.batch_size, device=self.device,
                                        predicate=lambda state: group_rich(state, cf),
                                        fraction=cf.group_state_fraction, max_tries=cf.max_group_tries)
        return sampler.sample(self.config.batch_size, device=self.device)

    @staticmethod
    def _penetration_buckets(batch):
        """Logging label per start: world-id penetration (p02..p40), else the observed CAV share."""
        from .start_states import penetration_of

        out = {}
        present = batch.state["present"].bool()
        share = ((present & batch.state["is_cav"].bool()).sum(-1).float()
                 / present.sum(-1).clamp_min(1).float())
        for index in range(present.shape[0]):
            meta = batch.metadata[index] if index < len(batch.metadata) else {}
            try:
                value = penetration_of(meta)
                out[index] = f"p{round(value * 100):02d}"
            except (ValueError, TypeError):
                nearest = min((0.02, 0.05, 0.1, 0.2, 0.4), key=lambda p: abs(p - float(share[index])))
                out[index] = f"share~p{round(nearest * 100):02d}"
        return out

    def _entry_slots(self, batch, generator=None):
        if not self.config.imagine_entries:
            return batch
        horizon_s = self.config.horizon * STEPS_PER_CONTROL * PHYSICS_DT_S
        return with_entry_slots(batch, horizon_s, self.config.entry_pad_max, generator)

    @staticmethod
    def _wait_metrics(imagined):
        """Per-control-step on-road delay, source wait and total wait (ensemble/horizon/all-CAV
        row means) and, with counterfactual base rows, total wait all-CAV minus all-native
        (summed over the horizon, mean over starts; negative = less waiting than native)."""
        comps, n_base = imagined["components"], imagined["n_base"]
        delay = comps["delay"]
        source = comps.get("source_wait", torch.zeros_like(delay))
        total = delay + source
        out = {"delay": float(delay[..., :n_base].mean()),
               "source_wait": float(source[..., :n_base].mean()),
               "total_wait": float(total[..., :n_base].mean())}
        variants = imagined.get("variants")
        if variants is not None and variants.base_row:
            per_row = total.sum(1).mean(0)                         # [N]
            diffs = [float(per_row[s] - per_row[r]) for s, r in variants.base_row.items()]
            out["total_wait_vs_native"] = sum(diffs) / len(diffs)
        return out

    def update(self):
        started = time.perf_counter()
        self.policy.train()
        self.group_encoder.train()
        self.critic.train()
        # Entry seeds from a per-update generator: enabling entries leaves the global RNG stream
        # (variant choice, policy noise) unchanged, and a resumed run draws the same seeds.
        batch = self._entry_slots(self._sample_starts(self.start_states),
                                  torch.Generator().manual_seed(1_000_003 * (self.update_count + 1)))
        variants = (build_variants(batch.state, self.counterfactual)
                    if self.counterfactual.enabled else None)
        profile_due = (self.config.credit_profile_every > 0
                       and (self.update_count + 1) % self.config.credit_profile_every == 0)
        imagined = self.imagine(batch, variants=variants, **({"credit_profile": True}
                                                            if profile_due else {}))
        imagined["penetration_buckets"] = self._penetration_buckets(batch)
        n_base, n_rows = imagined["n_base"], imagined["returns"].shape[-1]
        all_returns = imagined["returns"]
        returns = all_returns[..., :n_base]
        scale = self._return_scale(returns)
        weights = self.counterfactual.weights(self.update_count)
        entropy = imagined["entropy"][..., :n_base].mean()
        mean_cost = imagined["costs"][..., :n_base].mean()
        objective, extra = self.actor_objective(imagined, scale, weights)
        # V6EXPLORE: a learned entropy temperature replaces the fixed weight when a target is set.
        entropy_weight = (self.log_alpha.detach().exp() if self.log_alpha is not None
                          else self.config.entropy_weight)
        actor_loss = (objective - entropy_weight * entropy +
                      self.lagrange_multiplier * mean_cost)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        # clip_grad_norm_ returns the PRE-clip norm (logged below), including when V6
        # raises the imagination decay above the WM value. Keep the existing clip only.
        actor_norm = nn.utils.clip_grad_norm_(self.actor_parameters, self.config.grad_clip,
                                             error_if_nonfinite=True)
        self.actor_optimizer.step()
        # V6EXPLORE: dual steps on the entropy temperature(s), logged with the entropy above.
        extra.update(self._alpha_update(imagined, entropy))
        # The critic estimates the value of the all-CAV policy: fit it on the all-CAV rows.
        predictions = torch.stack([torch.stack([self.critic(_detach(_rows(state, n_base, n_rows)))
                                                for state in states[:-1]])
                                   for states in imagined["states"]])
        critic_loss = ((predictions - returns.detach()) / scale).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_norm = nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip,
                                              error_if_nonfinite=True)
        self.critic_optimizer.step()
        with torch.no_grad():
            for target, source in zip(self.target_critic.parameters(), self.critic.parameters()):
                target.lerp_(source, self.config.target_tau)
            for target, source in zip(self.target_critic.buffers(), self.critic.buffers()):
                if target.is_floating_point():
                    target.lerp_(source, self.config.target_tau)
                else:
                    target.copy_(source)
        self.update_lagrange(mean_cost.detach())
        self.update_count += 1
        selected = imagined["actions"][..., :n_base, :].detach()[imagined["action_mask"][..., :n_base, :]]
        action_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        if selected.numel():
            action_stats = {"mean": float(selected.mean()), "std": float(selected.std(unbiased=False)),
                            "min": float(selected.min()), "max": float(selected.max())}
        flags = [m.get("group_rich") for m in batch.metadata if isinstance(m, dict)]
        if any(flag is not None for flag in flags):
            extra["group_rich_fraction"] = mean_or_none(float(bool(f)) for f in flags)
        metrics = {"update": self.update_count, "returns": float(returns.detach().mean()),
                   **({"credit_profile": imagined["credit_profile"]} if profile_due else {}),
                   "reward": float(imagined["rewards"][..., :n_base].detach().mean()),
                   **extra,
                   "actor_loss": float(actor_loss.detach()), "critic_loss": float(critic_loss.detach()),
                   "lagrange_multiplier": self.lagrange_multiplier, "entropy": float(entropy.detach()),
                   "actor_grad_norm": float(actor_norm), "critic_grad_norm": float(critic_norm),
                   "ensemble_disagreement": float(imagined["disagreement"][..., :n_base].detach().mean()),
                   "return_scale": scale, "action_stats": action_stats,
                   **({"future_samples": imagined["n_futures"]} if imagined["n_futures"] > 1
                      or self.config.future_accel_noise else {}),
                   "wall_time_s": time.perf_counter() - started}
        for name in ("delay", "throughput", "smooth", "hard_brake", "fairness_delay", "cost_ttc",
                     "source_wait"):
            value = imagined["components"].get(name)
            metrics["fairness" if name == "fairness_delay" else name] = (
                float(value[..., :n_base].detach().mean()) if value is not None else 0.0)
        # Total wait per control step = on-road speed-deficit delay + entry (source) waiting.
        metrics["total_wait"] = metrics["delay"] + metrics["source_wait"]
        metrics["entries"] = bool(getattr(batch, "entry_spec", None) is not None
                                  and self.config.imagine_entries)
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            with (self.output_dir / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            if self.config.checkpoint_every and self.update_count % self.config.checkpoint_every == 0:
                self.save_checkpoint(self.output_dir / f"policy_{self.update_count:06d}.pt")
        return metrics

    def validate(self, sampler, batches=1):
        """Selection score on separate VAL windows (deterministic policy, no update): mean D_all
        (all-CAV minus all-native lambda-return, start-of-rollout) when counterfactuals are on,
        else the mean imagined lambda-return.  See :meth:`validate_detail`."""
        return self.validate_detail(sampler, batches)["score"]

    def validate_detail(self, sampler, batches=1):
        """``{"score", "returns", "D_all", "coop_gain_val", "n_groups"}`` on VAL windows.

        With counterfactuals the VAL starts are sampled group-rich like training and every start
        gets base / individual / group / flip rows (a fixed generator, so the variant choice is
        reproducible); ``coop_gain_val`` = mean C_G.  Training RNG, sampler state and module
        modes are restored."""
        if batches < 1:
            raise ValueError("validation requires at least one batch")
        rng = _rng_state(self.device)
        modes = (self.policy.training, self.group_encoder.training)
        self.policy.eval()
        self.group_encoder.eval()
        cf = self.counterfactual
        returns, d_all, coop, n_groups, waits = [], [], [], 0, []
        try:
            with torch.no_grad():
                generator = torch.Generator().manual_seed(12345)
                entry_generator = torch.Generator().manual_seed(54321)
                for batch_index in range(batches):
                    batch = self._entry_slots(self._sample_starts(sampler), entry_generator)
                    variants = build_variants(batch.state, cf, generator) if cf.enabled else None
                    # Fixed sampled futures (same worlds at every validation) when enabled.
                    imagined = self.imagine(batch, deterministic=True, variants=variants,
                                            future_seed=f"val{batch_index}")
                    returns.append(float(imagined["returns"][..., :imagined["n_base"]].mean()))
                    waits.append(self._wait_metrics(imagined))
                    if variants is not None and variants.base_row:
                        parts = decompose(imagined["returns"][:, 0].mean(0), variants)
                        d_all.extend(parts["D_all"])
                        coop.extend(parts["C_G"])
                        n_groups += len(parts["D_G"])
            mean_return = sum(returns) / len(returns)
            out = {"returns": mean_return, "D_all": mean_or_none(d_all),
                   "coop_gain_val": mean_or_none(coop), "n_groups": n_groups}
            for key in ("delay", "source_wait", "total_wait", "total_wait_vs_native"):
                out[key] = mean_or_none(w[key] for w in waits if key in w)
            out["score"] = out["D_all"] if out["D_all"] is not None else mean_return
            if not math.isfinite(out["score"]):
                raise ValueError("nonfinite validation return")
            return out
        finally:
            _restore_rng(rng, self.device)
            self.policy.train(modes[0])
            self.group_encoder.train(modes[1])

    def fit(self, max_updates, *, validation_states=None, validation_batches=1):
        """Train; best checkpoint ``policy_best.pt`` by the VAL score (D_all with counterfactuals)
        and ``policy_best_coop.pt`` by the VAL cooperation gain ``coop_gain_val``."""
        self.validation_states = validation_states
        if self._pending_validation_state is not None and validation_states is not None:
            if (getattr(self, "_loop_refresh_resume", False)
                    and not _same_groups(validation_states, self._pending_validation_state)):
                print("[loop-refresh] validation windows changed; sampler state not restored", flush=True)
            else:
                validation_states.load_state_dict(self._pending_validation_state)
            self._pending_validation_state = None
        last = None
        while self.update_count < max_updates:
            if self.output_dir is not None and (self.output_dir / "STOP").exists():
                # C4 Dreamer loop (scripts/c4/dreamer_loop.py): a STOP file in the output directory ends
                # training cleanly after the current update; policy_latest.pt is written below.
                print(f"[stop-file] STOP found at update {self.update_count}; saving policy_latest.pt",
                      flush=True)
                break
            last = self.update()
            if (validation_states is not None and self.config.validation_every and
                    self.update_count % self.config.validation_every == 0):
                detail = self.validate_detail(validation_states, validation_batches)
                score = detail["score"]
                if self.output_dir is not None:
                    with (self.output_dir / "validation.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"update": self.update_count, **detail},
                                                allow_nan=False) + "\n")
                if score > self.best_validation:
                    self.best_validation = score
                    if self.output_dir is not None:
                        self.save_checkpoint(self.output_dir / "policy_best.pt")
                coop = detail["coop_gain_val"]
                if coop is not None and coop > self.best_coop_validation:
                    self.best_coop_validation = coop
                    if self.output_dir is not None:
                        self.save_checkpoint(self.output_dir / "policy_best_coop.pt")
        if self.output_dir is not None:
            self.save_checkpoint(self.output_dir / "policy_latest.pt")
        return last

    def save_checkpoint(self, path):
        path = checked_path(path)
        from .policy import accel_bounds_payload
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": CHECKPOINT_FORMAT, "config": asdict(self.config),
                   "lane_rules": asdict(self.lane_rules) if self.lane_rules is not None else None,
                   "lane_head": self.use_lanes, "counterfactual": self.counterfactual.as_dict(),
                   "reward_weights": asdict(self.reward_weights),
                   "policy": self.policy.state_dict(), "group_encoder": self.group_encoder.state_dict(),
                   **accel_bounds_payload(self.policy),
                   "critic": self.critic.state_dict(), "target_critic": self.target_critic.state_dict(),
                   "actor_optimizer": self.actor_optimizer.state_dict(),
                   "critic_optimizer": self.critic_optimizer.state_dict(),
                   **({"log_alpha": self.log_alpha.detach().cpu(),
                       "alpha_optimizer": self.alpha_optimizer.state_dict()}
                      if self.log_alpha is not None else {}),
                   **({"log_lane_alpha": self.log_lane_alpha.detach().cpu(),
                       "lane_alpha_optimizer": self.lane_alpha_optimizer.state_dict()}
                      if self.log_lane_alpha is not None else {}),
                   "update_count": self.update_count, "lagrange_multiplier": self.lagrange_multiplier,
                   "return_low": self.return_low, "return_high": self.return_high,
                   "best_validation": self.best_validation, "rng": _rng_state(self.device),
                   "best_coop_validation": self.best_coop_validation,
                   "ensemble_size": len(self.world_models),
                   "world_model_ids": self.world_model_ids,
                   "validation_sampler": (self.validation_states.state_dict()
                                          if hasattr(self.validation_states, "state_dict") else None),
                   "sampler": self.start_states.state_dict() if hasattr(self.start_states, "state_dict") else None}
        torch.save(payload, path)

    def load_checkpoint(self, path, *, policy_only=False):
        payload = torch.load(checked_path(path), map_location=self.device, weights_only=True)
        if payload.get("format") not in READABLE_FORMATS:
            raise ValueError(f"expected a {' or '.join(READABLE_FORMATS)} checkpoint")
        if not policy_only and payload["format"] != CHECKPOINT_FORMAT:
            raise ValueError(f"full resume requires a {CHECKPOINT_FORMAT} checkpoint (use policy_only)")
        loop_refresh = os.environ.get("WMS_LOOP_REFRESH", "") == "1"
        if not policy_only and (payload["ensemble_size"] != len(self.world_models) or
                                (payload["world_model_ids"] != self.world_model_ids
                                 and not loop_refresh)):
            raise ValueError("resume requires the same frozen world-model ensemble")
        if not policy_only and loop_refresh and payload["world_model_ids"] != self.world_model_ids:
            # C4 Dreamer loop (2026-10-16): continue the full trainer (policy, critic, optimisers, duals)
            # on a REFRESHED world model of the same ensemble size; recorded in the run's log.
            print(f"[loop-refresh] resuming on new WM ids {self.world_model_ids} "
                  f"(was {payload['world_model_ids']})", flush=True)
        from .policy import check_payload_accel_bounds, upgrade_policy_state
        configured = (getattr(self.policy, "accel_min", None), getattr(self.policy, "accel_max", None))
        missing, unexpected = self.policy.load_state_dict(upgrade_policy_state(payload["policy"]),
                                                          strict=False)
        if unexpected or any(not key.startswith("lane_head.") for key in missing):
            raise ValueError(f"policy weights do not match: missing={missing} unexpected={unexpected}")
        check_payload_accel_bounds(self.policy, payload)
        loaded = (getattr(self.policy, "accel_min", None), getattr(self.policy, "accel_max", None))
        if loaded != configured:
            # The checkpoint's stored bound wins (legacy [-3, 1.5] when it has none): the same
            # weights must decode the same actions.  Re-decoding needs policy.set_accel_bounds.
            print(f"WARNING: {path}: policy accel bounds {list(loaded)} from the checkpoint override "
                  f"the configured {list(configured)} (ACCELCAP_NOTES.md)", file=sys.stderr, flush=True)
        # A c4_dreamer/1 actor has no lane head: the head keeps its uniform initialisation.
        self.group_encoder.load_state_dict(payload["group_encoder"])
        if policy_only:
            return
        self.config = DreamerConfig(**payload["config"])
        self.reward_weights = RewardWeights(**payload["reward_weights"])
        # V6EXPLORE: the loaded config may activate targets the run config did not have.
        self._init_entropy_temperature()
        for name in ("critic", "target_critic", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(payload[name])
        for key, parameter, optimizer_name, label in (
                ("log_alpha", self.log_alpha, "alpha_optimizer", "entropy"),
                ("log_lane_alpha", self.log_lane_alpha, "lane_alpha_optimizer", "lane entropy")):
            if key not in payload:
                if parameter is not None:
                    print(f"WARNING: checkpoint predates the {label} temperature; "
                          f"{key} restarts at log(weight) (V6EXPLORE)", file=sys.stderr, flush=True)
                continue
            if parameter is None:
                raise ValueError(f"checkpoint carries {key} but the loaded config sets no target")
            with torch.no_grad():
                parameter.copy_(payload[key].to(parameter.device))
            getattr(self, optimizer_name).load_state_dict(payload[optimizer_name])
        for name in ("update_count", "lagrange_multiplier", "return_low", "return_high", "best_validation"):
            setattr(self, name, payload[name])
        self.best_coop_validation = payload.get("best_coop_validation", -math.inf)
        if loop_refresh and payload["world_model_ids"] != self.world_model_ids:
            # Validation scores on a refreshed WM are not comparable with the old WM's: the new run
            # directory keeps its own policy_best*.pt (selected on this WM only).
            self.best_validation = self.best_coop_validation = -math.inf
        self._loop_refresh_resume = loop_refresh
        if payload["sampler"] is not None:
            if loop_refresh and not _same_groups(self.start_states, payload["sampler"]):
                # Loop iterations add the policy's own SUMO episodes to the start windows: the window
                # set changed, so the sampler continues with this run's fresh RNG.
                print("[loop-refresh] start windows changed; sampler state not restored", flush=True)
            else:
                self.start_states.load_state_dict(payload["sampler"])
        self._pending_validation_state = payload.get("validation_sampler")
        _restore_rng(payload["rng"], self.device)
