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
from dataclasses import asdict, dataclass, field, replace
from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from wms.core import speed_flags
from wms.physics.batched import road_end_exit_bound

from .contract import ACCEL_MAX_MPS2, PHYSICS_DT_S, STEPS_PER_CONTROL, RewardWeights
from .counterfactual import (KIND_ALL, CounterfactualConfig, baseline_index, build_variants,
                             decompose, expand_rows, group_rich, mean_or_none, native_fn,
                             row_coefficients)
from .entrance import DEFAULT_ENTRANCE_ZONE_M, make_entrance_zone_fn
from .futures import FuturesConfig, SampledFutures, lane_uniform_draws
from .real_returns14 import RealReturnBatch, RealReturnProvider
from .start_states import StartBatch, checked_path

# /2 (2026-10-12 late): lane head, lane rules, counterfactuals; /3: reports carry the other CAVs'
# last commanded acceleration and lane intent (older actors load with zero weights on them).
CHECKPOINT_FORMAT = "c4_dreamer/3"
READABLE_FORMATS = ("c4_dreamer/1", "c4_dreamer/2", CHECKPOINT_FORMAT)
MEMORY_CONFIG_FIELDS = ("imagination_microbatch", "imagination_checkpoint_segment",
                        "imagination_offload_futures", "imagination_offload_saved_tensors")


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
    # CAVLANE (2026-10-19; default True = current behaviour): allow the POLICY to issue CAV lane
    # actions in imagination.  The lane path is active only when this is True AND lane rules are
    # supplied AND the policy has a lane head (the existing condition).  Setting it False keeps
    # CAVs in lane (the bridge's forced ramp merge still runs), which is what
    # ``[policy] lane_control = "native"`` already does for the v7..v10 policy chain; the switch
    # lets the training loop disable policy lane actions independently of the actor/config.
    policy_lane_actions: bool = True
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
    # LONGMEM: default-off exact full-objective VJP replay, in start microbatches.
    imagination_microbatch: int = 0
    # Physics steps per checkpoint segment; zero disables segmented checkpointing.
    imagination_checkpoint_segment: int = 0
    # Keep replay's constant random futures on the host between forward/backward.
    imagination_offload_futures: bool = False
    # Host-spill saved autograd tensors as well (useful when the effective batch is one).
    imagination_offload_saved_tensors: bool = False
    # S-LONGSKIP: exact output-cotangent checks; independent, default OFF.
    imagination_skip_zero_vjp: bool = False
    imagination_replay_dense_prefix: bool = False
    # S-LONGGRAD: optimizer preconditioning only; the registered loss is unchanged.
    # Cap long-credit actor gradient norm at the dense-term norm before addition.
    long_grad_balance: bool = False
    # Explicit new-phase recovery of contaminated actor Adam history; not used
    # during updates or ordinary resume. Model weights/update count are retained.
    long_grad_reset_actor_adam: bool = False
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
    # V7POL (owner 2026-10-17; defaults = bit-identical).
    # Safety budget: "absolute" = mean all-CAV-row TTC cost vs ``ttc_budget`` (as before);
    # "native_relative" = per start, the policy row's cost vs its counterfactual all-native base
    # row (+ ``ttc_native_margin``), absolute ``ttc_budget`` for starts without a base row; the
    # actor penalty is lambda * mean_s relu(excess_s) (no credit for being safer than native) and
    # the dual ascends on mean_s excess_s over the native-referenced starts (all starts if none).
    ttc_budget_mode: str = "absolute"
    ttc_native_margin: float = 0.0
    # Divide the Lagrangian actor term by the same return scale as the objective.
    lagrange_scaled: bool = False
    # Entropy floor: alpha = exp(log_alpha) (and the lane temperature) never below this value;
    # log_alpha itself is clamped after every dual step, so recovery starts from the floor.
    alpha_min: float = 0.0
    # V8POL (2026-10-17; defaults = bit-identical). ``lagrange_max``: cap on the TTC multiplier
    # (None = uncapped); with a cap the update log gets ``lagrange_capped`` and
    # ``lagrange_pre_cap`` (this step's value before the cap).
    # ``lagrange_decay``: lambda <- (1 - decay) * lambda before each dual ascent step (0 = off).
    lagrange_max: Optional[float] = None
    lagrange_decay: float = 0.0
    # V9ENTRY (2026-10-18; default off = bit-identical): the imagined entrance (source) queue
    # starts from the recorded SUMO backlog at t0 and arrivals follow the demand (request) rate
    # instead of the observed insertion rate; a lane's same-step entrant gets no end-of-step
    # credit (EntryConfig.fresh_no_credit).  Needs start windows with ``entry_queue``
    # (converted sources with queue arrays) and imagine_entries; refuses batches without it.
    entry_queue_state: bool = False
    entrant_control: bool = False  # explicit continuity contract; ALL already controls births
    deterministic_crn: bool = False  # pair merge streams for every actor, including MAPPO
    # V9NOYIELD (2026-10-18; default empty = bit-identical): PhysicsConfig field -> value
    # overrides applied to EVERY PhysicsConfig holder of each frozen WM member right after it is
    # loaded (same holder-walk as tools/wm_eval_common.py / coop_cl_noyield.py), so the imagined
    # rollouts run with the override without touching the checkpoint.  The intended use is
    # ``{mainline_merge_yield = false}``: the WM's mainline yield rule deadlocks the merge zone and
    # penalises any CAV that deviates from its native CAV (ONROAD_DIAG.md).
    wm_physics_overrides: dict = field(default_factory=dict)
    # V10ENTZONE (2026-10-19; default off = bit-identical): a CAV inside an entrance zone (the
    # first ``entrance_zone_main_m`` metres downstream of the lane-0/1 insertion points and
    # ``entrance_zone_ramp_m`` for the ramp) drives native inside imagination -- its policy action
    # is replaced by the native/IDM twin command and the step is excluded from the actor loss.
    # Same zones as the SUMO autocheck (entrance_mask/entrance_mask_loader.py); the policy must
    # learn under the restriction it is evaluated with so it cannot win by metering entrances.
    entrance_zone_native: bool = False
    entrance_zone_main_m: float = DEFAULT_ENTRANCE_ZONE_M
    entrance_zone_ramp_m: float = DEFAULT_ENTRANCE_ZONE_M
    # POLICY14: critic-only fitting of authoritative negative-total-vehicle-second
    # returns. Convex mix of independently mean-reduced real/imagined MSE. The
    # real residual uses the return scale with the floor below. Zero never calls the provider.
    # Enable only with imagined rewards in these same raw units (no per-car mean).
    real_return_mix: float = 0.0
    # Independent real-target residual scale floor; raw critic values stay in veh-s.
    real_return_scale: float = 1.0
    # Extra autograd passes for weighted actor-loss components, plus pre-safety
    # acceleration saturation/entropy in m/s^2. False adds no diagnostic graph work.
    actor_gradient_audit: bool = False
    # Native residual representation learning; independent, default-off switches.
    native_residual_train_group: bool = False
    native_residual_train_trunk: bool = False
    # POLICY14 evidence guards; legacy trainers retain their original behaviour.
    low_penetration_horizon_cap: bool = False
    wm_safety_report_only: bool = False
    # S-ANTIEXPLOIT: fixed registered q10 paired reward credit; no dynamics changes.
    pessimistic_credit: bool = False
    native_anchor: bool = False
    native_anchor_calibration: Optional[str] = None
    # S-EXPLOIT: epistemic protection and diagnostics; independent default-OFF.
    exploit_ensemble_pessimism: bool = False
    exploit_monitor: bool = False
    exploit_stop_on_conflict: bool = False
    exploit_head_rmse_mps2: float = 0.0
    exploit_disable_direct_credit: bool = False
    # S-C4PREP AMENDMENT2: preserve a registered, nonzero individual baseline
    # when real-credit multipliers would otherwise erase cooperation entirely.
    real_credit_individual_floor: bool = False

    def __post_init__(self):
        for name in ('pessimistic_credit', 'native_anchor', 'real_credit_individual_floor',
                     'exploit_ensemble_pessimism', 'exploit_monitor', 'exploit_stop_on_conflict',
                     'exploit_disable_direct_credit'):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f'{name} must be a boolean')
        if (self.exploit_ensemble_pessimism or self.exploit_monitor) and not self.pessimistic_credit:
            raise ValueError('exploit safeguards require pessimistic_credit=true')
        if self.exploit_stop_on_conflict and not self.exploit_ensemble_pessimism:
            raise ValueError('exploit_stop_on_conflict requires ensemble pessimism')
        if self.pessimistic_credit or self.native_anchor:
            if self.future_samples not in (8, 16):
                raise ValueError('anti-exploit requires future_samples=8')
        if self.native_anchor and (not isinstance(self.native_anchor_calibration, str)
                                   or not self.native_anchor_calibration):
            raise ValueError('native_anchor requires a fitted calibration artifact')
        for name in ("imagination_microbatch", "imagination_checkpoint_segment"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("imagination_offload_futures", "imagination_offload_saved_tensors",
                     "imagination_skip_zero_vjp", "imagination_replay_dense_prefix",
                     "long_grad_balance", "long_grad_reset_actor_adam"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if (isinstance(self.real_return_mix, bool)
                or not isinstance(self.real_return_mix, (int, float))
                or not math.isfinite(self.real_return_mix) or not 0 <= self.real_return_mix <= 1):
            raise ValueError("real_return_mix must be a finite number in [0, 1]")
        if (isinstance(self.real_return_scale, bool)
                or not isinstance(self.real_return_scale, (int, float))
                or not math.isfinite(self.real_return_scale) or self.real_return_scale <= 0):
            raise ValueError("real_return_scale must be a finite positive number")
        if not isinstance(self.actor_gradient_audit, bool):
            raise ValueError("actor_gradient_audit must be a boolean")
        for name in ("native_residual_train_group", "native_residual_train_trunk"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
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
                        "entropy_target", "lane_entropy_target", "ttc_budget_mode",
                         "lagrange_max", "wm_physics_overrides", "native_anchor_calibration") or value is None:
                continue   # the string fields; the optional decay / entropy targets; the overrides
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
        if self.lagrange_max is not None and (
                isinstance(self.lagrange_max, bool) or not isinstance(self.lagrange_max, (int, float))
                or not math.isfinite(self.lagrange_max) or self.lagrange_max < 0):
            raise ValueError("lagrange_max must be None or a finite nonnegative number")
        if self.lagrange_decay >= 1:
            raise ValueError("lagrange_decay must be in [0, 1)")
        if self.ttc_budget_mode not in ("absolute", "native_relative"):
            raise ValueError("ttc_budget_mode must be 'absolute' or 'native_relative'")
        for name in ("checkpoint_imagination", "imagine_entries", "future_accel_noise",
                     "source_wait_grad", "critic_entry_backlog", "lagrange_scaled",
                     "entry_queue_state", "entrant_control", "deterministic_crn",
                     "entrance_zone_native", "policy_lane_actions",
                     "low_penetration_horizon_cap", "wm_safety_report_only"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.entrance_zone_native and not (
                math.isfinite(self.entrance_zone_main_m) and self.entrance_zone_main_m > 0.0
                and math.isfinite(self.entrance_zone_ramp_m) and self.entrance_zone_ramp_m > 0.0):
            raise ValueError("entrance_zone_native needs positive entrance_zone_main_m / "
                             "entrance_zone_ramp_m")
        if not isinstance(self.wm_physics_overrides, dict):
            raise ValueError("wm_physics_overrides must be a mapping of PhysicsConfig field "
                             "names to values")
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
        self.requires_state = getattr(fn, 'requires_state', False)

    def __call__(self, *args, **kwargs):
        if self.enabled and torch.is_grad_enabled():
            return activation_checkpoint(self.fn, *args, use_reentrant=False,
                                         preserve_rng_state=self.preserve_rng, **kwargs)
        return self.fn(*args, **kwargs)


class _ImmediateHolder:
    """Static (non-tensor) call structure for :class:`_ImmediateVJP`."""

    def __init__(self, fn, args):
        from torch.utils._pytree import tree_flatten
        self.fn = fn
        leaves, self.spec = tree_flatten(args)
        self.leaves = list(leaves)
        self.index = [i for i, leaf in enumerate(leaves) if torch.is_tensor(leaf)]
        self.out_spec = None

    def tensors(self):
        return [self.leaves[i] for i in self.index]

    def fill(self, tensors):
        from torch.utils._pytree import tree_unflatten
        leaves = list(self.leaves)
        for i, tensor in zip(self.index, tensors):
            leaves[i] = tensor
        return tree_unflatten(leaves, self.spec)


class _ImmediateVJP(torch.autograd.Function):
    """Pure-call activation recompute whose backward runs INSIDE one autograd node.

    S-LONGMEM300 (memory only, exact): ``checkpoint(use_reentrant=False)`` recomputes a frame when
    its first saved tensor is unpacked but frees the recomputed tensors only when every node of the
    frame has run. For the 600 per-step reward calls the engine unpacks hundreds of frames before the
    pairwise-TTC branches execute (16 GiB of [K,K] intermediates at 300 s). Here the forward runs
    under no_grad and the backward recomputes AND differentiates one call at a time, so at most one
    call's intermediates are live. Same ops on the same inputs: values and gradients are unchanged.
    """

    @staticmethod
    def forward(ctx, holder, *tensors):
        from torch.utils._pytree import tree_flatten
        out = holder.fn(*holder.fill(tensors))
        flat, holder.out_spec = tree_flatten(out)
        if not all(torch.is_tensor(x) for x in flat):
            raise ValueError('immediate recompute supports tensor-only outputs')
        ctx.holder = holder
        ctx.save_for_backward(*tensors)
        ctx.mark_non_differentiable(*(x for x in flat if not x.is_floating_point()))
        return tuple(flat)

    @staticmethod
    def backward(ctx, *grads):
        from torch.utils._pytree import tree_flatten
        holder, saved = ctx.holder, ctx.saved_tensors
        need = ctx.needs_input_grad[1:]
        with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
            inputs = [t.detach().requires_grad_(bool(n) and t.is_floating_point())
                      for t, n in zip(saved, need)]
            flat, _ = tree_flatten(holder.fn(*holder.fill(inputs)))
            pairs = [(o, g) for o, g in zip(flat, grads) if g is not None and o.requires_grad]
            wrt = [t for t in inputs if t.requires_grad]
            found = (torch.autograd.grad([o for o, _ in pairs], wrt, [g for _, g in pairs],
                                         allow_unused=True) if pairs and wrt else ())
        it = iter(found)
        return (None, *(next(it) if t.requires_grad else None for t in inputs))


class _ImmediateRecompute:
    """Drop-in for :class:`_Recompute` over a pure positional call (see :class:`_ImmediateVJP`)."""

    def __init__(self, fn, enabled):
        self.fn, self.enabled = fn, bool(enabled)
        self.requires_state = getattr(fn, 'requires_state', False)

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise ValueError('immediate recompute takes positional arguments only')
        if not (self.enabled and torch.is_grad_enabled()):
            return self.fn(*args)
        from torch.utils._pytree import tree_unflatten
        holder = _ImmediateHolder(self.fn, args)
        flat = _ImmediateVJP.apply(holder, *holder.tensors())
        return tree_unflatten(list(flat), holder.out_spec)


def _physics_holders(model):
    """Every ``(owner, attr)`` in the model graph that holds a PhysicsConfig instance.

    Same walk as ``tools/wm_eval_common.py`` / ``coop_cl_noyield.py``: an override must replace
    every reference (``model.physics`` and any submodule attribute), not just the top-level one.
    """
    from wms.physics.config import PhysicsConfig
    found, seen = [], set()

    def visit(obj):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        for name, value in list(vars(obj).items()):
            if isinstance(value, PhysicsConfig):
                found.append((obj, name))
        for sub in getattr(obj, "_modules", {}).values():
            if sub is not None:
                visit(sub)

    visit(model)
    return found


def apply_wm_physics_overrides(model, overrides):
    """Swap every PhysicsConfig holder of ``model`` for a copy with ``overrides`` applied.

    V9NOYIELD: ``{"mainline_merge_yield": False}`` turns the WM's mainline yield rule off inside
    imagination.  An empty mapping is a no-op (default path stays bit-identical) and the frozen
    member's loaded physics fields are remembered in a plain dict so repeated applications always
    start from the checkpoint's configuration.
    Returns the applied overrides and how many holders were replaced.
    """
    from wms.physics.config import PhysicsConfig
    changes = dict(overrides or {})
    holders = _physics_holders(model)
    if not changes:
        return {"overrides": {}, "holders": len(holders)}
    unknown = sorted(set(changes) - set(PhysicsConfig.__dataclass_fields__))
    if unknown:
        raise ValueError(f"unknown WM physics override fields {unknown}")
    if not holders:
        raise ValueError("no PhysicsConfig reference found on the model")
    # Remember the loaded fields as a plain dict (NOT a PhysicsConfig instance) so the walk above
    # never picks it up and repeated applications always start from the checkpoint's physics.
    base_fields = getattr(model, "_wm_physics_base_fields", None)
    if base_fields is None:
        base_fields = {name: getattr(model.physics, name)
                       for name in PhysicsConfig.__dataclass_fields__}
        model._wm_physics_base_fields = base_fields
    new = replace(PhysicsConfig(**base_fields), **changes)
    for owner, name in holders:
        setattr(owner, name, new)
    return {"overrides": {key: getattr(new, key) for key in changes}, "holders": len(holders)}


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


def _entropy_temperature(parameter, fixed, config):
    """V6EXPLORE/V7POL entropy weight: ``exp(log_alpha)`` floored at ``config.alpha_min`` (when > 0),
    or the fixed weight when no temperature is learned."""
    if parameter is None:
        return fixed
    alpha = parameter.detach().exp()
    floor = float(getattr(config, "alpha_min", 0.0))
    return alpha.clamp_min(floor) if floor > 0.0 else alpha


_ENTRY_PAD_SAFETY = 1.5
_ENTRY_PAD_MARGIN = 16
_PAD_FILL = {"lane": -1, "length_m": 5.0}


def _pad_slots(value, k, pad, fill=0):
    """Append ``pad`` slots along dim 1 of a [B, K, ...] tensor (other values unchanged)."""
    if not (torch.is_tensor(value) and value.ndim >= 2 and value.shape[1] == k):
        return value
    shape = (value.shape[0], pad, *value.shape[2:])
    return torch.cat((value, torch.full(shape, fill, dtype=value.dtype, device=value.device)), 1)


#: V9ENTRY: extra entrant slots for a t0 backlog, at most this many veh/s beyond the demand.
_ENTRY_PAD_BACKLOG_RATE = 0.5


def with_entry_slots(batch: StartBatch, horizon_s: float, pad_max: int, generator=None,
                     queue_state: bool = False):
    """Start batch with empty entrant slots appended and its entry spec (``batch.entry_spec``).

    The pad holds 1.5x the expected arrivals (observed entry-history rate x horizon) + 16, the
    batch maximum, capped at ``pad_max``; one process seed per start (rows of a start share it).
    Returns ``batch`` unchanged when it has no ``entry_feat``.

    ``queue_state`` (V9ENTRY): the spec also carries one ``EntranceQueue`` per start (t0 backlog,
    demand rate) from ``batch.entry_queue`` (required), and the pad is sized from the demand
    plus the backlog (at most ``_ENTRY_PAD_BACKLOG_RATE`` veh/s of extra service)."""
    if batch.entry_feat is None:
        return batch
    from wms.model.entries import EntranceHistory, EntranceQueue

    feats = batch.entry_feat.detach().cpu().numpy()
    histories = [EntranceHistory.from_features(row) for row in feats]
    k = batch.state["s_m"].shape[1]
    queues = None
    if queue_state:
        if batch.entry_queue is None:
            raise ValueError("entry_queue_state needs start windows with entry_queue "
                             "(a converted source with V9ENTRY queue arrays)")
        queues = [EntranceQueue.from_vector(row) for row in
                  batch.entry_queue.detach().cpu().numpy()]
        pad = max(int(math.ceil((float(np.sum(q.demand_rate)) * horizon_s
                                 + min(float(np.sum(q.backlog)),
                                       _ENTRY_PAD_BACKLOG_RATE * horizon_s))
                                * _ENTRY_PAD_SAFETY)) + _ENTRY_PAD_MARGIN for q in queues)
    else:
        pad = max(int(math.ceil(float(np.sum(h.rate)) * horizon_s * _ENTRY_PAD_SAFETY))
                  + _ENTRY_PAD_MARGIN for h in histories)
    pad = max(1, min(int(pad_max), pad))
    seeds = torch.randint(0, 2 ** 31 - 1, (len(histories),), generator=generator).tolist()
    state = {key: _pad_slots(value, k, pad, _PAD_FILL.get(key, 0))
             for key, value in batch.state.items()}
    kwargs = {key: _pad_slots(value, k, pad) for key, value in batch.rollout_kwargs.items()}
    spec = {"histories": histories, "seeds": seeds, "k0": k, "pad": pad}
    if queues is not None:
        spec["queues"] = queues
    return StartBatch(state, kwargs, batch.metadata, batch.entry_feat, spec,
                      entry_queue=batch.entry_queue)


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


def _entry_process(spec, rows, n_slots, insert_rule: str = "v2", learned_entry=None, *,
                   couple_draws: bool = False, sample_index: int = 0):
    from wms.model.entries import EntryConfig, TorchEntryProcessV2

    pool = np.zeros((len(rows), n_slots), dtype=bool)
    pool[:, spec["k0"]:] = True
    queues = spec.get("queues")         # V9ENTRY (absent = unchanged)
    # V9ENTRY: coupled draws = common random numbers across the counterfactual rows of a start
    # (arrival times and candidate identities keyed by (row seed, entrance, index), as in WM paired
    # training).  A shared row seed alone loses CRN as soon as one row defers an insertion; with a
    # standing backlog every arrival-count difference becomes a queue (source-wait) difference.
    extra = ({"couple_draws": True} if couple_draws else {}) if queues is None else {
        "queues": [queues[r] for r in rows], "couple_draws": True}
    cfg = (EntryConfig(max_pending=10 ** 9, insert_rule=insert_rule,
                       learned_entry=learned_entry) if queues is None else
           EntryConfig(max_pending=10 ** 9, insert_rule=insert_rule, fresh_no_credit=True,
                       learned_entry=learned_entry))
    # Uncapped backlog: every due arrival that cannot enter keeps waiting (as in SUMO).
    return TorchEntryProcessV2(histories=[spec["histories"][r] for r in rows],
                               dt_s=PHYSICS_DT_S, seed=0, batch=len(rows),
                               cfg=cfg, pool=pool,
                                row_seeds=[spec["seeds"][r] + 1000003 * sample_index for r in rows], **extra)


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
                 counterfactual=None, real_return_provider: RealReturnProvider | None = None,
                 uncertainty=None, exploration=None, horizon=None):
        self.config = config or DreamerConfig()
        # S-COOPW [exploration] (wms.control.coopw.ExplorationConfig; None/disabled = unchanged).
        self.exploration = exploration if exploration is not None and exploration.enabled else None
        # S-COOPW [horizon] (wms.control.coopw.HorizonConfig; gate resolved by HorizonConfig.bind).
        self.horizon_curriculum = horizon if horizon is not None and horizon.enabled else None
        # S-COOPW [uncertainty] (wms.control.coopw.UncertaintyConfig; None/disabled = unchanged).
        self.uncertainty = uncertainty if uncertainty is not None and uncertainty.enabled else None
        self._training_imagination = False
        self.longcritic = None  # S-LONGCRITIC: injected only by an enabled run section.
        self.paircrit = None  # S-PAIRCRIT: default-OFF signed cost-advantage regression.
        self.coop_anchor = None
        self._last_start_info = None
        self.real_return_provider = real_return_provider
        self._check_real_return_provider(self.config)
        # Lane head (owner 2026-10-12 late): active when lane rules are supplied AND the
        # policy has a categorical lane head; trained by REINFORCE (accelerations stay analytic).
        self.lane_rules = lane_rules
        # CAVLANE switch: False disables the policy lane path (CAVs stay in lane; forced ramp merge
        # still applies) without changing the actor or the rest of the trainer.
        self.use_lanes = (bool(self.config.policy_lane_actions) and lane_rules is not None
                          and bool(getattr(policy, "has_lane_head", False)))
        # V7POL ``[policy] lane_control = "native"``: an actor WITHOUT a lane head (attribute set by
        # train_cli); imagination keeps CAVs in lane (bridge forced ramp merge only) and the
        # checkpoint records it so the SUMO executor leaves CAV lane changes to SUMO.
        self.lane_control = getattr(policy, "lane_control", "learned")
        if self.lane_control not in ("learned", "native"):
            raise ValueError("lane_control must be 'learned' or 'native'")
        if self.lane_control == "native" and (self.use_lanes
                                              or getattr(policy, "has_lane_head", False)):
            raise ValueError("lane_control='native' requires a policy without a lane head "
                             "and no lane rules")
        self.counterfactual = counterfactual or CounterfactualConfig()
        self.device = torch.device(device)
        if not world_models:
            raise ValueError("at least one world model is required")
        self.world_models = [model.to(self.device).eval() for model in world_models]
        self.wm_physics_overrides = []
        for model in self.world_models:
            model.requires_grad_(False)
            model.zero_grad(set_to_none=True)
            dt = getattr(getattr(model, "physics", None), "dt_s", PHYSICS_DT_S)
            if not math.isclose(dt, PHYSICS_DT_S):
                raise ValueError("WM physics dt does not match the shared control contract")
            # V9NOYIELD: apply the configured physics switch to EVERY PhysicsConfig holder of
            # this member before its identity is fixed (empty overrides = bit-identical).
            applied = apply_wm_physics_overrides(model, self.config.wm_physics_overrides)
            self.wm_physics_overrides.append(applied)
            if applied["overrides"]:
                print(f"[wm-physics] member {len(self.wm_physics_overrides) - 1}: "
                      f"{applied['overrides']} applied to {applied['holders']} PhysicsConfig "
                      f"holder(s); effective mainline_merge_yield="
                      f"{model.physics.mainline_merge_yield}", flush=True)
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
        if getattr(self.policy, "residual_on_native", False):
            from .nativeinit import paired_reward
            self.reward_fn = paired_reward(reward_fn)
        self.controller_factory = controller_factory
        if getattr(self.policy, "residual_on_native", False) and controller_factory is None:
            raise ValueError("native residual requires the native-aware bridge controller factory")
        self.reward_weights = reward_weights or RewardWeights()
        self.actor_parameters = list(dict.fromkeys(
            list(self.policy.parameters()) + list(self.group_encoder.parameters())))
        if getattr(self.policy, "residual_on_native", False):
            self.policy.configure_native_training(group=self.config.native_residual_train_group,
                                                  trunk=self.config.native_residual_train_trunk)
            self.group_encoder.requires_grad_(self.config.native_residual_train_group)
            self.actor_parameters = [p for p in self.actor_parameters if p.requires_grad]
        elif self.config.native_residual_train_group or self.config.native_residual_train_trunk:
            raise ValueError("native residual training flags require native residual semantics")
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
        self._tilt_sampler = None
        if self.counterfactual.enabled and self.counterfactual.start_tilt is not None:
            from .coopw import TiltSampler
            self._tilt_sampler = TiltSampler(self.counterfactual.start_tilt)
        if self.counterfactual.enabled and self.counterfactual.headroom_starts:
            from .coopw import HeadroomSampler
            self._tilt_sampler = HeadroomSampler()

    def _check_real_return_provider(self, config):
        if self.real_return_provider is not None and not callable(self.real_return_provider):
            raise ValueError("real_return_provider must be callable or None")
        if config.real_return_mix > 0 and self.real_return_provider is None:
            raise ValueError("real_return_mix > 0 requires a real_return_provider")

    def _real_return_batch(self):
        """Provider boundary: no actor/WM graph can enter critic regression."""
        if self.config.real_return_mix == 0:
            return None
        self._check_real_return_provider(self.config)
        with torch.no_grad():
            batch = self.real_return_provider(batch_size=self.config.batch_size,
                                              device=self.device, update=self.update_count + 1)
            if not isinstance(batch, RealReturnBatch):
                raise ValueError("real_return_provider must return a RealReturnBatch")
            return batch.detached(device=self.device, gamma=self.config.gamma)

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

    def _clamp_log_alpha(self, parameter):
        """V7POL entropy floor: keep exp(log_alpha) >= ``alpha_min`` (no-op when 0)."""
        floor = float(getattr(self.config, "alpha_min", 0.0))
        if parameter is not None and floor > 0.0:
            with torch.no_grad():
                parameter.clamp_(min=math.log(floor))

    def _temperature(self, parameter, fixed):
        """Entropy weight in use: exp(log_alpha) floored at ``alpha_min``, else the fixed weight."""
        return _entropy_temperature(parameter, fixed, self.config)

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
            self._clamp_log_alpha(self.log_alpha)
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
            self._clamp_log_alpha(self.log_lane_alpha)
            out.update({"lane_alpha": float(self.log_lane_alpha.detach().exp()),
                        "lane_entropy_target": float(self.config.lane_entropy_target)})
        return out

    def _imagination_config(self, batch):
        """Cap low-support starts using observed present-CAV share, never metadata.

        A mixed batch conservatively shares its shortest allowed horizon so every
        ensemble/counterfactual row stays paired. Empty roads count as zero share;
        imagined entries cannot retroactively extend the start-state horizon.
        """
        cfg = self.config
        if self._training_imagination and self.horizon_curriculum is not None:
            # S-COOPW gated horizon curriculum: medium/high observed CAV share only (the low-share
            # test below keeps the base horizon for <= 10 % shares).
            present = batch.state["present"].bool()
            cav_count = (present & batch.state["is_cav"].bool()).sum(-1)
            if not bool((cav_count * 10 <= present.sum(-1)).any()):
                h = self.horizon_curriculum
                from .gated300 import score_position
                seconds = h.horizons_s[h.stage(score_position(self))]
                steps = int(round(seconds / (STEPS_PER_CONTROL * PHYSICS_DT_S)))
                if steps > cfg.horizon:
                    cfg = replace(cfg, horizon=steps)
        if cfg.low_penetration_horizon_cap:
            present = batch.state["present"].bool()
            cav_count = (present & batch.state["is_cav"].bool()).sum(-1)
            low = cav_count * 10 <= present.sum(-1)
            if bool(low.any()):
                max_steps = math.floor(60.0 / (STEPS_PER_CONTROL * PHYSICS_DT_S))
                cfg = replace(cfg, horizon=min(cfg.horizon, max_steps))
        if self._training_imagination and self.uncertainty is not None:
            # S-COOPW: sampled driver futures + the pessimism curriculum, training imagination only.
            u = self.uncertainty
            cfg = replace(cfg, future_samples=u.future_samples,
                          future_driver_scale=u.driver_noise_scale,
                          uncertainty_weight=u.weight(self._curriculum_position(advance=False)))
        if (self._training_imagination and getattr(self, 'long_credit_requested', False)
                and cfg.horizon > getattr(self, 'long_credit_trusted_s', 60)):
            cfg = replace(cfg, future_samples=8, future_driver_scale=1., uncertainty_weight=0.)
        return cfg

    def _curriculum_position(self, advance=True):
        """Counterfactual (and uncertainty) curriculum position: ``update_count`` (legacy) or, with
        ``curriculum_anchor = "first_update"``, the updates since this config first trained."""
        if self.counterfactual.curriculum_anchor != "first_update":
            return self.update_count
        if self.coop_anchor is None:
            if not advance:
                return 0
            self.coop_anchor = self.update_count
        return self.update_count - self.coop_anchor

    def imagine(self, batch: StartBatch, deterministic=False, variants=None, future_seed=None,
                credit_profile=False, difference_only=False, _memory_replay=False,
                _memory_config=None, _memory_full_batch=None, _memory_start_index=None,
                _memory_future_cache=None, _memory_active_future=None, _memory_row_index=None,
                _memory_collect_future=None, _memory_replay_horizon=None):
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
        cfg = _memory_config or self._imagination_config(batch)
        from . import gated300
        gated_rollout = gated300.requested_rollout(self)
        if gated300.measures_ctrl0(self):
            from .execution import mode
            if mode() != 2 or not getattr(self.policy, 'residual_on_native', False):
                raise ValueError('CTRL0 reference requires mode-2 native-residual execution')
            variants = gated300.add_ctrl0_reference(variants)
        dense_steps = self._imagination_config(batch).horizon if gated_rollout else cfg.horizon
        if _memory_config is None:
            cfg = gated300.rollout_config(self, cfg)
            if self._training_imagination:
                from .eslong import execution_config
                cfg = execution_config(self, cfg, batch)
        memory = (cfg.imagination_microbatch or cfg.imagination_checkpoint_segment
                  or cfg.imagination_offload_futures or cfg.imagination_offload_saved_tensors)
        if memory and torch.is_grad_enabled() and not _memory_replay:
            from .imagination_memory import memory_imagine
            if credit_profile:
                raise ValueError("LONGMEM requires credit_profile_every=0 (internal-action diagnostic)")
            return memory_imagine(self, batch, cfg, deterministic=deterministic,
                                  variants=variants, future_seed=future_seed,
                                  difference_only=difference_only)
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
        # V10ENTZONE: dynamic per-tick predicate; None keeps the default path bit-identical.
        zone_fn = (make_entrance_zone_fn(cfg.entrance_zone_main_m, cfg.entrance_zone_ramp_m)
                   if cfg.entrance_zone_native else None)
        need_bridge = self.controller_factory is not None and (
            self.use_lanes or act_mask is not None or zone_fn is not None
            or getattr(self.policy, "residual_on_native", False))
        crn = act_mask is not None and bool(getattr(self.policy, "supports_noise", False))
        # S-COOPW correlated exploration: training imagination only, stochastic, CRN path.
        explore = None
        if (crn and not deterministic and self._training_imagination and self.exploration is not None):
            from .coopw import exploration_plan
            rho = self.exploration.rho_at(self._curriculum_position(advance=False))
            if rho > 0.0:
                radius = self.exploration.radius_m or self.counterfactual.group_radius_m
                gid, shared = exploration_plan(
                    (_memory_full_batch or batch).state, radius, cfg.horizon,
                    self.exploration.persistence_at(
                        self._curriculum_position(advance=False)),
                    torch.Generator().manual_seed(3_000_017 * (self.update_count + 1)))
                own = None
                if self.exploration.mode == "additive_accel":
                    from .coopw import own_exploration_plan
                    state_b = (_memory_full_batch or batch).state
                    own = own_exploration_plan(
                        state_b["s_m"].shape[0], state_b["s_m"].shape[1], cfg.horizon,
                        self.exploration.persistence_at(self._curriculum_position(advance=False)),
                        torch.Generator().manual_seed(5_000_011 * (self.update_count + 1))
                    ).to(state_b["s_m"].device)
                if _memory_start_index is not None:
                    gid = gid.index_select(0, _memory_start_index)
                    shared = shared.index_select(1, _memory_start_index)
                    if own is not None:
                        own = own.index_select(1, _memory_start_index)
                explore = (gid, shared, rho) if own is None else (
                    gid, shared, rho, own, self.exploration.additive_correlation)
        recompute = cfg.checkpoint_imagination and torch.is_grad_enabled() and not memory
        from wms.model.cav import apply_exec_cap
        native = _Recompute(native_fn(self.counterfactual.native_cav,
                                      a_max=self.counterfactual.native_accel_max_mps2,
                                      head_path=self.counterfactual.native_cav_head_path), recompute)
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
        # S-LONGMEM300 (memory only, default OFF, exact): the per-step reward/TTC cost builds [B,K,K]
        # pairwise saved tensors OUTSIDE the WM segment checkpoints; in LONGMEM they were spilled to
        # host RAM (hundreds of GB at 300 s). Recomputing the pure reward in backward keeps only its inputs.
        _lm300 = os.environ.get('WMS_LM300_RECOMPUTE_REWARD', '')
        if bool(memory) and torch.is_grad_enabled() and _lm300 == '2':
            reward_call = _ImmediateRecompute(self.reward_fn, True)
        else:
            reward_call = _Recompute(self.reward_fn, recompute or (
                bool(memory) and torch.is_grad_enabled() and _lm300 == '1'))
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
        member_zone = []
        member_action_audit = []
        member_anchor = []
        profile = None
        rng = _rng_state(self.device)
        advanced_rng = None
        collections = dict(states=member_states, rewards=member_rewards, costs=member_costs,
            entropy=member_entropy, actions=member_actions, components=member_components,
            lanes=member_lanes, zone=member_zone, action_audit=member_action_audit, anchor=member_anchor)
        if _memory_collect_future is not None:
            from .imagination_memory import _current_tape
            _memory_future_cache['tape_prefix'] = _current_tape().position
        elif _memory_future_cache is not None and _memory_active_future is None:
            from .eslong import active as es_active
            if es_active(self):
                from .imagination_memory import _current_tape
                _memory_future_cache['eslong_tape_prefix'] = _current_tape().position
        # Common random numbers: disagreement must measure WM differences, not action noise.
        for future_index, (model_index, sample_index) in enumerate(
                [(e, s) for e in range(len(self.world_models)) for s in range(n_futures)]):
            if _memory_collect_future is not None and future_index != _memory_collect_future:
                continue
            if _memory_active_future is not None and future_index != _memory_active_future:
                from .imagination_memory import replay_cached_future
                total_rows = _memory_future_cache['items'][0]['payload']['rewards'].shape[1]
                replay_cached_future(_memory_future_cache, future_index, collections,
                                     _memory_row_index, total_rows)
                continue
            model = self.world_models[model_index]
            # Keep cfg.horizon for ALL stochastic plans. Only the active WM
            # execution is shortened; cached tails restore full output geometry.
            replay_steps = _memory_replay_horizon or cfg.horizon
            future = {} if future_extras is None else future_extras[model_index][sample_index]
            _restore_rng(rng, self.device)
            records = []
            step = 0
            held_out = None
            last_decision_step = -1
            zone_now = None

            def policy_proxy(obs, signal, **kwargs):
                nonlocal held_out, last_decision_step
                if step % STEPS_PER_CONTROL == 0 and last_decision_step != step:
                    if crn:
                        from .policy import PolicyNoise, draw_policy_noise
                        noise = draw_policy_noise((n_base, obs.cav_mask.shape[-1]), device=self.device)
                        if explore is not None and len(explore) == 5:
                            # additive_accel: eps [n_base,K] in m/s^2 (std = rho), added to the residual
                            from .coopw import additive_noise
                            gid, shared, rho, own, corr = explore
                            t = min(step // STEPS_PER_CONTROL, shared.shape[0] - 1)
                            noise = PolicyNoise(additive_noise(gid, shared[t], own[t], rho, corr)
                                                .to(noise.accel), noise.lane_gumbel)
                        elif getattr(self.policy, 'residual_additive_exploration', False):
                            # additive mode: outside the exploration plan (validation/scoring/rho 0)
                            # the unit-variance CRN draw must NOT enter the residual (m/s^2).
                            noise = PolicyNoise(torch.zeros_like(noise.accel), noise.lane_gumbel)
                        elif explore is not None:
                            from .coopw import correlate
                            gid, shared, rho = explore
                            noise = PolicyNoise(correlate(noise.accel, gid, shared[min(
                                step // STEPS_PER_CONTROL, shared.shape[0] - 1)], rho),
                                noise.lane_gumbel)
                        kwargs["noise"] = PolicyNoise(noise.accel.index_select(0, index),
                                                      noise.lane_gumbel.index_select(0, index))
                    held_out = policy_call(obs, signal, deterministic=deterministic, **kwargs)
                    acting = obs.cav_mask if act_mask is None else obs.cav_mask & act_mask
                    if zone_now is not None:      # V10ENTZONE: zone CAVs excluded from the actor loss
                        acting = acting & ~zone_now
                    log_prob = getattr(held_out, "log_prob_accel", None)
                    log_prob = held_out[1] if log_prob is None else log_prob
                    record = records[-1]
                    if credit_profile and model_index == 0 and sample_index == 0 and step == 0:
                        record["credit_action"] = held_out[0]
                        record["credit_mask"] = acting
                    record["entropy"] = -(log_prob * acting).sum(-1) / acting.sum(-1).clamp_min(1)
                    if cfg.pessimistic_credit or cfg.native_anchor:
                        # Actor-objective telemetry only; do not alter entrant control.
                        if getattr(self.policy, 'residual_on_native', False):
                            record['anchor_deviation'] = held_out[0]
                            record['anchor_mask'] = acting.detach()
                        elif cfg.native_anchor:
                            raise ValueError('native_anchor requires a native-residual actor')
                    if cfg.actor_gradient_audit or getattr(self.policy, "residual_on_native", False):
                        record["audit_action"] = held_out[0].detach()
                        record["audit_log_prob"] = log_prob.detach()
                        record["audit_acting"] = acting.detach()
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
            policy_proxy.residual_on_native = bool(getattr(self.policy, "residual_on_native", False))
            policy_proxy.residual_reference = getattr(self.policy, "residual_reference", "base")
            if gated300.measures_ctrl0(self):
                from .counterfactual import KIND_CTRL0, KIND_BASE
                # With CTRL0 credit, a non-intervened CAV (including an entrant)
                # means residual zero under the same executor, not native release.
                # The explicit dense learned-native baseline row is retained.
                policy_proxy.ctrl0_rows = (variants.row_kind != KIND_BASE if
                    gated_rollout and gated300.uses_ctrl0(self) else variants.row_kind == KIND_CTRL0)

            def lane_record(entry):
                records[-1]["lane_changed"] = entry["lane_changed"].detach()

            external_callback = None
            if need_bridge:
                extra = {}
                if self.use_lanes:
                    extra["lane_rules"] = self.lane_rules
                if act_mask is not None or zone_fn is not None or policy_proxy.residual_on_native:
                    extra["native"] = native
                    if _accepts(self.controller_factory, "native_accel_max"):
                        extra["native_accel_max"] = native_max
                if act_mask is not None:
                    extra["act_mask"] = act_mask
                if zone_fn is not None:
                    extra["zone"] = zone_fn
                external_callback = self.controller_factory(
                    policy_proxy, group_call, safety_call,
                    lane_record if self.use_lanes else None, **extra)
            elif self.controller_factory is not None:
                external_callback = self.controller_factory(policy_proxy, group_call,
                                                            safety_call, None)

            def controller(state):
                nonlocal step, zone_now
                trace = getattr(self, '_longgrad_state_trace', None)
                if trace is not None:
                    trace.watch(state, step)
                mask = state["present"].bool() & state["is_cav"].bool()
                if step % STEPS_PER_CONTROL == 0:
                    records.append({"mask": mask})
                    if zone_fn is not None:
                        zone_now = zone_fn(state)
                        records[-1]["entrance_zone"] = zone_now
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
                acting = None if act_mask is None else act_mask
                if zone_fn is not None:      # V10ENTZONE: zone CAVs released to the native twin
                    acting = (torch.ones_like(command, dtype=torch.bool)
                              if acting is None else acting)
                    acting = acting & ~zone_now
                if acting is not None:       # counterfactual rows / entrance zones: native CAVs
                    command = torch.where(acting, command, native(state))
                    ceiling = torch.where(acting, torch.full_like(command, policy_max),
                                          torch.full_like(command, native_max))
                accel = safety_call(state, command, accel_max=ceiling)
                cav = state["present"].bool() & state["is_cav"].bool()
                accel = torch.where(cav, accel, torch.zeros_like(accel))
                # EXECCAP_NOTES.md: policy CAVs only (identity unless the WM's cav_exec_cap is on).
                accel = apply_exec_cap(state, accel, cav if acting is None else cav & acting)
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

            if external_callback is not None and hasattr(external_callback, "native_lane_mask"):
                controller.native_lane_mask = external_callback.native_lane_mask
            initial = {key: value.clone() if torch.is_tensor(value) else value
                       for key, value in start_state.items()}
            if cfg.deterministic_crn or getattr(self.policy, "residual_on_native", False):
                from .nativeinit import pair_merge_streams
                pair_merge_streams(initial, index, n_base,
                                   (self.update_count if future_seed is None else future_seed) + sample_index)
            # Plain-bug fix: imagined vehicles leave at the road end (same gate as evaluation
            # and training); physics.exit_at_road_end=false keeps old replays bit-identical.
            exit_s = road_end_exit_bound(getattr(model, "physics", None))
            if exit_s is None:
                initial.pop("exit_s_m", None)
            else:
                initial.setdefault("exit_s_m", torch.full_like(
                    initial["s_m"], float(exit_s)))
            recorder = None
            member_kwargs = {**rollout_kwargs, **future}
            sample_lanes = lane_uniform
            if cfg.deterministic_crn and sample_index:
                sample_lanes = lane_uniform_draws(
                    self.world_models, n_base, cfg.horizon * STEPS_PER_CONTROL,
                    batch.state["s_m"].shape[1],
                    (self.update_count if future_seed is None else future_seed) + 1000003 * sample_index,
                    device=self.device)
                if sample_lanes is not None and index is not None:
                    sample_lanes = expand_rows(sample_lanes, index, n_base)
            if sample_lanes is not None:
                member_kwargs["lane_uniform"] = sample_lanes
            if entry_spec is not None:
                learned = None
                if bool(getattr(getattr(model, "physics", None), "entry_learned", False)):
                    from wms.model.entry_learned import load_learned_entry

                    learned = load_learned_entry()
                recorder = _EntryRecorder(_entry_process(entry_spec, entry_rows,
                                                         initial["s_m"].shape[1],
                                                         cfg.entry_insert_rule,
                                                          learned_entry=learned,
                                                          couple_draws=cfg.deterministic_crn,
                                                          sample_index=sample_index if cfg.deterministic_crn else 0))
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
                if cfg.imagination_checkpoint_segment and torch.is_grad_enabled():
                    from .imagination_memory import segmented_rollout
                    roll = segmented_rollout(
                        model, initial, replay_steps * STEPS_PER_CONTROL,
                        cfg.imagination_checkpoint_segment, controller, member_kwargs)
                else:
                    roll = model.rollout(initial, n_steps=replay_steps * STEPS_PER_CONTROL,
                                         cav_controller=controller, cav_teacher=None,
                                         truncation_steps=None, **member_kwargs)
            finally:
                if switch:
                    model.checkpoint_forward = previous_switch
                if cap_switch:
                    model.cav_exec_cap = previous_cap
                if decay_switch:
                    model.grad_decay_per_step = previous_decay
            if step != replay_steps * STEPS_PER_CONTROL:
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
                replay_steps, STEPS_PER_CONTROL, -1).sum(1) for key in keys}
            if credit_profile and model_index == 0 and sample_index == 0:
                profile = _credit_profile(components["reward"], records[0]["credit_action"],
                                          records[0]["credit_mask"])
            member_components.append(components)
            member_rewards.append(components["reward"])
            member_costs.append(components["cost_ttc"])
            member_states.append(physics_states[::STEPS_PER_CONTROL])
            member_entropy.append(torch.stack([record["entropy"] for record in records]))
            member_actions.append(torch.stack([record["action"] for record in records]))
            if 'anchor_deviation' in records[0]:
                member_anchor.append({name: torch.stack([record['anchor_' + name] for record in records])
                                      for name in ('deviation', 'mask')})
            if cfg.actor_gradient_audit or getattr(self.policy, "residual_on_native", False):
                member_action_audit.append({name: torch.stack([record[name] for record in records])
                                            for name in ("audit_action", "audit_log_prob",
                                                         "audit_acting")})
            if zone_fn is not None:
                member_zone.append(torch.stack([record["entrance_zone"] for record in records]))
            if self.use_lanes:
                member_lanes.append({name: torch.stack([record[name] for record in records])
                                     for name in ("lane_log_prob", "lane_entropy", "lane_active",
                                                   "lane_acting", "lane_changed")})
            if _memory_future_cache is not None and _memory_active_future is None:
                from .imagination_memory import capture_future
                capture_future(_memory_future_cache, collections)
            elif replay_steps < cfg.horizon:
                from .imagination_memory import restore_cached_tail
                restore_cached_tail(_memory_future_cache, future_index, collections,
                                    _memory_row_index, dense_steps)
        if _memory_future_cache is not None:
            if _memory_active_future is None:
                _memory_future_cache['advanced_rng'] = advanced_rng
            else:
                advanced_rng = _memory_future_cache['advanced_rng']
        _restore_rng(advanced_rng, self.device)
        rewards, costs = torch.stack(member_rewards), torch.stack(member_costs)
        disagreement = ensemble_penalty(rewards, 1.0)
        adjusted = rewards - cfg.uncertainty_weight * disagreement.unsqueeze(0)
        returns = []
        for e, states in enumerate(member_states):
            # Freeze critic weights, not state derivatives: the terminal bootstrap
            # must provide analytic actor credit through the imagined WM trajectory.
            # Long no-grad target construction consumes raw paired rewards only.
            # Avoid hundreds of irrelevant target-critic calls/bootstraps there.
            if difference_only or getattr(self, 'long_credit_requested', False) or gated_rollout:
                if difference_only and torch.is_grad_enabled():
                    raise ValueError("difference_only imagination must run under no_grad")
                values = rewards.new_zeros((len(states), rewards.shape[-1]))
            else:
                values = torch.stack([value_call(state) for state in states])
                if self.paircrit is not None and variants is not None:
                    # Native continuation has identically zero advantage. Its
                    # absolute value/gradient must never re-enter paired credit.
                    from .counterfactual import KIND_BASE
                    values = torch.where((variants.row_kind != KIND_BASE)[None, :], values, 0.)
            member_values.append(values)
            if getattr(self, 'long_credit_requested', False) or gated_rollout:
                # Accrued finite-horizon WM credit only. Neither TD mixing nor
                # terminal values may carry a real 300s target into imagination.
                returns.append(torch.flip(torch.flip(rewards[e], (0,)).cumsum(0), (0,)))
            else:
                returns.append(lambda_returns(adjusted[e], values, cfg.gamma, cfg.lambda_))
        masks = torch.stack([torch.stack([state["present"] & state["is_cav"]
                                         for state in states[:-1]]) for states in member_states])
        if act_mask is not None:
            masks = masks & act_mask
        zone_masks = None
        if member_zone:
            zone_masks = torch.stack(member_zone)
            masks = masks & ~zone_masks
        lanes = None
        if member_lanes:
            lanes = {name: torch.stack([m[name] for m in member_lanes]) for name in member_lanes[0]}
        # V10ENTZONE: fraction of all-CAV-row CAV-steps released to native inside an entrance zone
        # (the imagination analogue of the SUMO autocheck's masked fraction).
        zone_fraction = None
        if zone_masks is not None:
            cav = torch.stack([torch.stack([state["present"] & state["is_cav"]
                                            for state in states[:-1]]) for states in member_states])
            cav = cav[..., :n_base, :]
            zone_base = zone_masks[..., :n_base, :]
            zone_fraction = float((zone_base & cav).sum()) / max(1, int(cav.sum()))
        result = {"rewards": rewards, "costs": costs, "returns": torch.stack(returns),
                **({'antiexploit_commands': {name: torch.stack([m[name] for m in member_anchor])
                                             for name in ('deviation', 'mask')}} if member_anchor else {}),
                "imagination_horizon_s": cfg.horizon * STEPS_PER_CONTROL * PHYSICS_DT_S,
                "deterministic_policy": deterministic,
                **({"action_audit": {name: torch.stack([m[name] for m in member_action_audit])
                                     for name in member_action_audit[0]}}
                   if member_action_audit else {}),
                **({"credit_profile": profile} if profile is not None else {}),
                "values": torch.stack(member_values),
                "disagreement": disagreement, "states": member_states,
                "entropy": torch.stack(member_entropy), "actions": torch.stack(member_actions),
                "action_mask": masks, "lanes": lanes, "variants": variants, "n_base": n_base,
                "entrance_zone": zone_masks, "entrance_zone_fraction": zone_fraction,
                "n_members": len(self.world_models), "n_futures": n_futures,
                "uncertainty_weight": cfg.uncertainty_weight,
                **({"exploration_rho": explore[2]} if explore is not None else {}),
                "components": {key: torch.stack([c[key] for c in member_components])
                                for key in member_components[0]}}
        return gated300.prefix(self, result, dense_steps) if gated_rollout else result

    def update_lagrange(self, mean_cost, budget=None):
        """Dual ascent ``lambda += dual_lr * (mean_cost - budget)`` (budget default ``ttc_budget``;
        native_relative passes the mean excess over native with budget 0)."""
        value = float(mean_cost)
        if not math.isfinite(value):
            raise ValueError("nonfinite safety cost")
        if self.config.wm_safety_report_only:
            self._lagrange_pre_cap = self.lagrange_multiplier
            self._lagrange_capped = False
            return self.lagrange_multiplier
        budget = self.config.ttc_budget if budget is None else float(budget)
        current = self.lagrange_multiplier
        decay = float(getattr(self.config, "lagrange_decay", 0.0))
        if decay > 0.0:                                   # V8POL: leaky dual
            current = (1.0 - decay) * current
        self.lagrange_multiplier = max(0.0, current + self.config.dual_lr * (value - budget))
        cap = getattr(self.config, "lagrange_max", None)
        if cap is not None:                               # V8POL: capped dual
            was_capped = getattr(self, "_lagrange_capped", False)
            self._lagrange_pre_cap = self.lagrange_multiplier
            self._lagrange_capped = self.lagrange_multiplier > float(cap)
            if self._lagrange_capped and not was_capped:
                print(f"[lagrange-cap] update {self.update_count + 1}: lambda "
                      f"{self._lagrange_pre_cap:.4g} capped at lagrange_max={float(cap):.4g}", flush=True)
            self.lagrange_multiplier = min(self.lagrange_multiplier, float(cap))
        return self.lagrange_multiplier

    def _safety_terms(self, imagined, mean_cost):
        """``(actor penalty, dual value, dual budget, log metrics)`` of the TTC constraint.

        absolute (default): ``(mean_cost, mean_cost, None -> ttc_budget, {})`` = the original terms.
        native_relative (V7POL): per all-CAV row s, ``c_s`` = mean over members/futures/horizon of
        its per-control-step cost and ``ref_s`` = the same mean of its start's all-native base row
        (detached) + ``ttc_native_margin``, or ``ttc_budget`` when the start has no base row.
        Penalty ``mean_s relu(c_s - ref_s)``; the dual ascends on ``mean_s (c_s - ref_s)`` over the
        native-referenced starts (over the absolute ones only when no start has a base row)."""
        cfg = self.config
        if cfg.ttc_budget_mode == "absolute":
            if cfg.wm_safety_report_only:
                return mean_cost.detach().new_zeros(()), mean_cost.detach(), None, {
                    "wm_safety_report_only": True, "ttc_penalty_report": float(mean_cost.detach())}
            return mean_cost, mean_cost.detach(), None, {}
        n_base = imagined["n_base"]
        per_row = imagined["costs"].mean((0, 1))                 # [N]
        policy_cost = per_row[:n_base]
        reference = torch.full_like(policy_cost, float(cfg.ttc_budget))
        native = torch.zeros(n_base, dtype=torch.bool, device=policy_cost.device)
        variants = imagined.get("variants")
        if variants is not None:
            for start, row in variants.base_row.items():
                reference[start] = per_row[row].detach() + float(cfg.ttc_native_margin)
                native[start] = True
        excess = policy_cost - reference
        penalty = torch.relu(excess).mean()
        chosen = native if bool(native.any()) else ~native
        dual_value = excess.detach()[chosen].mean()
        extra = {"ttc_budget_mode": "native_relative", "ttc_native_starts": int(native.sum()),
                 "ttc_excess": float(dual_value),
                 "ttc_penalty": float(penalty.detach())}
        if bool(native.any()):
            extra["ttc_native_cost"] = float((reference[native] - float(cfg.ttc_native_margin)).mean())
            extra["ttc_policy_cost_native_starts"] = float(policy_cost.detach()[native].mean())
        if cfg.wm_safety_report_only:
            extra.update(wm_safety_report_only=True, ttc_penalty_report=float(penalty.detach()))
            penalty = penalty.detach().new_zeros(())
        return penalty, dual_value, 0.0, extra

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

    def actor_objective(self, imagined, scale, weights, *, gradient_components=None):
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
        paired_difference = difference
        if getattr(self, 'long_credit', None) is not None:
            from .long_credit import pessimistic_difference
            difference = pessimistic_difference(self, imagined, difference, base, has_base, coefficients)
        anti_metrics, anchor = {}, returns.new_zeros(())
        if self.config.pessimistic_credit or self.config.native_anchor:
            from .antiexploit import actor_terms
            difference, anchor, anti_metrics = actor_terms(self, imagined, difference, coefficients, scale,
                                                         raw_difference=paired_difference)
        loss = -(difference.mean((0, 1)) * coefficients).sum() / scale
        dense_loss = loss
        gated_metrics = {}
        if 'gated300_long_difference' in imagined:
            from .gated300 import actor_term
            long_term, gated_metrics = actor_term(self, imagined, coefficients, difference, scale)
            loss = loss + long_term
            if gradient_components is not None:
                gradient_components['gated300_long'] = long_term
        if self.config.pessimistic_credit or self.config.native_anchor:
            loss = loss + anchor
            if gradient_components is not None:
                gradient_components['native_anchor'] = anchor
                gradient_components['paired_risk_credit'] = (
                    dense_loss if 'gated300_long_difference' in imagined else loss - anchor)
        extra = {}
        if variants is not None and variants.base_row:
            extra = DreamerTrainer._cooperation_metrics(imagined, variants, weights)
        extra.update(anti_metrics)
        extra.update(gated_metrics)
        lanes = imagined.get("lanes")
        if lanes is not None:
            advantage = difference.detach() / scale
            values = imagined.get("values")
            if (values is not None and not self.config.pessimistic_credit
                    and not getattr(self, 'long_credit_requested', False) and getattr(self, 'long_credit', None) is None):
                advantage[..., :n_base] = (returns[..., :n_base] - values[:, :-1, :n_base]).detach() / scale
            active = lanes["lane_active"]                 # [E,H,N,K]
            joint = (lanes["lane_log_prob"] * active.to(returns.dtype)).sum(-1)   # [E,H,N]
            pg_loss = -((joint * advantage).mean((0, 1)) * coefficients).sum()
            lane_entropy = _masked_mean(lanes["lane_entropy"], active)
            # V6EXPLORE: a learned lane temperature replaces the fixed weight when a target is set.
            lane_weight = _entropy_temperature(getattr(self, "log_lane_alpha", None),
                                               self.config.lane_entropy_weight, self.config)
            loss = loss + self.config.lane_pg_weight * pg_loss - lane_weight * lane_entropy
            if gradient_components is not None:
                gradient_components["lane"] = (self.config.lane_pg_weight * pg_loss
                                                - lane_weight * lane_entropy)
            acting = lanes["lane_acting"][..., :n_base, :]
            changed = lanes["lane_changed"][..., :n_base, :] & acting
            extra.update({"lane_pg_loss": float(pg_loss.detach()),
                          "lane_entropy": float(lane_entropy.detach()),
                          "lane_change_rate": float(changed.sum() / acting.sum().clamp_min(1)),
                          "lane_decisions_with_options": int(active[..., :n_base, :].sum())})
        if gradient_components is not None:
            # Construct independently: loss - long_term would leave a cancelling
            # long branch in backward and can lose precision in its scalar value.
            dense_objective = dense_loss + anchor
            if lanes is not None:
                dense_objective = (dense_objective + self.config.lane_pg_weight * pg_loss
                                   - lane_weight * lane_entropy)
            gradient_components['_dense_objective'] = dense_objective
        return loss, extra

    def _actor_gradient_norms(self, imagined, scale, weights, components):
        """Pre-clip L2 parameter norms of actual weighted loss contributions.

        TD(lambda) is linear: split analytic return credit into raw reward,
        disagreement and frozen-critic bootstrap, preserving the row coefficients.
        Counterfactual baselines and lane advantages are detached in the loss.
        autograd.grad retains the training graph and never writes .grad buffers.
        """
        rewards, values = imagined["rewards"], imagined["values"]
        coefficients = row_coefficients(imagined.get("variants"), weights, rewards.shape[-1],
                                        imagined["n_base"], device=rewards.device,
                                        dtype=rewards.dtype)
        sources = {"reward": (rewards, torch.zeros_like(values)),
                   "critic": (torch.zeros_like(rewards), values)}
        if self.config.pessimistic_credit or self.config.native_anchor:
            # Tail selection is nonlinear. Audit the actual paired risk term
            # instead of mislabelling its pre-selection linear decomposition.
            sources = {}
        uncertainty_weight = imagined.get("uncertainty_weight", self.config.uncertainty_weight)
        if uncertainty_weight > 0 and not (self.config.pessimistic_credit or self.config.native_anchor):
            sources["uncertainty"] = (
                -uncertainty_weight * imagined["disagreement"].unsqueeze(0).expand_as(rewards),
                torch.zeros_like(values))
        for name, (reward, value) in sources.items():
            credit = torch.stack([lambda_returns(r, v, self.config.gamma, self.config.lambda_)
                                  for r, v in zip(reward, value)])
            components[name] = -(credit.mean((0, 1)) * coefficients).sum() / scale
        parameters = [p for p in self.actor_parameters if p.requires_grad]
        norms = {}
        for name, term in components.items():
            if name.startswith('_'):
                continue
            grads = (torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                     if term.requires_grad and parameters else ())
            norms[name] = math.sqrt(sum(float(g.detach().double().square().sum())
                                        for g in grads if g is not None))
        return norms

    def _critic_step(self, imagined, starts, returns, real_batch, scale, n_base, n_rows, extra):
        if getattr(self, 'long_credit_requested', False) or 'gated300_long_difference' in imagined:
            # Preserve checkpoint/optimizer compatibility, but keep the critic
            # OFF: no baseline has qualified within-world ranking evidence.
            extra.update(critic_bootstrap=False, critic_baseline=False)
            return returns.new_zeros(()), returns.new_zeros(())
        return self._legacy_critic_step(imagined, starts, returns, real_batch, scale, n_base, n_rows, extra)

    def _legacy_critic_step(self, imagined, starts, returns, real_batch, scale, n_base, n_rows, extra):
        mix = self.config.real_return_mix
        imagined_loss = None
        if mix < 1 and self.longcritic is None and self.paircrit is None:
            critic_scale = max(scale, self.config.real_return_scale)
            if (self.config.imagination_microbatch or self.config.imagination_checkpoint_segment
                    or self.config.imagination_offload_futures
                    or self.config.imagination_offload_saved_tensors):
                from .imagination_memory import memory_critic_loss
                imagined_loss = memory_critic_loss(self, imagined, returns, critic_scale)
            else:
                predictions = torch.stack([torch.stack([
                    self.critic(_detach(_rows(state, n_base, n_rows))) for state in states[:-1]])
                    for states in imagined['states']])
                imagined_loss = ((predictions - returns.detach()) / critic_scale).square().mean()
        if self.paircrit is not None:
            critic_loss, pair_metrics = self.paircrit.loss(self, starts, scale)
            extra.update(pair_metrics)
        elif real_batch is None:
            critic_loss = imagined_loss
        else:
            real_predictions = self.critic(real_batch.states)
            if real_predictions.shape != real_batch.returns.shape:
                raise ValueError('real critic predictions must have shape [B], matching real returns')
            real_scale = max(scale, self.config.real_return_scale)
            real_loss = ((real_predictions - real_batch.returns.to(real_predictions)) / real_scale).square().mean()
            critic_loss = mix * real_loss
            if imagined_loss is not None:
                critic_loss = critic_loss + (1 - mix) * imagined_loss
            extra.update(real_return_mix=mix, critic_real_loss=float(real_loss.detach()),
                         critic_imagined_loss=float(imagined_loss.detach()) if imagined_loss is not None else None,
                         real_return_mean=float(real_batch.returns.mean()),
                         real_return_count=real_batch.returns.numel(), real_return_units=real_batch.units)
        if self.longcritic is not None:
            auxiliary, long_metrics = self.longcritic.loss(self, starts, scale)
            if auxiliary is not None:
                critic_loss = critic_loss + (1 - mix) * auxiliary
            extra.update(long_metrics)
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
        return critic_loss, critic_norm

    def _action_audit_metrics(self, imagined):
        """Policy commands BEFORE safety/caps, only actively controlled all-CAV rows.

        CavPolicy's log density includes the affine-tanh Jacobian in m/s^2.
        exp(differential entropy) is an effective action width in those units.
        Saturation means within 1% of the action span of either stored bound.
        """
        audit, n_base = imagined.get("action_audit"), imagined["n_base"]
        if audit is None:
            return {}
        active = audit["audit_acting"][..., :n_base, :]
        actions = audit["audit_action"][..., :n_base, :][active]
        log_prob = audit["audit_log_prob"][..., :n_base, :][active]
        entropy = float(-log_prob.double().mean()) if log_prob.numel() else 0.0
        out = {"active_actions": actions.numel(), "action_unit": "m/s^2",
               "entropy_nats": entropy,
               "entropy_effective_width_mps2": math.exp(entropy) if log_prob.numel() else 0.0}
        lo, hi = getattr(self.policy, "accel_min", None), getattr(self.policy, "accel_max", None)
        if lo is not None and hi is not None:
            margin = 0.01 * (hi - lo)
            low, high = actions <= lo + margin, actions >= hi - margin
            out.update({"lower_bound_mps2": float(lo), "upper_bound_mps2": float(hi),
                        "saturation_margin_mps2": float(margin),
                        "saturation_low_fraction": float(low.float().mean()) if actions.numel() else 0.0,
                        "saturation_high_fraction": float(high.float().mean()) if actions.numel() else 0.0,
                        "saturation_fraction": float((low | high).float().mean()) if actions.numel() else 0.0})
        return out

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

    def _coopw_metrics(self, imagined, batch, variants, scale):
        """S-COOPW logs (empty unless switched on): row coverage, oversampling, curriculum
        position, pessimism weight, and every ``term_grad_audit_every`` updates the unit-weight
        gradient norms of the objective terms (``coopw.term_objectives``)."""
        cf, out = self.counterfactual, {}
        if getattr(self, 'long_credit', None) is not None:
            artifact = self.long_credit
            h = str(int(imagined['imagination_horizon_s']))
            out['long_credit_cqr_sha256'] = artifact.get('artifact_sha256')
            out['long_credit_k'] = imagined['rewards'].shape[0]
            out['long_credit_finite_rank'] = artifact['horizons'].get(h, {}).get('qualified', False)
            out['long_credit_off_policy_after_update'] = (
                not getattr(self, 'long_credit_on_policy', False) or
                self.update_count != getattr(self, 'long_credit_bound_update', None))
            out['long_credit_exploratory_unqualified'] = not imagined.get('deterministic_policy', False)
        if getattr(self, 'looplong_logging', False):
            from .long_credit import hold_metrics
            out.update(hold_metrics(imagined))
            position = self._curriculum_position(advance=False)
            out['exploration_persistence_s'] = (self.exploration.persistence_at(position)
                if self.exploration else 0.)
            out['exploration_correlated_active'] = (self.exploration is not None
                and self.exploration.rho_at(position) > 0)
        if cf.enabled and cf.coop_audit:
            from .coopw import coverage
            out["cf_coverage"] = coverage(variants, batch.state)
            out["cf_curriculum_position"] = self._curriculum_position()
            if self._last_start_info is not None:
                out["start_oversampling"] = self._last_start_info
            if self.uncertainty is not None:
                out["uncertainty_weight_effective"] = imagined.get("uncertainty_weight")
                out["n_futures"] = imagined.get("n_futures")
            if self.exploration is not None:
                out["exploration_rho"] = imagined.get("exploration_rho", 0.0)
            if self.horizon_curriculum is not None:
                out["imagination_horizon_stage"] = self.horizon_curriculum.stage(
                    self._curriculum_position(advance=False))
        every = cf.term_grad_audit_every if cf.enabled else 0
        if every and (self.update_count + 1) % every == 0 and variants is not None and variants.groups:
            from .coopw import term_gradient_norms, term_objectives
            terms, values = term_objectives(imagined, scale, self.config.lane_pg_weight)
            parameters = [p for p in self.actor_parameters if p.requires_grad]
            out["cf_term_grad_norms"] = term_gradient_norms(terms, parameters)
            out["cf_term_values"] = values
        return out

    def _sample_starts(self, sampler, validation=False):
        """Group-rich oversampling (``group_state_fraction`` of the batch satisfies
        :func:`wms.control.counterfactual.group_rich`) when the sampler supports it.
        S-COOPW: ``validation_uniform`` draws VAL starts from the target distribution;
        ``start_tilt`` tilts TRAINING starts (``coopw.TiltSampler``, observable features)."""
        cf = self.counterfactual
        if validation and cf.enabled and cf.validation_uniform:
            return sampler.sample(self.config.batch_size, device=self.device)
        if not validation and self._tilt_sampler is not None:
            from .coopw import start_features
            horizon_s = self.config.horizon * STEPS_PER_CONTROL * PHYSICS_DT_S

            def features(batch):
                state = {k: v[:1] for k, v in batch.state.items()
                         if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] == batch.state["s_m"].shape[0]}
                queue = batch.entry_queue[0] if batch.entry_queue is not None else None
                feature_fn = start_features
                if cf.headroom_starts:
                    from .coopw import headroom_features
                    feature_fn = headroom_features
                    resolved = self._imagination_config(batch).horizon * STEPS_PER_CONTROL * PHYSICS_DT_S
                else:
                    resolved = horizon_s
                return feature_fn(state, cf, resolved, entry_queue=queue,
                                      lane_rules=self.lane_rules)
            generator = torch.Generator().manual_seed(2_000_003 * (self.update_count + 1))
            batch, self._last_start_info = self._tilt_sampler.draw(
                sampler, self.config.batch_size, self.device, features,
                max_tries=cf.max_group_tries, generator=generator)
            return batch
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
        if self.config.entrant_control and batch.entry_feat_control is not None:
            batch = replace(batch, entry_feat=batch.entry_feat_control)
        if not self.config.imagine_entries:
            return batch
        horizon_steps = self._imagination_config(batch).horizon
        horizon_s = horizon_steps * STEPS_PER_CONTROL * PHYSICS_DT_S
        pad_max = self.config.entry_pad_max
        if (self._training_imagination and self.horizon_curriculum is not None
                and horizon_steps > self.config.horizon):
            # S-COOPW: a staged horizon gets its stage's entrant-slot cap (COOPW.md).
            pad_max = max(pad_max, self.horizon_curriculum.pad_cap(horizon_s))
        from .gated300 import requested_rollout, rollout_config
        if requested_rollout(self):
            long_cfg = rollout_config(self, self._imagination_config(batch))
            long_s = long_cfg.horizon * STEPS_PER_CONTROL * PHYSICS_DT_S
            # Reserve entrants once for the continuation, not separately per horizon.
            pad_max = max(pad_max, math.ceil(pad_max * long_s / max(horizon_s, 1.)))
            horizon_s = long_s
        if self.config.entry_queue_state:
            return with_entry_slots(batch, horizon_s, pad_max, generator, queue_state=True)
        return with_entry_slots(batch, horizon_s, pad_max, generator)

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
        self._longskip_records = []
        from .gated300 import active as gated300_active
        real_batch = (self._real_return_batch() if self.paircrit is None and
                      not getattr(self, 'long_credit_requested', False) and not gated300_active(self) else None)
        self.policy.train()
        self.group_encoder.train()
        self.critic.train()
        # Entry seeds from a per-update generator: enabling entries leaves the global RNG stream
        # (variant choice, policy noise) unchanged, and a resumed run draws the same seeds.
        profile_due = (self.config.credit_profile_every > 0
                       and (self.update_count + 1) % self.config.credit_profile_every == 0)
        # S-COOPW switches (uncertainty futures, correlated exploration, horizon stage) act on
        # this training imagination only; unset they change nothing.
        self._training_imagination = True
        try:
            starts = self._sample_starts(self.start_states)
            batch = self._entry_slots(starts,
                                      torch.Generator().manual_seed(1_000_003 * (self.update_count + 1)))
            variants = (build_variants(batch.state, self.counterfactual)
                        if self.counterfactual.enabled else None)
            imagined = self.imagine(batch, variants=variants, **({"credit_profile": True}
                                                                if profile_due else {}))
        finally:
            self._training_imagination = False
        imagined["penetration_buckets"] = self._penetration_buckets(batch)
        n_base, n_rows = imagined["n_base"], imagined["returns"].shape[-1]
        all_returns = imagined["returns"]
        returns = all_returns[..., :n_base]
        scale = self._return_scale(returns)
        weights = self.counterfactual.weights(self._curriculum_position())
        entropy = imagined["entropy"][..., :n_base].mean()
        mean_cost = imagined["costs"][..., :n_base].mean()
        components = {}
        audit_components = components if self.config.actor_gradient_audit else None
        objective, extra = self.actor_objective(
            imagined, scale, weights, gradient_components=components)
        direct_term = objective.new_zeros(())
        if getattr(self, 'looplong_real_credit', None) is not None:
            if self.config.exploit_disable_direct_credit:
                extra.update(exploit_direct_credit_disabled=True, direct_real_credit_grad_norm=0.0)
            else:
                direct, receipt = self.looplong_real_credit.loss(self, scale)
                from .looplong_real import direct_gradient_norm
                extra['direct_real_credit_grad_norm'] = direct_gradient_norm(direct, self.actor_parameters)
                objective = objective + direct
                direct_term = direct
                extra.update(receipt)
        # V6EXPLORE: a learned entropy temperature replaces the fixed weight when a target is set
        # (V7POL: floored at alpha_min).
        entropy_weight = self._temperature(self.log_alpha, self.config.entropy_weight)
        # V7POL safety terms (absolute + unscaled = the original lambda * mean_cost exactly).
        penalty, dual_value, dual_budget, safety_extra = self._safety_terms(imagined, mean_cost)
        extra.update(safety_extra)
        lagrange_term = self.lagrange_multiplier * penalty
        if self.config.lagrange_scaled:
            lagrange_term = lagrange_term / scale
        actor_loss = (objective - entropy_weight * entropy + lagrange_term)
        if audit_components is not None:
            audit_components.update(constraint=lagrange_term, entropy=-entropy_weight * entropy)
            # S-SPEED2 grad_audit_every: diagnostic only (never read by the update).
            if self.update_count % speed_flags.every("grad_audit_every") == 0:
                extra["actor_grad_norms"] = self._actor_gradient_norms(
                    imagined, scale, weights, audit_components)
            extra["action_audit"] = self._action_audit_metrics(imagined)
        extra.update(self._coopw_metrics(imagined, batch, variants, scale))
        from . import eslong
        es_grads = None
        if eslong.active(self):
            es_grads, es_metrics = eslong.gradient(self, imagined, weights, scale)
            extra.update(es_metrics)
        self.actor_optimizer.zero_grad(set_to_none=True)
        from .longgrad import actor_backward
        dense_loss = (components['_dense_objective'] + direct_term
                      - entropy_weight * entropy + lagrange_term)
        extra.update(eslong.backward(self, dense_loss, es_grads) if es_grads is not None else
                     actor_backward(self, dense_loss, components.get('gated300_long'), actor_loss))
        if hasattr(self, 'memory_spill_stats'):
            extra['memory_spill'] = dict(self.memory_spill_stats)
        if getattr(self.policy, "residual_on_native", False):
            modules = {"residual_head": self.policy.residual_head,
                       "own_encoder": self.policy.own_encoder,
                       "report_encoder": self.policy.report_encoder,
                       "attention": self.policy.attention, "group_encoder": self.group_encoder}
            extra["actor_module_grad_norms"] = {
                name: math.sqrt(sum(float(p.grad.detach().double().square().sum())
                                    for p in module.parameters() if p.grad is not None))
                for name, module in modules.items()}
            lrs = {id(p): g["lr"] for g in self.actor_optimizer.param_groups for p in g["params"]}
            extra["actor_module_lr"] = {
                name: sorted({lrs[id(p)] for p in module.parameters() if id(p) in lrs})
                for name, module in modules.items()}
            audit = imagined.get("action_audit")
            if audit is not None:
                residual = audit["audit_action"][..., :n_base, :][audit["audit_acting"][..., :n_base, :]]
                extra["physical_residual_mps2"] = {
                    "count": residual.numel(),
                    "mean_abs": float(residual.abs().mean()) if residual.numel() else 0.,
                    "max_abs": float(residual.abs().max()) if residual.numel() else 0.,
                    "rms": float(residual.square().mean().sqrt()) if residual.numel() else 0.}
        # clip_grad_norm_ returns the PRE-clip norm (logged below), including when V6
        # raises the imagination decay above the WM value. Keep the existing clip only.
        actor_norm = nn.utils.clip_grad_norm_(self.actor_parameters, self.config.grad_clip,
                                             error_if_nonfinite=True)
        self.actor_optimizer.step()
        # V6EXPLORE: dual steps on the entropy temperature(s), logged with the entropy above.
        extra.update(self._alpha_update(imagined, entropy))
        critic_loss, critic_norm = self._critic_step(
            imagined, starts, returns, real_batch, scale, n_base, n_rows, extra)
        self.update_lagrange(dual_value, dual_budget)
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
                   "imagination_horizon_s": imagined["imagination_horizon_s"],
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
        free = imagined["components"].get("free_speed_shortfall")
        if free is not None:                              # V8POL free-road CAV speed shortfall
            metrics["free_speed_shortfall"] = float(free[..., :n_base].detach().mean())
        if self.config.lagrange_max is not None:          # V8POL: log when the cap binds
            metrics["lagrange_capped"] = bool(self._lagrange_capped)
            metrics["lagrange_pre_cap"] = float(self._lagrange_pre_cap)
        metrics["entries"] = bool(getattr(batch, "entry_spec", None) is not None
                                  and self.config.imagine_entries)
        if imagined.get("entrance_zone_fraction") is not None:   # V10ENTZONE masked fraction
            metrics["entrance_zone_fraction"] = float(imagined["entrance_zone_fraction"])
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
                    batch = self._entry_slots(self._sample_starts(sampler, validation=True),
                                              entry_generator)
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
                   **({"lane_control": "native"} if self.lane_control == "native" else {}),
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
        coopw = {**({"anchor": self.coop_anchor} if self.coop_anchor is not None else {}),
                 **({"tilt_sampler": self._tilt_sampler.state_dict()}
                    if self._tilt_sampler is not None else {})}
        if coopw:
            payload["coopw"] = coopw
        if self.longcritic is not None:
            payload["longcritic"] = self.longcritic.state_dict()
        if self.paircrit is not None:
            payload['paircrit'] = self.paircrit.state_dict()
        if getattr(self, 'looplong_boundary', None) is not None:
            payload['looplong'] = {'boundary': self.looplong_boundary,
                'cqr': getattr(self, 'long_credit', None),
                'horizon': self.horizon_curriculum.as_dict() if self.horizon_curriculum else None,
                'exploration': self.exploration.as_dict() if self.exploration else None}
        if hasattr(self, 'breakdown_receipt'):
            payload['breakdown'] = self.breakdown_receipt
        if hasattr(self, 'gated300'):
            payload['gated300'] = self.gated300
        lane_switches = getattr(self.controller_factory, "ramp_merge", {})
        if lane_switches.get("native_lane_change", False):
            payload["wm_controller_kwargs"] = {"symmetric_exec_cap": True, "mask_forced_merge": True,
                                                **lane_switches}
        if getattr(self.policy, "residual_on_native", False):
            payload["residual_on_native"] = True
            from .execution import contract as execution_contract
            reference = getattr(self.policy, "residual_reference", "base")
            if reference != "base":
                payload["residual_reference"] = reference
            contract = execution_contract(reference)
            if contract is not None:
                payload["execution_contract"] = contract
        torch.save(payload, path)

    def load_checkpoint(self, path, *, policy_only=False, new_phase=False, resume_allow=()):
        if (new_phase or resume_allow) and policy_only:
            raise ValueError('new-phase migration requires full resume')
        if resume_allow and not new_phase:
            raise ValueError('resume_allow requires new_phase')
        requested_config, requested_reward = self.config, self.reward_weights
        # Memory is execution geometry: an explicitly enabled runtime request may
        # be applied to an old full-resume checkpoint without changing its objective.
        memory_request = {name: getattr(self.config, name) for name in MEMORY_CONFIG_FIELDS}
        training_request = {name: getattr(self.config, name) for name in
                            ("native_residual_train_group", "native_residual_train_trunk",
                             "real_credit_individual_floor")
                            if getattr(self.config, name)}
        anti_request = {}
        if self.config.pessimistic_credit or self.config.native_anchor:
            anti_request = {name: getattr(self.config, name) for name in
                ('pessimistic_credit', 'native_anchor', 'native_anchor_calibration',
                 'future_samples', 'uncertainty_weight', 'entrant_control', 'deterministic_crn')}
            # These runtime requests must survive historical C1 full resume.
            from .exploit import FIELDS
            anti_request.update({name: getattr(self.config, name) for name in FIELDS})
        elif self.config.exploit_disable_direct_credit:
            anti_request['exploit_disable_direct_credit'] = True
        if any(memory_request.values()):
            memory_request["credit_profile_every"] = self.config.credit_profile_every
        else:
            memory_request = {}
        payload = torch.load(checked_path(path), map_location=self.device, weights_only=True)
        from .execution import check_contract
        check_contract(payload)
        if (payload.get("residual_reference", "base") !=
                getattr(self.policy, "residual_reference", "base") and not new_phase):
            raise ValueError("residual_reference change requires an explicit new phase")
        if bool(payload.get("residual_on_native", False)) != bool(getattr(self.policy, "residual_on_native", False)):
            raise ValueError("cannot resume across absolute/native-residual action semantics")
        if payload.get("format") not in READABLE_FORMATS:
            raise ValueError(f"expected a {' or '.join(READABLE_FORMATS)} checkpoint")
        if not policy_only and payload["format"] != CHECKPOINT_FORMAT:
            raise ValueError(f"full resume requires a {CHECKPOINT_FORMAT} checkpoint (use policy_only)")
        if not policy_only:
            from .phase_resume import resume_diff, authorize_new_phase
            diff = resume_diff(requested_config, requested_reward, payload)
            if new_phase:
                authorize_new_phase(diff, resume_allow)
            self.resume_config_receipt = {'new_phase': new_phase, 'differences': diff,
                                          'approved': sorted(resume_allow)}
            if diff:
                print('[resume-config] ' + json.dumps(self.resume_config_receipt, sort_keys=True),
                      flush=True)
            self._check_real_return_provider(requested_config if new_phase else
                                             DreamerConfig(**payload["config"]))
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
        # Preserve the explicitly requested C4 boundary atomically across a
        # historical full-trainer resume (registered credit floor and entrant /
        # CRN settings did not exist in C1). Same floor repair as 96a60687.
        self.config = (requested_config if new_phase else
                       DreamerConfig(**{**payload["config"], **memory_request,
                                        **training_request, **anti_request}))
        if getattr(self.policy, "residual_on_native", False):
            self.policy.configure_native_training(group=self.config.native_residual_train_group,
                                                  trunk=self.config.native_residual_train_trunk)
            self.group_encoder.requires_grad_(self.config.native_residual_train_group)
            self.actor_parameters = [p for p in list(self.policy.parameters()) +
                                     list(self.group_encoder.parameters()) if p.requires_grad]
            self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=self.config.actor_lr)
            old = copy.deepcopy(payload["actor_optimizer"])
            old_cfg = payload["config"]
            if (not old_cfg.get("native_residual_train_group", False)
                    and not old_cfg.get("native_residual_train_trunk", False)
                    and len(self.actor_parameters) != 2):
                # Legacy optimizer is exactly the two head tensors. Keep its Adam
                # moments/step; the newly trainable random features start empty.
                assert len(old["param_groups"]) == 1 and len(old["param_groups"][0]["params"]) == 2
                fresh = self.actor_optimizer.state_dict()
                ids = {id(p): i for i, p in enumerate(self.actor_parameters)}
                head_ids = [ids[id(p)] for p in self.policy.residual_head.parameters()]
                fresh["state"] = {new: old["state"][prior] for prior, new in
                                  zip(old["param_groups"][0]["params"], head_ids) if prior in old["state"]}
                fresh["param_groups"][0].update({k: v for k, v in old["param_groups"][0].items()
                                                  if k not in ("params", "lr")})
                payload["actor_optimizer"] = fresh
                self.native_training_migration = "legacy head Adam retained; new features fresh Adam"
        self.reward_weights = requested_reward if new_phase else RewardWeights(**payload["reward_weights"])
        # V6EXPLORE: the loaded config may activate targets the run config did not have.
        self._init_entropy_temperature()
        for name in ("critic", "target_critic", "actor_optimizer", "critic_optimizer"):
            getattr(self, name).load_state_dict(payload[name])
        if self.longcritic is not None and "longcritic" in payload:
            self.longcritic.load_state_dict(payload["longcritic"])
        if (self.paircrit is None) != ('paircrit' not in payload):
            raise ValueError('absolute/paired checkpoint mismatch; fresh boundary fit required')
        if self.paircrit is not None:
            self.paircrit.load_state_dict(payload['paircrit'])
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
        if new_phase:
            from .phase_resume import restore_requested_rates
            restore_requested_rates(self)
            if self.config.long_grad_reset_actor_adam:
                count = len(self.actor_optimizer.state)
                self.actor_optimizer.state.clear()
                self.resume_config_receipt['actor_adam_reset'] = {'cleared_parameters': count}
        for name in ("update_count", "lagrange_multiplier", "return_low", "return_high", "best_validation"):
            setattr(self, name, payload[name])
        self.best_coop_validation = payload.get("best_coop_validation", -math.inf)
        coopw = payload.get("coopw") or {}
        if self.counterfactual.curriculum_anchor == "first_update" and "anchor" in coopw:
            self.coop_anchor = int(coopw["anchor"])
        if self._tilt_sampler is not None and "tilt_sampler" in coopw:
            self._tilt_sampler.load_state_dict(coopw["tilt_sampler"])
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
        from .gated300 import checkpoint_restore
        checkpoint_restore(self, payload)
        _restore_rng(payload["rng"], self.device)
