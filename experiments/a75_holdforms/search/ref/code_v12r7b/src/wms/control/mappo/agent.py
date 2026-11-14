"""MAPPO actor = the C4 actor, executed exactly like ``wms.control.policy.PolicyExecutor``.

Fairness by construction: MAPPO trains the SAME ``CavPolicy`` (lane head on, default sizes,
so ``wms.control.policy.load_policy_fn`` rebuilds it) and ``RsuGroupEncoder`` the C4 method
trains, on the SAME inputs, with the SAME deployment step:

    ages   = LaneDwell.step(state)                              (dwell clock, 1 s)
    obs    = build_observations(CommandMemory.augment(state))   (reports incl. last command + intent)
    signal = group_encoder(obs, state["s_m"], build_rsu_field(state))
    mask   = lane_feasibility(state, ages, lane_rules)
    out    = policy(obs, signal, lane_mask=mask, noise=PolicyNoise)
    lanes  = apply_lane_choices(...)                             (conflict-checked)
    safe   = safe_accel(state, accel) [min with the target-lane safe accel if a CAV changes]
    CommandMemory.record(state, safe, intent)

:class:`ActorStep` is that step (stochastic for rollouts, deterministic = ``PolicyExecutor``;
a unit test asserts equality).  It additionally returns what PPO needs per frame: the
memory-augmented state (``cmd_accel_mps2`` / ``lane_intent``), the lane mask, the pre-tanh
latent, the lane choice and both log-probabilities.

:func:`evaluate_actions` re-evaluates stored (latent, lane choice) under the current weights.
It runs the frozen ``CavPolicy.forward`` (deterministic, so no RNG is consumed) and reads the
acceleration-head and lane-head outputs with forward hooks, so no actor code is duplicated;
the joint per-CAV log-probability is ``log pi(accel) + log pi(lane | feasible)``.  Frames are
compacted to their CAV slots first (the encoder is permutation-equivariant over the masked
CAV set and the actor is per-slot, so this is exact; tested).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import asdict
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from wms.control.contract import (ACCEL_MIN_MPS2, CONTROL_DT_S, LEGACY_ACCEL_MAX_MPS2,
                                  Observation)
from wms.control.lanes import (
    LANE_DELTA,
    LaneDwell,
    LaneRules,
    apply_lane_choices,
    lane_feasibility,
)
from wms.control.obs import CommandMemory, build_observations
from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic, draw_policy_noise
from wms.control.rsu_field import build_rsu_field
from wms.control.safety import safe_accel

#: Keys of the decision-time state stored per frame (CavSumoEnv / contract state + memory).
STATE_KEYS = ("s_m", "v_mps", "a_mps2", "lane", "present", "is_cav", "length_m",
              "cmd_accel_mps2", "lane_intent")
EXPORT_FORMAT = "c4_dreamer/3"     # payload schema read by wms.control.policy.load_policy_fn
#: Legacy tanh scale / midpoint (1.5 cap); actors carry their own bounds (ACCELCAP_NOTES.md).
_SCALE = (LEGACY_ACCEL_MAX_MPS2 - ACCEL_MIN_MPS2) / 2.0
_MID = (LEGACY_ACCEL_MAX_MPS2 + ACCEL_MIN_MPS2) / 2.0
_HALF_LOG_2PI_E = 0.5 * (1.0 + math.log(2.0 * math.pi))


def build_networks(accel_max: float = LEGACY_ACCEL_MAX_MPS2) -> tuple[CavPolicy, RsuGroupEncoder,
                                                                      TeamCritic]:
    """The C4 networks at their default sizes (the only sizes ``load_policy_fn`` rebuilds).

    ``accel_max``: the actor's action cap.  MAPPO keeps the LEGACY 1.5 unless the config sets
    ``[policy] accel_max_mps2`` (ACCELCAP_NOTES.md); a loaded state dict brings its own bound."""
    return CavPolicy(lane_head=True, accel_max=accel_max), RsuGroupEncoder(), TeamCritic()


def policy_scale(policy) -> float:
    """tanh scale of ``policy`` (legacy 1.5-cap scale for objects without stored bounds)."""
    lo = float(getattr(policy, "accel_min", ACCEL_MIN_MPS2))
    hi = float(getattr(policy, "accel_max", LEGACY_ACCEL_MAX_MPS2))
    return (hi - lo) / 2.0


@contextmanager
def capture_heads(policy: CavPolicy):
    """Record the raw outputs of ``policy.head`` (mean, log_std) and ``policy.lane_head``."""
    store: dict[str, Tensor] = {}
    handles = [policy.head.register_forward_hook(lambda m, i, o: store.__setitem__("head", o))]
    if getattr(policy, "has_lane_head", False):
        handles.append(policy.lane_head.register_forward_hook(
            lambda m, i, o: store.__setitem__("lane_logits", o)))
    try:
        yield store
    finally:
        for handle in handles:
            handle.remove()


def latent_log_prob(mean: Tensor, log_std: Tensor, latent: Tensor, scale: float = _SCALE) -> Tensor:
    """log pi(accel) of the tanh-squashed Gaussian at pre-tanh ``latent`` (CavPolicy formula);
    ``scale`` = the actor's tanh scale (:func:`policy_scale`)."""
    std = log_std.clamp(-5.0, 1.0).exp()
    log_jacobian = math.log(scale) + 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))
    return torch.distributions.Normal(mean, std).log_prob(latent) - log_jacobian


def _feasible(lane_mask: Tensor, cav: Tensor) -> Tensor:
    feasible = lane_mask.bool() & cav[..., None]
    return torch.cat((torch.ones_like(feasible[..., :1]), feasible[..., 1:]), -1)


class ActorStep:
    """One deployment-identical decision per call (per episode: ``reset()``)."""

    def __init__(self, policy: CavPolicy, group_encoder: RsuGroupEncoder,
                 lane_rules: LaneRules | None = None):
        self.policy, self.group_encoder = policy, group_encoder
        self.lane_rules = lane_rules if lane_rules is not None else LaneRules()
        self.dwell = LaneDwell(self.lane_rules.dwell_s, CONTROL_DT_S)
        self.memory = CommandMemory()

    def reset(self) -> None:
        self.dwell.reset()
        self.memory.reset()

    @torch.no_grad()
    def __call__(self, state: Mapping[str, Tensor], *, deterministic: bool = False,
                 generator: torch.Generator | None = None) -> dict:
        rules = self.lane_rules
        ages = self.dwell.step(state)
        augmented = self.memory.augment(state)
        obs = build_observations(augmented)
        signal = self.group_encoder(obs, state["s_m"], build_rsu_field(state))
        mask = lane_feasibility(state, ages, rules)
        cav = obs.cav_mask
        noise = None if deterministic else draw_policy_noise(tuple(cav.shape), generator=generator)
        kwargs = {} if noise is None else {"noise": noise}
        with capture_heads(self.policy) as heads:
            out = self.policy(obs, signal, deterministic=deterministic, lane_mask=mask, **kwargs)
        mean, log_std = heads["head"].unbind(-1)
        latent = mean if deterministic else mean + log_std.clamp(-5.0, 1.0).exp() * noise.accel
        latent = torch.where(cav, latent, torch.zeros_like(latent))
        accel = out.accel
        new_lane, changed = apply_lane_choices(state, out.lane_choice, mask, rules, cav)
        intent = torch.tensor(LANE_DELTA, device=accel.device)[out.lane_choice].to(accel.dtype)
        accel = accel.float()
        bound = float(getattr(self.policy, "accel_max", LEGACY_ACCEL_MAX_MPS2))
        safe = safe_accel(state, accel, accel_max=bound)
        if bool(changed.any()):
            safe = torch.minimum(safe, safe_accel({**state, "lane": new_lane}, accel, accel_max=bound))
        self.memory.record(state, torch.where(cav, safe, torch.zeros_like(safe)),
                           torch.where(cav, intent, torch.zeros_like(intent)))
        command = torch.where(cav, safe, torch.full_like(safe, float("nan")))
        lanes = torch.where(changed, new_lane, torch.full_like(new_lane, -1))
        return {
            "state": {k: augmented[k] for k in STATE_KEYS if k in augmented},
            "lane_mask": mask, "cav_mask": cav, "latent": latent, "accel": out.accel,
            "lane_choice": out.lane_choice, "logp_accel": out.log_prob_accel,
            "logp_lane": out.log_prob_lane, "lane_changed": changed,
            "command": command, "lanes": lanes, "safe": safe,
        }


def compact_present(states: Mapping[str, Tensor]) -> tuple[dict[str, Tensor], Tensor]:
    """Drop absent slots: ``([B, Kp] states, [B, Kp] slot index)``, present slots first in
    their original slot order (Kp = max present per frame).  Exact for every consumer here:
    observations, the RSU field, safety and the critic depend only on present vehicles, and
    the stable order keeps the report tie-break by slot index."""
    present = states["present"].bool()
    kp = max(1, int(present.sum(-1).max())) if present.numel() else 1
    index = torch.argsort((~present).to(torch.int8), dim=-1, stable=True)[:, :kp]
    return {k: (v.gather(1, index) if torch.is_tensor(v) and v.dim() == 2 else v)
            for k, v in states.items()}, index


def compact_index(cav_mask: Tensor) -> Tensor:
    """[B, Kc] slot indices with the CAV slots first (Kc = max CAVs per frame, >= 1)."""
    kc = max(1, int(cav_mask.sum(-1).max())) if cav_mask.numel() else 1
    order = torch.argsort((~cav_mask).to(torch.int8), dim=-1, stable=True)
    return order[:, :kc]


def _gather(x: Tensor, index: Tensor) -> Tensor:
    view = index.reshape(*index.shape, *([1] * (x.dim() - 2)))
    return x.gather(1, view.expand(*index.shape, *x.shape[2:]))


def compact_observation(obs: Observation, index: Tensor) -> Observation:
    return Observation(_gather(obs.own, index), _gather(obs.reports, index),
                       _gather(obs.report_mask, index), _gather(obs.cav_mask, index))


def evaluate_actions(policy: CavPolicy, group_encoder: RsuGroupEncoder,
                     states: Mapping[str, Tensor], lane_mask: Tensor, latent: Tensor,
                     lane_choice: Tensor) -> dict:
    """Per-CAV log-probs / entropies of stored actions under the current weights.

    ``states`` are decision-time (memory-augmented) states [B, K]; ``lane_mask`` [B, K, 3],
    ``latent`` / ``lane_choice`` [B, K].  Returns compacted [B, Kc] tensors plus ``index``
    ([B, Kc] slot of each compacted entry) and ``cav`` (valid compacted entries).
    """
    obs = build_observations(states)
    field = build_rsu_field(states)
    index = compact_index(obs.cav_mask)
    obs_c = compact_observation(obs, index)
    cav = obs_c.cav_mask
    positions = states["s_m"].gather(1, index)
    mask_c = _gather(lane_mask.bool(), index)
    signal = group_encoder(obs_c, positions, field)
    with capture_heads(policy) as heads:
        policy(obs_c, signal, deterministic=True, lane_mask=mask_c)
    mean, log_std = heads["head"].unbind(-1)
    u = latent.gather(1, index)
    logp_accel = torch.where(cav, latent_log_prob(mean, log_std, u, policy_scale(policy)),
                             torch.zeros_like(mean))
    entropy_accel = torch.where(cav, _HALF_LOG_2PI_E + log_std.clamp(-5.0, 1.0),
                                torch.zeros_like(mean))
    feasible = _feasible(mask_c, cav)
    options = feasible.sum(-1)
    logits = torch.where(feasible, heads["lane_logits"], torch.full_like(heads["lane_logits"], -1e9))
    log_p = torch.log_softmax(logits, -1)
    choice = lane_choice.gather(1, index).long().clamp(0, 2)
    logp_lane = log_p.gather(-1, choice[..., None]).squeeze(-1)
    logp_lane = torch.where(cav & (options > 1), logp_lane, torch.zeros_like(logp_lane))
    entropy_lane = -torch.where(feasible, log_p.exp() * log_p, torch.zeros_like(log_p)).sum(-1)
    entropy_lane = torch.where(cav, entropy_lane, torch.zeros_like(entropy_lane))
    return {"index": index, "cav": cav, "logp_accel": logp_accel, "logp_lane": logp_lane,
            "entropy_accel": entropy_accel, "entropy_lane": entropy_lane}


def export_payload(policy: CavPolicy, group_encoder: RsuGroupEncoder, lane_rules: LaneRules,
                   **extra) -> dict:
    """Deploy actor in the ``c4_dreamer/3`` payload schema (policy, group_encoder, lane_rules),
    tagged ``trainer = "mappo"``; loads with ``wms.control.policy.load_policy_fn`` unchanged."""
    from wms.control.policy import accel_bounds_payload
    payload = {"format": EXPORT_FORMAT, "trainer": "mappo", "lane_head": True,
               "lane_rules": asdict(lane_rules), **accel_bounds_payload(policy),
               "policy": {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
               "group_encoder": {k: v.detach().cpu().clone()
                                 for k, v in group_encoder.state_dict().items()}}
    payload.update(extra)
    return payload
