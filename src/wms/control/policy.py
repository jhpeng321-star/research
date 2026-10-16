"""Small shared CAV actor, zone-conditioned RSU encoder and privileged critic."""
from __future__ import annotations

import math
from typing import Mapping, NamedTuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Normal

from .contract import (ACCEL_MIN_MPS2, F_OWN, F_REPORT, GROUP_SIGNAL_DIM, LEGACY_ACCEL_MAX_MPS2,
                       MERGE_S0_M, MERGE_S1_M, OWN_FEATURES, POLICY_ACCEL_MAX_MPS2,
                       REPORT_FEATURES, Observation)
from .contract import F_RSU_CELL
from .lanes import LANE_DELTA, N_LANE_CHOICES
from .obs import FEATURE_NORMALIZATION, normalize_features
from .rsu_field import FIELD_SCALE, cell_geometry


def _masked_attention(attention: nn.MultiheadAttention, query: Tensor, keys: Tensor,
                      mask: Tensor) -> Tensor:
    """Avoid all-masked softmax rows with a zero dummy key for empty sets."""
    empty = ~mask.any(-1, keepdim=True)
    keys = torch.cat((keys, keys.new_zeros(keys.shape[0], 1, keys.shape[-1])), 1)
    mask = torch.cat((mask, empty), 1)
    out, _ = attention(query, keys, keys, key_padding_mask=~mask, need_weights=False)
    return torch.where(empty[..., None], torch.zeros_like(out), out)


class PolicyNoise(NamedTuple):
    """Explicit exploration noise (common random numbers across counterfactual rows / WM members)."""
    accel: Tensor          # [B,K] standard normal for the acceleration latent
    lane_gumbel: Tensor    # [B,K,3] standard Gumbel for the lane choice


def draw_policy_noise(shape, *, generator=None, device=None, dtype=torch.float32) -> PolicyNoise:
    """Standard-normal accel noise and standard-Gumbel lane noise of ``shape`` = (B, K)."""
    eps = torch.randn(*shape, generator=generator, dtype=dtype)
    u = torch.rand(*shape, N_LANE_CHOICES, generator=generator, dtype=dtype)
    gumbel = -torch.log(-torch.log(u.clamp(1e-20, 1.0 - 1e-7)))
    return PolicyNoise(eps.to(device) if device is not None else eps,
                       gumbel.to(device) if device is not None else gumbel)


class PolicyOutput(NamedTuple):
    """Per-CAV actions and their statistics, all [B,K] (zeros on non-CAV slots)."""
    accel: Tensor              # direct acceleration in [policy.accel_min, policy.accel_max] (before safety)
    lane_choice: Tensor        # long: LANE_STAY / LANE_LEFT / LANE_RIGHT (feasible by the mask)
    log_prob_accel: Tensor
    log_prob_lane: Tensor      # log pi(choice | masked logits); 0 when only "stay" is feasible
    entropy_accel: Tensor      # single-sample estimate -log_prob_accel
    entropy_lane: Tensor       # exact categorical entropy over the feasible choices
    lane_options: Tensor       # number of feasible choices (1..3)


class CavPolicy(nn.Module):
    """Shared actor: (Observation, group [B,K,8]) -> per-CAV acceleration and lane choice.

    ``forward(obs, group)`` keeps the legacy contract ``(accel, log_prob)``.  With
    ``lane_mask`` ([B,K,3] feasible stay/left/right from :func:`wms.control.lanes.
    lane_feasibility`) it returns a :class:`PolicyOutput` including the categorical lane
    head (masked logits; ``stay`` always feasible).  ``noise`` (:class:`PolicyNoise`)
    replaces the internal draws (common random numbers).  ``lane_head=False`` builds the
    pre-2026-10-12 accel-only actor (``c4_dreamer/1`` checkpoints): it always stays.
    ``init_zero_accel=True`` (V6EXPLORE, default False = unchanged random start) zeroes the
    acceleration MEAN row of the head and biases it so the initial decoded mean acceleration
    is 0 m/s^2 for every input (see :meth:`init_zero_mean_accel`).

    Action bounds (ACCELCAP_NOTES.md): ``tanh`` is mapped to ``[accel_min, accel_max]``.  The
    bounds are a stored property of the actor: the persistent buffer ``accel_bounds`` (float64,
    in ``state_dict``) carries them through every checkpoint, and a state dict WITHOUT it (every
    pre-2026-10-13 checkpoint) loads with the legacy bound ``[-3, 1.5]``, whatever the constructor
    default was, so old checkpoints decode bit-identically.  Fresh actors default to
    ``POLICY_ACCEL_MAX_MPS2`` (2.6); trainers pass the configured value explicitly.
    """

    supports_noise = True

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4, lane_head: bool = True, *,
                 accel_min: float = ACCEL_MIN_MPS2, accel_max: float = POLICY_ACCEL_MAX_MPS2,
                 init_zero_accel: bool = False) -> None:
        super().__init__()
        self.register_buffer("accel_bounds", torch.zeros(2, dtype=torch.float64))
        self.init_zero_accel = bool(init_zero_accel)
        self.set_accel_bounds(accel_min, accel_max)
        self.own_encoder = nn.Sequential(nn.Linear(F_OWN, hidden_dim), nn.SiLU(),
                                         nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.report_encoder = nn.Sequential(nn.Linear(F_REPORT, hidden_dim), nn.SiLU())
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim + GROUP_SIGNAL_DIM, hidden_dim),
                                  nn.SiLU(), nn.Linear(hidden_dim, 2))
        self.has_lane_head = bool(lane_head)
        if self.has_lane_head:
            self.lane_head = nn.Sequential(
                nn.Linear(2 * hidden_dim + GROUP_SIGNAL_DIM + N_LANE_CHOICES, hidden_dim),
                nn.SiLU(), nn.Linear(hidden_dim, N_LANE_CHOICES))
            # Uniform start over the feasible choices: no designed lane preference.
            nn.init.zeros_(self.lane_head[-1].weight)
            nn.init.zeros_(self.lane_head[-1].bias)
        if self.init_zero_accel:
            self.init_zero_mean_accel()

    def init_zero_mean_accel(self) -> None:
        """V6EXPLORE: neutral zero-mean initial acceleration (call on a FRESH actor only).

        The tanh decode maps a zero pre-activation to the range MIDPOINT (-0.9 m/s^2 for the
        [-3, 1.2] bounds), so a random start mostly brakes (the v5 run opened at a -0.67 mean
        command).  This zeroes the acceleration MEAN row of the head and biases it so the
        decoded mean is 0 m/s^2 on any input; the log-std row and every other weight keep
        their random defaults, so it is an initialisation, not a designed controller (no imitation / hand-designed
        behaviour).  ``set_accel_bounds`` re-centres the
        bias while the flag is set, so fresh actors stay zero-mean under a configured cap;
        loading a state dict overwrites the bias as usual.
        """
        with torch.no_grad():
            nn.init.zeros_(self.head[-1].weight[0])
        self.init_zero_accel = True
        self._centre_mean_bias()

    def _centre_mean_bias(self) -> None:
        # tanh(b) = -midpoint/scale, so the decoded mean midpoint + scale * tanh(b) = 0.
        scale = (self.accel_max - self.accel_min) / 2.0
        midpoint = (self.accel_max + self.accel_min) / 2.0
        with torch.no_grad():
            self.head[-1].bias[0] = math.atanh(-midpoint / scale)

    def set_accel_bounds(self, accel_min: float, accel_max: float) -> None:
        """Set the action bounds.  This changes how the SAME weights decode actions: use it only
        for fresh actors or an explicit, documented re-decode of a loaded checkpoint."""
        accel_min, accel_max = float(accel_min), float(accel_max)
        if not (accel_min < 0.0 < accel_max):
            raise ValueError(f"accel bounds must satisfy accel_min < 0 < accel_max, got "
                             f"[{accel_min}, {accel_max}]")
        self.accel_min, self.accel_max = accel_min, accel_max
        with torch.no_grad():
            self.accel_bounds.data = torch.tensor((accel_min, accel_max), dtype=torch.float64,
                                                  device=self.accel_bounds.device)
        if getattr(self, "init_zero_accel", False) and hasattr(self, "head"):
            self._centre_mean_bias()

    def _sync_accel_bounds(self) -> None:
        # Rounded: a float32 cast of the module (``.float()``) must not perturb 2.6 -> 2.5999999.
        lo, hi = (round(float(x), 6) for x in self.accel_bounds.detach().cpu().double())
        self.accel_min, self.accel_max = lo, hi

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys,
                              unexpected_keys, error_msgs):
        key = prefix + "accel_bounds"
        if key not in state_dict:       # pre-2026-10-13 checkpoint: legacy bound, decoded as before
            state_dict[key] = torch.tensor((ACCEL_MIN_MPS2, LEGACY_ACCEL_MAX_MPS2), dtype=torch.float64)
        else:                           # keep the exact float64 value whatever the module dtype
            state_dict[key] = state_dict[key].to(torch.float64)
        self.accel_bounds.data = self.accel_bounds.data.to(torch.float64)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys,
                                      unexpected_keys, error_msgs)
        self._sync_accel_bounds()

    def forward(self, obs: Observation, group_signal: Tensor, *, deterministic: bool = False,
                lane_mask: Optional[Tensor] = None, noise: Optional[PolicyNoise] = None):
        b, k, _ = obs.own.shape
        own = self.own_encoder(normalize_features(obs.own, OWN_FEATURES))
        reports = torch.where(obs.report_mask[..., None], obs.reports,
                              torch.zeros_like(obs.reports))
        reports = self.report_encoder(normalize_features(reports, REPORT_FEATURES))
        context = _masked_attention(self.attention, own.reshape(b * k, 1, -1),
                                    reports.reshape(b * k, reports.shape[2], -1),
                                    obs.report_mask.reshape(b * k, -1)).reshape(b, k, -1)
        signal = torch.where(obs.cav_mask[..., None], group_signal, torch.zeros_like(group_signal))
        features = torch.cat((own, context, signal), -1)
        mean, log_std = self.head(features).unbind(-1)
        std = log_std.clamp(-5.0, 1.0).exp()
        distribution = Normal(mean, std)
        if deterministic:
            latent = mean
        else:
            eps = (noise.accel.to(mean) if noise is not None else torch.randn_like(mean))
            latent = mean + std * eps          # reparameterised (rsample)
        scale = (self.accel_max - self.accel_min) / 2.0
        midpoint = (self.accel_max + self.accel_min) / 2.0
        accel = midpoint + scale * torch.tanh(latent)
        # Stable log(1-tanh(z)^2), including saturated latent values.
        log_jacobian = math.log(scale) + 2.0 * (math.log(2.0) - latent - F.softplus(-2.0 * latent))
        log_prob = distribution.log_prob(latent) - log_jacobian
        cav = obs.cav_mask
        accel = torch.where(cav, accel, torch.zeros_like(accel))
        log_prob = torch.where(cav, log_prob, torch.zeros_like(log_prob))
        if lane_mask is None:
            return accel, log_prob
        feasible = lane_mask.bool() & cav[..., None]
        feasible = torch.cat((torch.ones_like(feasible[..., :1]), feasible[..., 1:]), -1)
        options = feasible.sum(-1)
        zeros = torch.zeros_like(log_prob)
        if self.has_lane_head:
            logits = self.lane_head(torch.cat((features, feasible.to(features.dtype)), -1))
            logits = torch.where(feasible, logits, torch.full_like(logits, -1e9))
            log_p = torch.log_softmax(logits, -1)
            if deterministic:
                choice = logits.argmax(-1)
            else:
                gumbel = (noise.lane_gumbel.to(logits) if noise is not None else
                          -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-20, 1.0 - 1e-7))))
                choice = (logits + gumbel).argmax(-1)
            log_prob_lane = log_p.gather(-1, choice[..., None]).squeeze(-1)
            entropy_lane = -torch.where(feasible, log_p.exp() * log_p, torch.zeros_like(log_p)).sum(-1)
            choice = torch.where(cav, choice, torch.zeros_like(choice))
            log_prob_lane = torch.where(cav & (options > 1), log_prob_lane, zeros)
            entropy_lane = torch.where(cav, entropy_lane, zeros)
        else:
            choice = torch.zeros_like(options)
            log_prob_lane = entropy_lane = zeros
        return PolicyOutput(accel, choice, log_prob, log_prob_lane, -log_prob, entropy_lane,
                            torch.where(cav, options, torch.ones_like(options)))


class _SetBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.SiLU(),
                                 nn.Linear(2 * hidden_dim, hidden_dim))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.norm1(x + _masked_attention(self.attention, x, x, mask))
        x = self.norm2(x + self.ffn(x))
        return torch.where(mask[..., None], x, torch.zeros_like(x))


class RsuGroupEncoder(nn.Module):
    """RSU encoder: (obs, positions [B,K], rsu_field [B,N,F] | None) -> per-CAV signals [B,K,8].

    CAV reports (set attention + pooling) plus the RSU own-sensor field (each CAV attends
    over the RSU cells); smooth upstream/merge/downstream zone memberships condition the
    output.  No true penetration or global input; the field is what the RSU sensors see.
    """

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4) -> None:
        super().__init__()
        self.input = nn.Linear(F_OWN + 1, hidden_dim)
        self.set_block = _SetBlock(hidden_dim, n_heads)
        # RSU own-sensor field: each CAV attends over the RSU cells.
        self.field_input = nn.Sequential(nn.Linear(F_RSU_CELL + 4, hidden_dim), nn.SiLU(),
                                         nn.Linear(hidden_dim, hidden_dim))
        self.field_attention = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.head = nn.Sequential(nn.Linear(3 * hidden_dim + 4, hidden_dim), nn.SiLU(),
                                  nn.Linear(hidden_dim, GROUP_SIGNAL_DIM), nn.Tanh())

    def _field_context(self, x: Tensor, rsu_field: Optional[Tensor]) -> Tensor:
        if rsu_field is None:
            return torch.zeros_like(x)
        b, k, h = x.shape
        scale = rsu_field.new_tensor(FIELD_SCALE)
        geometry = cell_geometry(rsu_field.device, rsu_field.dtype).expand(b, -1, -1)
        tokens = self.field_input(torch.cat((rsu_field / scale, geometry), -1))
        out, _ = self.field_attention(x, tokens, tokens, need_weights=False)
        return out

    def forward(self, obs: Observation, positions: Tensor,
                rsu_field: Optional[Tensor] = None) -> Tensor:
        mask = obs.cav_mask
        s = torch.where(mask, positions, torch.zeros_like(positions))
        position = normalize_features(s[..., None], ("s_m",))
        x = self.input(torch.cat((normalize_features(obs.own, OWN_FEATURES), position), -1))
        x = self.set_block(x, mask)
        pooled = x.sum(1, keepdim=True) / mask.sum(1).clamp_min(1)[:, None, None]
        upstream = torch.sigmoid((MERGE_S0_M - s) / 10.0)
        downstream = torch.sigmoid((s - MERGE_S1_M) / 10.0)
        zones = torch.stack((upstream, 1.0 - upstream - downstream, downstream), -1)
        field = self._field_context(x, rsu_field)
        signal = self.head(torch.cat((x, pooled.expand_as(x), field, position, zones), -1))
        return torch.where(mask[..., None], signal, torch.zeros_like(signal))


class TeamCritic(nn.Module):
    """Privileged all-present-vehicle set transformer: state -> value [B].

    ``entry_backlog=True`` (SRCGRAD_NOTES.md) also feeds the per-entrance entry queue
    ``state["entry_backlog"]`` [B, 2] (mainline, ramp; vehicles / ``BACKLOG_SCALE``; zeros when
    absent) to the value head, so the bootstrap can price queues beyond the imagined horizon.
    Its head has 2 more inputs: older critic weights fail to load (size mismatch)."""

    BACKLOG_SCALE = 10.0

    def __init__(self, hidden_dim: int = 64, n_heads: int = 4, n_layers: int = 2,
                 entry_backlog: bool = False) -> None:
        super().__init__()
        self.entry_backlog = bool(entry_backlog)
        self.input = nn.Linear(8, hidden_dim)
        self.blocks = nn.ModuleList([_SetBlock(hidden_dim, n_heads) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Linear(hidden_dim + (2 if self.entry_backlog else 0), hidden_dim),
                                  nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, state: Mapping[str, Tensor]) -> Tensor:
        mask = state["present"].bool()
        if all(name in state for name in ("s_m", "v_mps", "a_mps2", "lane")):
            # Scalar fields are authoritative after WM-S rollout steps. Keep
            # deployment networks independent of the world-model package.
            lane = state["lane"].long()
            onehot = F.one_hot(lane.clamp(0, 2), 3).to(state["s_m"].dtype)
            onehot = torch.where((lane >= 0)[..., None], onehot, torch.zeros_like(onehot))
            length = state.get("length_m", torch.full_like(
                state["s_m"], FEATURE_NORMALIZATION["length_m"]))
            kin = torch.cat((state["s_m"][..., None], onehot, state["v_mps"][..., None],
                             state["a_mps2"][..., None], length[..., None]), -1)
        else:
            kin = state["veh_kin"]
        kin = torch.where(mask[..., None], kin, torch.zeros_like(kin))
        features = ("s_m", "lane_0", "lane_1", "lane_2", "v_mps", "a_mps2", "length_m")
        cav = torch.where(mask, state["is_cav"].bool(), torch.zeros_like(mask)).to(kin.dtype)
        x = self.input(torch.cat((normalize_features(kin, features), cav[..., None]), -1))
        for block in self.blocks:
            x = block(x, mask)
        pooled = x.sum(1) / mask.sum(1).clamp_min(1)[..., None]
        if self.entry_backlog:
            backlog = state.get("entry_backlog")
            backlog = (torch.zeros(pooled.shape[0], 2, dtype=pooled.dtype, device=pooled.device)
                       if backlog is None else backlog.to(pooled.dtype).reshape(pooled.shape[0], 2))
            pooled = torch.cat((pooled, backlog / self.BACKLOG_SCALE), -1)
        value = self.head(pooled).squeeze(-1)
        return torch.where(mask.any(1), value, torch.zeros_like(value))


CHECKPOINT_FORMATS = ("c4_dreamer/1", "c4_dreamer/2", "c4_dreamer/3")


def policy_accel_bounds(policy) -> tuple[float, float]:
    """``(accel_min, accel_max)`` of a policy (or a wrapper exposing ``accel_max``); the legacy
    ``[-3, 1.5]`` for objects without stored bounds (e.g. ``None`` / hold)."""
    return (float(getattr(policy, "accel_min", ACCEL_MIN_MPS2)),
            float(getattr(policy, "accel_max", LEGACY_ACCEL_MAX_MPS2)))


def accel_bounds_payload(policy) -> dict:
    """Checkpoint payload fields recording the actor's action bounds (readable without torch
    state-dict inspection; the authoritative copy is the ``accel_bounds`` buffer)."""
    if getattr(policy, "accel_max", None) is None:      # custom actors without stored bounds
        return {}
    lo, hi = policy_accel_bounds(policy)
    return {"accel_min_mps2": lo, "accel_max_mps2": hi}


def check_payload_accel_bounds(policy, payload) -> None:
    """Payload without bound fields: whatever the state dict says (legacy when absent).  With
    fields: they must agree with the loaded actor (a state dict lacking the buffer but a payload
    carrying the fields adopts the payload's bounds)."""
    if "accel_max_mps2" not in payload or not hasattr(policy, "set_accel_bounds"):
        return
    lo = float(payload.get("accel_min_mps2", ACCEL_MIN_MPS2))
    hi = float(payload["accel_max_mps2"])
    state = payload.get("policy") or {}
    if not any(str(key).endswith("accel_bounds") for key in state):
        policy.set_accel_bounds(lo, hi)
    elif (abs(policy.accel_min - lo) > 1e-6) or (abs(policy.accel_max - hi) > 1e-6):
        raise ValueError(f"checkpoint accel bounds disagree: payload [{lo}, {hi}] vs state dict "
                         f"[{policy.accel_min}, {policy.accel_max}]")


def upgrade_policy_state(state: dict) -> dict:
    """Zero-pad the report encoder of pre-/3 actors (11 report features) to the current
    ``F_REPORT``: the new command/intent report features get zero weights, so the upgraded actor
    computes exactly what the old one did."""
    from .contract import LEGACY_F_REPORT

    key = "report_encoder.0.weight"
    weight = state.get(key)
    if weight is not None and weight.shape[-1] == LEGACY_F_REPORT < F_REPORT:
        state = dict(state)
        state[key] = torch.cat((weight, weight.new_zeros(weight.shape[0], F_REPORT - LEGACY_F_REPORT)), -1)
    return state


class PolicyExecutor:
    """Deployment callable ``state [1,K] -> accel`` (numpy, NaN on non-CAV slots).

    ``exe(state, with_lanes=True)`` also returns the lane targets ``[1,K]`` (int64, WM lane
    0/1/2 for a CAV the policy moves this tick, -1 = stay / not a CAV) for
    ``CavSumoEnv.step(accel, lanes=...)``.  Accel-only callers (``with_lanes=False``) get the
    acceleration computed for the CURRENT lanes (no lane change is executed).  Inputs are
    exactly the deployment ones: CAV observations + RSU-relayed reports + the RSU field; the
    lane mask additionally uses the target-lane gaps (same state the safety layer uses).
    ``hold=True`` is the direct-output fallback ``a = 0, stay`` (safety-filtered).
    ``noise_fn(shape) -> PolicyNoise`` makes a stochastic executor reproducible (CRN).
    One executor = one dwell clock (``reset()`` per episode) unless ``dwell`` is shared;
    with ``dwell_age=`` the caller supplies the ages (and owns the clock).
    """

    def __init__(self, policy=None, group_fn=None, *, lane_rules=None, deterministic: bool = True,
                 noise_fn=None, hold: bool = False, dwell=None, memory=None, device: str = "cpu",
                 extra_accel_noise: float = 0.0):
        from .contract import CONTROL_DT_S
        from .lanes import LaneDwell, LaneRules
        from .obs import CommandMemory

        if not hold and (policy is None or group_fn is None):
            raise ValueError("a policy and a group-signal function are required unless hold=True")
        self.policy, self.group_encoder = policy, group_fn
        # The policy's own stored action bounds bound its commands (legacy when no policy / hold).
        self.accel_min, self.accel_max = policy_accel_bounds(policy)
        self.lane_rules = lane_rules if lane_rules is not None else LaneRules()
        self.has_lane_head = bool(hold or getattr(policy, "has_lane_head", False))
        self.deterministic, self.noise_fn, self.hold, self.device = deterministic, noise_fn, hold, device
        # Data-collection exploration (Dreamer loop): extra N(0, sigma) on the commanded accel before safety.
        self.extra_accel_noise = float(extra_accel_noise)
        self.dwell = dwell if dwell is not None else LaneDwell(self.lane_rules.dwell_s, CONTROL_DT_S)
        # Last commanded accel / lane intent of every CAV, relayed in the other CAVs' reports.
        self.memory = memory if memory is not None else CommandMemory()
        self.last_lanes = None

    def reset(self) -> None:
        self.dwell.reset()
        self.memory.reset()
        self.last_lanes = None

    def act_with_lane(self, state, *, dwell_age=None):
        """``(accel [1,K] float64 NaN off-CAV, lane [1,K] int64: -1 = no change, 0/1/2 = target)``."""
        return self(state, with_lanes=True, dwell_age=dwell_age)

    def __call__(self, state, *, with_lanes: bool = False, dwell_age=None):
        import numpy as np

        from .lanes import apply_lane_choices, lane_feasibility
        from .obs import build_observations
        from .rsu_field import build_rsu_field
        from .safety import safe_accel

        with torch.no_grad():
            state = {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in state.items()}
            ages = dwell_age if dwell_age is not None else self.dwell.step(state)
            obs = build_observations(self.memory.augment(state))
            lane = state["lane"].long()
            new_lane, changed = lane, torch.zeros_like(obs.cav_mask)
            intent = torch.zeros_like(state["s_m"])
            if self.hold:
                accel = torch.zeros_like(state["s_m"])
            else:
                signal = self.group_encoder(obs, state["s_m"], build_rsu_field(state))
                noise = None if self.noise_fn is None else self.noise_fn(tuple(obs.cav_mask.shape))
                kwargs = {} if noise is None else {"noise": noise}
                if with_lanes and self.has_lane_head:
                    mask = lane_feasibility(state, ages, self.lane_rules)
                    out = self.policy(obs, signal, deterministic=self.deterministic,
                                      lane_mask=mask, **kwargs)
                    accel = out.accel
                    new_lane, changed = apply_lane_choices(state, out.lane_choice, mask,
                                                           self.lane_rules, obs.cav_mask)
                    intent = torch.tensor(LANE_DELTA, device=accel.device)[out.lane_choice].to(accel.dtype)
                else:
                    accel, _ = self.policy(obs, signal, deterministic=self.deterministic, **kwargs)
            accel = accel.float()
            if self.extra_accel_noise > 0.0 and not self.hold:
                accel = (accel + self.extra_accel_noise * torch.randn_like(accel)).clamp(
                    self.accel_min, self.accel_max)
            safe = safe_accel(state, accel, accel_max=self.accel_max)
            if bool(changed.any()):
                # SUMO executes the change during the next physics step (and may refuse it):
                # be safe for BOTH the current and the target-lane leader.
                safe = torch.minimum(safe, safe_accel({**state, "lane": new_lane}, accel,
                                                      accel_max=self.accel_max))
            self.memory.record(state, torch.where(obs.cav_mask, safe, torch.zeros_like(safe)),
                               torch.where(obs.cav_mask, intent, torch.zeros_like(intent)))
        out = safe.detach().cpu().numpy().astype(np.float64)
        cav = obs.cav_mask.cpu().numpy()
        out[~cav] = np.nan
        lanes = torch.where(changed, new_lane, torch.full_like(new_lane, -1)).cpu().numpy().astype(np.int64)
        self.last_lanes = lanes
        return (out, lanes) if with_lanes else out


def load_policy_components(path, *, device: str = "cpu"):
    """``(policy, group_encoder, lane_rules, payload)`` from a ``c4_dreamer/1|2`` checkpoint.

    ``c4_dreamer/1`` (accel-only actor) loads as ``CavPolicy(lane_head=False)``: it stays in lane."""
    from .lanes import LaneRules

    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") not in CHECKPOINT_FORMATS:
        raise ValueError(f"expected a {' or '.join(CHECKPOINT_FORMATS)} checkpoint")
    state = upgrade_policy_state(payload["policy"])
    lane_head = any(key.startswith("lane_head.") for key in state)
    policy, encoder = CavPolicy(lane_head=lane_head), RsuGroupEncoder()
    policy.load_state_dict(state)          # bounds: stored buffer, else legacy [-3, 1.5]
    check_payload_accel_bounds(policy, payload)
    group = {k[len("encoder."):] if k.startswith("encoder.") else k: v
             for k, v in payload["group_encoder"].items()}
    encoder.load_state_dict(group)
    policy.eval()
    encoder.eval()
    return policy.to(device), encoder.to(device), LaneRules.from_mapping(payload.get("lane_rules")), payload


def load_policy_fn(path, *, deterministic: bool = True, device: str = "cpu"):
    """Deployment :class:`PolicyExecutor` from a C4 trainer checkpoint (``c4_dreamer/1`` or
    ``/2``); used by the SUMO runner (``exe(state)`` -> accel, ``exe(state, with_lanes=True)``
    -> (accel, lane targets)).  ``exe.policy`` / ``exe.group_encoder`` expose the networks."""
    import os

    policy, encoder, rules, _ = load_policy_components(path, device=device)
    # Dreamer-loop data collection (exploration): C4_POLICY_STOCHASTIC=1 samples from the policy instead of
    # its mean; C4_POLICY_ACCEL_NOISE=<sigma> adds extra Gaussian accel noise (m/s^2). Deployment default: off.
    if os.environ.get("C4_POLICY_STOCHASTIC") == "1":
        deterministic = False
    noise = float(os.environ.get("C4_POLICY_ACCEL_NOISE", "0") or 0)
    return PolicyExecutor(policy, encoder, lane_rules=rules, deterministic=deterministic,
                          device=device, extra_accel_noise=noise)
