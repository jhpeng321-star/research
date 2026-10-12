"""CPU fakes for the C4 MAPPO baseline: a FakeCavEnv producing random WM-frame
states plus tiny stand-ins for the parallel C4 modules (observation builder,
team reward, safety layer, tanh-Gaussian actor, RSU group encoder, privileged
team critic).  They implement exactly the ``wms.control.contract`` protocols so
the trainer, CLI smoke mode and unit tests run with no SUMO and no partner
modules.  The tanh-Gaussian actor also serves as the reference implementation
of the ``evaluate_actions`` extension MAPPO needs from ``CavPolicy``.
"""

from __future__ import annotations

from typing import Mapping, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from wms.control.contract import (
    ACCEL_MAX_MPS2,
    ACCEL_MIN_MPS2,
    CONTROL_DT_S,
    F_OWN,
    F_REPORT,
    MAX_REPORTS,
    MERGE_S1_M,
    N_LANES,
    SENSOR_RANGE_M,
    SPEED_LIMIT_MPS,
    GROUP_SIGNAL_DIM,
    Observation,
    RewardWeights,
)

VEHICLE_LENGTH_M = 5.0


def _state_dict(
    s: Tensor, v: Tensor, a: Tensor, lane: Tensor, present: Tensor, is_cav: Tensor
) -> dict[str, Tensor]:
    return {"s_m": s, "v_mps": v, "a_mps2": a, "lane": lane, "present": present, "is_cav": is_cav}


class FakeCavEnv:
    """CavEnv protocol on random WM-frame states (deterministic given ``seed``)."""

    def __init__(self, k_slots: int = 10, min_vehicles: int = 5, churn_prob: float = 0.1):
        self.k_slots = k_slots
        self.min_vehicles = min_vehicles
        self.churn_prob = churn_prob
        self._gen: torch.Generator | None = None
        self._steps_left = 0
        self._penetration = 0.0
        self._s = self._v = self._a = self._lane = self._present = self._is_cav = None

    def _rand(self, *shape: int, lo: float = 0.0, hi: float = 1.0) -> Tensor:
        return lo + (hi - lo) * torch.rand(shape, generator=self._gen)

    def reset(
        self, *, seed: int, demand_vph: int, penetration: float, warmup_s: float, horizon_s: float
    ) -> dict[str, Tensor]:
        self._gen = torch.Generator().manual_seed(int(seed))
        self._penetration = float(penetration)
        k = self.k_slots
        n = int(torch.randint(self.min_vehicles, k + 1, (1,), generator=self._gen))
        present = torch.zeros(k, dtype=torch.bool)
        present[torch.randperm(k, generator=self._gen)[:n]] = True
        lane = torch.where(
            present,
            torch.randint(0, N_LANES, (k,), generator=self._gen),
            torch.full((k,), -1, dtype=torch.long),
        )
        self._s = self._rand(k, lo=-150.0, hi=MERGE_S1_M)
        self._v = self._rand(k, lo=8.0, hi=28.0)
        self._a = torch.zeros(k)
        is_cav = present & (self._rand(k) < self._penetration)
        if not bool(is_cav.any()):
            is_cav[torch.nonzero(present)[0]] = True
        self._present, self._lane, self._is_cav = present, lane, is_cav
        self._steps_left = int(round(float(horizon_s) / CONTROL_DT_S))
        return _state_dict(
            self._s[None],
            self._v[None],
            self._a[None],
            self._lane[None],
            self._present[None],
            self._is_cav[None],
        )

    def step(self, accel_by_slot: Tensor) -> tuple[dict[str, Tensor], dict]:
        dt = CONTROL_DT_S
        cmd = accel_by_slot.detach()[0].clamp(ACCEL_MIN_MPS2, ACCEL_MAX_MPS2) * self._is_cav
        hdv = self._rand(self.k_slots, lo=-1.0, hi=1.0) * (~self._is_cav).float()
        self._a = cmd + hdv
        self._v = (self._v + self._a * dt).clamp(0.0, SPEED_LIMIT_MPS)
        self._s = self._s + self._v * dt
        if float(torch.rand((1,), generator=self._gen)) < self.churn_prob:
            slot = int(torch.nonzero(self._present)[0])
            self._s[slot] = float(self._rand(1, lo=-150.0, hi=MERGE_S1_M)[0])
            self._v[slot] = float(self._rand(1, lo=8.0, hi=28.0)[0])
            self._is_cav[slot] = bool(torch.rand((1,), generator=self._gen) < self._penetration)
        self._steps_left -= 1
        state = _state_dict(
            self._s[None],
            self._v[None],
            self._a[None],
            self._lane[None],
            self._present[None],
            self._is_cav[None],
        )
        info = {"terminated": False, "steps_remaining": self._steps_left}
        return state, info


def _front_rear(
    s: Tensor, v: Tensor, lane: Tensor, present: Tensor, k: int
) -> tuple[float, float, float, float, float, float]:
    """(front_gap, front_dv, front_present, rear_gap, rear_dv, rear_present)."""
    same = present & (lane == lane[k])
    out = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ahead = torch.nonzero(same & (s > s[k])).reshape(-1)
    behind = torch.nonzero(same & (s < s[k])).reshape(-1)
    if ahead.numel():
        j = int(ahead[torch.argmin(s[ahead] - s[k])])
        gap = float(s[j] - s[k] - VEHICLE_LENGTH_M)
        if gap <= SENSOR_RANGE_M:
            out[0], out[1], out[2] = gap, float(v[j] - v[k]), 1.0
    if behind.numel():
        j = int(behind[torch.argmin(s[k] - s[behind])])
        gap = float(s[k] - s[j] - VEHICLE_LENGTH_M)
        if gap <= SENSOR_RANGE_M:
            out[3], out[4], out[5] = gap, float(v[j] - v[k]), 1.0
    return tuple(out)  # type: ignore[return-value]


def fake_build_observations(
    state: Mapping[str, Tensor], report_age_s: Optional[Tensor] = None
) -> Observation:
    """Contract-shaped observations from a WM-frame state (batch-first).

    Feature order follows ``contract.OWN_FEATURES`` / ``REPORT_FEATURES``
    verbatim; non-CAV and absent slots are zero rows.
    """
    s = state["s_m"].float()
    v = state["v_mps"].float()
    a = state["a_mps2"].float()
    lane = state["lane"].long()
    present = state["present"].bool()
    is_cav = state["is_cav"].bool()
    batch, k = s.shape
    own = torch.zeros(batch, k, F_OWN)
    reports = torch.zeros(batch, k, MAX_REPORTS, F_REPORT)
    report_mask = torch.zeros(batch, k, MAX_REPORTS, dtype=torch.bool)
    cav_mask = torch.zeros(batch, k, dtype=torch.bool)
    for b in range(batch):
        for i in range(k):
            if not (present[b, i] and is_cav[b, i] and lane[b, i] >= 0):
                continue
            cav_mask[b, i] = True
            one_hot = torch.zeros(N_LANES)
            one_hot[int(lane[b, i])] = 1.0
            fr = _front_rear(s[b], v[b], lane[b], present[b], i)
            own[b, i] = torch.tensor(
                [
                    float(v[b, i]),
                    float(a[b, i]),
                    *one_hot,
                    float(MERGE_S1_M - s[b, i]),
                    fr[0],
                    fr[1],
                    fr[2],
                    fr[3],
                    fr[4],
                    fr[5],
                    1.0 if 0.0 <= float(s[b, i]) <= MERGE_S1_M else 0.0,
                ]
            )
            others = torch.nonzero(present[b] & is_cav[b]).reshape(-1)
            others = others[others != i]
            if others.numel():
                ds = s[b][others] - s[b, i]
                order = torch.argsort(torch.abs(ds))[:MAX_REPORTS]
                for r, pos in enumerate(order):
                    j = int(others[pos])
                    lane_hot = torch.zeros(N_LANES)
                    lane_hot[int(lane[b, j])] = 1.0
                    frj = _front_rear(s[b], v[b], lane[b], present[b], j)
                    age = 0.0 if report_age_s is None else float(report_age_s[b, j])
                    reports[b, i, r] = torch.tensor(
                        [
                            float(ds[pos]),
                            float(v[b, j] - v[b, i]),
                            float(lane[b, j] - lane[b, i]),
                            *lane_hot,
                            float(v[b, j]),
                            frj[0],
                            frj[1],
                            frj[2],
                            age,
                        ]
                    )
                    report_mask[b, i, r] = True
    return Observation(own=own, reports=reports, report_mask=report_mask, cav_mask=cav_mask)


def _lane_ttc_cost(
    s: Tensor, v: Tensor, lane: Tensor, present: Tensor, ttc_limit_s: float
) -> float:
    total = 0.0
    for ln in range(N_LANES):
        idx = torch.nonzero(present & (lane == ln)).reshape(-1)
        if idx.numel() < 2:
            continue
        idx = idx[torch.argsort(s[idx])]
        gap = s[idx][1:] - s[idx][:-1] - VEHICLE_LENGTH_M
        closing = (v[idx][:-1] - v[idx][1:]).clamp(min=0.5)
        total += float(torch.relu(ttc_limit_s - gap / closing).sum())
    return total


def fake_team_reward(
    state: Mapping[str, Tensor],
    next_state: Mapping[str, Tensor],
    accel: Tensor,
    weights: RewardWeights,
) -> dict[str, Tensor]:
    """Stand-in team reward with the contract's return keys (all [B], >= 0 cost)."""
    present = state["present"].float()
    is_cav = state["is_cav"].float()
    count = present.sum(-1).clamp(min=1.0)
    cav_count = is_cav.sum(-1).clamp(min=1.0)
    v0, v1 = state["v_mps"].float(), next_state["v_mps"].float()
    progress = ((v1 - v0) * present).sum(-1) / count
    delay = ((SPEED_LIMIT_MPS - v1).clamp(min=0.0) * present).sum(-1) / count
    throughput = (
        (((state["s_m"] < MERGE_S1_M) & (next_state["s_m"] >= MERGE_S1_M)) & state["present"])
        .sum(-1)
        .float()
    )
    smooth = (accel.float() ** 2 * is_cav).sum(-1) / cav_count
    hard = (torch.relu(-accel.float() - 3.0) ** 2 * is_cav).sum(-1) / cav_count
    per_vehicle_delay = (SPEED_LIMIT_MPS - v1).clamp(min=0.0) * present
    top = max(1, int(0.1 * per_vehicle_delay.shape[-1]))
    fairness = torch.topk(per_vehicle_delay, top, dim=-1).values.mean(-1)
    cost = torch.zeros(v0.shape[0])
    for b in range(v0.shape[0]):
        c = _lane_ttc_cost(
            state["s_m"][b].float(),
            v0[b],
            state["lane"][b].long(),
            state["present"][b].bool(),
            weights.ttc_limit_s,
        )
        cost[b] = c / float(count[b])
    reward = (
        progress
        + weights.throughput * 0.01 * throughput
        - weights.delay * delay
        - weights.smooth * smooth
        - weights.hard_brake * hard
        - weights.fairness * 0.1 * fairness
    )
    return {
        "reward": reward,
        "cost_ttc": cost,
        "delay": delay,
        "throughput": throughput,
        "smooth": smooth,
        "hard_brake": hard,
        "fairness_delay": fairness,
    }


def fake_safe_accel(state: Mapping[str, Tensor], accel: Tensor) -> Tensor:
    """SafetyLayer stand-in: actuation bounds + no acceleration near the limit."""
    bounded = accel.clamp(ACCEL_MIN_MPS2, ACCEL_MAX_MPS2)
    at_limit = (state["v_mps"] >= SPEED_LIMIT_MPS - 0.5).float()
    return torch.minimum(bounded, bounded * (1.0 - at_limit))


class FakeGroupEncoder(nn.Module):
    """RsuGroupEncoder stand-in: pooled CAV report summary -> group signal."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(F_OWN, hidden), nn.Tanh())
        self.out = nn.Linear(hidden, GROUP_SIGNAL_DIM)

    def forward(self, obs: Observation) -> Tensor:
        mask = obs.cav_mask.unsqueeze(-1).float()
        pooled = (obs.own * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return torch.tanh(self.out(self.trunk(pooled)))


class FakeActor(nn.Module):
    """CavPolicy stand-in: shared per-CAV tanh-Gaussian over acceleration."""

    def __init__(self, hidden: int = 32, log_std_init: float = 0.0):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(F_OWN, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.mu = nn.Linear(hidden, 1)
        self.log_std = nn.Linear(hidden, 1)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, log_std_init)

    def _params(self, obs: Observation) -> tuple[Tensor, Tensor]:
        h = self.trunk(obs.own)
        mu = self.mu(h).squeeze(-1)
        std = self.log_std(h).squeeze(-1).clamp(-5.0, 2.0).exp()
        return mu, std

    @staticmethod
    def _log_det_jacobian(tanh_u: Tensor) -> Tensor:
        return torch.log(
            torch.clamp(1.0 - tanh_u**2, min=1e-6) * (ACCEL_MAX_MPS2 - ACCEL_MIN_MPS2) / 2.0
        )

    @staticmethod
    def _squash(tanh_u: Tensor) -> Tensor:
        return ACCEL_MIN_MPS2 + (ACCEL_MAX_MPS2 - ACCEL_MIN_MPS2) * (tanh_u + 1.0) / 2.0

    def forward(
        self, obs: Observation, group_signal: Tensor, *, deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        mu, std = self._params(obs)
        u = mu if deterministic else Normal(mu, std).rsample()
        tanh_u = torch.tanh(u).clamp(-0.999999, 0.999999)
        logp = Normal(mu, std).log_prob(u) - self._log_det_jacobian(tanh_u)
        return self._squash(tanh_u), logp

    def evaluate_actions(
        self, obs: Observation, group_signal: Tensor, accel: Tensor
    ) -> tuple[Tensor, Tensor]:
        mu, std = self._params(obs)
        w = ((accel - ACCEL_MIN_MPS2) / (ACCEL_MAX_MPS2 - ACCEL_MIN_MPS2)) * 2.0 - 1.0
        tanh_u = w.clamp(-0.999999, 0.999999)
        u = torch.atanh(tanh_u)
        logp = Normal(mu, std).log_prob(u) - self._log_det_jacobian(tanh_u)
        entropy = Normal(mu, std).entropy() + self._log_det_jacobian(tanh_u)
        return logp, entropy


class FakeTeamCritic(nn.Module):
    """TeamCritic stand-in: privileged centralised V(s) on the full WM state."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: Mapping[str, Tensor]) -> Tensor:
        present = state["present"].float().unsqueeze(-1)
        is_cav = state["is_cav"].float().unsqueeze(-1)
        feats = torch.cat(
            [
                state["v_mps"].float().unsqueeze(-1) / SPEED_LIMIT_MPS,
                state["a_mps2"].float().unsqueeze(-1) / 3.0,
                state["s_m"].float().unsqueeze(-1) / 200.0,
                state["lane"].float().unsqueeze(-1) / 3.0,
                present,
                is_cav,
            ],
            dim=-1,
        )
        pooled = (feats * present).sum(1) / present.sum(1).clamp(min=1.0)
        occupancy = present.squeeze(-1).mean(1, keepdim=True)
        return self.net(torch.cat([pooled, occupancy], dim=-1)).squeeze(-1)
