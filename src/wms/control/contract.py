"""C4 interface contract: learned multi-CAV control inside WM-S (docs/WM_C4_DESIGN.md).

Single source of truth for the shapes, constants and call signatures shared by the
parallel C4 workstreams.  Every module under ``wms.control`` (and the SUMO runner /
MAPPO baseline) imports its constants from here; change this file only by agreement.

State convention (WM-S model frame, batch-first, K vehicle slots):
    state = {"s_m" [B,K] (merge zone 0..75.44 m, upstream negative), "v_mps" [B,K],
             "a_mps2" [B,K], "lane" [B,K] long (0 = left mainline, 1 = right mainline,
             2 = ramp/acceleration lane, -1 = invalid), "present" [B,K] bool,
             "is_cav" [B,K] bool, optional "length_m", "exit_s_m", "entrant"}.
The SAME observation code runs in imagination and in SUMO: the SUMO runner converts
its frames to this state dict and calls :func:`wms.control.obs.build_observations`.
Nothing here may use true driver types, penetration or global density as an input.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Protocol

from torch import Tensor

# ---------------------------------------------------------------- timing / actuation
CONTROL_DT_S = 1.0          # one policy decision per CAV per second (traffic_shepherd/config.py)
PHYSICS_DT_S = 0.5          # WM-S / SUMO step
STEPS_PER_CONTROL = 2
ACCEL_MIN_MPS2 = -3.0       # CAV actuation bounds (traffic_shepherd/config.py:43-44)
ACCEL_MAX_MPS2 = 1.5
SPEED_LIMIT_MPS = 31.29

# ---------------------------------------------------------------- sensing / comms
SENSOR_RANGE_M = 100.0      # same-lane front/rear sensing (SumoLocalSensor)
REPORT_RANGE_M = 400.0      # RSU relays other CAVs' self reports within this |ds|
MAX_REPORTS = 16            # nearest other-CAV reports per CAV (padded, masked)
N_LANES = 3
MERGE_S0_M = 0.0
MERGE_S1_M = 75.44

#: Own-CAV observation features (order is the contract).
OWN_FEATURES = (
    "v_mps", "a_mps2", "lane_0", "lane_1", "lane_2",
    "s_to_merge_end_m",          # MERGE_S1_M - s (signed; negative after the merge)
    "front_gap_m", "front_dv_mps", "front_present",   # same lane, within SENSOR_RANGE_M
    "rear_gap_m", "rear_dv_mps", "rear_present",
    "in_merge_zone",
)
F_OWN = len(OWN_FEATURES)

#: Per-report features of another CAV, relative to the ego CAV.
REPORT_FEATURES = (
    "ds_m", "dv_mps", "dlane", "lane_0", "lane_1", "lane_2", "v_mps",
    "front_gap_m", "front_dv_mps", "front_present", "age_s",
)
F_REPORT = len(REPORT_FEATURES)
GROUP_SIGNAL_DIM = 8        # learned RSU broadcast vector


@dataclass
class Observation:
    own: Tensor            # [B, K, F_OWN]   (zeros for non-CAV / absent slots)
    reports: Tensor        # [B, K, MAX_REPORTS, F_REPORT]
    report_mask: Tensor    # [B, K, MAX_REPORTS] bool
    cav_mask: Tensor       # [B, K] bool: present CAV slots that act


# ---------------------------------------------------------------- reward
@dataclass
class RewardWeights:
    delay: float = 1.0          # per vehicle-second lost vs SPEED_LIMIT_MPS
    throughput: float = 0.5     # per vehicle crossing MERGE_S1_M ... exit
    smooth: float = 0.05        # sum a^2 (HDV + CAV)
    hard_brake: float = 1.0     # relu(-a - 3)^2
    fairness: float = 1.0       # CVaR(90%) of per-vehicle delay
    ttc_limit_s: float = 3.0    # safety constraint threshold (Lagrangian, not a weight)


# ---------------------------------------------------------------- signatures
class ObservationBuilder(Protocol):
    def __call__(self, state: Mapping[str, Tensor], report_age_s: Optional[Tensor] = None
                 ) -> Observation: ...


class TeamReward(Protocol):
    def __call__(self, state: Mapping[str, Tensor], next_state: Mapping[str, Tensor],
                 accel: Tensor, weights: RewardWeights) -> dict[str, Tensor]:
        """Per physics step: {"reward" [B], "cost_ttc" [B] (>= 0, constraint),
        "delay", "throughput", "smooth", "hard_brake", "fairness_delay" [B]}.
        Differentiable w.r.t. the next-state speeds/positions."""


class Policy(Protocol):
    def __call__(self, obs: Observation, group_signal: Tensor, *, deterministic: bool = False
                 ) -> tuple[Tensor, Tensor]:
        """Returns (accel [B,K] in [ACCEL_MIN, ACCEL_MAX] before the safety layer,
        log_prob [B,K]).  Reparameterised (rsample) so gradients flow."""


class SafetyLayer(Protocol):
    def __call__(self, state: Mapping[str, Tensor], accel: Tensor) -> Tensor:
        """Clamp CAV accelerations so the same-lane gap stays safe; [B,K] -> [B,K]."""


#: WM-S ``cav_controller`` callback: state -> [B,K,4] = (next s, lane, next v, a).
CavController = Callable[[Mapping[str, Tensor]], Tensor]


class CavEnv(Protocol):
    """SUMO environment for one episode (traffic_shepherd side), used by the data loop,
    the evaluation campaign and the MAPPO baseline."""

    def reset(self, *, seed: int, demand_vph: int, penetration: float,
              warmup_s: float, horizon_s: float) -> dict[str, Tensor]: ...

    def step(self, accel_by_slot: Tensor) -> tuple[dict[str, Tensor], dict]:
        """Apply one CONTROL_DT_S of per-CAV accelerations ([1,K]; non-CAV ignored);
        returns (WM-frame state dict with batch 1, info incl. SUMO metrics)."""
