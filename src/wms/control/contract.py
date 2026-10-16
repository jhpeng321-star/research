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
#: LEGACY policy acceleration cap (ACCELCAP_NOTES.md).  Actors now store their own bounds
#: (``CavPolicy.accel_min`` / ``accel_max``, persisted in the state dict and the checkpoint
#: payload); a checkpoint WITHOUT stored bounds decodes with this value, so every pre-2026-10-13
#: checkpoint (phase 1-4, policy_check, MAPPO) behaves exactly as before.  Safety / bridge helpers
#: keep it as their default ceiling; callers bounding a POLICY's commands pass the policy's bound.
ACCEL_MAX_MPS2 = 1.5
LEGACY_ACCEL_MAX_MPS2 = ACCEL_MAX_MPS2
#: Default cap of freshly constructed C4 actors: the ``normal_car`` driver
#: accel of scenarios/idm/driver_routes.rou.xml (aggressive 3.0, cautious 2.0).  Trainers select it
#: explicitly via config (``[policy] accel_max_mps2``).
POLICY_ACCEL_MAX_MPS2 = 2.6
#: SUMO-native CAV acceleration used by the counterfactual 'native' IDM rows.  CAVs have no
#: CAV-only vType: each CAV keeps its driver vType (aggressive 3.0 / normal 2.6 / cautious 2.0,
#: p = 0.1 / 0.7 / 0.2; the type is hidden from the controller), so the dominant normal_car value.
NATIVE_CAV_ACCEL_MAX_MPS2 = 2.6
#: Env-level ceiling of ``CavSumoEnv`` DIRECT commands (largest SUMO driver accel).  The policy's
#: own bound (and, under speedMode 27, the CAV's SUMO vType accel) bind first.
DIRECT_ACCEL_MAX_MPS2 = 3.0
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

#: Per-report features of another CAV, relative to the ego CAV.  The last two
#: are the reporting CAV's last commanded acceleration and lane intent (-1 left, 0 stay,
#: +1 right) from its previous decision, so CAVs can react to each other's plans.
REPORT_FEATURES = (
    "ds_m", "dv_mps", "dlane", "lane_0", "lane_1", "lane_2", "v_mps",
    "front_gap_m", "front_dv_mps", "front_present", "age_s",
    "cmd_accel_mps2", "lane_intent",
)
#: Report features of the pre-2026-10-13 actors (c4_dreamer/1, /2); their weights are zero-padded.
LEGACY_F_REPORT = 11
F_REPORT = len(REPORT_FEATURES)
GROUP_SIGNAL_DIM = 8        # learned RSU broadcast vector

# ---------------------------------------------------------------- RSU own sensors
# The RSU directly observes every vehicle in these WM-frame zones (in addition to CAV reports).
# Its group signal is computed from CAV reports AND this RSU field; the same field feeds MAPPO.
RSU_MAINLINE_S_M = (-800.0, 225.44)   # lanes 0 and 1: 800 m upstream .. ~150 m downstream of the merge end
RSU_RAMP_S_M = (-378.66, 75.44)       # lane 2: whole on-ramp + acceleration lane
RSU_CELL_M = 50.0
RSU_FIELD_FEATURES = (
    "density_veh_per_km",    # vehicles in the cell / cell length
    "mean_speed_mps",        # SPEED_LIMIT_MPS when empty
    "speed_std_mps",
    "queue",                 # 1 if mean speed < 5 m/s and occupied
    "occupied",
    "cav_share",             # CAVs / vehicles in the cell (observable: CAVs identify themselves)
)
F_RSU_CELL = len(RSU_FIELD_FEATURES)
# Cells: lane 0 and lane 1 over RSU_MAINLINE_S_M, then lane 2 over RSU_RAMP_S_M (ceil(length / RSU_CELL_M) each).


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


class RsuFieldBuilder(Protocol):
    def __call__(self, state: Mapping[str, Tensor]) -> Tensor:
        """RSU sensor field [B, N_RSU_CELLS, F_RSU_CELL] from the vehicles inside the RSU zones only
        (same function in imagination and in SUMO)."""


class GroupEncoder(Protocol):
    def __call__(self, obs: Observation, state_positions: Tensor, rsu_field: Tensor) -> Tensor:
        """Per-CAV group signal [B, K, GROUP_SIGNAL_DIM] from all CAV reports + the RSU field, conditioned on
        each CAV's position zone.  ``state_positions`` = CAV s_m [B, K] (CAVs report it)."""


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
