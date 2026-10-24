"""Architecture and bounding options of :class:`~wms.model.wms.WmSModel`."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .controllers import N_CONTROLLERS
from .history import HISTORY_F, HISTORY_H
from .residual_bound import validate_bounds

ROLLOUT_MODES = ("residual", "driver_params", "hybrid")
LANE_HEADS = ("off", "history")
LANE_RULES = ("mobil", "forced_merge_only", "learned_discretionary")
#: Lane-decision model (WMLANE A): the gated residual lane delta (default) or the
#: dedicated per-HDV lane-change hazard model (``wms.model.lane_hazard``).
LANE_MODELS = ("residual", "hazard", "hazard_v5")
#: In-rollout driver-setting update from observed motion (WMLANE C).
DRIVER_UPDATES = ("off", "ema")
#: LANEFIX (hazard mode only): rollout lane decision, hazard feature set, merge ownership.
LANE_DECISIONS = ("argmax", "sample", "gated")
#: V8LANE mainline decision rules (= wms.model.lane_v5.MAINLINE_MODES).
LANE_V5_MAINLINE = ("default", "sample", "budget", "budget_prop")
LANE_HAZARD_FEATURES = ("v1", "v2")
MERGE_HAZARD_ZONES = ("forced_late", "full")
#: R5THR/R5FR/SECUREGAP merge decision: per-step Bernoulli hazard (default), persistent
#: gap-acceptance threshold + first passage, per-step hazard with a persistent per-driver
#: latent frailty, or the learned LC2013-like two-sided secure-gap core.
MERGE_DECISIONS = ("hazard", "threshold", "frailty", "secure_gap", "first_passage")
#: SLOWRES: direct per-step temporal acceleration or a learned low-pass state.
TEMPORAL_RESIDUAL_MODES = ("direct", "slow")


@dataclass(frozen=True)
class ModelSpec:
    """Every option that determines the structure or bounding of the WM-S model.

    ``black_box`` is the controller-id-conditioned variant: CAVs become learned
    movers.  ``rollout_mode`` selects how the driver identity enters (``residual``:
    none; ``driver_params``: inferred IDM/MOBIL parameters without residual;
    ``hybrid``: inferred parameters plus gated residual conditioned on the driver
    embedding).  ``history_s`` (seconds, optional) enables the long driver history.
    ``lane_rule`` records the C4 lane-change policy of the training config; the
    runtime switch itself lives on :class:`~wms.physics.config.PhysicsConfig`
    (like ``merge_cal`` / ``sumo_fidelity``), which is what the model consults.

    The residual bounds are mutually exclusive: either the legacy tanh caps
    (``residual_cap_*``) or the soft bound (``residual_soft_bound_*``).
    """

    black_box: bool = False
    rollout_mode: str = "residual"
    lane_head: str = "off"
    lane_rule: str = "mobil"
    history_features: int = HISTORY_F
    history_len: int = HISTORY_H
    history_hidden: int = 64
    k_hidden: int = 32
    n_controllers: int = N_CONTROLLERS
    controller_embed_dim: int = 16
    checkpoint_residual: bool = False
    residual_cap_mps2: Optional[float] = None
    residual_cap_neg_mps2: Optional[float] = None
    residual_soft_bound_mps2: Optional[float] = None
    residual_soft_bound_neg_mps2: Optional[float] = None
    lane_logit_cap: Optional[float] = None
    history_s: Optional[float] = None
    history_stride_s: float = 1.0
    gate_kind: str = "cumsum"
    gate_g_max_init: float = 0.30
    gate_tau_steps_init: float = 50.0
    lane_gate_kind: str = "shared"
    lane_gate_g_max_init: float = 0.16
    lane_gate_tau_steps_init: float = 30.0
    checkpoint_step: bool = False
    grad_decay_per_step: float = 1.0
    # WMLANE (default OFF; created modules are appended after every existing one so
    # older checkpoints keep their key order and load unchanged).
    lane_model: str = "residual"
    lane_hazard_hidden: int = 64
    lane_hazard_cap: float = 10.0
    driver_update: str = "off"
    driver_update_rate: float = 0.3
    # LANEFIX (default OFF; hazard mode only; the v2 features widen the hazard input layer).
    lane_decision: str = "argmax"
    lane_logit_offsets: tuple[float, float, float] = (0.0, 0.0, 0.0)
    lane_hazard_features: str = "v1"
    merge_hazard_zone: str = "forced_late"
    # LANEV5 (lane_model='hazard_v5' only): gated-decision gates and configured head APs.
    lane_v5_gates: tuple[float, float, float] = (0.25, 0.10, 0.5)
    lane_v5_head_ap: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    # R5INPUTS (default OFF): append the extra merge inputs (follower acceleration / gap
    # trends / next neighbours / alignment times) to the lane-v5 feature vector.
    lane_v5_extra_features: bool = False
    # V8LANE (default 'default' = lane_decision): mainline overtake / return decision rule.
    lane_v5_mainline: str = "default"
    # V6: default values preserve the legacy encoder and algebraic EMA updates.
    driver_param_mode: str = "legacy"
    driver_ema_update: bool = True
    # V6INTEG: opt-in resolution of simultaneous lane-change conflicts.
    lane_conflict_resolve: bool = False
    # V10COURTESY (default OFF): ramp-car distance-to-end / speed / waiting time and
    # the merge head's current probability as extra residual inputs.
    courtesy_inputs: bool = False
    # R5MERGE (default OFF): learned pre-tanh divisor of the residual head, exactly
    # identity (and no new parameter) when off.
    residual_head_rescale: bool = False
    # V10NEGOT (default OFF): learned ramp/follower merge negotiation block (zero-init).
    merge_negotiation: bool = False
    # R5THR (default 'hazard' = the per-step Bernoulli merge draw): 'threshold' replaces the
    # merge decision by a persistent learned gap-acceptance threshold (first passage).
    merge_decision: str = "hazard"
    merge_st_width: float = 1.0
    # R9 first-passage law: strict preserves the legacy path; hybrid learns a width.
    merge_passage_law: str = "strict"
    merge_w_init: float = 1.0
    merge_quadrature_order: int = 32
    # False: no straight-through adjoint from the pivot lane weights into the next rollout step.
    merge_pivot_state_st: bool = True
    # SLOWRES (default 'direct' = bit-identical): 'slow' replaces the temporal path's
    # direct per-step acceleration with a learned low-pass residual state r_t carried in
    # the rollout state (see wms.model.temporal_slow).  New parameters are created only in
    # slow mode; a direct model has exactly the pre-SLOWRES key set.
    temporal_residual_mode: str = "direct"
    temporal_tau_init_s: float = 0.1
    temporal_tau_conditioned: bool = False
    temporal_slew_init_mps2: Optional[float] = None
    # R9K/R10 (default OFF, FAIL CLOSED): the leader-change tracker exists for recorded-state
    # replay only.  The runtime consumer (zero-init 14->d_model adapter on the residual net and
    # the rollout carry) is NOT installed, so enabling this key would silently do nothing;
    # ModelSpec therefore refuses it until the consumer lands.
    leader_change_features: bool = False
    # R10B (default OFF = bit-identical, no new tensor): learned pre-merge courtesy offset for the
    # target lane-1 follower of a ramp-lane car (wms.model.follower_courtesy).  ``hidden`` is the MLP
    # width; ``embedding`` (ablation only) adds the follower's driver embedding as an input;
    # ``rollout_grad`` lets rollout losses train the head (default False: only the recorded-state
    # courtesy loss reaches its parameters; the state adjoint through the head is kept either way).
    follower_courtesy: bool = False
    follower_courtesy_hidden: int = 32
    follower_courtesy_embedding: bool = False
    follower_courtesy_rollout_grad: bool = False
    # R11: courtesy input version (1 = round 10b; 2 = + ramp car accel t/t-1 + carried merge-zone time)
    follower_courtesy_version: int = 1
    # R13A (defaults = round 12): keep the pair features' continuous state derivative (discrete
    # pairing stays non-differentiable); intent input source: lane_v5 (round 10b-12), pivot (the
    # first-passage clearance CDF at the step-start state) or none (ablation: input slot = 0).
    follower_courtesy_state_grad: bool = False
    follower_courtesy_intent: str = "lane_v5"
    # R13F (default false = round 12): the temporal acceleration emission (row 0 of the temporal
    # MLP output) is replaced by an exact 0 (the BRAKE_DIAG zerotemp tensor); lane rows unchanged.
    temporal_accel_off: bool = False
    # R13M (default false = round 12): "keep the mean, remove the noise" -- the temporal acceleration
    # emission is replaced by a learned per-lane constant b[lane] (pre-gate, learned HDV rows only;
    # parameter ``temporal_lane_bias`` [3], created only when on); the history MLP no longer acts.
    temporal_accel_lane_bias: bool = False
    # R13S (default false = round 12): the emission is the S-FACTORIAL state-conditional mean table over
    # (lane, v, gap, closing, s, k) with backoff (wms.model.temporal_state_table); the tables are parameters.
    temporal_accel_state_table: bool = False

    def __post_init__(self) -> None:
        if (not isinstance(self.follower_courtesy_version, int)
                or isinstance(self.follower_courtesy_version, bool)
                or self.follower_courtesy_version not in (1, 2)):
            raise ValueError("follower_courtesy_version must be 1 or 2")
        if self.follower_courtesy_version == 2 and not self.follower_courtesy:
            raise ValueError("follower_courtesy_version = 2 needs follower_courtesy")
        if sum(getattr(self, name) is True for name in (
                "temporal_accel_off", "temporal_accel_lane_bias", "temporal_accel_state_table")) > 1:
            raise ValueError("temporal_accel_off, temporal_accel_lane_bias and temporal_accel_state_table "
                             "are exclusive")
        if self.follower_courtesy_intent not in ("lane_v5", "pivot", "none"):
            raise ValueError("follower_courtesy_intent must be lane_v5, pivot or none")
        if self.follower_courtesy_intent == "pivot" and self.merge_decision != "first_passage":
            raise ValueError("follower_courtesy_intent='pivot' needs merge_decision='first_passage'")
        for name in ("follower_courtesy", "follower_courtesy_embedding",
                     "follower_courtesy_rollout_grad", "follower_courtesy_state_grad",
                     "temporal_accel_off", "temporal_accel_lane_bias", "temporal_accel_state_table"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if not isinstance(self.follower_courtesy_hidden, int)                 or isinstance(self.follower_courtesy_hidden, bool)                 or self.follower_courtesy_hidden < 1:
            raise ValueError("follower_courtesy_hidden must be an integer >= 1")
        if self.follower_courtesy and self.black_box:
            raise ValueError("follower_courtesy needs the structured WM-S (black_box off)")
        if not isinstance(self.leader_change_features, bool):
            raise ValueError("leader_change_features must be a boolean")
        if self.leader_change_features:
            raise ValueError(
                "model.leader_change_features=true is refused: only the recorded-state replay "
                "tracker exists (R9_K.md coordinator hooks 1-6 are not installed), so the "
                "runtime model would ignore the key")
        if self.driver_param_mode not in ("legacy", "physical"):
            raise ValueError("driver_param_mode must be legacy or physical")
        if not isinstance(self.driver_ema_update, bool):
            raise ValueError("driver_ema_update must be a boolean")
        if self.residual_cap_mps2 is not None and float(self.residual_cap_mps2) <= 0.0:
            raise ValueError("residual_cap_mps2 must be > 0 when set")
        if self.lane_logit_cap is not None and float(self.lane_logit_cap) <= 0.0:
            raise ValueError("lane_logit_cap must be > 0 when set")
        if self.residual_cap_neg_mps2 is not None and (
                not math.isfinite(float(self.residual_cap_neg_mps2))
                or float(self.residual_cap_neg_mps2) <= 0.0):
            raise ValueError("residual_cap_neg_mps2 must be finite and > 0 when set")
        validate_bounds(self.residual_soft_bound_mps2, self.residual_soft_bound_neg_mps2,
                        self.residual_cap_mps2, self.residual_cap_neg_mps2)
        if self.rollout_mode not in ROLLOUT_MODES:
            raise ValueError("rollout_mode must be residual, driver_params or hybrid")
        if self.lane_head not in LANE_HEADS:
            raise ValueError("lane_head must be off or history")
        if self.lane_rule not in LANE_RULES:
            raise ValueError(f"lane_rule must be one of {LANE_RULES}")
        if self.gate_kind not in ("cumsum", "saturating"):
            raise ValueError("gate_kind must be cumsum or saturating")
        if not 0.0 < self.gate_g_max_init < 1.0:
            raise ValueError("gate_g_max_init must be in (0, 1)")
        if not math.isfinite(self.gate_tau_steps_init) or self.gate_tau_steps_init <= 0.0:
            raise ValueError("gate_tau_steps_init must be finite and positive")
        if self.lane_gate_kind not in ("shared", "saturating"):
            raise ValueError("lane_gate_kind must be shared or saturating")
        if not 0.0 < self.lane_gate_g_max_init < 1.0:
            raise ValueError("lane_gate_g_max_init must be in (0, 1)")
        if not math.isfinite(self.lane_gate_tau_steps_init) or self.lane_gate_tau_steps_init <= 0.0:
            raise ValueError("lane_gate_tau_steps_init must be finite and positive")
        if self.lane_head == "history" and self.rollout_mode == "residual":
            raise ValueError("history lane head requires driver_params or hybrid rollout")
        if self.lane_model not in LANE_MODELS:
            raise ValueError(f"lane_model must be one of {LANE_MODELS}")
        if self.driver_update not in DRIVER_UPDATES:
            raise ValueError(f"driver_update must be one of {DRIVER_UPDATES}")
        if (self.lane_model in ("hazard", "hazard_v5") or self.driver_update != "off")                 and (self.rollout_mode == "residual" or self.black_box):
            raise ValueError("lane_model='hazard' / driver_update need the structured "
                             "driver_params or hybrid rollout (driver embedding/settings)")
        if self.lane_model in ("hazard", "hazard_v5") and self.lane_head != "off":
            raise ValueError("lane_model='hazard' and the history lane head are exclusive")
        if self.lane_hazard_hidden < 1 or not (math.isfinite(self.lane_hazard_cap)
                                               and self.lane_hazard_cap > 0.0):
            raise ValueError("lane_hazard_hidden must be >= 1 and lane_hazard_cap > 0")
        if not 0.0 < self.driver_update_rate < 1.0:
            raise ValueError("driver_update_rate must be in (0, 1)")
        offsets = tuple(float(x) for x in self.lane_logit_offsets)
        object.__setattr__(self, "lane_logit_offsets", offsets)
        if len(offsets) != 3 or not all(math.isfinite(x) for x in offsets):
            raise ValueError("lane_logit_offsets must be three finite (overtake, return, merge)")
        if self.lane_decision not in LANE_DECISIONS:
            raise ValueError(f"lane_decision must be one of {LANE_DECISIONS}")
        if not isinstance(self.courtesy_inputs, bool):
            raise ValueError("courtesy_inputs must be a boolean")
        if not isinstance(self.residual_head_rescale, bool):
            raise ValueError("residual_head_rescale must be a boolean")
        if not isinstance(self.merge_negotiation, bool):
            raise ValueError("merge_negotiation must be a boolean")
        if self.temporal_residual_mode not in TEMPORAL_RESIDUAL_MODES:
            raise ValueError(f"temporal_residual_mode must be one of {TEMPORAL_RESIDUAL_MODES}")
        if not math.isfinite(float(self.temporal_tau_init_s)) \
                or float(self.temporal_tau_init_s) <= 0.0:
            raise ValueError("temporal_tau_init_s must be finite and positive")
        if not isinstance(self.temporal_tau_conditioned, bool):
            raise ValueError("temporal_tau_conditioned must be a boolean")
        if self.temporal_slew_init_mps2 is not None and (
                not math.isfinite(float(self.temporal_slew_init_mps2))
                or float(self.temporal_slew_init_mps2) <= 0.0):
            raise ValueError("temporal_slew_init_mps2 must be finite and positive when set")
        if self.temporal_residual_mode == "slow" and self.black_box:
            raise ValueError("temporal_residual_mode='slow' needs the structured model")
        if self.merge_negotiation and self.lane_model != "hazard_v5":
            raise ValueError("merge_negotiation needs lane_model='hazard_v5'")
        if self.merge_decision not in MERGE_DECISIONS:
            raise ValueError(f"merge_decision must be one of {MERGE_DECISIONS}")
        if not math.isfinite(self.merge_st_width) or self.merge_st_width <= 0.0:
            raise ValueError("merge_st_width must be finite and positive")
        if self.merge_passage_law not in ("strict", "hybrid"):
            raise ValueError("merge_passage_law must be strict or hybrid")
        if isinstance(self.merge_w_init, bool) or not math.isfinite(float(self.merge_w_init)) \
                or float(self.merge_w_init) <= 0.0:
            raise ValueError("merge_w_init must be finite and positive")
        if not isinstance(self.merge_quadrature_order, int) \
                or isinstance(self.merge_quadrature_order, bool) \
                or self.merge_quadrature_order < 2:
            raise ValueError("merge_quadrature_order must be an integer >= 2")
        if self.merge_decision == "first_passage" and self.merge_negotiation:
            raise ValueError("first_passage requires B merge_negotiation off")
        if self.merge_decision != "hazard" and self.lane_model != "hazard_v5":
            raise ValueError("merge_decision!='hazard' needs lane_model='hazard_v5'")
        if self.lane_hazard_features not in LANE_HAZARD_FEATURES:
            raise ValueError(f"lane_hazard_features must be one of {LANE_HAZARD_FEATURES}")
        if self.merge_hazard_zone not in MERGE_HAZARD_ZONES:
            raise ValueError(f"merge_hazard_zone must be one of {MERGE_HAZARD_ZONES}")
        gates = tuple(float(x) for x in self.lane_v5_gates)
        head_ap = tuple(float(x) for x in self.lane_v5_head_ap)
        object.__setattr__(self, "lane_v5_gates", gates)
        object.__setattr__(self, "lane_v5_head_ap", head_ap)
        if len(gates) != 3 or len(head_ap) != 3 \
                or not all(math.isfinite(x) for x in (*gates, *head_ap)):
            raise ValueError("lane_v5_gates / lane_v5_head_ap must be three finite values")
        if self.lane_model != "hazard_v5" and (
                self.lane_decision == "gated" or gates != (0.25, 0.10, 0.5)
                or head_ap != (-1.0, -1.0, -1.0)):
            raise ValueError("lane_decision='gated' / lane_v5_* need lane_model='hazard_v5'")
        if self.lane_model not in ("hazard", "hazard_v5") and (
                self.lane_decision != "argmax" or any(offsets)
                or self.lane_hazard_features != "v1" or self.merge_hazard_zone != "forced_late"):
            raise ValueError("lane_decision / lane_logit_offsets / lane_hazard_features / "
                             "merge_hazard_zone need lane_model='hazard'")
        if self.lane_v5_mainline not in LANE_V5_MAINLINE:
            raise ValueError(f"lane_v5_mainline must be one of {LANE_V5_MAINLINE}")
        if self.lane_v5_mainline != "default" and self.lane_model != "hazard_v5":
            raise ValueError("lane_v5_mainline needs lane_model='hazard_v5'")
        if not isinstance(self.lane_v5_extra_features, bool):
            raise ValueError("lane_v5_extra_features must be a boolean")
        if self.lane_v5_extra_features and self.lane_model != "hazard_v5":
            raise ValueError("lane_v5_extra_features needs lane_model='hazard_v5'")
        if self.history_features < 0:
            raise ValueError("history_features must be >= 0")
        if self.long_history and (self.black_box or self.history_features != HISTORY_F):
            raise ValueError("long history requires structured WM-S with six rolling features")
        if self.black_box and self.n_controllers < 1:
            raise ValueError("black_box needs at least one controller id")
        if self.black_box and self.controller_embed_dim < 1:
            raise ValueError("controller_embed_dim must be >= 1 in black_box")

    @property
    def long_history(self) -> bool:
        return self.history_s is not None

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any]) -> "ModelSpec":
        """Spec from a flat legacy training config (``checkpoint["config"]``).

        Keys absent from ``config`` keep their default, which reproduces the
        checkpoints written before the corresponding option existed.
        """
        return cls(**{key: config[key] for key in cls.__dataclass_fields__
                      if config.get(key) is not None})
