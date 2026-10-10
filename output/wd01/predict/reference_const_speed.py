"""Constant-speed predictor — NOT A MODEL, conformance reference only.

A deliberately trivial, fully deterministic, pure-numpy implementation of the
predict() contract v1 (wd01_predict/1). It exists so run_conformance() has a
known-good target and so implementers (planner perfect-sim, learned
world-model) can diff against a minimal working example.

What it does:
- every tracked vehicle holds its t0 speed for the whole Part A window
  (trajectories advance linearly; a vehicle is NaN from the step it exits the
  section on);
- native delay: a fixed deterministic function of the arrival draw and the
  tracked speeds (0.01 * remaining-horizon per arrival row + 0.05 *
  horizon * (1 - v0 / free-flow) per tracked vehicle);
- plan delay: native delay + a fixed cost per per_cav action and per wall
  phase (native 0 / damping 3 / pi-external-u 7 s; assemble 10 / hold 6 /
  release 0 s);
- safety fields are all-clear (collision False, min_ttc 30 s, exposures 0);
- M = 1 ("const_speed_ref"), part_a_horizon_s = 60 s, CPU, wall_clock_s = 0.
  weights_sha256 = sha256 of this file (no learned weights).

What it does NOT do: model anything. It is NOT A MODEL, conformance
reference only. The hybrid input it reads is the minimal synthetic mapping
built by predict.conformance.build_state_samples.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from .contract import (
    DEFAULT_HORIZON_S,
    DEFAULT_COMMIT_S,
    CONTINUATION_NATIVE,
    CONTINUATION_PLANNER,
    CONTINUATIONS,
    CONTRACT_VERSION,
    CONTRACT_VERSION_V1_1,
    CONTRACT_VERSION_V2,
    SUPPORTED_CONTRACT_VERSIONS,
    PHASE_ASSEMBLE,
    PHASE_HOLD,
    PHASE_RELEASE,
    ArrivalDraw,
    ControllerState,
    Meta,
    PlanCandidate,
    Prediction,
    StateSample,
    WallOutcome,
)
from .native_ref import native_reference
from .protocol import PredictorMixin
from .validate import validate_plan, validate_state_samples

MODEL_ID = "reference_const_speed/1"
PART_A_HORIZON_S = 60.0
FREE_FLOW_MPS = 30.0
MEMBER_ID = "const_speed_ref"

ACTION_COST_S = {"native": 0.0, "damping": 3.0, "pi-external-u": 7.0}
WALL_PHASE_COST_S = {PHASE_ASSEMBLE: 10.0, PHASE_HOLD: 6.0, PHASE_RELEASE: 0.0}
ASSEMBLED_AT_S = {PHASE_ASSEMBLE: 5.0, PHASE_HOLD: 0.0, PHASE_RELEASE: 0.0}
ARRIVAL_ROW_COST_S = 0.01
SPEED_LAG_COST_S = 0.05


def _tracked(hybrid: object) -> Tuple[Mapping[str, tuple], float]:
    """Read the synthetic hybrid mapping. NOT A MODEL, conformance reference
    only — real hybrid states belong to real implementations."""
    if not isinstance(hybrid, Mapping) or "tracked" not in hybrid:
        raise TypeError(
            "reference_const_speed reads the synthetic hybrid mapping built by "
            "predict.conformance ({'tracked': {id: (s_m, lane_id, speed_mps)}, "
            "'section_len_m': float}); got "
            f"{type(hybrid).__name__} — this predictor is NOT A MODEL, "
            "conformance reference only"
        )
    return hybrid["tracked"], float(hybrid.get("section_len_m", math.inf))


def _arrival_rows(arrivals: object) -> Tuple[tuple, ...]:
    """Explicit rows as passed; a draw expands deterministically (fixed rows
    derived from the seed) so the reference stays a pure function of inputs."""
    if isinstance(arrivals, ArrivalDraw):
        span = (10.0, 90.0, 300.0)
        return tuple(
            (arrivals.seed * 1.0 + dt, "mainline" if i % 2 == 0 else "ramp", 26.0)
            for i, dt in enumerate(span)
        )
    rows = []
    for a in arrivals:
        rows.append((a.t_s, a.entrance, a.speed_mps))
    return tuple(rows)


class ConstSpeedPredictor(PredictorMixin):
    """NOT A MODEL, conformance reference only."""

    model_id = MODEL_ID
    part_a_horizon_s = PART_A_HORIZON_S

    def __init__(
        self,
        free_flow_mps: float = FREE_FLOW_MPS,
        contract_version: str = CONTRACT_VERSION,
    ):
        if contract_version not in SUPPORTED_CONTRACT_VERSIONS:
            raise ValueError(
                f"contract_version must be one of {SUPPORTED_CONTRACT_VERSIONS}, "
                f"got {contract_version!r} (§9: all supported versions)"
            )
        self._free_flow_mps = float(free_flow_mps)
        self._contract_version = str(contract_version)

    # -- weights identity: this file's sha256 (no learned weights) ----------

    def weights_sha256(self) -> str:
        return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    # -- §2 API --------------------------------------------------------------

    def predict_many(
        self,
        state_samples: Sequence[StateSample],
        plans: Sequence[PlanCandidate],
        controller_state: ControllerState,
        *,
        horizon_s: float = DEFAULT_HORIZON_S,
        trajectory_s: Optional[float] = None,
    ) -> "list[Prediction]":
        horizon_s = float(horizon_s)
        part_a = float(self.part_a_horizon_s)
        if self._contract_version == CONTRACT_VERSION_V2:
            raise NotImplementedError(
                "ConstSpeedPredictor does not emit 'wd01_predict/2' outputs "
                "(no future-safety fields); use ConstSpeedPredictorV2"
            )
        if trajectory_s is None:
            traj_s = part_a
        else:
            traj_s = float(trajectory_s)
            if traj_s > part_a + 1e-9:
                raise ValueError(
                    f"trajectory_s {traj_s} > part_a_horizon_s {part_a}: this "
                    "implementation rejects horizons beyond its Part A window"
                )

        samples = validate_state_samples(list(state_samples))
        for plan in plans:
            validate_plan(plan, controller_state)
        if not plans:
            raise ValueError("plans must be non-empty")

        cav_ids = sorted(controller_state)
        native_plan, native_kept_walls = native_reference(controller_state, cav_ids)

        K = len(samples)
        T = int(round(traj_s)) + 1

        # native is computed ONCE per call and shared by every Prediction (§2)
        native_delay = np.array(
            [[self._native_delay_k(samples[k], horizon_s) for k in range(K)]]
        )  # [M=1, K]
        native_traj = self._trajectories(samples, T)

        return [
            self._predict_one(
                samples=samples,
                plan=plan,
                native_plan=native_plan,
                native_kept_walls=native_kept_walls,
                native_delay=native_delay,
                native_traj=native_traj,
                horizon_s=horizon_s,
                T=T,
            )
            for plan in plans
        ]

    # -- internals ------------------------------------------------------------

    def _native_delay_k(self, sample: StateSample, horizon_s: float) -> float:
        t0 = sample.meta.time_s
        total = 0.0
        for t_s, _entrance, _speed in _arrival_rows(sample.arrivals):
            rel = min(max(t_s - t0, 0.0), horizon_s)
            total += ARRIVAL_ROW_COST_S * (horizon_s - rel)
        tracked, _len_m = _tracked(sample.hybrid)
        for _vid, (_s0, _lane, v0) in tracked.items():
            total += SPEED_LAG_COST_S * horizon_s * max(0.0, 1.0 - v0 / self._free_flow_mps)
        return total

    def _trajectories(
        self, samples: Sequence[StateSample], T: int
    ) -> "dict[str, np.ndarray]":
        """Tracked trajectories [M=1, K, T, 4]: constant speed from t0; whole
        row NaN from the exit step on (§4.3). Ids must be stable across
        samples."""
        K = len(samples)
        ids = sorted({vid for s in samples for vid in _tracked(s.hybrid)[0]})
        arrs = {vid: np.full((1, K, T, 4), np.nan, dtype=np.float64) for vid in ids}
        for k, s in enumerate(samples):
            tracked, len_m = _tracked(s.hybrid)
            t0 = s.meta.time_s
            for vid in ids:
                if vid not in tracked:
                    raise KeyError(
                        f"tracked id {vid!r} missing in sample {k}: tracked ids must "
                        "be stable across samples"
                    )
                s0, lane, v0 = (
                    float(tracked[vid][0]),
                    int(tracked[vid][1]),
                    float(tracked[vid][2]),
                )
                arr = arrs[vid]
                if v0 > 0.0:
                    j_exit = min(int(math.ceil((len_m - s0) / v0)), T)
                else:
                    j_exit = T
                j_exit = max(j_exit, 0)
                if j_exit > 0:
                    steps = np.arange(j_exit)
                    arr[0, k, :j_exit, 0] = t0 + steps
                    arr[0, k, :j_exit, 1] = s0 + v0 * steps
                    arr[0, k, :j_exit, 2] = lane
                    arr[0, k, :j_exit, 3] = v0
        return arrs

    def _plan_extra_s(self, plan: PlanCandidate) -> float:
        extra = sum(ACTION_COST_S[a] for a in plan.per_cav.values())
        extra += sum(WALL_PHASE_COST_S[w.phase] for w in plan.walls)
        return float(extra)

    def _predict_one(
        self,
        *,
        samples: Sequence[StateSample],
        plan: PlanCandidate,
        native_plan: PlanCandidate,
        native_kept_walls: Tuple[str, ...],
        native_delay: "np.ndarray",
        native_traj: "dict[str, np.ndarray]",
        horizon_s: float,
        T: int,
    ) -> Prediction:
        K = len(samples)
        M = 1

        extra = self._plan_extra_s(plan)
        plan_delay = native_delay + extra  # [1, K]
        delay_vs = np.full((M, K), extra, dtype=np.float64)
        part_a_delay = 0.6 * plan_delay
        part_b_delay = 0.4 * plan_delay

        all_clear = np.zeros((M, K), dtype=bool)
        min_ttc = np.full((M, K), 30.0, dtype=np.float64)
        zeros = np.zeros((M, K), dtype=np.float64)

        plan_traj = native_traj  # constant-speed stub: same motion under any plan

        walls: "dict[str, WallOutcome]" = {}
        specs = {w.wall_id: w for w in plan.walls}
        for w in native_plan.walls:
            specs.setdefault(w.wall_id, w)
        for wall_id, w in specs.items():
            walls[wall_id] = WallOutcome(
                assembled=np.ones((M, K), dtype=bool),
                assembled_at_s=np.full((M, K), ASSEMBLED_AT_S[w.phase], dtype=np.float64),
                safety_abort=np.zeros((M, K), dtype=bool),
                abort_reason=np.zeros((M, K), dtype=np.int8),
            )

        meta = Meta(
            contract_version=self._contract_version,
            model_id=self.model_id,
            weights_sha256=(self.weights_sha256(),),
            sample_ids=tuple(s.meta.sample_id for s in samples),
            member_ids=(MEMBER_ID,),
            part_a_horizon_s=float(self.part_a_horizon_s),
            safety_window_s=float(self.part_a_horizon_s),
            horizon_s=horizon_s,
            native_kept_walls=tuple(native_kept_walls),
            native_unlocked_available=False,
            device="cpu",
            wall_clock_s=0.0,
        )

        # §9 (v1.1): required cell-validity outputs. The stub marks every cell
        # valid and reports a closing pair wherever it reports a finite
        # min_ttc_s (30 s) — consistent with the §9 NaN-iff rule.
        cell_kwargs: dict = {}
        if self._contract_version == CONTRACT_VERSION_V1_1:
            cell_kwargs = dict(
                cell_valid=np.ones((M, K), dtype=bool),
                native_cell_valid=np.ones((M, K), dtype=bool),
                has_closing_pair=np.ones((M, K), dtype=bool),
                native_has_closing_pair=np.ones((M, K), dtype=bool),
            )

        return Prediction(
            delay_vs_native_s=delay_vs,
            plan_delay_s=plan_delay,
            native_delay_s=native_delay,
            delay_part_a_s=part_a_delay,
            delay_part_b_s=part_b_delay,
            collision=all_clear,
            min_ttc_s=min_ttc,
            ttc_lt3_exposure_s=zeros,
            hard_brake_s=zeros,
            native_collision=all_clear,
            native_min_ttc_s=min_ttc,
            native_ttc_lt3_exposure_s=zeros,
            native_hard_brake_s=zeros,
            trajectories_tracked=plan_traj,
            native_trajectories_tracked=native_traj,
            walls=walls,
            meta=meta,
            **cell_kwargs,
        )


MODEL_ID_V2 = "reference_const_speed/2"
CONTINUATION_POLICY_ID = "reference_const_speed/2#pi"
VALUE_MODEL_ID = "reference_const_speed/2#value"


class ConstSpeedPredictorV2(ConstSpeedPredictor):
    """NOT A MODEL, conformance reference only — the v2 (commit-window) stub.

    Subclasses the v1 constant-speed reference and emits a full
    `wd01_predict/2` output: `delay_part_a_s` accrues over [t0, t0+commit_s],
    `delay_part_b_s` is the continuation value over the remainder, all
    future-safety fields are all-clear with a finite 30 s future min-TTC and a
    closing pair (the conservative learned-model encoding), and every
    future-wall head is p_assembly_completes=1 with a finite expected time.
    M = 1, part_a_horizon_s = 60 s; trajectory_s=None means commit_s.
    `future_min_ttc_validated` stays False (the stub's future TTC head is NOT
    validated), so live_mode_allowed() is False and the G-1 future floor is
    off — exactly the un-validated regime the contract starts in."""

    model_id = MODEL_ID_V2

    def __init__(
        self,
        free_flow_mps: float = FREE_FLOW_MPS,
        commit_s: float = DEFAULT_COMMIT_S,
        continuation: str = CONTINUATION_PLANNER,
    ):
        if continuation not in CONTINUATIONS:
            raise ValueError(
                f"continuation must be one of {CONTINUATIONS}, got {continuation!r}"
            )
        commit_s = float(commit_s)
        if not (math.isfinite(commit_s) and commit_s > 0.0):
            raise ValueError(f"commit_s must be finite > 0, got {commit_s!r}")
        if commit_s > self.part_a_horizon_s + 1e-9:
            raise ValueError(
                f"commit_s {commit_s} > part_a_horizon_s {self.part_a_horizon_s}: "
                "the commit window lies inside the Part A rollout"
            )
        super().__init__(free_flow_mps=free_flow_mps, contract_version=CONTRACT_VERSION_V2)
        self._commit_s = commit_s
        self._continuation = str(continuation)

    # -- §2 (v2) API ----------------------------------------------------------

    def predict_many(
        self,
        state_samples: Sequence[StateSample],
        plans: Sequence[PlanCandidate],
        controller_state: ControllerState,
        *,
        horizon_s: float = DEFAULT_HORIZON_S,
        commit_s: "float | None" = None,
        continuation: "str | None" = None,
        trajectory_s: "float | None" = None,
    ) -> "list[Prediction]":
        commit_s = self._commit_s if commit_s is None else float(commit_s)
        continuation = self._continuation if continuation is None else str(continuation)
        if continuation not in CONTINUATIONS:
            raise ValueError(
                f"continuation must be one of {CONTINUATIONS}, got {continuation!r}"
            )
        if not (math.isfinite(commit_s) and commit_s > 0.0):
            raise ValueError(f"commit_s must be finite > 0, got {commit_s!r}")
        part_a = float(self.part_a_horizon_s)
        if commit_s > part_a + 1e-9:
            raise ValueError(
                f"commit_s {commit_s} > part_a_horizon_s {part_a}: the commit "
                "window lies inside the Part A rollout"
            )
        if trajectory_s is None:
            traj_s = commit_s  # §2 (v2): trajectory_s=None means commit_s
        else:
            traj_s = float(trajectory_s)
            if traj_s > part_a + 1e-9:
                raise ValueError(
                    f"trajectory_s {traj_s} > part_a_horizon_s {part_a}: this "
                    "implementation rejects horizons beyond its Part A window"
                )

        samples = validate_state_samples(list(state_samples))
        for plan in plans:
            validate_plan(plan, controller_state)
        if not plans:
            raise ValueError("plans must be non-empty")

        horizon_s = float(horizon_s)
        cav_ids = sorted(controller_state)
        native_plan, native_kept_walls = native_reference(controller_state, cav_ids)

        K = len(samples)
        T = int(round(traj_s)) + 1
        native_delay = np.array(
            [[self._native_delay_k(samples[k], horizon_s) for k in range(K)]]
        )  # [M=1, K]
        native_traj = self._trajectories(samples, T)

        return [
            self._predict_one_v2(
                samples=samples,
                plan=plan,
                native_plan=native_plan,
                native_kept_walls=native_kept_walls,
                native_delay=native_delay,
                native_traj=native_traj,
                horizon_s=horizon_s,
                commit_s=commit_s,
                continuation=continuation,
                T=T,
            )
            for plan in plans
        ]

    # -- internals -------------------------------------------------------------

    def _predict_one_v2(
        self,
        *,
        samples: Sequence[StateSample],
        plan: PlanCandidate,
        native_plan: PlanCandidate,
        native_kept_walls: Tuple[str, ...],
        native_delay: "np.ndarray",
        native_traj: "dict[str, np.ndarray]",
        horizon_s: float,
        commit_s: float,
        continuation: str,
        T: int,
    ) -> Prediction:
        K = len(samples)
        M = 1

        extra = self._plan_extra_s(plan)
        plan_delay = native_delay + extra  # [1, K]
        delay_vs = np.full((M, K), extra, dtype=np.float64)
        # §3 (v2): part_a accrues inside the commit window, part_b is the
        # continuation value; the split is deterministic and sums to the total.
        frac = commit_s / horizon_s
        part_a_delay = plan_delay * frac
        part_b_delay = plan_delay * (1.0 - frac)

        all_clear = np.zeros((M, K), dtype=bool)
        min_ttc = np.full((M, K), 30.0, dtype=np.float64)
        zeros = np.zeros((M, K), dtype=np.float64)
        ones = np.ones((M, K), dtype=bool)

        plan_traj = native_traj  # constant-speed stub: same motion under any plan

        walls: "dict[str, WallOutcome]" = {}
        specs = {w.wall_id: w for w in plan.walls}
        for w in native_plan.walls:
            specs.setdefault(w.wall_id, w)
        for wall_id, w in specs.items():
            at_s = ASSEMBLED_AT_S[w.phase]
            walls[wall_id] = WallOutcome(
                assembled=np.ones((M, K), dtype=bool),
                assembled_at_s=np.full((M, K), at_s, dtype=np.float64),
                safety_abort=np.zeros((M, K), dtype=bool),
                abort_reason=np.zeros((M, K), dtype=np.int8),
                # §3 (v2) future-wall heads: the stub's walls always complete
                # (at the phase's assembly time); NaN never appears because
                # p_assembly_completes == 1 everywhere.
                p_assembly_completes=np.ones((M, K), dtype=np.float64),
                expected_assembly_time_s=np.full((M, K), at_s, dtype=np.float64),
                p_assembly_abort=np.zeros((M, K), dtype=np.float64),
            )

        meta = Meta(
            contract_version=CONTRACT_VERSION_V2,
            model_id=self.model_id,
            weights_sha256=(self.weights_sha256(),),
            sample_ids=tuple(s.meta.sample_id for s in samples),
            member_ids=(MEMBER_ID,),
            part_a_horizon_s=float(self.part_a_horizon_s),
            safety_window_s=float(commit_s),  # §3: the window fields cover Δ
            horizon_s=horizon_s,
            native_kept_walls=tuple(native_kept_walls),
            native_unlocked_available=False,
            device="cpu",
            wall_clock_s=0.0,
            commit_s=float(commit_s),
            continuation=continuation,
            continuation_policy_id=(
                None if continuation == CONTINUATION_NATIVE else CONTINUATION_POLICY_ID
            ),
            value_model_id=VALUE_MODEL_ID,
            value_model_sha256=(self.weights_sha256(),),
            future_min_ttc_validated=False,
            future_min_ttc_validation_id=None,
        )

        # §9 (v1.1) window cell books — REQUIRED on v2 outputs as well — plus
        # the §3 (v2) future-safety outputs: all-clear, conservative
        # closing-pair encoding (pair present, finite 30 s q05, p_collision
        # 0.0 <= p_max).
        cell_kwargs = dict(
            cell_valid=ones,
            native_cell_valid=ones,
            has_closing_pair=ones,
            native_has_closing_pair=ones,
            future_cell_valid=ones,
            native_future_cell_valid=ones,
            future_has_closing_pair=ones,
            native_future_has_closing_pair=ones,
            future_ttc_lt3_exposure_s=zeros,
            future_hard_brake_s=zeros,
            p_future_collision=zeros,
            future_min_ttc_q05_s=min_ttc,
            native_future_ttc_lt3_exposure_s=zeros,
            native_future_hard_brake_s=zeros,
            native_p_future_collision=zeros,
            native_future_min_ttc_q05_s=min_ttc,
        )

        return Prediction(
            delay_vs_native_s=delay_vs,
            plan_delay_s=plan_delay,
            native_delay_s=native_delay,
            delay_part_a_s=part_a_delay,
            delay_part_b_s=part_b_delay,
            collision=all_clear,
            min_ttc_s=min_ttc,
            ttc_lt3_exposure_s=zeros,
            hard_brake_s=zeros,
            native_collision=all_clear,
            native_min_ttc_s=min_ttc,
            native_ttc_lt3_exposure_s=zeros,
            native_hard_brake_s=zeros,
            trajectories_tracked=plan_traj,
            native_trajectories_tracked=native_traj,
            walls=walls,
            meta=meta,
            **cell_kwargs,
        )


__all__ = [
    "ConstSpeedPredictor",
    "ConstSpeedPredictorV2",
    "MODEL_ID",
    "MODEL_ID_V2",
    "PART_A_HORIZON_S",
]
