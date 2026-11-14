"""MockWorldModel: a deterministic, scriptable world model for planner tests.

Implements the frozen ``wd01_predict/1`` contract (see interface.py) with:

- SCRIPTED plans: per-plan-key overrides for delay and safety arrays
  (``plan_delay_override``, ``safety_script``);
- a DEFAULT ADDITIVE model: per-CAV action delay effects + pairwise
  interaction delay terms + trajectory position/speed offsets (own-action and
  coupling), so grouping tests can create measured interactions and safety
  tests can script per-cell arrays (scalar or any array broadcastable to
  [M, K]);

All values are plain Python floats, bools or array-likes; every call is
deterministic. Native arrays are computed ONCE per call and shared by every
Prediction (contract section 2). Structural contract validation (full
per_cav map, wall members, locked CAVs, trajectory horizon) mirrors contract
section 3 and raises ValueError.

Zero simulation: no SUMO/traci/libsumo import anywhere in this package.
"""

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from .config import PlannerConfig  # noqa: F401  (re-exported for convenience)
from .interface import (
    ACTIONS,
    CONTRACT_VERSION,
    CONTRACT_VERSION_V2,
    PlanCandidate,
    Prediction,
    StateSample,
    WallPrediction,
    WorldModel,
    locked_entries,
    plan_key,
)
from .walls import ACTIVE_WALL_PHASES

_SAMPLE_META_FIELDS = ("sample_id", "time_s", "merge_observed", "coverage")


class MockWorldModel(WorldModel):
    """Scriptable additive world model (see module docstring)."""

    def __init__(
        self,
        cav_ids: Sequence[str],
        *,
        n_members: int = 1,
        part_a_horizon_s: float = 60.0,
        native_delay_s: Any = 100.0,
        base_speed_mps: float = 20.0,
        model_id: str = "mock/1",
        contract_version: str = CONTRACT_VERSION,
    ):
        if contract_version not in ("wd01_predict/1", "wd01_predict/1.1", CONTRACT_VERSION_V2):
            raise ValueError(
                f"unsupported contract_version {contract_version!r}; expected "
                "'wd01_predict/1', 'wd01_predict/1.1' or 'wd01_predict/2'"
            )
        self.cav_ids = tuple(cav_ids)
        if len(set(self.cav_ids)) != len(self.cav_ids):
            raise ValueError("cav_ids must be unique")
        self.n_members = int(n_members)
        if self.n_members < 1:
            raise ValueError("n_members must be >= 1")
        self.part_a_horizon_s = float(part_a_horizon_s)
        self.native_delay_s = native_delay_s
        self.base_speed_mps = float(base_speed_mps)
        self.model_id = model_id
        self.contract_version = contract_version
        self.part_a_fraction = 0.5
        self.default_min_ttc_s = 10.0
        self.assembly_duration_s = 10.0
        # ---- scripting knobs (mutate directly in tests) ----
        self.action_delay: dict = {}  # {cav: {action: scalar | [M,K]-like}}
        self.pair_delay: dict = {}  # {(cav_i,cav_j): {(a_i,a_j): value}}
        self.plan_delay_override: dict = {}  # {plan_key: value} replaces plan_delay_s
        self.plan_dvn_override: dict = {}  # {plan_key: value} replaces delay_vs_native_s
        self.safety_script: dict = {}  # {plan_key: {field: value}}
        self.native_safety: dict = {}  # {field: value} for the native_* arrays
        self.lane_of: dict = {}  # {cav: lane_id} for tracked trajectories
        self.s0_of: dict = {}  # {cav: t0 longitudinal position [m]}
        self.wall_delay: dict = {}  # {wall_id: value} while assemble/hold
        self.wall_coupling_pos: dict = {}  # {(wall_id, cav): value} while active
        self.wall_coupling_speed: dict = {}  # {(wall_id, cav): value}
        self.traj_pos_offset: dict = {}  # {cav: {own action: scalar | [K]}}
        self.traj_speed_offset: dict = {}  # {cav: {own action: scalar | [K]}}
        self.traj_coupling_pos: dict = {}  # {(i,j): {action of i: value}} -> j's s
        self.traj_coupling_speed: dict = {}  # {(i,j): {action of i: value}} -> j's v
        self.exit_step: dict = {}  # {cav: {own action: t_index}} NaN from there
        self.native_exit_step: dict = {}  # {cav: t_index}
        self.wall_script: dict = {}  # {wall_id: {field: value}} (safety_abort etc.)
        # ---- contract v2 scripting (used when contract_version is v2) ----
        # merged into every meta (e.g. future_min_ttc_validated /
        # future_min_ttc_validation_id)
        self.meta_script: dict = {}
        # ---- contract v2.1 (Part A v2 general) scripting, OFF by default:
        # {field: scalar | [M,K]-like} for n_tracked / n_exited_tracked /
        # n_free_road_assumed / untracked_arrivals_expected. Empty (default)
        # leaves all four OPTIONAL Prediction fields None -- the mock keeps
        # working unchanged, like any predictor that does not emit them.
        self.part_a_v21_fields: dict = {}
        # ---- introspection ----
        self.calls: list = []

    # ------------------------------------------------------------------ API
    def predict_many(
        self,
        state_samples: Sequence[StateSample],
        plans: Sequence[PlanCandidate],
        *,
        controller_state: Optional[Mapping[str, Mapping[str, Any]]] = None,
        horizon_s: float = 600.0,
        trajectory_s: Optional[float] = None,
        commit_s: float = 30.0,
        continuation: str = "planner",
    ) -> "list[Prediction]":
        plans = list(plans)
        if not plans:
            return []
        samples = list(state_samples)
        if not samples:
            raise ValueError("state_samples must be non-empty")
        if continuation not in ("planner", "native"):
            raise ValueError("continuation must be 'planner' or 'native'")
        self._validate_samples(samples)
        used_trajectory_s = self.part_a_horizon_s if trajectory_s is None else float(trajectory_s)
        if used_trajectory_s > self.part_a_horizon_s + 1e-9:
            raise ValueError(
                f"trajectory_s {used_trajectory_s} > part_a_horizon_s {self.part_a_horizon_s}"
            )
        n_steps = int(round(used_trajectory_s)) + 1  # T includes t0
        shape = (self.n_members, len(samples))
        locked = locked_entries(controller_state)
        roster = set(self.cav_ids)
        for plan in plans:
            self._validate_plan(plan, roster, locked)
        is_v2 = self.contract_version == CONTRACT_VERSION_V2
        call = {
            "n_plans": len(plans),
            "n_samples": len(samples),
            "n_members": self.n_members,
            "horizon_s": float(horizon_s),
            "trajectory_s": used_trajectory_s,
        }
        if is_v2:
            # v2 log extension only: v1/v1.1 call records are unchanged
            call["commit_s"] = float(commit_s)
            call["continuation"] = str(continuation)
        self.calls.append(call)

        # Native reference: computed ONCE per call, shared by every plan.
        # N1 (contract section 5): walls in ASSEMBLE continue (locked); HOLD
        # and RELEASE walls are dissolved. Kept walls are read from
        # controller_state (what is actually on the road), never from plan
        # candidates.
        t0_s = float(samples[0].meta["time_s"])
        wall_phase, wall_time = self._controller_walls(controller_state)
        kept_wall_ids = sorted(wid for wid, phase in wall_phase.items() if phase == "assemble")
        native_delay = self._as_mk(self.native_delay_s, shape)
        for wall_id in kept_wall_ids:
            if wall_id in self.wall_delay:
                native_delay = native_delay + self._as_mk(self.wall_delay[wall_id], shape)
        native_safety = {
            "collision": self._as_bool(self.native_safety.get("collision", False), shape),
            "min_ttc_s": self._as_mk(
                self.native_safety.get("min_ttc_s", self.default_min_ttc_s), shape
            ),
            "ttc_lt3_exposure_s": self._as_mk(
                self.native_safety.get("ttc_lt3_exposure_s", 0.0), shape
            ),
            # T-A: the primary 1.5 s exposure twin (same script keys as the
            # 3 s fields, independent defaults)
            "ttc_lt1p5_exposure_s": self._as_mk(
                self.native_safety.get("ttc_lt1p5_exposure_s", 0.0), shape
            ),
            "hard_brake_s": self._as_mk(self.native_safety.get("hard_brake_s", 0.0), shape),
        }
        if is_v2:
            # contract v2 native future heads (defaults: clean future)
            native_safety.update(
                {
                    "native_future_ttc_lt3_exposure_s": self._as_mk(
                        self.native_safety.get("native_future_ttc_lt3_exposure_s", 0.0), shape
                    ),
                    # T-A: the 1.5 s future twin
                    "native_future_ttc_lt1p5_exposure_s": self._as_mk(
                        self.native_safety.get("native_future_ttc_lt1p5_exposure_s", 0.0), shape
                    ),
                    "native_future_hard_brake_s": self._as_mk(
                        self.native_safety.get("native_future_hard_brake_s", 0.0), shape
                    ),
                    "native_p_future_collision": self._as_mk(
                        self.native_safety.get("native_p_future_collision", 0.0), shape
                    ),
                    "native_future_min_ttc_q05_s": self._as_mk(
                        self.native_safety.get("native_future_min_ttc_q05_s", 2.5), shape
                    ),
                    "native_future_cell_valid": self._as_bool(
                        self.native_safety.get("native_future_cell_valid", True), shape
                    ),
                    "native_future_has_closing_pair": self._as_bool(
                        self.native_safety.get("native_future_has_closing_pair", True), shape
                    ),
                }
            )
        native_trajectories = {
            cav: self._trajectory(cav, None, shape, n_steps, native=True) for cav in self.cav_ids
        }

        out = []
        for plan in plans:
            key = plan_key(plan)
            if key in self.plan_dvn_override:
                # Scripted dvn is authoritative: no additive wall effects.
                delay_vs_native = self._as_mk(self.plan_dvn_override[key], shape)
                plan_delay = native_delay + delay_vs_native
            else:
                plan_delay = self._plan_delay(plan, key, native_delay, shape)
                for wall in plan.walls:
                    if wall.phase in ACTIVE_WALL_PHASES and wall.wall_id in self.wall_delay:
                        plan_delay = plan_delay + self._as_mk(self.wall_delay[wall.wall_id], shape)
                delay_vs_native = plan_delay - native_delay
            script = self.safety_script.get(key, {})
            collision = self._as_bool(script.get("collision", False), shape)
            min_ttc = self._as_mk(script.get("min_ttc_s", self.default_min_ttc_s), shape)
            trajectories = {
                cav: self._trajectory(cav, plan, shape, n_steps) for cav in self.cav_ids
            }
            meta = {
                "contract_version": self.contract_version,
                "model_id": self.model_id,
                "weights_sha256": [f"mock:{self.model_id}"] * self.n_members,
                "sample_ids": [sample.meta["sample_id"] for sample in samples],
                "member_ids": [f"m{i}" for i in range(self.n_members)],
                "part_a_horizon_s": used_trajectory_s,
                "safety_window_s": (float(commit_s) if is_v2 else used_trajectory_s),
                "horizon_s": float(horizon_s),
                "native_kept_walls": kept_wall_ids,
                "native_unlocked_available": False,
                "device": "cpu",
                "wall_clock_s": 0.0,
            }
            if is_v2:
                # contract v2 section 4 meta additions
                meta.update(
                    {
                        "commit_s": float(commit_s),
                        "continuation": str(continuation),
                        "continuation_policy_id": (
                            None if continuation == "native" else f"{self.model_id}:pi"
                        ),
                        "value_model_id": [f"{self.model_id}:v0"] * self.n_members,
                        "value_model_sha256": [f"mock-value:{self.model_id}"] * self.n_members,
                        "future_min_ttc_validated": bool(
                            self.meta_script.get("future_min_ttc_validated", False)
                        ),
                        "future_min_ttc_validation_id": self.meta_script.get(
                            "future_min_ttc_validation_id"
                        ),
                        # G-2 (contract v2 section 8): the relative-safety
                        # validation meta (scriptable like the G-1 pair; a
                        # validated producer -- e.g. the perfect simulator --
                        # sets True with its own validation id)
                        "relative_safety_validated": bool(
                            self.meta_script.get("relative_safety_validated", False)
                        ),
                        "relative_safety_validation_id": self.meta_script.get(
                            "relative_safety_validation_id"
                        ),
                    }
                )
                meta.update(self.meta_script)
            future_kwargs = {}
            if is_v2:
                future_kwargs = dict(
                    # contract v2 future heads (defaults: clean future; each
                    # scriptable per plan via safety_script)
                    future_ttc_lt3_exposure_s=self._as_mk(
                        script.get("future_ttc_lt3_exposure_s", 0.0), shape
                    ),
                    native_future_ttc_lt3_exposure_s=native_safety[
                        "native_future_ttc_lt3_exposure_s"
                    ],
                    # T-A: the 1.5 s future twin (scriptable per plan)
                    future_ttc_lt1p5_exposure_s=self._as_mk(
                        script.get("future_ttc_lt1p5_exposure_s", 0.0), shape
                    ),
                    native_future_ttc_lt1p5_exposure_s=native_safety[
                        "native_future_ttc_lt1p5_exposure_s"
                    ],
                    future_hard_brake_s=self._as_mk(script.get("future_hard_brake_s", 0.0), shape),
                    native_future_hard_brake_s=native_safety["native_future_hard_brake_s"],
                    p_future_collision=self._as_mk(script.get("p_future_collision", 0.0), shape),
                    native_p_future_collision=native_safety["native_p_future_collision"],
                    future_min_ttc_q05_s=self._as_mk(
                        script.get("future_min_ttc_q05_s", 2.5), shape
                    ),
                    native_future_min_ttc_q05_s=native_safety["native_future_min_ttc_q05_s"],
                    future_cell_valid=self._as_bool(script.get("future_cell_valid", True), shape),
                    native_future_cell_valid=native_safety["native_future_cell_valid"],
                    future_has_closing_pair=self._as_bool(
                        script.get("future_has_closing_pair", True), shape
                    ),
                    native_future_has_closing_pair=native_safety["native_future_has_closing_pair"],
                )
            # contract v2.1 optional per-cell fields (only when scripted):
            # values broadcast to [M, K]; absent fields stay None.
            part_a_v21_kwargs = {
                name: self._as_mk(value, shape)
                for name, value in self.part_a_v21_fields.items()
                if value is not None
            }
            out.append(
                Prediction(
                    delay_vs_native_s=delay_vs_native,
                    plan_delay_s=plan_delay,
                    native_delay_s=native_delay,
                    delay_part_a_s=plan_delay * self.part_a_fraction,
                    delay_part_b_s=plan_delay * (1.0 - self.part_a_fraction),
                    collision=collision,
                    min_ttc_s=min_ttc,
                    ttc_lt3_exposure_s=self._as_mk(script.get("ttc_lt3_exposure_s", 0.0), shape),
                    native_ttc_lt3_exposure_s=native_safety["ttc_lt3_exposure_s"],
                    # T-A: the primary 1.5 s exposure (scriptable per plan
                    # via "ttc_lt1p5_exposure_s"; independent of the frozen
                    # 3 s field)
                    ttc_lt1p5_exposure_s=self._as_mk(
                        script.get("ttc_lt1p5_exposure_s", 0.0), shape
                    ),
                    native_ttc_lt1p5_exposure_s=native_safety["ttc_lt1p5_exposure_s"],
                    hard_brake_s=self._as_mk(script.get("hard_brake_s", 0.0), shape),
                    native_hard_brake_s=native_safety["hard_brake_s"],
                    native_collision=native_safety["collision"],
                    native_min_ttc_s=native_safety["min_ttc_s"],
                    trajectories_tracked=trajectories,
                    native_trajectories_tracked=native_trajectories,
                    walls={
                        w.wall_id: self._wall_prediction(w, shape, wall_time, t0_s, v2=is_v2)
                        for w in plan.walls
                    },
                    meta=meta,
                    # contract v1.1 diagnostics: valid cells with a closing pair
                    # by default; scriptable per plan ("cell_valid",
                    # "has_closing_pair") and per native reference
                    # ("native_cell_valid", "native_has_closing_pair").
                    cell_valid=self._as_bool(script.get("cell_valid", True), shape),
                    native_cell_valid=self._as_bool(
                        self.native_safety.get("native_cell_valid", True), shape
                    ),
                    has_closing_pair=self._as_bool(script.get("has_closing_pair", True), shape),
                    native_has_closing_pair=self._as_bool(
                        self.native_safety.get("native_has_closing_pair", True), shape
                    ),
                    **future_kwargs,
                    **part_a_v21_kwargs,
                )
            )
        return out

    # -------------------------------------------------------------- helpers
    def _validate_samples(self, samples):
        times = set()
        for sample in samples:
            meta = sample.meta
            missing = [f for f in _SAMPLE_META_FIELDS if f not in meta]
            if missing:
                raise ValueError(f"sample meta missing fields: {missing}")
            times.add(float(meta["time_s"]))
        if len(times) > 1:
            raise ValueError("all K samples must share the same time_s")

    def _validate_plan(self, plan, roster, locked):
        if plan.plan_version != 1:
            raise ValueError("only plan_version 1 exists (wd01_predict/1)")
        for wall in plan.walls:
            for member in wall.members:
                if member not in roster:
                    raise ValueError(f"wall member {member!r} not in roster")
        members = {m: w.wall_id for w in plan.walls for m in w.members}
        expected = roster - set(members)
        if set(plan.per_cav) != expected:
            raise ValueError(
                f"per_cav must be the FULL map of non-wall CAVs: expected "
                f"{sorted(expected)}, got {sorted(plan.per_cav)}"
            )
        for cav, entry in locked.items():
            if cav in plan.per_cav:
                raise ValueError(f"locked CAV {cav!r} must not appear in per_cav")
            if members.get(cav) != entry.get("wall_id"):
                raise ValueError(
                    f"locked CAV {cav!r} may appear only as a member of its "
                    f"existing wall {entry.get('wall_id')!r}"
                )
        for cav, action in plan.per_cav.items():
            if action not in ACTIONS:
                raise ValueError(f"per_cav[{cav!r}] action {action!r} not in {ACTIONS}")

    def _as_mk(self, value, shape):
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(shape, float(arr))
        return np.array(np.broadcast_to(arr, shape), dtype=float)

    def _as_bool(self, value, shape):
        arr = np.asarray(value, dtype=bool)
        if arr.ndim == 0:
            return np.full(shape, bool(arr))
        return np.array(np.broadcast_to(arr, shape), dtype=bool)

    def _plan_delay(self, plan, key, native_delay, shape):
        if key in self.plan_delay_override:
            return self._as_mk(self.plan_delay_override[key], shape)
        total = native_delay.copy()
        for cav, action in plan.per_cav.items():
            effect = self.action_delay.get(cav, {}).get(action)
            if effect is not None:
                total = total + self._as_mk(effect, shape)
        for (cav_i, cav_j), table in self.pair_delay.items():
            pair = (plan.per_cav.get(cav_i), plan.per_cav.get(cav_j))
            effect = table.get(pair)
            if effect is not None:
                total = total + self._as_mk(effect, shape)
        return total

    def _traj_offset(self, value, members, n_samples):
        """Scalar or per-sample [K] offset -> float array broadcastable over
        the trajectory axes [M, K, T]."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return float(arr)
        return np.broadcast_to(arr.reshape(1, n_samples, 1), (members, n_samples, 1))

    def _trajectory(self, cav, plan, shape, n_steps, native=False):
        members, n_samples = shape
        t = np.arange(n_steps, dtype=float)
        pos = np.broadcast_to(
            self.s0_of.get(cav, 0.0) + self.base_speed_mps * t, (members, n_samples, n_steps)
        ).copy()
        speed = np.broadcast_to(
            np.asarray(self.base_speed_mps, dtype=float), (members, n_samples, n_steps)
        ).copy()
        exit_at = None
        if native:
            exit_at = self.native_exit_step.get(cav)
        else:
            action = plan.per_cav.get(cav)
            offset = self.traj_pos_offset.get(cav, {}).get(action)
            if offset is not None:
                pos = pos + self._traj_offset(offset, members, n_samples)
            offset = self.traj_speed_offset.get(cav, {}).get(action)
            if offset is not None:
                speed = speed + self._traj_offset(offset, members, n_samples)
            for other in self.cav_ids:
                if other == cav:
                    continue
                other_action = plan.per_cav.get(other)
                coupling = self.traj_coupling_pos.get((other, cav), {}).get(other_action)
                if coupling is not None:
                    pos = pos + self._traj_offset(coupling, members, n_samples)
                coupling = self.traj_coupling_speed.get((other, cav), {}).get(other_action)
                if coupling is not None:
                    speed = speed + self._traj_offset(coupling, members, n_samples)
            exit_at = self.exit_step.get(cav, {}).get(action)
        arr = np.empty((members, n_samples, n_steps, 4), dtype=float)
        arr[..., 0] = t
        arr[..., 1] = pos
        arr[..., 2] = self.lane_of.get(cav, 0.0)  # lane_id (main_left default)
        arr[..., 3] = speed
        if exit_at is not None:
            arr[..., int(exit_at) :, :] = np.nan  # NaN from the exit step on
        if not native and plan is not None:
            for wall in plan.walls:
                if wall.phase not in ACTIVE_WALL_PHASES:
                    continue
                coupling = self.wall_coupling_pos.get((wall.wall_id, cav))
                if coupling is not None:
                    arr[..., 1] = arr[..., 1] + self._traj_offset(coupling, members, n_samples)
                coupling = self.wall_coupling_speed.get((wall.wall_id, cav))
                if coupling is not None:
                    arr[..., 3] = arr[..., 3] + self._traj_offset(coupling, members, n_samples)
        return arr

    def _controller_walls(self, controller_state):
        """({wall_id: phase}, {wall_id: min time_in_phase_s}) from locked entries."""
        phases: "dict[str, str]" = {}
        times: "dict[str, list]" = {}
        for cav, entry in sorted((controller_state or {}).items()):
            wall_id = entry.get("wall_id")
            if not entry.get("locked") or not wall_id:
                continue
            phases.setdefault(wall_id, entry.get("wall_phase") or "assemble")
            times.setdefault(wall_id, []).append(float(entry.get("time_in_phase_s") or 0.0))
        return phases, {wid: min(ts) for wid, ts in times.items()}

    def _wall_prediction(self, wall, shape, wall_time, t0_s, v2=False):
        script = self.wall_script.get(wall.wall_id, {})
        if "assembled" in script:
            assembled = self._as_bool(script["assembled"], shape)
        elif wall.phase == "hold" or wall.phase == "release":
            assembled = self._as_bool(True, shape)
        else:  # assemble: done once every member has been in phase long enough
            assembled = self._as_bool(
                wall_time.get(wall.wall_id, 0.0) >= self.assembly_duration_s, shape
            )
        assembled_at = (
            self._as_mk(script.get("assembled_at_s", t0_s), shape)
            if assembled.any()
            else self._as_mk(np.nan, shape)
        )
        # contract v2 future-wall heads (None pre-v2; scriptable via
        # wall_script "p_assembly_completes" / "expected_assembly_time_s" /
        # "p_assembly_abort"; NaN expected_assembly_time_s only where the
        # completion probability is 0)
        p_assembly_completes = None
        expected_assembly_time_s = None
        p_assembly_abort = None
        if v2:
            completes = self._as_mk(script.get("p_assembly_completes", 1.0), shape)
            expected = self._as_mk(
                script.get("expected_assembly_time_s", t0_s + self.assembly_duration_s), shape
            )
            zero = completes == 0.0
            expected = np.where(zero, np.nan, expected)
            p_assembly_completes = completes
            expected_assembly_time_s = expected
            p_assembly_abort = self._as_mk(script.get("p_assembly_abort", 0.0), shape)
        return WallPrediction(
            assembled=assembled,
            assembled_at_s=assembled_at,
            safety_abort=self._as_bool(script.get("safety_abort", False), shape),
            abort_reason=np.full(shape, int(script.get("abort_reason", 0)), dtype=np.int8),
            p_assembly_completes=p_assembly_completes,
            expected_assembly_time_s=expected_assembly_time_s,
            p_assembly_abort=p_assembly_abort,
        )
