"""ContractPredictorAdapter: the planner's MockWorldModel behind the official
WD-01 predict() conformance kit (PL01-E).

The kit (the wd01b worktree's output/wd01/predict/, read-only reference; see its
REPORT_PREDICT.md "How implementations run the conformance kit") drives any
object satisfying its ``Predictor`` protocol with the kit's own frozen
dataclasses (``predict.contract.*``) and validates the returned kit
``Prediction`` objects with ``validate_prediction``. This module is the
translation layer between that surface and the planner's ``interface.py``
types:

- inbound: kit ``PlanCandidate`` / ``WallSpec`` / ``ControllerStateEntry`` /
  ``StateSample`` -> planner ``PlanCandidate`` / ``WallSpec`` / state dicts /
  meta-mapping samples. Field names coincide (wall_id, members, speed_mps,
  phase, release_at_s, hold_s), so walls carry over 1:1 including the kit's
  ``hold_s=math.inf`` native indefinite ASSEMBLE hold and
  ``release_at_s=0.0`` "release now" (§5 dissolution encoding). Actions
  coincide: native/damping/pi-external-u; ``wall`` appears only in the
  controller state (wall members never appear in per_cav on either side).
- outbound: planner ``Prediction`` -> kit ``Prediction`` with kit ``Meta``
  (the mock's meta mapping already carries every §4.6 field incl.
  weights_sha256, wall_clock_s, native_kept_walls,
  native_unlocked_available) and planner ``WallPrediction`` -> kit
  ``WallOutcome``. The optional §4.5 native_unlocked_* diagnostics stay
  unset (the mock reports ``native_unlocked_available=False``), and
  ``trajectories_hidden`` stays None — both are contract-optional.

The wrapped mock is built per call from the controller_state roster
(fresh, scripting-free => deterministic); pass ``mock_factory`` to script
behavior. The kit is imported lazily so the planner package never
hard-depends on the external reference tree.

Zero simulation: no SUMO/traci/libsumo import anywhere.
"""

import dataclasses
import time
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from .interface import PlanCandidate, Prediction, WallSpec
from .mock import MockWorldModel

__all__ = ["ContractPredictorAdapter"]

KitModule = Any


def kit_contract() -> KitModule:
    """The kit's ``predict.contract`` module (lazy, needs its tree on
    sys.path — see tests/planner/test_contract_conformance.py)."""
    try:
        from predict import contract as module
    except ImportError as exc:  # pragma: no cover - path-dependent
        raise RuntimeError(
            "the WD-01 predict conformance kit is not importable; insert "
            "its tree (e.g. the wd01b worktree's output/wd01) into sys.path "
            "first"
        ) from exc
    return module


def _as_bool_array(value):
    """None passthrough (contract-optional field); bool[M, K] otherwise."""
    if value is None:
        return None
    return np.asarray(value).astype(bool)


def _as_mapping_entry(entry: Any) -> Mapping[str, Any]:
    """Kit ControllerStateEntry (or planner dict) -> planner state dict."""
    if isinstance(entry, Mapping):
        return dict(entry)
    if dataclasses.is_dataclass(entry):
        return dataclasses.asdict(entry)
    raise TypeError(
        f"controller_state entry must be a mapping or dataclass, got {type(entry).__name__}"
    )


class ContractPredictorAdapter:
    """``MockWorldModel`` behind the kit's Predictor protocol (§2)."""

    def __init__(
        self,
        *,
        contract_version: str = "wd01_predict/1",
        part_a_horizon_s: float = 60.0,
        model_id: str = "mock/1",
        mock_factory: Optional[Callable[[Sequence[str]], MockWorldModel]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if contract_version not in ("wd01_predict/1", "wd01_predict/1.1"):
            raise ValueError(
                f"unsupported contract_version {contract_version!r}; "
                "expected 'wd01_predict/1' or 'wd01_predict/1.1'")
        if part_a_horizon_s <= 0:
            raise ValueError("part_a_horizon_s must be positive")
        self.contract_version = contract_version
        self.part_a_horizon_s = float(part_a_horizon_s)
        self.model_id = model_id
        self._mock_factory = mock_factory
        self._clock = clock

    # ------------------------------------------------------------------ api
    def predict_many(
        self,
        state_samples: Sequence[Any],
        plans: Sequence[Any],
        controller_state: Optional[Mapping[str, Any]] = None,
        *,
        horizon_s: float = 600.0,
        trajectory_s: Optional[float] = None,
    ) -> "list[Prediction]":
        kit = kit_contract()
        controller_state = dict(controller_state or {})
        mock = self._mock_for(controller_state)
        planner_plans = [self._plan_in(plan, kit) for plan in plans]
        planner_samples = [self._sample_in(sample) for sample in state_samples]
        planner_state = {cav: _as_mapping_entry(entry) for cav, entry in controller_state.items()}
        started = self._clock()
        inner = mock.predict_many(
            planner_samples,
            planner_plans,
            controller_state=planner_state,
            horizon_s=horizon_s,
            trajectory_s=trajectory_s,
        )
        wall_clock_s = self._clock() - started
        return [self._prediction_out(prediction, kit, wall_clock_s) for prediction in inner]

    def predict(
        self,
        state_samples: Sequence[Any],
        plan: Any,
        controller_state: Optional[Mapping[str, Any]] = None,
        **kw: Any,
    ) -> Prediction:
        return self.predict_many(state_samples, [plan], controller_state, **kw)[0]

    # ------------------------------------------------------------- inbound
    def _mock_for(self, controller_state: Mapping[str, Any]) -> MockWorldModel:
        if self._mock_factory is not None:
            return self._mock_factory(controller_state)
        return MockWorldModel(
            cav_ids=tuple(sorted(controller_state)),
            part_a_horizon_s=self.part_a_horizon_s,
            model_id=self.model_id,
        )

    @staticmethod
    def _plan_in(plan: Any, kit: KitModule) -> PlanCandidate:
        """Kit PlanCandidate -> planner PlanCandidate (fields coincide)."""
        if isinstance(plan, PlanCandidate):
            return plan
        if isinstance(plan, kit.PlanCandidate):
            walls = tuple(
                WallSpec(
                    wall_id=w.wall_id,
                    members=tuple(w.members),
                    speed_mps=w.speed_mps,
                    phase=w.phase,
                    release_at_s=w.release_at_s,
                    hold_s=w.hold_s,
                )
                for w in plan.walls
            )
            return PlanCandidate(
                per_cav=dict(plan.per_cav), walls=walls, plan_version=int(plan.plan_version)
            )
        raise TypeError(f"plan must be a kit or planner PlanCandidate, got {type(plan).__name__}")

    @staticmethod
    def _sample_in(sample: Any) -> SimpleNamespace:
        """Kit StateSample -> the meta-mapping sample the mock reads."""
        meta = getattr(sample, "meta", None)
        if isinstance(meta, Mapping):
            meta = dict(meta)
        elif dataclasses.is_dataclass(meta):
            meta = dataclasses.asdict(meta)
        else:
            raise TypeError("StateSample.meta must be a mapping or dataclass")
        return SimpleNamespace(
            hybrid=getattr(sample, "hybrid", None),
            arrivals=getattr(sample, "arrivals", ()),
            meta=meta,
        )

    # ------------------------------------------------------------ outbound
    def _prediction_out(
        self, prediction: Prediction, kit: KitModule, wall_clock_s: float
    ) -> Any:
        meta = dict(prediction.meta)
        meta["wall_clock_s"] = float(wall_clock_s)
        reported_version = (
            self.contract_version if self._is_v11() else str(meta["contract_version"])
        )
        kit_meta = kit.Meta(
            contract_version=reported_version,
            model_id=str(meta["model_id"]),
            weights_sha256=tuple(str(w) for w in meta["weights_sha256"]),
            sample_ids=tuple(str(s) for s in meta["sample_ids"]),
            member_ids=tuple(str(m) for m in meta["member_ids"]),
            part_a_horizon_s=float(meta["part_a_horizon_s"]),
            safety_window_s=float(meta["safety_window_s"]),
            horizon_s=float(meta["horizon_s"]),
            native_kept_walls=tuple(str(w) for w in meta["native_kept_walls"]),
            native_unlocked_available=bool(meta["native_unlocked_available"]),
            device=str(meta["device"]),
            wall_clock_s=float(wall_clock_s),
        )
        walls = {
            wall_id: kit.WallOutcome(
                assembled=np.asarray(block.assembled),
                assembled_at_s=np.asarray(block.assembled_at_s),
                safety_abort=np.asarray(block.safety_abort),
                abort_reason=np.asarray(block.abort_reason),
            )
            for wall_id, block in prediction.walls.items()
        }
        return kit.Prediction(
            delay_vs_native_s=np.asarray(prediction.delay_vs_native_s),
            plan_delay_s=np.asarray(prediction.plan_delay_s),
            native_delay_s=np.asarray(prediction.native_delay_s),
            delay_part_a_s=np.asarray(prediction.delay_part_a_s),
            delay_part_b_s=np.asarray(prediction.delay_part_b_s),
            collision=np.asarray(prediction.collision),
            min_ttc_s=np.asarray(prediction.min_ttc_s),
            ttc_lt3_exposure_s=np.asarray(prediction.ttc_lt3_exposure_s),
            hard_brake_s=np.asarray(prediction.hard_brake_s),
            native_collision=np.asarray(prediction.native_collision),
            native_min_ttc_s=np.asarray(prediction.native_min_ttc_s),
            native_ttc_lt3_exposure_s=np.asarray(prediction.native_ttc_lt3_exposure_s),
            native_hard_brake_s=np.asarray(prediction.native_hard_brake_s),
            trajectories_tracked=dict(prediction.trajectories_tracked),
            native_trajectories_tracked=dict(prediction.native_trajectories_tracked),
            walls=walls,
            meta=kit_meta,
            # contract v1.1 diagnostics (§9): emitted in 1.1 mode only -- a
            # v1 output carrying them is rejected by the kit's validator.
            # The planner Prediction carries them as optional fields and the
            # mock always reports them.
            **(
                {
                    "cell_valid": _as_bool_array(prediction.cell_valid),
                    "native_cell_valid": _as_bool_array(
                        prediction.native_cell_valid
                    ),
                    "has_closing_pair": _as_bool_array(
                        prediction.has_closing_pair
                    ),
                    "native_has_closing_pair": _as_bool_array(
                        prediction.native_has_closing_pair
                    ),
                }
                if self._is_v11()
                else {}
            ),
            trajectories_hidden=None,  # contract-optional; the mock has none
        )

    def _is_v11(self) -> bool:
        return self.contract_version == "wd01_predict/1.1"
