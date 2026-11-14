"""KitWorldModel + real-state builders for the PL-03a dry run (zero sim).

Two bridges, both READ-ONLY toward the WD-01 trees:

1. :class:`KitWorldModel` wraps any WD-01 kit ``Predictor``
   (``predict.protocol``, e.g. the reference ``ConstSpeedPredictor``) into
   the planner's own ``WorldModel`` interface -- the reverse of
   :mod:`traffic_shepherd.planner.contract_adapter`, whose converters are
   reused conceptually. The wrapped predictor is asked for contract v1.1
   outputs (``meta.contract_version = "wd01_predict/1.1"``), so the
   planner's fail-closed safety sees ``cell_valid`` /
   ``has_closing_pair``. With ``strict=True`` (default) every plan is
   kit-``validate_plan``-checked against the controller_state of the same
   call before it reaches the predictor, and every kit rejection is
   recorded, not swallowed.

2. :func:`build_real_state` builds the decision-time inputs from WD-01's
   p2C dataset (``partA_data.loader``, read-only): observed vehicles from
   the t0 state, CAVs = ``is_cav`` vehicles on WM-30 physical lanes
   (0 main_left / 1 main_right / 2 accel), all CAVs start ``damping``
   unlocked, arrivals converted to the contract's explicit rows, K = 1.
   Frame freshness for the E4 wall filter: every observed physical-lane
   CAV carries ``frames[vid] = t0`` (fresh at decision time); junction
   CAVs (lane == -1) and not-yet-inserted vehicles carry no frame and are
   excluded from wall proposals.

Prominently: predictions of the reference constant-speed predictor are
MEANINGLESS as traffic forecasts -- a PL-03a dry run is an
INTEGRATION/THROUGHPUT check only.
"""

import dataclasses
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .interface import PlanCandidate, Prediction, WallPrediction
from .paths import export_research_env, rd_wt_root

__all__ = [
    "KitUnavailable",
    "KitWorldModel",
    "P2C_DATA_ROOT",
    "PREDICT_KIT_ROOT",
    "RealStateBundle",
    "assert_train_only",
    "build_real_state",
    "load_p2c_loader",
    "load_predict_kit",
]

#: WD-01 predict kit root (read-only sibling worktree; env RD_WT_ROOT with
#: E:/rd-wt default, see paths.py; resolved at import).
PREDICT_KIT_ROOT = str(rd_wt_root() / "wd01" / "output" / "wd01")
P2C_DATA_ROOT = str(rd_wt_root() / "wd01" / "output" / "wd01" / "partA_data")
CONTRACT_VERSION_V1_1 = "wd01_predict/1.1"
ENTRANCE_LABELS = {0: "mainline", 1: "ramp"}
PHYSICAL_LANES = (0, 1, 2)


class KitUnavailable(RuntimeError):
    """A read-only WD-01 reference tree is not importable (precise path in
    the message)."""


def load_predict_kit(search_paths: Sequence[str] = (PREDICT_KIT_ROOT,)):
    """Lazy import of the WD-01 predict kit -> (contract, validate)."""
    import importlib

    for path in search_paths:
        if Path(path).is_dir() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        contract = importlib.import_module("predict.contract")
        validate = importlib.import_module("predict.validate")
    except ImportError as exc:
        raise KitUnavailable(
            f"WD-01 predict kit not importable from {list(search_paths)}: {exc!r}"
        ) from exc
    return contract, validate


def load_p2c_loader(data_root: str = P2C_DATA_ROOT):
    """Lazy import of the p2C dataset loader (read-only reference).

    ``data_root`` is the partA_data directory; its PARENT goes on sys.path
    because the directory is an import package without __init__.py (PEP
    420 namespace package)."""
    # the loader's own tree references (pdl.WD01_SRC, hybrid_state_v1's
    # repo.RESEARCH_DENSE_ROOT) default to stale D: junctions: publish the
    # resolved root BEFORE the sibling imports freeze them
    export_research_env()
    parent = str(Path(data_root).parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if data_root not in sys.path:
        sys.path.insert(0, data_root)
    import importlib

    try:
        return importlib.import_module("partA_data.loader")
    except ImportError as exc:
        raise KitUnavailable(
            f"p2C loader (partA_data.loader) not importable from {data_root}: {exc!r}"
        ) from exc


def assert_train_only(chosen: Sequence[str], splits: Mapping[str, Sequence[str]]) -> None:
    """PL-03a hard rule: the holdout split is never touched."""
    holdout = set(splits.get("holdout", ()))
    bad = sorted(set(chosen) & holdout)
    if bad:
        raise AssertionError(f"holdout roots selected for the dry run: {bad}")


# ------------------------------------------------------------- conversions
def plan_out(plan: PlanCandidate, kit) -> Any:
    """Planner PlanCandidate -> kit PlanCandidate (fields coincide)."""
    walls = tuple(
        kit.WallSpec(
            wall_id=w.wall_id,
            members=tuple(w.members),
            speed_mps=w.speed_mps,
            phase=w.phase,
            release_at_s=w.release_at_s,
            hold_s=w.hold_s,
        )
        for w in plan.walls
    )
    return kit.PlanCandidate(
        plan_version=int(plan.plan_version),
        per_cav=dict(plan.per_cav),
        walls=walls,
    )


def controller_state_out(controller_state, kit):
    """Planner state dicts -> kit ControllerStateEntry values (contract
    defaults filled for absent optional fields)."""
    out = {}
    for cav, entry in (controller_state or {}).items():
        if dataclasses.is_dataclass(entry):
            out[cav] = entry
            continue
        e = dict(entry)
        out[cav] = kit.ControllerStateEntry(
            action=str(e.get("action", "native")),
            wall_id=e.get("wall_id"),
            wall_phase=e.get("wall_phase"),
            time_in_phase_s=float(e.get("time_in_phase_s") or 0.0),
            wall_speed_mps=e.get("wall_speed_mps"),
            locked=bool(e.get("locked", False)),
        )
    return out


def prediction_in(kit_pred, kit) -> Prediction:
    """Kit Prediction -> planner Prediction (v1.1 fields kept)."""
    meta = kit_pred.meta
    meta_dict = {
        "contract_version": meta.contract_version,
        "model_id": meta.model_id,
        "weights_sha256": list(meta.weights_sha256),
        "sample_ids": list(meta.sample_ids),
        "member_ids": list(meta.member_ids),
        "part_a_horizon_s": float(meta.part_a_horizon_s),
        "safety_window_s": float(meta.safety_window_s),
        "horizon_s": float(meta.horizon_s),
        "native_kept_walls": list(meta.native_kept_walls),
        "native_unlocked_available": bool(meta.native_unlocked_available),
        "device": str(meta.device),
        "wall_clock_s": float(meta.wall_clock_s),
    }
    # contract v2.1 optional meta (None when the kit predictor does not
    # report it): part_a_model / param_set_id / recon_schema_version /
    # part_a_scope. A kit Meta without the attributes maps to None.
    for key in ("part_a_model", "param_set_id", "recon_schema_version", "part_a_scope"):
        meta_dict[key] = getattr(meta, key, None)
    walls = {
        wall_id: WallPrediction(
            assembled=np.asarray(wo.assembled),
            assembled_at_s=np.asarray(wo.assembled_at_s),
            safety_abort=np.asarray(wo.safety_abort),
            abort_reason=np.asarray(wo.abort_reason),
        )
        for wall_id, wo in (kit_pred.walls or {}).items()
    }
    return Prediction(
        delay_vs_native_s=np.asarray(kit_pred.delay_vs_native_s),
        plan_delay_s=np.asarray(kit_pred.plan_delay_s),
        native_delay_s=np.asarray(kit_pred.native_delay_s),
        delay_part_a_s=np.asarray(kit_pred.delay_part_a_s),
        delay_part_b_s=np.asarray(kit_pred.delay_part_b_s),
        collision=np.asarray(kit_pred.collision),
        min_ttc_s=np.asarray(kit_pred.min_ttc_s),
        ttc_lt3_exposure_s=np.asarray(kit_pred.ttc_lt3_exposure_s),
        hard_brake_s=np.asarray(kit_pred.hard_brake_s),
        native_collision=np.asarray(kit_pred.native_collision),
        native_min_ttc_s=np.asarray(kit_pred.native_min_ttc_s),
        native_ttc_lt3_exposure_s=np.asarray(kit_pred.native_ttc_lt3_exposure_s),
        native_hard_brake_s=np.asarray(kit_pred.native_hard_brake_s),
        trajectories_tracked=dict(kit_pred.trajectories_tracked),
        native_trajectories_tracked=dict(kit_pred.native_trajectories_tracked),
        trajectories_hidden=kit_pred.trajectories_hidden,
        walls=walls,
        meta=meta_dict,
        cell_valid=(None if kit_pred.cell_valid is None else np.asarray(kit_pred.cell_valid)),
        native_cell_valid=(
            None if kit_pred.native_cell_valid is None else np.asarray(kit_pred.native_cell_valid)
        ),
        has_closing_pair=(
            None if kit_pred.has_closing_pair is None else np.asarray(kit_pred.has_closing_pair)
        ),
        native_has_closing_pair=(
            None
            if kit_pred.native_has_closing_pair is None
            else np.asarray(kit_pred.native_has_closing_pair)
        ),
        # contract v2.1 optional per-cell fields (None when the kit
        # prediction does not carry the attribute)
        **{
            name: (
                None
                if getattr(kit_pred, name, None) is None
                else np.asarray(getattr(kit_pred, name))
            )
            for name in ("n_tracked", "n_exited_tracked", "n_free_road_assumed",
                         "untracked_arrivals_expected")
        },
    )


class KitWorldModel:
    """Planner ``WorldModel`` view of a WD-01 kit ``Predictor``.

    Every plan is kit-``validate_plan``-checked against the call's own
    controller_state before reaching the predictor (strict boundary), and
    every call is timed so the dry run can split predictor time from
    planner overhead. The predictor is asked for contract v1.1 outputs."""

    def __init__(
        self,
        predictor,
        *,
        contract_version: str = CONTRACT_VERSION_V1_1,
        strict: bool = True,
        search_paths: Sequence[str] = (PREDICT_KIT_ROOT,),
        clock: Optional[Callable[[], float]] = None,
    ):
        if contract_version not in ("wd01_predict/1", "wd01_predict/1.1"):
            raise ValueError(f"unsupported contract_version {contract_version!r}")
        self.contract, self.validate = load_predict_kit(search_paths)
        self.predictor = predictor
        self.contract_version = contract_version
        self.strict = strict
        self.clock = clock or time.monotonic
        self.calls: List[Dict[str, Any]] = []
        self.rejections: List[str] = []

    # ------------------------------------------------------------------ api
    def predict_many(
        self,
        state_samples: Sequence[Any],
        plans: Sequence[PlanCandidate],
        controller_state: Optional[Mapping[str, Any]] = None,
        *,
        horizon_s: float = 600.0,
        trajectory_s: Optional[float] = None,
    ) -> "list[Prediction]":
        kit = self.contract
        kit_state = controller_state_out(controller_state, kit)
        kit_plans = [plan_out(plan, kit) for plan in plans]
        if self.strict:
            for kp in kit_plans:
                try:
                    self.validate.validate_plan(kp, kit_state)
                except Exception as exc:  # noqa: BLE001 - recorded, re-raised
                    self.rejections.append(f"validate_plan: {exc}")
                    raise
        started = self.clock()
        kit_preds = self.predictor.predict_many(
            state_samples, kit_plans, kit_state, horizon_s=horizon_s, trajectory_s=trajectory_s
        )
        predictor_s = self.clock() - started
        self.calls.append({"n_plans": len(kit_plans), "predictor_s": predictor_s})
        return [prediction_in(kp, kit) for kp in kit_preds]

    def predict(self, state_samples, plan, controller_state=None, **kw) -> Prediction:
        return self.predict_many(state_samples, [plan], controller_state, **kw)[0]


# ------------------------------------------------------------ real states
_WITH_FRAMES = None


def _state_sample_with_frames(contract):
    """(Lazy) kit StateSample plus the planner-side frame freshness
    extension (PL01-E4 interface request to WD-01: frames[vehicle] =
    last-frame time; the contract schema does not carry it yet)."""
    global _WITH_FRAMES
    if _WITH_FRAMES is None:
        import dataclasses as dc

        @dataclasses.dataclass(frozen=True)
        class _WithFrames(contract.StateSample):
            frames: Dict[str, float] = dc.field(default_factory=dict)

        _WITH_FRAMES = _WithFrames
    return _WITH_FRAMES


@dataclass
class RealStateBundle:
    """Decision-time inputs built from one p2C root at one t0."""

    root: str
    t0_s: float
    state_samples: List[Any]  # kit StateSample (K = 1)
    controller_state: Dict[str, Dict[str, Any]]
    frames: Dict[str, float]  # {cav: last-frame time} (fresh at t0)
    cav_ids: Tuple[str, ...]  # physical-lane CAVs (wall-eligible)
    observed_ids: Tuple[str, ...]  # every observed vehicle at t0
    merge_observed: bool
    coverage: float
    n_junction_vehicles: int
    rsu_report_missing: bool = False
    merge_observed_reason: Optional[str] = None
    rsu_staleness_bound_s: Optional[float] = None
    rsu_report_age_s: Optional[Mapping[str, float]] = None


def _sample_meta(
    contract,
    sample_id: str,
    t0: float,
    merge_observed: bool,
    coverage: float,
    merge_observed_reason: Optional[str] = None,
):
    """SampleMeta that ALSO supports meta["merge_observed"] subscripting
    (the planner's rules.py reads it that way; the kit reads attributes).
    A frozen-dataclass subclass adding __getitem__ satisfies both."""

    class _Meta(contract.SampleMeta):
        def __getitem__(self, key):
            return getattr(self, key)

    meta = _Meta(
        sample_id=sample_id,
        time_s=float(t0),
        merge_observed=bool(merge_observed),
        coverage=float(coverage),
    )
    if merge_observed_reason:
        object.__setattr__(meta, "merge_observed_reason", merge_observed_reason)
    return meta


def _state_sample_with_frames(contract):
    """Frozen kit StateSample subclass carrying the planner-side frame
    freshness extension (PL01-E4 interface request to WD-01: frames[vehicle]
    = last-frame time; the contract schema does not carry it yet)."""
    import dataclasses

    @dataclasses.dataclass(frozen=True)
    class _WithFrames(contract.StateSample):
        frames: Dict[str, float] = dataclasses.field(default_factory=dict)

    return _WithFrames


def build_real_state(
    root: str,
    t0: float,
    *,
    loader,
    horizon_s: float = 60.0,
    sample_id: Optional[str] = None,
    rsu_report_missing: bool = False,
    merge_observed: Optional[bool] = True,
    merge_observed_reason: Optional[str] = None,
    rsu_staleness_bound_s: Optional[float] = None,
    rsu_report_age_s: Optional[Mapping[str, float]] = None,
) -> RealStateBundle:
    """Real decision-time inputs from the p2C dataset (read-only).

    Observed vehicles = the t0 state rows (positions/lanes/speeds are REAL).
    CAVs = ``is_cav`` vehicles on WM-30 physical lanes; all start ``damping``
    unlocked. merge_observed follows contract §8.15 + ADDENDUM
    (document sha256
    2619390e95ae2d37fcdffd6dc1fefeb5141ef9d2dc1cfb2676b9c02559bd83a8,
    read-only): True IFF the whole merge region is within sensor
    coverage at t0 REGARDLESS of occupancy -- the p2C full-state
    recording is ideal sensing with an RSU, so it is always True;
    the optional rsu_report_missing flag models a missing/stale RSU
    report at t0 (then False and the planner abstains). It is NEVER
    derived from vehicle presence. coverage = share of t0 vehicles
    on physical lanes. K = 1 (the loader supplies one realised
    state per root). Frame freshness: every observed physical-lane
    CAV gets frames[vid] = t0 (fresh at decision time); junction
    vehicles get no frame (excluded from wall proposals,
    §3.2-consistent).
    """
    contract, validate = load_predict_kit([PREDICT_KIT_ROOT])
    window = loader.windows(root, t0, 0.5)
    state = window["state"]

    observed: Dict[str, tuple] = {}
    frames: Dict[str, float] = {}
    cav_ids: List[str] = []
    n_physical = 0
    n_junction = 0
    for i, vid_name in enumerate(state["id"]):
        vid_name = str(vid_name)
        lane = int(state["lane"][i])
        if lane < 0:
            n_junction += 1
            continue  # junction vehicles are not observable lane traffic
        n_physical += 1
        observed[vid_name] = (float(state["s"][i]), lane, float(state["v"][i]))
        if int(state["is_cav"][i]) == 1:
            cav_ids.append(vid_name)
            frames[vid_name] = t0

    controller_state = {
        cav: {
            "action": "damping",
            "wall_id": None,
            "wall_phase": None,
            "time_in_phase_s": 0.0,
            "wall_speed_mps": None,
            "locked": False,
        }
        for cav in cav_ids
    }
    arrivals = []
    for k in range(len(window["arrivals"]["t_s"])):
        entrance = int(window["arrivals"]["entrance"][k])
        arrivals.append(
            contract.Arrival(
                t_s=float(window["arrivals"]["t_s"][k]),
                entrance=ENTRANCE_LABELS.get(entrance, "mainline"),
                speed_mps=float(window["arrivals"]["speed_mps"][k]),
                is_cav=bool(window["arrivals"]["is_cav"][k]),
                vehicle_type=str(window["arrivals"]["vehicle_type"][k]),
            )
        )
    # §8.15: occupancy-independent. Ideal sensing with an RSU covering the
    # merge region (the p2C full-state recording) -> True; the optional
    # rsu_report_missing flag models a missing/stale RSU report at t0.
    # §8.15 ADDENDUM (hybrid_state/2): the RSU meta may report
    # merge_observed = None (no RSU inputs) -- that FAILS CLOSED to False
    # and counts toward ABSTAIN; the reason travels with the sample meta.
    if merge_observed is None:
        resolved_merge_observed = False
        if merge_observed_reason is None:
            merge_observed_reason = "no RSU inputs for t0"
    else:
        resolved_merge_observed = bool(merge_observed)
    if rsu_report_missing:
        resolved_merge_observed = False
    coverage = n_physical / (n_physical + n_junction) if (n_physical + n_junction) else 0.0
    sample_cls = _state_sample_with_frames(contract)
    sample = sample_cls(
        hybrid={
            "tracked": observed,
            "section_len_m": 3199.0,
            "t0": t0,
            "meta": {
                "merge_observed": merge_observed,
                "merge_observed_reason": merge_observed_reason,
                "rsu_staleness_bound_s": rsu_staleness_bound_s,
                "rsu_report_age_s": dict(rsu_report_age_s) if rsu_report_age_s else {},
            },
        },
        arrivals=tuple(arrivals),
        meta=_sample_meta(
            contract,
            sample_id or f"{root}@{t0:g}",
            t0,
            resolved_merge_observed,
            coverage,
            merge_observed_reason=merge_observed_reason,
        ),
        frames=dict(frames),
    )
    bundle = RealStateBundle(
        root=root,
        t0_s=t0,
        state_samples=[sample],
        controller_state=controller_state,
        frames=frames,
        cav_ids=tuple(sorted(cav_ids)),
        observed_ids=tuple(sorted(observed)),
        merge_observed=resolved_merge_observed,
        coverage=coverage,
        n_junction_vehicles=n_junction,
        rsu_report_missing=rsu_report_missing,
        merge_observed_reason=merge_observed_reason,
        rsu_staleness_bound_s=rsu_staleness_bound_s,
        rsu_report_age_s=rsu_report_age_s,
    )
    validate.validate_state_samples(bundle.state_samples)
    return bundle
