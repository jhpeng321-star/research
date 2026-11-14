"""Model adapters and the frozen evaluation model names."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from typing import Any

from pathlib import Path

import numpy as np

from .data import WORKSPACE, DataError, initial_frame, recorded_for

MODEL_SPECS = ("keep_speed", "keep_speed_tf", "pure_physics", "pure_physics_tf",
               "a11_closed", "a11_open", "a11_open_fid", "wms",
               "blackbox_id", "blackbox_traj")

_DT_S = 0.5


class ModelUnavailable(RuntimeError):
    """A requested model cannot be loaded without hidden/external artifacts."""


@dataclass(frozen=True)
class RolloutFlags:
    residual_enabled: bool = True
    physical_exits: bool = False
    open_entries: bool = False
    sumo_fidelity: bool = False
    sumo_idm: bool = True
    entries_mode: str = "v1"
    entries_v2: bool = False


FLAGS = {
    "pure_physics": RolloutFlags(residual_enabled=False),
    "a11_closed": RolloutFlags(),
    "a11_open": RolloutFlags(open_entries=True, physical_exits=True),
    "a11_open_fid": RolloutFlags(open_entries=True, physical_exits=True, sumo_fidelity=True),
    "wms": RolloutFlags(),
    "blackbox_id": RolloutFlags(),
    "blackbox_traj": RolloutFlags(),
}


def _keep_speed(state: dict, _controllers: dict, horizon_s: float, _seed: int) -> list[dict]:
    initial = initial_frame(state)
    dt = float(state.get("dt_s", 0.5))
    n_steps = int(round(float(horizon_s) / dt))
    out = []
    for step in range(n_steps + 1):
        t = initial["time_s"] + step * dt
        rows = []
        for item in initial["vehicles"]:
            row = dict(item)
            row["s"] = item["s"] + item["v"] * step * dt
            row["time_s"] = t
            rows.append(row)
        out.append({"time_s": t, "vehicles": rows})
    return out


def _nearest(frame_by_round: dict, index: int) -> dict | None:
    if not frame_by_round:
        return None
    if index in frame_by_round:
        return frame_by_round[index]
    return frame_by_round[min(frame_by_round, key=lambda key: abs(key - index))]


def teacher_force_cavs(frames: list[dict], state: dict, arm: str) -> list[dict]:
    """Replay the recorded-arm CAV trajectory on top of a baseline rollout.

    CAVs are never modelled movers in the WM-S harness: the ``wms`` adapter
    forces each CAV from the recorded arm trajectory.  This helper gives any
    baseline the identical CAV motion so car metrics compare like with like.
    Presence mirrors the ``wms`` teacher: a CAV on a junction lane (``lane < 0``)
    or missing from the recorded step is dropped, and drops are permanent
    (cumulative mask), matching ``wms.model.rollout``.
    """
    if not frames:
        return frames
    init = initial_frame(state)
    cav_ids = [str(v["id"]) for v in init["vehicles"]
               if v.get("present", True) and v.get("is_cav")
               and int(v.get("lane", -1)) >= 0]
    if not cav_ids:
        return frames
    try:
        recorded = recorded_for(state, arm)
    except DataError:
        return frames
    t0 = float(frames[0]["time_s"])
    by_round = {int(round((float(f["time_s"]) - t0) / _DT_S)): f for f in recorded}
    initial = {str(v["id"]): v for v in init["vehicles"]}
    live = {vid: True for vid in cav_ids}
    out = []
    for index, frame in enumerate(frames):
        cav_rows = []
        if index == 0:
            for vid in cav_ids:
                row = initial.get(vid, {})
                cav_rows.append({"id": vid, "s": float(row.get("s", 0.0)),
                                 "lane": int(row.get("lane", -1)),
                                 "v": float(row.get("v", 0.0)),
                                 "present": True, "is_cav": True})
        else:
            src = _nearest(by_round, index)
            rows = {}
            if src is not None:
                for row in src["vehicles"]:
                    if row.get("present", True):
                        rows[str(row["id"])] = row
            for vid in cav_ids:
                row = rows.get(vid)
                if row is None or int(row.get("lane", -1)) < 0:
                    live[vid] = False
                    continue
                if not live[vid]:
                    continue
                cav_rows.append({"id": vid, "s": float(row["s"]),
                                 "lane": int(row["lane"]), "v": float(row["v"]),
                                 "present": True, "is_cav": True})
        vehicles = [dict(row) for row in frame["vehicles"] if not row.get("is_cav")]
        vehicles.extend(cav_rows)
        out.append({"time_s": frame["time_s"], "vehicles": vehicles})
    return out


class TeacherForcedCavs:
    """Baseline wrapper whose CAVs replay the recorded arm (see ``wms``)."""

    def __init__(self, base):
        self.base = base

    def predict_states(self, state: dict, controllers: dict,
                       horizon_s: float, seed: int) -> list[dict]:
        arm = str((controllers or {}).get("arm", "native"))
        return teacher_force_cavs(self.base(state, controllers, horizon_s, seed),
                                  state, arm)


class FrozenA11Model:
    """Adapter for the local, pinned A11 next-state rollout."""

    def __init__(self, flags: RolloutFlags):
        self.flags = flags
        self.bundle = None

    def _bundle(self):
        if self.bundle is None:
            from closedloop.a11_predictor import load_a11_bundle
            from closedloop.fast_hdv import fast_bundle, install_fast_gap_projection

            install_fast_gap_projection()
            self.bundle = fast_bundle(load_a11_bundle(
                path=WORKSPACE / "_ext" / "a11_member" / "member_1" / "member_1.pt",
                climatology=WORKSPACE / "_ext" / "climatology_dir" / "TRAIN_CLIMATOLOGY.json",
            ).bundle)
        return self.bundle

    @staticmethod
    def _path(value):
        path = Path(str(value))
        return path if path.is_absolute() else WORKSPACE / path

    @staticmethod
    def _live_from_state(state):
        """Build the A11 decision-time state from a materialised state record.

        The frozen A11 adapter normally reconstructs the decision row from the
        raw physics file (``state_row`` + ``base_run_dir``).  A pack-materialised
        C1 state has no raw run, but it already carries the SAME 7-column
        kinematics view (``s_down``, lane one-hot, ``v``, ``a``, length) plus the
        CAV flags and vehicle identities, so the live state is assembled from
        the initial frame directly.  The RSU-observable t0 entrance counter is
        taken from the materialised ``counter`` field when present (the pack
        ships it), so ``--entries`` sees the same history for every spec; it
        defaults to zeros only for legacy records that lack it.
        """
        import numpy as np
        from features import constants as C
        from features.idea_a import kinematics_view
        from closedloop.live_state import LiveState

        init = initial_frame(state)
        rows = [row for row in init["vehicles"]
                if row.get("present", True) and int(row.get("lane", -1)) >= 0]
        rows.sort(key=lambda row: str(row["id"]))
        n = max(1, len(rows))
        veh = np.zeros((n, C.VEH_F), np.float16)
        mask = np.zeros(n, bool)
        is_cav = np.zeros(n, bool)
        slot_vid = [None] * n
        for i, row in enumerate(rows):
            lane = int(row.get("lane", -1))
            veh[i, C.S_DOWN] = float(row.get("s", 0.0))
            if 0 <= lane <= 2:
                veh[i, C.LANE_LEFT + lane] = 1.0
            veh[i, C.V_IDX] = float(row.get("v", 0.0))
            veh[i, C.LENGTH_IDX] = float(C.NOMINAL_LENGTH_M)
            veh[i, C.PRESENT_IDX] = 1.0
            mask[i] = True
            is_cav[i] = bool(row.get("is_cav", False))
            veh[i, C.IS_CAV_IDX] = 1.0 if is_cav[i] else 0.0
            slot_vid[i] = str(row["id"])
        counter = state.get("counter")
        counter = (np.asarray(counter, np.float32).reshape(2, 30)
                   if counter is not None else np.zeros((2, 30), np.float32))
        return LiveState(
            time_s=float(init["time_s"]),
            veh_kin=np.asarray(kinematics_view(veh), np.float32),
            veh_mask=mask, is_cav=is_cav,
            slot_id=np.arange(n, dtype=np.int64),
            counter=counter,
            slot_vid=slot_vid,
            meta={"source": "wms_eval_materialized"},
        )

    def predict_states(self, state, cav_controllers, horizon_s, seed):
        from closedloop.a11_predictor import NONE_CODE, PLANNER_ACTION_CODES
        from closedloop.live_state import DepartureHistory, live_state_from_row
        from closedloop.a11_predictor import batched_rollout

        t0_s = float(state.get("t0_s", state.get("time_s", 0.0)))
        if "state_row" in state:
            history = DepartureHistory(self._path(state["base_run_dir"]), t0_s)
            live = live_state_from_row(state["state_row"], history)
        else:
            live = self._live_from_state(state)
        arm = str((cav_controllers or {}).get("arm", "native"))
        if arm not in ("native", "pi"):
            raise ModelUnavailable(f"unsupported CAV controller arm {arm!r}")
        codes = np.full((1, len(live.slot_vid)), NONE_CODE, dtype=np.int64)
        action = PLANNER_ACTION_CODES["native" if arm == "native" else "pi-external-u"]
        active = np.asarray(live.is_cav, bool) & live.rollout_mask
        codes[0, active] = action
        roll = batched_rollout(
            self._bundle(), live, codes, n_steps=int(round(float(horizon_s) / 0.5)),
            t0_s=t0_s, seed=int(seed), residual_enabled=self.flags.residual_enabled,
            physical_exits=self.flags.physical_exits, open_entries=self.flags.open_entries,
            sumo_fidelity=self.flags.sumo_fidelity, sumo_idm=self.flags.sumo_idm,
        )
        result = []
        for j in range(roll.pos.shape[0]):
            t = t0_s + j * 0.5
            rows = []
            for k, vid in enumerate(live.slot_vid):
                if vid is None:
                    continue
                if j == 0:
                    present = bool(live.veh_mask[k])
                    s = float(live.veh_kin[k, 0])
                    lane_onehot = live.veh_kin[k, 1:4]
                    lane = int(np.argmax(lane_onehot)) if lane_onehot.sum() else -1
                    v = float(live.veh_kin[k, 4])
                else:
                    present = bool(roll.msk[j, 0, k])
                    s, lane, v = (float(roll.pos[j, 0, k]), int(roll.lns[j, 0, k]),
                                  float(roll.spd[j, 0, k]))
                rows.append({"id": str(vid), "s": s, "lane": lane, "v": v,
                             "present": present, "is_cav": bool(live.is_cav[k])})
            result.append({"time_s": t, "vehicles": rows})
        return result


class PluginModel:
    def __init__(self, model: Any, flags: RolloutFlags):
        self.model = model
        self.flags = flags

    def predict_states(self, state, cav_controllers, horizon_s, seed):
        configure = getattr(self.model, "configure_rollout", None)
        if configure is not None:
            configure(
                residual_enabled=self.flags.residual_enabled,
                physical_exits=self.flags.physical_exits,
                open_entries=self.flags.open_entries,
                sumo_fidelity=self.flags.sumo_fidelity,
                sumo_idm=self.flags.sumo_idm,
                entries_mode=self.flags.entries_mode,
                entries_v2=self.flags.entries_v2,
            )
        fn = getattr(self.model, "predict_states", None)
        if fn is None and callable(self.model):
            fn = self.model
        if fn is None:
            raise ModelUnavailable("plugin has no predict_states(state, controllers, horizon_s, seed)")
        return fn(state, cav_controllers, horizon_s, seed)


def _load_plugin(spec: str, flags: RolloutFlags) -> PluginModel:
    module_name, sep, attr = spec.partition(":")
    if not sep:
        raise ModelUnavailable("plugin must be module:object")
    try:
        obj = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise ModelUnavailable(f"cannot load plugin {spec!r}: {exc}") from exc
    if isinstance(obj, type):
        obj = obj()
    return PluginModel(obj, flags)


def flags_for(spec: str, *, entries: bool = False,
              entries_mode: str | None = None,
              entries_v2: bool = False) -> RolloutFlags:
    """Resolve the rollout flags for ``spec`` with the optional entries override.

    ``--entries`` turns observed-counter vehicle entries on for any spec that
    supports them (``wms`` and the ``pure_physics``/A11 baselines).  Default OFF
    keeps every spec byte-identical to its frozen flag set.

    ``entries_mode`` is the forward hook for the entry2 branch: ``"v1"`` is the
    observed-counter process merged here; ``"v2"`` is reserved for the
    history/rate-trend process on ``wms/entry2``.  Requesting a mode that is not
    installed fails closed at rollout time, never silently.
    """
    if spec not in FLAGS:
        raise ModelUnavailable(f"model {spec!r} has no rollout flags")
    flags = FLAGS[spec]
    if entries_v2:
        if not flags.open_entries:
            flags = replace(flags, open_entries=True)
        return replace(flags, entries_v2=True)
    mode = str(entries_mode) if entries_mode else ("v1" if entries else flags.entries_mode)
    if entries or mode == "v2":
        flags = replace(flags, open_entries=True, entries_mode=mode)
    return flags


def load_model(spec: str, plugin: str | None = None, checkpoint: str | None = None,
               *, entries: bool = False, entries_mode: str | None = None,
               entries_v2: bool = False, physics: dict | None = None):
    if spec not in MODEL_SPECS:
        raise ModelUnavailable(f"unknown model {spec!r}; choose from {', '.join(MODEL_SPECS)}")
    flags = flags_for(spec, entries=entries, entries_mode=entries_mode,
                      entries_v2=entries_v2) if spec in FLAGS else FLAGS.get(spec)
    if spec == "keep_speed":
        return _keep_speed, RolloutFlags(residual_enabled=False)
    if spec == "keep_speed_tf":
        return TeacherForcedCavs(_keep_speed).predict_states, RolloutFlags(residual_enabled=False)
    if spec == "pure_physics_tf":
        return TeacherForcedCavs(FrozenA11Model(FLAGS["pure_physics"]).predict_states).predict_states, \
            FLAGS["pure_physics"]
    if spec == "wms":
        if not checkpoint:
            raise ModelUnavailable(
                "model 'wms' requires --ckpt PATH (a wm_s/1 checkpoint from the wms/p2 trainer)")
        from .wms_plugin import WmsCheckpointModel

        return WmsCheckpointModel(checkpoint, flags, physics=physics).predict_states, flags
    if spec in ("blackbox_id", "blackbox_traj"):
        if not checkpoint:
            raise ModelUnavailable(f"model {spec!r} requires --ckpt PATH")
        from .blackbox import load_blackbox

        return load_blackbox(spec, checkpoint, flags).predict_states, flags
    if not plugin:
        model = FrozenA11Model(flags)
        return model.predict_states, flags
    return _load_plugin(plugin, flags).predict_states, flags
