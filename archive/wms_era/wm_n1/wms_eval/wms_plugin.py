"""WM-S ``wms`` model specification for the offline evaluation CLI.

The specification loads a checkpoint written by the WM-S trainer on
``wms/p2`` (``output/wd01/wm_n1/wms/train.py``), whose format is

    {"format": "wm_s/1", "model": <state_dict>, "config": <train config dict>, ...}

The default evaluation mode teacher-forces recorded CAV trajectories. The
opt-in closed-loop mode executes the selected CAV controller from each
predicted state, at one-second control boundaries, while the WM-S model
continues to advance the HDVs at its 0.5 s physics interval.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence
import sys

import numpy as np

from .data import WORKSPACE, DataError, initial_frame, recorded_for

DT_S = 0.5
CHECKPOINT_FORMAT = "wm_s/1"
DEFAULT_LENGTH_M = 5.0

#: Model kinds share the ``wm_s/1`` checkpoint container but differ in the
#: network class and the CAV-rollout contract:
#: ``structured``  physics+residual HDV, CAVs teacher-forced from the recorded arm;
#: ``blackbox_id`` physics+residual HDV with a controller-id embedding, CAVs
#:                 predicted by the network (no controller code is executed);
#: ``blackbox_traj`` learned absolute HDV (no physics base), CAVs teacher-forced
#:                 like ``structured`` so only the HDV mechanism differs.
MODEL_KINDS = ("structured", "blackbox_id", "blackbox_traj")


class WmsCheckpointError(RuntimeError):
    """The WM-S checkpoint or its inputs cannot be scored."""


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise WmsCheckpointError("torch is required to score a WM-S checkpoint") from exc
    return torch


def _checkpoint_path(value: str | Path) -> Path:
    path = Path(str(value))
    path = path if path.is_absolute() else WORKSPACE / path
    try:
        path.resolve().relative_to(WORKSPACE)
    except ValueError as exc:
        raise WmsCheckpointError(f"checkpoint must be inside workspace: {path}") from exc
    if not path.is_file():
        raise WmsCheckpointError(f"checkpoint not found: {path}")
    return path


def _nearest(frame_by_round: Mapping[int, dict], index: int) -> dict | None:
    if not frame_by_round:
        return None
    if index in frame_by_round:
        return frame_by_round[index]
    nearest = min(frame_by_round, key=lambda key: abs(key - index))
    return frame_by_round[nearest]


def prior_history_features(kin_hist, mask_hist, target_hist, present_now,
                           *, history_len: int = 10):
    """Mirror ``wms.train._prior_history``: t0 features ``[K,H,F]`` + mask ``[K,H]``.

    The pre-t0 window is right-aligned to ``history_len`` with zero front-pad,
    the same continuity rule (backward ``target_mask`` product) the trainer uses
    to learn the DriverEncoder.  Only ticks ``<= t0`` are read.
    """
    import torch
    from wms.model import history_step_features

    kin = torch.as_tensor(np.asarray(kin_hist, dtype=np.float32))
    present = torch.as_tensor(np.asarray(mask_hist).astype(bool))
    if kin.ndim != 3 or kin.shape[-1] != 7:
        raise WmsCheckpointError("driver history kin must be [T,K,7]")
    t, k = kin.shape[:2]
    tgt = np.asarray(target_hist, dtype=bool) if target_hist is not None else None
    if tgt is None or tgt.size == 0:
        tm = torch.zeros(max(0, t - 1), k, dtype=torch.bool)
    else:
        tm = torch.as_tensor(tgt)
    same = torch.ones(t, k, dtype=torch.bool)
    for j in range(t - 2, -1, -1):
        same[j] = same[j + 1] & tm[j]
    valid = same & present
    valid[-1] = torch.as_tensor(np.asarray(present_now).astype(bool))
    lane = kin[..., 1:4].argmax(-1)
    s_ = kin[..., 0]
    gap = torch.full((t, k), 250.0)
    for ti in range(t):
        ds = s_[ti].unsqueeze(0) - s_[ti].unsqueeze(1)
        ok = valid[ti].unsqueeze(0) & (lane[ti].unsqueeze(0) == lane[ti].unsqueeze(1)) \
            & (ds > 0)
        g = torch.where(ok, ds - kin[ti, :, 6].unsqueeze(0),
                        torch.full_like(ds, 250.0))
        gap[ti] = g.min(-1).values.clamp(0.3, 250.0)
    feat = history_step_features(kin, gap)                    # [T,K,F]
    want = max(1, int(history_len))
    out = torch.zeros(k, want, feat.shape[-1], dtype=feat.dtype)
    out_mask = torch.zeros(k, want, dtype=torch.bool)
    use = min(t, want)
    out[:, want - use:] = feat[-use:].transpose(0, 1)
    out_mask[:, want - use:] = valid[-use:].transpose(0, 1)
    return out, out_mask


class WmsCheckpointModel:
    """Load a ``wm_s/1`` checkpoint and roll out next states for the harness."""

    def __init__(self, checkpoint: str | Path, flags: Any = None,
                 physics: Mapping[str, Any] | None = None,
                 model_kind: str = "structured") -> None:
        if model_kind not in MODEL_KINDS:
            raise WmsCheckpointError(
                f"unknown model kind {model_kind!r}; choose from {', '.join(MODEL_KINDS)}")
        self.checkpoint = _checkpoint_path(checkpoint)
        self.flags = flags
        self.model_kind = model_kind
        self.model = None
        self.config: dict = {}
        self.projection = None
        self._physics = dict(physics) if physics else None
        self._torch = _torch()

    def configure_rollout(self, **flags: Any) -> None:
        """The harness advertises rollout flags; the network keeps its own config."""
        self.rollout_flags = dict(flags)

    def _load(self):
        torch = self._torch
        # Honour the harness per-worker thread cap (c1v2 sets WMS_EVAL_THREADS
        # from --workers so the whole worker fleet stays within four threads);
        # fall back to the frozen four threads for standalone callers.
        torch.set_num_threads(int(os.environ.get("WMS_EVAL_THREADS", "4")))
        try:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # older torch without ``weights_only``
            payload = torch.load(self.checkpoint, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
            raise WmsCheckpointError(
                f"unsupported checkpoint format: {payload.get('format') if isinstance(payload, dict) else type(payload).__name__}"
            )
        if "model" not in payload:
            raise WmsCheckpointError("checkpoint has no model state_dict")
        try:
            from wms.model import PhysicsConfig
        except ImportError as exc:
            raise WmsCheckpointError(
                "WM-S trainer package 'wms' is missing; merge branch wms/p2 into wms/eval"
            ) from exc
        self.config = dict(payload.get("config") or {})
        history_s = self.config.get("history_s")
        history_stride_s = self.config.get("history_stride_s", 1.0)
        if history_s is not None and self.model_kind != "structured":
            raise WmsCheckpointError("long history is supported only for structured checkpoints")
        # The residual net was trained against the trainer's physics switches
        # (open road + SUMO vType IDM); replay the same base at scoring time.
        # Prefer an explicit receipt override, an embedded config block, then a
        # sibling ``physics.json`` captured from the run's TRAIN_RECEIPT.
        raw_physics = self._physics or self.config.get("physics")
        if raw_physics is None:
            sibling = self.checkpoint.with_name("physics.json")
            if sibling.is_file():
                raw_physics = json.loads(sibling.read_text(encoding="utf-8"))
        if isinstance(raw_physics, Mapping):
            allowed = set(PhysicsConfig.__dataclass_fields__)
            raw_physics = {key: value for key, value in raw_physics.items()
                           if key in allowed}
        physics = PhysicsConfig(**raw_physics) if raw_physics else PhysicsConfig()
        history_features = int(self.config.get("history_features", 6))
        weights = payload["model"]
        # Checkpoints trained before the action-timeline merge carry no
        # command projection; they never receive an action command here.
        optional_prefixes = ("driver_encoder.", "command_proj.")
        if self.model_kind == "blackbox_traj":
            from wms.blackbox_traj import BlackBoxTrajModel

            model = BlackBoxTrajModel(
                physics=physics, history_features=history_features,
                action_conditioned="controller_embed.weight" in weights)
        elif self.model_kind == "blackbox_id":
            from wms.model import WmSModel

            model = WmSModel(
                physics=physics,
                history_features=history_features,
                black_box=True,
                n_controllers=int(self.config.get("n_controllers", 6)),
                controller_embed_dim=int(self.config.get("controller_embed_dim", 16)),
            )
        else:
            from wms.model import WmSModel

            # Run-2 bounded correction (B) + feasibility projection (A): the
            # trainer applies these INSIDE the training rollout (see wms/train.py
            # ``residual_cap_mps2``/``lane_logit_cap``/``projection_enabled``), so a
            # run-2 checkpoint must be scored with the same switches or it runs the
            # residual net outside its training distribution.  Run-1/C1 checkpoints
            # leave every field absent and stay byte-identical.
            model = WmSModel(
                physics=physics,
                history_features=history_features,
                # WM-S2 driver-parameter recovery requires the matching rollout mode;
                # a residual checkpoint keeps the byte-identical default.
                rollout_mode=str(self.config.get("rollout_mode", "residual")),
                lane_head=str(self.config.get("lane_head", "off")),
                residual_cap_mps2=self.config.get("residual_cap_mps2"),
                residual_cap_neg_mps2=self.config.get("residual_cap_neg_mps2"),
                residual_soft_bound_mps2=self.config.get("residual_soft_bound_mps2"),
                residual_soft_bound_neg_mps2=self.config.get("residual_soft_bound_neg_mps2"),
                lane_logit_cap=self.config.get("lane_logit_cap"),
                history_s=history_s,
                history_stride_s=history_stride_s,
            )
        self.projection = None
        if self.config.get("projection_enabled"):
            try:
                from wms.stability import FeasibilityProjection, ProjectionConfig
            except ImportError as exc:
                raise WmsCheckpointError(
                    "checkpoint needs wms.stability for projection_enabled; "
                    "merge wms/run2b") from exc
            min_gap = float(self.config.get("projection_min_gap_m", 2.0))
            self.projection = FeasibilityProjection(ProjectionConfig(min_gap_m=min_gap))
        result = model.load_state_dict(weights, strict=False)
        # A residual/black-box checkpoint predates the WM-S2 driver encoder, so
        # its ``driver_encoder.*`` weights are legitimately absent and unused.
        # driver_params/hybrid/lane-head checkpoints DO use the encoder.
        uses_encoder = (str(self.config.get("rollout_mode", "residual")) != "residual"
                        or str(self.config.get("lane_head", "off")) != "off"
                        or history_s is not None)
        if uses_encoder:
            optional_prefixes = tuple(p for p in optional_prefixes
                                      if p != "driver_encoder.")
        missing = [key for key in result.missing_keys
                   if key not in ("history",) and not key.startswith(optional_prefixes)]
        if missing:
            raise WmsCheckpointError(f"checkpoint is missing network weights: {missing[:5]}")
        model.eval()
        self.model = model
        return model

    def _inputs(self, state: dict, arm: str):
        init = initial_frame(state)
        t0_s = float(init["time_s"])
        rows = [row for row in init["vehicles"] if row.get("present", True)]
        rows = [row for row in rows if int(row.get("lane", -1)) >= 0]
        rows.sort(key=lambda row: str(row["id"]))
        if not rows:
            raise WmsCheckpointError("state has no vehicles on the modelled road")
        ids = [str(row["id"]) for row in rows]
        kin = np.zeros((len(rows), 7), dtype=np.float32)
        is_cav = np.zeros(len(rows), dtype=bool)
        for index, row in enumerate(rows):
            lane = int(row.get("lane", -1))
            kin[index, 0] = float(row.get("s", 0.0))
            if 0 <= lane < 3:
                kin[index, 1 + lane] = 1.0
            kin[index, 4] = float(row.get("v", 0.0))
            kin[index, 6] = DEFAULT_LENGTH_M
            is_cav[index] = bool(row.get("is_cav", False))
        return t0_s, ids, kin, is_cav

    def _driver_history(self, state: dict, ids: Sequence[str]):
        """Rebuild the pre-t0 DriverEncoder input for the modelled t0 slots.

        Two materialised schemas are accepted:

        * id-keyed (``wms/evalshift``): ``history.ids`` + raw pre-t0 ``kin`` /
          ``mask`` / ``target_mask`` on the pack columns of the t0 vehicles; the
          columns are re-ordered to the sorted rollout slot order and featurised
          exactly as the trainer did.  Used for every checkpoint kind.
        * slot-indexed (``wms/lanehead``): ``history.kin``/``mask`` ``[T,K,7]``
          addressed by ``slotN`` vehicle ids; required (fail closed) by a
          history lane-head or ``hybrid`` checkpoint.

        Returns ``(None, None)`` otherwise (residual / plain ``driver_params``
        checkpoints then use the encoder's empty-history defaults, as before).
        Long-history checkpoints always require duration-declared native history
        and use the shared trainer/evaluator featuriser for either schema.
        """
        hist = state.get("history")
        if self._long_history_config()[0] is not None and not isinstance(hist, Mapping):
            raise WmsCheckpointError("long-history checkpoint requires pre-t0 history")
        if isinstance(hist, Mapping) and hist.get("ids"):
            return self._driver_history_by_id(hist, ids)
        return self._driver_history_by_slot(state, ids)

    def _long_history_config(self):
        return (getattr(self.model, "history_s", self.config.get("history_s")),
                getattr(self.model, "history_stride_s",
                        self.config.get("history_stride_s", 1.0)))

    def _validate_long_history(self, hist: Mapping) -> None:
        """Fail closed on legacy, downsampled, or insufficient pre-t0 states."""
        history_s, stride_s = self._long_history_config()
        from wms.history import history_spec

        try:
            history_spec(history_s, stride_s, DT_S)
            kin = np.asarray(hist["kin"], dtype=np.float32)
            mask = np.asarray(hist["mask"], dtype=bool)
            target = np.asarray(hist["target_mask"], dtype=bool)
            declared_s = float(hist["history_s"])
            declared_stride = float(hist["history_stride_s"])
            dt_s = float(hist["dt_s"])
            available_s = float(hist["available_span_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise WmsCheckpointError(
                "long history requires native tensors and duration metadata") from exc
        if (kin.ndim != 3 or kin.shape[-1] != 7 or kin.shape[0] < 1
                or mask.shape != kin.shape[:2]):
            raise WmsCheckpointError("long history input must be [T,K,7]/[T,K], T >= 1")
        t, k = kin.shape[:2]
        if target.shape != (t - 1, k) and not (t == 1 and target.size == 0):
            raise WmsCheckpointError("long history target_mask must cover all native transitions")
        if (not np.isfinite([declared_s, declared_stride, dt_s, available_s]).all()
                or not np.isclose(dt_s, DT_S, rtol=0.0, atol=1e-6)
                or not np.isclose(declared_stride, float(stride_s), rtol=0.0, atol=1e-6)):
            raise WmsCheckpointError("long history requires dt_s=0.5 and matching history_stride_s")
        if not np.isclose(available_s, (t - 1) * DT_S, rtol=0.0, atol=1e-6):
            raise WmsCheckpointError("long history available_span_s does not match native ticks")
        if declared_s + 1e-6 < float(history_s):
            raise WmsCheckpointError("state history_s is shorter than checkpoint history_s")
        native_steps, _, _ = history_spec(history_s, stride_s, DT_S)
        if t < native_steps and hist.get("run_start_limited") is not True:
            raise WmsCheckpointError(
                "short long history requires explicit run_start_limited=true")

    def _long_history_features(self, kin, mask, target, present_now):
        from wms.history import long_history_features

        history_s, stride_s = self._long_history_config()
        torch = self._torch
        target = np.asarray(target, dtype=bool).reshape(max(0, kin.shape[0] - 1),
                                                       kin.shape[1])
        return long_history_features(
            torch.as_tensor(kin, dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.bool),
            torch.as_tensor(target, dtype=torch.bool),
            torch.as_tensor(present_now, dtype=torch.bool),
            history_s=history_s, history_stride_s=stride_s, dt_s=DT_S)

    def _driver_history_by_id(self, hist: Mapping, ids: Sequence[str]):
        import torch

        long_history = self._long_history_config()[0] is not None
        if long_history:
            self._validate_long_history(hist)
        src_ids = [str(v) for v in hist["ids"]]
        col_of = {vid: i for i, vid in enumerate(src_ids)}
        kin = np.asarray(hist["kin"], dtype=np.float32)          # [T,S,7]
        mask = np.asarray(hist["mask"]).astype(bool)             # [T,S]
        target = np.asarray(hist.get("target_mask", []), dtype=bool)
        if kin.ndim != 3:
            raise WmsCheckpointError("state.history.kin must be [T,S,7]")
        t, s = kin.shape[:2]
        if long_history and (len(src_ids) != s or len(col_of) != s):
            raise WmsCheckpointError("long history ids must be unique and match its columns")
        k = len(ids)
        if long_history:
            # Featurise before selecting the t0 roster so an observed leader
            # which left before t0 remains available as a historical cue.
            all_features, all_masks = self._long_history_features(
                kin, mask, target, mask[-1])
            model = self.model or self._load()
            want = model.driver_history_len if hasattr(model, "driver_history_len") \
                else all_features.shape[1]
            feat = torch.zeros(k, want, all_features.shape[-1])
            hmask = torch.zeros(k, want, dtype=torch.bool)
            for j, vid in enumerate(ids):
                col = col_of.get(str(vid))
                if col is not None:
                    feat[j], hmask[j] = all_features[col], all_masks[col]
            return feat[None], hmask[None]
        kin_hist = np.zeros((t, k, 7), dtype=np.float32)
        mask_hist = np.zeros((t, k), dtype=bool)
        target_hist = np.zeros((max(0, t - 1), k), dtype=bool)
        present_now = np.zeros(k, dtype=bool)
        for j, vid in enumerate(ids):
            col = col_of.get(str(vid))
            if col is None or col >= s:
                continue
            kin_hist[:, j] = kin[:, col]
            mask_hist[:, j] = mask[:, col]
            if t > 1:
                target_hist[:, j] = target[:t - 1, col]
            present_now[j] = True
        model = self.model or self._load()
        history_len = int(getattr(model, "history_len", 10))
        feat, hmask = prior_history_features(kin_hist, mask_hist, target_hist,
                                             present_now, history_len=history_len)
        return torch.as_tensor(feat[None]), torch.as_tensor(hmask[None])

    def _driver_history_by_slot(self, state: dict, ids: Sequence[str]):
        """Slot-aligned causal history for history-conditioned checkpoints.

        Required by a ``--lane-head history`` checkpoint and by the ``hybrid``
        rollout (whose driver encoder feeds both the physics and the residual
        net); a plain ``driver_params`` checkpoint without the lane head
        reconstructs the encoder's empty-history defaults, as before.
        """
        long_history = self._long_history_config()[0] is not None
        conditioned = (self.config.get("lane_head", "off") == "history"
                       or self.config.get("rollout_mode", "residual") == "hybrid"
                       or long_history)
        if not conditioned:
            return None, None
        raw = state.get("history")
        if not isinstance(raw, Mapping) or "kin" not in raw or "mask" not in raw:
            raise WmsCheckpointError(
                "history-conditioned checkpoint requires pre-t0 history.kin/mask")
        if long_history:
            self._validate_long_history(raw)
        kin = np.asarray(raw["kin"], dtype=np.float32)
        mask = np.asarray(raw["mask"], dtype=bool)
        if kin.ndim != 3 or kin.shape[-1] != 7 or mask.shape != kin.shape[:2]:
            raise WmsCheckpointError("history lane-head input must be [T,K,7]/[T,K]")
        try:
            slots = [int(str(vid).removeprefix("slot")) for vid in ids]
        except ValueError as exc:
            raise WmsCheckpointError(
                "history lane-head input requires slot-indexed vehicle ids") from exc
        if not slots or max(slots) >= kin.shape[1] or (long_history and min(slots) < 0):
            raise WmsCheckpointError("history lane-head slots do not align with vehicles")
        if long_history:
            target = np.asarray(raw["target_mask"], dtype=bool).reshape(
                max(0, kin.shape[0] - 1), kin.shape[1])
            features, feature_mask = self._long_history_features(
                kin, mask, target, mask[-1])
            return features[slots][None], feature_mask[slots][None]
        kin = kin[:, slots]
        mask = mask[:, slots]
        from wms_eval.driver import history_features_from_kin
        features, feature_mask = history_features_from_kin(
            kin, mask, history_len=self.model.history_len)
        return (self._torch.as_tensor(features[None], dtype=self._torch.float32),
                self._torch.as_tensor(feature_mask[None], dtype=self._torch.bool))

    def _teacher(self, state: dict, t0_s: float, ids: Sequence[str],
                 kin: np.ndarray, is_cav: np.ndarray, n_steps: int, arm: str):
        """Recorded-arm CAV teacher + presence sequence (offline exogenous labels).

        A vehicle on a junction internal lane (``lane < 0``) is carried with its
        last known state and kept present instead of being permanently dropped;
        only a genuinely absent recorded row (true exit) or the end of the
        recorded identity ends the slot.  The junction steps are returned
        separately so the rollout can carry the slot through them.
        """
        from wms.presence import junction_carry_presence

        torch = self._torch
        recorded_present = np.zeros((n_steps + 1, len(ids)), dtype=bool)
        junction = np.zeros((n_steps + 1, len(ids)), dtype=bool)
        teacher = np.zeros((n_steps + 1, len(ids), 4), dtype=np.float32)
        teacher[:, :, 0] = kin[:, 0]
        teacher[:, :, 1] = kin[:, 1:4].argmax(-1)
        teacher[:, :, 2] = kin[:, 4]
        try:
            recorded = recorded_for(state, arm)
        except DataError:
            recorded = None
        if not recorded:
            return None, None, None
        by_round: dict[int, dict] = {}
        for frame in recorded:
            by_round[int(round((float(frame["time_s"]) - t0_s) / DT_S))] = frame
        previous = {vid: (float(kin[k, 0]), int(kin[k, 1:4].argmax(-1)), float(kin[k, 4]))
                    for k, vid in enumerate(ids)}
        speeds: dict[str, tuple[int, float]] = {}
        for index in range(n_steps + 1):
            frame = _nearest(by_round, index)
            rows = {}
            if frame is not None:
                for row in frame["vehicles"]:
                    if row.get("present", True):
                        rows[str(row["id"])] = row
            for k, vid in enumerate(ids):
                row = rows.get(vid)
                if row is None:
                    # genuinely absent from the recorded frame: a real exit.
                    recorded_present[index, k] = index == 0
                    s, lane, v = previous[vid]
                elif int(row.get("lane", -1)) < 0:
                    # junction internal lane: physical transit, carried forward.
                    junction[index, k] = True
                    s, lane, v = previous[vid]
                else:
                    lane = int(row["lane"])
                    v = float(row["v"])
                    s = float(row["s"])
                    recorded_present[index, k] = True
                    previous[vid] = (s, lane, v)
                    speeds[vid] = (index, v)
                teacher[index, k] = (s, lane, v, 0.0)
            if index > 0:
                for k, vid in enumerate(ids):
                    hit = speeds.get(vid)
                    if hit is not None and hit[0] == index:
                        prior = teacher[index - 1, k, 2]
                        teacher[index, k, 3] = (teacher[index, k, 2] - prior) / DT_S
        # t0 is a modelled-lane snapshot by construction.
        recorded_present[0] = True
        junction[0] = False
        present, junction_step = junction_carry_presence(recorded_present, junction)
        teacher_t = torch.as_tensor(teacher[None], dtype=torch.float32)
        present_t = torch.as_tensor(present[None])
        junction_t = torch.as_tensor(junction_step[None])
        return teacher_t, present_t, junction_t

    def _entries_enabled(self) -> bool:
        flags = getattr(self, "rollout_flags", None) or {}
        if flags.get("open_entries"):
            return True
        return bool(getattr(self.flags, "open_entries", False))

    def _closed_loop_enabled(self, controllers: Mapping[str, Any]) -> bool:
        mode = str(controllers.get("cav_mode", "teacher"))
        if mode not in ("teacher", "closed_loop"):
            raise WmsCheckpointError(f"unsupported CAV evaluation mode: {mode}")
        return mode == "closed_loop"

    @staticmethod
    def _controller_dependencies():
        """Load the frozen project controller sources, without copying them."""
        root = WORKSPACE / "_in" / "pi_deps"
        if not root.is_dir():
            # Integrated checkouts carry the same sources in-tree (repo root
            # traffic_shepherd/ and output/controller_verification_v*/).
            in_tree = (WORKSPACE / "traffic_shepherd" / "damping.py",
                       WORKSPACE / "output" / "controller_verification_v4" / "pi_external_u.py",
                       WORKSPACE / "output" / "controller_verification_v2" / "pi_saturation_v1.py")
            if not all(path.is_file() for path in in_tree):
                raise WmsCheckpointError("closed-loop PI source tree is missing: _in/pi_deps")
            root = WORKSPACE
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from traffic_shepherd.config import load_config
            from traffic_shepherd.damping import CirclesController, direct_target
            from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame
            from output.controller_verification_v4.pi_external_u import (  # type: ignore
                PIExternalU, UReport,
            )
            from output.controller_verification_v2.pi_saturation_v1 import (  # type: ignore
                PISaturationParameters,
            )
        except ImportError as exc:
            raise WmsCheckpointError(f"closed-loop controller source import failed: {exc}") from exc
        return (load_config(), CirclesController, direct_target, NeighborMeasurement,
                SensorFrame, PIExternalU, UReport, PISaturationParameters)

    def _closed_loop_callback(self, ids: Sequence[str], is_cav: np.ndarray,
                              controllers: Mapping[str, Any], control_steps: int,
                              initial_slots: int | None = None):
        """Create an episode-persistent controller dispatcher for model.rollout."""
        (cfg, circles_type, direct_target_fn, neighbor_type, frame_type,
         pi_type, u_report_type, pi_params_type) = self._controller_dependencies()
        mode_values = controllers.get("vehicle_controllers", {})
        default_mode = str(controllers.get("arm", "native"))
        aliases = {"pi": "pi-external-u", "fleet-pi": "pi-external-u"}
        selected = [aliases.get(str(mode_values.get(vid, default_mode)),
                                str(mode_values.get(vid, default_mode)))
                    for vid in ids]
        allowed = {"native", "damping", "pi-external-u", "direct"}
        if any(mode not in allowed for mode, cav in zip(selected, is_cav) if cav):
            raise WmsCheckpointError("closed-loop supports native, damping, and pi-external-u CAV actions")
        initial_cav_indices = [i for i, cav in enumerate(is_cav) if cav]
        entrant_mode = str(controllers.get("default_entrant_controller", "native"))
        entrant_mode = aliases.get(entrant_mode, entrant_mode)
        if entrant_mode not in allowed:
            raise WmsCheckpointError(
                "default_entrant_controller must be native, damping, or pi-external-u")
        if initial_slots is None:
            initial_slots = len(is_cav)
        for index in range(int(initial_slots), len(ids)):
            selected[index] = entrant_mode
        damping = {ids[i]: circles_type(cfg) for i in initial_cav_indices
                   if selected[i] == "damping"}
        pi = pi_type(sample_period_s=1.0, parameters=pi_params_type())
        u_report = u_report_type(31.29, "fixed_registered_setpoint", "cv07-registered-u31.29")
        pi_active: set[str] = set()
        active_accel: dict[str, float] = {}

        def callback(state):
            import torch
            s = state["s_m"][0].detach().cpu().numpy()
            v = state["v_mps"][0].detach().cpu().numpy()
            lanes = state["lane"][0].detach().cpu().numpy()
            present = state.get("present", torch.ones_like(state["lane"], dtype=torch.bool))[0].detach().cpu().numpy()
            live_cav = state.get("is_cav", torch.zeros_like(state["lane"], dtype=torch.bool))[0].detach().cpu().numpy()
            out = self.model.physics_accel(state)[0].detach()
            accel = out.clone()
            # Callback runs at every 0.5 s physics step. Re-evaluate observations
            # only on each 1 s control boundary and hold that slowDown target.
            if not hasattr(callback, "_physics_index"):
                callback._physics_index = 0
            step_index = int(state.get("physics_step_index", callback._physics_index))
            callback._physics_index = step_index + 1
            control_index = step_index // 2
            if step_index % 2 == 0:
                action_ids = state.get("action_id")
                if action_ids is not None:
                    from wms.model import CONTROLLER_VOCAB

                    selected[:] = [aliases.get(CONTROLLER_VOCAB[int(value)],
                                               CONTROLLER_VOCAB[int(value)])
                                   for value in action_ids[0].tolist()]
                now_s = float(controllers.get("t0_s", 0.0)) + control_index
                for i, vid in enumerate(ids):
                    if not live_cav[i]:
                        if vid in pi_active:
                            pi.invalidate(vehicle_ids=[vid], missing_ids=[vid])
                            pi_active.discard(vid)
                        active_accel.pop(vid, None)
                        continue
                    if not present[i] or lanes[i] < 0:
                        pi.invalidate(vehicle_ids=[vid], missing_ids=[vid])
                        pi_active.discard(vid)
                        active_accel.pop(vid, None)
                        continue
                    lane = lanes[i]
                    ahead = [j for j in range(len(ids)) if j != i and present[j]
                             and lanes[j] == lane and s[j] > s[i]
                             and s[j] - s[i] <= cfg.sensor_range_m]
                    leader = min(ahead, key=lambda j: s[j]) if ahead else None
                    coverage = cfg.sensor_range_m
                    front = None
                    if leader is not None:
                        gap = max(0.0, float(s[leader] - DEFAULT_LENGTH_M - s[i]))
                        front = neighbor_type(gap, float(v[leader] - v[i]))
                        coverage = max(gap, min(cfg.sensor_range_m, float(s[leader] - s[i])))
                    frame = frame_type(
                        vehicle_id=vid, timestamp_s=now_s, speed_mps=float(v[i]),
                        acceleration_mps2=0.0,
                        allowed_speed_mps=float(self.model.physics.speed_limit_mps),
                        lane_index=int(lane), front=front,
                        front_coverage_m=coverage, rear_coverage_m=cfg.sensor_range_m,
                        position_m=float(s[i]))
                    action = selected[i]
                    if action != "pi-external-u" and vid in pi_active:
                        pi.invalidate(vehicle_ids=[vid], native_ids=[vid])
                        pi_active.discard(vid)
                    if action in ("native", "direct"):
                        active_accel.pop(vid, None)
                    elif action == "damping":
                        if vid not in damping:
                            # Padding slots become live only when an entrant is
                            # spawned; create its controller at that boundary.
                            damping[vid] = circles_type(cfg)
                        decision = damping[vid].decide(frame, now_s=now_s)
                        if decision.target_speed_mps is None:
                            active_accel.pop(vid, None)
                        else:
                            active_accel[vid] = min(cfg.acceleration_max_mps2,
                                max(cfg.acceleration_min_mps2,
                                    (float(decision.target_speed_mps) - float(v[i])) /
                                    cfg.control_dt_s))
                    else:
                        if vid not in pi_active:
                            pi.reset(vid, measured_ego_speed_mps=float(v[i]))
                            pi_active.add(vid)
                        if front is None or frame.front_coverage_m <= 0:
                            pi.invalidate(vehicle_ids=[vid], native_ids=[vid])
                            pi_active.discard(vid)
                            active_accel.pop(vid, None)
                        else:
                            proposal = pi.propose(vid, timestamp_s=now_s,
                                ego_speed_mps=float(v[i]),
                                leader_speed_mps=float(v[i] + front.relative_speed_mps),
                                gap_m=front.gap_m, u_report=u_report)
                            command, target = direct_target_fn(frame,
                                (proposal.command_mps - float(v[i])) / cfg.control_dt_s, cfg)
                            if command is None:
                                pi.discard(proposal)
                                pi.invalidate(vehicle_ids=[vid], native_ids=[vid])
                                pi_active.discard(vid)
                                active_accel.pop(vid, None)
                            else:
                                pi.commit(proposal, writer_ack=True)
                                active_accel[vid] = min(cfg.acceleration_max_mps2,
                                    max(cfg.acceleration_min_mps2,
                                        (float(target) - float(v[i])) / cfg.control_dt_s))
            # slowDown's acceleration is held for BOTH half-second physics
            # integrations; native/direct handback clears it at the boundary.
            for i, vid in enumerate(ids):
                if live_cav[i] and present[i] and vid in active_accel:
                    accel[i] = active_accel[vid]
            # Integrate acceleration chosen by slowDown over the 0.5 s plant
            # interval. Native uses the model's own IDM acceleration unchanged.
            dt = float(self.model.physics.dt_s)
            speed = (state["v_mps"][0] + accel * dt).clamp_min(0.0)
            position = state["s_m"][0] + 0.5 * (state["v_mps"][0] + speed) * dt
            return torch.stack((position, state["lane"][0].float(), speed, accel),
                               dim=-1).unsqueeze(0)

        return callback

    def _action_timeline(self, state, controllers, ids, n_pad):
        """Align a validated action contract to sorted t0 and entrant slots."""
        from wms.action_timeline import ActionTimeline, CONTROLLERS

        raw = controllers.get("action_timeline", state.get("action_timeline"))
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise WmsCheckpointError("action_timeline must be a schema-declared mapping")
        try:
            timeline = ActionTimeline.from_dict(raw)
            roster = [str(vid) for vid in raw.get("vehicle_ids", ids)]
            if len(roster) != timeline.controller.shape[1] or len(set(roster)) != len(roster):
                raise ValueError("timeline needs unique vehicle_ids matching its columns")
            if set(roster) - set(ids):
                raise ValueError("timeline vehicle_ids do not match t0 identities")
            entrant_mode = str(controllers.get("default_entrant_controller", "native"))
            entrant_mode = {"pi": "pi-external-u", "fleet-pi": "pi-external-u"}.get(
                entrant_mode, entrant_mode)
            default = CONTROLLERS.index(entrant_mode)
            shape = (timeline.controller.shape[0], len(ids) + n_pad)
            ctrl = np.full(shape, default, dtype=np.int64)
            accel = np.zeros(shape, dtype=np.float32)
            target = np.zeros(shape, dtype=np.float32)
            modes = controllers.get("vehicle_controllers", {})
            arm = str(controllers.get("arm", "native"))
            for col, vid in enumerate(ids):
                mode = str(modes.get(vid, arm))
                mode = {"pi": "pi-external-u", "fleet-pi": "pi-external-u"}.get(mode, mode)
                ctrl[:, col] = CONTROLLERS.index(mode)
            for source, vid in enumerate(roster):
                col = ids.index(vid)
                ctrl[:, col] = timeline.controller[:, source]
                accel[:, col] = timeline.acceleration_mps2[:, source]
                target[:, col] = timeline.target_speed_mps[:, source]
        except (ValueError, KeyError, TypeError) as exc:
            raise WmsCheckpointError(f"invalid action timeline: {exc}") from exc
        return {"controller": self._torch.as_tensor(ctrl[None]),
                "acceleration_mps2": self._torch.as_tensor(accel[None]),
                "target_speed_mps": self._torch.as_tensor(target[None]),
                "control_dt_s": timeline.control_dt_s}

    def _resolve_counter(self, state: dict, t0_s: float) -> np.ndarray:
        """RSU-observable t0 entrance counter [2,30] (never a future arrival)."""
        counter = state.get("counter")
        if counter is not None:
            return np.asarray(counter, np.float32).reshape(2, 30)
        base = state.get("base_run_dir")
        if not base:
            raise WmsCheckpointError(
                "entries require state.counter or state.base_run_dir (t0 counters)")
        base = Path(str(base))
        base = base if base.is_absolute() else WORKSPACE / base
        if not base.is_dir():
            raise WmsCheckpointError(f"entries base_run_dir not found: {base}")
        try:
            from closedloop.live_state import base_departures_upto
            from features.counters import DepartureLog
        except ImportError as exc:  # pragma: no cover - torch workspace only
            raise WmsCheckpointError(
                "entries need the closedloop/features counter reader") from exc
        log = DepartureLog(0.0)
        for time_s, entrance in base_departures_upto(base, t0_s):
            log.add(time_s, entrance)
        return log.counter_at(t0_s).astype(np.float32)

    def _entries_v2_enabled(self) -> bool:
        flags = getattr(self, "rollout_flags", None) or {}
        if flags.get("entries_v2"):
            return True
        return bool(getattr(self.flags, "entries_v2", False))

    def _base_dir(self, state: dict) -> Path | None:
        base = state.get("base_run_dir")
        if not base:
            return None
        path = Path(str(base))
        path = path if path.is_absolute() else WORKSPACE / path
        return path if path.is_dir() else None

    def _entries_process(self, state: dict, t0_s: float, horizon_s: float, seed: int,
                         n_t0: int):
        """Build the seeded observed-counter (v1) entry process and its pad size.

        Newcomers are confined to the padding pool (indices >= ``n_t0``): a
        vacated t0 slot carries the recorded ``present_seq`` that would drop an
        entrant on the next step.
        """
        from wms.entries import EntryConfig, TorchEntryProcess, observed_rates

        cfg = EntryConfig()
        counter = self._resolve_counter(state, t0_s)
        rates = observed_rates(counter, cfg.window_bins)
        # 1.5x the expected arrivals + 16, bounded; overflow is counted as no_slot.
        n_pad = min(1024, int(np.ceil(float(rates.sum()) * float(horizon_s) * 1.5)) + 16)
        n_pad = max(1, n_pad)
        pool = np.zeros(n_t0 + n_pad, dtype=bool)
        pool[n_t0:] = True
        proc = TorchEntryProcess(counter, dt_s=DT_S, seed=int(seed), batch=1,
                                 cfg=cfg, pool=pool)
        return proc, n_pad

    def _entry_history(self, state: dict, t0_s: float, kin: np.ndarray):
        """Observed t<=t0 entry history (recency rate, headways, how-enter)."""
        from wms.entries import build_entrance_history, read_insertion_history

        counter = self._resolve_counter(state, t0_s)
        lane = kin[:, 1:4].argmax(-1)
        mainline = (lane == 0) | (lane == 1)
        ml = lane[mainline]
        lane_p1 = float((ml == 1).mean()) if ml.size else 0.5

        def rear_speed(sel):
            idx = np.flatnonzero(sel)
            if idx.size == 0:
                return None
            idx = idx[np.argsort(kin[idx, 0])]
            return float(kin[idx[:3], 4].mean())

        local = np.array([rear_speed(mainline) or 20.0,
                          rear_speed(lane == 2) or 20.0], dtype=np.float64)
        gap_vals, gap_ent = [], []
        for e, sel in ((0, mainline), (1, lane == 2)):
            idx = np.flatnonzero(sel)
            idx = idx[np.argsort(kin[idx, 0])]
            for a_i, b_i in zip(idx[:-1], idx[1:]):
                g = float(kin[b_i, 0] - kin[a_i, 0] - kin[a_i, 6])
                if g > 0.0:
                    gap_vals.append(g)
                    gap_ent.append(e)
        gaps = np.asarray(gap_vals, dtype=np.float64)
        gap_e = np.asarray(gap_ent, dtype=np.int64)

        base = self._base_dir(state)
        obs = read_insertion_history(base, t0_s) if base is not None else None
        if obs is not None and obs["times"].size:
            return build_entrance_history(
                counter=counter, times=obs["times"], entrance=obs["entrance"],
                speed=obs["speed"], local_speed=local[obs["entrance"]],
                is_cav=obs["is_cav"], vtype=obs["vtype"],
                mainline_lane_share=lane_p1, gap=gaps, gap_entrance=gap_e, t0_s=t0_s)
        return build_entrance_history(counter=counter, mainline_lane_share=lane_p1,
                                      gap=gaps, gap_entrance=gap_e, t0_s=t0_s)

    def _entries_process_v2(self, state: dict, t0_s: float, horizon_s: float,
                            seed: int, kin: np.ndarray):
        """Build the seeded observed-history (v2) entry process and its pad size."""
        from wms.entries import EntryConfig, TorchEntryProcessV2

        cfg = EntryConfig()
        history = self._entry_history(state, t0_s, kin)
        n_t0 = int(kin.shape[0])
        n_pad = min(1024, int(np.ceil(float(history.rate.sum())
                                      * float(horizon_s) * 1.5)) + 16)
        n_pad = max(1, n_pad)
        pool = np.zeros(n_t0 + n_pad, dtype=bool)
        pool[n_t0:] = True
        proc = TorchEntryProcessV2(history=history, dt_s=DT_S, seed=int(seed),
                                   batch=1, cfg=cfg, pool=pool)
        return proc, n_pad

    def predict_states(self, state: dict, cav_controllers: dict,
                       horizon_s: float, seed: int) -> list[dict]:
        torch = self._torch
        model = self.model or self._load()
        arm = str((cav_controllers or {}).get("arm", "native"))
        controller_id = None
        if self.model_kind == "blackbox_id":
            # The black-box-id model predicts CAVs itself from the arm's
            # controller id; the recorded CAV trajectory is not the only hint.
            from wms.model import controller_index

            controller_id = torch.as_tensor(controller_index(arm), dtype=torch.long)
        t0_s, ids, kin, is_cav = self._inputs(state, arm)
        driver_history, driver_history_mask = self._driver_history(state, ids)
        closed_loop = self._closed_loop_enabled(cav_controllers or {})
        n_steps = max(1, int(round(float(horizon_s) / DT_S)))
        entries_on = self._entries_enabled()
        entries_v2 = self._entries_v2_enabled()
        proc, n_pad = (None, 0)
        if entries_on:
            if entries_v2:
                proc, n_pad = self._entries_process_v2(state, t0_s, horizon_s, seed, kin)
            else:
                proc, n_pad = self._entries_process(state, t0_s, horizon_s, seed, len(ids))
        n_slots = len(ids) + n_pad
        action_timeline = self._action_timeline(state, cav_controllers or {}, ids, n_pad)
        initial = {
            "s_m": torch.as_tensor(kin[None, :, 0]),
            "v_mps": torch.as_tensor(kin[None, :, 4]),
            "a_mps2": torch.zeros((1, len(ids))),
            "lane": torch.as_tensor(kin[None, :, 1:4].argmax(-1)),
            "length_m": torch.as_tensor(kin[None, :, 6]),
            "present": torch.ones((1, len(ids)), dtype=torch.bool),
            "is_cav": torch.as_tensor(is_cav[None]),
        }
        teacher, present_seq, junction_seq = self._teacher(
            state, t0_s, ids, kin, is_cav, n_steps, arm)
        closed_callback = None
        if closed_loop:
            default_entrant_controller = str(
                (cav_controllers or {}).get("default_entrant_controller", "native"))
            if default_entrant_controller not in {
                    "native", "damping", "pi", "fleet-pi", "pi-external-u"}:
                raise WmsCheckpointError(
                    "default_entrant_controller must be native, damping, or pi-external-u")
            callback_ids = list(ids) + [f"__entry_{i}" for i in range(n_pad)]
            callback_cav = np.concatenate((is_cav, np.zeros(n_pad, dtype=bool)))
            closed_callback = self._closed_loop_callback(
                callback_ids, callback_cav,
                dict(cav_controllers or {}, t0_s=t0_s), n_steps,
                initial_slots=len(ids))
            teacher = None
            present_seq = junction_seq = None
        if entries_on:
            pad_shape = (1, n_steps + 1, n_pad)
            initial = {
                "s_m": torch.cat((initial["s_m"],
                                  torch.zeros(1, n_pad)), dim=1),
                "v_mps": torch.cat((initial["v_mps"],
                                    torch.zeros(1, n_pad)), dim=1),
                "a_mps2": torch.cat((initial["a_mps2"],
                                     torch.zeros(1, n_pad)), dim=1),
                "lane": torch.cat((initial["lane"],
                                   torch.zeros(1, n_pad, dtype=torch.long)), dim=1),
                "length_m": torch.cat((initial["length_m"],
                                       torch.full((1, n_pad), DEFAULT_LENGTH_M)), dim=1),
                "present": torch.cat((initial["present"],
                                      torch.zeros(1, n_pad, dtype=torch.bool)), dim=1),
                "is_cav": torch.cat((initial["is_cav"],
                                     torch.zeros(1, n_pad, dtype=torch.bool)), dim=1),
            }
            if present_seq is not None:
                present_seq = torch.cat(
                    (present_seq, torch.ones(*pad_shape, dtype=torch.bool)), dim=2)
            if teacher is not None:
                teacher = torch.cat(
                    (teacher, torch.zeros(1, n_steps + 1, n_pad, 4)), dim=2)
            if junction_seq is not None:
                # Padding/entry slots are never junction transits; pad the
                # junction mask to the entrant slot count or the rollout's
                # ``present | junction`` reduction broadcasts against the wrong
                # width (entries + junction carry-through).
                junction_seq = torch.cat(
                    (junction_seq, torch.zeros(*pad_shape, dtype=torch.bool)), dim=2)
            if driver_history is not None:
                # Entry padding slots carry no observed history; the encoder
                # falls back to the physics defaults for them (mask False).
                dh_pad = torch.zeros(1, n_pad, driver_history.shape[2],
                                     driver_history.shape[3],
                                     dtype=driver_history.dtype)
                driver_history = torch.cat((driver_history, dh_pad), dim=1)
                dhm_pad = torch.zeros(1, n_pad, driver_history_mask.shape[2],
                                      dtype=torch.bool)
                driver_history_mask = torch.cat((driver_history_mask, dhm_pad), dim=1)
        with torch.no_grad():
            if teacher is None:
                roll = model.rollout(
                    initial, n_steps=n_steps, k0=0,
                    cav_controller=(closed_callback if closed_loop else
                                    model.hold_teacher
                                    if (bool(is_cav.any()) or entries_on) else None),
                    present_seq=present_seq, entries=proc,
                    driver_history=driver_history,
                    driver_history_mask=driver_history_mask,
                    action_timeline=action_timeline, controller_id=controller_id,
                    projection=self.projection,
                )
            else:
                roll = model.rollout(
                    initial, n_steps=n_steps, k0=0,
                    cav_teacher=teacher, present_seq=present_seq,
                    junction_seq=junction_seq,
                    cav_controller=model.hold_teacher if entries_on else None,
                    entries=proc, driver_history=driver_history,
                    driver_history_mask=driver_history_mask,
                    controller_id=controller_id,
                    action_timeline=action_timeline,
                    projection=self.projection,
                )
        positions = roll["positions"][0].numpy()
        speeds = roll["speeds"][0].numpy()
        lanes = roll["lanes"][0].numpy()
        masks = roll["masks"][0].numpy()
        slot_vid = list(ids) + [f"__entry_{i}" for i in range(n_pad)]
        final_cav = roll["state"]["is_cav"][0].numpy()
        cav_slot = np.array([bool(is_cav[k]) if k < len(ids) else bool(final_cav[k])
                             for k in range(n_slots)])
        frames = [{"time_s": t0_s, "vehicles": [
            {"id": slot_vid[k], "s": float(kin[k, 0]),
             "lane": int(kin[k, 1:4].argmax(-1)), "v": float(kin[k, 4]),
             "present": True, "is_cav": bool(is_cav[k])}
            for k in range(len(ids))]}]
        for index in range(n_steps):
            vehicles = []
            for k in range(n_slots):
                if not bool(masks[index, k]):
                    continue
                vehicles.append({
                    "id": slot_vid[k], "s": float(positions[index, k]),
                    "lane": int(lanes[index, k]), "v": float(speeds[index, k]),
                    "present": True, "is_cav": bool(cav_slot[k]),
                })
            frames.append({"time_s": t0_s + (index + 1) * DT_S, "vehicles": vehicles})
        if proc is not None:
            self.entries_summary = proc.summary()
        return frames
