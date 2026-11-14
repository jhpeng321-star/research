"""Pre-t0 driver-history tensors for history-conditioned checkpoints.

Two materialised schemas are accepted (see :mod:`wms.eval.states`):

* id-keyed: ``history.ids`` + raw pre-t0 ``kin``/``mask``/``target_mask`` on the pack
  columns of the t0 vehicles; columns are re-ordered to the sorted rollout slot order and
  featurised as the trainer did.  Valid for every checkpoint.
* slot-indexed: ``history.kin``/``mask`` ``[T,K,7]`` addressed by ``slotN`` ids; required
  (fail closed) by a history lane-head or ``hybrid`` checkpoint.

Long-history checkpoints (``config.history_s``) always require duration-declared native
history and use the shared trainer featuriser for either schema.  Residual or plain
``driver_params`` checkpoints without history conditioning get ``(None, None)`` and fall
back to the encoder's empty-history defaults.  Only ticks ``<= t0`` are ever read.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .backend import WmsBackend
from .errors import PredictorError
from .normalize import DT_S

GAP_FLOOR_M = 0.3
GAP_FREE_M = 250.0
KIN_FEATURES = 7


class HistoryInputs:
    """Builds ``(driver_history, driver_history_mask)`` for one checkpoint."""

    def __init__(self, backend: WmsBackend) -> None:
        self._model = backend.model
        self._config = backend.config
        self._tools = backend.tools

    # ------------------------------------------------------------------ config
    def long_history_config(self) -> tuple[float | None, float]:
        history_s = getattr(self._model, "history_s", self._config.get("history_s"))
        stride_s = getattr(self._model, "history_stride_s",
                           self._config.get("history_stride_s", 1.0))
        return history_s, stride_s

    @property
    def is_long(self) -> bool:
        return self.long_history_config()[0] is not None

    # ------------------------------------------------------------------ public
    def build(self, state: Mapping[str, Any], ids: Sequence[str]):
        hist = state.get("history")
        if self.is_long and not isinstance(hist, Mapping):
            raise PredictorError("long-history checkpoint requires pre-t0 history")
        if isinstance(hist, Mapping) and hist.get("ids"):
            return self._by_id(hist, ids)
        return self._by_slot(state, ids)

    # ------------------------------------------------------------- validation
    def validate_long(self, hist: Mapping[str, Any]) -> None:
        """Fail closed on legacy, downsampled, or insufficient pre-t0 states."""
        history_s, stride_s = self.long_history_config()
        try:
            self._tools.history_spec(history_s, stride_s, DT_S)
            kin = np.asarray(hist["kin"], dtype=np.float32)
            mask = np.asarray(hist["mask"], dtype=bool)
            target = np.asarray(hist["target_mask"], dtype=bool)
            declared_s = float(hist["history_s"])
            declared_stride = float(hist["history_stride_s"])
            dt_s = float(hist["dt_s"])
            available_s = float(hist["available_span_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PredictorError(
                "long history requires native tensors and duration metadata") from exc
        if (kin.ndim != 3 or kin.shape[-1] != KIN_FEATURES or kin.shape[0] < 1
                or mask.shape != kin.shape[:2]):
            raise PredictorError("long history input must be [T,K,7]/[T,K], T >= 1")
        t, k = kin.shape[:2]
        if target.shape != (t - 1, k) and not (t == 1 and target.size == 0):
            raise PredictorError("long history target_mask must cover all native transitions")
        if (not np.isfinite([declared_s, declared_stride, dt_s, available_s]).all()
                or not np.isclose(dt_s, DT_S, rtol=0.0, atol=1e-6)
                or not np.isclose(declared_stride, float(stride_s), rtol=0.0, atol=1e-6)):
            raise PredictorError("long history requires dt_s=0.5 and matching history_stride_s")
        if not np.isclose(available_s, (t - 1) * DT_S, rtol=0.0, atol=1e-6):
            raise PredictorError("long history available_span_s does not match native ticks")
        if declared_s + 1e-6 < float(history_s):
            raise PredictorError("state history_s is shorter than checkpoint history_s")
        native_steps, _, _ = self._tools.history_spec(history_s, stride_s, DT_S)
        if t < native_steps and hist.get("run_start_limited") is not True:
            raise PredictorError("short long history requires explicit run_start_limited=true")

    # ---------------------------------------------------------------- builders
    def _long_features(self, kin, mask, target, present_now):
        history_s, stride_s = self.long_history_config()
        target = np.asarray(target, dtype=bool).reshape(max(0, kin.shape[0] - 1), kin.shape[1])
        return self._tools.long_history_features(
            torch.as_tensor(kin, dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.bool),
            torch.as_tensor(target, dtype=torch.bool),
            torch.as_tensor(present_now, dtype=torch.bool),
            history_s=history_s, history_stride_s=stride_s, dt_s=DT_S)

    def _by_id(self, hist: Mapping[str, Any], ids: Sequence[str]):
        if self.is_long:
            self.validate_long(hist)
        src_ids = [str(v) for v in hist["ids"]]
        col_of = {vid: i for i, vid in enumerate(src_ids)}
        kin = np.asarray(hist["kin"], dtype=np.float32)          # [T,S,7]
        mask = np.asarray(hist["mask"]).astype(bool)             # [T,S]
        target = np.asarray(hist.get("target_mask", []), dtype=bool)
        if kin.ndim != 3:
            raise PredictorError("state.history.kin must be [T,S,7]")
        t, s = kin.shape[:2]
        if self.is_long and (len(src_ids) != s or len(col_of) != s):
            raise PredictorError("long history ids must be unique and match its columns")
        k = len(ids)
        if self.is_long:
            # Featurise before selecting the t0 roster so an observed leader which left
            # before t0 stays available as a historical cue.
            all_features, all_masks = self._long_features(kin, mask, target, mask[-1])
            want = getattr(self._model, "driver_history_len", all_features.shape[1])
            feat = torch.zeros(k, want, all_features.shape[-1])
            hmask = torch.zeros(k, want, dtype=torch.bool)
            for j, vid in enumerate(ids):
                col = col_of.get(str(vid))
                if col is not None:
                    feat[j], hmask[j] = all_features[col], all_masks[col]
            return feat[None], hmask[None]
        kin_hist = np.zeros((t, k, KIN_FEATURES), dtype=np.float32)
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
        history_len = int(getattr(self._model, "history_len", 10))
        feat, hmask = self.prior_history_features(
            kin_hist, mask_hist, target_hist, present_now, history_len=history_len)
        return feat[None], hmask[None]

    def _by_slot(self, state: Mapping[str, Any], ids: Sequence[str]):
        conditioned = (self._config.get("lane_head", "off") == "history"
                       or self._config.get("rollout_mode", "residual") == "hybrid"
                       or self.is_long)
        if not conditioned:
            return None, None
        raw = state.get("history")
        if not isinstance(raw, Mapping) or "kin" not in raw or "mask" not in raw:
            raise PredictorError("history-conditioned checkpoint requires pre-t0 history.kin/mask")
        if self.is_long:
            self.validate_long(raw)
        kin = np.asarray(raw["kin"], dtype=np.float32)
        mask = np.asarray(raw["mask"], dtype=bool)
        if kin.ndim != 3 or kin.shape[-1] != KIN_FEATURES or mask.shape != kin.shape[:2]:
            raise PredictorError("history lane-head input must be [T,K,7]/[T,K]")
        try:
            slots = [int(str(vid).removeprefix("slot")) for vid in ids]
        except ValueError as exc:
            raise PredictorError(
                "history lane-head input requires slot-indexed vehicle ids") from exc
        if not slots or max(slots) >= kin.shape[1] or (self.is_long and min(slots) < 0):
            raise PredictorError("history lane-head slots do not align with vehicles")
        if self.is_long:
            target = np.asarray(raw["target_mask"], dtype=bool).reshape(
                max(0, kin.shape[0] - 1), kin.shape[1])
            features, feature_mask = self._long_features(kin, mask, target, mask[-1])
            return features[slots][None], feature_mask[slots][None]
        features, feature_mask = self.history_features_from_kin(
            kin[:, slots], mask[:, slots], history_len=int(self._model.history_len))
        return features[None], feature_mask[None]

    # --------------------------------------------------------- featurisation
    @staticmethod
    def _leader_gap(kin: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Same-lane bumper gap to the nearest valid leader per ``[T,K]`` step, clamped."""
        lane = kin[..., 1:4].argmax(-1)
        s_ = kin[..., 0]
        t, k = s_.shape
        gap = torch.full((t, k), GAP_FREE_M)
        for ti in range(t):
            ds = s_[ti].unsqueeze(0) - s_[ti].unsqueeze(1)
            ok = valid[ti].unsqueeze(0) & (lane[ti].unsqueeze(0) == lane[ti].unsqueeze(1)) \
                & (ds > 0)
            g = torch.where(ok, ds - kin[ti, :, 6].unsqueeze(0),
                            torch.full_like(ds, GAP_FREE_M))
            gap[ti] = g.min(-1).values.clamp(GAP_FLOOR_M, GAP_FREE_M)
        return gap

    def prior_history_features(self, kin_hist, mask_hist, target_hist, present_now,
                               *, history_len: int = 10):
        """Trainer-identical t0 features ``[K,H,F]`` + mask ``[K,H]`` (right-aligned, zero-padded).

        Continuity follows the backward ``target_mask`` product so a slot reused by another
        vehicle never leaks the departed vehicle's motion.
        """
        kin = torch.as_tensor(np.asarray(kin_hist, dtype=np.float32))
        present = torch.as_tensor(np.asarray(mask_hist).astype(bool))
        if kin.ndim != 3 or kin.shape[-1] != KIN_FEATURES:
            raise PredictorError("driver history kin must be [T,K,7]")
        t, k = kin.shape[:2]
        tgt = np.asarray(target_hist, dtype=bool) if target_hist is not None else None
        tm = (torch.zeros(max(0, t - 1), k, dtype=torch.bool) if tgt is None or tgt.size == 0
              else torch.as_tensor(tgt))
        same = torch.ones(t, k, dtype=torch.bool)
        for j in range(t - 2, -1, -1):
            same[j] = same[j + 1] & tm[j]
        valid = same & present
        valid[-1] = torch.as_tensor(np.asarray(present_now).astype(bool))
        feat = self._tools.history_step_features(kin, self._leader_gap(kin, valid))  # [T,K,F]
        want = max(1, int(history_len))
        out = torch.zeros(k, want, feat.shape[-1], dtype=feat.dtype)
        out_mask = torch.zeros(k, want, dtype=torch.bool)
        use = min(t, want)
        out[:, want - use:] = feat[-use:].transpose(0, 1)
        out_mask[:, want - use:] = valid[-use:].transpose(0, 1)
        return out, out_mask

    def history_features_from_kin(self, kin_hist, mask_hist, *, history_len: int):
        """Slot-indexed ``[T,K,7]`` history -> ``[K,H,F]`` tensors (front-pad / keep newest)."""
        kin = torch.as_tensor(np.asarray(kin_hist, dtype=np.float32))
        present = torch.as_tensor(np.asarray(mask_hist).astype(bool))
        if kin.ndim != 3 or kin.shape[-1] != KIN_FEATURES:
            raise PredictorError("history kin must be [T,K,7]")
        if history_len < 1:
            raise PredictorError("history_len must be positive")
        feat = self._tools.history_step_features(kin, self._leader_gap(kin, present))
        out = feat.transpose(0, 1).numpy()                  # [K,T,F]
        out_mask = present.transpose(0, 1).numpy()          # [K,T]
        if out.shape[1] > history_len:
            out, out_mask = out[:, -history_len:], out_mask[:, -history_len:]
        elif out.shape[1] < history_len:
            pad = history_len - out.shape[1]
            out = np.concatenate((np.zeros((out.shape[0], pad, out.shape[2]), dtype=out.dtype),
                                  out), axis=1)
            out_mask = np.concatenate(
                (np.zeros((out_mask.shape[0], pad), dtype=bool), out_mask), axis=1)
        return (torch.as_tensor(out, dtype=torch.float32),
                torch.as_tensor(out_mask, dtype=torch.bool))
