"""Prioritised mixed replay (V10HORIZON): merge-zone / CAV-follower / brake-event / high-error.

``PriorityReplayDataset`` wraps whatever mixed TRAIN dataset the assembly built (loop data
+ original native/random sources, already combined by :mod:`wms.data.mixdata` /
:mod:`wms.data.c4_episodes`) and supplies the per-stage sampling weights the trainer already
honours.  Every window gets a multiplier

    base_source_weight(stage) x merge-zone_mult x cav-follower_mult
        x brake-event_mult x high-error_mult

so the source mix ratios are preserved and the priority terms compose on top.  ``merge-zone``,
``cav-follower`` and ``brake-event`` are static per-window flags
(:mod:`wms.data.replay_flags`); the ``high-error`` multiplier is updated online from the
forward-only per-window positional error
(:meth:`wms.train.unroll.UnrollRunner.window_errors`), so windows the model predicts poorly
are drawn more often.  All terms default to 1 (OFF = the weights the base dataset already
produced, bit-for-bit).
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from wms.core.errors import TrainDataRefusal
from wms.data.replay_flags import REPLAY_SCHEMA, replay_flags  # noqa: F401 (re-exported)

REPLAY_RECEIPT_SCHEMA = "wms_priority_replay/1"


class ReplayRefusal(TrainDataRefusal):
    """A priority-replay contract violation (fail closed)."""


class PriorityReplayDataset(Dataset):
    """Priority-weighted view of a training dataset (default OFF = unchanged weights)."""

    def __init__(self, dataset: Dataset, *, merge_zone, cav_follower,
                 brake_event=None, merge_oversample: float = 1.0,
                 cav_oversample: float = 1.0, brake_oversample: float = 1.0,
                 error_alpha: float = 1.0, error_decay: float = 0.9,
                 error_max_boost: float = 8.0) -> None:
        n = len(dataset)
        merge = np.asarray(merge_zone, dtype=bool)
        follower = np.asarray(cav_follower, dtype=bool)
        brake = (np.zeros(n, dtype=bool) if brake_event is None
                 else np.asarray(brake_event, dtype=bool))
        if merge.shape != (n,) or follower.shape != (n,) or brake.shape != (n,):
            raise ReplayRefusal("replay flags must give one bool per window")
        for name, value in (("merge_oversample", merge_oversample),
                            ("cav_oversample", cav_oversample),
                            ("brake_oversample", brake_oversample),
                            ("error_alpha", error_alpha),
                            ("error_max_boost", error_max_boost)):
            if not math.isfinite(float(value)) or float(value) < 1.0:
                raise ReplayRefusal(f"replay {name} must be finite and >= 1")
        if not (0.0 <= float(error_decay) < 1.0):
            raise ReplayRefusal("replay error_decay must be in [0, 1)")
        self.dataset = dataset
        self.merge_zone, self.cav_follower, self.brake_event = merge, follower, brake
        self.merge_oversample = float(merge_oversample)
        self.cav_oversample = float(cav_oversample)
        self.brake_oversample = float(brake_oversample)
        self.error_alpha = float(error_alpha)
        self.error_decay = float(error_decay)
        self.error_max_boost = float(error_max_boost)
        self._error = np.zeros(n, dtype=np.float64)
        self._error_seen = np.zeros(n, dtype=bool)
        base = dict(getattr(dataset, "receipt", {}) or {})
        self.receipt = dict(base)
        self.receipt["replay"] = {
            "schema": REPLAY_RECEIPT_SCHEMA, "n_windows": n,
            "n_merge_zone": int(merge.sum()), "n_cav_follower": int(follower.sum()),
            "n_brake_event": int(brake.sum()), "n_both": int((merge & follower).sum()),
            "merge_oversample": self.merge_oversample,
            "cav_oversample": self.cav_oversample,
            "brake_oversample": self.brake_oversample,
            "error_alpha": self.error_alpha, "error_decay": self.error_decay,
            "error_max_boost": self.error_max_boost,
            "error_weighted_windows": 0}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        return self.dataset[index]

    # ------------------------------------------------------------- weights
    def base_stage_weights(self, stage_s: float) -> np.ndarray:
        """The base dataset's stage weights (or uniform when it has none)."""
        fn = getattr(self.dataset, "stage_weights", None)
        if callable(fn):
            weights = np.asarray(fn(stage_s), dtype=np.float64)
            if weights.shape != (len(self),):
                raise ReplayRefusal("base stage_weights length mismatch")
            return weights
        return np.ones(len(self), dtype=np.float64)

    def static_multiplier(self) -> np.ndarray:
        mult = np.ones(len(self), dtype=np.float64)
        if self.merge_oversample > 1.0:
            mult = mult * np.where(self.merge_zone, self.merge_oversample, 1.0)
        if self.cav_oversample > 1.0:
            mult = mult * np.where(self.cav_follower, self.cav_oversample, 1.0)
        if self.brake_oversample > 1.0:
            mult = mult * np.where(self.brake_event, self.brake_oversample, 1.0)
        return mult

    def error_multiplier(self) -> np.ndarray:
        mult = np.ones(len(self), dtype=np.float64)
        if not self._error_seen.any():
            return mult
        seen = self._error
        mean = float(seen[self._error_seen].mean())
        if not math.isfinite(mean) or mean <= 0.0:
            return mult
        ratio = np.zeros_like(mult)
        ratio[self._error_seen] = seen[self._error_seen] / mean
        boost = np.clip(ratio ** float(self.error_alpha), 1.0, float(self.error_max_boost))
        boost[~self._error_seen] = 1.0
        return boost

    def stage_weights(self, stage_s: float) -> list[float]:
        weights = (self.base_stage_weights(stage_s) * self.static_multiplier()
                   * self.error_multiplier())
        total = float(weights.sum())
        if not math.isfinite(total) or total <= 0.0:
            weights = np.ones(len(self), dtype=np.float64)
            total = float(weights.sum())
        return (weights / total).tolist()

    # -------------------------------------------------------------- errors
    def update_errors(self, indices, values, *, decay: Optional[float] = None) -> None:
        """EMA-update the per-window error used by the high-error multiplier."""
        d = self.error_decay if decay is None else float(decay)
        if not (0.0 <= d < 1.0):
            raise ReplayRefusal("replay error decay must be in [0, 1)")
        idx = np.asarray(indices, dtype=np.int64).ravel()
        val = np.asarray(values, dtype=np.float64).ravel()
        if idx.shape != val.shape:
            raise ReplayRefusal("update_errors needs one value per index")
        for i, v in zip(idx.tolist(), val.tolist()):
            if not 0 <= i < len(self):
                raise ReplayRefusal(f"update_errors index {i} out of range")
            if not math.isfinite(v):
                continue
            self._error[i] = d * self._error[i] + (1.0 - d) * v
            self._error_seen[i] = True
        self.receipt["replay"]["error_weighted_windows"] = int(self._error_seen.sum())

    def error_snapshot(self) -> dict:
        """Priority bookkeeping for the stage/validation log."""
        if not self._error_seen.any():
            return {"n_seen": 0, "mean_error": None, "max_error": None,
                    "mean_boost": 1.0, "max_boost": 1.0}
        seen = self._error_seen
        boost = self.error_multiplier()
        return {"n_seen": int(seen.sum()),
                "mean_error": float(self._error[seen].mean()),
                "max_error": float(self._error[seen].max()),
                "mean_boost": float(boost[seen].mean()),
                "max_boost": float(boost[seen].max())}

    def describe_stages(self, stages_s: Sequence[float]) -> list[dict]:
        """Expected per-stage draw weights of the priority terms (receipt evidence)."""
        rows = []
        for stage in stages_s:
            static = self.static_multiplier()
            n = len(self)
            rows.append({"stage_s": float(stage), "n_windows": n,
                         "merge_draw_fraction": round(
                             float((static[self.merge_zone]).sum() / static.sum()), 4),
                         "cav_follower_draw_fraction": round(
                             float((static[self.cav_follower]).sum() / static.sum()), 4),
                         "brake_event_draw_fraction": round(
                             float((static[self.brake_event]).sum() / static.sum()), 4),
                         "error_weighted_windows": int(self._error_seen.sum())})
        return rows


__all__ = ["REPLAY_RECEIPT_SCHEMA", "ReplayRefusal", "PriorityReplayDataset", "replay_flags"]
