"""Per-target, per-tau outcome-target normalization (M1FIX2 item 1).

The outcome targets (``delay_s``, ``ttc_lt3_s``, ``hard_brake_s``) are raw and
span several orders of magnitude, so the CRPS/contrast terms are dominated by
their scale alone.  This module fits a robust per-(tau, target) location and
scale ONCE from TRAIN items only and is applied to the outcome-quantile
targets before the correction-CRPS / response-contrast losses; predictions are
de-normalized for evaluation, the API and the incident controller.

The statistics ride inside the checkpoint (``outcome_norm`` payload) and are
also written as ``OUTCOME_NORM.json`` next to the member checkpoint; the sha256
of the canonical JSON is recorded in ``TRAIN_RECEIPT.json``.  CAL/TEST items
are never used to fit the statistics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from . import constants as C

SCHEMA = "wm_n1_outcome_norm/1"
DEFAULT_MIN_SCALE = 1e-3


def _canonical_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, indent=2).encode("utf-8")


class OutcomeNorm:
    """Robust per-(tau, target) location/scale for outcome targets.

    ``loc``/``scale`` are ``[N_TAU, N_TGT]`` nested lists.  ``normalize`` maps
    a raw target tensor to zero-median unit-scale space and ``denormalize``
    inverts it (predictions live in normalized space during training).
    """

    def __init__(self, loc, scale, *, targets=None, taus=None,
                 min_scale: float = DEFAULT_MIN_SCALE) -> None:
        loc = np.asarray(loc, dtype=np.float64).reshape(C.N_TAU,
                                                        C.N_OUTCOME_TARGETS)
        scale = np.asarray(scale, dtype=np.float64).reshape(
            C.N_TAU, C.N_OUTCOME_TARGETS)
        if not np.isfinite(loc).all() or not np.isfinite(scale).all():
            raise ValueError("outcome-normalization stats must be finite")
        if (scale <= 0.0).any():
            raise ValueError("outcome-normalization scales must be positive")
        self.loc = loc
        self.scale = scale
        self.targets = tuple(targets) if targets is not None else C.OUTCOME_TARGETS
        self.taus = tuple(taus) if taus is not None else C.TAUS
        self.min_scale = float(min_scale)

    # -- construction -------------------------------------------------------
    @classmethod
    def identity(cls) -> "OutcomeNorm":
        return cls(np.zeros((C.N_TAU, C.N_OUTCOME_TARGETS)),
                   np.ones((C.N_TAU, C.N_OUTCOME_TARGETS)))

    @classmethod
    def fit(cls, targets, masks, *, min_scale: float = DEFAULT_MIN_SCALE
            ) -> "OutcomeNorm":
        """Fit robust stats from TRAIN ``(targets, masks)`` pairs ONLY.

        ``targets`` are raw ``[N_TAU, N_TGT]`` arrays/tensors; ``masks`` the
        per-item ``[N_TAU]`` availability masks.  Per (tau, target) the
        location is the median and the scale the IQR (q75 - q25); a degenerate
        IQR falls back to the standard deviation and then to 1.0, so a constant
        TRAIN target cannot blow the normalized values up.
        """
        values = [[] for _ in range(C.N_TAU * C.N_OUTCOME_TARGETS)]
        for y, m in zip(targets, masks):
            arr = np.asarray(y, dtype=np.float64).reshape(
                C.N_TAU, C.N_OUTCOME_TARGETS)
            mask = np.asarray(m, dtype=bool).reshape(C.N_TAU)
            for i in range(C.N_TAU):
                if not bool(mask[i]):
                    continue
                for j in range(C.N_OUTCOME_TARGETS):
                    v = float(arr[i, j])
                    if np.isfinite(v):
                        values[i * C.N_OUTCOME_TARGETS + j].append(v)
        loc = np.zeros((C.N_TAU, C.N_OUTCOME_TARGETS))
        scale = np.ones((C.N_TAU, C.N_OUTCOME_TARGETS))
        for i in range(C.N_TAU):
            for j in range(C.N_OUTCOME_TARGETS):
                vals = np.asarray(values[i * C.N_OUTCOME_TARGETS + j],
                                  dtype=np.float64)
                if vals.size == 0:
                    loc[i, j] = 0.0
                    scale[i, j] = 1.0
                    continue
                loc[i, j] = float(np.median(vals))
                iqr = float(np.quantile(vals, 0.75)
                            - np.quantile(vals, 0.25))
                if iqr < min_scale:
                    iqr = float(np.std(vals))
                scale[i, j] = float(max(iqr, min_scale))
        return cls(loc, scale, min_scale=min_scale)

    @classmethod
    def from_items(cls, items, **kw) -> "OutcomeNorm":
        """Fit from dataset items carrying ``outcome_quantiles``/``outcome_mask``.

        A window may carry a leading batch axis (the canonical windows do), so
        any leading dimensions are flattened into separate per-item targets.
        """
        targets, masks = [], []
        for item in items:
            y = item.get("outcome_quantiles")
            m = item.get("outcome_mask")
            if y is None or m is None:
                continue
            if isinstance(y, Tensor):
                y = y.detach().cpu().numpy()
            if isinstance(m, Tensor):
                m = m.detach().cpu().numpy()
            arr = np.asarray(y, dtype=np.float64)
            msk = np.asarray(m, dtype=bool)
            arr = arr.reshape(-1, C.N_TAU, C.N_OUTCOME_TARGETS)
            msk = msk.reshape(-1, C.N_TAU)
            for b in range(arr.shape[0]):
                targets.append(arr[b])
                masks.append(msk[b])
        return cls.fit(targets, masks, **kw)

    # -- transforms ---------------------------------------------------------
    def _stats_for(self, ref: Tensor) -> tuple:
        """``(loc, scale)`` broadcastable to ``ref``.

        A quantile-shaped tensor ``[..., N_TAU, N_TGT, NQ]`` gets a trailing
        singleton on the stats (aligned to the tau/target axes); a target
        tensor ``[..., N_TAU, N_TGT]`` does not.  Both are left-padded with
        singletons for any leading batch/ensemble axes.
        """
        loc = torch.as_tensor(self.loc, dtype=ref.dtype, device=ref.device)
        scale = torch.as_tensor(self.scale, dtype=ref.dtype, device=ref.device)
        nq = C.N_QUANTILES
        if ref.dim() >= 3 and int(ref.shape[-1]) == nq \
                and int(ref.shape[-3]) == C.N_TAU:
            loc = loc.unsqueeze(-1)
            scale = scale.unsqueeze(-1)
        while loc.dim() < ref.dim():
            loc = loc.unsqueeze(0)
            scale = scale.unsqueeze(0)
        return loc, scale

    def normalize(self, y: Tensor) -> Tensor:
        loc, scale = self._stats_for(y)
        return (y - loc) / scale

    def denormalize(self, q: Tensor) -> Tensor:
        loc, scale = self._stats_for(q)
        return q * scale + loc

    # -- io -----------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "taus": [float(t) for t in self.taus],
            "targets": list(self.targets),
            "loc": self.loc.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OutcomeNorm":
        if data.get("schema") != SCHEMA:
            raise ValueError(
                f"outcome-norm schema {data.get('schema')!r} != {SCHEMA!r}")
        return cls(data["loc"], data["scale"], targets=data.get("targets"),
                   taus=data.get("taus"))

    def sha256(self) -> str:
        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()

    def write(self, path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = _canonical_bytes(self.to_dict())
        path.write_bytes(body)
        return hashlib.sha256(body).hexdigest()

    @classmethod
    def read(cls, path) -> "OutcomeNorm":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def denormalize_quantiles(q: Tensor, norm: OutcomeNorm) -> Tensor:
    """De-normalize outcome quantiles ``[..., N_TAU, N_TGT, NQ]``.

    The location/scale broadcast over the trailing quantile axis; a tensor is
    returned in the input dtype/device.
    """
    return norm.denormalize(q)


__all__ = ["OutcomeNorm", "denormalize_quantiles", "SCHEMA",
           "DEFAULT_MIN_SCALE"]
