"""Learned native-CAV twin (V10NATIVECAV).

The counterfactual ``base`` rows and the entrance-zone CAVs are driven by a *native* CAV
behaviour, never by the policy.  The legacy twin is a fixed-parameter IDM
(:func:`wms.control.counterfactual.idm_native_accel`).  On TRAIN-val native episodes that
twin hard-brakes (a < -2.5 m/s^2) several times as often as the recorded SUMO CAVs, which
makes the policy look better than it is.  This module supplies a small supervised head
fitted to recorded native-CAV accelerations (``tools/native_cav_fit.py``) behind the
``counterfactual.native_cav = "head"`` switch; the default remains the legacy IDM.

Inputs are the same information the IDM uses: own speed, the same-lane leader's bumper gap
and speed difference, lane, position in the merge frame and presence.  The head is a
9-32-32-1 tanh MLP stored as JSON so a checkpoint/config is portable without the training
data.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Optional

import torch
from torch import Tensor

from wms.core.constants import DEFAULT_LENGTH_M

from .contract import ACCEL_MIN_MPS2, MERGE_S1_M, SPEED_LIMIT_MPS

HEAD_SCHEMA = "wms_native_cav_head/1"
HEAD_PATH = Path(__file__).resolve().parents[3] / "configs" / "native_cav_head_v5.json"
FEATURE_NAMES = ("s_scaled", "v_scaled", "gap_scaled", "dv_scaled", "has_leader",
                 "merge_scaled", "lane_0", "lane_1", "lane_2")
HEAD_ACCEL_MAX_MPS2 = 1.2      # SUMO cav_typical accel (speedMode 27)
HEAD_ACCEL_MIN_MPS2 = ACCEL_MIN_MPS2


def _length(state: Mapping[str, Tensor], s: Tensor) -> Tensor:
    length = state.get("length_m", torch.full_like(s, DEFAULT_LENGTH_M))
    return torch.where(state["present"].bool(), length, torch.zeros_like(length))


def leader_gap_dv(state: Mapping[str, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
    """Same-lane nearest-leader bumper gap, speed difference (ego - leader) and presence.

    The geometry follows :func:`wms.control.counterfactual.idm_native_accel` exactly so the
    learned head and the legacy twin see the same leader.
    """
    present = state["present"].bool()
    lane = state["lane"].long()
    valid = present & (lane >= 0)
    s = torch.where(valid, state["s_m"], torch.zeros_like(state["s_m"]))
    v = torch.where(valid, state["v_mps"], torch.zeros_like(s)).clamp_min(0.0)
    length = _length(state, s)
    k = s.shape[-1]
    ds = s[:, None, :] - s[:, :, None]
    other = ~torch.eye(k, dtype=torch.bool, device=s.device)[None]
    ahead = valid[:, :, None] & valid[:, None, :] & (lane[:, :, None] == lane[:, None, :]) \
        & (ds >= 0) & other
    distance, index = torch.where(ahead, ds, torch.full_like(ds, math.inf)).min(-1)
    has = torch.isfinite(distance)
    gap = (torch.where(has, distance, torch.zeros_like(distance))
           - length.gather(-1, index)).clamp_min(0.1)
    dv = v - v.gather(-1, index)
    return torch.where(has, gap, torch.zeros_like(gap)), dv, has


def head_features(state: Mapping[str, Tensor]) -> Tensor:
    """[B,K,9] features of the learned native-CAV head (see ``FEATURE_NAMES``)."""
    valid = state["present"].bool() & (state["lane"] >= 0)
    s = torch.where(valid, state["s_m"], torch.zeros_like(state["s_m"]))
    v = torch.where(valid, state["v_mps"], torch.zeros_like(s)).clamp_min(0.0)
    gap, dv, has = leader_gap_dv(state)
    lane = state["lane"].long()
    onehot = torch.stack((lane == 0, lane == 1, lane == 2), -1).to(s.dtype)
    feats = torch.stack((
        s / 1000.0,
        v / SPEED_LIMIT_MPS,
        gap.clamp(0.0, 150.0) / 100.0,
        dv.clamp(-20.0, 20.0) / 20.0,
        has.to(s.dtype),
        (MERGE_S1_M - s) / 100.0,
    ), -1)
    return torch.cat((feats, onehot), -1)


def feature_matrix(s, v, a, lane, present, length=None):
    """NumPy twin of :func:`head_features` for the offline fit ([N,K] arrays -> [N,K,9])."""
    import numpy as np

    s = np.asarray(s, np.float32)
    v = np.asarray(v, np.float32)
    lane = np.asarray(lane, np.int64)
    present = np.asarray(present, bool)
    valid = present & (lane >= 0)
    s = np.where(valid, s, 0.0).astype(np.float32)
    v = np.where(valid, v, 0.0).clip(0.0).astype(np.float32)
    length = np.full_like(s, DEFAULT_LENGTH_M) if length is None else np.asarray(length, np.float32)
    length = np.where(valid, length, 0.0)
    n, k = s.shape
    ds = s[:, None, :] - s[:, :, None]
    eye = np.eye(k, dtype=bool)[None]
    ahead = (valid[:, :, None] & valid[:, None, :] & (lane[:, :, None] == lane[:, None, :])
             & (ds >= 0) & ~eye)
    dist = np.where(ahead, ds, np.inf)
    index = dist.argmin(-1)
    nearest = dist.min(-1)
    has = np.isfinite(nearest)
    gap = np.where(has, nearest - length[np.arange(n)[:, None], index], 0.0).clip(0.1)
    dv = v - v[np.arange(n)[:, None], index]
    gap = np.where(has, gap, 0.0)
    onehot = np.stack((lane == 0, lane == 1, lane == 2), -1).astype(np.float32)
    feats = np.stack((s / 1000.0, v / SPEED_LIMIT_MPS, np.clip(gap, 0, 150) / 100.0,
                      np.clip(dv, -20, 20) / 20.0, has.astype(np.float32),
                      (MERGE_S1_M - s) / 100.0), -1)
    return np.concatenate((feats, onehot), -1).astype(np.float32)


class NativeCavHead:
    """Loaded JSON weights; ``forward`` is differentiable and device-following."""

    def __init__(self, payload: Mapping, source: Optional[Path] = None):
        if int(payload.get("input_dim", -1)) != len(FEATURE_NAMES):
            raise ValueError("native-CAV head input_dim does not match FEATURE_NAMES")
        if list(payload.get("features", FEATURE_NAMES)) != list(FEATURE_NAMES):
            raise ValueError("native-CAV head feature order does not match this build")
        self.layers = [(torch.as_tensor(layer["w"], dtype=torch.float32),
                        torch.as_tensor(layer["b"], dtype=torch.float32))
                       for layer in payload["layers"]]
        self.activation = str(payload.get("activation", "tanh"))
        if self.activation != "tanh":
            raise ValueError("native-CAV head supports only tanh activations")
        self.accel_min = float(payload.get("accel_min", HEAD_ACCEL_MIN_MPS2))
        self.accel_max = float(payload.get("accel_max", HEAD_ACCEL_MAX_MPS2))
        self.source = source
        self.meta = dict(payload.get("fit", {}))
        self._device_layers: dict[str, list[tuple[Tensor, Tensor]]] = {}

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "NativeCavHead":
        path = Path(path or HEAD_PATH)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema") != HEAD_SCHEMA:
            raise ValueError(f"{path}: not a {HEAD_SCHEMA} file")
        return cls(payload, path)

    def parameters(self):
        for weight, bias in self.layers:
            yield weight
            yield bias

    def to(self, device) -> "NativeCavHead":
        self.layers = [(w.to(device), b.to(device)) for w, b in self.layers]
        self._device_layers.clear()
        return self

    def _for(self, device) -> list[tuple[Tensor, Tensor]]:
        key = str(device)
        if key not in self._device_layers:
            self._device_layers[key] = [(w.to(device), b.to(device)) for w, b in self.layers]
        return self._device_layers[key]

    def __call__(self, features: Tensor) -> Tensor:
        x = features
        layers = self._for(x.device)
        for index, (weight, bias) in enumerate(layers):
            x = x @ weight.transpose(0, 1) + bias
            if index < len(layers) - 1:
                x = torch.tanh(x)
        return x.squeeze(-1).clamp(self.accel_min, self.accel_max)


_CACHE: dict[str, NativeCavHead] = {}


def load_head(path: Optional[Path | str] = None) -> NativeCavHead:
    key = str(Path(path or HEAD_PATH).resolve())
    if key not in _CACHE:
        _CACHE[key] = NativeCavHead.load(Path(key))
    return _CACHE[key]


def head_native_accel(state: Mapping[str, Tensor], head: Optional[NativeCavHead] = None,
                      path: Optional[Path | str] = None) -> Tensor:
    """Learned native-CAV acceleration [B,K]; zero on absent / lane-less slots."""
    head = head if head is not None else load_head(path)
    valid = state["present"].bool() & (state["lane"] >= 0)
    accel = head(head_features(state))
    return torch.where(valid, accel, torch.zeros_like(accel))
