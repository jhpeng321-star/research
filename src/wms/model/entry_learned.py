"""Learned entrance insertion model (V10INSERT, default off).

Two small feed-forward networks fitted to *recorded SUMO entrants* on TRAIN-val data
(``merge_calib/data/{v5_native,genrand_plain}``):

* ``speed`` - the insertion speed ``v0`` (m/s) of a SUMO entrant as a function of the
  observable entrance state (bumper gap to the nearest downstream vehicle in the entry
  lane, that vehicle's speed, the entrance backlog, the demand rate, the lane speed cap
  and the entrance id);
* ``hazard`` - the per-step acceptance probability of a *waiting* entrant (an entrance
  with a positive backlog), fitted on the same state and calibrated per entrance so the
  insertion rate over time matches SUMO.

The networks are exported to ``entry_learned.json`` next to this module; inference is
pure numpy (no torch, no learned rule is evaluated at import time).  The switch is
``PhysicsConfig.entry_learned`` (default ``False`` = the hand-set ``sumo_like_v2``
insertion rule, bit-identical to the pre-V10INSERT behaviour).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

#: Observable features, in the order the exported networks expect them.
FEATURE_NAMES = (
    "gap_50",
    "sqrt_gap_5",
    "gap_per_step",
    "v_lead_10",
    "cap_10",
    "entrance",
    "free_gate",
    "free_gate_cap_10",
    "free_gate_v_lead_10",
    "queued_gate",
    "exp_gap_1",
    "exp_gap_2",
    "exp_gap_5",
    "exp_gap_15",
    "exp_gap_2_v_lead_10",
    "log_backlog_3",
    "demand",
)
N_FEATURES = len(FEATURE_NAMES)
SCHEMA = "wms_entry_learned/1"
DEFAULT_MODEL_PATH = Path(__file__).with_name("entry_learned.json")


def _broadcast(gap, v_lead, backlog, demand, cap, entrance):
    gap = np.asarray(gap, dtype=np.float64)
    v_lead = np.asarray(v_lead, dtype=np.float64)
    backlog = np.asarray(backlog, dtype=np.float64)
    demand = np.asarray(demand, dtype=np.float64)
    cap = np.asarray(cap, dtype=np.float64)
    entrance = np.asarray(entrance, dtype=np.float64)
    gap, v_lead, backlog, demand, cap, entrance = np.broadcast_arrays(
        gap, v_lead, backlog, demand, cap, entrance)
    g = np.clip(gap, 0.0, 250.0)
    vl = np.clip(v_lead, 0.0, 45.0)
    bl = np.clip(backlog, 0.0, 1000.0)
    dm = np.clip(demand, 0.0, 5.0)
    cp = np.clip(cap, 0.0, 45.0)
    return g, vl, bl, dm, cp, np.clip(entrance, 0.0, 1.0)


def state_features(gap, v_lead, backlog, demand, cap, entrance) -> np.ndarray:
    """``[..., N_FEATURES]`` observable entrance-state features (shared by fit and rollout)."""
    g, vl, bl, dm, cp, en = _broadcast(gap, v_lead, backlog, demand, cap, entrance)
    e1, e2, e5, e15 = np.exp(-g), np.exp(-g / 2.0), np.exp(-g / 5.0), np.exp(-g / 15.0)
    # Dimensionless physical state: one physics step at the lane cap is the free-flow scale.
    step_m = np.maximum(cp * 0.5, 1.0)
    gap_per_step = np.clip(g / step_m, 0.0, 3.0)
    free_gate = 1.0 / (1.0 + np.exp(-4.0 * (gap_per_step - 1.0)))
    return np.stack([
        g / 50.0,
        np.sqrt(g) / 5.0,
        gap_per_step,
        vl / 10.0,
        cp / 10.0,
        en,
        free_gate,
        free_gate * cp / 10.0,
        free_gate * vl / 10.0,
        1.0 - free_gate,
        e1,
        e2,
        e5,
        e15,
        e2 * vl / 10.0,
        np.log1p(bl) / 3.0,
        dm,
    ], axis=-1)


@dataclass
class _MLP:
    """One-hidden-layer tanh network with an exported (numpy) forward pass."""

    mean: np.ndarray
    scale: np.ndarray
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: float
    out_scale: float = 1.0
    out_shift: float = 0.0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "_MLP":
        return cls(mean=np.asarray(raw["mean"], np.float64),
                   scale=np.asarray(raw["scale"], np.float64),
                   w1=np.asarray(raw["w1"], np.float64),
                   b1=np.asarray(raw["b1"], np.float64),
                   w2=np.asarray(raw["w2"], np.float64),
                   b2=float(raw["b2"]),
                   out_scale=float(raw.get("out_scale", 1.0)),
                   out_shift=float(raw.get("out_shift", 0.0)))

    def to_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "w1": self.w1.tolist(), "b1": self.b1.tolist(),
                "w2": self.w2.tolist(), "b2": float(self.b2),
                "out_scale": float(self.out_scale), "out_shift": float(self.out_shift)}

    def forward(self, features: np.ndarray) -> np.ndarray:
        x = (np.asarray(features, np.float64).reshape(-1, self.mean.size) - self.mean) / self.scale
        hidden = np.tanh(x @ self.w1 + self.b1)
        return (hidden @ self.w2 + self.b2) * self.out_scale + self.out_shift


@dataclass
class LearnedEntryModel:
    """Fitted insertion speed and per-step acceptance hazard (see the module doc)."""

    speed: _MLP
    hazard: _MLP
    hazard_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))
    speed_clip: tuple[float, float] = (0.0, 35.0)
    hazard_clip: tuple[float, float] = (1.0e-4, 0.999)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "LearnedEntryModel":
        if raw.get("schema") != SCHEMA:
            raise ValueError(f"entry-learned model schema must be {SCHEMA!r}, "
                             f"got {raw.get('schema')!r}")
        if list(raw.get("features", [])) != list(FEATURE_NAMES):
            raise ValueError("entry-learned model features do not match this code")
        return cls(speed=_MLP.from_dict(raw["speed"]), hazard=_MLP.from_dict(raw["hazard"]),
                   hazard_offset=np.asarray(raw.get("hazard_offset", [0.0, 0.0]), np.float64),
                   speed_clip=tuple(raw.get("speed_clip", (0.0, 35.0))),
                   hazard_clip=tuple(raw.get("hazard_clip", (1.0e-4, 0.999))),
                   metadata=dict(raw.get("metadata") or {}))

    @classmethod
    def from_json(cls, path: str | Path = DEFAULT_MODEL_PATH) -> "LearnedEntryModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "features": list(FEATURE_NAMES),
                "speed": self.speed.to_dict(), "hazard": self.hazard.to_dict(),
                "hazard_offset": np.asarray(self.hazard_offset, np.float64).tolist(),
                "speed_clip": list(self.speed_clip), "hazard_clip": list(self.hazard_clip),
                "metadata": self.metadata}

    def insertion_speed(self, gap, v_lead, backlog, demand, cap, entrance) -> np.ndarray:
        raw = self.speed.forward(state_features(gap, v_lead, backlog, demand, cap, entrance))
        lo, hi = self.speed_clip
        return np.clip(raw, float(lo), float(hi))

    def acceptance_hazard(self, gap, v_lead, backlog, demand, cap, entrance) -> np.ndarray:
        en = np.asarray(entrance, np.int64).reshape(-1)
        offset = np.asarray(self.hazard_offset, np.float64)[np.clip(en, 0, 1)]
        logit = self.hazard.forward(
            state_features(gap, v_lead, backlog, demand, cap, entrance)) + offset
        lo, hi = self.hazard_clip
        return np.clip(1.0 / (1.0 + np.exp(-logit)), float(lo), float(hi))


def load_learned_entry(path: str | Path | None = None) -> LearnedEntryModel:
    """Load the shipped fit (or an explicit JSON path); raises when it is absent."""
    target = Path(path) if path is not None else DEFAULT_MODEL_PATH
    if not target.is_file():
        raise FileNotFoundError(
            f"physics.entry_learned is on but the fitted model is missing: {target}")
    return LearnedEntryModel.from_json(target)
