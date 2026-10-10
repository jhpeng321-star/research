"""Honest fixed baselines for the WM-N1 CAL-tune scorer (E9, owner option c).

Owner decision (verbatim): ``cav-motion teacher; comparator option (c)``.
Option (c) scores model A against fixed, non-tuned baselines on the registered
paired-dV label:

* :class:`ZeroEffectBaseline` -- paired dV = 0 as a point mass (CRPS = |dY|, so
  it is reported but its sign is a genuine tie and is reported UNDEFINED);
* :class:`TrainClimatologyBaseline` -- the per-``(contrast, tau)`` empirical
  distribution of paired dV built from TRAIN labels by
  :mod:`evalh.scorer.build_climatology`, as the 19 registered
  ``evalh.protocol.QUANTILE_LEVELS``.

Both predict paired-dV quantiles DIRECTLY (``dv_quantiles``), so the metric loop
can consume them alongside model A's per-row level quantiles.  Everything fails
closed: the climatology verifies its file SHA-256 against an explicit pin and
refuses a missing ``(contrast, tau)`` cell -- there is NO zero substitution.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from evalh.protocol import N_QUANTILES, QUANTILE_LEVELS

CLIMATOLOGY_SCHEMA = "wm_n1_train_climatology/1"


class BaselineRefusal(RuntimeError):
    """A baseline input is absent, unverified, or covers no requested cell."""


@runtime_checkable
class DirectDVBaseline(Protocol):
    """A model that predicts paired-dV quantiles directly for whole pairs."""

    name: str
    #: ``False`` when the model's sign is a genuine tie (e.g. zero effect) and
    #: sign accuracy is undefined rather than a measured value.
    sign_defined: bool

    def dv_quantiles(self, pairs: Sequence) -> np.ndarray:
        ...


def _field(pair, name, default=None):
    if isinstance(pair, dict):
        return pair.get(name, default)
    return getattr(pair, name, default)


class ZeroEffectBaseline:
    """Paired dV is exactly zero: every quantile is ``0.0`` (a point mass)."""

    name = "zero"
    sign_defined = False

    def __init__(self) -> None:
        self.quantile_levels = np.asarray(QUANTILE_LEVELS, dtype=float)

    def dv_quantiles(self, pairs: Sequence) -> np.ndarray:
        return np.zeros((len(pairs), N_QUANTILES), dtype=float)


class TrainClimatologyBaseline:
    """TRAIN per-``(contrast, tau)`` paired-dV empirical quantiles."""

    name = "train_climatology"
    sign_defined = True

    def __init__(self, path, *, expected_sha256: str,
                 levels=QUANTILE_LEVELS) -> None:
        if not isinstance(expected_sha256, str) or not expected_sha256.strip():
            raise BaselineRefusal(
                "train_climatology requires an explicit expected_sha256 pin")
        pin = expected_sha256.strip().lower()
        raw = Path(path).read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != pin:
            raise BaselineRefusal(
                f"climatology {path} sha256 {sha} != pinned {pin}; refusing")
        doc = json.loads(raw.decode("utf-8"))
        if doc.get("schema") != CLIMATOLOGY_SCHEMA:
            raise BaselineRefusal(
                f"climatology schema {doc.get('schema')!r} != "
                f"{CLIMATOLOGY_SCHEMA!r}")
        lv = np.asarray(levels, dtype=float)
        if lv.shape != (N_QUANTILES,):
            raise BaselineRefusal(
                f"levels must be the {N_QUANTILES} registered quantiles")
        self.path = str(path)
        self.sha256 = sha
        self.quantile_levels = lv
        self._cells: dict = {}
        for contrast, by_tau in (doc.get("cells") or {}).items():
            for tau_key, cell in by_tau.items():
                q = np.asarray(cell["quantiles"], dtype=float)
                if q.shape != (N_QUANTILES,):
                    raise BaselineRefusal(
                        f"climatology cell {contrast}@{tau_key} has "
                        f"{q.shape} quantiles, expected ({N_QUANTILES},)")
                self._cells[(str(contrast), round(float(tau_key), 6))] = q

    def available(self) -> list:
        return sorted(self._cells)

    def dv_quantiles(self, pairs: Sequence) -> np.ndarray:
        out = np.empty((len(pairs), N_QUANTILES), dtype=float)
        for i, pair in enumerate(pairs):
            contrast = str(_field(pair, "contrast"))
            tau = round(float(_field(pair, "tau_s")), 6)
            key = (contrast, tau)
            q = self._cells.get(key)
            if q is None:
                raise BaselineRefusal(
                    f"train_climatology has no cell for contrast "
                    f"{contrast!r} at tau {tau:g}s; refusing (no zero "
                    "substitution)")
            out[i] = q
        return out


def make_comparator(name, *, climatology=None, climatology_sha256=None):
    """Build one registered comparator by name (fail closed on unknown)."""
    key = str(name).strip().lower()
    if key == "zero":
        return ZeroEffectBaseline()
    if key == "train_climatology":
        if not climatology:
            raise BaselineRefusal(
                "comparator 'train_climatology' requires --climatology")
        return TrainClimatologyBaseline(
            climatology, expected_sha256=climatology_sha256)
    raise BaselineRefusal(
        f"unknown comparator {name!r}; the registered set is "
        "{'zero', 'train_climatology'}")
