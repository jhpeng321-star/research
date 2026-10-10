"""Per-vehicle driver-parameter ground-truth sidecar + recovery evaluation (WM-S2).

The heterogeneous-driver generator (``traffic_shepherd.hetero_drivers``, being
defined on branch ``wms/hetpack``) emits, per vehicle, the 11 SUMO IDM/LC2013
parameters plus a driver class.  This module loads that ground truth from an
optional sidecar path and scores how well the WM-S2 driver encoder recovers it.

Contract
--------
* The sidecar is EVALUATION-ONLY.  It is never passed into the training loss,
  never reaches the encoder as an input, and never influences an optimizer step.
* The path is optional: ``load_driver_sidecar(None)`` returns ``None`` and every
  consumer treats a missing sidecar as "no recovery report".
* Accepted formats (the loader infers from the suffix):

  - ``.csv``  hetdata truth table (``hetdata_truth.csv``): a header row with
    ``vehicle_id``, ``driver_class`` and the 11 :data:`PARAM_ORDER` columns.
  - ``.json`` / ``.jsonl``  a list of row objects with the same fields.
  - ``.npz``  slot-aligned arrays ``params`` ``[N,11]``, ``class`` (int or str)
    and optional ``vehicle_ids``.
  - ``.npy``  a bare ``[N,11]`` parameter array (no ids/classes).
  - a hetpack driver bundle: a directory with ``driver_params.npy``
    ``[n_items, k_max, 11]`` (+ optional ``driver_class.npy``
    ``[n_items, k_max]``, ``global_positions.npy`` and ``vehicle_params.json``),
    or a bare ``.npy``/``.npz`` carrying a 3-D ``params`` array.  Use
    :func:`load_driver_bundle` / :meth:`DriverBundle.item_sidecar` to obtain the
    slot-aligned ``[k_max, 11]`` view of one packed item.
  - the dense per-run truth directory produced by :mod:`wms.dense_truth`
    (``DENSE_TRUTH_RECEIPT.json`` + ``runs.jsonl``): :func:`load_driver_sidecar`
    returns a :class:`~wms.dense_truth.DenseDriverTruth` read view.

Recovery scoring maps only the encoder parameters that are directly comparable
to a SUMO vType attribute (:data:`ENCODER_TO_SIDECAR`) and reports per-parameter
MAE/RMSE/bias plus a calibration summary of the encoder's predicted uncertainty.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from .driver_encoder import PARAMETER_NAMES

#: The 11 ground-truth parameters, in the generator's fixed draw/emit order.
PARAM_ORDER = ("speed_factor", "tau_s", "min_gap_m", "accel_mps2", "decel_mps2",
               "delta", "lc_strategic", "lc_cooperative", "lc_speed_gain",
               "lc_assertive", "impatience")
#: Driver classes (the audited 0.10 / 0.70 / 0.20 aggressive/normal/cautious mix).
DRIVER_CLASSES = ("aggressive_car", "normal_car", "cautious_car")

#: Encoder parameter -> directly comparable sidecar parameter.  Parameters with
#: no vType counterpart (politeness, mobil_threshold, merge_urgency) or that need
#: a speed-limit rescale (v0) are intentionally left out of the error summary.
ENCODER_TO_SIDECAR = {
    "time_headway": "tau_s",
    "min_gap": "min_gap_m",
    "accel": "accel_mps2",
    "decel": "decel_mps2",
}


@dataclass(frozen=True)
class DriverSidecar:
    """Slot/vehicle-aligned ground truth (order follows :data:`PARAM_ORDER`)."""

    params: np.ndarray                       # [N, 11] float
    classes: np.ndarray                      # [N] int index into class_names (-1 unknown)
    vehicle_ids: Optional[tuple] = None      # [N] hashable keys, or None
    param_order: tuple = PARAM_ORDER
    class_names: tuple = DRIVER_CLASSES

    def __post_init__(self) -> None:
        params = np.asarray(self.params, dtype=np.float64)
        classes = np.asarray(self.classes, dtype=np.int64)
        if params.ndim != 2 or params.shape[1] != len(PARAM_ORDER):
            raise ValueError(f"params must be [N, {len(PARAM_ORDER)}]")
        if classes.shape != (params.shape[0],):
            raise ValueError("classes must be [N]")
        if self.vehicle_ids is not None and len(self.vehicle_ids) != params.shape[0]:
            raise ValueError("vehicle_ids must match params rows")
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "classes", classes)

    @property
    def n(self) -> int:
        return int(self.params.shape[0])


def _class_index(value, class_names: Sequence[str]) -> int:
    if value is None or value == "":
        return -1
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    return list(class_names).index(text) if text in class_names else -1


def _rows_to_sidecar(rows: Sequence[Mapping]) -> DriverSidecar:
    if not rows:
        raise ValueError("sidecar contains no rows")
    params = np.asarray([[float(row[name]) for name in PARAM_ORDER] for row in rows],
                        dtype=np.float64)
    classes = np.asarray([_class_index(row.get("driver_class"), DRIVER_CLASSES)
                          for row in rows], dtype=np.int64)
    ids = [row.get("vehicle_id") for row in rows]
    vehicle_ids = tuple(ids) if all(i is not None for i in ids) else None
    return DriverSidecar(params=params, classes=classes, vehicle_ids=vehicle_ids)


def _load_csv(path: Path) -> DriverSidecar:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return _rows_to_sidecar(rows)


def _load_json(path: Path) -> DriverSidecar:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        rows = json.loads(text)
    else:  # JSONL
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list):
        raise ValueError("JSON sidecar must be a list of row objects")
    return _rows_to_sidecar(rows)


def _load_npz(path: Path) -> DriverSidecar:
    with np.load(path, allow_pickle=False) as data:
        if "params" not in data:
            raise ValueError("npz sidecar must contain a 'params' array")
        params = np.asarray(data["params"], dtype=np.float64)
        raw_classes = data["class"] if "class" in data else data.get("classes")
        if raw_classes is None:
            classes = np.full(params.shape[0], -1, dtype=np.int64)
        else:
            classes = np.asarray([_class_index(v, DRIVER_CLASSES)
                                  for v in np.asarray(raw_classes).reshape(-1)],
                                 dtype=np.int64)
        ids = data["vehicle_ids"] if "vehicle_ids" in data else None
        vehicle_ids = None if ids is None else tuple(np.asarray(ids).reshape(-1).tolist())
    return DriverSidecar(params=params, classes=classes, vehicle_ids=vehicle_ids)


def load_driver_sidecar(path: Optional[str | Path]) -> Optional[DriverSidecar]:
    """Load a driver-parameter sidecar; ``None`` path means "no sidecar"."""
    if path is None or str(path).strip() == "":
        return None
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"driver sidecar not found: {resolved}")
    if resolved.is_dir():
        if _is_dense_truth_dir(resolved):
            from .dense_truth import DenseDriverTruth

            return DenseDriverTruth(resolved)
        if _bundle_files(resolved):
            return _load_bundle_dir(resolved)
        raise ValueError(f"{resolved} is neither a driver bundle nor dense truth")
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return _load_csv(resolved)
    if suffix in (".json", ".jsonl"):
        return _load_json(resolved)
    if suffix == ".npz":
        with np.load(resolved, allow_pickle=False) as data:
            if np.asarray(data["params"]).ndim == 3:
                return load_driver_bundle(resolved)
        return _load_npz(resolved)
    if suffix == ".npy":
        params = np.asarray(np.load(resolved, allow_pickle=False), dtype=np.float64)
        if params.ndim == 3:
            return DriverBundle(params=params,
                                classes=np.full(params.shape[:2], -1, dtype=np.int64))
        return DriverSidecar(params=params,
                             classes=np.full(params.shape[0], -1, dtype=np.int64))
    raise ValueError(f"unsupported driver-sidecar suffix: {suffix!r}")


def align_to_slots(sidecar: DriverSidecar, slot_keys: Sequence
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align a per-vehicle sidecar to slot order by key.

    Returns ``(params [S,11], classes [S], present [S])``; ``present`` is False
    for slots whose key is absent from the sidecar (their rows are NaN / -1).
    """
    if sidecar.vehicle_ids is None:
        raise ValueError("sidecar has no vehicle_ids; supply a slot-aligned sidecar")
    lookup = {key: i for i, key in enumerate(sidecar.vehicle_ids)}
    n_slots = len(slot_keys)
    params = np.full((n_slots, len(PARAM_ORDER)), np.nan, dtype=np.float64)
    classes = np.full(n_slots, -1, dtype=np.int64)
    present = np.zeros(n_slots, dtype=bool)
    for slot, key in enumerate(slot_keys):
        row = lookup.get(key)
        if row is None:
            continue
        params[slot] = sidecar.params[row]
        classes[slot] = sidecar.classes[row]
        present[slot] = True
    return params, classes, present


@dataclass(frozen=True)
class DriverBundle:
    """An ``[n_items, k_max, 11]`` hetpack driver bundle.

    One row per packed item; columns are the item's slots at its snapshot tick
    (the same alignment the pack windows use).  ``item_sidecar`` returns the
    slot-aligned :class:`DriverSidecar` view for one item so the positional
    recovery path in :func:`driver_recovery` applies unchanged.
    """

    params: np.ndarray                       # [n_items, k_max, 11] float
    classes: np.ndarray                      # [n_items, k_max] int (-1 unknown)
    global_positions: Optional[np.ndarray] = None   # [n_items] pack positions
    vehicle_params: Optional[Mapping] = None        # global key -> row (metadata)
    param_order: tuple = PARAM_ORDER
    class_names: tuple = DRIVER_CLASSES

    def __post_init__(self) -> None:
        params = np.asarray(self.params, dtype=np.float64)
        classes = np.asarray(self.classes, dtype=np.int64)
        if params.ndim != 3 or params.shape[2] != len(PARAM_ORDER):
            raise ValueError(f"bundle params must be [n_items, k_max, {len(PARAM_ORDER)}]")
        if classes.shape != params.shape[:2]:
            raise ValueError("bundle classes must be [n_items, k_max]")
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "classes", classes)

    @property
    def n_items(self) -> int:
        return int(self.params.shape[0])

    @property
    def k_max(self) -> int:
        return int(self.params.shape[1])

    def item_sidecar(self, index: int) -> DriverSidecar:
        i = int(index)
        if not (0 <= i < self.n_items):
            raise IndexError(f"bundle item {i} outside [0, {self.n_items})")
        return DriverSidecar(params=self.params[i], classes=self.classes[i])

    def item_index_of(self, position: int) -> int:
        """Item whose pack manifest position is ``position`` (if recorded)."""
        if self.global_positions is None:
            raise ValueError("bundle has no global_positions mapping")
        matches = np.flatnonzero(np.asarray(self.global_positions) == int(position))
        if matches.size != 1:
            raise ValueError(f"position {position} does not map to exactly one item")
        return int(matches[0])


def _bundle_files(root: Path) -> bool:
    return (root / "driver_params.npy").is_file()


def _load_bundle_dir(root: Path) -> DriverBundle:
    params = np.load(root / "driver_params.npy", allow_pickle=False)
    class_path = root / "driver_class.npy"
    classes = (np.load(class_path, allow_pickle=False) if class_path.is_file()
               else np.full(params.shape[:2], -1, dtype=np.int64))
    gp_path = root / "global_positions.npy"
    positions = np.load(gp_path, allow_pickle=False) if gp_path.is_file() else None
    vp_path = root / "vehicle_params.json"
    vehicle_params = None
    if vp_path.is_file():
        vehicle_params = json.loads(vp_path.read_text(encoding="utf-8"))
    return DriverBundle(params=params, classes=classes,
                        global_positions=positions, vehicle_params=vehicle_params)


def load_driver_bundle(path: str | Path) -> DriverBundle:
    """Load a hetpack ``[n_items, k_max, 11]`` driver bundle (dir, .npy, .npz)."""
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"driver bundle not found: {resolved}")
    if resolved.is_dir():
        if not _bundle_files(resolved):
            raise ValueError(f"{resolved} is not a hetpack driver bundle")
        return _load_bundle_dir(resolved)
    suffix = resolved.suffix.lower()
    if suffix == ".npz":
        with np.load(resolved, allow_pickle=False) as data:
            params = np.asarray(data["params"], dtype=np.float64)
            raw = data["class"] if "class" in data else data.get("classes")
            classes = (np.asarray([_class_index(v, DRIVER_CLASSES)
                                   for v in np.asarray(raw).reshape(-1)], dtype=np.int64)
                       .reshape(params.shape[:2]) if raw is not None
                       else np.full(params.shape[:2], -1, dtype=np.int64))
    elif suffix == ".npy":
        params = np.asarray(np.load(resolved, allow_pickle=False), dtype=np.float64)
        classes = np.full(params.shape[:2], -1, dtype=np.int64)
    else:
        raise ValueError(f"unsupported driver-bundle suffix: {suffix!r}")
    if params.ndim != 3:
        raise ValueError("driver bundle params must be 3-D [n_items, k_max, 11]")
    return DriverBundle(params=params, classes=classes)


def _is_dense_truth_dir(path: Path) -> bool:
    return (path / "DENSE_TRUTH_RECEIPT.json").is_file()


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), dependency-free."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0.0 else float("nan")


def _nearest_class(values: np.ndarray, names: Sequence[str],
                   centres: Mapping[str, Mapping[str, float]]) -> np.ndarray:
    """Nearest class centre in the mapped-parameter space (sidecar units)."""
    usable = [name for name in names
              if centres and all(name in centres[c] for c in centres)]
    if not usable:
        return np.full(values.shape[0], -1, dtype=np.int64)
    out = np.full(values.shape[0], -1, dtype=np.int64)
    centre_matrix = np.asarray([[centres[c][name] for name in usable]
                                for c in DRIVER_CLASSES], dtype=np.float64)
    for i, row in enumerate(values):
        if not np.all(np.isfinite(row)):
            continue
        out[i] = int(np.argmin(np.linalg.norm(centre_matrix - row, axis=1)))
    return out


def driver_recovery(pred_values: np.ndarray, pred_uncertainty: Optional[np.ndarray],
                    truth_params: np.ndarray, truth_classes: Optional[np.ndarray] = None,
                    *, present: Optional[np.ndarray] = None,
                    param_order: Sequence[str] = PARAM_ORDER,
                    class_centres: Optional[Mapping] = None,
                    ) -> dict:
    """Per-parameter error and predicted-uncertainty calibration.

    ``pred_values`` / ``pred_uncertainty`` are ``[..., len(PARAMETER_NAMES)]`` in
    encoder order; ``truth_params`` is ``[..., 11]`` in ``param_order``.  Only the
    :data:`ENCODER_TO_SIDECAR` parameters are scored.  Nothing here is
    differentiable or connected to the training loss.
    """
    values = np.asarray(pred_values, dtype=np.float64)
    truth = np.asarray(truth_params, dtype=np.float64)
    if values.shape[:-1] != truth.shape[:-1]:
        raise ValueError("pred_values and truth_params must share leading shape")
    if values.shape[-1] != len(PARAMETER_NAMES):
        raise ValueError(f"pred_values last dim must be {len(PARAMETER_NAMES)}")
    if truth.shape[-1] != len(param_order):
        raise ValueError(f"truth_params last dim must be {len(param_order)}")
    uncertainty = (None if pred_uncertainty is None
                   else np.asarray(pred_uncertainty, dtype=np.float64))
    flat_values = values.reshape(-1, values.shape[-1])
    flat_truth = truth.reshape(-1, truth.shape[-1])
    flat_unc = None if uncertainty is None else uncertainty.reshape(-1, uncertainty.shape[-1])
    if present is None:
        mask = np.ones(flat_values.shape[0], dtype=bool)
    else:
        mask = np.asarray(present, dtype=bool).reshape(-1)
    mask = mask & np.all(np.isfinite(flat_truth), axis=-1) & np.all(np.isfinite(flat_values), axis=-1)

    per_parameter: dict[str, dict] = {}
    mapped_encoder = [name for name in PARAMETER_NAMES if name in ENCODER_TO_SIDECAR]
    mapped_truth = [param_order.index(ENCODER_TO_SIDECAR[name]) for name in mapped_encoder]
    mapped_pred = [PARAMETER_NAMES.index(name) for name in mapped_encoder]
    for name, t_idx, p_idx in zip(mapped_encoder, mapped_truth, mapped_pred):
        errors = (flat_values[mask, p_idx] - flat_truth[mask, t_idx])
        entry: dict[str, float] = {
            "n": int(errors.size),
            "mae": float(np.mean(np.abs(errors))) if errors.size else float("nan"),
            "rmse": float(np.sqrt(np.mean(errors ** 2))) if errors.size else float("nan"),
            "bias": float(np.mean(errors)) if errors.size else float("nan"),
            "rank_correlation": _spearman(flat_values[mask, p_idx],
                                          flat_truth[mask, t_idx]),
            "truth_name": ENCODER_TO_SIDECAR[name],
        }
        if flat_unc is not None and errors.size:
            sigma = flat_unc[mask, p_idx]
            entry["uncertainty_mean"] = float(np.mean(sigma))
            entry["coverage_1sigma"] = float(np.mean(np.abs(errors) <= sigma))
            entry["coverage_2sigma"] = float(np.mean(np.abs(errors) <= 2.0 * sigma))
            entry["coverage_3sigma"] = float(np.mean(np.abs(errors) <= 3.0 * sigma))
            err_std = float(np.std(errors))
            entry["error_std"] = err_std
            entry["calibration_ratio"] = (float(np.mean(sigma)) / err_std
                                          if err_std > 0.0 else float("nan"))
        per_parameter[name] = entry

    class_report = None
    if truth_classes is not None and class_centres:
        truth_idx = np.asarray(truth_classes, dtype=np.int64).reshape(-1)[mask]
        usable_names = [name for name in mapped_encoder]
        centre_slice = {
            cls: {name: class_centres[cls][ENCODER_TO_SIDECAR[name]]
                  for name in usable_names}
            for cls in DRIVER_CLASSES if cls in class_centres
        }
        pred_idx = _nearest_class(flat_values[mask][:, mapped_pred], usable_names,
                                  centre_slice)
        valid = (truth_idx >= 0) & (pred_idx >= 0)
        class_report = {
            "n": int(valid.sum()),
            "accuracy": (float(np.mean(pred_idx[valid] == truth_idx[valid]))
                         if valid.any() else float("nan")),
            "classes": list(DRIVER_CLASSES),
        }
    per_class = None
    if truth_classes is not None:
        truth_idx_all = np.asarray(truth_classes, dtype=np.int64).reshape(-1)
        per_class = {}
        for ci, cname in enumerate(DRIVER_CLASSES):
            sel = mask & (truth_idx_all == ci)
            entry_cls: dict = {"n": int(sel.sum())}
            for name, t_idx, p_idx in zip(mapped_encoder, mapped_truth, mapped_pred):
                errs = flat_values[sel, p_idx] - flat_truth[sel, t_idx]
                entry_cls[name] = (float(np.mean(np.abs(errs))) if errs.size
                                   else float("nan"))
            per_class[cname] = entry_cls
    return {"n_scored": int(mask.sum()), "per_parameter": per_parameter,
            "class": class_report, "per_class": per_class}


def _describe(sidecar: DriverSidecar) -> dict:
    out: dict = {"n": sidecar.n, "param_order": list(sidecar.param_order),
                 "class_names": list(sidecar.class_names)}
    for i, name in enumerate(PARAM_ORDER):
        column = sidecar.params[:, i]
        out[name] = {"min": float(np.min(column)), "median": float(np.median(column)),
                     "max": float(np.max(column))}
    known = sidecar.classes >= 0
    out["class_counts"] = {cls: int(np.sum(sidecar.classes == i))
                           for i, cls in enumerate(sidecar.class_names)}
    out["class_unknown"] = int(np.sum(~known))
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Driver-parameter sidecar (WM-S2)")
    parser.add_argument("sidecar", nargs="?", default=None,
                        help="path to hetdata truth CSV/JSON/JSONL/NPZ/NPY (optional)")
    args = parser.parse_args(argv)
    sidecar = load_driver_sidecar(args.sidecar)
    if sidecar is None:
        print(json.dumps({"sidecar": None, "status": "absent"}, indent=2))
        return 0
    print(json.dumps({"sidecar": str(args.sidecar), "status": "loaded",
                      **_describe(sidecar)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
