"""JAM-WINDOW OVERSAMPLING (ROUND-3 fix #2; every flag defaults OFF).

A *jam window* is a training window whose recorded HDV speed drops below
``--jam-speed-mps`` somewhere in its own future horizon (the recorded
trajectory).  Jam windows are then drawn ``--jam-oversample`` times more often
than the rest.  The per-window flag is computed once from the memory-mapped
pack windows, cached in a sidecar JSON with counts, and recorded in
``TRAIN_RECEIPT.json``.

``--jam-oversample 1`` (the default) is the OFF path: no scan runs, no sidecar
is written and the sampling weights are byte-identical to the pre-fix trainer.
The multiplier composes multiplicatively with the switching-source
``--event-oversample`` / ``--switch-mix-ratio`` weights.

TRAIN only: nothing here reads CAL / HOLDOUT / sealed / chunk-5 / J4B / INC1
bytes, an outcome, or a pack array other than one window's mmap slice at a time.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from wms.core.errors import TrainDataRefusal

JAM_SCHEMA = "wms_jam_windows/1"
DEFAULT_SPEED_MPS = 5.0
SPEED_FEATURE_INDEX = 4


class JamRefusal(TrainDataRefusal):
    """A jam-window contract violation (fail closed)."""


def record_hdv_min_speed(record: dict) -> tuple[float, int]:
    """``(min recorded HDV speed, n HDV observations)`` of one window record.

    HDVs are the present, non-CAV slots of ``states[..., SPEED_FEATURE_INDEX]``;
    a window with no observed HDV returns ``(+inf, 0)`` and is never a jam.
    """
    states = np.asarray(record["states"])
    if states.ndim != 3 or states.shape[-1] <= SPEED_FEATURE_INDEX:
        raise JamRefusal("jam scan needs states [T, K, F>=5]")
    present = np.asarray(record["present"], dtype=bool)
    if present.shape != states.shape[:2]:
        raise JamRefusal("jam scan: present shape differs from states")
    is_cav = record.get("is_cav")
    hdv = present if is_cav is None else (present & ~np.asarray(is_cav, dtype=bool))
    if hdv.shape != present.shape:
        raise JamRefusal("jam scan: is_cav shape differs from present")
    speed = states[..., SPEED_FEATURE_INDEX].astype(np.float64)
    valid = hdv & np.isfinite(speed)
    n = int(valid.sum())
    if n == 0:
        return float("inf"), 0
    return float(speed[valid].min()), n


def record_hdv_jam_fraction(record: dict, speed_mps: float) -> float:
    """INFRAV5: fraction of the recorded HDV observations of a window below ``speed_mps``."""
    states = np.asarray(record["states"])
    present = np.asarray(record["present"], dtype=bool)
    is_cav = record.get("is_cav")
    hdv = present if is_cav is None else (present & ~np.asarray(is_cav, dtype=bool))
    speed = states[..., SPEED_FEATURE_INDEX].astype(np.float64)
    valid = hdv & np.isfinite(speed)
    n = int(valid.sum())
    return float((speed[valid] < float(speed_mps)).sum() / n) if n else 0.0


def dataset_source_key(dataset: Dataset) -> str:
    """Stable content key of a dataset's window provenance (for the cache)."""
    receipt = getattr(dataset, "receipt", None) or {}
    payload = json.dumps(
        {"packs": receipt.get("packs"), "window_filter": receipt.get("window_filter"),
         "n_records": receipt.get("n_records"),
         "item_filter": receipt.get("item_filter"),
         "window_list": receipt.get("train_window_list") or receipt.get("window_list")},
        sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_document(speed_mps: float, source_key: str, flags: np.ndarray,
                    n_hdv_windows: int, n_hdv_scanned: int,
                    min_speed_mps: Optional[float]) -> dict:
    return {"schema": JAM_SCHEMA, "speed_mps": float(speed_mps),
            "source_key": str(source_key), "n_windows": int(flags.shape[0]),
            "n_jam": int(flags.sum()), "n_hdv_windows": int(n_hdv_windows),
            "n_hdv_scanned": int(n_hdv_scanned),
            "min_speed_mps": (None if min_speed_mps is None
                              else float(min_speed_mps)),
            "flags": [bool(x) for x in flags]}


_SCAN_STATE: dict = {}


def _scan_one(i: int) -> tuple[float, int, float]:
    dataset, speed, min_fraction = _SCAN_STATE["args"]
    loader = getattr(dataset, "_loader", None)
    fast = getattr(loader, "jam_view", None)
    record = fast(i) if callable(fast) else dataset[i]
    window_min, n_hdv = record_hdv_min_speed(record)
    frac = 0.0
    if math.isfinite(window_min) and window_min < speed and min_fraction > 0.0:
        frac = record_hdv_jam_fraction(record, speed)
    return window_min, n_hdv, frac


def _scan_windows(dataset: Dataset, speed: float, min_fraction: float):
    """Per-window ``(min HDV speed, n HDV obs, slow fraction)`` in index order.

    ``WMS_JAM_WORKERS`` > 1 (POSIX fork only) scans in that many forked processes; the
    per-window values and their order are identical to the serial scan."""
    import os
    workers = int(os.environ.get("WMS_JAM_WORKERS", "1") or 1)
    n = len(dataset)
    _SCAN_STATE["args"] = (dataset, speed, min_fraction)
    try:
        if workers > 1 and n > 1000 and os.name == "posix":
            import multiprocessing as mp
            with mp.get_context("fork").Pool(workers) as pool:
                yield from pool.imap(_scan_one, range(n), chunksize=64)
        else:
            for i in range(n):
                yield _scan_one(i)
    finally:
        _SCAN_STATE.pop("args", None)


def compute_jam_flags(dataset: Dataset, *, speed_mps: float,
                      cache_path: Optional[str | Path] = None,
                      source_key: Optional[str] = None, min_fraction: float = 0.0) -> dict:
    """Per-window jam flags + counts, cached in a sidecar JSON when requested.

    Reads one window at a time through the dataset loader (the A11 packs stay
    memory-mapped); a matching cache sidecar is reused without rescanning.
    ``min_fraction`` (INFRAV5, 0 = legacy any-observation rule) flags a window only when
    more than that fraction of its HDV observations are below ``speed_mps``.
    """
    speed = float(speed_mps)
    if not math.isfinite(speed) or speed <= 0.0:
        raise JamRefusal("--jam-speed-mps must be finite and positive")
    min_fraction = float(min_fraction)
    if not (math.isfinite(min_fraction) and 0.0 <= min_fraction < 1.0):
        raise JamRefusal("jam min_fraction must be in [0, 1)")
    key = str(source_key) if source_key is not None else dataset_source_key(dataset)
    n = len(dataset)
    cache_file = None if cache_path is None else Path(cache_path)
    hit = False
    if cache_file is not None and cache_file.is_file():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if (cached.get("schema") == JAM_SCHEMA
                and float(cached.get("speed_mps", -1.0)) == speed
                and float(cached.get("min_fraction", 0.0)) == min_fraction
                and str(cached.get("source_key")) == key
                and int(cached.get("n_windows", -1)) == n
                and len(cached.get("flags") or []) == n):
            flags = np.asarray(cached["flags"], dtype=bool)
            hit = True
            return {"flags": flags, "n_windows": n, "n_jam": int(flags.sum()),
                    "n_hdv_windows": int(cached.get("n_hdv_windows", 0)),
                    "n_hdv_scanned": int(cached.get("n_hdv_scanned", 0)),
                    "min_speed_mps": cached.get("min_speed_mps"),
                    "speed_mps": speed, "source_key": key,
                    "cache": (None if cache_file is None else str(cache_file)),
                    "cache_hit": hit}
    if not hit:
        flags = np.zeros(n, dtype=bool)
        n_hdv_windows = n_hdv_scanned = 0
        min_speed = None
        for i, (window_min, n_hdv, frac) in enumerate(_scan_windows(dataset, speed, min_fraction)):
            n_hdv_windows += int(n_hdv > 0)
            n_hdv_scanned += n_hdv
            if not math.isfinite(window_min):
                continue
            if min_speed is None or window_min < min_speed:
                min_speed = window_min
            if window_min < speed and (min_fraction <= 0.0 or frac > min_fraction):
                flags[i] = True
        if cache_file is not None:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            document = _cache_document(speed, key, flags, n_hdv_windows,
                                       n_hdv_scanned, min_speed)
            if min_fraction > 0.0:
                document["min_fraction"] = min_fraction
            cache_file.write_text(json.dumps(document, indent=1) + "\n",
                                  encoding="utf-8")
    return {"flags": flags, "n_windows": n, "n_jam": int(flags.sum()),
            "n_hdv_windows": int(n_hdv_windows), "n_hdv_scanned": int(n_hdv_scanned),
            "min_speed_mps": min_speed, "speed_mps": speed, "source_key": key,
            "cache": (None if cache_file is None else str(cache_file)),
            "cache_hit": hit}


def jam_receipt(report: dict) -> dict:
    """The compact TRAIN_RECEIPT block of a jam report (flags omitted)."""
    return {"schema": JAM_SCHEMA, "speed_mps": float(report["speed_mps"]),
            "oversample": float(report.get("oversample", 1.0)),
            "source_key": str(report["source_key"]),
            "n_windows": int(report["n_windows"]), "n_jam": int(report["n_jam"]),
            "n_hdv_windows": int(report["n_hdv_windows"]),
            "n_hdv_scanned": int(report["n_hdv_scanned"]),
            "min_speed_mps": report.get("min_speed_mps"),
            "cache": report.get("cache"), "cache_hit": bool(report.get("cache_hit"))}


class JamOversampledDataset(Dataset):
    """Single-source jam oversampling (no ``--switch-pack``).

    Delegates ``__len__``/``__getitem__``/``receipt`` and supplies the per-stage
    ``stage_weights`` the trainer already honours for mixed datasets, so jam
    windows are drawn ``oversample`` times as often.
    """

    def __init__(self, dataset: Dataset, flags: np.ndarray, *, oversample: float,
                 report: Optional[dict] = None):
        n = len(dataset)
        flags = np.asarray(flags, dtype=bool)
        if flags.shape != (n,):
            raise JamRefusal("jam flags must be one bool per window")
        if not math.isfinite(float(oversample)) or float(oversample) < 1.0:
            raise JamRefusal("--jam-oversample must be >= 1")
        self.dataset = dataset
        self.flags = flags
        self.oversample = float(oversample)
        base = dict(getattr(dataset, "receipt", {}) or {})
        block = jam_receipt({"oversample": self.oversample,
                             "speed_mps": (report or {}).get("speed_mps",
                                                             DEFAULT_SPEED_MPS),
                             "source_key": (report or {}).get("source_key", ""),
                             "n_windows": n, "n_jam": int(flags.sum()),
                             "n_hdv_windows": (report or {}).get("n_hdv_windows", 0),
                             "n_hdv_scanned": (report or {}).get("n_hdv_scanned", 0),
                             "min_speed_mps": (report or {}).get("min_speed_mps"),
                             "cache": (report or {}).get("cache"),
                             "cache_hit": (report or {}).get("cache_hit", False)})
        self.receipt = dict(base)
        self.receipt["jam"] = block

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        record = dict(self.dataset[index])
        record.setdefault("mix_source", "original")
        return record

    def stage_weights(self, stage_s: float) -> list[float]:
        if self.oversample <= 1.0:
            w = np.ones(len(self), dtype=np.float64)
        else:
            w = np.where(self.flags, self.oversample, 1.0).astype(np.float64)
        total = float(w.sum())
        return (w / total).tolist()

    def describe_stages(self, stages_s: Sequence[float]) -> list[dict]:
        n = len(self)
        return [{"stage_s": float(stage), "sources": {"original": {
            "share": 1.0, "n_items": n, "n_jam_items": int(self.flags.sum()),
            "jam_fraction_uniform": round(float(self.flags.sum()) / max(1, n), 4)}}}
            for stage in stages_s]


__all__ = ["JAM_SCHEMA", "DEFAULT_SPEED_MPS", "JamRefusal", "record_hdv_min_speed",
           "dataset_source_key", "compute_jam_flags", "jam_receipt",
           "JamOversampledDataset"]


JamOversampler = JamOversampledDataset  # architecture name (docs/ARCHITECTURE_V4.md 2.3)
