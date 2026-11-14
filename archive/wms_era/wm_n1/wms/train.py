"""WM-S multi-step TRAIN-only trainer and pack adapters.

This module intentionally does not import or modify ``train.idea_a``.  It
accepts trajectory packs directly, while also accepting an in-memory batch in
the same ``states [B,T+1,K,7]`` contract used by the CPU tests.  A trajectory
pack may be an ``npz`` or a directory containing ``trajectory.npz``/``states``;
the ordinary A11 packed snapshot arrays are refused when no sequence metadata
is available rather than silently training a one-step surrogate.
"""
from __future__ import annotations

import functools
import json
import hashlib
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_WM_ROOT = Path(__file__).resolve().parents[1]
if str(_WM_ROOT) not in sys.path:
    sys.path.insert(0, str(_WM_ROOT))

from .driver_encoder import PARAMETER_NAMES  # noqa: E402
from .history import history_spec, long_history_features  # noqa: E402
from .action_timeline import ActionTimeline, CONTROLLERS, SCHEMA  # noqa: E402
from .driver_sidecar import align_to_slots, driver_recovery  # noqa: E402
from .lane_head import balanced_lane_loss  # noqa: E402
from .model import (CONTROLLER_VOCAB, N_CONTROLLERS, WmSModel, _state_kin,  # noqa: E402
                    controller_index)
from .presence import junction_carry_presence, junction_mask_from_lane  # noqa: E402
from .traffloss import (TrafficLossConfig, entrant_steps,  # noqa: E402
                        scheduled_traffic_weight, traffic_bin_loss)
from .stability import (ConservationConfig, ConservationTracker,  # noqa: E402
                        FeasibilityProjection, ProjectionConfig, combine_losses,
                        stability_regulariser)
from . import lpo  # noqa: E402
from .long_horizon import (CONFIG_KEYS as LONG_HORIZON_CONFIG_KEYS, MONITORS,  # noqa: E402
                           LongHorizonSelection, residual_mag_loss,
                           validate_bounds, validation_horizons)
from .brakefix import (BrakefixConfig, braking_terms,  # noqa: E402
                       leader_brake_events, propagation_terms, propagation_window,
                       realized_acceleration)

BRAKEFIX_CONFIG_KEYS = ("asym_speed_alpha", "decel_weight", "decel_threshold_mps2",
                        "propagation_weight", "prop_window_s", "prop_max_gap_m",
                        "residual_cap_neg_mps2")

FORBIDDEN_TOKENS = ("CAL_SPLIT_A4.json", "HOLDOUT", "CALIBRATION", "chunk-5",
                    "chunk5", "J4B", "INC1", "sealed")


class TrainDataRefusal(PermissionError):
    """A source is not demonstrably TRAIN-only."""


def _process_rss_mb() -> Optional[float]:
    """Best-effort current resident-set sample (MB) for the CPU timing receipt."""
    try:
        import psutil
        return float(psutil.Process().memory_info().rss) / (1024.0 ** 2)
    except ImportError:
        pass
    return _win_mem("WorkingSetSize")


def _peak_rss_mb() -> Optional[float]:
    """Process peak working set (MB); POSIX ``resource`` else Windows psapi."""
    try:
        import resource  # POSIX
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except ImportError:
        return _win_mem("PeakWorkingSetSize")


def _win_mem(field: str) -> Optional[float]:
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        counters = PMC()
        counters.cb = ctypes.sizeof(PMC)
        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC),
                                               wintypes.DWORD]
        psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(),
                                   ctypes.byref(counters), counters.cb)
        return float(getattr(counters, field)) / (1024.0 ** 2)
    except (AttributeError, OSError, ImportError):
        return None


def run_key(meta: dict) -> str:
    """Pack run key ``<batch>/<world>__<arm>`` used by the LPO window lists.

    ``batch`` is the campaign directory (``chunk2`` / ``truearms_chunk2``), which
    the pack INDEX uses as the key prefix; ``chunk`` is NOT unique (the same
    world/arm can appear in two campaigns), so it must never be used alone.
    """
    batch = meta.get("batch") or meta.get("chunk")
    return f"{batch}/{meta['world']}__{meta['arm']}"


def load_exclusion_keys(paths) -> set[str]:
    """Normalise ``--exclude-window-list`` files to a set of pack run keys.

    Accepts the LPO ``pi_excluded_runs.json`` shape (``{"keys": [...]}``), a
    ``{"windows": [{"key": ...}, ...]}`` shape, or a bare list of key strings.
    Metadata only; no pack array is touched.
    """
    keys: set[str] = set()
    for path in paths or ():
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "keys" in data:
            keys.update(str(k) for k in data["keys"])
        elif isinstance(data, dict) and "windows" in data:
            keys.update(str(w["key"]) for w in data["windows"])
        elif isinstance(data, list):
            keys.update(str(w["key"] if isinstance(w, dict) else w) for w in data)
        else:
            raise ValueError(f"unrecognised exclusion list shape: {path}")
    return keys


def load_window_list(path: str | Path) -> set[tuple[str, str]]:
    """Normalise a validation ``--window-list`` file to ``(pack, key)`` pairs.

    The pack name scopes the key because pack_damping re-packs pack_v3's base
    runs under identical run keys.  Metadata only.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    windows = data["windows"] if isinstance(data, dict) and "windows" in data else data
    pairs: set[tuple[str, str]] = set()
    for w in windows:
        if not isinstance(w, dict) or "key" not in w or "pack" not in w:
            raise ValueError(f"window-list entry lacks pack/key: {w!r}")
        pairs.add((str(w["pack"]), str(w["key"])))
    return pairs


def validate_train_source(path: str | Path, manifest: Optional[dict] = None) -> Path:
    """Validate source naming and manifest split claims before reading arrays."""
    p = Path(path)
    text = str(p).replace("\\", "/")
    for token in FORBIDDEN_TOKENS:
        if token.lower() in text.lower():
            raise TrainDataRefusal(f"forbidden TRAIN source token: {token}")
    if manifest is not None:
        split = str(manifest.get("split", "TRAIN")).upper()
        if split not in ("TRAIN",):
            raise TrainDataRefusal(f"source manifest split is not TRAIN: {split}")
        for item in manifest.get("items", ()):  # metadata only, before arrays
            item_split = str(item.get("split", "TRAIN")).upper()
            if item_split != "TRAIN":
                raise TrainDataRefusal(f"item split is not TRAIN: {item_split}")
    return p


def _fixed_sidecar_dir(value: str) -> str:
    """Picklable replacement for ``IdeaACorpus._sidecar_dir`` (per-corpus identity dir)."""
    return value


CAV_DRIVE_MODES = ("teacher", "closed_loop")


@dataclass
class WmSTrainConfig:
    """Small, explicit schedule for WM-S preparation and GPU execution."""

    unroll_curriculum_s: tuple[float, ...] = (10.0, 30.0, 60.0, 120.0)
    stage_steps: Optional[tuple[int, ...]] = None   # optimizer steps per stage
    num_workers: int = 0
    epochs_per_stage: int = 1
    steps_per_epoch: int = 2000
    batch_size: int = 8
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    position_huber_delta_m: float = 5.0
    speed_huber_delta_mps: float = 1.0
    lane_weight: float = 1.0
    position_weight: float = 1.0
    speed_weight: float = 1.0
    step_weight_decay: float = 0.01
    truncation_steps: int = 20
    history_features: int = 6
    history_s: Optional[float] = None
    history_stride_s: float = 1.0
    device: str = "cpu"
    threads: int = 6
    state_noise_std_m: float = 0.0
    state_noise_std_mps: float = 0.0
    rollout_mode: str = "residual"
    lane_head: str = "off"
    # Hybrid rollout only: optional penalty on the learned residual magnitude so
    # the driver-parameter encoder cannot be trivially absorbed by the residual
    # (identifiability guard).  ``None`` resolves to a modest 1e-3 in hybrid mode
    # and to 0.0 in residual/driver_params mode (which stay byte-identical).
    residual_magnitude_weight: Optional[float] = None
    # Evaluation-only driver-parameter ground-truth sidecar (never in the loss).
    driver_sidecar: Optional[str] = None
    out_dir: str = "output/wd01/wm_n1/wms/runs"
    seed: int = 101
    black_box: bool = False
    n_controllers: int = N_CONTROLLERS
    controller_embed_dim: int = 16
    checkpoint_residual: bool = False
    # Run 2 / B1: differentiable traffic-level loss (sections x 30 s bins).
    traffic_loss: bool = False
    traffic_bin_s: float = 30.0
    traffic_weight: float = 0.1
    traffic_free_speed_mps: float = 13.89
    traffic_ramp_start_s: float = 30.0
    traffic_ramp_end_s: float = 120.0
    # Run 2 / B4: Gaussian noise on predicted HDV states during training.
    state_noise_std_m: float = 0.0
    state_noise_std_mps: float = 0.0
    entries_v2: bool = False
    entry_pad: int = 8
    window_list: Optional[str] = None
    exclude_window_list: Optional[str] = None
    val_window_list: Optional[str] = None
    val_batches: int = 8
    # Run-2 stability features (all default OFF; off == byte-identical).
    projection_enabled: bool = False
    projection_min_gap_m: float = 2.0
    residual_cap_mps2: Optional[float] = None
    lane_logit_cap: Optional[float] = None
    stability_weight: float = 0.0
    stability_margin: float = 1.0
    stability_probes: int = 1
    # Conservation: the *metric* can be on with the loss term off.
    conservation_enabled: bool = False
    conservation_weight: float = 0.0
    # Run 3: log each loss term's share of the total gradient norm per stage
    # (a diagnostic-only extra backward on the last chunk of one batch).
    loss_share_logging: bool = False
    switch_window_fraction: float = 0.0
    # Braking-response fix (wms/brakefix.py); every field default OFF, and with
    # the defaults the loss/log/receipt are byte-identical to the pre-fix trainer.
    asym_speed_alpha: float = 1.0           # --asym-speed-loss (>1 on)
    decel_weight: float = 1.0               # --decel-weight (>1 on)
    decel_threshold_mps2: float = 3.0       # --decel-threshold (A, m/s^2 > 0)
    propagation_weight: float = 0.0         # --propagation-loss (>0 on)
    prop_window_s: float = 3.0              # --prop-window (S)
    prop_max_gap_m: Optional[float] = None
    residual_cap_neg_mps2: Optional[float] = None   # --residual-cap-neg-mps2
    # CAV driving during training rollouts.  ``teacher`` (default, as in all
    # earlier training): every CAV follows the recorded trajectory.  The action
    # timeline is still bound/validated/over-sampled but a ``direct`` entry does
    # NOT move the CAV. ``closed_loop`` runs native/damping/PI/direct via an
    # episode-persistent timeline dispatcher on predicted states.
    cav_drive: str = "teacher"
    residual_soft_bound_mps2: Optional[float] = None
    residual_soft_bound_neg_mps2: Optional[float] = None
    residual_mag_weight: float = 0.0
    lane_rate_weight: float = 0.0
    lane_rate_target: float = 0.0123
    select_by_long_horizon: bool = False
    val_every: int = 200  # inert unless long-horizon selection is enabled

    def __post_init__(self) -> None:
        if not self.unroll_curriculum_s or any(float(x) <= 0 for x in self.unroll_curriculum_s):
            raise ValueError("unroll_curriculum_s must be positive")
        if self.epochs_per_stage < 1 or self.steps_per_epoch < 1:
            raise ValueError("epochs_per_stage and steps_per_epoch must be positive")
        if self.truncation_steps < 1:
            raise ValueError("truncation_steps must be positive")
        if self.entry_pad < 0:
            raise ValueError("entry_pad must be non-negative")
        if self.stage_steps is not None and (
                len(self.stage_steps) != len(self.unroll_curriculum_s)
                or any(int(x) < 1 for x in self.stage_steps)):
            raise ValueError("stage_steps must give one positive count per curriculum stage")
        if float(self.traffic_bin_s) <= 0.0 or float(self.traffic_weight) < 0.0:
            raise ValueError("traffic_bin_s must be positive and traffic_weight non-negative")
        if float(self.traffic_ramp_end_s) <= float(self.traffic_ramp_start_s):
            raise ValueError("traffic_ramp_end_s must exceed traffic_ramp_start_s")
        if float(self.state_noise_std_m) < 0.0 or float(self.state_noise_std_mps) < 0.0:
            raise ValueError("state noise stds must be non-negative")
        if self.rollout_mode not in ("residual", "driver_params", "hybrid"):
            raise ValueError(
                "rollout_mode must be residual, driver_params or hybrid")
        history_spec(self.history_s, self.history_stride_s)
        if self.history_s is not None and self.black_box:
            raise ValueError("long history requires structured WM-S")
        if self.lane_head not in ("off", "history"):
            raise ValueError("lane_head must be off or history")
        if self.lane_head == "history" and self.rollout_mode not in (
                "driver_params", "hybrid"):
            raise ValueError(
                "history lane head requires rollout_mode=driver_params or hybrid")
        if self.residual_magnitude_weight is not None \
                and float(self.residual_magnitude_weight) < 0.0:
            raise ValueError("residual_magnitude_weight must be non-negative "
                             "when set")
        if self.residual_cap_mps2 is not None and float(self.residual_cap_mps2) <= 0.0:
            raise ValueError("residual_cap_mps2 must be > 0 when set")
        if self.lane_logit_cap is not None and float(self.lane_logit_cap) <= 0.0:
            raise ValueError("lane_logit_cap must be > 0 when set")
        if float(self.stability_weight) < 0.0 or float(self.conservation_weight) < 0.0:
            raise ValueError("stability/conservation weights must be non-negative")
        if float(self.projection_min_gap_m) < 0.0:
            raise ValueError("projection_min_gap_m must be non-negative")
        if not 0.0 <= float(self.switch_window_fraction) <= 1.0:
            raise ValueError("switch_window_fraction must be in [0,1]")
        if self.cav_drive not in CAV_DRIVE_MODES:
            raise ValueError(f"cav_drive must be one of {CAV_DRIVE_MODES}")
        if self.residual_cap_neg_mps2 is not None and (
                not np.isfinite(float(self.residual_cap_neg_mps2))
                or float(self.residual_cap_neg_mps2) <= 0.0):
            raise ValueError("residual_cap_neg_mps2 must be finite and > 0 when set")
        self.brakefix_config()  # validates the braking-fix hyper-parameters
        validate_bounds(self.residual_soft_bound_mps2, self.residual_soft_bound_neg_mps2,
                        self.residual_cap_mps2, self.residual_cap_neg_mps2)
        for value in (self.residual_mag_weight, self.lane_rate_weight):
            if not np.isfinite(value) or value < 0:
                raise ValueError("residual_mag/lane_rate weights must be finite and non-negative")
        if not np.isfinite(self.lane_rate_target) or not 0 <= self.lane_rate_target <= 1:
            raise ValueError("lane_rate_target must be finite and in [0,1]")
        if self.val_every < 1:
            raise ValueError("val_every must be positive")

    def brakefix_config(self, dt_s: float = 0.5) -> BrakefixConfig:
        """The braking-fix term configuration (all-off by default)."""
        return BrakefixConfig(
            asym_alpha=float(self.asym_speed_alpha),
            decel_weight=float(self.decel_weight),
            decel_threshold_mps2=float(self.decel_threshold_mps2),
            prop_weight=float(self.propagation_weight),
            prop_window_s=float(self.prop_window_s),
            prop_max_gap_m=(None if self.prop_max_gap_m is None
                            else float(self.prop_max_gap_m)), dt_s=float(dt_s))

    @classmethod
    def from_dict(cls, data: dict) -> "WmSTrainConfig":
        unknown = sorted(set(data) - {f.name for f in fields(cls)})
        if unknown:
            raise ValueError(f"unknown WM-S config keys: {unknown}")
        data = dict(data)
        if "unroll_curriculum_s" in data:
            data["unroll_curriculum_s"] = tuple(data["unroll_curriculum_s"])
        if data.get("stage_steps") is not None:
            data["stage_steps"] = tuple(int(x) for x in data["stage_steps"])
        return cls(**data)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["unroll_curriculum_s"] = list(self.unroll_curriculum_s)
        out["stage_steps"] = None if self.stage_steps is None else list(self.stage_steps)
        # Braking-fix keys are only written when a term is on / non-default, so
        # receipts and checkpoints of old (flags-off) runs stay byte-identical.
        for key in (*BRAKEFIX_CONFIG_KEYS, *LONG_HORIZON_CONFIG_KEYS):
            if out.get(key) == type(self).__dataclass_fields__[key].default:
                out.pop(key, None)
        if self.cav_drive == "teacher":
            out.pop("cav_drive", None)
        if self.history_s is None:
            out.pop("history_s", None)
            out.pop("history_stride_s", None)
        return out


def _timeline_controller_ids(ids: np.ndarray, arrays: dict) -> np.ndarray:
    """Normalize documented compact/legacy ids without guessing overlapping labels.

    ``wms_action_timeline/1`` declares the compact action vocabulary. An explicit
    ``controller_vocab`` array can also declare either vocabulary (including the
    six-label legacy vocabulary before ``direct`` was added). Unmarked legacy
    arrays with ids >= 4 retain the historical adapter; ids 0 and 2 have identical
    meanings after normalization. Unmarked ids 1/3 are ambiguous and are refused.
    """
    schema = arrays.get("controller_timeline_schema")
    if schema is None and "schema" in arrays:
        candidate = np.asarray(arrays["schema"])
        if candidate.size == 1 and str(candidate.item()).startswith("wms_action_timeline/"):
            schema = str(candidate.item())
    if schema is not None:
        schema_array = np.asarray(schema)
        if schema_array.size != 1 or str(schema_array.item()) != SCHEMA:
            raise ValueError("unsupported controller_timeline_schema")
    vocab = arrays.get("controller_vocab")
    legacy = None
    if vocab is not None:
        vocabulary = tuple(np.asarray(vocab).reshape(-1).tolist())
        if vocabulary == CONTROLLERS:
            legacy = False
        elif vocabulary in (CONTROLLER_VOCAB, CONTROLLER_VOCAB[:-1]):
            legacy = True
        else:
            raise ValueError("unknown controller_vocab for controller_timeline")
        if schema is not None and legacy:
            raise ValueError("controller_vocab conflicts with action timeline schema")
    if ids.dtype.kind in "US":
        if legacy:
            raise ValueError("legacy controller_timeline must use numeric ids")
        return np.asarray([CONTROLLERS.index(label) for label in ids.reshape(-1)],
                          dtype=np.int64).reshape(ids.shape)
    if ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("controller_timeline must contain non-negative integer ids")
    if legacy is None:
        if schema is not None:
            legacy = False
        elif np.any(ids >= len(CONTROLLERS)):
            legacy = True
        elif np.any((ids == 1) | (ids == 3)):
            raise ValueError("ambiguous controller_timeline ids; declare "
                             "wms_action_timeline/1 schema or controller_vocab")
        else:
            legacy = False
    if legacy:
        legacy_to_action = np.asarray((0, 0, 2, 1, 0, 0, 3), dtype=np.int64)
        limit = len(CONTROLLER_VOCAB) if vocab is None else len(vocabulary)
        if np.any(ids >= limit):
            raise ValueError("controller_timeline contains an unknown legacy controller id")
        return legacy_to_action[ids]
    if np.any(ids >= len(CONTROLLERS)):
        raise ValueError("controller_timeline contains an unknown action controller id")
    return ids.astype(np.int64)


class TrajectoryDataset(Dataset):
    """In-memory sequence dataset with explicit identity/presence sidecars."""

    def __init__(self, records: list[dict], *, loader=None, receipt: Optional[dict] = None):
        if not records and loader is None:
            raise ValueError("TRAIN trajectory dataset is empty")
        self.records = records
        self._loader = loader
        self.receipt = receipt or {"mode": "arrays", "n_records": len(records)}

    @classmethod
    def from_arrays(cls, arrays: dict[str, np.ndarray], *, split: str = "TRAIN",
                    history_s=None, history_stride_s=1.0):
        if str(split).upper() != "TRAIN":
            raise TrainDataRefusal("TrajectoryDataset only accepts TRAIN")
        states = np.asarray(arrays["states" if "states" in arrays else "veh_kin"],
                            dtype=np.float32)
        if states.ndim == 3:
            states = states[None]
        if states.ndim != 4 or states.shape[-1] != 7 or states.shape[1] < 2:
            raise ValueError("states must be [N,T+1,K,7], with T >= 1")
        n, t1, k, _ = states.shape
        history_spec(history_s, history_stride_s)
        present = np.asarray(arrays.get("present", np.ones((n, t1, k), bool)))
        if present.ndim == 2:
            present = present[None]
        slot = np.asarray(arrays.get("slot_index", np.broadcast_to(
            np.arange(k, dtype=np.int64), (n, t1, k))))
        if slot.ndim == 2:
            slot = slot[None]
        identities = arrays.get("identities")
        if identities is not None:
            identities = np.asarray(identities)
            if identities.ndim == 2:
                identities = identities[None]
        is_cav = arrays.get("is_cav")
        if is_cav is not None:
            is_cav = np.asarray(is_cav, dtype=bool)
            if is_cav.ndim == 2:
                is_cav = is_cav[None]
        cav = arrays.get("cav_teacher", arrays.get("cav_states"))
        # Explicit junction transits may be supplied (``junction``); otherwise a
        # slot that still carries a vehicle but has no modelled lane one-hot is a
        # junction internal-lane transit.  Both feed ``junction_carry_presence``
        # so one junction step never permanently drops the slot.
        junction = arrays.get("junction")
        if junction is not None:
            junction = np.asarray(junction, dtype=bool)
            if junction.ndim == 2:
                junction = junction[None]
        records = []
        timeline_ids = arrays.get("controller_timeline")
        timeline_accel = arrays.get("issued_acceleration_mps2")
        timeline_target = arrays.get("issued_target_speed_mps")
        if timeline_ids is not None:
            timeline_ids = np.asarray(timeline_ids)
            if timeline_ids.ndim == 2:
                timeline_ids = timeline_ids[None]
            if timeline_ids.ndim != 3 or timeline_ids.shape[0] != n \
                    or timeline_ids.shape[1] < 1 or timeline_ids.shape[2] != k:
                raise ValueError("controller_timeline must be [N,control_steps,K]")
            timeline_ids = _timeline_controller_ids(timeline_ids, arrays)
            if np.any(timeline_ids == CONTROLLERS.index("direct")) \
                    and (timeline_accel is None or timeline_target is None):
                raise ValueError("direct actions require both issued command fields")
            commands = []
            for command in (timeline_accel, timeline_target):
                command = (np.zeros_like(timeline_ids, dtype=np.float32) if command is None
                           else np.asarray(command, dtype=np.float32))
                if command.ndim == 2:
                    command = command[None]
                if command.shape != timeline_ids.shape:
                    raise ValueError("issued command timeline shape differs from controller_timeline")
                commands.append(command)
            timeline_accel, timeline_target = commands
        for i in range(n):
            actual_present = present[i].astype(bool)
            on_road = states[i, :, :, 1:4].sum(-1) > 0
            if junction is not None:
                jn = junction[i] if junction.ndim == 3 else junction
            else:
                jn = junction_mask_from_lane(states[i, :, :, 1:4], actual_present)
            if identities is None:
                present_eff, jn_eff = actual_present, jn
            else:
                first_id = identities[i, 0]
                same = (identities[i] == first_id) & (first_id >= 0)
                present_eff, jn_eff = actual_present & same, jn & same
            alive, _ = junction_carry_presence(present_eff, jn_eff)
            rec = {"states": states[i], "present": alive,
                   "loss_valid": alive & on_road,
                   "slot_index": slot[i].astype(np.int64)}
            if history_s is not None:
                # Optional explicit pre-t0 native observations. Without them,
                # an array window starts at t0 and has one observed tick only.
                if "history_kin" in arrays:
                    hk = np.asarray(arrays["history_kin"], np.float32)
                    hm = np.asarray(arrays["history_mask"], bool)
                    ht = np.asarray(arrays["history_target_mask"], bool)
                    if hk.ndim == 3:
                        hk, hm, ht = hk[None], hm[None], ht[None]
                    if hk.shape[0] != n or hk.shape[2:] != (k, 7):
                        raise ValueError("history_kin must be [N,H,K,7]")
                    if not np.array_equal(hk[i, -1], states[i, 0]):
                        raise ValueError("history_kin must end at the window t0 state")
                    raw, observed, continuous = hk[i], hm[i], ht[i]
                else:
                    raw, observed = states[i, :1], actual_present[:1]
                    continuous = np.zeros((0, k), bool)
                dh, dm = long_history_features(
                    torch.as_tensor(raw), torch.as_tensor(observed),
                    torch.as_tensor(continuous), torch.as_tensor(alive[0]),
                    history_s=history_s, history_stride_s=history_stride_s)
                rec["driver_hist"], rec["driver_hist_mask"] = dh.numpy(), dm.numpy()
            if identities is not None:
                rec["identities"] = identities[i]
            if is_cav is not None:
                cav0 = is_cav[i, 0] if is_cav.ndim == 3 else is_cav[i]
                rec["is_cav"] = np.broadcast_to(cav0, alive.shape).copy() & alive
            if cav is not None:
                rec["cav_teacher"] = np.asarray(cav[i], dtype=np.float32)
            elif is_cav is not None:
                # Structured rollouts teacher-force CAVs; the recorded CAV slot
                # of ``states`` IS the true next state, so no controller code is
                # needed when no explicit teacher is supplied.  [T,K,4]=s,lane,v,a.
                rec["cav_teacher"] = np.stack(
                    (states[i, ..., 0],
                     states[i, ..., 1:4].argmax(-1).astype(np.float32),
                     states[i, ..., 4], states[i, ..., 5]), axis=-1).astype(np.float32)
            if timeline_ids is not None:
                ActionTimeline(timeline_ids[i], timeline_accel[i], timeline_target[i])
                rec["action_timeline"] = {
                    "controller": timeline_ids[i].astype(np.int64),
                    "acceleration_mps2": timeline_accel[i],
                    "target_speed_mps": timeline_target[i], "control_dt_s": 1.0}
            records.append(rec)
        return cls(records)

    @classmethod
    def from_pack(cls, path: str | Path, *, history_s=None,
                  history_stride_s=1.0) -> "TrajectoryDataset":
        p = validate_train_source(path)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.is_file():
            if p.suffix.lower() != ".npz":
                raise ValueError("trajectory source file must be .npz")
            with np.load(p, allow_pickle=False) as data:
                return cls.from_arrays({k: data[k] for k in data.files},
                                       history_s=history_s, history_stride_s=history_stride_s)
        manifest_path = p / "manifest.json"
        manifest = None
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            validate_train_source(p, manifest)
        for name in ("trajectory.npz", "states.npz"):
            candidate = p / name
            if candidate.is_file():
                return cls.from_pack(candidate, history_s=history_s,
                                     history_stride_s=history_stride_s)
        for name in ("states.npy", "veh_kin.npy"):
            candidate = p / name
            if candidate.is_file():
                states = np.load(candidate, mmap_mode="r", allow_pickle=False)
                if states.ndim < 4:
                    raise TrainDataRefusal(
                        f"{candidate} has snapshots only; no multi-step sequence")
                arrays = {"states": states}
                for sidecar in ("present.npy", "slot_index.npy", "cav_teacher.npy"):
                    f = p / sidecar
                    if f.is_file():
                        arrays[sidecar.removesuffix(".npy")] = np.load(
                            f, mmap_mode="r", allow_pickle=False)
                for sidecar in ("controller_timeline.npy", "issued_acceleration_mps2.npy",
                                "issued_target_speed_mps.npy", "controller_vocab.npy",
                                "controller_timeline_schema.npy"):
                    f = p / sidecar
                    if f.is_file():
                        arrays[sidecar.removesuffix(".npy")] = np.load(
                            f, mmap_mode="r", allow_pickle=False)
                if history_s is not None:
                    for sidecar in ("history_kin", "history_mask", "history_target_mask"):
                        f = p / f"{sidecar}.npy"
                        if f.is_file():
                            arrays[sidecar] = np.load(f, mmap_mode="r", allow_pickle=False)
                return cls.from_arrays(arrays, history_s=history_s,
                                       history_stride_s=history_stride_s)
        # A11's real pack stores contiguous per-run ticks plus INDEX ranges,
        # rather than a second trajectory file.  Reuse its read-only window
        # adapter so pack_v3, dspeed and damping packs retain their identity and
        # exit masks.  No idea_a trainer code is changed or called for losses.
        if manifest and "veh_kin" in (manifest.get("arrays") or {}):
            try:
                from train.idea_a import IdeaACorpus
                corpus = IdeaACorpus(p, split="TRAIN", horizon_steps=240,
                                     allow_single_run_fallback=False)
            except Exception as exc:  # noqa: BLE001
                raise TrainDataRefusal(
                    f"A11 TRAIN pack could not establish its verified window "
                    f"and identity interface: {exc}") from exc
            records = []
            for position in range(len(corpus._order)):
                item = corpus._items[corpus._order[position]]
                window = corpus.window(item)
                teacher = window.get("teacher")
                if teacher is None:
                    raise TrainDataRefusal("A11 pack item has no trajectory teacher")
                teacher = teacher[:, 0].numpy().astype(np.float32)
                horizon = teacher.shape[0]
                on_road = teacher[..., 1].astype(np.int64) >= 0
                lane = np.clip(teacher[..., 1].astype(np.int64), 0, 2)
                one_hot = np.eye(3, dtype=np.float32)[lane]
                states = np.concatenate((teacher[..., 0:1], one_hot,
                                         teacher[..., 2:4],
                                         np.broadcast_to(
                                             window["length0"][0].numpy(),
                                             (horizon, teacher.shape[1], 1))), axis=-1)
                present = np.concatenate((
                    window["present0"][0].numpy().astype(bool)[None],
                    window["rollout_mask"][:, 0].numpy().astype(bool)), axis=0)
                record = {"states": states, "present": present,
                          "loss_valid": present & on_road,
                          "slot_index": window.get(
                              "vehicle_ids", torch.arange(states.shape[1]))[0].numpy()
                          if window.get("vehicle_ids") is not None else
                          np.arange(states.shape[1], dtype=np.int64),
                          "is_cav": (np.broadcast_to(
                              window["is_cav"][0].numpy().astype(bool),
                              present.shape) & present),
                          "cav_teacher": teacher}
                if history_s is not None:
                    dh, dm = _prior_history(corpus, int(item["tick"]), present[0],
                                            history_s=history_s,
                                            history_stride_s=history_stride_s)
                    record["driver_hist"], record["driver_hist_mask"] = dh, dm
                records.append(record)
            return cls(records)
        raise TrainDataRefusal(
            f"{p} has no trajectory sequence; snapshot-only A11 packs are refused")

    @classmethod
    def from_real_packs(cls, paths: list[str | Path], *, identity_dirs: list[str | Path],
                        overlays: Optional[list[str | Path]] = None,
                        horizon_steps: int = 60,
                        start_shift_prob: float = 0.0,
                        exclude_keys: Optional[set[str]] = None,
                        select_windows: Optional[set[tuple[str, str]]] = None,
                        window_list: Optional[str | Path] = None,
                        exclude_window_list: Optional[str | Path | list[str]] = None,
                        pack_name_list: Optional[list[str]] = None,
                        entries_v2: bool = False,
                        timelines: Optional[dict] = None,
                        item_filter: Optional[Callable] = None,
                        identity_env_dir: Optional[str | Path] = None,
                        history_s=None, history_stride_s=1.0
                        ) -> "TrajectoryDataset":
        """Build a lazy TRAIN view over A11 packs and verified identity windows.

        ``IdeaACorpus`` performs the authoritative split and INDEX range checks.
        The sidecars are read through the literal workspace junction paths here;
        no resolved external target is followed and no source array is copied.

        ``exclude_keys`` drops windows whose pack run key is listed (leave-PI-out);
        every listed key must match a corpus window or the load refuses.
        ``select_windows`` (``(pack, key)`` pairs) keeps only a validation subset;
        every requested pair must match or the load refuses.  Both filters run
        after the corpus split checks and before identity/position validation.

        Switching/direct packs (``wms.mixdata``): ``timelines`` maps a pack index
        to a provider whose ``timeline(position, n_control)`` returns that item's
        ``wms_action_timeline/1`` arrays bound to the pack slots (it replaces the
        legacy constant timeline); ``item_filter(corpus, kept_orig)`` returns the
        kept subset and a receipt (metadata/mmap only; used for the zero-shot
        held-out guard); ``identity_env_dir`` overrides the process-wide
        ``WM_N1_TRAIN_IDENTITIES`` for this corpus so two corpora with different
        identity sidecars can coexist.  Start-shift windows are refused with a
        timeline because their slot columns would no longer match the timeline.

        ``window_list`` (P5 ``windows_train_noPI.json``) keeps only listed
        ``(batch, world, arm, t0_s)`` windows — the leave-one-controller-out
        TRAIN set; ``exclude_window_list`` drops a bare key list
        (``pi_excluded_runs.json``).  Both are metadata-only and fail closed on
        an unmatched keep key.
        """
        if not paths:
            raise ValueError("at least one trajectory pack is required")
        history_spec(history_s, history_stride_s)
        if len(identity_dirs) != len(paths):
            raise TrainDataRefusal(
                f"expected one identity sidecar per pack ({len(paths)}), got "
                f"{len(identity_dirs)}")
        if overlays is not None and len(overlays) != max(0, len(paths) - 1):
            raise TrainDataRefusal(
                f"expected one overlay per extra pack ({len(paths) - 1}), got "
                f"{len(overlays)}")
        for path in paths:
            validate_train_source(path)
        if window_list is not None:
            validate_train_source(window_list)
        if timelines and float(start_shift_prob) > 0.0:
            raise TrainDataRefusal(
                "start-shift windows are refused with action timelines: the shifted "
                "window's slot columns do not match the t0-bound timeline columns")
        try:
            from train.idea_a import IdeaACorpus
            corpus = IdeaACorpus(
                str(paths[0]), split="TRAIN", horizon_steps=int(horizon_steps),
                extra_pack_dirs=[str(p) for p in paths[1:]],
                extra_overlay_dir=None if overlays is None else [str(p) for p in overlays],
                extra_identity_dirs=[str(p) for p in identity_dirs[1:]] or None,
            )
        except (ImportError, KeyError, OSError, ValueError, RuntimeError) as exc:
            raise TrainDataRefusal(f"A11 TRAIN window adapter refused: {exc}") from exc
        if len(corpus) == 0:
            raise TrainDataRefusal("A11 TRAIN selection is empty")
        if identity_env_dir is not None:
            corpus._sidecar_dir = functools.partial(_fixed_sidecar_dir,
                                                    str(identity_env_dir))

        names = list(pack_name_list) if pack_name_list else lpo.pack_names(paths)
        if len(names) != len(paths):
            raise TrainDataRefusal(
                f"pack_name_list has {len(names)} entries for {len(paths)} packs")
        identity_rows = []
        for path, sidecar in zip(paths, identity_dirs):
            identity_rows.append(_read_identity_rows(sidecar, corpus_k=corpus.k_max))
        pack_names = [Path(p).name for p in paths]
        # Optional structured (batch, world, arm, t0_s) leave-controller-out
        # keep/exclude filter (P5 window lists).  Metadata-only and fail-closed.
        filter_receipt = None
        lpo_keep: Optional[set[int]] = None
        if window_list is not None or exclude_window_list is not None:
            lpo_selected, filter_receipt = _select_lpo_positions(
                corpus, names, window_list=window_list,
                exclude_window_list=exclude_window_list)
            lpo_keep = set(lpo_selected)
        # Filter by pack run key AFTER the corpus split checks and BEFORE any
        # identity/position validation (leave-PI-out exclusion + val selection).
        kept_orig: list[int] = []
        keys_present: set[str] = set()
        total_by_pack: Counter = Counter()
        kept_by_pack: Counter = Counter()
        matched_select: set[tuple[str, str]] = set()
        n_excluded = 0
        for i in range(len(corpus)):
            if lpo_keep is not None and i not in lpo_keep:
                continue
            meta = corpus._items[corpus._order[i]]
            key = run_key(meta)
            pack = int(corpus._global_packs[i])
            pname = pack_names[pack] if 0 <= pack < len(pack_names) else str(pack)
            keys_present.add(key)
            total_by_pack[pname] += 1
            pair = (pname, key)
            excluded = exclude_keys is not None and key in exclude_keys
            if excluded:
                n_excluded += 1
            if select_windows is not None and pair in select_windows:
                matched_select.add(pair)
                if excluded:
                    raise TrainDataRefusal(
                        f"window-list entry is also in the exclusion list: {pname}/{key}")
            if excluded or (select_windows is not None and pair not in select_windows):
                continue
            kept_orig.append(i)
            kept_by_pack[pname] += 1
        if exclude_keys is not None:
            absent = sorted(k for k in exclude_keys if k not in keys_present)
            if absent:
                raise TrainDataRefusal(
                    f"exclusion list has {len(absent)} key(s) absent from the TRAIN "
                    f"corpus, e.g. {absent[:3]}")
        if select_windows is not None:
            missing = sorted(select_windows - matched_select)
            if missing:
                raise TrainDataRefusal(
                    f"window-list has {len(missing)} entry(ies) absent from the TRAIN "
                    f"corpus, e.g. {missing[:3]}")
        if not kept_orig:
            raise TrainDataRefusal("TRAIN window selection is empty after filtering")
        item_filter_receipt = None
        if item_filter is not None:
            kept_orig, item_filter_receipt = item_filter(corpus, list(kept_orig))
            kept_orig = [int(i) for i in kept_orig]
            if not kept_orig:
                raise TrainDataRefusal("item filter left no TRAIN windows")
        # Use the corpus-resolved ORIGINAL per-pack manifest positions (never the
        # TRAIN-filtered/composite-offset dataset position).
        positions = []
        for i in kept_orig:
            pack = int(corpus._global_packs[i])
            position = int(corpus._global_positions[i])
            if position not in identity_rows[pack]:
                raise TrainDataRefusal(
                    f"identity sidecar {identity_dirs[pack]} lacks TRAIN manifest "
                    f"position {position}")
            positions.append((pack, position))

        def load(index: int) -> dict:
            item = corpus._v1[kept_orig[index]]
            shift = 0
            if start_shift_prob > 0.0 and float(torch.rand(())) < start_shift_prob:
                tick0 = int(item["tick"])
                for r0, r1 in corpus._runs:
                    if r0 <= tick0 < r1:
                        room = r1 - tick0 - 1 - int(horizon_steps)
                        if room > 0:
                            shift = int(torch.randint(1, room + 1, ()))
                        break
            if shift:
                item = _shifted_item(corpus, item, int(item["tick"]) + shift)
                orig_ids = corpus._ids_for_item
                corpus._ids_for_item = lambda _it: None   # identity rows are keyed to the original t0
                try:
                    window = corpus.window(item)
                finally:
                    del corpus._ids_for_item
                    assert corpus._ids_for_item == orig_ids
            else:
                window = corpus.window(item)
            teacher = window.get("teacher")
            if teacher is None or teacher.shape[0] < 2:
                raise TrainDataRefusal(f"TRAIN item {index} has no multi-step target")
            teacher = teacher[:, 0].numpy().astype(np.float32)
            horizon = teacher.shape[0]
            on_road = teacher[..., 1].astype(np.int64) >= 0
            lane = np.clip(teacher[..., 1].astype(np.int64), 0, 2)
            one_hot = np.eye(3, dtype=np.float32)[lane]
            states = np.concatenate(
                (teacher[..., 0:1], one_hot, teacher[..., 2:4],
                 np.broadcast_to(window["length0"][0].numpy()[:, None],
                                 (horizon, teacher.shape[1], 1))), axis=-1)
            present = np.concatenate(
                (window["present0"][0].numpy().astype(bool)[None],
                 window["rollout_mask"][:, 0].numpy().astype(bool)), axis=0)
            # A junction internal-lane step still carries the vehicle (the pack
            # ``target_mask`` already keeps it), but its lane target is undefined,
            # so exclude only that step from the loss.
            loss_valid = present & on_road
            pack, position = positions[index]
            ids = identity_rows[pack][position]
            policy = (window.get("meta") or [{}])[0].get("policy")
            hist, hist_mask = _prior_history(
                corpus, int(item["tick"]), window["present0"][0].numpy().astype(bool))
            result = {
                "hist": hist, "hist_mask": hist_mask,
                "states": states,
                "present": present,
                "loss_valid": loss_valid,
                "slot_index": np.broadcast_to(
                    np.arange(states.shape[1], dtype=np.int64), present.shape).copy(),
                "vehicle_ids": ids.copy() if not shift else np.full_like(ids, -1),
                "start_shift_ticks": int(shift),
                "controller_id": int(controller_index(policy or window.get("arm"))),
                "is_cav": (np.broadcast_to(
                    window["is_cav"][0].numpy().astype(bool), present.shape).copy()
                    & present),
                "cav_teacher": teacher,
                "meta": window["meta"][0],
            }
            action_name = str(policy or window.get("arm") or "").lower()
            action_name = {"pi": "pi-external-u", "fleet-pi": "pi-external-u",
                           "dspeed": "direct", "direct-speed": "direct",
                           "direct_speed": "direct", "trajectory": "direct"}.get(action_name, action_name)
            if action_name in ("native", "damping", "direct", "pi-external-u"):
                # Legacy fixed-controller runs become constant action timelines.
                action_name = {"pi-external-u": "pi-external-u"}.get(action_name,
                                                                       action_name)
                n_control = max(1, int(np.ceil((horizon - 1) * 0.5)))
                fixed = ActionTimeline.constant(action_name, n_control, states.shape[1])
                if action_name == "direct":
                    # Original direct-speed packs: fixed controller identity,
                    # recorded issued command channels, never invented zeros.
                    stop = min(int(item["tick"]) + horizon, corpus._run_end(int(item["tick"])))
                    commands = np.asarray(corpus._array("cav_plan_ch")[int(item["tick"]):stop:2], np.float32)
                    count = min(n_control, commands.shape[0])
                    fixed.acceleration_mps2[:count] = commands[:count, :, 1] * 3.0
                    fixed.target_speed_mps[:count] = commands[:count, :, 0] * 40.0
                result["action_timeline"] = {
                    "controller": fixed.controller,
                    "acceleration_mps2": fixed.acceleration_mps2,
                    "target_speed_mps": fixed.target_speed_mps,
                    "control_dt_s": fixed.control_dt_s}
            provider = None if timelines is None else timelines.get(pack)
            if provider is not None:
                n_control = max(1, int(np.ceil((horizon - 1) * 0.5)))
                result["action_timeline"] = provider.timeline(position, n_control)
            elif "action_timeline" not in result:
                # Heterogeneous legacy snapshots: bind the recorded t0 controller
                # per slot instead of silently labelling every CAV native.
                source = np.asarray(corpus._array("veh_step")[int(item["tick"])])
                cav0 = source[:, 7] > .5
                ids0 = np.zeros(states.shape[1], np.int64)
                actions = source[:, 8:12]
                if np.any(cav0 & (actions.sum(-1) < .5)):
                    raise TrainDataRefusal("original CAV has no supported controller label")
                if np.any(cav0 & (actions[:, 3] > .5)):
                    raise TrainDataRefusal("wall timeline unsupported in mixed training")
                ids0[cav0] = np.asarray((0, 1, 2, 0))[actions[cav0].argmax(-1)]
                n_control = max(1, int(np.ceil((horizon - 1) * .5)))
                shape = (n_control, states.shape[1])
                result["action_timeline"] = {"controller": np.broadcast_to(ids0, shape).copy(),
                    "acceleration_mps2": np.zeros(shape, np.float32),
                    "target_speed_mps": np.zeros(shape, np.float32), "control_dt_s": 1.0}
            if entries_v2:
                result["entry_feat"] = _entry_features_from_corpus(corpus, item)
            if history_s is not None:
                dh, dm = _prior_history(
                    corpus, int(item["tick"]), window["present0"][0].numpy().astype(bool),
                    history_s=history_s, history_stride_s=history_stride_s)
                result["driver_hist"], result["driver_hist_mask"] = dh, dm
            return result

        receipt = {
            "mode": "a11-index-window",
            "split": "TRAIN",
            "packs": [str(p) for p in paths],
            "identity_sidecars": [str(p) for p in identity_dirs],
            "overlays": [] if overlays is None else [str(p) for p in overlays],
            "n_records": len(kept_orig),
            "n_corpus_windows": len(corpus),
            "horizon_steps": int(horizon_steps),
            "start_shift_prob": float(start_shift_prob),
            "identity_rows": len(identity_rows),
            "kept_by_pack": dict(sorted(kept_by_pack.items())),
            "total_by_pack": dict(sorted(total_by_pack.items())),
            "exclude_keys_requested": (0 if exclude_keys is None else len(exclude_keys)),
            "excluded_records": int(n_excluded),
            "select_windows_requested": (None if select_windows is None
                                         else len(select_windows)),
            "window_filter": filter_receipt,
            "item_filter": item_filter_receipt,
            "timeline_packs": (None if not timelines else sorted(int(k) for k in timelines)),
        }
        if history_s is not None:
            receipt["driver_history"] = {
                "history_s": history_s, "history_stride_s": history_stride_s,
                "short_history_policy": "front-pad and mask; no windows dropped"}
        dataset = cls([{} for _ in range(len(kept_orig))], loader=load, receipt=receipt)
        dataset.positions = list(positions)      # [(pack, original manifest position)]
        return dataset

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self._loader(index) if self._loader is not None else self.records[index]


def _resolve_window_pack_names(corpus, names: list[str],
                               keep_raw: dict[str, set]) -> tuple[dict[str, set], dict]:
    """Map window-list pack names onto the corpus pack names (FAIL CLOSED).

    The P5/LPWO lists name packs canonically (``pack_v3``), while a deployed
    pack directory may be staged under a different basename (``pack``,
    ``pack_v3_sparse``).  Matching by basename alone silently drops *every*
    window of the renamed pack: the list entries are filtered out before the
    keep-set is compared, so they are neither trained nor reported missing.
    Resolve each unmatched list name to the unique corpus pack whose TRAIN
    items cover its window keys (metadata only); refuse if it is absent or
    ambiguous instead of quietly returning zero windows.  Returns the resolved
    keep-set and the ``{window-list pack name: corpus pack name}`` mapping.
    """
    name_index = {str(n): i for i, n in enumerate(names)}
    if len(name_index) != len(names):
        raise TrainDataRefusal("pack names must be unique for window-list matching")
    mapping = {str(q): str(q) for q in keep_raw if str(q) in name_index}
    if set(mapping) == set(str(q) for q in keep_raw):
        return keep_raw, mapping
    unmatched = sorted(str(q) for q in keep_raw if str(q) not in name_index)
    free = [i for i, n in enumerate(names) if n not in keep_raw]
    if len(unmatched) > len(free):
        raise TrainDataRefusal(
            f"window-list pack name(s) {unmatched} do not match any pack "
            f"directory {names}; refusing rather than silently dropping them")
    keys_by_pack: dict[int, set] = {}
    for i in range(len(corpus)):
        p = int(corpus._global_packs[i])
        keys_by_pack.setdefault(p, set()).add(lpo.item_key(corpus._items[corpus._order[i]]))
    used: set[int] = set()
    for q in unmatched:
        best_p, best, ties = None, 0, 0
        for p in free:
            if p in used:
                continue
            n = len(set(keep_raw[q]) & keys_by_pack.get(p, set()))
            if n > best:
                best, best_p, ties = n, p, 1
            elif n == best and n > 0:
                ties += 1
        if best_p is None or best <= 0 or ties > 1:
            raise TrainDataRefusal(
                f"cannot uniquely resolve window-list pack {q!r} to a pack "
                f"directory (best overlap {best}, ties {ties}); refusing")
        mapping[q] = str(names[best_p])
        used.add(best_p)
    resolved: dict[str, set] = {}
    for q, keys in keep_raw.items():
        resolved.setdefault(mapping[str(q)], set()).update(keys)
    return resolved, mapping


def _select_lpo_positions(corpus, names: list[str], *,
                          window_list: Optional[str | Path] = None,
                          exclude_window_list: Optional[str | Path] = None):
    """Dataset positions kept by the C1 leave-controller-out window filter.

    Reads only the raw per-item metadata (``corpus._items``), never an array.
    """
    items = [corpus._items[corpus._order[i]] for i in range(len(corpus))]
    packs = [int(p) for p in corpus._global_packs]
    keep = None
    resolution = None
    if window_list is not None:
        keep_raw = lpo.window_keys(window_list, packs=None)
        keep, mapping = _resolve_window_pack_names(corpus, names, keep_raw)
        resolution = {k: v for k, v in mapping.items() if k != v} or None
    exclude = None
    if exclude_window_list is not None:
        paths = (list(exclude_window_list)
                 if isinstance(exclude_window_list, (list, tuple))
                 else [exclude_window_list])
        exclude = set()
        for path in paths:
            exclude |= lpo.exclude_runs(path)
    selected, receipt = lpo.filter_positions(items, packs, names,
                                             keep=keep, exclude=exclude)
    receipt.update({"window_list": None if window_list is None else str(window_list),
                    "pack_name_resolution": resolution,
                    "exclude_window_list": None if exclude_window_list is None
                    else [str(p) for p in (
                        exclude_window_list
                        if isinstance(exclude_window_list, (list, tuple))
                        else [exclude_window_list])]})
    return selected, receipt


def _read_identity_rows(sidecar: str | Path, *, corpus_k: int) -> dict[int, np.ndarray]:
    """Validate a TRAIN identity bundle without following its junction target."""
    root = validate_train_source(sidecar)
    receipt_path = root / "TRAIN_IDENTITY_RECEIPT.json"
    if not receipt_path.is_file():
        raise TrainDataRefusal(f"identity receipt missing: {root}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("split") != "TRAIN" or receipt.get("schema") != \
            "wm_n1_train_identity_sidecar/1":
        raise TrainDataRefusal(f"identity sidecar is not TRAIN: {root}")
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise TrainDataRefusal(f"identity sidecar has no file hashes: {root}")
    for name in ("identities.npy", "global_positions.npy", "vehicle_map.json"):
        path = root / name
        if not path.is_file():
            raise TrainDataRefusal(f"identity sidecar file missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != files.get(name):
            raise TrainDataRefusal(f"identity sidecar hash mismatch: {path}")
    ids = np.load(root / "identities.npy", mmap_mode="r", allow_pickle=False)
    positions = np.load(root / "global_positions.npy", mmap_mode="r", allow_pickle=False)
    if ids.dtype != np.int64 or positions.dtype != np.int64 or ids.ndim != 2 \
            or ids.shape[1] != int(corpus_k) or positions.shape != (ids.shape[0],):
        raise TrainDataRefusal(f"identity sidecar shape/dtype mismatch: {root}")
    if len(set(positions.tolist())) != len(positions):
        raise TrainDataRefusal(f"identity positions are duplicated: {root}")
    count = len(json.loads((root / "vehicle_map.json").read_text(encoding="utf-8")))
    if np.any(ids < -1) or np.any(ids >= count):
        raise TrainDataRefusal(f"identity IDs out of range: {root}")
    return {int(position): np.asarray(row, dtype=np.int64)
            for position, row in zip(positions.tolist(), ids)}


def _shifted_item(corpus, item: dict, tick: int) -> dict:
    """The same run's item re-based to a later tick (real pre-t0 history exists).

    Only per-tick arrays at ``tick`` are replaced; world/arm/plan/outcome rows are
    those of the original item.  The slot->vehicle identity row of the original
    t0 does not apply, so shifted windows carry ``vehicle_ids = -1``; same-vehicle
    continuity inside the window is given by the pack's ``target_mask``.
    """
    out = dict(item)
    arr = corpus._array
    out["tick"] = int(tick)
    out["veh_step"] = torch.as_tensor(np.asarray(arr("veh_step")[tick], np.float32))
    out["veh_mask"] = torch.as_tensor(np.asarray(arr("veh_mask")[tick]).astype(bool))
    out["accel"] = torch.as_tensor(np.asarray(arr("accel_next")[tick], np.float32))
    out["target_mask"] = torch.as_tensor(np.asarray(arr("target_mask")[tick]).astype(bool))
    out["lane"] = torch.as_tensor(np.asarray(arr("lane_next")[tick]).astype(np.int64))
    out["counter"] = torch.as_tensor(np.asarray(arr("counter")[tick], np.float32))
    out["arrivals"] = torch.as_tensor(np.asarray(arr("arrivals")[tick], np.float32))
    return out


def _prior_history(corpus, tick: int, present_now: np.ndarray, *,
                  history_s=None, history_stride_s=1.0):
    """Real pre-t0 motion history ``[K,H,F]`` + validity ``[K,H]`` (ticks <= t0 only).

    A slot's entry at ``tick-j`` is valid only if the same vehicle occupied it:
    ``target_mask[u] == 1`` iff the slot's vehicle is still present at ``u+1``, so
    validity is the backward cumulative product of ``target_mask`` and presence.
    Never crosses the run start.
    """
    from .model import HISTORY_F, HISTORY_H, history_step_features
    run_start = 0
    for start, end in corpus._runs:
        if start <= tick < end:
            run_start = start
            break
    native, _, _ = history_spec(history_s, history_stride_s)
    lo = max(run_start, tick - native + 1)
    kin = torch.as_tensor(np.asarray(corpus._array("veh_kin")[lo:tick + 1], np.float32))
    pres = torch.as_tensor(np.asarray(corpus._array("veh_mask")[lo:tick + 1]).astype(bool))
    t, k = kin.shape[:2]
    tm = torch.as_tensor(np.asarray(corpus._array("target_mask")[lo:tick], bool)) \
        if tick > lo else torch.zeros(0, k, dtype=torch.bool)
    if history_s is not None:
        feat, mask = long_history_features(kin, pres, tm, torch.as_tensor(present_now),
                                           history_s=history_s,
                                           history_stride_s=history_stride_s)
        return feat.numpy(), mask.numpy()
    same = torch.ones(t, k, dtype=torch.bool)       # same vehicle as at t0
    for j in range(t - 2, -1, -1):                  # row j -> j+1 continuity
        same[j] = same[j + 1] & tm[j]
    valid = same & pres
    valid[-1] = torch.as_tensor(present_now)
    ln = kin[..., 1:4].argmax(-1)
    s_ = kin[..., 0]
    gap = torch.full((t, k), 250.0)
    for ti in range(t):
        ds = s_[ti].unsqueeze(0) - s_[ti].unsqueeze(1)      # s_j - s_i
        ok = valid[ti].unsqueeze(0) & (ln[ti].unsqueeze(0) == ln[ti].unsqueeze(1)) & (ds > 0)
        g = torch.where(ok, ds - kin[ti, :, 6].unsqueeze(0), torch.full_like(ds, 250.0))
        gap[ti] = g.min(-1).values.clamp(0.3, 250.0)
    feat = history_step_features(kin, gap)                  # [t,K,F]
    out = torch.zeros(k, HISTORY_H, HISTORY_F)
    mask = torch.zeros(k, HISTORY_H, dtype=torch.bool)
    out[:, HISTORY_H - t:] = feat.transpose(0, 1)
    mask[:, HISTORY_H - t:] = valid.transpose(0, 1)
    return out.numpy(), mask.numpy()


def _entry_features_from_corpus(corpus, item: dict) -> np.ndarray:
    """Observed pre-t0 entry-history feature vector for the entries-v2 trainer.

    Reads only the window's t0 ``counter`` and its pre-t0 ``veh_kin``/
    ``veh_mask``/``target_mask`` ticks (t <= t0).  A slot whose vehicle first
    appears at tick ``u`` (present at ``u`` but not the same vehicle continued
    from ``u-1``) is an observed entrant; its entrance, speed, same-lane gap to
    the leader and the entry times are collected and passed to
    :func:`wms.entries.build_entrance_history`.  No recorded future is read.
    """
    from .entries import build_entrance_history

    tick = int(item["tick"])
    counter = np.asarray(corpus._array("counter")[tick], np.float32)
    run_start = 0
    for start, end in corpus._runs:
        if start <= tick < end:
            run_start = start
            break
    lo = max(run_start, tick - 120)          # bounded pre-t0 history (60 s)
    kin = np.asarray(corpus._array("veh_kin")[lo:tick + 1], np.float32)
    mask = np.asarray(corpus._array("veh_mask")[lo:tick + 1]).astype(bool)
    tm = (np.asarray(corpus._array("target_mask")[lo:tick], bool)
          if tick > lo else np.zeros((0, kin.shape[1]), bool))
    times: list = []
    entrance: list = []
    speed: list = []
    local_speed: list = []
    lane_obs: list = []
    gaps: list = []
    gap_ent: list = []
    for j in range(kin.shape[0]):
        pres = mask[j]
        prev_same = np.zeros_like(pres) if j == 0 else (mask[j - 1] & tm[j - 1])
        lanes_j = np.argmax(kin[j, :, 1:4], axis=-1)
        for k in np.flatnonzero(pres & ~prev_same):
            ln = int(lanes_j[k])
            if ln not in (0, 1, 2):
                continue
            e = 0 if ln in (0, 1) else 1
            v_self = float(kin[j, k, 4])
            same_lane = pres & (lanes_j == ln)
            idx = np.flatnonzero(same_lane)
            idx = idx[np.argsort(kin[j, idx, 0])]
            local = float(kin[j, idx[:3], 4].mean()) if idx.size else v_self
            ahead = np.flatnonzero(same_lane & (kin[j, :, 0] > kin[j, k, 0]))
            if ahead.size:
                lead = ahead[np.argmin(kin[j, ahead, 0])]
                gap = float(kin[j, lead, 0] - kin[j, k, 0] - kin[j, k, 6])
                if gap > 0.0:
                    gaps.append(gap)
                    gap_ent.append(e)
            times.append((lo + j) * 0.5)
            entrance.append(e)
            speed.append(v_self)
            local_speed.append(max(local, 1.0))
            # One lane per observed entrant (aligned with ``entrance``); the
            # builder keeps only the mainline (e == 0) lanes for lane_p1.
            lane_obs.append(ln)
    history = build_entrance_history(
        counter=counter, times=np.asarray(times, np.float64),
        entrance=np.asarray(entrance, np.int64),
        speed=np.asarray(speed, np.float64),
        local_speed=np.asarray(local_speed, np.float64),
        lane=(np.asarray(lane_obs, np.int64) if lane_obs else None),
        gap=np.asarray(gaps, np.float64),
        gap_entrance=np.asarray(gap_ent, np.int64), t0_s=float(tick) * 0.5)
    return history.to_features().astype(np.float32)


def collate(records: list[dict]) -> dict[str, Tensor | list]:
    keys = ("states", "present", "slot_index")
    out = {key: torch.as_tensor(np.stack([r[key] for r in records])) for key in keys}
    out["states"] = out["states"].float()
    out["present"] = out["present"].bool()
    out["slot_index"] = out["slot_index"].long()
    if all("loss_valid" in r for r in records):
        out["loss_valid"] = torch.as_tensor(
            np.stack([r["loss_valid"] for r in records]), dtype=torch.bool)
    if all("vehicle_ids" in r for r in records):
        out["vehicle_ids"] = torch.as_tensor(
            np.stack([r["vehicle_ids"] for r in records]), dtype=torch.int64)
    if all("cav_teacher" in r for r in records):
        out["cav_teacher"] = torch.as_tensor(
            np.stack([r["cav_teacher"] for r in records]), dtype=torch.float32)
    if all("is_cav" in r for r in records):
        out["is_cav"] = torch.as_tensor(
            np.stack([r["is_cav"] for r in records]), dtype=torch.bool)
    if all("hist" in r for r in records):
        out["hist"] = torch.as_tensor(np.stack([r["hist"] for r in records])).float()
        out["hist_mask"] = torch.as_tensor(np.stack([r["hist_mask"] for r in records])).bool()
    if any("driver_hist" in r for r in records):
        if not all("driver_hist" in r and "driver_hist_mask" in r for r in records):
            raise ValueError("all mixed records must supply configured driver history")
        out["driver_hist"] = torch.as_tensor(np.stack([r["driver_hist"] for r in records])).float()
        out["driver_hist_mask"] = torch.as_tensor(
            np.stack([r["driver_hist_mask"] for r in records])).bool()
    if all("entry_feat" in r for r in records):
        out["entry_feat"] = torch.as_tensor(
            np.stack([r["entry_feat"] for r in records]), dtype=torch.float32)
    if all("controller_id" in r for r in records):
        out["controller_id"] = torch.as_tensor(
            [int(r["controller_id"]) for r in records], dtype=torch.long)
    if all("action_timeline" in r for r in records):
        out["action_timeline"] = {
            key: (torch.as_tensor(np.stack([r["action_timeline"][key] for r in records]))
                  if key != "control_dt_s" else float(records[0]["action_timeline"][key]))
            for key in ("controller", "acceleration_mps2", "target_speed_mps", "control_dt_s")}
    if all("entrant" in r for r in records):
        out["entrant"] = torch.as_tensor(
            np.stack([r["entrant"] for r in records]), dtype=torch.bool)
    out["identities"] = [r.get("identities") for r in records]
    return out


def _state_from_kin(kin: Tensor, present: Tensor) -> dict[str, Tensor]:
    lane = kin[..., 1:4].argmax(-1)
    lane = torch.where(kin[..., 1:4].sum(-1) > 0, lane,
                      torch.full_like(lane, -1))
    return {"veh_kin": kin, "s_m": kin[..., 0], "v_mps": kin[..., 4],
            "a_mps2": kin[..., 5], "length_m": kin[..., 6], "lane": lane,
            "present": present}


GATE_PROBE_K = (0, 1, 2, 5, 10, 20, 40, 80, 160, 240)


class WmSTrainer:
    """One WM-S member; no CAL/HOLDOUT access exists in this class."""

    def __init__(self, model: WmSModel, config: Optional[WmSTrainConfig] = None):
        self.config = config or WmSTrainConfig()
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA is unavailable; WM-S fails closed")
        torch.set_num_threads(int(self.config.threads))
        torch.manual_seed(int(self.config.seed))
        random.seed(int(self.config.seed))
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        if (self.model.history_s != self.config.history_s
                or (self.config.history_s is not None
                    and self.model.history_stride_s != self.config.history_stride_s)):
            raise ValueError("model and trainer history configuration must match")
        # teacher: recorded trajectory drives every CAV (direct commands unused);
        # closed_loop: the full timeline callback drives every CAV.
        self.model.apply_direct_commands = False  # teacher or the full dispatcher owns CAVs
        # Flag-gated run-2 bounded correction (B).  Setting the attributes is a
        # no-op when the config leaves them None.
        if self.config.residual_cap_mps2 is not None:
            self.model.residual_cap_mps2 = float(self.config.residual_cap_mps2)
        if self.config.lane_logit_cap is not None:
            self.model.lane_logit_cap = float(self.config.lane_logit_cap)
        if self.config.residual_cap_neg_mps2 is not None:
            self.model.residual_cap_neg_mps2 = float(self.config.residual_cap_neg_mps2)
        if self.config.residual_soft_bound_mps2 is not None:
            self.model.residual_soft_bound_mps2 = self.config.residual_soft_bound_mps2
            self.model.residual_soft_bound_neg_mps2 = self.config.residual_soft_bound_neg_mps2
        validate_bounds(self.model.residual_soft_bound_mps2,
                        self.model.residual_soft_bound_neg_mps2,
                        self.model.residual_cap_mps2, self.model.residual_cap_neg_mps2)
        self.projection = (
            FeasibilityProjection(ProjectionConfig(
                min_gap_m=float(self.config.projection_min_gap_m)))
            if self.config.projection_enabled else None)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr,
                                          weight_decay=self.config.weight_decay)
        self.history: list[dict] = []

    def _entry_process(self, batch: dict, present: Tensor):
        """Build the seeded entries-v2 process over free padding slots, or None.

        Padding slots are the last ``entry_pad`` columns (empty at t0); entrants
        occupy them, move by the model physics (+ residual) and perturb their
        neighbours, but they have no recorded target so ``loss_valid`` excludes
        them from the per-car loss.  Reusing the same process seed makes a run
        deterministic; the per-window observed-history features keep the
        arrival process data-driven.
        """
        cfg = self.config
        if not cfg.entries_v2 or "entry_feat" not in batch or int(cfg.entry_pad) <= 0:
            return None
        import numpy as _np

        from .entries import EntranceHistory, TorchEntryProcessV2

        b_n, k_n = int(present.shape[0]), int(present.shape[2])
        pad = int(cfg.entry_pad)
        feat = batch["entry_feat"].to(self.device).cpu().numpy()
        if feat.shape[1] != EntranceHistory.defaults().to_features().size:
            raise ValueError("entry_feat has an unexpected width")
        histories = [EntranceHistory.from_features(feat[b]) for b in range(b_n)]
        pool = _np.zeros((b_n, k_n + pad), dtype=bool)
        pool[:, k_n:] = True
        return TorchEntryProcessV2(histories=histories,
                                   dt_s=float(self.model.physics.dt_s),
                                   seed=int(cfg.seed), batch=b_n, pool=pool)

    @staticmethod
    def _extend_for_entries(states, present, is_cav, teacher, hist, hist_mask,
                            loss_valid, pad: int):
        """Append ``pad`` padding slots to every per-slot tensor.

        Returns ``(states, present_init, present_seq, is_cav, teacher, hist,
        hist_mask, loss_valid)``.  ``present_init`` (the t0 snapshot) keeps the
        padding slots empty, while ``present_seq`` marks them present so an
        entrant is not dropped by the recorded presence mask; ``loss_valid`` is
        False there so entrants never enter the supervised loss.
        """
        b_n, t_n = int(present.shape[0]), int(present.shape[1])
        device, dtype = present.device, states.dtype
        zeros_k = torch.zeros(b_n, t_n, pad, dtype=dtype, device=device)
        states = torch.cat((states, zeros_k.unsqueeze(-1).expand(
            -1, -1, -1, states.shape[-1])), dim=2)
        present_init = torch.cat((present, torch.zeros(b_n, t_n, pad,
                                                        dtype=torch.bool, device=device)), dim=2)
        present_seq = torch.cat((present, torch.ones(b_n, t_n, pad,
                                                     dtype=torch.bool, device=device)), dim=2)
        is_cav = torch.cat((is_cav, torch.zeros(b_n, t_n, pad, dtype=torch.bool,
                                                device=device)), dim=2)
        loss_valid = torch.cat((loss_valid, torch.zeros(b_n, t_n, pad,
                                                        dtype=torch.bool, device=device)), dim=2)
        if teacher is not None:
            teacher = torch.cat((teacher, torch.zeros(
                b_n, teacher.shape[1], pad, teacher.shape[-1], dtype=teacher.dtype,
                device=device)), dim=2)
        if hist is not None:
            hist = torch.cat((hist, torch.zeros(b_n, pad, hist.shape[2], hist.shape[3],
                                                dtype=hist.dtype, device=device)), dim=1)
            hist_mask = torch.cat((hist_mask, torch.zeros(
                b_n, pad, hist_mask.shape[2], dtype=torch.bool, device=device)), dim=1)
        return states, present_init, present_seq, is_cav, teacher, hist, hist_mask, loss_valid

    @staticmethod
    def _step_weights(steps: int, dtype, device, decay: float) -> Tensor:
        w = torch.exp(-float(decay) * torch.arange(steps, device=device,
                                                   dtype=dtype))
        return w / w.sum().clamp_min(1.0e-8)

    @staticmethod
    def _driver_param_stats(values: Optional[Tensor],
                            valid: Optional[Tensor]) -> Optional[dict]:
        """Per-driver parameter min/median/max over valid HDV slots at t0.

        Vectorised WM-S2 speed fix: a single boolean gather plus ONE host
        transfer, replacing the per-parameter loop that issued a leading
        ``bool(valid.any())`` sync and 3 device syncs per parameter (28 syncs
        per optimizer step in ``driver_params`` mode).  Numerically identical to
        the element-wise reference; parity is asserted in
        ``tests/test_wms_s2_vectorization_parity.py``.
        """
        if values is None or valid is None:
            return None
        valid = valid.to(values.device).bool()
        gathered = values[valid].reshape(-1, len(PARAMETER_NAMES))
        if gathered.numel() == 0:
            return None
        stats_t = torch.stack((gathered.min(dim=0).values,
                               gathered.median(dim=0).values,
                               gathered.max(dim=0).values), dim=1)
        rows = stats_t.detach().cpu().tolist()
        return {name: {"min": row[0], "median": row[1], "max": row[2]}
                for name, row in zip(PARAMETER_NAMES, rows)}

    @torch.no_grad()
    def driver_recovery_report(self, batch: dict, sidecar, *,
                               slot_keys=None) -> Optional[dict]:
        """Evaluation-only driver-parameter recovery (never used in the loss).

        Runs one forward pass of the driver encoder on the batch's t0 HDV slots
        and scores the predicted parameters + uncertainty against the optional
        ground-truth sidecar.  Returns ``None`` unless ``rollout_mode`` is
        ``driver_params`` and a sidecar is supplied.
        """
        if sidecar is None or self.model.rollout_mode not in (
                "driver_params", "hybrid"):
            return None
        states = batch["states"].to(self.device).float()
        present = batch["present"].to(self.device).bool()
        is_cav = batch.get("is_cav", torch.zeros_like(present)).to(self.device).bool()
        hist = batch.get("hist")
        hist_mask = batch.get("hist_mask")
        if hist is not None:
            hist, hist_mask = hist.to(self.device), hist_mask.to(self.device)
        initial = _state_from_kin(states[:, 0], present[:, 0])
        initial["is_cav"] = is_cav[:, 0]
        dh = batch.get("driver_hist", hist)
        dm = batch.get("driver_hist_mask", hist_mask)
        if dh is not None:
            dh, dm = dh.to(self.device), dm.to(self.device)
        out = self.model(initial, k_steps=0, history=hist, history_mask=hist_mask,
                         driver_history=dh, driver_history_mask=dm)
        if "driver_parameters" not in out:
            return None
        pred = out["driver_parameters"].detach().cpu().numpy()
        unc = out["driver_uncertainty"].detach().cpu().numpy()
        hdv = (present[:, 0] & ~is_cav[:, 0]).cpu().numpy()
        n_truth = len(sidecar.param_order)
        if sidecar.vehicle_ids is None:
            n = min(sidecar.n, pred.shape[1])
            truth = np.broadcast_to(sidecar.params[:n], (pred.shape[0], n, n_truth))
            classes = np.broadcast_to(sidecar.classes[:n], (pred.shape[0], n))
            present_mask = np.broadcast_to(hdv[:, :n], (pred.shape[0], n))
            pred, unc = pred[:, :n], unc[:, :n]
        else:
            keys = batch.get("vehicle_ids") if slot_keys is None else slot_keys
            if keys is None:
                return None
            keys = np.asarray(keys)
            truth = np.full((*pred.shape[:2], n_truth), np.nan, dtype=np.float64)
            classes = np.full(pred.shape[:2], -1, dtype=np.int64)
            present_mask = np.zeros(pred.shape[:2], dtype=bool)
            for b in range(pred.shape[0]):
                t, c, m = align_to_slots(sidecar, keys[b].tolist())
                truth[b], classes[b], present_mask[b] = t, c, m
        present_mask = present_mask & hdv[:, :pred.shape[1]]
        return driver_recovery(pred, unc, truth, classes, present=present_mask)

    def _forward_backward(self, batch: dict, *, unroll_steps: int,
                          backward: bool = True, share: bool = False,
                          _lane_rate_probe: bool = False) -> dict:
        """Chunked truncated BPTT: backward every ``truncation_steps`` steps, so
        activation memory is bounded by one chunk regardless of unroll length.
        All loss terms use the same global valid-count normaliser.

        ``share`` (diagnostic, requires ``backward=False``) additionally returns
        ``loss_share``: each term's share of the total gradient norm, measured on
        a single probe chunk at the traffic-ramp saturation point (so every loss
        term has a live gradient and per-car targets are still present), or the
        last chunk when traffic is off.  One chunk keeps the diagnostic cheap.
        This lets the multi-stage smoke gate detect a single term dominating the
        update.  Logged loss terms are per-chunk means, so they are horizon-
        invariant across curriculum stages."""
        cfg = self.config
        rate_reference = None
        if cfg.lane_rate_weight and (backward or share) and not _lane_rate_probe:
            # Replay identical RNG draws (dropout/state noise), without retaining
            # a whole-horizon graph. Each BPTT chunk receives the derivative of
            # the single whole-batch rate loss, not a mean of chunkwise squares.
            devices = [self.device.index or 0] if self.device.type == "cuda" else []
            with torch.random.fork_rng(devices=devices), torch.no_grad():
                probe = self._forward_backward(batch, unroll_steps=unroll_steps,
                                               backward=False, _lane_rate_probe=backward)
            rate_reference = probe["predicted_lane_change_rate"]
        if share and backward:
            raise ValueError("loss-share logging is a forward-only diagnostic "
                             "(backward=False)")
        states = batch["states"].to(self.device).float()
        present = batch["present"].to(self.device).bool()
        is_cav = batch.get("is_cav", torch.zeros_like(present)).to(self.device).bool()
        teacher = batch.get("cav_teacher")
        if teacher is not None:
            teacher = teacher.to(self.device).float()
        hist = batch.get("hist")
        hist_mask = batch.get("hist_mask")
        if hist is not None:
            hist, hist_mask = hist.to(self.device), hist_mask.to(self.device)
        observed_hist = batch.get("driver_hist", hist)
        observed_hist_mask = batch.get("driver_hist_mask", hist_mask)
        if observed_hist is not None:
            observed_hist = observed_hist.to(self.device)
            observed_hist_mask = observed_hist_mask.to(self.device)
        if self.model.long_history and "driver_hist" not in batch:
            raise ValueError("long-history trainer requires pre-t0 driver_hist")
        loss_valid = batch.get("loss_valid")
        loss_valid = (present if loss_valid is None
                      else loss_valid.to(self.device).bool())
        # entries-v2: append free padding slots, then let observed-history
        # entrants occupy them during the rollout (excluded from the loss).
        entries_proc = self._entry_process(batch, present)
        present_seq = present
        if entries_proc is not None:
            (states, present, present_seq, is_cav, teacher, hist, hist_mask,
             loss_valid) = self._extend_for_entries(
                 states, present, is_cav, teacher, hist, hist_mask, loss_valid,
                  int(cfg.entry_pad))
            if self.model.long_history:
                pad = int(cfg.entry_pad)
                observed_hist = F.pad(observed_hist, (0, 0, 0, 0, 0, pad))
                observed_hist_mask = F.pad(observed_hist_mask, (0, 0, 0, pad))
        n_steps = min(int(unroll_steps), states.shape[1] - 1)
        if n_steps < 1:
            raise ValueError("batch does not contain requested unroll")
        initial = _state_from_kin(states[:, 0], present[:, 0])
        initial["is_cav"] = is_cav[:, 0]
        # WM-S2 t0 state noise (driver_params mode): perturb only t0 HDV slots.
        if (self.model.rollout_mode == "driver_params"
                and (cfg.state_noise_std_m or cfg.state_noise_std_mps)):
            hdv0 = present[:, 0] & ~is_cav[:, 0]
            initial["s_m"] = (initial["s_m"] + torch.randn_like(initial["s_m"])
                              * cfg.state_noise_std_m * hdv0)
            initial["v_mps"] = (initial["v_mps"] + torch.randn_like(initial["v_mps"])
                                * cfg.state_noise_std_mps * hdv0).clamp_min(0)
            initial["veh_kin"] = _state_from_kin(states[:, 0], present[:, 0])["veh_kin"].clone()
            initial["veh_kin"][..., 0] = initial["s_m"]
            initial["veh_kin"][..., 4] = initial["v_mps"]
        controller_id = batch.get("controller_id")
        if controller_id is not None:
            controller_id = controller_id.to(self.device)
        action_timeline = batch.get("action_timeline")
        if action_timeline is not None:
            action_timeline = {key: (value.to(self.device) if torch.is_tensor(value) else value)
                               for key, value in action_timeline.items()}
            if entries_proc is not None:
                # Trainer entrants use the documented native default. Their
                # commands are unused; original columns retain the exact plan.
                action_timeline = dict(action_timeline)
                for key in ("controller", "acceleration_mps2", "target_speed_mps"):
                    value = action_timeline[key]
                    fill = CONTROLLERS.index("native") if key == "controller" else 0
                    padding = value.new_full((*value.shape[:2], int(cfg.entry_pad)), fill)
                    action_timeline[key] = torch.cat((value, padding), dim=2)
        cav_controller = None
        if cfg.cav_drive == "closed_loop":
            if cfg.black_box:
                raise ValueError("closed_loop CAV driving requires structured WM-S")
            from .train_controllers import TimelineControllers

            cav_controller = TimelineControllers(self.model, action_timeline)
            teacher = None
        weights = self._step_weights(n_steps, states.dtype, self.device,
                                     cfg.step_weight_decay).view(1, n_steps, 1)
        # Structured WM-S supervises HDVs only (CAVs are teacher-forced).  The
        # black-box variant supervises every learned mover, CAVs included.
        # ``loss_valid`` keeps a slot through a junction transit but skips the
        # junction step itself (undefined lane target); it falls back to
        # ``present`` when a dataset does not supply it.
        hdv_mask = (loss_valid[:, 1:n_steps + 1] if cfg.black_box
                    else loss_valid[:, 1:n_steps + 1] & ~is_cav[:, 1:n_steps + 1])
        # Run 2 / B1: an optional per-step entrant mask (vehicles entering after
        # t0) is excluded from the per-car loss but kept in the traffic loss
        # below (the rollout's own presence mask counts entrants).
        entrant = entrant_steps(batch.get("entrant"), n_steps, states.shape[1] - 1,
                                self.device)
        if entrant is not None:
            hdv_mask = hdv_mask & ~entrant
        aux_mask = hdv_mask & ~is_cav[:, 1:n_steps + 1] & present[:, :n_steps]
        aux_count = int(aux_mask.sum())
        raw_samples, rate_samples = [], []
        denom = (weights * hdv_mask.to(weights.dtype)).sum().clamp_min(1.0)
        lane_target_all = states[:, 1:n_steps + 1, :, 1:4].argmax(-1)
        sums = {"position": 0.0, "speed": 0.0, "lane": 0.0}
        if cfg.residual_mag_weight:
            sums["residual_mag"] = 0.0
        if cfg.lane_rate_weight:
            sums["lane_rate"] = 0.0
        grad_vectors = {}
        measure = bool(getattr(self, "_capture_loss_shares", False)) and backward
        parameters = [p for p in self.model.parameters() if p.requires_grad]

        def capture(name, term):
            if not measure:
                return
            grads = (torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                     if term.requires_grad else [None] * len(parameters))
            previous = grad_vectors.setdefault(name, [None] * len(parameters))
            for j, grad in enumerate(grads):
                if grad is not None:
                    previous[j] = (grad.detach().clone() if previous[j] is None
                                   else previous[j] + grad.detach())
        entrant_count = 0
        total = 0.0
        # Traffic-level loss config: section geometry comes from the model's
        # frozen s_down frame so training and wms_eval share one contract.
        traffic_cfg = None
        if cfg.traffic_loss:
            traffic_cfg = TrafficLossConfig(
                dt_s=float(self.model.physics.dt_s), bin_s=float(cfg.traffic_bin_s),
                free_speed_mps=float(cfg.traffic_free_speed_mps),
                section_bounds_m=(float(self.model.physics.merge_zone_s0_m),
                                  float(self.model.physics.merge_zone_s1_m)),
                ramp_start_s=float(cfg.traffic_ramp_start_s),
                ramp_end_s=float(cfg.traffic_ramp_end_s),
                weight=float(cfg.traffic_weight))
            sums["traffic"] = 0.0
        chunk = max(1, int(cfg.truncation_steps))
        n_chunks = max(1, (n_steps + chunk - 1) // chunk)
        share_params = [p for p in self.model.parameters() if p.requires_grad]
        share_sq: dict[str, float] = {}
        # Diagnostic probe chunk for the gradient-norm shares: the first chunk at
        # or after the traffic ramp saturates (every loss term active, per-car
        # targets still present), or the last chunk when traffic is off.
        if traffic_cfg is not None:
            ramp_end_steps = int(round(float(traffic_cfg.ramp_end_s)
                                       / float(traffic_cfg.dt_s)))
            share_probe = (ramp_end_steps // chunk) * chunk
        else:
            share_probe = ((n_steps - 1) // chunk) * chunk
        if not 0 <= share_probe < n_steps:
            share_probe = ((n_steps - 1) // chunk) * chunk
        # Braking-fix (flag-gated; nothing below runs unless a term is on).  The
        # recorded transition acceleration is ``states[:, t+1, :, 5]``; the
        # propagation mask is precomputed once for the whole unroll (vectorised
        # pairwise leader search per time chunk, no vehicle loop, no host sync).
        bf = cfg.brakefix_config(dt_s=self.model.physics.dt_s)
        bf_sums: dict[str, Tensor] = {}
        a_rec_all = None
        prop_mask_all = None
        brake_steps = None
        if bf.any_on:
            bf_mask = hdv_mask & ~is_cav[:, 1:n_steps + 1]
            a_rec_all = states[:, 1:n_steps + 1, :, 5]
            brake_steps = (bf_mask & (a_rec_all < -float(bf.decel_threshold_mps2))).sum()
            if bf.prop_on:
                events = leader_brake_events(
                    states[:, :n_steps + 1, :, 0],
                    states[:, :n_steps + 1, :, 1:4].argmax(-1),
                    loss_valid[:, :n_steps + 1], states[:, :n_steps + 1, :, 6],
                    a_rec_all, threshold=float(bf.decel_threshold_mps2),
                    max_gap_m=bf.prop_max_gap_m)
                prop_mask_all = propagation_window(events, bf.window_steps) & bf_mask
        current = initial
        # (D) optional per-step entrant mask supplied by the wms/entry2 branch:
        # [B, n_steps, K], the slots inserted on each step.  ``None`` keeps the
        # no-entries behaviour and the tracker treats appearances as entries.
        entrant_seq = batch.get("entrant_seq")
        if entrant_seq is not None:
            entrant_seq = entrant_seq.to(self.device).bool()
        conservation = (ConservationTracker(present[:, 0],
                                            ConservationConfig(enabled=True))
                        if (cfg.conservation_enabled or cfg.conservation_weight) else None)
        proj_totals: dict[str, float] = {}
        # WM-S2 driver_params: the encoder infers parameters from the observed
        # pre-t0 history on the first chunk; they are fixed through the rollout.
        driver_params = None
        driver_embedding = None
        driver_encoded = None
        if not self.model.long_history:
            observed_hist = hist
            observed_hist_mask = hist_mask
        # Hybrid identifiability diagnostics: the share of the learned HDV
        # acceleration magnitude contributed by physics vs the gated residual,
        # and the resolved residual-magnitude penalty weight (None -> modest
        # default in hybrid mode, 0.0 elsewhere so those modes are unchanged).
        hybrid = self.model.rollout_mode == "hybrid"
        residual_weight = cfg.residual_magnitude_weight
        if residual_weight is None:
            residual_weight = 1.0e-3 if hybrid else 0.0
        residual_weight = float(residual_weight)
        hybrid_phys_mag = None
        hybrid_res_mag = None
        hybrid_mag_count = 0
        for c0 in range(0, n_steps, chunk):
            c1 = min(n_steps, c0 + chunk)
            roll = self.model.rollout(
                current, n_steps=c1 - c0, k0=c0,
                cav_teacher=None if teacher is None else teacher[:, c0:c1 + 1],
                present_seq=present_seq[:, c0:c1 + 1],
                is_cav_seq=is_cav[:, c0:c1 + 1],
                history=hist, history_mask=hist_mask,
                controller_id=controller_id,
                action_timeline=action_timeline,
                cav_controller=cav_controller,
                state_noise_std_m=(float(cfg.state_noise_std_m)
                                   if (backward or _lane_rate_probe) else 0.0),
                state_noise_std_mps=(float(cfg.state_noise_std_mps)
                                      if (backward or _lane_rate_probe) else 0.0),
                projection=self.projection, entries=entries_proc,
                driver_history=observed_hist,
                driver_history_mask=observed_hist_mask,
                driver_params=driver_params,
                driver_embedding=driver_embedding)
            if "entrant" in roll:
                entrant_count = max(entrant_count, int(roll["entrant"].sum()))
            for key, value in (roll.get("projection_stats") or {}).items():
                proj_totals[key] = proj_totals.get(key, 0.0) + float(value)
            if conservation is not None:
                seq = None if entrant_seq is None else entrant_seq[:, c0:c1]
                conservation.track_rollout(roll["masks"], entrant_seq=seq)
            if driver_params is None and "driver_parameters" in roll:
                driver_encoded = roll["driver_parameters"].detach()
                driver_params = {name: driver_encoded[..., i]
                                 for i, name in enumerate(PARAMETER_NAMES)}
            if driver_embedding is None and "driver_embedding" in roll:
                driver_embedding = roll["driver_embedding"].detach()
            if hybrid and roll.get("residual_magnitude") is not None:
                hybrid_res_mag = (roll["residual_magnitude"] if hybrid_res_mag is None
                                  else hybrid_res_mag + roll["residual_magnitude"])
                hybrid_phys_mag = (roll["physics_accel_magnitude"] if hybrid_phys_mag is None
                                   else hybrid_phys_mag + roll["physics_accel_magnitude"])
                hybrid_mag_count += int(roll.get("residual_count", 0))
            w = weights[:, c0:c1]
            m = hdv_mask[:, c0:c1].to(w.dtype)
            pos = F.huber_loss(roll["positions"], states[:, c0 + 1:c1 + 1, :, 0],
                               delta=cfg.position_huber_delta_m, reduction="none")
            spd = F.huber_loss(roll["speeds"], states[:, c0 + 1:c1 + 1, :, 4],
                               delta=cfg.speed_huber_delta_mps, reduction="none")
            if cfg.lane_head == "history":
                prev_lane = states[:, c0:c1, :, 1:4].argmax(-1)
                next_lane = lane_target_all[:, c0:c1]
                action_target = torch.where(
                    next_lane == prev_lane, 0,
                    torch.where(next_lane < prev_lane, 1, 2))
                lane_loss = balanced_lane_loss(
                    roll["lane_logits"], action_target, hdv_mask[:, c0:c1])
                lane = None
            else:
                lane = F.cross_entropy(roll["lane_logits"].reshape(-1, 3),
                                       lane_target_all[:, c0:c1].reshape(-1),
                                       reduction="none").reshape_as(m)
                lane_loss = (lane * w * m).sum() / denom
            comp = {"position": (pos * w * m).sum() / denom,
                    "speed": (spd * w * m).sum() / denom,
                    "lane": lane_loss}
            loss = (cfg.position_weight * comp["position"] + cfg.speed_weight * comp["speed"]
                    + cfg.lane_weight * comp["lane"])
            valid_aux = aux_mask[:, c0:c1]
            raw = roll.get("raw_residual_mps2")
            if raw is None:
                if cfg.residual_mag_weight:
                    raise ValueError("residual-mag loss requires model raw_residual_mps2")
                raw = torch.zeros_like(roll["speeds"])
            change_p = roll.get("lane_change_probabilities")
            if change_p is None:
                # Historical/custom rollout adapters may supply logits only.
                probs = torch.softmax(roll["lane_logits"], dim=-1)
                change_p = 1 - (probs[..., 0] if cfg.lane_head == "history" else
                                probs.gather(-1, torch.cat((current["lane"].unsqueeze(1),
                                    roll["lanes"][:, :-1]), 1).clamp(0, 2).unsqueeze(-1)).squeeze(-1))
            raw_samples.append(raw.detach().masked_select(valid_aux).abs())
            rate_samples.append(change_p.detach().masked_select(valid_aux))
            new_chunk: dict[str, Tensor] = {}
            if cfg.residual_mag_weight:
                # Chunk contribution to one unweighted, whole-batch L2 mean.
                n_valid = int(valid_aux.sum())
                new_chunk["residual_mag"] = residual_mag_loss(
                    raw, valid_aux, cfg.residual_mag_weight) * n_valid / max(aux_count, 1)
            if cfg.lane_rate_weight and rate_reference is not None:
                mass = change_p.masked_select(valid_aux).sum()
                derivative = (2 * cfg.lane_rate_weight * (rate_reference - cfg.lane_rate_target)
                              * mass / max(aux_count, 1))
                value = (cfg.lane_rate_weight * (rate_reference - cfg.lane_rate_target) ** 2
                         * int(valid_aux.sum()) / max(aux_count, 1))
                new_chunk["lane_rate"] = derivative - derivative.detach() + value
            for name, tensor in new_chunk.items():
                loss = loss + tensor
                capture(name, tensor)
                sums[name] += float(tensor.detach())
            bf_chunk: dict[str, Tensor] = {}
            if bf.any_on:
                rec_pos = states[:, c0 + 1:c1 + 1, :, 0]
                rec_spd = states[:, c0 + 1:c1 + 1, :, 4]
                if bf.asym_on or bf.decel_on:
                    bf_chunk.update(braking_terms(
                        bf, pos=pos, spd=spd, pred_pos=roll["positions"],
                        rec_pos=rec_pos, pred_spd=roll["speeds"], rec_spd=rec_spd,
                        a_rec=a_rec_all[:, c0:c1], hdv_mask=bf_mask[:, c0:c1],
                        weights=w, denom=denom,
                        position_weight=float(cfg.position_weight),
                        speed_weight=float(cfg.speed_weight)))
                if bf.prop_on:
                    bf_chunk["propagation"] = float(bf.prop_weight) * propagation_terms(
                        roll["speeds"], realized_acceleration(
                            roll["speeds"], current["v_mps"], self.model.physics.dt_s),
                        rec_spd, a_rec_all[:, c0:c1],
                        prop_mask_all[:, c0:c1], w, denom,
                        speed_delta=float(cfg.speed_huber_delta_mps),
                        accel_delta=float(bf.prop_accel_huber_mps2))
                for name, tensor in bf_chunk.items():
                    loss = loss + tensor
                    capture(name, tensor)
                    bf_sums[name] = (tensor.detach() if name not in bf_sums
                                     else bf_sums[name] + tensor.detach())
            traffic_term = None
            for name, weight in (("position", cfg.position_weight), ("speed", cfg.speed_weight),
                                 ("lane", cfg.lane_weight)):
                capture(name, weight * comp[name])
            if traffic_cfg is not None:
                idx = torch.arange(c0, c1, device=self.device)
                tloss, _ = traffic_bin_loss(
                    roll["positions"], roll["speeds"], roll["masks"],
                    states[:, c0 + 1:c1 + 1, :, 0], states[:, c0 + 1:c1 + 1, :, 4],
                    loss_valid[:, c0 + 1:c1 + 1], step_indices=idx, cfg=traffic_cfg)
                term = scheduled_traffic_weight(idx, cfg=traffic_cfg) * tloss
                traffic_term = term
                loss = loss + term
                capture("traffic", term)
                sums["traffic"] += float(term.detach())
            if hybrid and residual_weight > 0.0 \
                    and roll.get("residual_magnitude") is not None \
                    and int(roll.get("residual_count", 0)) > 0:
                # Identifiability guard: penalise the mean residual magnitude so
                # the encoder cannot be absorbed by the gated residual.
                term = residual_weight * roll["residual_magnitude"] / int(
                    roll["residual_count"])
                loss = loss + term
                sums["residual_penalty"] = (sums.get("residual_penalty", 0.0)
                                            + float(term.detach()))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("non-finite WM-S loss")
            if backward:
                loss.backward()
            for key in ("position", "speed", "lane"):
                sums[key] += float(comp[key].detach())
            if share and c0 == share_probe:
                share_terms = {
                    "position": float(cfg.position_weight) * comp["position"],
                    "speed": float(cfg.speed_weight) * comp["speed"],
                    "lane": float(cfg.lane_weight) * comp["lane"],
                }
                if traffic_cfg is not None:
                    # Preserve historical flags-off diagnostics, including their
                    # term reuse in hybrid mode. On-path uses the traffic tensor.
                    share_terms["traffic"] = traffic_term if bf.any_on else term
                share_terms.update(bf_chunk)
                share_terms.update(new_chunk)
                for name, tensor in share_terms.items():
                    if not tensor.requires_grad:
                        continue
                    grads = torch.autograd.grad(
                        tensor, share_params, retain_graph=True, allow_unused=True)
                    share_sq[name] = share_sq.get(name, 0.0) + sum(
                        float((g.detach() ** 2).sum()) for g in grads if g is not None)
            total += float(loss.detach())
            current = {key: (value.detach() if torch.is_tensor(value) else value)
                       for key, value in roll["state"].items()}
            if entries_proc is None:
                current["is_cav"] = is_cav[:, c1]
                current["length_m"] = initial["length_m"]
            else:
                # Recorded overrides own original slots only. The entry process
                # owns newcomer type/length, including across BPTT boundaries.
                newcomers = current["entrant"].bool()
                current["is_cav"] = torch.where(newcomers, current["is_cav"], is_cav[:, c1])
                current["length_m"] = torch.where(
                    newcomers, current["length_m"], initial["length_m"])
            hist, hist_mask = roll["history"].detach(), roll["history_mask"]
        raw_values = torch.cat(raw_samples)
        rate_values = torch.cat(rate_samples)
        predicted_rate = float(rate_values.mean()) if aux_count else 0.0
        if cfg.lane_rate_weight and rate_reference is None:
            rate_term = (cfg.lane_rate_weight * (predicted_rate - cfg.lane_rate_target) ** 2
                         if aux_count else 0.0)
            sums["lane_rate"] = rate_term
            total += rate_term
        if bf_sums:
            # One host transfer for every braking-fix term and event count.
            names = list(bf_sums)
            values = torch.stack([bf_sums[n] for n in names]
                                 + [brake_steps.to(states.dtype)]
                                 + ([prop_mask_all.sum().to(states.dtype)]
                                    if prop_mask_all is not None else [])).tolist()
            for name, value in zip(names, values):
                sums[name] = float(value)
            sums["brake_steps"] = int(values[len(names)])
            if prop_mask_all is not None:
                sums["prop_steps"] = int(values[len(names) + 1])
        # (B) contractivity regulariser on the residual net (off by default).
        cons_loss = conservation.loss().to(self.device) if conservation is not None else None
        stability = None
        if cfg.stability_weight:
            kin0 = _state_kin(initial)
            stability = stability_regulariser(
                self.model.residual_net, kin0, present[:, 0].bool(),
                margin=float(cfg.stability_margin),
                n_probes=int(cfg.stability_probes))
            if backward:
                capture("stability", float(cfg.stability_weight) * stability)
                (float(cfg.stability_weight) * stability).backward()
        if share:
            if stability is not None:
                tensor = float(cfg.stability_weight) * stability
                share_sq["stability"] = (
                    sum(float((g.detach() ** 2).sum())
                        for g in torch.autograd.grad(
                            tensor, share_params, retain_graph=True,
                            allow_unused=True) if g is not None)
                    if tensor.requires_grad else 0.0)
            if cons_loss is not None and cfg.conservation_weight:
                tensor = float(cfg.conservation_weight) * cons_loss
                share_sq["conservation"] = (
                    sum(float((g.detach() ** 2).sum())
                        for g in torch.autograd.grad(
                            tensor, share_params, retain_graph=True,
                            allow_unused=True) if g is not None)
                    if tensor.requires_grad else 0.0)
        extra = combine_losses(
            torch.zeros((), device=self.device), conservation=cons_loss,
            conservation_weight=float(cfg.conservation_weight),
            stability=stability, stability_weight=float(cfg.stability_weight))
        extra_scalar = float(extra.detach())
        if cfg.conservation_weight:
            capture("conservation", cfg.conservation_weight * cons_loss)
        # Only surface the new keys when their flag is on, so the training
        # receipt is unchanged when the run-2 features are off.
        if cfg.conservation_weight:
            sums["conservation"] = float(cons_loss.detach()) if cons_loss is not None else 0.0
        if cfg.stability_weight:
            sums["stability"] = float(stability.detach()) if stability is not None else 0.0
        # Run-2 stability metrics (log-only; the conservation *metric* can be on
        # with its loss term off).  All values are finite by construction.
        if self.projection is not None:
            denom_present = float(present[:, 1:n_steps + 1].sum().clamp_min(1.0).detach())
            active = sum(proj_totals.get(k, 0.0) for k in
                         ("gap_active", "speed_high", "speed_low", "invalid_lane"))
            sums["projection_active_total"] = float(active)
            sums["projection_active_fraction"] = float(active) / denom_present
            sums["projection_gap_active"] = float(proj_totals.get("gap_active", 0.0))
            sums["projection_s_clamped"] = float(proj_totals.get("s_clamped", 0.0))
            sums["projection_invalid_lane"] = float(proj_totals.get("invalid_lane", 0.0))
            sums["projection_present_changed"] = float(
                proj_totals.get("present_changed", 0.0))
        if conservation is not None:
            sums["conservation_metric"] = (max(conservation.history)
                                           if conservation.history else 0.0)
            sums["conservation_created"] = float(
                conservation.created.max().detach()) if conservation.created.numel() else 0.0
        gates = self.model.gate(torch.arange(n_steps, device=self.device))
        ks = torch.tensor(GATE_PROBE_K, device=self.device)
        curve = [float(x) for x in self.model.gate(ks).detach()]
        # Horizon-invariant loss logging.  ``position``/``speed``/``lane`` are
        # already fractions of the *global* valid-count ``denom``, so summing
        # their per-chunk contributions reconstructs the whole-unroll mean and
        # must NOT be divided by the chunk count.  The traffic term is a mean
        # over 30 s bins per chunk, so summing it scales with the number of
        # chunks (run 2's stage-300 value was ~30x its stage-10 value from the
        # chunk count alone); it is logged per-chunk.  ``stability``/
        # ``conservation`` are computed once per unroll and are not divided.
        traffic_total = sums.get("traffic", 0.0)
        logged = dict(sums)
        loss_logged = total + extra_scalar
        if traffic_cfg is not None:
            # Only surface the traffic key when the term is on, so the off-path
            # record (and the training receipt) stays byte-identical to run 1.
            logged["traffic"] = traffic_total / n_chunks
            loss_logged = (total - traffic_total) + logged["traffic"] + extra_scalar
        record = {"loss": loss_logged, "gate_curve": curve,
                  "predicted_lane_change_rate": predicted_rate,
                  "raw_residual_abs_mean_mps2": float(raw_values.mean()) if aux_count else 0.0,
                  "raw_residual_abs_p95_mps2": float(torch.quantile(raw_values, .95)) if aux_count else 0.0,
                  **logged,
                  "gate_g0": float(gates[0].detach()),
                  "gate_g_last": float(gates[-1].detach()),
                  "gate_monotone": bool(torch.all(gates[1:] >= gates[:-1])),
                  "unroll_steps": int(n_steps), "valid_targets": int(hdv_mask.sum()),
                  "driver_param_stats": self._driver_param_stats(driver_encoded, hdv_mask[:, 0]),
                  "entrants": int(entrant_count),
                  "valid_hdv_slots_t0": int(hdv_mask[:, 0].sum())}
        if traffic_cfg is not None:
            record["chunks"] = int(n_chunks)
        if hybrid and hybrid_mag_count > 0:
            phys_mag = float(hybrid_phys_mag.detach())
            res_mag = float(hybrid_res_mag.detach())
            denom_mag = phys_mag + res_mag
            record["physics_share"] = (phys_mag / denom_mag) if denom_mag > 0.0 else 0.0
            record["residual_share"] = (res_mag / denom_mag) if denom_mag > 0.0 else 0.0
            record["residual_abs_mean_mps2"] = res_mag / max(hybrid_mag_count, 1)
            record["residual_magnitude_weight"] = residual_weight
        if share:
            total_norm = sum(value ** 0.5 for value in share_sq.values())
            record["loss_share"] = {
                name: (value ** 0.5 / total_norm if total_norm > 0.0 else 0.0)
                for name, value in share_sq.items()}
        if measure:
            norms = {name: sum(float(g.square().sum()) for g in grads if g is not None) ** .5
                     for name, grads in grad_vectors.items()}
            denom_norm = max(sum(norms.values()), 1.e-12)
            record.update(loss_share={name: value / denom_norm for name, value in norms.items()},
                          loss_grad_norm=norms)
        return record


    def train_batch(self, batch: dict, *, unroll_steps: int) -> dict:
        started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)
        parts = self._forward_backward(batch, unroll_steps=unroll_steps)
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
               for p in self.model.parameters()):
            raise RuntimeError("non-finite WM-S gradient")
        if self.config.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()
        if any(not bool(torch.isfinite(p).all()) for p in self.model.parameters()):
            raise RuntimeError("non-finite WM-S parameter")
        rec = {**parts, "seconds": time.perf_counter() - started, "finite": True}
        if self.device.type == "cuda":
            rec["cuda_peak_mb"] = torch.cuda.max_memory_allocated(self.device) / 2 ** 20
        return rec

    @torch.no_grad()
    def evaluate(self, dataset: Dataset, *, unroll_steps: int,
                 require_full_horizon: bool = False) -> dict:
        """Forward-only batch means; strict horizons cannot silently truncate."""
        was_training = self.model.training
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False,
                            collate_fn=collate, num_workers=0, drop_last=False)
        total = {"loss": 0.0, "position": 0.0, "speed": 0.0, "lane": 0.0}
        if self.config.traffic_loss:
            total["traffic"] = 0.0
        for name in self.config.brakefix_config().log_terms:
            total[name] = 0.0
        for name, weight in (("residual_mag", self.config.residual_mag_weight),
                             ("lane_rate", self.config.lane_rate_weight)):
            if weight:
                total[name] = 0.0
        total.update({name: 0.0 for name in MONITORS})
        n_batches = 0
        valid_targets = 0
        reached_steps = None
        available_steps = 0
        for batch in loader:
            available_steps = max(available_steps, min(int(unroll_steps), batch["states"].shape[1] - 1))
            if require_full_horizon and batch["states"].shape[1] <= unroll_steps:
                continue  # unavailable horizon is not selection evidence
            parts = self._forward_backward(batch, unroll_steps=int(unroll_steps),
                                           backward=False)
            reached_steps = (parts["unroll_steps"] if reached_steps is None else
                             min(reached_steps, parts["unroll_steps"]))
            for key in total:
                total[key] += float(parts.get(key, 0.0))
            valid_targets += int(parts["valid_targets"])
            n_batches += 1
        if was_training:
            self.model.train()
        mean = {key: (value / n_batches if n_batches else float("nan"))
                for key, value in total.items()}
        return {"n_batches": n_batches, "valid_targets": valid_targets,
                **({"reached_steps": reached_steps, "available_steps": available_steps}
                   if require_full_horizon else {}),
                "unroll_steps": int(unroll_steps), **{f"mean_{k}": v for k, v in mean.items()}}

    @torch.no_grad()
    def monitor(self, dataset: Dataset, *, unroll_steps: int,
                max_batches: Optional[int] = None) -> Optional[float]:
        """Mean forward-only loss over up to ``max_batches`` val batches.

        Monitoring only: no gradient step, no early stopping, no checkpoint
        selection.
        """
        was_training = self.model.training
        self.model.eval()
        loader = DataLoader(dataset, batch_size=self.config.batch_size,
                            shuffle=False, collate_fn=collate, num_workers=0)
        losses = []
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= int(max_batches):
                break
            parts = self._forward_backward(batch, unroll_steps=unroll_steps,
                                           backward=False)
            losses.append(float(parts["loss"]))
        if was_training:
            self.model.train()
        return float(np.mean(losses)) if losses else None

    def fit(self, dataset: Dataset, val_dataset: Optional[Dataset] = None) -> dict:
        if self.config.select_by_long_horizon and val_dataset is None:
            raise ValueError("select-by-long-horizon requires a validation dataset")
        sampler = None
        fraction = float(self.config.switch_window_fraction)
        if fraction > 0.0:
            switched = []
            # Lazy indexed inputs are inspected one window at a time. Retain
            # only switch membership, never a corpus of loaded states/arrays.
            records = (dataset.records if getattr(dataset, "_loader", None) is None
                       else (dataset[i] for i in range(len(dataset))))
            for record in records:
                action = record.get("action_timeline")
                switched.append(bool(action is not None and
                                     np.any(action["controller"][1:] !=
                                            action["controller"][:-1])))
            n_switch, n_fixed = sum(switched), len(switched) - sum(switched)
            if not n_switch:
                raise ValueError("switch_window_fraction requested but dataset has no switches")
            if fraction < 1.0 and not n_fixed:
                raise ValueError("switch_window_fraction requires fixed windows too")
            weights = [((fraction / n_switch) if flag else
                        ((1.0 - fraction) / n_fixed if n_fixed else 0.0))
                       for flag in switched]
            sampler = WeightedRandomSampler(weights, num_samples=len(dataset),
                                             replacement=True)
        def make_loader(stage_sampler):
            return DataLoader(dataset, batch_size=self.config.batch_size,
                              shuffle=stage_sampler is None, sampler=stage_sampler,
                              collate_fn=collate,
                              num_workers=int(self.config.num_workers),
                              persistent_workers=int(self.config.num_workers) > 0,
                              prefetch_factor=(4 if int(self.config.num_workers) > 0
                                               else None),
                              drop_last=False)

        # A mixed dataset (wms.mixdata) supplies per-stage sampling weights
        # (source mix ratio + over-sampling of switch/direct windows visible at
        # that stage's horizon); otherwise one loader serves every stage.
        stage_weights = getattr(dataset, "stage_weights", None)
        loader = make_loader(sampler) if stage_weights is None else None
        output = Path(self.config.out_dir)
        output.mkdir(parents=True, exist_ok=True)
        long_horizon_on = any(self.config.to_dict().get(key) for key in
                              ("residual_soft_bound_mps2", "residual_mag_weight",
                               "lane_rate_weight", "select_by_long_horizon"))
        if long_horizon_on:
            (output / "train_config.json").write_text(
                json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        if self.config.brakefix_config().any_on:
            (output / "BRAKEFIX_CONFIG.json").write_text(
                json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
        log_path = output / "train_log.jsonl"
        log_path.write_text("", encoding="utf-8")
        val_log_path = output / "val_log.jsonl"
        if val_dataset is not None:
            val_log_path.write_text("", encoding="utf-8")
        share_log_path = output / "loss_share.jsonl"
        if self.config.loss_share_logging:
            share_log_path.write_text("", encoding="utf-8")
        val_by_stage: list[dict] = []
        global_step = 0
        val_by_stage = []
        last_batch = None
        selector = LongHorizonSelection()

        def validate_long(stage_s):
            rows = []
            for horizon_s in validation_horizons(stage_s):
                horizon_steps = max(1, round(horizon_s / self.model.physics.dt_s))
                row = self.evaluate(val_dataset, unroll_steps=horizon_steps,
                                    require_full_horizon=True)
                if row["n_batches"] == 0 and row.get("available_steps", 0) > 0:
                    horizon_steps = row["available_steps"]
                    horizon_s = horizon_steps * self.model.physics.dt_s
                    row = self.evaluate(val_dataset, unroll_steps=horizon_steps,
                                        require_full_horizon=True)
                if any(r["horizon_s"] == horizon_s for r in rows):
                    continue
                row.update(stage_s=float(stage_s), step=global_step,
                           horizon_s=horizon_s, selection=False, early_stop=False)
                row["selection"] = selector.consider(row)
                if row["selection"]:
                    torch.save({"format": "wm_s/1", "model": self.model.state_dict(),
                                "config": self.config.to_dict(), "stage_s": float(stage_s),
                                "step": global_step, "val": row}, output / "wm_s_best.pt")
                val_by_stage.append(row)
                rows.append(row)
                with val_log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
            return rows[-1]

        for stage_i, stage_s in enumerate(self.config.unroll_curriculum_s):
            steps = max(1, round(float(stage_s) / self.model.physics.dt_s))
            done = 0
            target = (self.config.stage_steps[stage_i] if self.config.stage_steps
                      else self.config.steps_per_epoch * self.config.epochs_per_stage)
            steps_before_stage = global_step
            last_validation = None
            if stage_weights is not None:
                loader = None
                gen = torch.Generator().manual_seed(int(self.config.seed) + 1000 * stage_i)
                loader = make_loader(WeightedRandomSampler(
                    torch.as_tensor(stage_weights(float(stage_s)), dtype=torch.double),
                    num_samples=len(dataset), replacement=True, generator=gen))
            while done < target:
                for batch in loader:
                    last_batch = batch
                    # Closed-loop diagnostics share the actual training pass so
                    # stateful controllers are not replayed a second time.
                    self._capture_loss_shares = (self.config.loss_share_logging
                                                and self.config.cav_drive == "closed_loop"
                                                and done == 0)
                    record = self.train_batch(batch, unroll_steps=steps)
                    record.update({"stage_s": float(stage_s),
                                   "epoch": stage_i + 1})
                    global_step += 1
                    record["step"] = global_step
                    self.history.append(record)
                    with log_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record, sort_keys=True) + "\n")
                    if "loss_share" in record:
                        with (output / "loss_share.jsonl").open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps({key: record[key] for key in
                                ("stage_s", "step", "loss_share", "loss_grad_norm")}) + "\n")
                    done += 1
                    if self.config.select_by_long_horizon and (
                            global_step % self.config.val_every == 0 or done >= target):
                        last_validation = validate_long(stage_s)
                    if done >= target:
                        break
            if global_step == steps_before_stage:
                raise RuntimeError("training stage completed zero optimizer steps")
            stage_record = None
            if self.config.select_by_long_horizon:
                stage_record = last_validation
            elif val_dataset is not None:
                # Monitoring only: log the per-stage validation loss, then keep
                # training the fixed budget.  No early stopping and no checkpoint
                # selection is derived from this value.
                stage_record = self.evaluate(val_dataset, unroll_steps=steps)
                stage_record.update({"stage_s": float(stage_s), "step": global_step,
                                     "selection": False, "early_stop": False})
            if (self.config.loss_share_logging and last_batch is not None
                    and self.config.cav_drive != "closed_loop"):
                # Diagnostic-only per-stage gradient-norm shares.  Written to a
                # dedicated file so the multi-stage smoke can log them without a
                # validation set (the val log is only opened when one exists).
                share_parts = self._forward_backward(
                    last_batch, unroll_steps=steps, backward=False, share=True)
                share_record = {"stage_s": float(stage_s), "step": global_step,
                                "unroll_steps": int(steps),
                                "loss_share": share_parts.get("loss_share")}
                with share_log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(share_record, sort_keys=True) + "\n")
                if stage_record is not None:
                    stage_record["loss_share"] = share_record["loss_share"]
            if stage_record is not None and not self.config.select_by_long_horizon:
                val_by_stage.append(stage_record)
                with val_log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(stage_record, sort_keys=True) + "\n")
            torch.save({"format": "wm_s/1", "model": self.model.state_dict(),
                        "config": self.config.to_dict(), "stage_s": float(stage_s),
                        "step": global_step, "val": stage_record},
                       output / f"wm_s_stage{int(stage_s)}.pt")
            # Optional after-each-stage snapshot (env hook): copy the just-saved
            # stage checkpoint with a timestamp so stages can be scored while
            # training continues.  Unset env == no copy (byte-identical run).
            snapshot_dir = os.environ.get("WM_S_STAGE_CKPT_COPY_DIR")
            if snapshot_dir:
                import shutil

                dest = Path(snapshot_dir)
                dest.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%dT%H%M%S")
                shutil.copy2(output / f"wm_s_stage{int(stage_s)}.pt",
                             dest / f"wm_s_stage{int(stage_s)}_{stamp}.pt")
        checkpoint = output / "wm_s.pt"
        torch.save({"format": "wm_s/1", "model": self.model.state_dict(),
                    "config": self.config.to_dict(), "history": self.history,
                    "val_by_stage": val_by_stage}, checkpoint)
        receipt = {"schema": "wm_s_train_receipt/1", "split": "TRAIN",
                   "device": str(self.device), "threads": self.config.threads,
                   "curriculum_s": list(self.config.unroll_curriculum_s),
                   "checkpoint": str(checkpoint), "steps": len(self.history),
                   "val_by_stage": val_by_stage, "log": str(log_path),
                   "physics": asdict(self.model.physics),
                   "config": self.config.to_dict(),
                   "val_monitoring": (None if val_dataset is None else {
                       "mode": "monitor-only", "early_stopping": False,
                       "checkpoint_selection": False,
                       "log": str(val_log_path), "by_stage": val_by_stage,
                       "data": getattr(val_dataset, "receipt", {})}),
                   "data": getattr(dataset, "receipt", {})}
        if self.config.select_by_long_horizon:
            receipt["long_horizon_selection"] = {
                "enabled": True, "metric": "mean_position", "minimum_stage_s": 60.0,
                "policy": "longest-eligible-horizon-then-lowest-mean-position",
                "val_every": self.config.val_every, "best": selector.best,
                "checkpoint": str(output / "wm_s_best.pt") if selector.best else None,
                "longest_horizon_s": selector.horizon_s,
                "final_checkpoint_policy": "last-optimizer-step"}
            receipt["val_monitoring"].update(mode="long-horizon-selection",
                                             checkpoint_selection=True)
        (output / "TRAIN_RECEIPT.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return receipt


def dry_run(dataset: Dataset, *, model: Optional[WmSModel] = None,
            config: Optional[WmSTrainConfig] = None) -> dict:
    """Two CPU updates at the registered 10 s stage plus 10 s / 30 s profiles."""
    cfg = config or WmSTrainConfig(device="cpu", steps_per_epoch=2,
                                   unroll_curriculum_s=(10.0,))
    cfg = WmSTrainConfig.from_dict({**cfg.to_dict(), "device": "cpu",
                                    "steps_per_epoch": 2, "stage_steps": None,
                                    "unroll_curriculum_s": [10.0]})
    trainer = WmSTrainer(model or WmSModel(
        black_box=cfg.black_box, n_controllers=cfg.n_controllers,
        controller_embed_dim=cfg.controller_embed_dim,
        checkpoint_residual=cfg.checkpoint_residual,
        rollout_mode=cfg.rollout_mode, lane_head=cfg.lane_head,
        history_s=cfg.history_s, history_stride_s=cfg.history_stride_s), cfg)
    loader = DataLoader(dataset, batch_size=min(4, len(dataset)), shuffle=False,
                        collate_fn=collate, num_workers=0)
    batch = next(iter(loader))
    if batch["states"].shape[1] < 21:
        raise ValueError("CPU dry run requires 10 s (21 state boundaries)")
    records = [trainer.train_batch(batch, unroll_steps=20) for _ in range(2)]
    profiles = []
    for seconds in (10.0, 30.0):
        steps = round(seconds / trainer.model.physics.dt_s)
        if batch["states"].shape[1] <= steps:
            continue
        rss_before = _process_rss_mb()
        trainer.optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        trainer._forward_backward(batch, unroll_steps=steps)
        elapsed = time.perf_counter() - started
        profiles.append({"unroll_s": seconds, "steps": steps,
                         "fwd_bwd_time_s": elapsed,
                         "fwd_bwd_time_per_sim_step_ms": 1000.0 * elapsed / steps,
                         "rss_before_mb": rss_before,
                         "rss_after_mb": _process_rss_mb(),
                         "peak_working_set_mb_cumulative": _peak_rss_mb()})
    recovery = None
    if cfg.driver_sidecar:
        from .driver_sidecar import load_driver_sidecar
        recovery = trainer.driver_recovery_report(
            batch, load_driver_sidecar(cfg.driver_sidecar))
    result = {"status": "passed", "updates": records, "unroll_s": 10.0,
              "batch_size": int(batch["states"].shape[0]),
              "threads": cfg.threads, "split": "TRAIN", "optimizer_steps": 2,
              "black_box": bool(cfg.black_box),
              "driver_recovery": recovery,
              "profiles": profiles, "data": getattr(dataset, "receipt", {})}
    output = Path(cfg.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "wm_s_train_receipt/1", **result}
    (output / "TRAIN_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "train_log.jsonl").write_text(
        "".join(json.dumps({"step": i + 1, **row}, sort_keys=True) + "\n"
                for i, row in enumerate(records)), encoding="utf-8")
    return receipt
