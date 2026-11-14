"""CPU, kinematics-only TRAIN audit; no fitting, outcomes or checkpoint reads.

Global lower acceleration envelopes are necessary bounds, NOT state-dependent
cap coverage. Needed corrections use teacher-forced recorded states and fixed
default IDM parameter scenarios, NOT an inferred/trained driver encoder.
The destination acceleration channel is ASSUMED to label the preceding step;
this audit has not verified that alignment on real data.

Input: by default workspace-local, link-free wm_n1_pack/1 directories, with INDEX run
ranges and .npy arrays. A wm_s_brakefix_synthetic/1 manifest with the same array
layout explicitly labels synthetic evidence. Only selected TRAIN ranges are
sliced. No identities/outcome sidecars are required: slot continuity relies on
target_mask, as in IdeaACorpus.window. Canonical identity is not verified.
The explicit --readonly-external-packs option supports a coordinator's separately
authorized real-data audit; this execution agent does not invoke that option.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import stat
import sys

import numpy as np
import torch

from .brakefix import leader_brake_events, min_accel_reach, propagation_window
from .driver_encoder import PARAMETER_BOUNDS, PARAMETER_NAMES
from .model import PhysicsConfig, WmSModel
from . import lpo

WORKSPACE = Path(__file__).absolute().parents[4]
DECEL_DEFAULT = PhysicsConfig(sumo_fidelity=True).idm()[1]
DECEL_ENC_BOUND = PARAMETER_BOUNDS[PARAMETER_NAMES.index("decel")]
SETTINGS = (
    ("residual / no cap", "residual", DECEL_DEFAULT, None, None),
    ("residual / cap 1.5", "residual", DECEL_DEFAULT, 1.5, None),
    ("driver_params / defaults", "driver_params", DECEL_DEFAULT, None, None),
    ("hybrid / no cap", "hybrid", DECEL_DEFAULT, None, None),
    ("hybrid / cap 1.5", "hybrid", DECEL_DEFAULT, 1.5, None),
    ("hybrid / cap 1.5 / decel floor", "hybrid", DECEL_DEFAULT - DECEL_ENC_BOUND, 1.5, None),
    ("hybrid / cap 1.5 / decel ceiling", "hybrid", DECEL_DEFAULT + DECEL_ENC_BOUND, 1.5, None),
    ("hybrid / negative cap 3", "hybrid", DECEL_DEFAULT, 1.5, 3.0),
    ("hybrid / negative cap 3 / decel floor", "hybrid", DECEL_DEFAULT - DECEL_ENC_BOUND, 1.5, 3.0),
    ("hybrid / negative cap 4.5", "hybrid", DECEL_DEFAULT, 1.5, 4.5),
)
BRAKE_THRESHOLDS = (1.5, 2.0, 3.0, 3.5, 4.0, 4.5)
CORRECTION_CAPS = (1.0, 1.5, 2.0, 3.0, 4.5)
HIST_EDGES = np.arange(-12.0, 4.0 + 1e-9, 0.5)
PHYSICS_SCENARIOS = {"residual_defaults": None,
                     "driver_defaults": DECEL_DEFAULT,
                     "driver_decel_floor": DECEL_DEFAULT - DECEL_ENC_BOUND,
                     "driver_decel_ceiling": DECEL_DEFAULT + DECEL_ENC_BOUND}
LIMITATIONS = (
    "Acceleration alignment: states[t+1,:,5] is an assumption; real-data alignment is NOT verified here.",
    "Global lower-envelope inclusion is necessary only; it is not recorded-state cap coverage or learned-model skill.",
    "Needed corrections are one-step teacher-forced, fixed-default IDM scenarios; all other driver parameters stay at defaults. No checkpoint or encoder inference.",
    "Encoder decel 2.7..5.7 is a default-centred outer envelope, not observed recovery: shrinkage narrows it and tanh endpoints are asymptotic; supplied vType defaults can differ.",
    "Negative-cap diagnostics ignore positive-side capacity. Tanh cap equality is a limiting boundary, not a guaranteed finite-weight correction; gate(0)=0 permits no residual.",
    "Physics uses IDM clips [-9,3], driver decel/accel clamps and ramp-end obstacle; merge_cal and injected physics are excluded.",
    "Lane-logit cap constrains logit adjustments, not braking. Baseline MOBIL has safety-masked logits; residual adjustments can change argmax. History lane head replaces these logits and has separate feasibility/age gates.",
    "Projection bounds post-step speed/lane/position but does not rewrite accel_mps2. Its effect, lane changes, speed-floor effective acceleration and rollout drift are not measured by this audit.",
    "Propagation counts are recorded nearest-leader masks, not causal wave validation; windows truncate at their ends and overlapping sampled windows can count steps repeatedly.",
    "TRAIN metadata/target_mask continuity is checked, not canonical vehicle identity, source authenticity or independently registered split membership.",
    "old_hist_chain.sh uses driver_params with no residual, projection or stability; cap changes do not affect that acceleration path. Its full command supplies a VAL window list despite the TRAIN-only header.",
)


def workspace_path(path: str | Path, *, new: bool = False) -> Path:
    """Refuse external paths and every link/reparse ancestor before file reads."""
    p = Path(os.path.abspath(path))
    try:
        relative = p.relative_to(WORKSPACE)
    except ValueError as exc:
        raise PermissionError(f"outside workspace: {p}") from exc
    current = WORKSPACE
    for part in (None, *relative.parts):
        if part is not None:
            current = current / part
        try:
            info = current.lstat()  # inspect the link itself, never its target
        except FileNotFoundError:
            if new and current == p:
                break
            raise
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise PermissionError(f"link/reparse path prohibited: {current}")
    return p


def _json(path: Path) -> dict:
    return json.loads(workspace_path(path).read_text(encoding="utf-8"))


def _split_guard(record: dict) -> str:
    split = str(record.get("split", "")).strip().upper()
    chunk = re.sub(r"[\s\-_.]+", "", str(record.get("chunk", ""))).lower()
    if split not in ("TRAIN", "CALIBRATION") or chunk.startswith("chunk5"):
        raise PermissionError("unknown/sealed split or chunk in metadata")
    return split


def _manifest_dtype(value) -> np.dtype:
    """Normalise a manifest ``dtype`` token to a numpy dtype.

    Real packs spell the token two ways: pack_v3 stores the canonical name
    (``"float16"``) while pack_dspeed/pack_damping store ``str(np.float16)``
    (``"<class 'numpy.float16'>"``).  The trainer's ``MemmapCorpus._array``
    reads the array without comparing the token at all, so both spellings are
    accepted here; an unknown or malformed token still fails closed.
    """
    text = str(value).strip()
    match = re.fullmatch(r"<class 'numpy\.([A-Za-z0-9_]+)'>", text)
    if match:
        text = match.group(1)
    try:
        return np.dtype(text)
    except TypeError as exc:
        raise ValueError(f"unrecognised manifest dtype: {value!r}") from exc


class AuditPack:
    """Minimal .npy reader: avoids _dataset's outcome/identity-sidecar access.

    All files actually opened are path-guarded; INDEX source-path strings are
    metadata only and are never followed. Array mappings are lazy, read-only.
    """

    def __init__(self, path: str | Path, *, readonly_external: bool = False):
        self.readonly_external = readonly_external
        self.root = Path(os.path.abspath(path)) if readonly_external else workspace_path(path)
        from .train import validate_train_source
        validate_train_source(self.root)
        self.manifest = json.loads(self.input_path("manifest.json").read_text(encoding="utf-8"))
        schema = self.manifest.get("schema")
        if schema not in ("wm_n1_pack/1", "wm_s_brakefix_synthetic/1"):
            raise ValueError("unsupported audit pack schema")
        self.synthetic = schema == "wm_s_brakefix_synthetic/1"
        items = self.manifest.get("items")
        if not isinstance(items, list):
            raise ValueError("manifest items must be a list")
        self.items = [r for r in items if _split_guard(r) == "TRAIN"]
        index = json.loads(self.input_path("INDEX.json").read_text(encoding="utf-8"))
        runs = index.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError("INDEX run ranges are required")
        self.runs = sorted(runs, key=lambda r: int(r["tick_start"]))
        previous_end = 0
        for run in self.runs:
            _split_guard(run)
            start, end = int(run["tick_start"]), int(run["tick_end"])
            if start < previous_end or end <= start:
                raise ValueError("invalid/overlapping INDEX ranges")
            previous_end = end
        self._mm = {}
        for name in ("veh_kin", "veh_mask", "target_mask", "veh_step"):
            spec = self.manifest["arrays"][name]
            file = Path(spec["file"])
            if file.is_absolute() or ".." in file.parts or file.suffix != ".npy":
                raise PermissionError("array must be a pack-local .npy file")
            self.input_path(file)
        for item in self.items:
            self.run_for(item)  # metadata-only: prove TRAIN before array access

    def input_path(self, filename: str | Path) -> Path:
        """Only pack-local inputs; external mode never relaxes the output guard."""
        name = Path(filename)
        if name.is_absolute() or ".." in name.parts:
            raise PermissionError("input must be pack-local")
        path = self.root / name
        return path if self.readonly_external else workspace_path(path)

    def run_for(self, item: dict) -> dict:
        tick = int(item["tick"])
        found = [r for r in self.runs if int(r["tick_start"]) <= tick < int(r["tick_end"])]
        if len(found) != 1 or _split_guard(found[0]) != "TRAIN":
            raise PermissionError("TRAIN item has no unique TRAIN INDEX range")
        run = found[0]
        for key in ("world", "arm", "chunk", "batch"):
            if key in run and str(run[key]) != str(item.get(key)):
                raise ValueError(f"item/INDEX identity mismatch: {key}")
        return run

    def array(self, name: str) -> np.memmap:
        if name not in self._mm:
            spec = self.manifest["arrays"][name]
            arr = np.load(self.input_path(spec["file"]),
                          mmap_mode="r", allow_pickle=False)
            if not isinstance(arr, np.memmap) or list(arr.shape) != spec["shape"] \
                    or arr.dtype != _manifest_dtype(spec["dtype"]):
                raise ValueError(f"array metadata mismatch: {name}")
            self._mm[name] = arr
        return self._mm[name]

    def window(self, item: dict, horizon_steps: int) -> dict:
        run = self.run_for(item)
        tick = int(item["tick"])
        end = int(run["tick_end"])
        kin = self.array("veh_kin")
        mask, tm, step = (self.array(n) for n in ("veh_mask", "target_mask", "veh_step"))
        if kin.ndim != 3 or kin.shape[-1] != 7 or kin.shape[1] <= 0 \
                or mask.shape != kin.shape[:2] or tm.shape != mask.shape \
                or step.shape[:2] != mask.shape or step.shape[-1] < 8 \
                or int(self.runs[-1]["tick_end"]) > kin.shape[0]:
            raise ValueError("invalid kinematic array shapes / INDEX endpoint")
        stop = min(end, tick + horizon_steps + 1)
        states = np.array(kin[tick:stop], dtype=np.float32, copy=True)
        # Same t0 roster as IdeaACorpus.window; never resurrect a reused slot.
        alive = np.concatenate((np.asarray(mask[tick], bool)[None],
                                np.asarray(tm[tick:stop - 1], bool)), axis=0)
        alive = np.logical_and.accumulate(alive, axis=0)
        valid = alive & (states[..., 1:4].sum(-1) > 0)
        cav = np.broadcast_to(np.asarray(step[tick, :, 7] > 0.5), valid.shape)
        return {"states": states, "loss_valid": valid, "is_cav": cav}


def _ratio(count: int | float, total: int) -> float | None:
    return count / total if total else None


class Accumulator:
    """Count vectors only; tensors are limited to chunk + propagation lookback."""

    def __init__(self) -> None:
        self.windows = self.hdv_steps = self.correction_steps = 0
        self.min_a = None
        self.hist = np.zeros(len(HIST_EDGES) - 1, dtype=np.int64)
        self.underflow = self.overflow = 0
        self.below = {t: 0 for t in BRAKE_THRESHOLDS}
        self.reach = {s[0]: min_accel_reach(rollout_mode=s[1], decel_i=s[2],
                                          cap_pos=s[3], cap_neg=s[4]) for s in SETTINGS}
        self.unreachable = {s[0]: {t: 0 for t in BRAKE_THRESHOLDS} for s in SETTINGS}
        self.prop_steps = {t: 0 for t in BRAKE_THRESHOLDS}
        self.interval_total = {s[0]: 0 for s in SETTINGS}
        self.interval_reachable = {s[0]: {t: 0 for t in BRAKE_THRESHOLDS} for s in SETTINGS}
        self.alignment_n = 0
        self.alignment_moments = np.zeros(5, dtype=np.float64)
        self.alignment_abs_sum = self.alignment_abs_max = 0.0
        self.need = {name: {t: {"steps": 0, "sum": 0.0,
                               "exceeds": {c: 0 for c in CORRECTION_CAPS},
                               "boundary": {c: 0 for c in CORRECTION_CAPS},
                               "below_final_clip": 0}
                           for t in BRAKE_THRESHOLDS} for name in PHYSICS_SCENARIOS}

    @staticmethod
    def _physics(model, states: np.ndarray, valid: np.ndarray, decel: float | None):
        kin = torch.tensor(states, dtype=torch.float32)
        present = torch.tensor(valid, dtype=torch.bool)
        params = None
        if decel is not None:
            p = model.physics
            a, _, headway, gap, _ = p.idm()
            defaults = (p.v0_mps, headway, gap, a, decel, 0.2,
                        p.mobil_threshold_mps2, p.lane_safe_gap_m, 1.0)
            params = {n: torch.full(kin.shape[:2], value)
                      for n, value in zip(PARAMETER_NAMES, defaults)}
        return model.physics_accel({"veh_kin": kin, "present": present}, params).numpy()

    def add(self, states: np.ndarray, loss_valid: np.ndarray, is_cav: np.ndarray,
            window_steps: int = 6, max_gap_m: float | None = None, model=None,
            chunk: int = 60) -> None:
        if chunk < 1 or window_steps < 1 or (max_gap_m is not None and
                                           (not np.isfinite(max_gap_m) or max_gap_m <= 0)):
            raise ValueError("chunk, window_steps and max_gap_m must be positive")
        if states.ndim != 3 or states.shape[-1] != 7 \
                or loss_valid.shape != states.shape[:2] or is_cav.shape != loss_valid.shape:
            raise ValueError("expected states [T+1,K,7] and aligned validity/CAV masks")
        if model is not None and (model.physics != PhysicsConfig(sumo_fidelity=True)
                                  or model.physics_step is not None):
            raise ValueError("needed-correction audit supports default SUMO-fidelity in-module physics only")
        self.windows += 1
        if states.shape[1] == 0:
            return
        n_steps = max(0, states.shape[0] - 1)
        with torch.no_grad():
            for t0 in range(0, n_steps, chunk):
                t1 = min(n_steps, t0 + chunk)  # source excludes final destination row
                lo = max(0, t0 - window_steps)
                st = np.array(states[lo:t1 + 1], dtype=np.float32, copy=True)
                vd = np.asarray(loss_valid[lo:t1 + 1], bool)
                if not np.isfinite(st[vd]).all():
                    raise ValueError("non-finite valid kinematics")
                # Invalid slots can contain NaNs; they must not poison physics.
                st[~vd] = 0.0
                offset = t0 - lo
                a = st[offset + 1:, :, 5].astype(np.float64)
                hdv = (vd[offset:-1] & vd[offset + 1:]
                       & ~np.asarray(is_cav[t0:t1], bool)
                       & ~np.asarray(is_cav[t0 + 1:t1 + 1], bool))
                x = a[hdv]
                difference = ((st[offset + 1:, :, 4] - st[offset:-1, :, 4]) / 0.5)[hdv].astype(np.float64)
                self.alignment_n += x.size
                self.alignment_moments += [x.sum(), difference.sum(), (x*x).sum(),
                                           (difference*difference).sum(), (x*difference).sum()]
                if x.size:
                    err = np.abs(x - difference)
                    self.alignment_abs_sum += err.sum()
                    self.alignment_abs_max = max(self.alignment_abs_max, float(err.max()))
                self.hdv_steps += x.size
                if x.size:
                    self.min_a = float(x.min()) if self.min_a is None else min(self.min_a, float(x.min()))
                    self.hist += np.histogram(x, bins=HIST_EDGES)[0]
                    self.underflow += int((x < HIST_EDGES[0]).sum())
                    self.overflow += int((x > HIST_EDGES[-1]).sum())
                for t in BRAKE_THRESHOLDS:
                    self.below[t] += int((x < -t).sum())
                    for name, reach in self.reach.items():
                        self.unreachable[name][t] += int(((x < -t) & (x < reach)).sum())
                if model is not None and hdv.any():
                    self.correction_steps += int(x.size)
                    physics_scenarios = {}
                    for name, decel in PHYSICS_SCENARIOS.items():
                        physics = self._physics(model, st[offset:-1], vd[offset:-1], decel)
                        physics_scenarios[decel] = physics.astype(np.float64)
                        delta = (a - physics.astype(np.float64))[hdv]
                        for t in BRAKE_THRESHOLDS:
                            selected = x < -t
                            row = self.need[name][t]
                            row["steps"] += int(selected.sum())
                            row["sum"] += float(delta[selected].sum())
                            row["below_final_clip"] += int((x[selected] < model.physics.accel_min_mps2).sum())
                            for cap in CORRECTION_CAPS:
                                row["exceeds"][cap] += int((delta[selected] < -cap).sum())
                                row["boundary"][cap] += int(np.isclose(delta[selected], -cap, atol=1e-6, rtol=0).sum())
                    for name, mode, decel, pos, neg in SETTINGS:
                        physics = physics_scenarios[None if mode == "residual" else decel]
                        lower = (physics.copy() if mode == "driver_params" else
                                 physics - (neg if neg is not None else pos)
                                 if (neg is not None or pos is not None) else
                                 np.full_like(physics, model.physics.accel_min_mps2))
                        upper = (physics.copy() if mode == "driver_params" else
                                 physics + pos if pos is not None else
                                 np.full_like(physics, model.physics.accel_max_mps2))
                        lower = np.maximum(lower, model.physics.accel_min_mps2)
                        upper = np.minimum(upper, model.physics.accel_max_mps2)
                        # At k=0 the residual gate is exactly zero in every mode.
                        if t0 == 0:
                            lower[0] = upper[0] = physics[0]
                        reachable = ((a >= lower) & (a <= upper))[hdv]
                        self.interval_total[name] += int(x.size)
                        for threshold in BRAKE_THRESHOLDS:
                            self.interval_reachable[name][threshold] += int((reachable & (x < -threshold)).sum())
                tensor = torch.tensor(st[None])
                for t in BRAKE_THRESHOLDS:
                    events = leader_brake_events(tensor[..., 0], tensor[..., 1:4].argmax(-1),
                                                 torch.tensor(vd[None]), tensor[..., 6],
                                                 tensor[:, 1:, :, 5], threshold=t,
                                                 max_gap_m=max_gap_m)
                    mask = propagation_window(events, window_steps)[:, offset:]
                    self.prop_steps[t] += int((mask & torch.tensor(hdv[None])).sum())

    def report(self) -> dict:
        settings = []
        for name, mode, decel, pos, neg in SETTINGS:
            settings.append({"setting": name, "rollout_mode": mode, "decel_i": decel,
                             "cap_pos": pos, "cap_neg": neg,
                              "global_lower_envelope_mps2": self.reach[name],
                              "recorded_state_raw_accel_interval_share_of_braking": {
                                  f"a<-{t:g}": _ratio(self.interval_reachable[name][t], self.below[t])
                                  if self.interval_total[name] else None for t in BRAKE_THRESHOLDS},
                             "within_global_lower_envelope_share_of_braking": {
                                 f"a<-{t:g}": _ratio(self.below[t] - self.unreachable[name][t], self.below[t])
                                 for t in BRAKE_THRESHOLDS}})
        needed = {}
        for name, thresholds in self.need.items():
            needed[name] = {}
            for t, row in thresholds.items():
                needed[name][f"a<-{t:g}"] = {
                    "steps": row["steps"],
                    "mean_signed_needed_correction_mps2": _ratio(row["sum"], row["steps"]),
                    "share_below_final_accel_clip": _ratio(row["below_final_clip"], row["steps"]),
                    "share_negative_correction_exceeding_cap": {
                        f"{c:g}": _ratio(row["exceeds"][c], row["steps"]) for c in CORRECTION_CAPS},
                    "at_negative_tanh_boundary_steps_tol_1e-6": {
                        f"{c:g}": row["boundary"][c] for c in CORRECTION_CAPS}}
        # Clipped tails get explicit tail labels, not invented 0.5-bin quantiles.
        cdf = self.underflow + np.cumsum(self.hist)
        quantiles = {}
        for q in (0.001, 0.005, 0.01, 0.02, 0.05, 0.5):
            rank = max(1, int(np.ceil(q * self.hdv_steps)))
            i = int(np.searchsorted(cdf, rank))
            quantiles[f"q{q:g}"] = (None if not self.hdv_steps else
                                    {"below": float(HIST_EDGES[0])} if rank <= self.underflow else
                                    {"above": float(HIST_EDGES[-1])} if i >= len(self.hist) else
                                    {"bin": [float(HIST_EDGES[i]), float(HIST_EDGES[i + 1])]})
        n = self.alignment_n
        sx, sy, sxx, syy, sxy = self.alignment_moments
        corr_denom = max(0.0, n*sxx-sx*sx) * max(0.0, n*syy-sy*sy)
        correlation = (n*sxy-sx*sy) / np.sqrt(corr_denom) if corr_denom > 0 else None
        return {"windows": self.windows, "hdv_steps": int(self.hdv_steps),
                "acceleration_alignment_diagnostics": {
                    "samples": int(n), "channel_vs_speed_fd_correlation": correlation,
                    "mean_abs_difference_mps2": _ratio(self.alignment_abs_sum, n),
                    "max_abs_difference_mps2": self.alignment_abs_max if n else None,
                    "dt_s": 0.5},
                "needed_correction_evaluated_hdv_steps": self.correction_steps,
                "min_recorded_accel_mps2": self.min_a,
                "steps_below": {f"-{t:g}": self.below[t] for t in BRAKE_THRESHOLDS},
                "share_steps_below": {f"-{t:g}": _ratio(self.below[t], self.hdv_steps) for t in BRAKE_THRESHOLDS},
                "quantiles_mps2_binned_0.5": quantiles,
                "prop_window_hdv_steps_by_threshold": {f"-{t:g}": self.prop_steps[t] for t in BRAKE_THRESHOLDS},
                "prop_window_share_of_hdv_steps": {f"-{t:g}": _ratio(self.prop_steps[t], self.hdv_steps) for t in BRAKE_THRESHOLDS},
                "needed_negative_correction": needed, "settings": settings,
                "histogram": {"edges": HIST_EDGES.tolist(), "counts": self.hist.tolist(),
                              "underflow": self.underflow, "overflow": self.overflow},
                "acceleration_alignment": "assumed_destination_channel_not_verified_real",
                "limitations": list(LIMITATIONS)}


def pick_indices(n_records: int, n_windows: int) -> list[int]:
    """Evenly spaced records; small samples do NOT guarantee every pack is covered."""
    if n_records < 0 or n_windows < 0:
        raise ValueError("sample counts must be nonnegative")
    if not n_records or not n_windows:
        return []
    return sorted(set(int(i) for i in np.linspace(0, n_records - 1, min(n_windows, n_records))))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Workspace-only TRAIN braking audit (CPU, no fitting)")
    p.add_argument("--pack", action="append", required=True)
    p.add_argument("--readonly-external-packs", action="store_true",
                   help="coordinator only: read explicitly supplied external/junction pack roots; outputs stay workspace-local")
    p.add_argument("--window-list")
    p.add_argument("--exclude-window-list", action="append", default=[])
    p.add_argument("--n-windows", type=int, default=60)
    p.add_argument("--horizon-steps", type=int, default=240)
    p.add_argument("--chunk-steps", type=int, default=60)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--prop-window-s", type=float, default=3.0)
    p.add_argument("--out")
    args = p.parse_args(argv)
    if args.n_windows < 0 or min(args.horizon_steps, args.chunk_steps, args.threads) < 1 \
            or not np.isfinite(args.prop_window_s) or args.prop_window_s <= 0:
        p.error("nonnegative n-windows and positive finite bounds are required")
    out = workspace_path(args.out, new=True) if args.out else None
    if out is not None and out.exists():
        raise FileExistsError(f"audit output already exists: {out}")
    torch.set_num_threads(args.threads)
    packs = [AuditPack(path, readonly_external=args.readonly_external_packs) for path in args.pack]
    names = [pack.root.name for pack in packs]
    if len(set(names)) != len(names):
        raise ValueError("pack basenames must be unique for window-list matching")
    items, pack_ids = [], []
    for i, pack in enumerate(packs):
        items.extend(pack.items)
        pack_ids.extend([i] * len(pack.items))
    keep = None
    if args.window_list:
        path = workspace_path(args.window_list)
        data = lpo.load_window_manifest(path)
        if data.get("split", "TRAIN") != "TRAIN" \
                or any(w.get("split", "TRAIN") != "TRAIN" for w in data["windows"]):
            raise PermissionError("window list must be TRAIN")
        if any(str(w.get("pack")) not in names for w in data["windows"]):
            raise ValueError("window list references an unprovided pack")
        keep = lpo.window_keys(path, packs=names)
    excluded = set()
    for path in args.exclude_window_list:
        excluded |= lpo.exclude_runs(workspace_path(path))
    selected, receipt = lpo.filter_positions(items, pack_ids, names, keep=keep, exclude=excluded)
    indices = pick_indices(len(selected), args.n_windows)
    acc = Accumulator()
    model = WmSModel(physics=PhysicsConfig(sumo_fidelity=True), rollout_mode="hybrid").cpu().eval()
    counts = {name: 0 for name in names}
    steps = max(1, int(round(args.prop_window_s / model.physics.dt_s)))
    for index in indices:
        position = selected[index]
        pack_id = pack_ids[position]
        acc.add(**packs[pack_id].window(items[position], args.horizon_steps),
                window_steps=steps, chunk=args.chunk_steps, model=model)
        counts[names[pack_id]] += 1
    rep = acc.report()
    synthetic = all(pack.synthetic for pack in packs)
    rep.update({"schema": "wm_s_brakefix_audit/2", "split": "TRAIN",
                "evidence_kind": "synthetic_only" if synthetic else "recorded_or_mixed_unverified_provenance",
                "real_data_alignment_verified": False, "device": "cpu",
                "records_in_corpus": len(items), "records_after_filter": len(selected),
                "sampled_filtered_indices": indices, "sampled_windows_by_pack": counts,
                "window_filter": receipt, "horizon_steps": args.horizon_steps,
                "chunk_steps": args.chunk_steps, "prop_window_steps": steps,
                "sources": [{"pack": str(pack.root), "synthetic_declared": pack.synthetic} for pack in packs]})
    text = json.dumps(rep, indent=1, sort_keys=True, allow_nan=False)
    if out is not None:
        with workspace_path(out, new=True).open("x", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
