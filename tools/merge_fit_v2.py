"""Offline CPU fit of merge-decision candidate models M0-M3 on recorded TRAIN episodes.

The fitter builds per-ramp-car whole-wait sequences from the recorded C4 TRAIN-val
starts (``recs_native.pkl`` / ``recs_genrand.pkl``), reusing the deployed core
features (:func:`wms.model.merge_core.core_features`) and the per-car sequence
survival semantics of :mod:`wms.train.losses.merge_core_seqtf`:

    S_q(t) = prod_{u < t} (1 - h_uq),      h_uq = g(context_u) sigmoid(m_uq / T)
    event at e:    -logsumexp_q(log w_q + log S_q(e) + log h_eq)
    censored at c: -logsumexp_q(log w_q + sum_{u <= c} log(1 - h_uq))

with the persistent per-car assertiveness ``alpha_q = exp(sigma z_q)`` integrated
by Gauss-Hermite quadrature.  ``g`` is the learned unblocked gate of M3 (identically
1 for M0-M2) and the margin ``m`` is the deployed core margin plus the model's learned
context term (zero for M0 and M3).

Candidate models
----------------
* ``M0`` - deployed core: ``s0, T, b, tau_a, temperature, sigma``.
* ``M1`` - core + linear context terms in the margin.
* ``M2`` - core + small tanh MLP on the context in the margin.
* ``M3`` - learned sigmoid "unblocked" gate ``g(context)`` multiplying the core hazard.

The temperature is fitted by default (``--fit-temperature``) or frozen at the deployed
initialisation (``--fix-temperature`` / ``--temperature VALUE``).  Every behavioural
quantity is learned; the only constants are the deployed geometry, the quadrature order
and the recorded-data split.

Data discipline
---------------
Evaluation seeds 964000-964199 are refused.  Paths containing ``sealed``, ``holdout``,
``heldout``, ``cal_split``, ``chunk5``, ``j4b`` or ``inc1`` are refused (sources,
optional LCSTATE directory, outputs).  No driver truth tables are read.  The split is
world-grouped: sorted unique worlds, every fifth held out.

Usage
-----
    python tools/merge_fit_v2.py --models M0,M1,M2,M3 --out fit.json --md fit.md
    python tools/merge_fit_v2.py --models M0 --max-worlds 8 --out smoke.json
    python tools/merge_fit_v2.py --source v5_native=DIR --source genrand_plain=DIR \
        --lcstate E:/research-dense/runs/active/C4-LCSTATE-20261021/data

``recorded_context`` (agent A, ``wms/model/merge_context.py``) is imported lazily.
Without it, M0 still fits; requesting M1-M3 fails with an explicit message rather than
silently substituting hand-built context.  ``--context-module`` may point at a compatible
module for integration tests.
"""
from __future__ import annotations

import argparse
import copy
import csv
import gzip
import importlib
import json
import math
import os
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from wms.model.merge_core import DEFAULT_INIT, core_features  # noqa: E402
from wms.model.merge_frailty import gauss_hermite  # noqa: E402
from wms.train.losses.base import UnrollTargets  # noqa: E402
from wms.train.losses.merge_core import _core_margin  # noqa: E402
from wms.train.losses.merge_core_seqtf import build_recorded_scene  # noqa: E402

DT_DEFAULT = 0.5
MERGE_LANE = 2
TARGET_LANE = 1
ZONE_S0_M = 0.0
ZONE_S1_M = 75.44
BODY_LENGTH_M = 5.0
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
EVAL_SEEDS = range(964000, 964200)
DEFAULT_DATA = "E:/research-dense/runs/active/C4-V7FT-20261017/merge_calib/data"
DEFAULT_SOURCES = (f"v5_native={DEFAULT_DATA}", f"genrand_plain={DEFAULT_DATA}")
DEFAULT_OUT = "runs/active/C4-V7FT-20261017/merge_sumo_diag/merge_fit_v2.json"
DEFAULT_MD = "runs/active/C4-V7FT-20261017/merge_sumo_diag/merge_fit_v2.md"
DEFAULT_CONTEXT_MODULE = "wms.model.merge_context"
MODELS = ("M0", "M1", "M2", "M3")
CORE_FEATURES = ("gap_l", "gap_f", "v", "v_l", "v_f", "a", "a_l", "a_f", "has_l", "has_f")


def guard(path) -> Path:
    """Refuse restricted paths (data discipline is enforced at every open/write)."""
    text = str(path).replace("\\", "/").lower()
    if any(token in text for token in FORBIDDEN):
        raise SystemExit(f"refusing restricted path {path}")
    return Path(path)


def check_seed(seed) -> int:
    seed = int(seed)
    if seed in EVAL_SEEDS:
        raise SystemExit(f"evaluation seed {seed} refused")
    return seed


def _inv_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("softplus init must be positive")
    return value if value > 20.0 else math.log(math.expm1(value))


def _softplus(raw: Tensor) -> Tensor:
    return F.softplus(raw) + 1.0e-6


# --------------------------------------------------------------------------------------
# Record loading (shared layout with tools/merge_core_fit.py)
# --------------------------------------------------------------------------------------
_PICKLE_CACHE: dict[Path, list[dict]] = {}


def _parse_sources(items) -> list[tuple[str, Path]]:
    sources = []
    for item in items:
        name, sep, directory = str(item).partition("=")
        if not sep or not name.strip() or not directory.strip():
            raise SystemExit(f"--source expects NAME=DIR, got {item!r}")
        sources.append((name.strip(), guard(Path(directory.strip()))))
    return sources


def _load_pickle(path: Path) -> list[dict]:
    path = guard(path)
    cached = _PICKLE_CACHE.get(path)
    if cached is None:
        with path.open("rb") as fh:
            cached = pickle.load(fh)
        _PICKLE_CACHE[path] = cached
    return cached


def load_records(sources: list[tuple[str, Path]]) -> tuple[list[dict], list[str]]:
    """Records of every ``NAME=DIR`` source, deduplicated by record key.

    ``DIR`` may be the pickles directory itself or a source directory whose parent holds
    ``recs_native.pkl`` / ``recs_genrand.pkl`` (the C4 layout).  Records are matched to
    ``NAME`` by their recorded ``source`` field, then by the key; if the directory is an
    explicit single-source export without a matching tag every record is used.
    """
    records: list[dict] = []
    seen: set[str] = set()
    notes: list[str] = []
    for name, directory in sources:
        candidates: list[Path] = []
        for base in (directory, directory.parent):
            for filename in ("recs_native.pkl", "recs_genrand.pkl"):
                path = base / filename
                if path.is_file():
                    candidates.append(path.resolve())
        token = name.lower()
        preferred = []
        if any(t in token for t in ("native", "v5")):
            preferred = [p for p in candidates if p.name == "recs_native.pkl"]
        elif any(t in token for t in ("genrand", "generic")):
            preferred = [p for p in candidates if p.name == "recs_genrand.pkl"]
        candidates = sorted(set(preferred or candidates))
        if not candidates:
            notes.append(f"{name}: no recs_*.pkl under {directory} or its parent")
            continue
        pool: list[dict] = []
        for path in candidates:
            pool.extend(_load_pickle(path))
        matched = [r for r in pool if str(r.get("source", "")).lower() == token]
        if not matched:
            matched = [r for r in pool if token in str(r.get("key", "")).lower()]
        if not matched:
            matched = pool
            notes.append(f"{name}: no record source tag match; using all {len(pool)} records")
        added = 0
        for record in matched:
            key = str(record.get("key", id(record)))
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
            added += 1
        notes.append(f"{name}: {added} records from {[p.name for p in candidates]}")
    if not records:
        raise SystemExit("no recorded starts found")
    return records, notes


def _frames(record: dict) -> list[dict]:
    arm = str(record.get("arm", "native"))
    frames = record.get("recorded", {}).get(arm)
    if not frames:
        frames = next(iter(record.get("recorded", {}).values()), None)
    if not frames:
        raise ValueError(f"{record.get('key')}: no recorded frames")
    return frames


def record_targets(record: dict) -> tuple[UnrollTargets, list[str], float]:
    """Recorded frames -> ``UnrollTargets`` (same construction as the seq-TF tests)."""
    dt = float(record.get("dt_s", DT_DEFAULT))
    frames = _frames(record)
    ids: list[str] = []
    index: dict[str, int] = {}
    for frame in frames:
        for row in frame["vehicles"]:
            if row["id"] not in index:
                index[row["id"]] = len(ids)
                ids.append(row["id"])
    n_frames, n_slots = len(frames), len(ids)
    s = np.zeros((n_frames, n_slots), np.float32)
    v = np.zeros_like(s)
    lane = np.full((n_frames, n_slots), -1, np.int64)
    present = np.zeros((n_frames, n_slots), bool)
    is_cav = np.zeros((n_frames, n_slots), bool)
    for t, frame in enumerate(frames):
        for row in frame["vehicles"]:
            j = index[row["id"]]
            if row.get("present", True):
                present[t, j] = True
                s[t, j] = float(row["s"])
                v[t, j] = float(row["v"])
                lane[t, j] = int(row["lane"])
                is_cav[t, j] = bool(row.get("is_cav", False))
    accel = np.clip(np.where(present, np.gradient(v, dt, axis=0), 0.0), -8.0, 4.0)
    n = n_frames - 1
    states = torch.zeros(1, n_frames, n_slots, 7)
    states[0, :, :, 0] = torch.from_numpy(s)
    for lane_id in range(3):
        states[0, :, :, 1 + lane_id] = torch.from_numpy((lane == lane_id).astype(np.float32))
    states[0, :, :, 4] = torch.from_numpy(v)
    states[0, :, :, 5] = torch.from_numpy(accel.astype(np.float32))
    states[0, :, :, 6] = BODY_LENGTH_M
    targets = UnrollTargets(
        states=states, n_steps=n,
        weights=torch.ones(1, n, 1), hdv_mask=torch.ones(1, n, n_slots),
        aux_mask=torch.ones(1, n, n_slots), aux_count=n * n_slots,
        denom=torch.tensor(1.0),
        lane_target=torch.from_numpy(lane[:, 1:]).unsqueeze(0),
        loss_valid=torch.from_numpy(present).unsqueeze(0),
        is_cav=torch.from_numpy(is_cav).unsqueeze(0),
        dt_s=dt, position_delta_m=5.0, speed_delta_mps=1.0, history_lane_head=False,
    )
    return targets, ids, dt


# --------------------------------------------------------------------------------------
# Context (agent A) - lazy import, no local re-derivation of the contract
# --------------------------------------------------------------------------------------
def load_context(context_module: str = DEFAULT_CONTEXT_MODULE):
    """Return ``(recorded_context, CONTEXT_FEATURES, module_name)`` or ``(None, (), name)``."""
    try:
        module = importlib.import_module(context_module)
    except ModuleNotFoundError:
        return None, (), context_module
    fn = getattr(module, "recorded_context", None)
    names = getattr(module, "CONTEXT_FEATURES", None)
    if fn is None or not names:
        return None, (), context_module
    return fn, tuple(str(n) for n in names), context_module


def _call_context(fn, kin: Tensor, present: Tensor, is_cav: Tensor, dt: float) -> dict:
    try:
        return fn(kin, present, is_cav, dt=dt)
    except TypeError:
        return fn(kin, present, is_cav)


# --------------------------------------------------------------------------------------
# Per-car sequences
# --------------------------------------------------------------------------------------
def sequences_from_scene(scene: dict, *, context=None, context_names=(), meta=None):
    """Extract one whole-wait sequence per supervised ramp car from a recorded scene.

    ``scene`` is the output of :func:`wms.train.losses.merge_core_seqtf.build_recorded_scene`
    (batch size 1).  The returned masks are exactly the seq-TF survival / event rows, so the
    M0 likelihood on these sequences reproduces ``seq_tf_chunk``.
    """
    meta = {} if meta is None else dict(meta)
    dt = float(scene.get("dt_s", DT_DEFAULT))
    n_steps = int(scene["n_steps"])
    n_slots = int(scene["gap_l"].shape[-1])
    step = torch.arange(n_steps)
    context_names = tuple(context_names)
    sequences: list[dict] = []
    zero_exposure = 0
    unsupported_events = 0
    for slot in range(n_slots):
        identity = int(scene["identity"][0, slot])
        if identity < 0:
            continue
        feasible = scene["feasible"][0, :, slot]
        supervised = scene["supervised"][0, :, slot]
        same_car = scene["run_start"][0, :, slot] == identity
        supported = feasible & supervised & same_car
        rec_event = int(scene["rec_event"][0, slot])
        last_obs = int(scene["last_obs"][0, slot])
        event_ok = rec_event >= 0 and bool(supported[rec_event])
        before = (step < rec_event) if rec_event >= 0 else torch.ones_like(supervised)
        survival = supported & (step <= last_obs) & before
        at_event = supported & (step == rec_event) & event_ok
        keep = survival | at_event
        if not bool(keep.any()):
            zero_exposure += 1
            continue
        if rec_event >= 0 and not event_ok:
            unsupported_events += 1
        lo = int(torch.nonzero(keep, as_tuple=False)[0])
        hi = int(torch.nonzero(keep, as_tuple=False)[-1])
        sequence = {
            "survival": survival[lo:hi + 1].clone(),
            "event": at_event[lo:hi + 1].clone(),
            "support": (survival | at_event)[lo:hi + 1].clone(),
            "step": torch.arange(lo, hi + 1),
            "recorded_event_step": rec_event,
            "terminal_step": rec_event if event_ok else last_obs,
            "zone_entry_step": lo,
            "is_merged": bool(at_event.any()),
            "is_censored": not bool(at_event.any()),
            "unsupported_event": bool(rec_event >= 0 and not event_ok),
            "slot": slot,
            "dt_s": dt,
        }
        for name in CORE_FEATURES:
            sequence[name] = scene[name][0, lo:hi + 1, slot].float().clone()
        if context is not None and context_names:
            values, valid = [], []
            for name in context_names:
                raw = context.get(name)
                if raw is None:
                    values.append(torch.zeros(hi - lo + 1))
                    valid.append(torch.zeros(hi - lo + 1, dtype=torch.bool))
                    continue
                # ``recorded_context`` may return [1, N, K] (or [1, N+1, K]).
                raw = torch.as_tensor(raw).float()
                if raw.shape[1] < n_steps:
                    pad = torch.zeros(raw.shape[0], n_steps - raw.shape[1], raw.shape[2])
                    raw = torch.cat((raw, pad), dim=1)
                value = raw[0, lo:hi + 1, slot]
                value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                flag = context.get(f"{name}_valid")
                if flag is None:
                    ok = torch.ones_like(value, dtype=torch.bool)
                else:
                    flag = torch.as_tensor(flag).bool()
                    if flag.shape[1] < n_steps:
                        pad = torch.zeros(flag.shape[0], n_steps - flag.shape[1], flag.shape[2],
                                          dtype=torch.bool)
                        flag = torch.cat((flag, pad), dim=1)
                    ok = flag[0, lo:hi + 1, slot] & value.isfinite()
                values.append(value)
                valid.append(ok)
            sequence["ctx"] = torch.stack(values, dim=-1) if values else torch.zeros(
                hi - lo + 1, 0)
            sequence["ctx_valid"] = torch.stack(valid, dim=-1) if valid else torch.zeros(
                hi - lo + 1, 0, dtype=torch.bool)
        for key, value in meta.items():
            sequence[key] = value
        sequences.append(sequence)
    stats = {"zero_exposure": zero_exposure, "unsupported_events": unsupported_events}
    return sequences, stats


def build_dataset(records: list[dict], *, context_fn=None, context_names=(), verbose=True):
    """Whole-wait sequences of every record plus descriptive statistics."""
    sequences: list[dict] = []
    stats = {"zero_exposure": 0, "unsupported_events": 0, "records": 0,
             "missing_context": 0}
    for idx, record in enumerate(records):
        seed = check_seed(record.get("seed", -1))
        world = str(record.get("world", record.get("key", "?")))
        source = str(record.get("source", "?"))
        key = str(record.get("key", "?"))
        targets, ids, dt = record_targets(record)
        n = int(targets.n_steps)
        scene = build_recorded_scene(targets, merge_lane=MERGE_LANE, zone_s0_m=ZONE_S0_M,
                                     zone_s1_m=ZONE_S1_M, k=int(targets.states.shape[2]),
                                     target_lane=TARGET_LANE)
        context = None
        if context_fn is not None:
            kin = targets.states[:, :n]
            present = targets.loss_valid[:, :n]
            cav = targets.is_cav[:, :n]
            context = _call_context(context_fn, kin, present, cav, dt)
            context = {str(name): torch.as_tensor(value).detach()
                       for name, value in dict(context).items()}
        meta = {"world": world, "source": source, "key": key, "seed": seed,
                "vehicle_ids": list(ids)}
        new, one = sequences_from_scene(scene, context=context, context_names=context_names,
                                        meta=meta)
        for sequence in new:
            sequence["vehicle_id"] = ids[int(sequence["slot"])]
        sequences.extend(new)
        stats["records"] += 1
        stats["zero_exposure"] += one["zero_exposure"]
        stats["unsupported_events"] += one["unsupported_events"]
        if verbose and (idx + 1) % 20 == 0:
            print(f"  built {idx + 1}/{len(records)} records, {len(sequences)} cars",
                  flush=True)
    return sequences, stats


def pad_sequences(sequences: list[dict], n_context: int) -> dict:
    """Pad per-car sequences into tensors ``[C, L]`` (``ctx`` ``[C, L, F]``)."""
    count = len(sequences)
    if count == 0:
        raise SystemExit("no supervised ramp cars found")
    length = max(int(seq["survival"].numel()) for seq in sequences)
    batch: dict = {}
    for name in CORE_FEATURES:
        dtype = torch.bool if name in ("has_l", "has_f") else torch.float32
        fill = False if dtype is torch.bool else 0.0
        out = torch.full((count, length), fill, dtype=dtype)
        for i, seq in enumerate(sequences):
            out[i, :seq[name].numel()] = torch.as_tensor(seq[name], dtype=dtype)
        batch[name] = out
    for name in ("survival", "event", "support"):
        out = torch.zeros(count, length, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            out[i, :seq[name].numel()] = seq[name].bool()
        batch[name] = out
    step = torch.full((count, length), -1, dtype=torch.long)
    for i, seq in enumerate(sequences):
        step[i, :seq["step"].numel()] = seq["step"].long()
    batch["step"] = step
    if n_context:
        ctx = torch.zeros(count, length, n_context)
        valid = torch.zeros(count, length, n_context, dtype=torch.bool)
        for i, seq in enumerate(sequences):
            n = int(seq["survival"].numel())
            if "ctx" in seq and seq["ctx"].numel():
                ctx[i, :n] = seq["ctx"]
                valid[i, :n] = seq["ctx_valid"]
        batch["ctx"], batch["ctx_valid"] = ctx, valid
    else:
        batch["ctx"] = torch.zeros(count, length, 0)
        batch["ctx_valid"] = torch.zeros(count, length, 0, dtype=torch.bool)
    batch["dt_s"] = float(sequences[0]["dt_s"])
    batch["is_merged"] = torch.tensor([bool(s["is_merged"]) for s in sequences])
    batch["is_censored"] = torch.tensor([bool(s["is_censored"]) for s in sequences])
    batch["unsupported_event"] = torch.tensor(
        [bool(s["unsupported_event"]) for s in sequences])
    batch["recorded_event_step"] = torch.tensor(
        [int(s["recorded_event_step"]) for s in sequences], dtype=torch.long)
    batch["zone_entry_step"] = torch.tensor(
        [int(s["zone_entry_step"]) for s in sequences], dtype=torch.long)
    batch["world"] = [str(s["world"]) for s in sequences]
    batch["source"] = [str(s["source"]) for s in sequences]
    batch["vehicle_id"] = [str(s["vehicle_id"]) for s in sequences]
    batch["key"] = [str(s["key"]) for s in sequences]
    batch["n_context"] = int(n_context)
    return batch


def standardize_context(batch: dict, fit_mask: Tensor) -> dict:
    """Per-feature mean/std on fit cars; invalid rows become zero after scaling."""
    n_context = int(batch["n_context"])
    if n_context == 0:
        return {"mean": [], "std": []}
    ctx = batch["ctx"]
    valid = batch["ctx_valid"] & batch["support"].unsqueeze(-1)
    use = valid & fit_mask.view(-1, 1, 1)
    count = use.sum(dim=(0, 1)).clamp_min(1)
    mean = (ctx * use).sum(dim=(0, 1)) / count
    var = (((ctx - mean) ** 2) * use).sum(dim=(0, 1)) / count
    std = var.sqrt().clamp_min(1.0e-6)
    scaled = torch.where(valid, (ctx - mean) / std, torch.zeros_like(ctx))
    # Invalid rows stay zero so a missing quantity contributes no learned offset.
    scaled = torch.where(batch["support"].unsqueeze(-1), scaled, torch.zeros_like(scaled))
    batch["ctx"] = scaled
    return {"mean": mean.tolist(), "std": std.tolist()}


# --------------------------------------------------------------------------------------
# Candidate models
# --------------------------------------------------------------------------------------
class MergeModel(nn.Module):
    """One candidate decision model (M0-M3) with the deployed core parameterisation."""

    def __init__(self, kind: str, context_features=(), *, fit_temperature: bool = True,
                 temperature: float | None = None, hidden: int = 16, seed: int = 0):
        super().__init__()
        if kind not in MODELS:
            raise ValueError(f"unknown model {kind}")
        self.kind = kind
        self.context_features = tuple(context_features)
        self.fit_temperature = bool(fit_temperature)
        torch.manual_seed(int(seed))
        init = DEFAULT_INIT
        self.raw_s0 = nn.Parameter(torch.tensor(_inv_softplus(init["s0_m"])))
        self.raw_T = nn.Parameter(torch.tensor(_inv_softplus(init["T_s"])))
        self.raw_b = nn.Parameter(torch.tensor(_inv_softplus(init["b_mps2"])))
        self.raw_tau = nn.Parameter(torch.tensor(_inv_softplus(init["tau_a_s"])))
        self.raw_sigma = nn.Parameter(torch.tensor(
            _inv_softplus(init["alpha_log_sigma"])))
        temp = float(init["temperature_m"] if temperature is None else temperature)
        if self.fit_temperature:
            self.raw_temperature = nn.Parameter(torch.tensor(_inv_softplus(temp)))
        else:
            self.register_buffer("fixed_temperature", torch.tensor(temp))
        n_features = len(self.context_features)
        if kind in ("M1", "M2", "M3") and n_features == 0:
            raise ValueError(f"{kind} needs context features")
        if kind == "M1":
            self.context_linear = nn.Linear(n_features, 1)
            nn.init.zeros_(self.context_linear.weight)
            nn.init.zeros_(self.context_linear.bias)
        elif kind == "M2":
            self.context_mlp = nn.Sequential(
                nn.Linear(n_features, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            nn.init.zeros_(self.context_mlp[2].weight)
            nn.init.zeros_(self.context_mlp[2].bias)
        elif kind == "M3":
            self.gate = nn.Linear(n_features, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def core(self) -> dict[str, Tensor]:
        return {"s0": _softplus(self.raw_s0), "T": _softplus(self.raw_T),
                "b": _softplus(self.raw_b), "tau_a": _softplus(self.raw_tau),
                "sigma": _softplus(self.raw_sigma)}

    @property
    def temperature(self) -> Tensor:
        if self.fit_temperature:
            return _softplus(self.raw_temperature) + 1.0e-3
        return self.fixed_temperature.clamp_min(1.0e-3)

    def context_offset(self, batch: dict) -> Tensor:
        ctx = batch["ctx"]
        if self.kind == "M1":
            return self.context_linear(ctx).squeeze(-1)
        if self.kind == "M2":
            return self.context_mlp(ctx).squeeze(-1)
        return torch.zeros(ctx.shape[0], ctx.shape[1])

    def gate_prob(self, batch: dict) -> Tensor:
        if self.kind != "M3":
            return torch.ones(batch["support"].shape[0], batch["support"].shape[1])
        return torch.sigmoid(self.gate(batch["ctx"]).squeeze(-1))

    def hazard_terms(self, batch: dict, n_quad: int = 8):
        """``(log_h, log1m, margin, weights)`` with ``[C, Q, L]`` hazard tensors."""
        core = self.core()
        nodes, weights = gauss_hermite(int(n_quad))
        alpha = torch.exp((core["sigma"] * nodes).clamp(-8.0, 8.0)).view(1, -1, 1)
        args = (batch["gap_l"].unsqueeze(1), batch["gap_f"].unsqueeze(1),
                batch["v"].unsqueeze(1), batch["v_l"].unsqueeze(1),
                batch["v_f"].unsqueeze(1), batch["a"].unsqueeze(1),
                batch["a_l"].unsqueeze(1), batch["a_f"].unsqueeze(1),
                batch["has_l"].unsqueeze(1), batch["has_f"].unsqueeze(1))
        margin = _core_margin(alpha, *args, core["s0"], core["T"], core["b"],
                              core["tau_a"], None, None)
        margin = margin + self.context_offset(batch).unsqueeze(1)
        logit = margin / self.temperature
        if self.kind == "M3":
            gate = self.gate_prob(batch).unsqueeze(1)
            hazard = (gate * torch.sigmoid(logit)).clamp(0.0, 1.0 - 1.0e-6)
            log_h = torch.log(hazard)
            log1m = torch.log1p(-hazard)
        else:
            log_h = F.logsigmoid(logit)
            log1m = F.logsigmoid(-logit)
        return log_h, log1m, margin, weights

    def batch_nll(self, batch: dict, n_quad: int = 8) -> Tensor:
        log_h, log1m, _, weights = self.hazard_terms(batch, n_quad)
        survival = batch["survival"].unsqueeze(1)
        event = batch["event"].unsqueeze(1)
        g = (survival * log1m).sum(dim=-1) + (event * log_h).sum(dim=-1)
        log_weights = torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))
        return -torch.logsumexp(g + log_weights.view(1, -1), dim=1)

    def deterministic_margin(self, batch: dict) -> Tensor:
        """Core + learned margin offset at ``alpha = 1`` (gate excluded)."""
        core = self.core()
        args = (batch["gap_l"], batch["gap_f"], batch["v"], batch["v_l"], batch["v_f"],
                batch["a"], batch["a_l"], batch["a_f"], batch["has_l"], batch["has_f"])
        one = torch.ones(1, 1, 1)
        margin = _core_margin(one, *(x.unsqueeze(1) for x in args), core["s0"],
                              core["T"], core["b"], core["tau_a"], None, None)
        return margin.squeeze(1) + self.context_offset(batch)

    def hazard_alpha1(self, batch: dict) -> Tensor:
        margin = self.deterministic_margin(batch)
        hazard = torch.sigmoid(margin / self.temperature)
        if self.kind == "M3":
            hazard = hazard * self.gate_prob(batch)
        return hazard

    def event_distribution(self, batch: dict, n_quad: int = 8) -> dict:
        """Per-car marginal event-time distribution and its median step."""
        log_h, log1m, _, weights = self.hazard_terms(batch, n_quad)
        support = batch["support"].unsqueeze(1)
        log1m = torch.where(support, log1m, torch.zeros_like(log1m))
        log_h = torch.where(support, log_h, torch.full_like(log_h, -1.0e30))
        log_survival = torch.cumsum(log1m, dim=-1)
        log_survival_prev = log_survival - log1m
        log_prob = (log_h + log_survival_prev).clamp_min(-1.0e30)
        prob = (log_prob.exp() * weights.view(1, -1, 1)).sum(dim=1)
        prob = torch.where(batch["support"], prob, torch.zeros_like(prob))
        cumulative = torch.cumsum(prob, dim=-1).clamp(0.0, 1.0)
        reached = cumulative >= 0.5
        has_median = reached.any(dim=-1)
        local = reached.float().argmax(dim=-1)
        median_step = torch.where(
            has_median, batch["step"].gather(1, local.unsqueeze(1)).squeeze(1),
            torch.full_like(has_median, -1, dtype=torch.long))
        return {"prob": prob, "cumulative": cumulative,
                "final_cumulative": cumulative[:, -1],
                "median_step": median_step, "has_median": has_median}

    def core_parameters(self):
        params = [self.raw_s0, self.raw_T, self.raw_b, self.raw_tau, self.raw_sigma]
        if self.fit_temperature:
            params.append(self.raw_temperature)
        return params

    def learned_parameters(self):
        modules = [m for m in (getattr(self, "context_linear", None),
                               getattr(self, "context_mlp", None),
                               getattr(self, "gate", None)) if m is not None]
        return [p for module in modules for p in module.parameters()]


def _param_dict(model: MergeModel) -> dict:
    core = {name: float(value.detach()) for name, value in model.core().items()}
    out = {"s0_m": core["s0"], "T_s": core["T"], "b_mps2": core["b"],
           "tau_a_s": core["tau_a"], "temperature_m": float(model.temperature.detach()),
           "fit_temperature": bool(model.fit_temperature),
           "alpha_log_sigma": core["sigma"]}
    if model.kind == "M1":
        out["context_linear"] = {
            name: float(value)
            for name, value in zip(model.context_features,
                                   model.context_linear.weight.detach().flatten())}
        out["context_linear_bias"] = float(model.context_linear.bias.detach())
    elif model.kind == "M2":
        layers = [p.detach().flatten().tolist() for p in model.context_mlp.parameters()]
        out["context_mlp_parameters"] = layers
    elif model.kind == "M3":
        out["gate"] = {
            name: float(value)
            for name, value in zip(model.context_features,
                                   model.gate.weight.detach().flatten())}
        out["gate_bias"] = float(model.gate.bias.detach())
    return out


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def _quantile(values: list[float], q: float):
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), q))


def evaluate(batch: dict, model: MergeModel, mask: Tensor, n_quad: int = 8) -> dict:
    """Held-out (or fit) per-car NLL and merge-timing diagnostics."""
    dt = float(batch["dt_s"])
    support = batch["support"]
    exposed = support.any(dim=-1)
    selected = mask.bool() & exposed
    if not bool(selected.any()):
        return {"n_cars": 0, "n_events": 0, "n_censored": 0,
                "nll_mean": None, "nll_total": None}
    with torch.no_grad():
        nll = model.batch_nll(batch, n_quad)[selected]
        dist = model.event_distribution(batch, n_quad)
        margin = model.deterministic_margin(batch)
    events = selected & batch["is_merged"]
    censored = selected & batch["is_censored"]
    event_rows = batch["event"] & events.unsqueeze(1)
    event_margin = (margin * event_rows).sum(dim=1)
    positive_margin = event_margin[events] > 0.0
    has_median = dist["has_median"] & events
    recorded = batch["recorded_event_step"]
    entry = batch["zone_entry_step"]
    median = dist["median_step"]
    valid_error = has_median & (median >= 0)
    recorded_wait = (recorded - entry).float() * dt
    predicted_wait = (median - entry).float() * dt
    error = (predicted_wait - recorded_wait)[valid_error]
    errors = error.tolist()
    recorded_waits = recorded_wait[events].tolist()
    predicted_waits = predicted_wait[valid_error].tolist()
    n_events = int(events.sum())
    timing = {
        "signed_median_s": float(np.median(errors)) if errors else None,
        "abs_median_s": float(np.median(np.abs(errors))) if errors else None,
        "n_scored": len(errors),
        "recorded_wait_p25_s": _quantile(recorded_waits, 0.25),
        "recorded_wait_p50_s": _quantile(recorded_waits, 0.50),
        "recorded_wait_p75_s": _quantile(recorded_waits, 0.75),
        "predicted_wait_p25_s": _quantile(predicted_waits, 0.25),
        "predicted_wait_p50_s": _quantile(predicted_waits, 0.50),
        "predicted_wait_p75_s": _quantile(predicted_waits, 0.75),
    }
    if recorded_waits or predicted_waits:
        top = max(recorded_waits + predicted_waits + [1.0])
        grid = np.linspace(0.0, top, num=min(41, max(6, int(math.ceil(top)) + 1)))
        rec = np.asarray(recorded_waits, dtype=float)
        pred = np.asarray(predicted_waits, dtype=float)
        timing["cumulative_curve"] = [
            {"wait_s": float(t),
             "recorded": float((rec <= t).mean()) if rec.size else None,
             "predicted": float((pred <= t).mean()) if pred.size else None}
            for t in grid]
    per_world = {}
    with torch.no_grad():
        event_indices = torch.nonzero(events, as_tuple=False).flatten().tolist()
        for world in sorted({batch["world"][i] for i in event_indices}):
            idx_t = torch.tensor([i for i, name in enumerate(batch["world"])
                                  if name == world], dtype=torch.long)
            sel = events[idx_t]
            loc = sel & valid_error[idx_t]
            if not bool(loc.any()):
                continue
            world_error = (predicted_wait[idx_t] - recorded_wait[idx_t])[loc]
            per_world[world] = {
                "n_events": int(sel.sum()), "n_scored": int(loc.sum()),
                "signed_median_s": float(np.median(world_error.tolist())),
                "abs_median_s": float(np.median(np.abs(world_error.tolist()))),
            }
    return {
        "n_cars": int(selected.sum()),
        "n_events": n_events,
        "n_censored": int(censored.sum()),
        "n_unsupported_events": int((selected & batch["unsupported_event"]).sum()),
        "nll_mean": float(nll.mean()),
        "nll_total": float(nll.sum()),
        "event_recall": float(has_median.sum() / max(int(events.sum()), 1)),
        "positive_margin_share": (float(positive_margin.float().mean())
                                  if int(events.sum()) else None),
        "timing": timing,
        "per_world": per_world,
    }


def _fit_mask(worlds: list[str], held: set[str]) -> Tensor:
    return torch.tensor([world not in held for world in worlds], dtype=torch.bool)


def fit_model(kind: str, batch: dict, fit_mask: Tensor, *, context_features=(),
              fit_temperature: bool = True, temperature: float | None = None,
              seed: int = 0, iters: int = 500, n_quad: int = 8,
              weight_decay: float = 1.0e-3) -> MergeModel:
    """Deterministic Adam fit of one candidate by per-car whole-wait NLL."""
    model = MergeModel(kind, context_features, fit_temperature=fit_temperature,
                       temperature=temperature, seed=seed)
    core = model.core_parameters()
    learned = model.learned_parameters()
    lr = {"M0": 0.10, "M1": 0.05, "M2": 0.03, "M3": 0.05}[kind]
    groups = [{"params": core, "lr": lr}]
    if learned:
        groups.append({"params": learned, "lr": lr})
    optimizer = torch.optim.Adam(groups)
    exposed = batch["support"].any(dim=-1)
    use = fit_mask.bool() & exposed
    if not bool(use.any()):
        raise SystemExit("no exposed cars in the fit split")
    best = None
    best_nll = float("inf")
    for step in range(int(iters)):
        model.train()
        optimizer.zero_grad()
        nll = model.batch_nll(batch, n_quad)
        loss = nll[use].mean()
        learned_params = model.learned_parameters()
        if learned_params and weight_decay:
            loss = loss + weight_decay * sum(p.square().sum() for p in learned_params)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters()
                                        if p.requires_grad], 5.0)
        optimizer.step()
        if step % 25 == 0 or step == int(iters) - 1:
            with torch.no_grad():
                value = float(model.batch_nll(batch, n_quad)[use].mean())
            if math.isfinite(value) and value < best_nll:
                best_nll = value
                best = copy.deepcopy(model.state_dict())
    if best is not None:
        model.load_state_dict(best)
    model.eval()
    return model


# --------------------------------------------------------------------------------------
# Optional LC2013 state-log agreement (population level)
# --------------------------------------------------------------------------------------
def _zone_flags(row: dict, bits: dict | None) -> dict | None:
    def flag(name: str):
        value = row.get(name)
        return None if value is None else bool(int(value))

    leader = flag("f_blocked_left_leader")
    follower = flag("f_blocked_left_follower")
    urgent = flag("f_urgent")
    cooperative = flag("f_cooperative")
    if leader is None and follower is None and bits:
        state = row.get("stw", row.get("st"))
        if state is not None:
            state = int(state)
            leader = bool(state & int(bits.get("LCA_BLOCKED_BY_LEFT_LEADER", 512)))
            follower = bool(state & int(bits.get("LCA_BLOCKED_BY_LEFT_FOLLOWER", 1024)))
    if leader is None and follower is None:
        blocked = flag("f_blocked")
        if blocked is None:
            return None
        leader = follower = blocked
    return {"blocked_left_leader": bool(leader or False),
            "blocked_left_follower": bool(follower or False),
            "urgent": bool(urgent or False),
            "cooperative": bool(cooperative or False)}


def _lcstate_in_zone(row: dict) -> bool:
    for key in ("d_end", "distance_to_end_m", "dist_to_lane_end_m"):
        value = row.get(key)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return 0.0 <= value <= ZONE_S1_M
    for key in ("s_wm", "s_in_zone_m"):
        value = row.get(key)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return -ZONE_S1_M <= value <= 0.0
    return False


def _iter_lcstate(path: Path, per_file):
    name = path.name.lower()
    if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    per_file(obj)
    elif name.endswith(".json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        rows = data if isinstance(data, list) else data.get("rows", data.get("states", []))
        for obj in rows if isinstance(rows, list) else []:
            if isinstance(obj, dict):
                per_file(obj)
    elif name.endswith(".csv"):
        with path.open("r", encoding="utf-8", newline="") as fh:
            for obj in csv.DictReader(fh):
                per_file(dict(obj))


def load_lcstate(path, *, zone_s1_m: float = ZONE_S1_M) -> dict:
    """Load SUMO LC2013 state logs into per-world zone-state counts.

    Expected fields (accepted aliases in brackets): ``world`` (or filename prefix before
    ``__``), ``vid`` [vehicle_id, id], ``t`` [time_s, time] and/or ``step``, ``d_end``
    [distance_to_end_m, dist_to_lane_end_m] or ``s_wm`` [s_in_zone_m] for in-zone
    membership, and the flags ``f_blocked_left_leader``, ``f_blocked_left_follower``,
    ``f_urgent``, ``f_cooperative`` (fallback: ``f_blocked`` or SUMO bit states ``st`` /
    ``stw`` decoded with the ``lca_bits`` of the first metadata line).  ``.jsonl``,
    ``.jsonl.gz``, ``.json`` (list or ``rows``/``states``) and ``.csv`` are accepted;
    files with ``incomplete`` in the name are skipped.
    """
    root = guard(Path(path))
    if not root.exists():
        return {"status": "unavailable", "path": str(root), "reason": "directory missing",
                "worlds": {}}
    files = []
    for pattern in ("**/*.jsonl", "**/*.jsonl.gz", "**/*.json", "**/*.csv"):
        files.extend(root.glob(pattern))
    files = sorted(p for p in set(files) if "incomplete" not in p.name.lower())
    if not files:
        return {"status": "unavailable", "path": str(root), "reason": "no log files",
                "worlds": {}}
    worlds: dict[str, dict] = defaultdict(
        lambda: {"states": 0, "in_zone": 0, "unblocked": 0, "urgent": 0,
                 "cooperative": 0, "vids": set()})
    errors: list[str] = []
    total = 0
    for path in files:
        world_name = path.name.split("__")[0]
        bits: dict | None = None

        def per_file(obj: dict, world_name=world_name):
            nonlocal total, bits
            if "_meta" in obj:
                if isinstance(obj.get("lca_bits"), dict):
                    bits = obj["lca_bits"]
                return
            row_bits = obj.get("lca_bits")
            use_bits = row_bits if isinstance(row_bits, dict) else bits
            if not _lcstate_in_zone(obj):
                return
            flags = _zone_flags(obj, use_bits)
            if flags is None:
                return
            world = str(obj.get("world") or obj.get("world_id") or world_name)
            vid = str(obj.get("vid", obj.get("vehicle_id", obj.get("id", ""))))
            entry = worlds[world]
            entry["states"] += 1
            total += 1
            entry["in_zone"] += 1
            if vid:
                entry["vids"].add(vid)
            if not (flags["blocked_left_leader"] or flags["blocked_left_follower"]):
                entry["unblocked"] += 1
            entry["urgent"] += int(flags["urgent"])
            entry["cooperative"] += int(flags["cooperative"])

        try:
            _iter_lcstate(path, per_file)
        except (OSError, EOFError, ValueError) as exc:
            errors.append(f"{path.name}: {exc}")
    out_worlds = {}
    for world, entry in worlds.items():
        count = max(int(entry["in_zone"]), 1)
        out_worlds[world] = {
            "in_zone_states": int(entry["in_zone"]),
            "unblocked_share": float(entry["unblocked"] / count),
            "urgent_share": float(entry["urgent"] / count),
            "cooperative_share": float(entry["cooperative"] / count),
            "vids": sorted(entry["vids"]),
        }
    return {"status": "ok" if total else "empty", "path": str(root),
            "n_files": len(files), "n_zone_states": int(total),
            "worlds": out_worlds, "errors": errors}


def _pearson(x, y):
    if len(x) < 2:
        return None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.std() == 0.0 or y.std() == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def lcstate_agreement(batch: dict, model: MergeModel, lcstate: dict, *, n_quad: int = 8,
                      held_worlds: set[str] | None = None) -> dict:
    """Population-level agreement of model merge readiness with SUMO's LC2013 flags.

    The current LCSTATE logs are a separate SUMO run whose vehicle ids (``ramp.N``) do not
    coincide with the recorded ids (``v###``), so agreement is aggregated per world over
    in-zone states: model share of ``margin > 0`` (and mean merge hazard) against SUMO's
    blocked-free, urgent and cooperative shares.  Exact id/time matching is reported as
    unavailable rather than fabricated.
    """
    if lcstate.get("status") != "ok" or not lcstate.get("worlds"):
        return {"status": lcstate.get("status", "unavailable"),
                "reason": lcstate.get("reason"), "table": []}
    available = set(lcstate["worlds"])
    worlds = [w for w in dict.fromkeys(batch["world"]) if w in available]
    if held_worlds:
        preferred = [w for w in worlds if w in held_worlds]
        if preferred:
            worlds = preferred
    if not worlds:
        return {"status": "unmatched", "reason": "no recorded world in LCSTATE logs",
                "table": []}
    with torch.no_grad():
        margin = model.deterministic_margin(batch)
        hazard = model.hazard_alpha1(batch)
    support = batch["support"]
    table = []
    recorded_ids = set(zip(batch["world"], batch["vehicle_id"]))
    id_overlap = 0
    for world in worlds:
        idx = [i for i, name in enumerate(batch["world"]) if name == world]
        idx_t = torch.tensor(idx, dtype=torch.long)
        sup = support[idx_t]
        if not bool(sup.any()):
            continue
        count = int(sup.sum())
        pos = float((((margin[idx_t] > 0.0) & sup).sum()) / count)
        mean_hazard = float(((hazard[idx_t] * sup).sum()) / count)
        sumo = lcstate["worlds"][world]
        sumo_vids = set(sumo["vids"])
        id_overlap += sum(1 for name, vid in recorded_ids
                          if name == world and vid in sumo_vids)
        table.append({
            "world": world, "n_states": count,
            "model_positive_margin_share": pos,
            "model_mean_hazard": mean_hazard,
            "sumo_unblocked_share": sumo["unblocked_share"],
            "sumo_urgent_share": sumo["urgent_share"],
            "sumo_cooperative_share": sumo["cooperative_share"],
            "abs_diff": abs(pos - sumo["unblocked_share"]),
        })
    if not table:
        return {"status": "unmatched", "reason": "no in-zone states", "table": []}
    return {
        "status": "ok",
        "match": "vehicle_time" if id_overlap else "world_aggregate",
        "n_worlds": len(table),
        "n_states": int(sum(row["n_states"] for row in table)),
        "vehicle_id_overlap": int(id_overlap),
        "mean_abs_unblocked_diff": float(np.mean([r["abs_diff"] for r in table])),
        "pearson_positive_margin_vs_unblocked": _pearson(
            [r["model_positive_margin_share"] for r in table],
            [r["sumo_unblocked_share"] for r in table]),
        "pearson_mean_hazard_vs_unblocked": _pearson(
            [r["model_mean_hazard"] for r in table],
            [r["sumo_unblocked_share"] for r in table]),
        "pearson_positive_margin_vs_urgent": _pearson(
            [r["model_positive_margin_share"] for r in table],
            [r["sumo_urgent_share"] for r in table]),
        "pearson_positive_margin_vs_cooperative": _pearson(
            [r["model_positive_margin_share"] for r in table],
            [r["sumo_cooperative_share"] for r in table]),
        "table": table,
        "notes": ["world-level comparison; current LCSTATE vehicle ids differ from the "
                  "recorded ids, so per-vehicle agreement is not claimed"],
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def _fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.{digits}f}"
    return str(value)


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# Merge fit v2", ""]
    data = report["data"]
    split = report["split"]
    lines += [f"- sources: {', '.join(report['sources'])}",
              f"- records: {data['records']}, worlds: {data['worlds']}, "
              f"cars: {data['cars']} (zero-exposure {data['zero_exposure']})",
              f"- held-out worlds: {len(split['held_worlds'])} "
              f"(every fifth of {split['n_worlds']} sorted worlds)",
              f"- seed: {report['seed']}, context: {data['context_description']}",
              f"- temperature: {'fitted' if report['config']['fit_temperature'] else 'fixed'}",
              ""]
    lines += ["## Held-out summary", "",
              "| model | NLL/car | signed median s | abs median s | recall | "
              "positive margin | p25/p50/p75 recorded | p25/p50/p75 predicted |",
              "|---|---|---|---|---|---|---|---|"]
    for name, result in report["models"].items():
        held = result["held_out"]
        timing = held.get("timing", {})
        rec = "/".join(_fmt(timing.get(f"recorded_wait_p{q}_s"), 1) for q in (25, 50, 75))
        pred = "/".join(_fmt(timing.get(f"predicted_wait_p{q}_s"), 1) for q in (25, 50, 75))
        lines.append(
            f"| {name} | {_fmt(held.get('nll_mean'))} | "
            f"{_fmt(timing.get('signed_median_s'))} | {_fmt(timing.get('abs_median_s'))} | "
            f"{_fmt(held.get('event_recall'))} | "
            f"{_fmt(held.get('positive_margin_share'))} | {rec} | {pred} |")
    lines.append("")
    for name, result in report["models"].items():
        held = result["held_out"]
        params = result["params"]
        lines += [f"## {name}", "", f"- cars: {held.get('n_cars')}, "
                  f"events: {held.get('n_events')}, censored: {held.get('n_censored')}, "
                  f"unsupported events: {held.get('n_unsupported_events')}",
                  f"- NLL/car: {_fmt(held.get('nll_mean'))} "
                  f"(total {_fmt(held.get('nll_total'), 2)})",
                  f"- event recall: {_fmt(held.get('event_recall'))}; "
                  f"positive margin at event: {_fmt(held.get('positive_margin_share'))}",
                  f"- core: s0={_fmt(params['s0_m'])}, T={_fmt(params['T_s'])}, "
                  f"b={_fmt(params['b_mps2'])}, tau_a={_fmt(params['tau_a_s'])}, "
                  f"temperature={_fmt(params['temperature_m'])}, "
                  f"sigma={_fmt(params['alpha_log_sigma'])}"]
        if "context_linear" in params:
            terms = ", ".join(f"{key}={_fmt(value)}"
                              for key, value in params["context_linear"].items())
            lines.append(f"- context linear (margin m): {terms}; "
                         f"bias={_fmt(params['context_linear_bias'])}")
        if "gate" in params:
            terms = ", ".join(f"{key}={_fmt(value)}"
                              for key, value in params["gate"].items())
            lines.append(f"- unblocked gate logit: {terms}; "
                         f"bias={_fmt(params['gate_bias'])}")
        curve = timing.get("cumulative_curve") or []
        if curve:
            points = [curve[i] for i in np.linspace(0, len(curve) - 1, min(8, len(curve)))
                      .astype(int)]
            lines.append("- cumulative merge curve (wait s: recorded/predicted): " + "; ".join(
                f"{_fmt(row['wait_s'], 1)}: {_fmt(row['recorded'], 2)}/"
                f"{_fmt(row['predicted'], 2)}" for row in points))
        lines.append("")
    lcstate = report.get("lcstate") or {}
    lines += ["## LC2013 state-log agreement", ""]
    if lcstate.get("status") != "ok":
        lines.append(f"- {lcstate.get('status', 'not requested')}: "
                     f"{lcstate.get('reason', '')}")
    else:
        lines.append(f"- {lcstate['match']}, {lcstate['n_worlds']} worlds, "
                     f"{lcstate['n_states']} states, vehicle-id overlap "
                     f"{lcstate['vehicle_id_overlap']}")
        lines += ["", "| world | model margin>0 | SUMO unblocked | urgent | cooperative |",
                  "|---|---|---|---|---|"]
        for row in lcstate["table"][:40]:
            lines.append(f"| {row['world']} | {_fmt(row['model_positive_margin_share'])} | "
                         f"{_fmt(row['sumo_unblocked_share'])} | "
                         f"{_fmt(row['sumo_urgent_share'])} | "
                         f"{_fmt(row['sumo_cooperative_share'])} |")
    lines += ["", f"Fitted with seed {report['seed']}; see the JSON for full parameters.",
              ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    guard(path).write_text("\n".join(lines), encoding="utf-8")


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy().tolist()
    return value


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_models(text: str) -> list[str]:
    names = [part.strip().upper() for part in str(text).split(",") if part.strip()]
    if not names:
        raise SystemExit("--models must name at least one of M0,M1,M2,M3")
    for name in names:
        if name not in MODELS:
            raise SystemExit(f"unknown model {name}; choose from {', '.join(MODELS)}")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit merge-decision models M0-M3 on recorded TRAIN episodes (CPU).")
    parser.add_argument("--source", action="append", default=None,
                        help="NAME=DIR with recs_native.pkl / recs_genrand.pkl (repeatable)")
    parser.add_argument("--models", default="M0,M1,M2,M3",
                        help="comma-separated subset of M0,M1,M2,M3")
    temp = parser.add_mutually_exclusive_group()
    temp.add_argument("--fit-temperature", dest="fit_temperature", action="store_true",
                      help="fit the smooth-decision temperature (default)")
    temp.add_argument("--fix-temperature", dest="fit_temperature", action="store_false",
                      help="freeze the temperature at --temperature / the deployed init")
    parser.set_defaults(fit_temperature=True)
    parser.add_argument("--temperature", type=float, default=None,
                        help="temperature used with --fix-temperature (m)")
    parser.add_argument("--lcstate", default=None,
                        help="optional SUMO LC2013 state-log directory")
    parser.add_argument("--out", default=DEFAULT_OUT, help="JSON report path")
    parser.add_argument("--md", default=DEFAULT_MD, help="markdown report path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-worlds", type=int, default=None,
                        help="use only the first N sorted worlds (smoke runs)")
    parser.add_argument("--n-quad", type=int, default=8)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--context-module", default=DEFAULT_CONTEXT_MODULE)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    torch.set_num_threads(1)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed) % (2 ** 32))
    models = parse_models(args.models)
    source_items = args.source or list(DEFAULT_SOURCES)
    sources = _parse_sources(source_items)
    out_path = guard(Path(args.out))
    md_path = guard(Path(args.md))
    if args.lcstate:
        guard(Path(args.lcstate))
    needs_context = any(name != "M0" for name in models)
    context_fn, context_names, context_module = (None, (), None)
    if needs_context:
        context_fn, context_names, context_module = load_context(args.context_module)
        if context_fn is None:
            raise SystemExit(
                f"{', '.join(n for n in models if n != 'M0')} need "
                f"'{args.context_module}' (agent A's recorded_context/CONTEXT_FEATURES); "
                "it is not importable in this worktree. Fit M0 only until the coordinator "
                "integrates merge_context.py, or pass --context-module.")
    records, source_notes = load_records(sources)
    for record in records:
        check_seed(record.get("seed", -1))
    records = sorted(records, key=lambda r: (str(r.get("world", "")), str(r.get("key", ""))))
    all_worlds = sorted({str(r.get("world", "?")) for r in records})
    if args.max_worlds is not None:
        if int(args.max_worlds) <= 0:
            raise SystemExit("--max-worlds must be positive")
        keep = set(all_worlds[:int(args.max_worlds)])
        records = [r for r in records if str(r.get("world", "?")) in keep]
        all_worlds = sorted(keep)
    print(f"loading {len(records)} records from {len(all_worlds)} worlds", flush=True)
    sequences, build_stats = build_dataset(
        records, context_fn=context_fn, context_names=context_names)
    if not sequences:
        raise SystemExit("no ramp-car sequences found")
    batch = pad_sequences(sequences, len(context_names))
    worlds = list(dict.fromkeys(batch["world"]))
    if args.max_worlds is not None:
        worlds = [w for w in all_worlds if w in set(worlds)]
    held_worlds = {world for i, world in enumerate(worlds) if i % 5 == 4}
    if not held_worlds and len(worlds) > 1:
        held_worlds = {worlds[-1]}
    fit_mask = _fit_mask(batch["world"], held_worlds)
    held_mask = ~fit_mask
    scaler = standardize_context(batch, fit_mask)
    data = {
        "records": len(records), "worlds": len(worlds), "cars": len(sequences),
        "zero_exposure": int(build_stats["zero_exposure"]),
        "unsupported_events": int(build_stats["unsupported_events"]),
        "events": int(sum(seq["is_merged"] for seq in sequences)),
        "censored": int(sum(seq["is_censored"] for seq in sequences)),
        "context_features": list(context_names),
        "context_description": (f"{context_module} ({len(context_names)} features)"
                                if context_names else "none (M0)"),
        "context_scaler": scaler,
    }
    split = {"n_worlds": len(worlds), "held_worlds": sorted(held_worlds),
             "fit_worlds": sorted(set(worlds) - held_worlds),
             "fit_cars": int(fit_mask.sum()), "held_cars": int(held_mask.sum())}
    print(f"worlds {len(worlds)}: fit {len(split['fit_worlds'])}, "
          f"held {len(held_worlds)}; cars {len(sequences)}", flush=True)
    config = {"seed": int(args.seed), "fit_temperature": bool(args.fit_temperature),
              "temperature": args.temperature, "n_quad": int(args.n_quad),
              "iters": int(args.iters), "models": models,
              "source_spec": [f"{name}={path}" for name, path in sources],
              "max_worlds": args.max_worlds}
    results = {}
    for name in models:
        print(f"fitting {name}", flush=True)
        model = fit_model(name, batch, fit_mask, context_features=context_names,
                          fit_temperature=bool(args.fit_temperature),
                          temperature=args.temperature, seed=int(args.seed),
                          iters=int(args.iters), n_quad=int(args.n_quad))
        held = evaluate(batch, model, held_mask, int(args.n_quad))
        fit = evaluate(batch, model, fit_mask, int(args.n_quad))
        results[name] = {"params": _param_dict(model), "held_out": held, "fit": fit}
        print(f"  {name}: held NLL/car {_fmt(held.get('nll_mean'))}, "
              f"abs median {_fmt(held.get('timing', {}).get('abs_median_s'))} s, "
              f"recall {_fmt(held.get('event_recall'))}", flush=True)
    lcstate = {"status": "not_requested"}
    if args.lcstate:
        lcstate = load_lcstate(args.lcstate)
        if lcstate.get("status") == "ok":
            best_name = min(models, key=lambda n: (
                results[n]["held_out"].get("nll_mean") if
                results[n]["held_out"].get("nll_mean") is not None else float("inf")))
            # Recreate the selected model state for the agreement table.
            best_model = fit_model(
                best_name, batch, fit_mask, context_features=context_names,
                fit_temperature=bool(args.fit_temperature), temperature=args.temperature,
                seed=int(args.seed), iters=int(args.iters), n_quad=int(args.n_quad))
            lcstate = lcstate_agreement(batch, best_model, lcstate,
                                        n_quad=int(args.n_quad), held_worlds=held_worlds)
            lcstate["model"] = best_name
        print(f"lcstate: {lcstate.get('status')}", flush=True)
    report = {
        "tool": "merge_fit_v2", "version": 1, "seed": int(args.seed),
        "sources": [f"{name}={path}" for name, path in sources],
        "source_notes": source_notes, "config": config, "data": data, "split": split,
        "models": results, "lcstate": lcstate,
    }
    report = _jsonable(report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_markdown(report, md_path)
    print(f"wrote {out_path} and {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
