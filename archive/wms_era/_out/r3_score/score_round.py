"""Round-3 scoring harness: one candidate WM-S run vs the current best (old C1).

CPU only; no training, no SUMO.  The scoring core is adapted verbatim in mode
from ``E:/rd-wt/ws-main2score/_out/main2_score/main2_score.py`` (one model x one
state at a time, <=4 process workers, CAVs teacher-forced from the recorded arm,
entries per flag, default OFF); the quick-subset rule and the per-stage 60 s
table come from ``E:/rd-wt/ws-bisect-stages/_out/bisect_stages_score.py``; the
instrumented fingerprints (gated residual g*delta_a, lane-change rate,
wrong-lane share, signed speed bias) come from
``E:/rd-wt/ws-bisect-residual/_out/bisect_residual/run_bisect.py``.

Inputs: ``--candidate`` run dir (``wm_s.pt``, optional ``wm_s_best.pt`` /
``wm_s_stage*.pt``, ``TRAIN_RECEIPT.json``) and ``--reference`` (run dir or
``wm_s.pt`` path; default: the old C1 struct).  Checkpoints are read-only
staged copies inside this workspace (the loader requires that).

Sets: ``wall_same`` (primary), ``wall_pilot`` (secondary), optional
``seen_controls``.  ``--quick`` takes, per set, the fixed 20-state subset
(sort by (world, t0_s, key); first state per world; first 20 worlds) --
for ``wall_same`` this is exactly the registered bisect-settings subset.

Outputs (one markdown + one json plus per-model state files): HDV
median/IQR/p90 |ds| and |dv| at 10/30/60/120 s, blow-up rate, paired world
bootstrap candidate-minus-reference (negative favours the candidate), and the
fingerprints: mean/p95 |g*delta_a|, predicted vs recorded lane-change rate,
wrong-lane share, best-vs-last checkpoint, per-stage checkpoints at 60 s (quick
subset), jam vs free-flow slices (a state is 'jam' when any recorded HDV is
< 5 m/s within the horizon; plus |ds| on recorded-slow HDVs), signed speed
bias, optional inference toggles (projection, merge_cal) as extra rows, and a
``--history-states DIR`` hook (long-history eval states) with
``--history-cut-s`` (history fingerprint; a no-op for checkpoints without a
driver-encoder history path).

Each model is scored with its own ``TRAIN_RECEIPT`` physics flags.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

WORKSPACE = Path("E:/rd-wt/ws-r3-score2")
WM_N1 = WORKSPACE / "output" / "wd01" / "wm_n1"
if str(WM_N1) not in sys.path:
    sys.path.insert(0, str(WM_N1))

from wms_eval.c1v2 import _car_compact, iter_windows  # noqa: E402
from wms_eval.data import DataError, recorded_for  # noqa: E402
from wms_eval.metrics import car_metrics  # noqa: E402
from wms_eval.models import flags_for  # noqa: E402
from wms_eval.horizon_report import blowup_rate, paired_bootstrap  # noqa: E402
from wms_eval.wms_plugin import WmsCheckpointModel  # noqa: E402

STATES_DIR = Path("E:/rd-wt/ws-wallc1/_out/wallc1_states_v2")
SETS = {
    "wall_same": STATES_DIR / "wall_same.json",
    "wall_pilot": STATES_DIR / "wall_pilot.json",
    "seen_controls": STATES_DIR / "seen_controls.json",
}
DEFAULT_REFERENCE = Path(
    "E:/research-dense/runs/active/WM-S-C1FAST-20261006/box/full_struct")

HORIZONS = (10.0, 30.0, 60.0, 120.0)
FP_WINDOWS = (60.0, 120.0)
PREDICT_HORIZON = 120.0
STAGE_HORIZON = 60.0
QUICK_WORLDS = 20
DT_S = 0.5
JAM_SPEED_MPS = 5.0
SLOW_SPEED_MPS = 5.0
N_BOOT = 1000
MAX_WORKERS = 2
SEED = 0
OUT_ROOT = WORKSPACE / "_out" / "r3_score"
CKPT_ROOT = OUT_ROOT / "_ckpt"


CACHE_ROOT = OUT_ROOT / "_cache"


def cache_key(spec: dict, kind: str, recs: list[dict], extra="") -> str:
    """Identity of a deterministic (checkpoint, physics, states) computation."""
    def hsig(r):
        h = r.get("history")
        if not isinstance(h, dict):
            return None
        try:
            return [h.get("history_s"), h.get("run_start_limited"),
                    len(h.get("kin") or []), int(sum(
                        sum(1 for v in row if v) for row in h.get("mask") or []))]
        except Exception:
            return "?"

    ids = json.dumps([[str(r.get("world")), float(r.get("t0_s", 0.0)),
                       str(r.get("key")), hsig(r)] for r in recs])
    blob = json.dumps([kind, spec["sha256"], spec["physics"],
                       bool(spec.get("entries")), bool(spec.get("entries_v2")),
                       bool(spec.get("merge_cal_on")), ids, extra],
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:24]


def cached(key: str, compute):
    """Reuse an identical earlier result (reference model, repeated tags)."""
    path = CACHE_ROOT / f"{key}.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")), True
    value = compute()
    text = json.dumps(value)
    if '"error": "' not in text:  # never cache failed arms
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return value, False


def synthetic_history(record: dict, history_s: float, stride_s: float) -> dict:
    """SMOKE ONLY: declare the legacy 10-tick history as a start-limited long one."""
    out = dict(record)
    h = dict(record["history"])
    t = len(h["kin"])
    h.update({"history_s": float(history_s), "history_stride_s": float(stride_s),
              "dt_s": DT_S, "available_span_s": (t - 1) * DT_S,
              "run_start_limited": True})
    out["history"] = h
    return out


def cut_history(record: dict, cut_s: float) -> dict:
    """Mask pre-t0 history older than ``cut_s`` seconds (causal-history probe)."""
    out = dict(record)
    h = dict(record["history"])
    mask = [list(row) for row in h["mask"]]
    target = [list(row) for row in h.get("target_mask", [])]
    t = len(mask)
    keep = int(round(float(cut_s) / DT_S))  # ticks before t0 that stay visible
    for i in range(max(0, t - 1 - keep)):
        mask[i] = [False] * len(mask[i])
    for i in range(min(len(target), max(0, t - 1 - keep))):
        target[i] = [False] * len(target[i])
    h["mask"], h["target_mask"] = mask, target
    out["history"] = h
    return out


def scored_arm(set_name: str) -> str:
    return "native" if set_name == "seen_controls" else "wall"


# --------------------------------------------------------------------------- #
# checkpoints: discovery, staging, variants
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def receipt_physics(run: Path) -> dict:
    return json.loads((run / "TRAIN_RECEIPT.json").read_text(encoding="utf-8"))["physics"]


def run_of(value: Path) -> Path:
    """Accept a run dir or a direct ``wm_s.pt`` path; return the run dir."""
    value = Path(value)
    if value.is_file() and value.name.endswith(".pt"):
        return value.parent
    return value


def stage_copy(src: Path, tag: str) -> Path:
    dest = CKPT_ROOT / tag / "wm_s.pt"
    if not dest.is_file() or dest.stat().st_size != src.stat().st_size:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return dest


def rewrite_config(staged: Path, settings: dict) -> Path:
    """Rewrite the STAGED copy's embedded inference config (never the source)."""
    import torch

    payload = torch.load(staged, map_location="cpu", weights_only=True)
    cfg = dict(payload.get("config") or {})
    for key, value in settings.items():
        cfg[key] = value
    payload["config"] = cfg
    torch.save(payload, staged)
    return staged


def ckpt_config(staged: Path) -> dict:
    import torch

    payload = torch.load(staged, map_location="cpu", weights_only=True)
    return dict(payload.get("config") or {})


def has_history_encoder(cfg: dict) -> bool:
    """True when the checkpoint actually consumes pre-t0 driver history."""
    if cfg.get("history_s") is not None:
        return True
    if str(cfg.get("rollout_mode", "residual")) != "residual":
        return True
    if str(cfg.get("lane_head", "off")) != "off":
        return True
    return bool(cfg.get("history_conditioned", False))


# --------------------------------------------------------------------------- #
# subset selection (bisect-settings rule)
# --------------------------------------------------------------------------- #
def select_subset(path: Path, max_worlds: int = QUICK_WORLDS) -> list[dict]:
    """Sort by (world, t0_s, key); first state per world; first N worlds."""
    by_world: dict[str, tuple] = {}
    for record in iter_windows(path):
        world = str(record.get("world"))
        ordered = (world, float(record.get("t0_s", 0.0)), str(record.get("key")))
        if world not in by_world or ordered < by_world[world][0]:
            by_world[world] = (ordered, record)
    worlds = sorted(by_world)
    chosen = [by_world[w][1] for w in worlds[:max_worlds]]
    chosen.sort(key=lambda r: (str(r.get("world")), float(r.get("t0_s", 0.0)),
                               str(r.get("key"))))
    return chosen


# --------------------------------------------------------------------------- #
# plain scoring (main2_score contract)
# --------------------------------------------------------------------------- #
def evaluate(record: dict, predictor, horizon=PREDICT_HORIZON,
             horizons=HORIZONS, seed=0) -> dict:
    t0_s = float(record["t0_s"])
    controllers = record.get("cav_controllers", {})
    arms = {}
    for arm in sorted((record.get("recorded") or {}).keys()):
        try:
            predicted = predictor(record, {**controllers, "arm": arm}, horizon, seed)
            recorded = recorded_for(record, arm)
            car = _car_compact(car_metrics(predicted, recorded, t0_s=t0_s,
                                           horizons=tuple(horizons)))
            arms[arm] = {"error": None, "car": car}
        except Exception as exc:  # one arm must not abort the rest
            arms[arm] = {"error": f"{type(exc).__name__}: {exc}"}
    return {
        "world": str(record.get("world")), "t0_s": t0_s, "key": record.get("key"),
        "source_arm": record.get("source_arm"), "own_arm": record.get("arm"),
        "source_set": record.get("source_set"), "arms": arms,
    }


_W: dict = {}
_CUT: list = [None]


def _unused_patch_history_cut(cut_s: float):
    """Cap the pre-t0 driver-history window at ``cut_s`` seconds of ticks."""
    import wms_eval.wms_plugin as plugin

    original = plugin.prior_history_features
    ticks = max(1, int(round(float(cut_s) / DT_S)))

    def wrapped(*args, **kwargs):
        kwargs["history_len"] = min(int(kwargs.get("history_len", 10)), ticks)
        return original(*args, **kwargs)

    plugin.prior_history_features = wrapped


def build_predictor(tag: str, spec: dict):
    staged = Path(spec["staged"])
    physics = dict(spec["physics"])
    if spec.get("merge_cal_on"):
        physics = {**physics, "merge_cal": True}
    return WmsCheckpointModel(
        staged, flags_for("wms", entries=spec.get("entries", False),
                          entries_v2=spec.get("entries_v2", False)),
        physics=physics, model_kind="structured").predict_states


def _init_worker(tag: str, spec: dict, history_cut_s=None):
    os.environ["WMS_EVAL_THREADS"] = "1"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _CUT[0] = history_cut_s
    _W["predictor"] = build_predictor(tag, spec)


def _work(payload):
    index, record = payload
    return index, evaluate(record, _W["predictor"])


def score_model(tag: str, spec: dict, records: list[dict], workers: int,
                inflight: int, history_cut_s=None, horizons=HORIZONS,
                horizon_s=PREDICT_HORIZON) -> list[dict]:
    if workers <= 1:
        _init_worker(tag, spec, history_cut_s)
        return [res for _, res in (_work((i, rec)) for i, rec in enumerate(records))]
    results = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(tag, spec, history_cut_s)) as pool:
        pending = {}
        it = enumerate(records)
        exhausted = False

        def submit():
            nonlocal exhausted
            if exhausted:
                return
            try:
                i, rec = next(it)
            except StopIteration:
                exhausted = True
                return
            pending[pool.submit(_work, (i, rec))] = i

        for _ in range(max(1, inflight)):
            submit()
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for fut in done:
                i, res = fut.result()
                results[i] = res
                del pending[fut]
                submit()
    return [results[i] for i in sorted(results)]


# --------------------------------------------------------------------------- #
# instrumented fingerprints (run_bisect contract)
# --------------------------------------------------------------------------- #
def fingerprint_model(tag: str, spec: dict, records: list[dict], arm: str,
                      history_cut_s=None, predict_horizon=PREDICT_HORIZON) -> dict:
    """Per-step instrumented rollout: g*delta_a, lane rates, wrong-lane, bias."""
    os.environ["WMS_EVAL_THREADS"] = "2"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    import types

    import numpy as np
    import wms.model as WM

    _CUT[0] = history_cut_s
    physics = dict(spec["physics"])
    if spec.get("merge_cal_on"):
        physics = {**physics, "merge_cal": True}
    model = WmsCheckpointModel(Path(spec["staged"]),
                               flags_for("wms", entries=spec.get("entries", False),
                                         entries_v2=spec.get("entries_v2", False)),
                               physics=physics, model_kind="structured")
    model._load()
    cap = model.model.residual_cap_mps2
    soft = getattr(model.model, "residual_soft_bound_mps2", None)

    store: dict = {"diag": [], "ids": []}
    captures: list = []
    orig_bv = WM.bounded_value

    def bv(x, c):
        captures.append(x.detach())
        return orig_bv(x, c)

    WM.bounded_value = bv
    orig_inputs = model._inputs

    def inputs(state, arm_):
        t0, ids, kin, is_cav = orig_inputs(state, arm_)
        store["ids"] = list(ids)
        return t0, ids, kin, is_cav

    model._inputs = inputs
    orig_step = model.model.step

    def step(self, state, **kw):
        captures.clear()
        out = orig_step(state, **kw)
        kin = WM._state_kin(state)
        present = WM._present(state, kin).detach().cpu().numpy()[0]
        is_cav = state.get("is_cav")
        is_cav = (is_cav.detach().cpu().numpy()[0] if is_cav is not None
                  else np.zeros_like(present))
        # residual g*delta_a (pre-bound): model output key (also correct for the
        # soft-bound path); fall back to the first bounded_value capture.
        raw = out.get("raw_residual_mps2")
        if raw is None:
            raw = captures[0] if captures else None
        store["diag"].append({
            "k": int(kw.get("k_steps", 0).item()
                     if hasattr(kw.get("k_steps", 0), "item")
                     else kw.get("k_steps", 0)),
            "raw": (raw.detach().cpu().numpy()[0].astype(float).reshape(-1)
                    if raw is not None else None),
            "gate": out["gate"].detach().cpu().numpy()[0].astype(float).reshape(-1),
            "next_s": out["next_s_m"].detach().cpu().numpy()[0].astype(float),
            "next_v": out["next_v_mps"].detach().cpu().numpy()[0].astype(float),
            "next_lane": out["next_lane"].detach().cpu().numpy()[0].astype(int),
            "next_present": out["next_present"].detach().cpu().numpy()[0].astype(bool),
            "present": present,
            "is_cav": is_cav,
        })
        return out

    model.model.step = types.MethodType(step, model.model)

    def recorded_index(frames, t0):
        by = {}
        for f in frames:
            idx = int(round((float(f["time_s"]) - t0) / DT_S))
            by[idx] = {str(r["id"]): r for r in f.get("vehicles", [])
                       if r.get("present", True)}
        return by

    per_state = []
    for record in records:
        t0 = float(record["t0_s"])
        controllers = record.get("cav_controllers", {})
        store["diag"] = []
        # arm == "first_sorted" reproduces the BISECT_RESIDUAL diagnostic, which
        # used sorted(recorded)[0] (the native arm when a state has one); the
        # harness default is the registered arm (wall for wall_same).
        arm_r = (sorted((record.get("recorded") or {}).keys())[0]
                 if arm == "first_sorted" else arm)
        model.predict_states(record, {**controllers, "arm": arm_r},
                             predict_horizon, 0)
        diag = store["diag"]
        ids = store["ids"]
        try:
            recorded = recorded_for(record, arm_r)
        except DataError:
            recorded = []
        rec_by = recorded_index(recorded, t0)

        # step-level statistics are accumulated for the first 60 s (the window
        # of the BISECT_RESIDUAL diagnostic; primary) and for the full 120 s
        # rollout (later steps are dominated by drifted states).
        wins = {w: {"g_abs": [], "g_signed": [], "cap_bind": 0, "hdv_steps": 0,
                    "lc_pred": 0, "lc_rec": 0, "steps_pred": 0, "steps_rec": 0,
                    "speed_errs": []} for w in FP_WINDOWS}
        prev_pred: dict[str, int] = {}
        prev_rec: dict[str, int] = {}
        for i, d in enumerate(diag):
            active = [wins[w] for w in FP_WINDOWS if i < int(round(w / DT_S))]
            learned = d["present"] & (~d["is_cav"])
            if d["raw"] is not None:
                for j in np.flatnonzero(learned):
                    v = float(d["raw"][j])
                    for a in active:
                        a["g_signed"].append(v)
                        a["g_abs"].append(abs(v))
                        if cap is not None and abs(v) > cap:
                            a["cap_bind"] += 1
                for a in active:
                    a["hdv_steps"] += int(learned.sum())
            for j in np.flatnonzero(learned & d["next_present"]):
                vid = ids[j] if j < len(ids) else f"__entry_{j - len(ids)}"
                if vid in prev_pred:
                    for a in active:
                        a["steps_pred"] += 1
                        if prev_pred[vid] != int(d["next_lane"][j]):
                            a["lc_pred"] += 1
                prev_pred[vid] = int(d["next_lane"][j])
            rows = rec_by.get(i + 1)
            if rows:
                for vid, r in rows.items():
                    if bool(r.get("is_cav")):
                        continue
                    ln = int(r.get("lane", -1))
                    if vid in prev_rec:
                        for a in active:
                            a["steps_rec"] += 1
                            if prev_rec[vid] != ln:
                                a["lc_rec"] += 1
                    prev_rec[vid] = ln
                for j in np.flatnonzero(learned & d["next_present"]):
                    vid = ids[j] if j < len(ids) else None
                    r = rows.get(vid) if vid else None
                    if r is None or bool(r.get("is_cav")):
                        continue
                    for a in active:
                        a["speed_errs"].append(float(d["next_v"][j]) - float(r["v"]))

        # recorded jam classification + slow ids over the scored horizon
        horizon_rows = [f for f in recorded
                        if t0 <= float(f["time_s"]) <= t0 + PREDICT_HORIZON]
        slow_ids: set[str] = set()
        for f in horizon_rows:
            for r in f.get("vehicles", []):
                if not r.get("present", True) or bool(r.get("is_cav")):
                    continue
                if float(r.get("v", 0.0)) < SLOW_SPEED_MPS:
                    slow_ids.add(str(r["id"]))

        horizons_out = {}
        for h in (60.0, 120.0):
            hsteps = int(round(h / DT_S))
            hrows = rec_by.get(hsteps)
            hpred = diag[hsteps - 1] if hsteps - 1 < len(diag) else None
            ds_all: list[float] = []
            ds_lane: list[float] = []
            ds_slow: list[float] = []
            if hrows is not None and hpred is not None:
                for j in np.flatnonzero(hpred["next_present"] & (~hpred["is_cav"])):
                    vid = ids[j] if j < len(ids) else None
                    r = hrows.get(vid) if vid else None
                    if r is None or bool(r.get("is_cav")):
                        continue
                    ds = abs(float(hpred["next_s"][j]) - float(r["s"]))
                    ds_all.append(ds)
                    if int(hpred["next_lane"][j]) != int(r.get("lane", -1)):
                        ds_lane.append(ds)
                    if vid in slow_ids:
                        ds_slow.append(ds)
            horizons_out[f"{h:g}"] = {
                "ds": ds_all, "ds_lane": ds_lane, "ds_slow": ds_slow}

        per_state.append({
            "world": str(record.get("world")), "t0_s": t0, "key": record.get("key"),
            "jam": bool(slow_ids), "n_slow_ids": len(slow_ids),
            "windows": wins, "horizons": horizons_out,
        })

    WM.bounded_value = orig_bv
    agg = aggregate_fingerprints(per_state, cap, FP_WINDOWS[0])
    agg["window_s"] = FP_WINDOWS[0]
    agg["full"] = aggregate_fingerprints(per_state, cap, FP_WINDOWS[-1])
    agg["full"]["window_s"] = FP_WINDOWS[-1]
    agg["residual_soft_bound_mps2"] = soft
    return agg


def aggregate_fingerprints(per_state: list[dict], cap, window: float = 60.0) -> dict:
    # flatten the requested step window back into the per-state keys
    per_state = [{**s, **s["windows"][window]} for s in per_state]
    g_abs = [v for s in per_state for v in s["g_abs"]]
    g_signed = [v for s in per_state for v in s["g_signed"]]
    tot_bind = sum(s["cap_bind"] for s in per_state)
    tot_steps = sum(s["hdv_steps"] for s in per_state)

    def med(vals):
        return float(statistics.median(vals)) if vals else None

    out = {
        "residual_cap_mps2": cap,
        "n_states": len(per_state), "hdv_steps": tot_steps,
        "g_delta_mean_abs": (statistics.fmean(g_abs) if g_abs else None),
        "g_delta_p95_abs": (float(np_p95(g_abs)) if g_abs else None),
        "g_delta_mean": (statistics.fmean(g_signed) if g_signed else None),
        "g_delta_frac_pos": (sum(1 for v in g_signed if v > 0) / len(g_signed)
                             if g_signed else None),
        "cap_bind_rate": (tot_bind / tot_steps if tot_steps else None),
        "lane_rate_pred": None, "lane_rate_rec": None,
        "speed_bias_mean": (statistics.fmean([v for s in per_state
                                              for v in s["speed_errs"]])
                            if any(s["speed_errs"] for s in per_state) else None),
        "speed_bias_median_states": med([statistics.fmean(s["speed_errs"])
                                         for s in per_state if s["speed_errs"]]),
        "jam_free": {}, "slow_hdv_ds": {}, "wrong_lane_share": {},
        "states": [{k: s[k] for k in ("world", "t0_s", "key", "jam",
                                      "n_slow_ids")} for s in per_state],
    }
    lc_p = sum(s["lc_pred"] for s in per_state)
    sp = sum(s["steps_pred"] for s in per_state)
    lc_r = sum(s["lc_rec"] for s in per_state)
    sr = sum(s["steps_rec"] for s in per_state)
    out["lane_changes_pred"] = lc_p
    out["lane_steps_pred"] = sp
    out["lane_changes_rec"] = lc_r
    out["lane_steps_rec"] = sr
    out["lane_rate_pred"] = (lc_p / sp if sp else None)
    out["lane_rate_rec"] = (lc_r / sr if sr else None)
    for h in ("60", "120"):
        all_ds = [v for s in per_state for v in s["horizons"][h]["ds"]]
        lane_ds = [v for s in per_state for v in s["horizons"][h]["ds_lane"]]
        out["wrong_lane_share"][h] = (sum(lane_ds) / sum(all_ds)
                                      if all_ds and sum(all_ds) > 0 else None)
        out["wrong_lane_share"][f"{h}_n_lane"] = len(lane_ds)
        jam_states = [s for s in per_state if s["jam"]]
        free_states = [s for s in per_state if not s["jam"]]
        out["jam_free"] = out.get("jam_free") or {}
        for name, group in (("jam", jam_states), ("free", free_states)):
            block = out["jam_free"].setdefault(name, {"n_states": len(group)})
            vals = [med([v for v in s["horizons"][h]["ds"]]) for s in group]
            vals = [v for v in vals if v is not None]
            block[f"ds_med_{h}"] = med(vals)
        slow_vals = [med(s["horizons"][h]["ds_slow"]) for s in per_state
                     if s["horizons"][h]["ds_slow"]]
        out["slow_hdv_ds"][f"ds_med_{h}"] = med(slow_vals)
        out["slow_hdv_ds"][f"n_states_with_slow_{h}"] = len(slow_vals)
    return out


def np_p95(vals):
    import numpy as np
    return np.percentile(vals, 95)


# --------------------------------------------------------------------------- #
# analysis helpers
# --------------------------------------------------------------------------- #
def normalize_seen(states: list[dict]) -> list[dict]:
    out = []
    for state in states:
        arms = state.get("arms") or {}
        if "native" in arms or len(arms) != 1:
            out.append(state)
            continue
        (own,) = arms.keys()
        new = dict(state)
        new["arms"] = {"native": arms[own]}
        new["source_arm"] = own
        out.append(new)
    return out


def _car(state: dict, arm: str, horizon: float, field="ds_med"):
    block = (state.get("arms") or {}).get(arm)
    if not isinstance(block, dict) or block.get("error"):
        return None
    value = ((block.get("car") or {}).get(str(float(horizon))) or {}) \
        .get("hdv", {}).get(field)
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(values[lo])
    return float(values[lo] + (values[hi] - values[lo]) * (pos - lo))


def block_field(states, arm, horizon, field="ds_med"):
    vals = [v for v in (_car(s, arm, horizon, field) for s in states)
            if v is not None]
    if not vals:
        return {"n": 0, "median": None, "iqr": None, "p90": None}
    q1, q3 = _quantile(vals, 0.25), _quantile(vals, 0.75)
    return {"n": len(vals), "median": float(statistics.median(vals)),
            "iqr": q3 - q1, "p90": _quantile(vals, 0.9)}


def state_value(states, tag, set_name, horizon, field="ds_med"):
    return [v for v in (_car(s, scored_arm(set_name), horizon, field)
                        for s in states[tag][set_name]) if v is not None]


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def f(v, d=2):
    if v is None:
        return "n/a"
    return f"{v:.{d}f}"


def ci(block):
    if not block or block.get("point") is None:
        return "n/a"
    return f"{f(block['point'])} [{f(block.get('lo'))}, {f(block.get('hi'))}]"


def markdown(r: dict) -> str:
    L = []
    L.append(f"# Round-3 score: {r['candidate']['run']} vs {r['reference']['run']}")
    L.append("")
    L.append(f"Generated {r['generated_utc']}; mode `{r['mode']}`; sets "
             f"`{','.join(r['sets'])}`; entries `{r['entries']}`; workers "
             f"{r['workers']}. CPU only, no training, no SUMO. CAVs "
             "teacher-forced from the recorded arm; each checkpoint scored with "
             "its own TRAIN_RECEIPT physics. Metric = across-states median of "
             "the per-state HDV median |pred-recorded| (arm `wall`, `native` "
             "for seen_controls). Negative paired differences favour the "
             "candidate.")
    L.append("")
    for note in r.get("notes") or []:
        L.append(f"> NOTE: {note}")
        L.append("")
    L.append("## Checkpoints")
    L.append("")
    L.append("| role | source | sha256 | physics merge_cal |")
    L.append("|---|---|---|---|")
    for tag, spec in r["models"].items():
        L.append(f"| {tag} | {spec['source']} | {spec['sha256'][:16]}... | "
                 f"{spec['physics'].get('merge_cal')} |")
    L.append("")
    for metric, key in (("|ds| (m)", "horizon_ds"), ("|dv| (m/s)", "horizon_dv")):
        L.append(f"## HDV {metric} - median / IQR / p90 across states")
        L.append("")
        for set_name in r["sets"]:
            arm = scored_arm(set_name)
            L.append(f"### `{set_name}` (arm `{arm}`)")
            L.append("")
            L.append("| model | h | n | median | IQR | p90 |")
            L.append("|---|---|---:|---:|---:|---:|")
            for tag in r["models"]:
                if set_name not in r[key].get(tag, {}):
                    continue
                for h in HORIZONS:
                    b = r[key][tag][set_name][f"{h:g}"]
                    L.append(f"| {tag} | {h:g}s | {b['n']} | {f(b['median'])} | "
                             f"{f(b['iqr'])} | {f(b['p90'])} |")
            L.append("")
    L.append("## Blow-up rate (|ds|>50 m @60 s and >100 m @120 s)")
    L.append("")
    L.append("| set | model | rate | blowups / evaluated | excluded |")
    L.append("|---|---|---:|---:|---:|")
    for set_name in r["sets"]:
        for tag in r["models"]:
            if set_name not in r["blowup"].get(tag, {}):
                continue
            b = r["blowup"][tag][set_name]
            L.append(f"| {set_name} | {tag} | {f(b['rate'], 3)} | "
                     f"{b['n_blowup']}/{b['n_eval']} | {b['n_excluded']} |")
    L.append("")
    L.append("## Paired world bootstrap of HDV |ds| (row - column; negative "
             "favours the first model)")
    L.append("")
    L.append("| comparison | set | h | difference [95% CI over worlds] | worlds | pairs |")
    L.append("|---|---|---|---|---:|---:|")
    for cmp_name, per_set in r["paired"].items():
        for set_name in r["sets"]:
            if set_name not in per_set:
                continue
            for h in HORIZONS:
                b = per_set[set_name][f"{h:g}"]
                L.append(f"| {cmp_name} | {set_name} | {h:g}s | {ci(b)} | "
                         f"{b['n_worlds']} | {b['n_pairs']} |")
    L.append("")
    if r.get("stages"):
        L.append("## Per-stage checkpoints (quick subset of `wall_same`, 60 s)")
        L.append("")
        L.append("| stage | 60 s median | vs ref_last (paired) |")
        L.append("|---|---:|---|")
        for tag, row in r["stages"].items():
            L.append(f"| {tag} | {f(row['median60'])} | {ci(row.get('paired_vs_ref'))} |")
        L.append("")
    fp = r.get("fingerprints") or {}
    if fp:
        L.append("## Fingerprints (instrumented rollout, quick subset of "
                 "`wall_same`, arm `wall`)")
        L.append("")
        L.append("| fingerprint | " + " | ".join(fp) + " |")
        L.append("|---" * (len(fp) + 1) + "|")

        def row(label, key, fmt=f, h=None):
            cells = []
            for tag in fp:
                v = fp[tag].get(key)
                if h is not None and isinstance(v, dict):
                    v = v.get(h)
                cells.append(fmt(v) if v is not None else "n/a")
            L.append(f"| {label} | " + " | ".join(cells) + " |")

        def block(suffix, getter):
            def r2(label, key, fmt=f, h=None):
                cells = []
                for tag in fp:
                    v = getter(fp[tag]).get(key)
                    if h is not None and isinstance(v, dict):
                        v = v.get(h)
                    cells.append(fmt(v) if v is not None else "n/a")
                L.append(f"| {label}{suffix} | " + " | ".join(cells) + " |")
            r2("mean |g*delta_a| (m/s^2)", "g_delta_mean_abs")
            r2("p95 |g*delta_a| (m/s^2)", "g_delta_p95_abs")
            r2("cap-bind rate", "cap_bind_rate", lambda v: f(v, 4))
            r2("lane-change rate pred (per HDV-step)", "lane_rate_pred", lambda v: f(v, 4))
            r2("lane-change rate recorded", "lane_rate_rec", lambda v: f(v, 4))
            r2("signed speed bias (m/s; >0 too fast)", "speed_bias_mean",
               lambda v: f(v, 3))
        block(" [first 60 s]", lambda d: d)
        block(" [full 120 s]", lambda d: d.get("full") or {})
        row("wrong-lane share of |ds| @60 s", "wrong_lane_share", lambda v: f(v, 3), "60")
        row("wrong-lane share of |ds| @120 s", "wrong_lane_share", lambda v: f(v, 3), "120")
        for name in ("jam", "free"):
            cells = [f"{fp[tag]['jam_free'].get(name, {}).get('n_states', 0)} / "
                     f"60s {f(fp[tag]['jam_free'].get(name, {}).get('ds_med_60'))} / "
                     f"120s {f(fp[tag]['jam_free'].get(name, {}).get('ds_med_120'))}"
                     for tag in fp]
            L.append(f"| {name} slice (n / 60s / 120s median |ds|) | "
                     + " | ".join(cells) + " |")
        cells = [f"60s {f(fp[tag]['slow_hdv_ds'].get('ds_med_60'))} / "
                 f"120s {f(fp[tag]['slow_hdv_ds'].get('ds_med_120'))}" for tag in fp]
        L.append("| recorded-slow HDV |ds| median | " + " | ".join(cells) + " |")
        L.append("")
    hist = r.get("history") or {}
    if hist:
        L.append("## History states")
        L.append("")
        L.append(f"States dir: `{hist.get('states_dir')}`; synthetic short-history "
                 f"adapter: {hist.get('synthetic')}; candidate has history "
                 f"encoder: {hist.get('encoder')}.")
        for name, path in (hist.get("set_files") or {}).items():
            L.append(f"- `{name}`: `{path}`")
        L.append("")
        if hist.get("cut_s") is not None:
            L.append(f"History cut to the last {hist['cut_s']} s (cand_hcut, "
                     "primary set; difference = cut - full, positive = the long "
                     "history helps):")
            L.append("")
            L.append("| h | cut median |ds| | cut - full (paired) |")
            L.append("|---|---:|---|")
            for h in HORIZONS:
                b = hist["cut_horizon_ds"][f"{h:g}"]
                L.append(f"| {h:g}s | {f(b['median'])} | "
                         f"{ci(hist['cut_minus_full'][f'{h:g}'])} |")
            if hist.get("note"):
                L.append("")
                L.append(f"Note: {hist['note']}")
            L.append("")
    rt = r.get("runtime_s") or {}
    L.append("## Runtime")
    L.append("")
    L.append(f"scoring {f(rt.get('scoring_s'), 1)} s; fingerprints "
             f"{f(rt.get('fingerprints_s'), 1)} s; bootstrap "
             f"{f(rt.get('bootstrap_s'), 1)} s; total {f(rt.get('total_s'), 1)} s.")
    L.append("")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--candidate", required=True,
                    help="candidate run dir (wm_s.pt, TRAIN_RECEIPT.json, ...)")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE),
                    help="reference run dir or wm_s.pt (default: old C1 struct)")
    ap.add_argument("--sets", default="wall_same,wall_pilot")
    ap.add_argument("--quick", action="store_true",
                    help="fixed 20-state subset per set (default)")
    ap.add_argument("--full", action="store_true", help="all states of each set")
    ap.add_argument("--workers", type=int, default=2,
                    help="process workers (hard cap 2)")
    ap.add_argument("--inflight", type=int, default=6)
    ap.add_argument("--entries", choices=("off", "v1", "v2"), default="off")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--toggles", default="",
                    help="comma list from {projection,merge_cal} scored as "
                         "extra candidate rows")
    ap.add_argument("--history-states", default=None,
                    help="dir holding 60 s-history state files named like the "
                         "default sets (wall_same.json, wall_pilot.json, ...); "
                         "each set uses DIR/<set>.json when present")
    ap.add_argument("--history-cut-s", type=float, default=None,
                    help="also score the candidate with its pre-t0 history masked "
                         "to the last N seconds (e.g. 4.5) on the primary set")
    ap.add_argument("--synthetic-history", action="store_true",
                    help="SMOKE ONLY: wrap the legacy 10-tick history as a "
                         "run_start_limited long history so history-encoder "
                         "checkpoints can be exercised on the default states")
    ap.add_argument("--no-stages", action="store_true",
                    help="skip per-stage checkpoint scoring (speed)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fingerprint-limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="output dir under _out/r3_score")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t_start = time.time()
    mode = "full" if args.full else "quick"
    workers = max(1, min(MAX_WORKERS, int(args.workers)))
    set_names = [s.strip() for s in args.sets.split(",") if s.strip()]
    for s in set_names:
        if s not in SETS:
            raise SystemExit(f"unknown set {s}; choose from {', '.join(SETS)}")
    primary = "wall_same" if "wall_same" in set_names else set_names[0]
    toggles = [t.strip() for t in args.toggles.split(",") if t.strip()]
    for t in toggles:
        if t not in ("projection", "merge_cal"):
            raise SystemExit(f"unknown toggle {t}")

    tag_out = (f"{mode}_{'_'.join(set_names)}"
               f"{('_' + args.out) if args.out else ''}")
    out_dir = OUT_ROOT / args.out if args.out else OUT_ROOT / tag_out
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = {"scoring_s": 0.0, "fingerprints_s": 0.0, "bootstrap_s": 0.0}

    # ---- models ----------------------------------------------------------- #
    cand_run = run_of(Path(args.candidate))
    ref_run = run_of(Path(args.reference))
    specs: dict[str, dict] = {}
    entries = args.entries != "off"
    entries_v2 = args.entries == "v2"

    def add_spec(tag: str, src: Path, run: Path, *, config_override=None,
                 merge_cal_on=False, physics_run=None):
        staged = stage_copy(src, tag)
        if config_override:
            staged = rewrite_config(staged, config_override)
        specs[tag] = {"source": str(src), "staged": str(staged),
                      "sha256": _sha256(staged),
                      "physics": receipt_physics(physics_run or (
                          run if (run / "TRAIN_RECEIPT.json").is_file()
                          else ref_run)),
                      "entries": entries, "entries_v2": entries_v2,
                      "merge_cal_on": merge_cal_on,
                      "run": str(run)}

    cand_last = cand_run / "wm_s.pt"
    notes = []
    if not cand_last.is_file():
        if (cand_run / "wm_s_best.pt").is_file():
            cand_last = cand_run / "wm_s_best.pt"
            notes.append("candidate has no wm_s.pt: wm_s_best.pt used as "
                         "cand_last (cand_best not scored separately)")
        else:
            raise SystemExit(f"candidate has no wm_s.pt/wm_s_best.pt: {cand_run}")
    ref_phys_run = ref_run
    if not (cand_run / "TRAIN_RECEIPT.json").is_file():
        notes.append("candidate has no TRAIN_RECEIPT.json: physics flags taken "
                     "from the reference receipt (merge_cal etc. may be wrong)")
        add_spec("cand_last", cand_last, cand_run, physics_run=ref_phys_run)
    else:
        add_spec("cand_last", cand_last, cand_run)
    ref_last = ref_run / "wm_s.pt"
    if not ref_last.is_file():
        raise SystemExit(f"reference has no wm_s.pt: {ref_run}")
    ref_receipt_run = ref_run
    add_spec("ref_last", ref_last, ref_receipt_run)

    best_src = cand_run / "wm_s_best.pt"
    cand_best = best_src.is_file() and cand_last.resolve() != best_src.resolve()
    if cand_best:
        add_spec("cand_best", best_src, cand_run)

    stage_srcs = []
    for p in sorted(cand_run.glob("wm_s_stage*.pt")):
        m = re.search(r"wm_s_stage(\d+)\.pt$", p.name)
        if m:
            stage_srcs.append((int(m.group(1)), p))
    stage_srcs.sort()
    if args.no_stages:
        stage_srcs = []
    stage_tags = []
    for stage, src in stage_srcs:
        tag = f"cand_stage{stage}"
        add_spec(tag, src, cand_run)
        stage_tags.append(tag)

    for toggle in toggles:
        if toggle == "projection":
            cfg = ckpt_config(Path(specs["cand_last"]["staged"]))
            if cfg.get("projection_enabled"):
                continue
            add_spec("cand_proj_on", cand_last, cand_run,
                     config_override={"projection_enabled": True,
                                      "projection_min_gap_m": 2.0})
        elif toggle == "merge_cal":
            if receipt_physics(cand_run).get("merge_cal"):
                continue
            add_spec("cand_mergecal_on", cand_last, cand_run, merge_cal_on=True)

    # ---- states ----------------------------------------------------------- #
    subsets: dict[str, str] = {}
    records: dict[str, list[dict]] = {}
    cand_cfg = ckpt_config(Path(specs["cand_last"]["staged"]))
    cand_has_history = has_history_encoder(cand_cfg)
    long_cfg = cand_cfg.get("history_s")
    stride_cfg = float(cand_cfg.get("history_stride_s", 1.0))
    set_files: dict[str, Path] = {}
    for set_name in set_names:
        path = SETS[set_name]
        if args.history_states:
            alt = Path(args.history_states) / f"{set_name}.json"
            if alt.is_file():
                path = alt
            else:
                print(f"WARNING: {alt} missing; {set_name} uses default states",
                      flush=True)
        set_files[set_name] = path

    def prep(recs: list[dict]) -> list[dict]:
        if args.synthetic_history and long_cfg is not None:
            recs = [synthetic_history(r, long_cfg, stride_cfg) for r in recs]
        return recs

    for set_name in set_names:
        recs = prep(select_subset(set_files[set_name]) if mode == "quick"
                    else list(iter_windows(set_files[set_name])))
        if long_cfg is not None and recs:
            h = recs[0].get("history")
            if not (isinstance(h, dict) and h.get("history_s") is not None):
                raise SystemExit(
                    f"candidate uses a {long_cfg} s history encoder but "
                    f"{set_files[set_name]} has no long-history states; pass "
                    "--history-states DIR (60 s-history states) or, for a smoke "
                    "test only, --synthetic-history")
        if args.limit:
            recs = recs[:args.limit]
        records[set_name] = recs
        subsets[set_name] = {
            "n_states": len(recs),
            "n_worlds": len({str(r.get("world")) for r in recs}),
            "worlds": sorted({str(r.get("world")) for r in recs})
            if mode == "quick" else None,
        }
    fp_records = prep(select_subset(set_files[primary]))
    if args.fingerprint_limit:
        fp_records = fp_records[:args.fingerprint_limit]

    # ---- scoring ---------------------------------------------------------- #
    sets_by_tag: dict[str, list[str]] = {}
    for tag in (("cand_last", "ref_last")
                + (("cand_best",) if cand_best else ())
                + tuple(t for t in specs if t.endswith("_on"))):
        sets_by_tag[tag] = set_names
    for tag in stage_tags:
        sets_by_tag[tag] = [primary]

    states: dict[str, dict[str, list]] = {}
    stage_records = prep(select_subset(set_files[primary]))  # stages: always quick subset
    for tag, tag_sets in sets_by_tag.items():
        for set_name in tag_sets:
            t0 = time.time()
            horizons = HORIZONS
            horizon_s = PREDICT_HORIZON
            if tag.startswith("cand_stage"):
                horizons = (STAGE_HORIZON,)
                horizon_s = STAGE_HORIZON
            recs = stage_records if tag.startswith("cand_stage") \
                else records[set_name]
            rows, hit = cached(
                cache_key(specs[tag], "score", recs, repr(horizons)),
                lambda: score_model(tag, specs[tag], recs, workers,
                                    args.inflight, horizons=horizons,
                                    horizon_s=horizon_s))
            if set_name == "seen_controls":
                rows = normalize_seen(rows)
            states.setdefault(tag, {})[set_name] = rows
            timing["scoring_s"] += time.time() - t0
            print(f"scored {tag} [{set_name}]: {len(rows)} states "
                  f"({time.time() - t0:.1f}s){' [cache]' if hit else ''}",
                  flush=True)
            dest = out_dir / f"states__{tag}__{set_name}.json"
            dest.write_text(json.dumps({"model": tag, "set": set_name,
                                        "states": rows}), encoding="utf-8")

    # ---- aggregate tables ------------------------------------------------- #
    horizon_ds, horizon_dv, blowups = {}, {}, {}
    for tag, tag_sets in states.items():
        horizon_ds[tag], horizon_dv[tag], blowups[tag] = {}, {}, {}
        for set_name in tag_sets:
            arm = scored_arm(set_name)
            st = states[tag][set_name]
            horizon_ds[tag][set_name] = {
                f"{h:g}": block_field(st, arm, h, "ds_med") for h in HORIZONS}
            horizon_dv[tag][set_name] = {
                f"{h:g}": block_field(st, arm, h, "dv_med") for h in HORIZONS}
            blowups[tag][set_name] = blowup_rate(st, arm)

    # ---- paired bootstrap ------------------------------------------------- #
    t0 = time.time()
    paired: dict[str, dict] = {}
    comparisons = [("cand_last", "ref_last")]
    if cand_best:
        comparisons += [("cand_best", "ref_last"), ("cand_best", "cand_last")]
    for tag in specs:
        if tag.endswith("_proj_on") or tag.endswith("_mergecal_on"):
            comparisons.append((tag, "ref_last"))
    for a, b in comparisons:
        block = {}
        for set_name in set_names:
            if set_name not in states.get(a, {}) or set_name not in states.get(b, {}):
                continue
            arm = scored_arm(set_name)
            block[set_name] = {
                f"{h:g}": paired_bootstrap(states[a][set_name], states[b][set_name],
                                           arm, h, n_boot=args.n_boot,
                                           seed=args.seed)
                for h in HORIZONS}
        paired[f"{a}-{b}"] = block
    timing["bootstrap_s"] = time.time() - t0

    # ---- stage table (60 s, quick subset of primary) ---------------------- #
    stage_rows = {}
    if stage_tags:
        stage_keys = {(str(r.get("world")), round(float(r.get("t0_s", 0.0)), 3))
                      for r in stage_records}
        ref_subset = [s for s in states["ref_last"][primary]
                      if (str(s.get("world")), round(float(s.get("t0_s", 0.0)), 3))
                      in stage_keys]
        fp60_ref = block_field(ref_subset, scored_arm(primary), STAGE_HORIZON)
        for tag in stage_tags:
            blk = block_field(states[tag][primary], scored_arm(primary),
                              STAGE_HORIZON)
            pr = paired_bootstrap(states[tag][primary], states["ref_last"][primary],
                                  scored_arm(primary), STAGE_HORIZON,
                                  n_boot=args.n_boot, seed=args.seed)
            stage_rows[tag] = {"median60": blk["median"],
                               "n": blk["n"], "paired_vs_ref": pr,
                               "ref_last_median60": fp60_ref["median"]}

    # ---- fingerprints ------------------------------------------------------ #
    fingerprints = {}
    t0 = time.time()
    arm = scored_arm(primary)
    for tag in ("cand_last", "ref_last"):
        print(f"fingerprint {tag} on {len(fp_records)} states", flush=True)
        fingerprints[tag], _hit = cached(
            cache_key(specs[tag], "fingerprint", fp_records, arm + "|fp2"),
            lambda tag=tag: fingerprint_model(tag, specs[tag], fp_records, arm))
    timing["fingerprints_s"] = time.time() - t0
    (out_dir / "fingerprints.json").write_text(
        json.dumps(fingerprints, indent=1), encoding="utf-8")

    # ---- history cut (causal-history fingerprint) -------------------------- #
    history = {"states_dir": args.history_states, "synthetic": bool(args.synthetic_history),
               "set_files": {k: str(v) for k, v in set_files.items()},
               "cut_s": args.history_cut_s, "encoder": cand_has_history}
    if args.history_cut_s is not None:
        recs = records[primary]
        cut_recs = [cut_history(r, args.history_cut_s) for r in recs]
        rows, hit = cached(
            cache_key(specs["cand_last"], "score", cut_recs,
                      f"{HORIZONS!r}cut{args.history_cut_s}"),
            lambda: score_model("cand_hcut", specs["cand_last"], cut_recs, workers,
                                args.inflight))
        states["cand_hcut"] = {primary: rows}
        arm_p = scored_arm(primary)
        history["cut_horizon_ds"] = {f"{h:g}": block_field(rows, arm_p, h)
                                     for h in HORIZONS}
        history["cut_minus_full"] = {
            f"{h:g}": paired_bootstrap(rows, states["cand_last"][primary], arm_p, h,
                                       n_boot=args.n_boot, seed=args.seed)
            for h in HORIZONS}
        (out_dir / f"states__cand_hcut__{primary}.json").write_text(
            json.dumps({"model": "cand_hcut", "set": primary, "states": rows}),
            encoding="utf-8")
        if not cand_has_history:
            history["note"] = ("candidate has no long-history encoder; the cut "
                               "changes only the legacy 10-tick driver history")

    timing["total_s"] = time.time() - t_start

    result = {
        "schema": "r3_score_round/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "mode": mode,
        "subset_rule": "sort (world,t0_s,key); first state per world; "
                       f"first {QUICK_WORLDS} worlds (quick mode)",
        "sets": {s: subsets[s] for s in set_names},
        "entries": args.entries,
        "n_boot": args.n_boot, "seed": args.seed,
        "workers": workers,
        "candidate": {"run": str(cand_run)},
        "reference": {"run": str(ref_run),
                      "default": str(ref_run) == str(DEFAULT_REFERENCE)},
        "models": specs,
        "notes": notes,
        "horizon_ds": horizon_ds,
        "horizon_dv": horizon_dv,
        "blowup": blowups,
        "paired": paired,
        "stages": stage_rows,
        "fingerprints": fingerprints,
        "history": history,
        "runtime_s": {k: round(v, 1) for k, v in timing.items()},
        "sources": {
            "scoring_core": "ws-main2score/_out/main2_score/main2_score.py",
            "subset_and_stages": "ws-bisect-stages/_out/bisect_stages_score.py",
            "fingerprints": "ws-bisect-residual/_out/bisect_residual/run_bisect.py",
            "states_dir": str(STATES_DIR),
        },
    }
    (out_dir / "ROUND_SCORE.json").write_text(
        json.dumps(result, indent=1), encoding="utf-8")
    (out_dir / "ROUND_SCORE.md").write_text(markdown(result), encoding="utf-8")
    print(f"wrote {out_dir / 'ROUND_SCORE.json'} and .md "
          f"({timing['total_s']:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
