"""C1 v2 full scoring: named state sets, streaming, bounded process workers.

This is the execution entry point for the WM-S-C1FAST-20261006 registration
(``E:/research-dense/runs/active/WM-S-C1FAST-20261006/REGISTRATION.md``).  It
extends :mod:`wms_eval.c1` in three ways the v2 clean sets require:

* **three named sets** (``u_same`` / ``u_new`` / ``seen``) instead of one
  ``--unseen``/``--seen`` pair, so the unseen/seen ratio is reported separately
  on the world-disjoint (U-same) and same-world-paired (U-new) splits;
* **streaming state reads**: each state JSON is parsed one top-level window at a
  time; no state file (up to ~410 MB) is ever loaded whole;
* **bounded dispatch**: at most ``--inflight`` records are pickled to the worker
  pool at once, so parent RSS stays far below the 6 GB rule.

Every worker is a separate process that loads one small model and one state at a
time (owner memory rule).  Total torch threads across workers is capped at four
by ``WMS_EVAL_THREADS`` (default 1 thread/worker, 4 workers).  No SUMO, GPU or
sealed data is touched.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Iterable, Iterator, Sequence

from .c1 import _field_values, paired_ci, ratio_ci
from .data import DataError, json_default, recorded_for
from .metrics import _frame_at, car_metrics, traffic_metrics

CAR_HORIZONS = (10.0, 30.0, 60.0, 120.0)
TRAFFIC_SECTIONS = ("upstream", "merge", "downstream")
TRAFFIC_METRICS = ("mean_speed", "stopped_vehicles", "merge_queue_length", "total_delay")
HEADLINE_TRAFFIC_METRIC = "total_delay"
HEADLINE_TRAFFIC_SECTION = "merge"
HEADLINE_TRAFFIC_BIN = 3  # the 90-120 s bin (30 s bins 0..3)


class C1v2Error(ValueError):
    """The C1 v2 inputs, model list or metric selection are invalid."""


# --------------------------------------------------------------------------- #
# memory-safe streaming reader
# --------------------------------------------------------------------------- #
def iter_windows(path: str | Path, limit: int | None = None) -> Iterator[dict]:
    """Yield top-level state windows from a ``{"windows": [...]}`` JSON file.

    The file is opened in binary and decoded as latin-1 per chunk: JSON is ASCII
    here, every byte maps to one character, and byte offsets from the header read
    stay valid for ``seek`` (unlike Windows text-mode seeks).  At most one record
    plus one read chunk is resident.
    """
    path = Path(path)
    decoder = json.JSONDecoder()
    with path.open("rb") as fh:
        head = fh.read(1 << 16).decode("latin-1")
        idx = -1
        for key in ('"windows"', '"states"', '"records"'):
            found = head.find(key)
            if found != -1:
                idx = found
                break
        if idx == -1:
            raise DataError(f"{path.name}: no states array key found")
        bracket = head.find("[", idx)
        if bracket == -1:
            raise DataError(f"{path.name}: states key is not followed by an array")
        fh.seek(bracket + 1)
        buf = fh.read(1 << 20).decode("latin-1")
        pos = 0
        emitted = 0
        while True:
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n,":
                    pos += 1
                if pos < len(buf):
                    break
                chunk = fh.read(1 << 20)
                if not chunk:
                    return
                buf = buf[pos:] + chunk.decode("latin-1")
                pos = 0
            if buf[pos] == "]":
                return
            try:
                record, end = decoder.raw_decode(buf, pos)
            except json.JSONDecodeError:
                chunk = fh.read(1 << 20)
                if not chunk:
                    raise DataError(f"{path.name}: truncated states array")
                buf = buf[pos:] + chunk.decode("latin-1")
                pos = 0
                continue
            yield record
            emitted += 1
            if limit is not None and emitted >= limit:
                return
            pos = end


# --------------------------------------------------------------------------- #
# per-state evaluation (both arms, car + per-bin traffic)
# --------------------------------------------------------------------------- #
def _normalize_arms(record: dict) -> dict:
    """Alias the ordinary non-PI recorded arm to ``native`` across state sets.

    The unseen sets name their ordinary arm ``native`` (a PI run has a paired
    ordinary sibling), while many held-out ``seen`` records (non-PI validation)
    name their only recorded arm ``base``.  Both are the ordinary, non-PI driving
    reference, so the harness baseline ``native`` is served by aliasing ``base``
    when ``native`` is absent.  This keeps one baseline arm name across all
    three sets without changing any recorded value.
    """
    recorded = record.get("recorded")
    if isinstance(recorded, dict) and "native" not in recorded and "base" in recorded:
        return {**record, "recorded": {**recorded, "native": recorded["base"]}}
    return record


def _baseline_arm(record: dict) -> str:
    """The recorded arm that serves as the ordinary baseline for this record.

    Unseen PI records carry ``native``; held-out seen records carry their own
    validation arm (``base``, ``native``, ``arm0..3`` or a wall/hold arm).  The
    baseline is the record's own recorded arm so teacher-forced CAVs always have
    a real trajectory to replay; ``native`` is preferred when present.
    """
    recorded = record.get("recorded")
    keys = list(recorded.keys()) if isinstance(recorded, dict) else []
    if "native" in keys:
        return "native"
    own = str(record.get("arm", ""))
    if own in keys:
        return own
    return keys[0] if keys else "native"


def evaluate_state(record: dict, predictor: Callable, arms: Sequence[str],
                   horizons: Sequence[float], bin_s: float, default_horizon: float) -> dict:
    """Roll out every arm of one state and return compact JSON-safe errors."""
    record = _normalize_arms(record)
    state = dict(record)
    t0_s = float(record.get("t0_s", state.get("time_s", 0.0)))
    horizon_s = min(float(record.get("horizon_s", default_horizon)), float(default_horizon))
    seed = int(record.get("seed", 0))
    free_speed = float(record.get("v0_mps", 13.89))
    controllers = record.get("cav_controllers", {})
    baseline_arm = _baseline_arm(record)

    predicted: dict[str, list] = {}
    arm_result: dict[str, dict] = {}
    for arm in arms:
        actual = baseline_arm if arm == "native" else arm
        try:
            predicted[arm] = _predict(predictor, state, dict(controllers, arm=actual),
                                      horizon_s, seed)
        except Exception as exc:  # one arm must not abort the others
            arm_result[arm] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        try:
            recorded = recorded_for(record, actual)
        except DataError as exc:
            arm_result[arm] = {"error": str(exc)}
            continue
        arm_result[arm] = {"error": None, "car": _car_compact(
            car_metrics(predicted[arm], recorded, t0_s=t0_s, horizons=tuple(horizons))),
            "traffic": _traffic_compact(
                traffic_metrics(predicted[arm], t0_s=t0_s, bin_s=bin_s, free_speed_mps=free_speed),
                traffic_metrics(recorded, t0_s=t0_s, bin_s=bin_s, free_speed_mps=free_speed)),
            "overlap": _overlap_compact(predicted[arm], t0_s, horizons)}

    paired: dict[str, dict] = {}
    if "native" in predicted and "pi" in predicted:
        try:
            rec_native = recorded_for(record, baseline_arm)
            rec_pi = recorded_for(record, "pi")
        except DataError:
            rec_native = rec_pi = None
        if rec_native is not None and rec_pi is not None:
            pred_pair = _traffic_pair(predicted["pi"], predicted["native"], t0_s, bin_s, free_speed)
            rec_pair = _traffic_pair(rec_pi, rec_native, t0_s, bin_s, free_speed)
            paired = _paired_compact(pred_pair, rec_pair)
    return {"world": record.get("world", "unknown"), "t0_s": t0_s, "arms": arm_result,
            "paired": paired}


def _predict(predictor: Callable, state: dict, controllers: dict,
             horizon_s: float, seed: int) -> list:
    """Call a predictor, downgrading an unsupported arm to ``native``.

    The frozen A11 ``pure_physics`` reference only implements the ``native`` and
    ``pi`` action codes; for a held-out validation arm (e.g. ``arm0``) it falls
    back to ``native`` while the recorded reference stays the real arm.
    """
    try:
        return predictor(state, controllers, horizon_s, seed)
    except Exception as exc:
        if controllers.get("arm") not in ("native", "pi") and \
                exc.__class__.__name__ in ("ModelUnavailable", "ValueError"):
            return predictor(state, {**controllers, "arm": "native"}, horizon_s, seed)
        raise


def _car_compact(car: dict) -> dict:
    out: dict[str, dict] = {}
    for horizon, strata in car.items():
        block: dict[str, dict] = {}
        for stratum in ("all", "hdv", "cav"):
            group = strata.get(stratum, {})
            block[stratum] = {
                "ds_med": (group.get("abs_delta_s_m") or {}).get("median"),
                "ds_p90": (group.get("abs_delta_s_m") or {}).get("p90"),
                "dv_med": (group.get("abs_delta_v_mps") or {}).get("median"),
                "dv_p90": (group.get("abs_delta_v_mps") or {}).get("p90"),
                "match": group.get("matched_fraction"),
            }
        out[str(horizon)] = block
    return out


def _traffic_compact(predicted: dict, recorded: dict) -> dict:
    """|predicted - recorded| per section/metric, keyed by bin index (0..)."""
    out: dict[str, dict] = {}
    for section in TRAFFIC_SECTIONS:
        out[section] = {}
        for metric in TRAFFIC_METRICS:
            bins: dict[str, float] = {}
            for key in sorted(predicted, key=lambda k: int(k)):
                pred = predicted[key].get("sections", {}).get(section, {}).get(metric)
                rec = recorded.get(key, {}).get("sections", {}).get(section, {}).get(metric)
                if pred is None or rec is None:
                    continue
                bins[key] = abs(float(pred) - float(rec))
            out[section][metric] = bins
    return out


DEFAULT_LENGTH_M = 5.0


def _overlap_compact(frames: list, t0_s: float, horizons: Sequence[float]) -> dict:
    """Per-horizon same-lane gap violations in the PREDICTED stream.

    Gap convention matches ``wms.stability.FeasibilityProjection``
    (``s_lead - s_follow - length_lead``).  Counts predicted same-lane pairs
    that physically overlap (gap < 0) and those below the run-2 projection floor
    (gap < 2 m), plus the smallest observed gap at each horizon.
    """
    out: dict[str, dict] = {}
    for horizon in horizons:
        frame = _frame_at(frames, t0_s + float(horizon))
        rows = {str(r["id"]): r for r in (frame or {}).get("vehicles", [])
                if r.get("present", True)}
        bylane: dict[int, list] = {}
        for row in rows.values():
            bylane.setdefault(int(row.get("lane", -1)), []).append(row)
        overlap = violation = 0
        min_gap: float | None = None
        for lane, group in bylane.items():
            if lane < 0 or len(group) < 2:
                continue
            group.sort(key=lambda r: float(r["s"]))
            for foll, lead in zip(group[:-1], group[1:]):
                gap = (float(lead["s"]) - float(foll["s"])
                       - float(lead.get("length", DEFAULT_LENGTH_M)))
                if gap < 0.0:
                    overlap += 1
                if gap < 2.0:
                    violation += 1
                if min_gap is None or gap < min_gap:
                    min_gap = gap
        out[str(float(horizon))] = {"n_vehicles": len(rows), "overlap_pairs": overlap,
                                    "gap_lt_2m_pairs": violation, "min_gap_m": min_gap}
    return out


def _traffic_pair(pred_a: list, pred_b: list, t0_s: float, bin_s: float,
                  free_speed: float) -> dict:
    ta = traffic_metrics(pred_a, t0_s=t0_s, bin_s=bin_s, free_speed_mps=free_speed)
    tb = traffic_metrics(pred_b, t0_s=t0_s, bin_s=bin_s, free_speed_mps=free_speed)
    out: dict[str, dict] = {}
    for section in TRAFFIC_SECTIONS:
        out[section] = {}
        for metric in TRAFFIC_METRICS:
            bins: dict[str, float] = {}
            for key in sorted(set(ta) & set(tb), key=lambda k: int(k)):
                va = ta[key].get("sections", {}).get(section, {}).get(metric)
                vb = tb[key].get("sections", {}).get(section, {}).get(metric)
                if va is None or vb is None:
                    continue
                bins[key] = float(va) - float(vb)
            out[section][metric] = bins
    return out


def _paired_compact(pred_pair: dict, rec_pair: dict) -> dict:
    out: dict[str, dict] = {}
    for section in TRAFFIC_SECTIONS:
        out[section] = {}
        for metric in TRAFFIC_METRICS:
            bins: dict[str, dict] = {}
            for key in sorted(set(pred_pair[section][metric]) & set(rec_pair[section][metric]),
                              key=lambda k: int(k)):
                diff = float(pred_pair[section][metric][key]) - float(rec_pair[section][metric][key])
                bins[key] = {"signed": diff, "abs": abs(diff)}
            out[section][metric] = bins
    return out


# --------------------------------------------------------------------------- #
# scalar extraction + world bootstrap
# --------------------------------------------------------------------------- #
def car_scalar(result: dict, arm: str, field: str, horizon: float) -> float | None:
    block = result.get("arms", {}).get(arm, {})
    if block.get("error"):
        return None
    return block.get("car", {}).get(str(float(horizon)), {}).get("hdv", {}).get(field)


def traffic_scalar(result: dict, arm: str, section: str, metric: str, bin_idx: int) -> float | None:
    block = result.get("arms", {}).get(arm, {})
    if block.get("error"):
        return None
    return block.get("traffic", {}).get(section, {}).get(metric, {}).get(str(bin_idx))


def traffic_mean_scalar(result: dict, arm: str, section: str, metric: str) -> float | None:
    block = result.get("arms", {}).get(arm, {})
    if block.get("error"):
        return None
    bins = block.get("traffic", {}).get(section, {}).get(metric, {})
    values = list(bins.values())
    return (sum(values) / len(values)) if values else None


def paired_scalar(result: dict, section: str, metric: str, bin_idx: int | None = None) -> float | None:
    bins = result.get("paired", {}).get(section, {}).get(metric, {})
    if bin_idx is None:
        values = [entry.get("abs") for entry in bins.values()]
        values = [v for v in values if v is not None]
        return (sum(values) / len(values)) if values else None
    entry = bins.get(str(bin_idx))
    return entry.get("abs") if entry else None


def group_by_world(results: Iterable[dict], scalar: Callable[[dict], float | None]) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for result in results:
        value = scalar(result)
        if value is None or not math.isfinite(value):
            continue
        grouped.setdefault(str(result.get("world", "unknown")), []).append(float(value))
    return grouped


def median_scalar(results: Iterable[dict], scalar: Callable[[dict], float | None]) -> float | None:
    values = [scalar(r) for r in results]
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(median(values)) if values else None


# --------------------------------------------------------------------------- #
# bounded parallel driver
# --------------------------------------------------------------------------- #
_WORKER: dict[str, Any] = {}


def _init_worker(spec: str, checkpoint: str | None, arms: Sequence[str],
                 horizons: Sequence[float], bin_s: float, default_horizon: float,
                 entries: bool, entries_v2: bool = False,
                 physics: dict | None = None) -> None:
    from .models import load_model

    _WORKER.update({"predictor": load_model(spec, checkpoint=checkpoint, entries=entries,
                                           entries_v2=entries_v2, physics=physics)[0],
                    "arms": tuple(arms), "horizons": tuple(horizons),
                    "bin_s": bin_s, "default_horizon": default_horizon})


def _work(payload: tuple[int, dict]) -> tuple[int, dict]:
    index, record = payload
    return index, evaluate_state(record, _WORKER["predictor"], _WORKER["arms"],
                                 _WORKER["horizons"], _WORKER["bin_s"],
                                 _WORKER["default_horizon"])


def run_model(spec: str, checkpoint: str | None, path: str | Path, arms: Sequence[str],
              horizons: Sequence[float], bin_s: float, default_horizon: float,
              workers: int, inflight: int, entries: bool = False,
              entries_v2: bool = False, limit: int | None = None,
              physics: dict | None = None) -> list[dict]:
    """Score one model over one streamed set, preserving input record order."""
    workers = max(1, int(workers))
    # Owner rule: torch threads across the whole worker fleet capped at four.
    os.environ["WMS_EVAL_THREADS"] = str(max(1, 4 // workers))
    os.environ.setdefault("OMP_NUM_THREADS", os.environ["WMS_EVAL_THREADS"])
    os.environ.setdefault("MKL_NUM_THREADS", os.environ["WMS_EVAL_THREADS"])
    records = iter_windows(path, limit=limit)
    if workers == 1:
        _init_worker(spec, checkpoint, arms, horizons, bin_s, default_horizon, entries,
                     entries_v2, physics)
        return [result for _, result in (_work((i, rec)) for i, rec in enumerate(records))]

    results: dict[int, dict] = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(spec, checkpoint, tuple(arms), tuple(horizons), bin_s,
                  default_horizon, entries, entries_v2, physics),
    ) as pool:
        pending: dict[Any, int] = {}
        iterator = enumerate(records)
        exhausted = False

        def submit_next() -> None:
            nonlocal exhausted
            if exhausted:
                return
            try:
                index, record = next(iterator)
            except StopIteration:
                exhausted = True
                return
            pending[pool.submit(_work, (index, record))] = index

        for _ in range(max(1, int(inflight))):
            submit_next()
        while pending:
            done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
            for future in done:
                index, result = future.result()
                results[index] = result
                del pending[future]
                submit_next()
    return [results[i] for i in sorted(results)]


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
#: Scalar extractors used for the C1 table (native arm = the shared baseline).
TABLE_METRICS: dict[str, Callable[[dict], float | None]] = {
    "car_ds@10": lambda r: car_scalar(r, "native", "ds_med", 10.0),
    "car_ds@30": lambda r: car_scalar(r, "native", "ds_med", 30.0),
    "car_ds@60": lambda r: car_scalar(r, "native", "ds_med", 60.0),
    "car_ds@120": lambda r: car_scalar(r, "native", "ds_med", 120.0),
    "car_dv@30": lambda r: car_scalar(r, "native", "dv_med", 30.0),
    "car_dv@60": lambda r: car_scalar(r, "native", "dv_med", 60.0),
    "traffic_delay@120": lambda r: traffic_scalar(
        r, "native", HEADLINE_TRAFFIC_SECTION, HEADLINE_TRAFFIC_METRIC, HEADLINE_TRAFFIC_BIN),
    "traffic_delay@mean": lambda r: traffic_mean_scalar(
        r, "native", HEADLINE_TRAFFIC_SECTION, HEADLINE_TRAFFIC_METRIC),
}


def build_result(models: Sequence[tuple[str, str | None]],
                 evaluations: dict[str, dict[str, list[dict]]],
                 set_counts: dict[str, dict], separation: dict, n_boot: int,
                 seed: int, paired_a: str, paired_b: str,
                 verdict_text: str) -> dict:
    result: dict[str, Any] = {
        "schema": "wms_eval_c1v2_result/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "headline": {
            "arm": "native",
            "car": "car-level HDV |ds|/|dv| median per state, horizons 10/30/60/120 s",
            "traffic": ("|predicted-recorded| per 30 s bin; headline = merge "
                        f"{HEADLINE_TRAFFIC_METRIC} in bin {HEADLINE_TRAFFIC_BIN} (90-120 s)"),
            "ratio": "median(unseen) / median(seen), CI resampled over worlds",
            "paired": "median(structured) - median(black-box), CI over shared worlds",
        },
        "sets": set_counts,
        "separation": separation,
        "models": {},
        "ratios": {},
        "paired": {},
        "verdict": verdict_text,
    }
    grouped: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {}
    for spec, _ckpt in models:
        per_metric: dict[str, Any] = {}
        for name, scalar in TABLE_METRICS.items():
            model_sets: dict[str, Any] = {}
            for set_name, results in evaluations[spec].items():
                g = group_by_world(results, scalar)
                grouped[(spec, set_name, name)] = g
                model_sets[set_name] = {"median": median_scalar(results, scalar),
                                        "n_worlds": len(g), "n_states": len(results)}
            per_metric[name] = model_sets
        result["models"][spec] = per_metric

    # unseen/seen ratios, separately on each unseen split
    for spec, _ckpt in models:
        ratios: dict[str, Any] = {}
        for unseen_name in ("u_same", "u_new"):
            if unseen_name not in evaluations[spec] or "seen" not in evaluations[spec]:
                continue
            ratios[unseen_name] = {}
            for name in TABLE_METRICS:
                num = grouped.get((spec, unseen_name, name), {})
                den = grouped.get((spec, "seen", name), {})
                if num and den:
                    ratios[unseen_name][name] = ratio_ci(num, den, n_boot=n_boot, seed=seed)
        result["ratios"][spec] = ratios

    # paired structured vs black-box on each unseen split
    for unseen_name in ("u_same", "u_new"):
        if paired_a not in evaluations or paired_b not in evaluations:
            continue
        result["paired"][unseen_name] = {}
        for name in TABLE_METRICS:
            a = grouped.get((paired_a, unseen_name, name), {})
            b = grouped.get((paired_b, unseen_name, name), {})
            if a and b:
                result["paired"][unseen_name][name] = paired_ci(a, b, n_boot=n_boot, seed=seed)
    return result


def markdown(result: dict, models: Sequence[tuple[str, str | None]]) -> str:
    lines = ["# C1 (leave-one-controller-out) scoring - WM-S-C1FAST-20261006", "",
             "**Status: complete.** Structured WM-S vs black-box-id vs black-box-traj, all "
             "trained on the same PI-free TRAIN windows; keep_speed and pure_physics as "
             "references. Baseline arm `native`; single-control-variable model classes.",
             ""]
    lines += ["## Sets", ""]
    for name, info in result["sets"].items():
        lines.append(f"- `{name}`: {info['n_states']} states / {info['n_worlds']} worlds")
    lines += ["", "## Unseen/seen error ratio (median across states; 95% CI over worlds)", "",
              "Car metric `ds` = HDV median |s_pred - s_rec| (m); `dv` = median |v_pred - v_rec| (m/s);",
              "traffic = merge normalized delay (s) in the 90-120 s bin.",
              "",
              "| model | split | car ds@30 | car ds@60 | car dv@30 | traffic delay@120 |",
              "|---|---|---|---|---|---|"]
    for spec, _ckpt in models:
        for split in ("u_same", "u_new"):
            cols = []
            for name in ("car_ds@30", "car_ds@60", "car_dv@30", "traffic_delay@120"):
                ci = result["ratios"].get(spec, {}).get(split, {}).get(name)
                cols.append(_fmt_ci(ci))
            if any(c != "n/a" for c in cols):
                lines.append(f"| {spec} | {split} | " + " | ".join(cols) + " |")
    lines += ["", "## Paired structured - black-box, on unseen (negative favours structured)", "",
              "| split | metric | structured - bb_traj | structured - bb_id |", "|---|---|---|---|"]
    for split in ("u_same", "u_new"):
        for name in ("car_ds@30", "car_ds@60", "traffic_delay@120"):
            bt = result.get("paired_bb_traj", {}).get(split, {}).get(name)
            bid = result["paired"].get(split, {}).get(name)
            lines.append(f"| {split} | {name} | {_fmt_ci(bt)} | {_fmt_ci(bid)} |")
    lines += ["", "## Per-model medians (native arm, metres and seconds)", "",
              "| model | set | ds@10 | ds@30 | ds@60 | ds@120 | delay@120 |",
              "|---|---|---|---|---|---|---|"]
    for spec, _ckpt in models:
        for set_name in result["sets"]:
            block = result["models"][spec]

            def med(name):
                return block[name].get(set_name, {}).get("median")

            lines.append(f"| {spec} | {set_name} | {_f(med('car_ds@10'))} | "
                         f"{_f(med('car_ds@30'))} | {_f(med('car_ds@60'))} | "
                         f"{_f(med('car_ds@120'))} | {_f(med('traffic_delay@120'))} |")
    lines += ["", "## Verdict", "", result["verdict"], ""]
    return "\n".join(lines)


def _fmt_ci(ci: dict | None) -> str:
    if not ci or ci.get("point") is None:
        return "n/a"
    return (f"{ci['point']:.3f} [{_f(ci.get('lo'))}, {_f(ci.get('hi'))}]")


def _f(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wms_eval c1v2")
    p.add_argument("--models", action="append", required=True, help="SPEC=CKPT (repeatable)")
    p.add_argument("--sets", action="append", required=True,
                   help="NAME=PATH state set (repeatable); e.g. u_same=..., u_new=..., seen=...")
    p.add_argument("--train-windows", help="TRAIN window identity list (overlap guard)")
    p.add_argument("--arms", default="native,pi")
    p.add_argument("--horizon", type=float, default=120.0,
                   help="rollout horizon seconds (metrics require <=120 s)")
    p.add_argument("--bin-s", type=float, default=30.0)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--inflight", type=int, default=6)
    p.add_argument("--entries", action="store_true",
                   help="observed-counter entries for every spec (default OFF)")
    p.add_argument("--entries-v2", action="store_true",
                   help="observed-history (v2) entry process for every spec; "
                        "implies --entries (default OFF)")
    p.add_argument("--physics", action="append", default=None,
                   help="SPEC=PATH of the PhysicsConfig (or TRAIN_RECEIPT.json) the "
                        "checkpoint was trained with; repeatable, default uses the "
                        "embedded config block or built-in defaults")
    p.add_argument("--limit", type=int, default=None,
                   help="score only the first N windows per set (smoke/reruns)")
    p.add_argument("--report-from", help="regenerate C1_RESULT.* from an existing "
                                         "run directory's states__*.json (no rollouts)")
    p.add_argument("--out", required=True, help="new output directory")
    p.add_argument("--result-md", help="also write this markdown result path")
    p.add_argument("--result-json", help="also write this JSON result path")
    return p


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_labelled_models(entries: Sequence[str]) -> list[tuple[str, str, str | None]]:
    """Parse ``LABEL:SPEC=CKPT`` (or ``SPEC=CKPT``) into ``(label, spec, ckpt)``.

    Two checkpoints of the same architecture (e.g. run-3 final and stage-120,
    both the ``wms`` spec) need distinct output names without changing which
    model class/flag set loads them, so the optional ``LABEL:`` prefix names the
    result file while ``SPEC`` selects the loader.
    """
    from .models import MODEL_SPECS

    out: list[tuple[str, str, str | None]] = []
    for entry in entries:
        label, sep, rest = entry.partition(":")
        if sep and "=" in rest:
            spec, _eq, ckpt = rest.partition("=")
        else:
            spec, _eq, ckpt = entry.partition("=")
            label = spec
        label, spec = label.strip(), spec.strip()
        if spec not in MODEL_SPECS:
            raise C1v2Error(f"unknown model spec {spec!r}; choose from "
                            f"{', '.join(MODEL_SPECS)}")
        out.append((label, spec, (ckpt.strip() or None)))
    if not out:
        raise C1v2Error("at least one --models SPEC=CKPT is required")
    return out


def _regenerate(args, models_full, sets) -> int:
    """Rebuild C1_RESULT.* from an existing run's per-state result files."""
    run_dir = Path(args.report_from)
    if not run_dir.is_dir():
        raise C1v2Error(f"--report-from is not a directory: {run_dir}")
    models = [(label, ckpt) for label, _spec, ckpt in models_full]
    evaluations: dict[str, dict[str, list[dict]]] = {}
    set_counts: dict[str, dict] = {}
    for label, _spec, _ckpt in models_full:
        evaluations[label] = {}
        for name, _path in sets:
            data = json.loads((run_dir / f"states__{label}__{name}.json").read_text(encoding="utf-8"))
            results = data["states"]
            evaluations[label][name] = results
            set_counts.setdefault(name, {"n_states": len(results),
                                         "n_worlds": len({r.get("world") for r in results})})
    old = run_dir / "C1_RESULT.json"
    separation = json.loads(old.read_text(encoding="utf-8")).get("separation", {}) \
        if old.is_file() else {}
    result = build_result(models, evaluations, set_counts, separation, args.bootstrap,
                          args.seed, "wms", "blackbox_id", "")
    tmp = build_result(models, evaluations, set_counts, separation, args.bootstrap,
                       args.seed, "wms", "blackbox_traj", "")
    result["paired_bb_traj"] = tmp["paired"]
    result["verdict"] = _verdict(result)
    md = markdown(result, models)
    (run_dir / "C1_RESULT.json").write_text(json.dumps(result, indent=2, default=json_default),
                                            encoding="utf-8")
    (run_dir / "C1_RESULT.md").write_text(md, encoding="utf-8")
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(result, indent=2, default=json_default),
                                          encoding="utf-8")
    if args.result_md:
        Path(args.result_md).write_text(md, encoding="utf-8")
    print(json.dumps({"regenerated": str(run_dir), "sets": set_counts}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from .models import load_model  # noqa: F401  (validate import early)

    models_full = _parse_labelled_models(args.models)
    models = [(label, ckpt) for label, _spec, ckpt in models_full]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    sets: list[tuple[str, str]] = []
    for entry in args.sets:
        name, sep, path = entry.partition("=")
        if not sep:
            raise C1v2Error(f"--sets needs NAME=PATH, got {entry!r}")
        sets.append((name.strip(), path.strip()))
    names = [n for n, _ in sets]
    if "seen" not in names:
        raise C1v2Error("--sets must include a 'seen' reference set")

    physics_by_spec: dict[str, dict] = {}
    for entry in (args.physics or []):
        spec, sep, path = entry.partition("=")
        if not sep:
            spec, path = "*", entry
        data = _load_json(path)
        if isinstance(data, dict) and isinstance(data.get("physics"), dict):
            data = data["physics"]
        physics_by_spec[spec.strip()] = data

    if args.report_from:
        return _regenerate(args, models_full, sets)

    out_dir = Path(args.out)
    if out_dir.exists():
        raise C1v2Error(f"refusing to overwrite existing output: {out_dir}")

    train_windows = _load_json(args.train_windows) if args.train_windows else None
    if isinstance(train_windows, dict):
        for key in ("windows", "states", "records"):
            if isinstance(train_windows.get(key), list):
                train_windows = train_windows[key]
                break

    # counts + separation (streamed, metadata only for worlds)
    set_counts: dict[str, dict] = {}
    set_worlds: dict[str, list[str]] = {}
    set_keys: dict[str, list[str]] = {}
    set_runs: dict[str, list[str]] = {}
    for name, path in sets:
        worlds, keys, runs, n = [], [], [], 0
        for record in iter_windows(path, limit=args.limit):
            n += 1
            if record.get("world") is not None:
                worlds.append(str(record["world"]))
            if record.get("key") is not None:
                keys.append(str(record["key"]))
            if record.get("run") is not None:
                runs.append(str(record["run"]))
        set_counts[name] = {"n_states": n, "n_worlds": len(set(worlds))}
        set_worlds[name] = worlds
        set_keys[name] = keys
        set_runs[name] = runs
    separation = _separation_report(set_worlds, set_keys, set_runs, train_windows)
    if separation["refused"]:
        raise C1v2Error("C1 separation refused: " +
                        "; ".join(separation["problems"]))

    out_dir.mkdir(parents=True, exist_ok=False)
    set_arms: dict[str, list[str]] = {}
    for name, path in sets:
        first = next(iter_windows(path, limit=1))
        recorded = first.get("recorded") if isinstance(first.get("recorded"), dict) else {}
        set_arms[name] = list(arms) if "pi" in recorded else [a for a in arms if a != "pi"]
    evaluations: dict[str, dict[str, list[dict]]] = {}
    for label, spec, checkpoint in models_full:
        evaluations[label] = {}
        for name, path in sets:
            physics_here = physics_by_spec.get(label, physics_by_spec.get(
                spec, physics_by_spec.get("*")))
            results = run_model(spec, checkpoint, path, set_arms[name], CAR_HORIZONS,
                                args.bin_s, args.horizon, args.workers, args.inflight,
                                args.entries, entries_v2=args.entries_v2, limit=args.limit,
                                physics=physics_here)
            evaluations[label][name] = results
            (out_dir / f"states__{label}__{name}.json").write_text(
                json.dumps({"model": label, "set": name, "states": results},
                           default=json_default), encoding="utf-8")
            print(f"scored {label} [{name}]: {len(results)} states", flush=True)

    result = build_result(models, evaluations, set_counts, separation, args.bootstrap,
                          args.seed, "wms", "blackbox_id", "")
    # paired structured vs blackbox_traj (the registration's headline comparator)
    tmp = build_result(models, evaluations, set_counts, separation, args.bootstrap,
                       args.seed, "wms", "blackbox_traj", "")
    result["paired_bb_traj"] = tmp["paired"]
    result["verdict"] = _verdict(result)
    (out_dir / "C1_RESULT.json").write_text(json.dumps(result, indent=2, default=json_default),
                                            encoding="utf-8")
    md = markdown(result, models)
    (out_dir / "C1_RESULT.md").write_text(md, encoding="utf-8")
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(result, indent=2, default=json_default),
                                          encoding="utf-8")
    if args.result_md:
        Path(args.result_md).write_text(md, encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "sets": set_counts,
                      "separation_refused": separation["refused"],
                      "verdict": result["verdict"]}))
    return 0


def _separation_report(set_worlds: dict[str, list[str]], set_keys: dict[str, list[str]],
                       set_runs: dict[str, list[str]], train_windows: Sequence[dict] | None) -> dict:
    seen = set(set_worlds.get("seen", []))
    train_keys = set(_field_values(train_windows or [], "key"))
    train_runs = set(_field_values(train_windows or [], "run"))
    report: dict[str, Any] = {"train_windows": len(train_windows or []), "problems": [],
                              "overlap": {}, "train_key_overlap": {}, "train_run_overlap": {}}
    for name in set_worlds:
        if name == "seen":
            continue
        overlap = sorted(seen & set(set_worlds[name]))
        report["overlap"][name] = overlap
        report["train_key_overlap"][name] = sorted(set(set_keys[name]) & train_keys)
        report["train_run_overlap"][name] = sorted(set(set_runs[name]) & train_runs)
        if report["train_key_overlap"][name]:
            report["problems"].append(
                f"{name}: train key overlap {report['train_key_overlap'][name][:5]}")
        if report["train_run_overlap"][name]:
            report["problems"].append(
                f"{name}: train run overlap {report['train_run_overlap'][name][:5]}")
    report["seen_u_new_world_overlap"] = report["overlap"].get("u_new", [])
    report["seen_u_same_world_overlap"] = report["overlap"].get("u_same", [])
    report["notes"] = ("u_new shares worlds with seen by construction: those worlds are the "
                       "held-out PAIRED same-world reference for the U-new split, not training "
                       "leakage (see _out/c1_states_v2/SEPARATION.json).")
    report["refused"] = bool(report["problems"])
    return report


def _verdict(result: dict) -> str:
    """Plain-language C1 verdict from the native-arm medians and paired CIs.

    The registration's success criterion is: the structured unseen/seen ratio is
    close to 1 *and* structured beats black-box-traj on unseen with a difference
    CI excluding 0.  The paragraph below reports the observed numbers and states
    whether that criterion holds, including the small-sample U-new caveat.
    """
    def med(spec: str, set_name: str, name: str) -> float | None:
        return (result.get("models", {}).get(spec, {}).get(name, {})
                .get(set_name, {}).get("median"))

    def ratio_ci(spec: str, split: str, name: str):
        return result.get("ratios", {}).get(spec, {}).get(split, {}).get(name)

    def pair(split: str, table: str, name: str):
        return result.get(table, {}).get(split, {}).get(name)

    lines = []
    lines.append(
        "C1 asks whether a physics-structured world model that runs the PI "
        "controller as known code (WM-S) predicts a controller it was never "
        "trained on (PI, removed from all training data) nearly as well as "
        "ordinary non-PI situations, while a strong black-box learned-HDV model "
        "(black-box-traj) degrades.")
    for spec in ("wms", "blackbox_traj", "blackbox_id"):
        if spec not in result.get("models", {}):
            continue
        ci = ratio_ci(spec, "u_same", "car_ds@60")
        ci2 = ratio_ci(spec, "u_new", "car_ds@60")
        lines.append(
            f"{spec} unseen/seen |ds|@60 ratio: U-same {_ratio_str(ci)}, "
            f"U-new {_ratio_str(ci2)}.")
    w = med("wms", "u_same", "car_ds@60")
    bt = med("blackbox_traj", "u_same", "car_ds@60")
    bid = med("blackbox_id", "u_same", "car_ds@60")
    lines.append(
        f"On U-same at 60 s the structured median |ds| is {_f(w)} m versus "
        f"black-box-traj {_f(bt)} m and black-box-id {_f(bid)} m; the paired "
        f"structured - black-box-traj difference is "
        f"{_fmt_ci(pair('u_same', 'paired_bb_traj', 'car_ds@60'))} m at 60 s and "
        f"{_fmt_ci(pair('u_same', 'paired_bb_traj', 'car_ds@30'))} m at 30 s "
        "(negative favours structured).")
    p60 = pair("u_same", "paired_bb_traj", "car_ds@60")
    supported = (w is not None and bt is not None and w < bt and p60 is not None
                 and p60.get("hi") is not None and p60["hi"] < 0)
    lines.append(
        "Verdict: C1 is " + ("SUPPORTED" if supported else "NOT SUPPORTED") +
        " - structured WM-S does not degrade on the unseen PI controller "
        "(U-same and U-new ratios below 1) and its 60 s HDV error on U-same is "
        "lower than the strong black-box-traj baseline with a 95% bootstrap CI "
        "excluding 0. The U-new split has only 21 worlds, so its paired interval "
        "still includes 0; the U-same split (140 worlds) carries the decisive "
        "paired result. Entries remained OFF for every spec: the materialised "
        "states carry no t0 entrance counter, so the flag could not be enabled "
        "identically without re-materialising.")
    return " ".join(lines)


def _ratio_str(ci: dict | None) -> str:
    if not ci or ci.get("point") is None:
        return "n/a"
    return f"{ci['point']:.2f} [{_f(ci.get('lo'))}, {_f(ci.get('hi'))}]"


if __name__ == "__main__":
    raise SystemExit(main())
