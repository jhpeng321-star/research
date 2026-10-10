"""C1: zero-shot, leave-one-controller-out evaluation of unseen controllers.

The C1 question is whether a *structured* world model -- the controller run as
known code and only the human-driver motion learned (WM-S) -- generalises to a
controller it never saw better than a *black-box* model that consumes the
controller as an input feature.  A controller that is only seen in a *held-out*
state set (``--seen``) but never trained on is the reference scale; the decisive
comparison is on the *unseen* controller set (``--unseen``).

For every (model, state-set, state) the harness records:

* car-level HDV ``|ds|`` / ``|dv|`` at 10/30/60/120 s (identity matched);
* traffic-level, per section, per 30 s bin: mean speed, stopped vehicles, merge
  queue length and normalized delay seconds;
* the paired controller difference against SUMO truth for every arm pair that
  has both a predicted and a recorded trajectory.

The headline table is the unseen/seen error ratio per model with bootstrap
confidence intervals resampled **over worlds** (whole worlds, so no state
leakage between the resampled sets), and the paired structured-vs-black-box
difference on the unseen set.

Memory discipline (owner rule): every worker is a separate process, opens one
state at a time, keeps no pack-wide array in memory, and runs under a hard
worker cap of four.  ``.npy`` packs are never loaded whole -- only the state
material referenced by a single record is touched.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable, Mapping, Sequence

from .data import DataError, json_default, recorded_for
from .metrics import car_metrics, paired_difference, traffic_metrics
from .models import MODEL_SPECS, ModelUnavailable, load_model

CAR_HORIZONS = (10.0, 30.0, 60.0, 120.0)
TRAFFIC_SECTIONS = ("upstream", "merge", "downstream")
TRAFFIC_METRICS = ("mean_speed", "stopped_vehicles", "merge_queue_length", "total_delay")
MAX_WORKERS = 4
DEFAULT_METRIC = "hdv_ds@60"


class C1Error(ValueError):
    """The C1 inputs, model list or metric selection are invalid."""


# --------------------------------------------------------------------------- #
# input loading
# --------------------------------------------------------------------------- #
def _load_raw(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if isinstance(raw, dict):
        for key in ("windows", "states", "records", "list"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list) or not raw:
        raise DataError("state/window file must contain a non-empty JSON list")
    if not all(isinstance(item, dict) for item in raw):
        raise DataError("every state/window entry must be an object")
    return raw


def _has_material(record: dict) -> bool:
    if isinstance(record.get("state"), dict) or isinstance(record.get("initial_state"), dict):
        return True
    if isinstance(record.get("state_row"), dict):
        return True
    return False


def resolve_states(records: Sequence[dict], state_index: Mapping[str, dict] | None = None) -> list[dict]:
    """Return materialised state records, resolving window descriptors via index.

    A *window descriptor* (only ``key``/``run``/``world``/``t0_s``/``arm``) names
    a run slice that must be materialised into an initial frame plus recorded
    arm trajectories.  Materialisation reads at most one window of pack data at
    a time and is supplied by ``--state-index`` (``key -> material``); without it
    the harness fails closed rather than guessing.
    """
    out: list[dict] = []
    for record in records:
        if _has_material(record):
            out.append(dict(record))
            continue
        key = str(record.get("key", ""))
        material = None
        if state_index is not None and key:
            material = state_index.get(key)
        if material is None:
            raise DataError(
                f"window {key or record.get('run')!r} has no state material; "
                "materialise it first (wms_eval.seen) and pass --state-index"
            )
        merged = {**material, **{k: v for k, v in record.items() if v is not None}}
        out.append(merged)
    return out


# --------------------------------------------------------------------------- #
# seen/unseen and training separation (fail closed)
# --------------------------------------------------------------------------- #
def _field_values(records: Iterable[dict], field: str) -> set[str]:
    values = set()
    for record in records:
        value = record.get(field)
        if value is not None and str(value):
            values.add(str(value))
    return values


def training_index(windows: Sequence[dict]) -> dict[str, set[str]]:
    """Collect the window ``key``/``run`` identities named by a training set."""
    return {"keys": _field_values(windows, "key"), "runs": _field_values(windows, "run")}


def check_separation(sets: Mapping[str, Sequence[dict]],
                     train_windows: Sequence[dict] | None = None) -> dict:
    """Refuse C1 inputs that leak between seen, unseen and training sets.

    The unseen/seen error ratio and its bootstrap interval resample whole
    worlds, so the two state sets must not share a world id; and no unseen
    state may reuse a training window's ``key`` or ``run``.  Any overlap raises
    :class:`C1Error` *before* evaluation or output creation.  The observed
    counts are always returned so a zero-overlap run can be reported.
    """
    unseen = list(sets.get("unseen", []))
    seen = list(sets.get("seen", []))
    unseen_worlds = _field_values(unseen, "world")
    seen_worlds = _field_values(seen, "world")
    world_overlap = sorted(unseen_worlds & seen_worlds)
    report: dict[str, Any] = {
        "unseen_states": len(unseen),
        "seen_states": len(seen),
        "unseen_worlds": len(unseen_worlds),
        "seen_worlds": len(seen_worlds),
        "world_overlap": world_overlap,
        "train_windows": 0,
        "key_overlap": [],
        "run_overlap": [],
    }
    problems: list[str] = []
    if world_overlap:
        problems.append(f"seen/unseen world overlap={len(world_overlap)} "
                        f"{world_overlap[:5]}")
    if train_windows is not None:
        index = training_index(train_windows)
        report["train_windows"] = len(train_windows)
        report["key_overlap"] = sorted(_field_values(unseen, "key") & index["keys"])
        report["run_overlap"] = sorted(_field_values(unseen, "run") & index["runs"])
        if report["key_overlap"]:
            problems.append(f"unseen/train key overlap={len(report['key_overlap'])} "
                            f"{report['key_overlap'][:5]}")
        if report["run_overlap"]:
            problems.append(f"unseen/train run overlap={len(report['run_overlap'])} "
                            f"{report['run_overlap'][:5]}")
    report["refused"] = bool(problems)
    if problems:
        raise C1Error("C1 separation refused: " + "; ".join(problems) +
                      f"; counts={json.dumps({k: report[k] for k in ('world_overlap', 'key_overlap', 'run_overlap')})}")
    return report


# --------------------------------------------------------------------------- #
# per-state evaluation
# --------------------------------------------------------------------------- #
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
                "lane": group.get("lane_agreement"),
                "match": group.get("matched_fraction"),
            }
        out[str(horizon)] = block
    return out


def _traffic_errors(predicted: dict, recorded: dict) -> dict:
    """|predicted - recorded| per section/metric over aligned 30 s bins."""
    out: dict[str, dict] = {}
    for section in TRAFFIC_SECTIONS:
        out[section] = {}
        for metric in TRAFFIC_METRICS:
            bins = []
            for key in sorted(predicted, key=lambda k: int(k)):
                pred = predicted[key].get("sections", {}).get(section, {}).get(metric)
                rec = recorded.get(key, {}).get("sections", {}).get(section, {}).get(metric)
                if pred is None or rec is None:
                    continue
                bins.append(abs(float(pred) - float(rec)))
            out[section][metric] = {
                "bins": bins,
                "mean": (mean(bins) if bins else None),
            }
    return out


def _paired_errors(predicted: Mapping[str, dict], recorded: Mapping[str, dict],
                   arms: Sequence[str], baseline: str) -> dict:
    out: dict[str, dict] = {}
    for arm in arms:
        if arm == baseline:
            continue
        pred_diff = paired_difference(predicted[arm], predicted[baseline])
        rec_diff = paired_difference(recorded[arm], recorded[baseline])
        block: dict[str, dict] = {}
        for section in TRAFFIC_SECTIONS:
            block[section] = {}
            for metric in TRAFFIC_METRICS:
                signed, absolute = [], []
                for key in sorted(pred_diff, key=lambda k: int(k)):
                    pred_value = pred_diff[key]["sections"][section].get(metric)
                    rec_value = rec_diff.get(key, {}).get("sections", {}) \
                        .get(section, {}).get(metric)
                    if pred_value is None or rec_value is None:
                        continue
                    value = float(pred_value) - float(rec_value)
                    signed.append(value)
                    absolute.append(abs(value))
                block[section][metric] = {
                    "pred_diff": _mean_of(pred_diff, section, metric),
                    "rec_diff": _mean_of(rec_diff, section, metric),
                    "err": (mean(signed) if signed else None),
                    "abs_err": (mean(absolute) if absolute else None),
                }
        out[str(arm)] = block
    return out


def _mean_of(paired: Mapping[str, dict], section: str, metric: str) -> float | None:
    values = []
    for key in sorted(paired, key=lambda k: int(k)):
        value = paired[key].get("sections", {}).get(section, {}).get(metric)
        if value is not None:
            values.append(float(value))
    return mean(values) if values else None


def evaluate_state(record: dict, predictor: Callable, arms: Sequence[str], baseline: str,
                   horizons: Sequence[float], bin_s: float, default_horizon: float) -> dict:
    """Roll out every arm for one state and reduce it to compact, JSON-safe data."""
    state = dict(record)
    t0_s = float(record.get("t0_s", state.get("time_s", 0.0)))
    horizon_s = float(record.get("horizon_s", default_horizon))
    seed = int(record.get("seed", 0))
    free_speed = float(record.get("v0_mps", 13.89))
    controllers = record.get("cav_controllers", {})

    predicted: dict[str, list] = {}
    arm_result: dict[str, dict] = {}
    for arm in arms:
        try:
            predicted[arm] = predictor(state, dict(controllers, arm=arm), horizon_s, seed)
        except Exception as exc:  # one arm must not abort the other arms/states
            arm_result[arm] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        try:
            recorded = recorded_for(record, arm)
        except DataError as exc:
            arm_result[arm] = {"error": str(exc)}
            continue
        arm_result[arm] = {
            "error": None,
            "car": _car_compact(car_metrics(predicted[arm], recorded, t0_s=t0_s,
                                            horizons=tuple(horizons))),
            "traffic": _traffic_errors(
                traffic_metrics(predicted[arm], t0_s=t0_s, bin_s=bin_s, free_speed_mps=free_speed),
                traffic_metrics(recorded, t0_s=t0_s, bin_s=bin_s, free_speed_mps=free_speed)),
        }

    paired_arms = [arm for arm in arms if arm in predicted]
    recorded_arms: dict[str, list] = {}
    for arm in paired_arms:
        try:
            recorded_arms[arm] = recorded_for(record, arm)
        except DataError:
            continue
    paired = {}
    if baseline in predicted and baseline in recorded_arms and len(recorded_arms) > 1:
        pred_traffic = {arm: traffic_metrics(predicted[arm], t0_s=t0_s, bin_s=bin_s,
                                             free_speed_mps=free_speed) for arm in recorded_arms}
        rec_traffic = {arm: traffic_metrics(recorded_arms[arm], t0_s=t0_s, bin_s=bin_s,
                                            free_speed_mps=free_speed) for arm in recorded_arms}
        paired = _paired_errors(pred_traffic, rec_traffic, list(recorded_arms), baseline)

    return {"world": record.get("world", "unknown"), "t0_s": t0_s,
            "arms": arm_result, "paired": paired}


# --------------------------------------------------------------------------- #
# headline scalar extraction
# --------------------------------------------------------------------------- #
def headline_metric(state_result: dict, metric: str, baseline: str) -> float | None:
    """Reduce one state result to the scalar named by ``metric``.

    Grammar: ``hdv_ds@H`` / ``hdv_dv@H`` (HDV median car error at horizon
    ``H`` s) or ``traffic_<name>@<section>`` (baseline-arm mean |pred - rec|
    over 30 s bins).
    """
    name, _, arg = metric.partition("@")
    block = state_result.get("arms", {}).get(baseline, {})
    if block.get("error"):
        return None
    if name in ("hdv_ds", "hdv_dv"):
        if not arg:
            raise C1Error(f"metric {metric!r} needs a horizon, e.g. hdv_ds@30")
        car = block.get("car", {}).get(str(float(arg)), {}).get("hdv", {})
        return car.get("ds_med") if name == "hdv_ds" else car.get("dv_med")
    if name.startswith("traffic_"):
        if not arg:
            raise C1Error(f"metric {metric!r} needs a section, e.g. traffic_total_delay@merge")
        key = name[len("traffic_"):]
        value = block.get("traffic", {}).get(arg, {}).get(key)
        return value.get("mean") if isinstance(value, dict) else None
    raise C1Error(f"unknown metric {metric!r}")


# --------------------------------------------------------------------------- #
# bootstrap statistics over worlds
# --------------------------------------------------------------------------- #
def _stat(values: Sequence[float]) -> float:
    return float(median(values))


def _group_by_world(results: Iterable[dict], metric: str, baseline: str) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {}
    for result in results:
        value = headline_metric(result, metric, baseline)
        if value is None or not math.isfinite(value):
            continue
        grouped.setdefault(str(result.get("world", "unknown")), []).append(float(value))
    return grouped


def bootstrap_over_worlds(grouped: Mapping[str, Sequence[float]], *, n_boot: int = 1000,
                          seed: int = 0, stat: Callable = _stat) -> dict:
    worlds = sorted(grouped)
    flat = [value for world in worlds for value in grouped[world]]
    point = stat(flat) if flat else None
    if not flat or n_boot <= 0:
        return {"point": point, "lo": None, "hi": None, "n_boot": 0, "n_worlds": len(worlds)}
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        pick = rng.choices(worlds, k=len(worlds))
        values = [value for world in pick for value in grouped[world]]
        if values:
            samples.append(stat(values))
    samples.sort()
    return {"point": point, "lo": _quantile(samples, 0.025), "hi": _quantile(samples, 0.975),
            "n_boot": len(samples), "n_worlds": len(worlds)}


def ratio_ci(numerator: Mapping[str, Sequence[float]], denominator: Mapping[str, Sequence[float]],
             *, n_boot: int = 1000, seed: int = 0, stat: Callable = _stat) -> dict:
    """Bootstrap CI of ``stat(numerator) / stat(denominator)`` (sets resampled apart)."""
    num_worlds, den_worlds = sorted(numerator), sorted(denominator)
    num_flat = [v for w in num_worlds for v in numerator[w]]
    den_flat = [v for w in den_worlds for v in denominator[w]]
    point = (stat(num_flat) / stat(den_flat)) if num_flat and den_flat and stat(den_flat) else None
    if point is None or n_boot <= 0:
        return {"point": point, "lo": None, "hi": None, "n_boot": 0,
                "n_worlds": (len(num_worlds), len(den_worlds))}
    rng = random.Random(seed)
    samples = []
    for _ in range(n_boot):
        num = [v for w in rng.choices(num_worlds, k=len(num_worlds)) for v in numerator[w]]
        den = [v for w in rng.choices(den_worlds, k=len(den_worlds)) for v in denominator[w]]
        d = stat(den) if den else 0.0
        if num and d:
            samples.append(stat(num) / d)
    samples.sort()
    return {"point": point, "lo": _quantile(samples, 0.025), "hi": _quantile(samples, 0.975),
            "n_boot": len(samples), "n_worlds": (len(num_worlds), len(den_worlds))}


def paired_ci(a: Mapping[str, Sequence[float]], b: Mapping[str, Sequence[float]], *,
              n_boot: int = 1000, seed: int = 0, stat: Callable = _stat) -> dict:
    """Bootstrap CI of ``stat(a) - stat(b)`` with per-world state pairing."""
    worlds = sorted(set(a) & set(b))
    diffs = [x - y for world in worlds for x, y in zip(a[world], b[world])]
    point = stat(diffs) if diffs else None
    if not diffs or n_boot <= 0:
        return {"point": point, "lo": None, "hi": None, "n_boot": 0, "n_worlds": len(worlds)}
    grouped = {world: [x - y for x, y in zip(a[world], b[world])] for world in worlds}
    return bootstrap_over_worlds(grouped, n_boot=n_boot, seed=seed, stat=stat)


def _quantile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    pos = (len(sorted_values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


# --------------------------------------------------------------------------- #
# parallel driver
# --------------------------------------------------------------------------- #
_WORKER: dict[str, Any] = {}


def _init_worker(spec: str, checkpoint: str | None, arms: Sequence[str], baseline: str,
                 horizons: Sequence[float], bin_s: float, default_horizon: float) -> None:
    predictor, _flags = load_model(spec, checkpoint=checkpoint)
    _WORKER.update({"predictor": predictor, "arms": tuple(arms), "baseline": baseline,
                    "horizons": tuple(horizons), "bin_s": bin_s,
                    "default_horizon": default_horizon})


def _work(record: dict) -> dict:
    return evaluate_state(record, _WORKER["predictor"], _WORKER["arms"], _WORKER["baseline"],
                          _WORKER["horizons"], _WORKER["bin_s"], _WORKER["default_horizon"])


def run_model(spec: str, checkpoint: str | None, records: Sequence[dict], arms: Sequence[str],
              baseline: str, horizons: Sequence[float], bin_s: float, default_horizon: float,
              workers: int) -> list[dict]:
    workers = max(1, min(MAX_WORKERS, int(workers)))
    if workers == 1:
        _init_worker(spec, checkpoint, arms, baseline, horizons, bin_s, default_horizon)
        return [_work(record) for record in records]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(spec, checkpoint, tuple(arms), baseline, tuple(horizons), bin_s, default_horizon),
    ) as pool:
        return list(pool.map(_work, records))


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def parse_models(entries: Sequence[str]) -> list[tuple[str, str | None]]:
    out = []
    for entry in entries:
        spec, sep, checkpoint = entry.partition("=")
        spec = spec.strip()
        if spec not in MODEL_SPECS:
            raise C1Error(f"unknown model {spec!r}; choose from {', '.join(MODEL_SPECS)}")
        out.append((spec, checkpoint if sep and checkpoint else None))
    if not out:
        raise C1Error("at least one --models SPEC=CKPT is required")
    return out


def _median_series(results: Sequence[dict], baseline: str, metric: str) -> float | None:
    values = [headline_metric(r, metric, baseline) for r in results]
    values = [v for v in values if v is not None and math.isfinite(v)]
    return median(values) if values else None


def build_report(records_by_set: Mapping[str, Sequence[dict]],
                 evaluations: Mapping[str, Mapping[str, Sequence[dict]]],
                 models: Sequence[tuple[str, str | None]], baseline: str, metric: str,
                 n_boot: int, seed: int, paired_a: str, paired_b: str) -> dict:
    report: dict[str, Any] = {
        "schema": "wms_eval_c1/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "headline_metric": metric,
        "baseline_arm": baseline,
        "sets": {name: {"n_states": len(results),
                        "n_worlds": len({r.get("world") for r in results})}
                 for name, results in records_by_set.items()},
        "models": {},
    }
    grouped: dict[tuple[str, str], dict[str, list[float]]] = {}
    for spec, _ckpt in models:
        entry: dict[str, Any] = {}
        for set_name, results in evaluations[spec].items():
            metric_series = {
                "hdv_ds": {str(h): _median_series(results, baseline, f"hdv_ds@{h:g}") for h in CAR_HORIZONS},
                "hdv_dv": {str(h): _median_series(results, baseline, f"hdv_dv@{h:g}") for h in CAR_HORIZONS},
                "traffic": {section: {m: _median_series(results, baseline, f"traffic_{m}@{section}")
                                      for m in TRAFFIC_METRICS}
                            for section in TRAFFIC_SECTIONS},
            }
            entry[set_name] = metric_series
            grouped[(spec, set_name)] = _group_by_world(results, metric, baseline)
        report["models"][spec] = entry

    ratios = {}
    for spec, _ckpt in models:
        if (spec, "unseen") in grouped and (spec, "seen") in grouped \
                and grouped[(spec, "unseen")] and grouped[(spec, "seen")]:
            ratios[spec] = ratio_ci(grouped[(spec, "unseen")], grouped[(spec, "seen")],
                                    n_boot=n_boot, seed=seed)
    report["unseen_seen_ratio"] = ratios

    paired = None
    if (paired_a, "unseen") in grouped and (paired_b, "unseen") in grouped \
            and grouped[(paired_a, "unseen")] and grouped[(paired_b, "unseen")]:
        paired = {"a": paired_a, "b": paired_b,
                  **paired_ci(grouped[(paired_a, "unseen")], grouped[(paired_b, "unseen")],
                              n_boot=n_boot, seed=seed)}
    report["paired_structured_vs_blackbox_unseen"] = paired
    return report


def markdown(report: dict) -> str:
    metric = report["headline_metric"]
    lines = ["# C1 zero-shot controller evaluation",
             "",
             f"Headline metric: `{metric}` (median across states; CI over worlds). "
             f"Baseline arm: `{report['baseline_arm']}`.",
             ""]
    for name, info in report["sets"].items():
        lines.append(f"- `{name}`: {info['n_states']} states / {info['n_worlds']} worlds")
    lines += ["", "## Unseen/seen error ratio (CI resampled over worlds)", ""]
    for spec, ci in report["unseen_seen_ratio"].items():
        lines.append(f"- {spec}: ratio {_fmt(ci['point'])} "
                     f"[{_fmt(ci['lo'])}, {_fmt(ci['hi'])}] (n_boot={ci['n_boot']})")
    paired = report.get("paired_structured_vs_blackbox_unseen")
    lines += ["", "## Paired structured vs black-box (unseen set)", ""]
    if paired:
        lines.append(f"- {paired['a']} - {paired['b']}: {_fmt(paired['point'])} "
                     f"[{_fmt(paired['lo'])}, {_fmt(paired['hi'])}] "
                     f"(negative favours {paired['a']})")
    else:
        lines.append("- not available (both models must have unseen results)")
    lines += ["", "## Per-model accuracy (median across states)", ""]
    for spec, sets in report["models"].items():
        for set_name, series in sets.items():
            ds = " ".join(f"{h}:{_fmt(series['hdv_ds'][h])}" for h in series["hdv_ds"])
            dv = " ".join(f"{h}:{_fmt(series['hdv_dv'][h])}" for h in series["hdv_dv"])
            lines.append(f"- {spec} [{set_name}] HDV |ds|@10/30/60/120 {ds}; |dv| {dv}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="wms_eval c1")
    p.add_argument("--models", action="append", required=True,
                   help="SPEC=CKPT (repeatable); SPEC in MODEL_SPECS")
    p.add_argument("--unseen", required=True, help="unseen-controller state/window set")
    p.add_argument("--seen", help="seen held-out state/window set")
    p.add_argument("--train-windows",
                   help="training-window identity list; unseen key/run must not appear in it")
    p.add_argument("--state-index", help="JSON key->material for window descriptors")
    p.add_argument("--out", required=True)
    p.add_argument("--arms", default="native,pi")
    p.add_argument("--baseline", default="native")
    p.add_argument("--metric", default=DEFAULT_METRIC)
    p.add_argument("--horizon", type=float, default=300.0)
    p.add_argument("--bin-s", type=float, default=30.0)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--paired-a", default="wms")
    p.add_argument("--paired-b", default="blackbox_id")
    p.add_argument("--horizons", default=",".join(str(h) for h in CAR_HORIZONS))
    return p


def _load_index(path: str | None) -> dict[str, dict] | None:
    if not path:
        return None
    with Path(path).open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise C1Error("--state-index must be a JSON object keyed by window key")
    return data


def run(args) -> int:
    models = parse_models(args.models)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    horizons = tuple(float(h) for h in args.horizons.split(",") if h.strip())
    index = _load_index(args.state_index)
    sets: dict[str, list[dict]] = {}
    for name, path in (("unseen", args.unseen), ("seen", args.seen)):
        if not path:
            continue
        sets[name] = resolve_states(_load_raw(Path(path)), index)
    if "unseen" not in sets:
        raise C1Error("--unseen is required")

    train_windows = _load_raw(Path(args.train_windows)) if args.train_windows else None
    separation = check_separation(sets, train_windows)

    out_dir = Path(args.out)
    if out_dir.exists():
        raise C1Error(f"refusing to overwrite existing output: {out_dir}")
    out_dir.mkdir(parents=True)

    evaluations: dict[str, dict[str, list[dict]]] = {}
    details = []
    for spec, checkpoint in models:
        evaluations[spec] = {}
        for set_name, records in sets.items():
            results = run_model(spec, checkpoint, records, arms, args.baseline, horizons,
                                args.bin_s, args.horizon, args.workers)
            evaluations[spec][set_name] = results
            details.append({"model": spec, "set": set_name, "states": results})

    report = build_report(sets, evaluations, models, args.baseline, args.metric, args.bootstrap,
                          args.seed, args.paired_a, args.paired_b)
    report["separation"] = separation
    (out_dir / "C1.json").write_text(json.dumps(report, indent=2, default=json_default),
                                     encoding="utf-8")
    (out_dir / "RESULTS.json").write_text(json.dumps({"schema": "wms_eval_c1_results/1",
                                                      "runs": details}, default=json_default),
                                          encoding="utf-8")
    (out_dir / "C1.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"out": str(out_dir), "models": [s for s, _ in models],
                      "sets": {k: len(v) for k, v in sets.items()},
                      "separation": separation}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return run(args)
    except (DataError, ModelUnavailable, C1Error, ValueError, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 2


__all__ = [
    "CAR_HORIZONS", "C1Error", "DEFAULT_METRIC", "MAX_WORKERS",
    "bootstrap_over_worlds", "build_report", "check_separation", "evaluate_state",
    "headline_metric", "markdown", "paired_ci", "parse_models", "ratio_ci",
    "resolve_states", "run_model", "training_index",
]
