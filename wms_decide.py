"""TRAIN-only decision scoring adapter for the WM next-state interface.

The model boundary is deliberately small: ``predict_states(state, plan,
horizon_s, seed)``.  A release is a JSON document containing TRAIN decision
windows and labels.  This module never starts SUMO and refuses sealed splits.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


TAUS_S = (30.0, 60.0, 120.0, 300.0)
E2_ARMS = ("native", "pi", "fixed_mix", "switching_mix", "wall", "arm0", "arm1", "arm2", "arm3")
_ARM_RANK = {arm: index for index, arm in enumerate(E2_ARMS)}
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z", re.IGNORECASE)


class DecisionRefusal(RuntimeError):
    """Fail-closed refusal of an incomplete or non-TRAIN decision input."""


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def total_delay(states: Any, *, t0_s: float, tau_s: float) -> float:
    """Integrate per-vehicle normalized delay (seconds) over a state window.

    State mappings contain ``time_s``, ``speed_mps`` and ``free_speed_mps``
    arrays, with shape ``[T,V]``; ``mask`` is optional and has the same shape.
    The registered WM-N1 delay metric is
    ``sum(dt * clip(1 - v/v_free, 0, 1))`` over vehicles and steps
    (output/wd01/wm_n1/evalh/scorer/comparator_v2g.py:289-290,
    ``_window_delay``); this adapter reuses that definition, so the result
    carries second units and equals one second per vehicle-step spent at a
    standstill.
    """
    if not isinstance(states, Mapping):
        raise DecisionRefusal("predict_states must return a state mapping")
    try:
        times = np.asarray(states["time_s"], dtype=float)
        speed = np.asarray(states["speed_mps"], dtype=float)
        free_value = states.get("free_speed_mps")
        if free_value is None:
            free_value = states.get("free_speed")
        free = np.asarray(free_value, dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise DecisionRefusal("state mapping lacks numeric time/speed/free-speed arrays") from exc
    mask = np.asarray(states.get("mask", np.ones_like(speed, dtype=bool)), dtype=bool)
    if times.ndim != 1 or speed.ndim != 2 or free.shape != speed.shape or mask.shape != speed.shape:
        raise DecisionRefusal("state arrays must be aligned as [T] and [T,V]")
    if len(times) < 2 or not (np.isfinite(times).all() and np.isfinite(speed).all()
                              and np.isfinite(free).all()):
        raise DecisionRefusal("state arrays contain insufficient/non-finite values")
    if (free <= 0.0).any():
        raise DecisionRefusal("free-flow speed must be positive for normalized delay")
    lo, hi = float(t0_s), float(t0_s + tau_s)
    if times[0] > lo + 1e-9 or times[-1] < hi - 1e-9:
        raise DecisionRefusal("prediction is shorter than the requested delay window")
    keep = (times >= lo - 1e-9) & (times <= hi + 1e-9)
    if keep.sum() < 2:
        raise DecisionRefusal("delay window has fewer than two samples")
    grid = np.diff(times[keep])
    if not np.allclose(grid, grid[0], rtol=0.0, atol=1e-8):
        raise DecisionRefusal("prediction time grid is not regular")
    deficit = np.clip(1.0 - speed[keep][:-1] / free[keep][:-1], 0.0, 1.0)
    valid = mask[keep][:-1] & mask[keep][1:]
    return float(np.sum(np.where(valid, deficit, 0.0) * grid[:, None]))


def frames_to_arrays(frames: Any, *, free_speed_mps: float) -> dict[str, np.ndarray]:
    """Convert ``wms_eval`` frame predictions to the delay-aligned state arrays.

    A ``wms_eval`` predictor returns ``[{"time_s": ..., "vehicles": [...]}]``.
    The delay metric is defined on ``[T, V]`` arrays, so this adapter fixes one
    column per observed vehicle id and marks presence with a mask.
    """
    if not isinstance(frames, (list, tuple)) or len(frames) < 2:
        raise DecisionRefusal("wms_eval predictor must return at least two frames")
    ids: list[str] = []
    seen: set[str] = set()
    for frame in frames:
        if not isinstance(frame, Mapping):
            raise DecisionRefusal("wms_eval frame is not an object")
        for row in frame.get("vehicles", ()):
            vid = str(row.get("id"))
            if vid not in seen:
                seen.add(vid)
                ids.append(vid)
    if not ids:
        raise DecisionRefusal("wms_eval frames contain no vehicles")
    if not _finite(free_speed_mps) or float(free_speed_mps) <= 0.0:
        raise DecisionRefusal("free-flow speed must be positive for normalized delay")
    index = {vid: position for position, vid in enumerate(ids)}
    times = np.asarray([float(frame.get("time_s")) for frame in frames], dtype=float)
    speed = np.zeros((len(frames), len(ids)), dtype=float)
    mask = np.zeros_like(speed, dtype=bool)
    for ti, frame in enumerate(frames):
        for row in frame.get("vehicles", ()):
            if not bool(row.get("present", True)):
                continue
            speed[ti, index[str(row.get("id"))]] = float(row.get("v", 0.0))
            mask[ti, index[str(row.get("id"))]] = True
    free = np.full_like(speed, float(free_speed_mps))
    return {"time_s": times, "speed_mps": speed, "free_speed_mps": free, "mask": mask}


class WmsEvalPredictor:
    """Adapt a ``wms_eval`` predictor to the ``predict_states`` delay boundary."""

    def __init__(self, predictor: Callable):
        if not callable(predictor):
            raise DecisionRefusal("wms_eval predictor is not callable")
        self.predictor = predictor

    def predict_states(self, state: Any, plan: Any, horizon_s: float, seed: int) -> dict:
        controllers = ({key: value for key, value in state.get("cav_controllers", {}).items()}
                       if isinstance(state, Mapping) else {})
        controllers["arm"] = str(plan)
        frames = self.predictor(state, controllers, float(horizon_s), int(seed))
        free_speed = state.get("v0_mps", 27.78) if isinstance(state, Mapping) else 27.78
        return frames_to_arrays(frames, free_speed_mps=free_speed)


def wms_eval_root() -> Path:
    return Path(__file__).resolve().parent / "output" / "wd01" / "wm_n1"


def load_wms_eval_model(spec: str, plugin: str | None = None,
                        checkpoint: str | None = None) -> WmsEvalPredictor:
    """Load a ``wms_eval.MODEL_SPECS`` predictor (keep_speed/pure_physics/a11_*/wms)."""
    root = wms_eval_root()
    if not (root / "wms_eval" / "__init__.py").is_file():  # legacy shim -> archive/wms_era
        raise DecisionRefusal("wms_eval harness is absent from this workspace")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from wms_eval.models import load_model
    predictor, _flags = load_model(spec, plugin, checkpoint)
    return WmsEvalPredictor(predictor)


def recorded_delay_seconds(path: str | Path, *, t0_s: float, taus_s: Sequence[float],
                           free_speed_mps: float, dt_s: float = 0.5) -> dict[str, float]:
    """Stream one SUMO ``physics_steps.jsonl`` and integrate normalized delay.

    Each JSON line is consumed once; only scalar per-horizon accumulators are
    retained, so memory stays bounded regardless of file size. ``live_ids`` is
    the presence set emitted by the recorder. The window is ``[t0, t0 + tau)``.
    """
    if not _finite(free_speed_mps) or float(free_speed_mps) <= 0.0:
        raise DecisionRefusal("free-flow speed must be positive for normalized delay")
    taus = sorted(float(t) for t in taus_s)
    if not taus or taus[0] <= 0.0:
        raise DecisionRefusal("recorded delay requires positive horizons")
    accum = {t: 0.0 for t in taus}
    horizon = taus[-1]
    source = Path(path)
    if not source.is_file():
        raise DecisionRefusal(f"recorded trajectory is absent: {source}")
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            time_s = float(row.get("time_s"))
            if time_s < t0_s or time_s >= t0_s + horizon:
                continue
            vehicles = row.get("vehicles") or {}
            live = row.get("live_ids")
            if live is None:
                live = list(vehicles)
            deficit = 0.0
            for vid in live:
                record = vehicles.get(vid)
                if not isinstance(record, Mapping):
                    continue
                speed = float(record.get("speed_mps", record.get("v", 0.0)))
                deficit += max(0.0, 1.0 - speed / float(free_speed_mps))
            for tau in taus:
                if time_s < t0_s + tau:
                    accum[tau] += deficit * float(dt_s)
    return {str(t): accum[t] for t in taus}


def _load_symbol(spec: str) -> Callable:
    module_name, sep, symbol = spec.partition(":")
    if not sep or not module_name or not symbol:
        raise DecisionRefusal("model spec must be MODULE:FACTORY")
    factory = getattr(importlib.import_module(module_name), symbol, None)
    if not callable(factory):
        raise DecisionRefusal(f"model factory {spec!r} is not callable")
    return factory


def _load_model(spec: str):
    path = Path(spec)
    if path.is_file():
        document = json.loads(path.read_text(encoding="utf-8"))
        target = document.get("factory") or document.get("module")
        if not isinstance(target, str):
            raise DecisionRefusal("model JSON requires a factory or module string")
        return _load_symbol(target)(document)
    return _load_symbol(spec)()


def _sha256_file(path: Path) -> str:
    """Stream a file in bounded chunks; never load whole artifacts into RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _row_source_path(row: Mapping[str, Any], index: int) -> Path:
    """Resolve the source artifact a release row binds its hash to.

    The registered release format references the frozen recorder run through
    ``state.base_run_dir``; its immutable artifact is
    ``<base_run_dir>/saved_state/STATE_MANIFEST.json`` (the bytes hashed by
    ``_out/run_p4_e2e.py`` when the release was built). A direct
    ``source_file`` reference is accepted as an explicit override.
    """
    state = row.get("state")
    base = state.get("base_run_dir") if isinstance(state, Mapping) else None
    if isinstance(base, str) and base.strip():
        source = Path(base.strip()) / "saved_state" / "STATE_MANIFEST.json"
    else:
        override = row.get("source_file", row.get("source_path"))
        if not isinstance(override, str) or not override.strip():
            raise DecisionRefusal(
                f"release row {index} does not reference a source artifact "
                "(state.base_run_dir or source_file); provenance is unverifiable")
        source = Path(override.strip())
    if not source.is_absolute():
        source = Path(__file__).resolve().parent / source
    return source


def _validate_row_provenance(row: Any, index: int) -> None:
    """Refuse any row whose immutable provenance is not TRAIN or unverifiable."""
    if not isinstance(row, Mapping):
        raise DecisionRefusal(f"release row {index} is not an object")
    split = str(row.get("split", "")).strip().upper()
    if split != "TRAIN":
        raise DecisionRefusal(
            f"release row {index} carries non-TRAIN provenance "
            f"(split={split or 'missing'!r}); CAL/HOLDOUT/test rows are refused")
    source_sha = row.get("source_sha256")
    if not isinstance(source_sha, str) or not _SHA256_RE.match(source_sha.strip()):
        raise DecisionRefusal(
            f"release row {index} lacks an immutable 64-hex source_sha256")
    source = _row_source_path(row, index)
    if not source.is_file():
        raise DecisionRefusal(
            f"release row {index} references a missing source artifact: {source}")
    actual = _sha256_file(source)
    if actual != source_sha.strip().lower():
        raise DecisionRefusal(
            f"release row {index} source_sha256 does not match its source "
            f"artifact {source}: registered {source_sha.strip().lower()} "
            f"!= recomputed {actual}")


def _load_release(path: str) -> list[dict]:
    release = Path(path)
    if not release.is_file():
        raise DecisionRefusal(f"release spec is absent: {release}")
    document = json.loads(release.read_text(encoding="utf-8"))
    if str(document.get("split", "")).upper() != "TRAIN":
        raise DecisionRefusal("only a TRAIN release is permitted")
    if document.get("sealed") is True or str(document.get("half", "")).upper() in {
        "CAL", "CAL-TUNE", "HOLDOUT", "TEST_REPL"
    }:
        raise DecisionRefusal("sealed/CAL/HOLDOUT/TEST_REPL release is refused")
    rows = document.get("decisions", document.get("rows"))
    if not isinstance(rows, list) or not rows:
        raise DecisionRefusal("release must contain non-empty decisions/rows")
    for index, row in enumerate(rows):
        _validate_row_provenance(row, index)
    return rows


def _candidate_plans(row: Mapping[str, Any]) -> dict[str, Any]:
    plans = row.get("candidates", row.get("plans"))
    if not isinstance(plans, Mapping) or "native" not in plans:
        raise DecisionRefusal("decision row lacks candidate plans including native")
    return dict(plans)


def _delay_or_na(states: Any, *, t0_s: float, tau_s: float) -> float | None:
    try:
        return total_delay(states, t0_s=t0_s, tau_s=tau_s)
    except DecisionRefusal as exc:
        if "shorter than the requested" in str(exc):
            return None
        raise


def _decision_quality_result(rows: Sequence[Mapping[str, Any]], *, n_boot: int = 2000) -> dict | None:
    """Reuse the checked-in scorer when TRAIN truth is present.

    The scorer expects its registered 19-quantile interface; a deterministic
    next-state rollout is represented by nineteen identical point quantiles.
    """
    if not rows or not all(isinstance(row.get("true_delay_s"), Mapping) for row in rows):
        return None
    import sys

    scorer_root = Path(__file__).resolve().parent / "output" / "wd01" / "wm_n1"
    if not (scorer_root / "evalh" / "scorer" / "decision_quality.py").is_file():
        return None
    sys.path.insert(0, str(scorer_root))
    try:
        from evalh.scorer.decision_quality import (E2_POOL_ARMS, _e2_candidate_arm,
                                                   build_fixed_best,
                                                   decision_points_from_rows, evaluate)

        labels = []
        predictions = []
        for row in rows:
            t0 = float(row["t0_s"])
            true = row["true_delay_s"]
            tau = float(row["tau_s"])
            for arm, value in row["predicted_delay_s"].items():
                if arm not in true:
                    continue
                labels.append({"world": str(row["world"]), "t0_s": t0,
                               "decision_t_s": t0, "tau_s": tau,
                               "source": arm, "arm_kind": arm,
                               "y_delay_s": float(true[arm])})
                predictions.append(np.full(19, value))
        points, _ = decision_points_from_rows(
            labels, np.asarray(predictions), cost="delay_s", taus=TAUS_S,
            allowed_arms=("native", "pi", "fixed_mix", "switching_mix", "wall",
                          "damping", "arm0", "arm1", "arm2", "arm3"),
            normalizer=_e2_candidate_arm, pool_arms=E2_POOL_ARMS)
        if not points:
            return None
        fixed = build_fixed_best(points, costs=("delay_s",), taus=TAUS_S)
        # T7's fixed comparator is the mandatory native retention policy, not a
        # TRAIN-selected arm. Keep the scorer's CI/regret machinery unchanged.
        for cell in fixed.get("cells", {}).get("delay_s", {}).values():
            if "native" in cell.get("arm_mean", {}):
                cell["ranking"] = ["native"]
                cell["best_arm"] = "native"
        return evaluate(points, fixed_best=fixed, costs=("delay_s",), taus=TAUS_S,
                        n_boot=n_boot, seed=20261005, source="deterministic_rollout")
    finally:
        if sys.path and sys.path[0] == str(scorer_root):
            sys.path.pop(0)


def score_rows(model: Any, rows: Sequence[Mapping[str, Any]], *, seed: int) -> dict:
    """Roll out every candidate with the same seed and calculate decision rows."""
    output = []
    for row in rows:
        t0 = row.get("t0_s")
        if not _finite(t0):
            raise DecisionRefusal("decision row has non-finite t0_s")
        world = str(row.get("world", ""))
        if not world:
            raise DecisionRefusal("decision row lacks world")
        plans = _candidate_plans(row)
        taus = tuple(float(t) for t in row.get("taus_s", TAUS_S))
        if any(t not in TAUS_S for t in taus):
            raise DecisionRefusal("release requests an unregistered tau")
        predicted: dict[str, dict[str, float]] = {}
        realized: dict[str, dict[str, float]] = {}
        for arm, plan in plans.items():
            if hasattr(model, "predict_states"):
                states = model.predict_states(row.get("state"), plan, max(taus), seed)
            elif callable(model):
                states = model(row.get("state"), plan, max(taus), seed)
            else:
                raise DecisionRefusal("model lacks predict_states(state, plan, horizon, seed)")
            predicted[arm] = {
                str(t): _delay_or_na(states, t0_s=float(t0), tau_s=t) for t in taus
            }
            truth = row.get("true", {}).get(arm) if isinstance(row.get("true"), Mapping) else None
            if truth is not None:
                if isinstance(truth, Mapping) and "time_s" not in truth:
                    realized[arm] = {str(t): float(truth[str(t)]) for t in taus}
                else:
                    realized[arm] = {
                        str(t): _delay_or_na(truth, t0_s=float(t0), tau_s=t) for t in taus
                    }
        for tau in taus:
            key = str(tau)
            choices = {arm: values[key] for arm, values in predicted.items()
                       if values[key] is not None}
            if "native" not in choices:
                output.append({"world": world, "t0_s": float(t0), "tau_s": tau,
                               "status": "n/a", "reason": "native rollout shorter than tau"})
                continue
            # Native-first rank (registered abstention rule); equal predicted
            # delays must select native, not the lexically smallest arm.
            chosen = min(choices, key=lambda arm: (choices[arm],
                                                   _ARM_RANK.get(arm, len(_ARM_RANK)),
                                                   arm))
            item: dict[str, Any] = {
                "world": world,
                "t0_s": float(t0),
                "tau_s": tau,
                "predicted_delay_s": choices,
                "argmin": chosen,
                "paired_difference_vs_native_s": {
                    arm: value - choices["native"] for arm, value in choices.items()
                },
            }
            if realized:
                true_values = {arm: values[key] for arm, values in realized.items()
                               if values[key] is not None}
                best = min(true_values.values()) if true_values else None
                item["true_delay_s"] = true_values
                item["argmin_regret_s"] = (true_values[chosen] - best
                                             if best is not None and chosen in true_values else None)
                if "pi" in true_values and "native" in true_values:
                    item["pi_native_true_difference_s"] = true_values["pi"] - true_values["native"]
                    if "pi" in choices:
                        item["pi_native_pred_difference_s"] = choices["pi"] - choices["native"]
            output.append(item)
    result = {"schema": "wms_decide/1", "seed": int(seed), "n_decisions": len(output),
              "taus_s": list(TAUS_S), "rows": output}
    pairs = [(item["pi_native_pred_difference_s"], item["pi_native_true_difference_s"])
             for item in output if "pi_native_pred_difference_s" in item]
    if pairs:
        errors = np.asarray([pred - true for pred, true in pairs], dtype=float)
        result["paired_difference_accuracy"] = {
            "target": "pi_minus_native_delay_s",
            "n": len(pairs),
            "mae_s": float(np.mean(np.abs(errors))),
            "bias_s": float(np.mean(errors)),
            "sign_accuracy": float(np.mean([
                np.sign(pred) == np.sign(true) for pred, true in pairs
            ])),
        }
    dq = _decision_quality_result(output)
    if dq is not None:
        result["decision_quality"] = dq
    summary = decision_summary(output)
    if summary:
        result["decision_summary"] = summary
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(sorted_values):
        j = i
        while j + 1 < len(sorted_values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(predicted: np.ndarray, truth: np.ndarray) -> float | None:
    if len(predicted) < 2:
        return None
    ra, rb = _rankdata(predicted), _rankdata(truth)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    if denom <= 0.0:
        return None
    return float((ra * rb).sum() / denom)


def decision_summary(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Per-horizon sign agreement, rank correlation and native-retention regret.

    ``regret`` is the recorded delay penalty of the arm chosen from the predicted
    difference; ``always_native_s`` is the same sum for the mandatory native
    retention policy. ``negative`` regret improvement means the predictor saved
    delay relative to always-native.
    """
    grouped: dict[float, list[Mapping[str, Any]]] = {}
    for item in rows:
        if "pi_native_pred_difference_s" not in item:
            continue
        grouped.setdefault(float(item["tau_s"]), []).append(item)
    out: dict[str, Any] = {}
    for tau, items in sorted(grouped.items()):
        predicted = np.asarray([item["paired_difference_vs_native_s"]["pi"]
                                for item in items], dtype=float)
        truth = np.asarray([item["pi_native_true_difference_s"] for item in items], dtype=float)
        regret = 0.0
        native_regret = 0.0
        correct = 0
        counted = 0
        for item in items:
            true_values = item.get("true_delay_s") or {}
            if "native" not in true_values or "pi" not in true_values:
                continue
            best = min(float(true_values["native"]), float(true_values["pi"]))
            chosen = item["argmin"]
            if chosen not in true_values:
                continue
            regret += float(true_values[chosen]) - best
            native_regret += float(true_values["native"]) - best
            correct += int(float(true_values[chosen]) <= best + 1e-9)
            counted += 1
        out[str(tau)] = {
            "n": len(items),
            "sign_agreement": float(np.mean(np.sign(predicted) == np.sign(truth))),
            "rank_correlation": _spearman(predicted, truth),
            "best_arm_accuracy": (correct / counted if counted else None),
            "regret_s": regret,
            "always_native_regret_s": native_regret,
            "regret_improvement_vs_native_s": native_regret - regret,
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m wms_decide")
    parser.add_argument("--model", default=None,
                        help="MODULE:FACTORY or model JSON (native next-state boundary)")
    parser.add_argument("--wms-eval-model", default=None,
                        help="wms_eval MODEL_SPECS name: keep_speed, pure_physics, a11_*, wms")
    parser.add_argument("--wms-eval-plugin", default=None,
                        help="wms_eval plugin module:object override")
    parser.add_argument("--wms-eval-ckpt", default=None,
                        help="wm_s/1 checkpoint when --wms-eval-model wms is selected")
    parser.add_argument("--release-spec", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repl", action="store_true", help="reserved TEST_REPL mode; refused")
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args(argv)
    try:
        if args.repl:
            raise DecisionRefusal("TEST_REPL is prohibited for the TRAIN-only T7 dry run")
        if bool(args.model) == bool(args.wms_eval_model):
            raise DecisionRefusal("select exactly one of --model or --wms-eval-model")
        if args.wms_eval_model:
            model = load_wms_eval_model(args.wms_eval_model, args.wms_eval_plugin,
                                        args.wms_eval_ckpt)
        else:
            model = _load_model(args.model)
        result = score_rows(model, _load_release(args.release_spec), seed=args.seed)
        out = Path(args.out)
        if out.exists():
            raise DecisionRefusal(f"output already exists: {out}")
        out.mkdir(parents=True)
        (out / "DECISIONS.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"schema": result["schema"], "n_decisions": result["n_decisions"],
                          "out": str(out)}))
        return 0
    except (DecisionRefusal, OSError, ValueError, KeyError) as exc:
        parser.exit(87, f"wms_decide refused: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
