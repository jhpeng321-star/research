"""Streaming, strict native/wall C1 scoring; no unsupported-arm substitution.

Run from output/wd01/wm_n1: python -m lpwo.score_wall --help.
Only inline materialised C1 states are accepted. Recorded CAVs are exogenous
offline inputs for wms/blackbox_traj; blackbox_id keeps its learned-CAV contract.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from wms_eval import horizon_report, traffic_eval
from wms_eval.c1v2 import _car_compact, iter_windows
from wms_eval.data import DataError, initial_frame, recorded_for
from wms_eval.metrics import car_metrics, traffic_metrics

WORKSPACE = Path(__file__).resolve().parents[4]
HORIZONS = (10.0, 30.0, 60.0, 120.0)
ARMS = ("native", "wall")
MODELS = ("wms", "blackbox_traj", "blackbox_id")
RANKING_BACKEND = "lpwo.score_wall two-arm regret + wms_eval.traffic_eval sign/rank"


class WallScoreError(ValueError):
    """An input or rollout cannot support this wall comparison."""


def local_path(value: str | Path) -> Path:
    path = Path(value).resolve()
    try:
        path.relative_to(WORKSPACE)
    except ValueError as exc:
        raise WallScoreError(f"path outside workspace: {path}") from exc
    return path


def checked_path(value, guard) -> Path:
    path = local_path(value)
    if guard.check_path(path) is False:
        raise WallScoreError(f"guard denied path: {path}")
    return path


def checked_record(record, guard) -> None:
    if guard.check_record(record) is False:
        raise WallScoreError(f"guard denied record: {record.get('key')}")


def validate_frames(frames: list[dict], t0_s: float, horizon_s: float) -> None:
    """Require a complete 0.5 s grid; nearest-frame metrics must not extrapolate."""
    expected = int(round(horizon_s / 0.5)) + 1
    if len(frames) != expected:
        raise WallScoreError(f"expected {expected} frames, got {len(frames)}")
    for index, frame in enumerate(frames):
        time_s = float(frame["time_s"])
        if not math.isfinite(time_s) or abs(time_s - (t0_s + index * 0.5)) > 1e-6:
            raise WallScoreError(f"invalid 0.5 s timestamp at frame {index}")
        seen = set()
        for row in frame["vehicles"]:
            vid = str(row["id"])
            if vid in seen:
                raise WallScoreError(f"duplicate vehicle {vid!r} at frame {index}")
            seen.add(vid)
            for key in ("s", "v", "lane"):
                if not math.isfinite(float(row[key])):
                    raise WallScoreError(f"nonfinite {key} at frame {index}")


def _snapshot(frame: dict) -> dict:
    return {str(v["id"]): (float(v["s"]), float(v["v"]), int(v["lane"]),
                           bool(v.get("is_cav", False)))
            for v in frame["vehicles"] if v.get("present", True)}


def source_arm(record: dict) -> str:
    if record.get("source_arm") is not None:
        return str(record["source_arm"])
    if record.get("arm") is not None:
        return str(record["arm"])
    arms = record.get("recorded") or {}
    return "wall" if "wall" in arms else next(iter(arms), "unknown")


def validate_record(record: dict) -> tuple[float, dict[str, list[dict]], dict[str, str]]:
    if not isinstance(record, dict) or not record.get("world"):
        raise WallScoreError("state needs an explicit world identity")
    # The generic adapters may consult these paths even with inline trajectories.
    if any(record.get(k) for k in ("recorded_paths", "base_run_dir", "state_row")):
        raise WallScoreError("only inline materialised states; raw/path-backed states refused")
    if float(record.get("dt_s", 0.5)) != 0.5:
        raise WallScoreError("scoring requires dt_s=0.5")
    t0_s = float(record["t0_s"])
    if not math.isfinite(t0_s):
        raise WallScoreError("nonfinite t0_s")
    if float(record.get("horizon_s", 120.0)) < 120.0:
        raise WallScoreError("record is shorter than the required 120 s")
    inline = record.get("recorded")
    if not isinstance(inline, dict) or not inline:
        raise WallScoreError("at least one inline recorded trajectory is required")
    own = str(record.get("arm") or ("wall" if "wall" in inline else next(iter(inline))))
    if own not in inline:
        raise WallScoreError(f"own arm {own!r} has no inline recording")
    initial = initial_frame(record)
    if abs(float(initial["time_s"]) - t0_s) > 1e-6:
        raise WallScoreError("initial state time differs from t0_s")
    recorded, unavailable = {}, {a: "missing_recorded_arm" for a in ARMS if a not in inline}
    for arm in dict.fromkeys((*ARMS, own)):
        if arm not in inline:
            continue
        frames = recorded_for(record, arm)
        # Longer recordings are cropped, never padded or nearest-extrapolated.
        frames = [f for f in frames if float(f["time_s"]) <= t0_s + 120.0 + 1e-6]
        validate_frames(frames, t0_s, 120.0)
        if _snapshot(frames[0]) != _snapshot(initial):
            if arm == own:
                raise WallScoreError(f"{arm} is not paired to the identical initial state")
            unavailable[arm] = "initial_state_mismatch"
            continue
        recorded[arm] = frames
    return t0_s, recorded, unavailable


def delay_cost(frames: list[dict], t0_s: float, horizon: float, v0: float) -> float:
    """All-section normalized delay seconds over [t0, t0+h), local metric ops."""
    selected = [f for f in frames if t0_s <= float(f["time_s"]) < t0_s + horizon]
    bins = traffic_metrics(selected, t0_s=t0_s, bin_s=horizon, free_speed_mps=v0)
    values = [block["sections"][section]["total_delay"]
              for block in bins.values() for section in traffic_eval.SECTIONS]
    if not values or any(v is None or not math.isfinite(float(v)) for v in values):
        raise WallScoreError("normalized delay cost unavailable/nonfinite")
    return sum(float(v) for v in values)


def rank_pair(predicted_delta: float, recorded_delta: float) -> dict:
    """Wall-minus-native cost; lower wins, predicted ties select native."""
    if not all(math.isfinite(v) for v in (predicted_delta, recorded_delta)):
        raise WallScoreError("nonfinite paired delta")
    chosen = "wall" if predicted_delta < -traffic_eval.EPS else "native"
    realized_relative = recorded_delta if chosen == "wall" else 0.0
    return {"predicted_delta": predicted_delta, "recorded_delta": recorded_delta,
            "sign_agreement": traffic_eval._sign(predicted_delta) ==
                              traffic_eval._sign(recorded_delta),
            "selected_arm": chosen,
            "regret_s": realized_relative - min(0.0, recorded_delta)}


def evaluate_state(record: dict, predictor, *, mode: str = "recorded_cav", guard=None) -> dict:
    if mode != "recorded_cav":
        raise WallScoreError("wall_exec unavailable: rollout needs the documented wall-session hook")
    if guard is not None:
        # Guard exceptions/denials must not be swallowed as model arm errors.
        checked_record(record, guard)
    t0_s, recorded, unavailable = validate_record(record)
    v0 = float(record.get("v0_mps", 13.89))
    if not math.isfinite(v0) or v0 <= 0:
        raise WallScoreError("v0_mps must be finite and positive")
    arms = {a: {"error": None, "unavailable": reason} for a, reason in unavailable.items()}
    predicted_costs, recorded_costs = {}, {}
    for arm in recorded:
        try:
            # Deliberately NOT c1v2.evaluate_state/_predict: they permit wall->native.
            predicted = predictor(record, {**record.get("cav_controllers", {}), "arm": arm},
                                  120.0, int(record.get("seed", 0)))
            validate_frames(predicted, t0_s, 120.0)
            if _snapshot(predicted[0]) != _snapshot(initial_frame(record)):
                raise WallScoreError("prediction changed the initial snapshot")
            arms[arm] = {"error": None, "car": _car_compact(car_metrics(
                predicted, recorded[arm], t0_s=t0_s, horizons=HORIZONS))}
            predicted_costs[arm] = {str(h): delay_cost(predicted, t0_s, h, v0)
                                   for h in HORIZONS}
            recorded_costs[arm] = {str(h): delay_cost(recorded[arm], t0_s, h, v0)
                                  for h in HORIZONS}
        except Exception as exc:
            arms[arm] = {"error": f"{type(exc).__name__}: {exc}"}
    paired = {}
    is_pair = all(a in recorded and not arms[a].get("error") for a in ARMS)
    # Materialisation may have additional physical matching evidence. A declared
    # nonpair cannot be promoted just because its compact snapshots match.
    is_pair = is_pair and record.get("paired", True) is True
    if is_pair:
        for h in HORIZONS:
            key = str(h)
            paired[key] = rank_pair(predicted_costs["wall"][key] - predicted_costs["native"][key],
                                    recorded_costs["wall"][key] - recorded_costs["native"][key])
    return {"world": str(record["world"]), "t0_s": t0_s,
            "key": record.get("key"), "source_arm": source_arm(record),
            "source_set": record.get("source_set"), "arms": arms, "paired": paired,
            "pair_status": "identical_initial_pair" if is_pair else
                           record.get("native_pair_status", "unavailable_or_failed_pair")}


def state_key(row: dict) -> tuple:
    return (row["world"], float(row["t0_s"]), str(row.get("key") or ""),
            str(row.get("source_arm") or ""))


def bootstrap_keys(mine: dict, theirs: dict, keys: list, arm: str, horizon: float,
                   *, n_boot: int, seed: int) -> dict:
    # horizon_report keys only (world,round(t0,3)). After exact four-part matching,
    # use unique summary-only indices so distinct sources at one t0 cannot collapse.
    left = [{**mine[k], "t0_s": float(i)} for i, k in enumerate(keys)]
    right = [{**theirs[k], "t0_s": float(i)} for i, k in enumerate(keys)]
    return horizon_report.paired_bootstrap(left, right, arm, horizon,
                                           n_boot=n_boot, seed=seed)


def summarize_results(evaluations: dict[str, list[dict]], *, n_boot=1000, seed=0) -> dict:
    """Common-state comparisons, retaining explicit error/exclusion denominators."""
    keyed = {}
    if not evaluations:
        raise WallScoreError("no model evaluations")
    for model, rows in evaluations.items():
        keyed[model] = {}
        for row in rows:
            key = state_key(row)
            if key in keyed[model]:
                raise WallScoreError(f"duplicate (world,t0_s,key,source_arm) for {model}: {key}")
            keyed[model][key] = row
    universe = set().union(*(set(v) for v in keyed.values()))
    common = set.intersection(*(set(v) for v in keyed.values()))
    out = {"ranking_backend": RANKING_BACKEND, "cost": "all-section normalized delay (s)",
           "delta": "wall minus native; negative = improvement", "tie_rule": "native",
           "n_union_states": len(universe), "n_shared_states": len(common),
           "models": {}, "structured_minus_blackboxes": {}}
    for model, rows in evaluations.items():
        arms = sorted(set(ARMS) | {a for r in rows for a in r["arms"]})
        out["models"][model] = {
            "n_states": len(rows),
            "n_arm_errors": {a: sum(bool(r["arms"].get(a, {}).get("error")) for r in rows) for a in arms},
            "n_arm_unavailable": {a: sum(a not in r["arms"] or
                                         bool(r["arms"][a].get("unavailable")) for r in rows)
                                  for a in arms},
            "hdv_ds": {a: {str(h): horizon_report.horizon_block(rows, a, h)
                            for h in HORIZONS} for a in arms},
            "blowup": {a: horizon_report.blowup_rate(rows, a) for a in arms},
            "ranking_common_states": {}}
    for h in HORIZONS:
        hk = str(h)
        valid = sorted(k for k in common if all(hk in keyed[m][k]["paired"] for m in keyed))
        for model in keyed:
            pairs = [keyed[model][k]["paired"][hk] for k in valid]
            pred, rec = ([p[name] for p in pairs]
                         for name in ("predicted_delta", "recorded_delta"))
            # All models must score the same recorded labels, not just matching keys.
            if any(keyed[model][k]["paired"][hk]["recorded_delta"] !=
                   keyed[next(iter(keyed))][k]["paired"][hk]["recorded_delta"] for k in valid):
                raise WallScoreError("recorded paired labels differ across models")
            regrets = [p["regret_s"] for p in pairs]
            out["models"][model]["ranking_common_states"][hk] = {
                "n": len(pairs), "n_excluded": len(universe) - len(pairs),
                "sign_agreement": sum(p["sign_agreement"] for p in pairs) / len(pairs) if pairs else None,
                "rank_corr": traffic_eval.spearman(pred, rec),
                "regret_s": horizon_report.summarize(regrets),
                "mean_regret_s": sum(regrets) / len(regrets) if regrets else None}
        for bb in ("blackbox_traj", "blackbox_id"):
            if "wms" not in keyed or bb not in keyed:
                continue
            blocks = out["structured_minus_blackboxes"].setdefault(bb, {})
            # Reuse the existing WORLD-resampled paired bootstrap for compact scalar rows.
            arows, brows = {}, {}
            for k in valid:
                for model, target in (("wms", arows), (bb, brows)):
                    target[k] = {"world": k[0], "t0_s": k[1], "arms": {
                        "regret": {"car": {hk: {"hdv": {
                            "ds_med": keyed[model][k]["paired"][hk]["regret_s"]}}}}}}
            hdv_keys = sorted(k for k in set(keyed["wms"]) & set(keyed[bb])
                              if all(horizon_report._car_value(keyed[m][k], "wall", h, "ds_med")
                                     is not None for m in ("wms", bb)))
            blocks[hk] = {"regret_s": bootstrap_keys(
                arows, brows, valid, "regret", h, n_boot=n_boot, seed=seed),
                "hdv_ds_wall_m": bootstrap_keys(
                    keyed["wms"], keyed[bb], hdv_keys,
                    "wall", h, n_boot=n_boot, seed=seed)}
    return out


def _entries(values: list[str], *, models=False) -> dict[str, Path]:
    out = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not path or name in out:
            raise WallScoreError(f"expected unique NAME=PATH: {value!r}")
        if models and name not in MODELS:
            raise WallScoreError(f"unsupported model {name!r}; choose {MODELS}")
        out[name] = local_path(path)
    return out


def physics_source(checkpoint: Path, guard) -> dict:
    """Require explicit local physics; the shared adapter otherwise defaults it."""
    path = checked_path(checkpoint.with_name("physics.json"), guard)
    if not path.is_file():
        raise WallScoreError(f"missing explicit checkpoint physics: {path}")
    physics = guard.read_json(path)
    if not isinstance(physics, dict) or not physics:
        raise WallScoreError(f"empty/invalid physics configuration: {path}")
    if float(physics.get("dt_s", 0.0)) != 0.5:
        raise WallScoreError("checkpoint physics must declare dt_s=0.5")
    for value in physics.values():
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            raise WallScoreError("nonfinite checkpoint physics")
    return {"path": str(path), "physics": physics}


def checkpoint_predictor(model: str, checkpoint: Path, physics: dict):
    # All three local factories use this adapter. Passing physics explicitly also
    # avoids load_model's blackbox branch dropping its physics keyword override.
    from wms_eval.models import flags_for
    from wms_eval.wms_plugin import WmsCheckpointModel

    kind = "structured" if model == "wms" else model
    return WmsCheckpointModel(checkpoint, flags_for(model), physics=physics,
                              model_kind=kind).predict_states


def run(args, *, guard) -> int:
    if args.mode != "recorded_cav":
        raise WallScoreError("wall_exec unavailable: see SCORING_AND_CHAINS.md rollout hook")
    models = _entries(args.models, models=True)
    if set(models) != set(MODELS):
        raise WallScoreError("supply wms, blackbox_traj and blackbox_id checkpoints")
    sets = _entries(args.sets)
    if args.limit is not None and args.limit <= 0:
        raise WallScoreError("--limit must be positive")
    if args.bootstrap < 0:
        raise WallScoreError("--bootstrap must be nonnegative")
    # Check every input path before ANY data open, including lazy torch loads.
    for path in (*models.values(), *(p.with_name("physics.json") for p in models.values()),
                 *sets.values()):
        checked_path(path, guard)
    physics = {m: physics_source(p, guard) for m, p in models.items()}
    out = checked_path(args.out, guard)
    out.mkdir(parents=True, exist_ok=False)
    args._score_output_created = True

    report = {"schema": "lpwo_wall_score/1", "mode": args.mode,
              "status": "scoring", "limitation": "recorded CAV motion is offline/exogenous",
              "cav_contract": {"wms": "recorded teacher", "blackbox_traj": "recorded teacher",
                               "blackbox_id": "learned CAVs (not teacher-forced)"},
              "flag_identity": "pending outside receipt confirmation",
              "sources": {"models": {m: str(p) for m, p in models.items()},
                          "sets": {s: str(p) for s, p in sets.items()},
                          "physics": physics}, "sets": {}}
    evaluations = {s: {} for s in sets}
    errors = 0
    for model, checkpoint in models.items():
        checked_path(checkpoint, guard)
        predictor = checkpoint_predictor(model, checkpoint, physics[model]["physics"])
        for name, path in sets.items():
            rows = []  # Compact summaries only; raw windows/trajectories never accumulate.
            with checked_path(out / f"states__{model}__{name}.json", guard).open("x", encoding="utf-8") as fh:
                fh.write(json.dumps({"schema": "wms_eval_c1v2_states/1", "model": model,
                                     "set": name})[:-1] + ', "states": [')
                for record in iter_windows(checked_path(path, guard), limit=args.limit):
                    result = evaluate_state(record, predictor, mode=args.mode, guard=guard)
                    if rows:
                        fh.write(",")
                    fh.write(json.dumps(result, allow_nan=False))
                    rows.append(result)
                    errors += sum(bool(block.get("error")) for block in result["arms"].values())
                fh.write("]}\n")
            if not rows:
                raise WallScoreError(f"empty state set {name!r}")
            evaluations[name][model] = rows
    for name, rows in evaluations.items():
        report["sets"][name] = summarize_results(rows, n_boot=args.bootstrap, seed=args.seed)
    report["status"] = "incomplete_arm_errors" if errors else "scored"
    report["n_arm_errors"] = errors
    checked_path(out / "WALL_SCORE.json", guard).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    print(json.dumps({"out": str(out), "status": report["status"], "n_arm_errors": errors}))
    return 2 if errors else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", action="append", required=True, help="SPEC=CKPT (all three models)")
    p.add_argument("--sets", action="append", required=True, help="NAME=PATH streamed C1 windows JSON")
    p.add_argument("--mode", choices=("recorded_cav", "wall_exec"), default="recorded_cav")
    p.add_argument("--limit", type=int)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="new directory within this workspace")
    p.add_argument("--guard-spec", required=True)
    p.add_argument("--guard-tool", required=True)
    p.add_argument("--allowed-root", action="append", required=True,
                   help="workspace-local DataGuard allowed root (repeatable, include output)")
    args = p.parse_args(argv)
    guard = None
    rc = 2
    try:
        if args.mode != "recorded_cav":
            raise WallScoreError("wall_exec unavailable: see SCORING_AND_CHAINS.md rollout hook")
        from .guard import DataGuard

        guard = DataGuard(local_path(args.guard_spec), local_path(args.guard_tool),
                          [local_path(p) for p in args.allowed_root])
        rc = run(args, guard=guard)
    except (WallScoreError, DataError, OSError, ValueError, RuntimeError, ImportError) as exc:
        print(json.dumps({"status": "refused", "error": f"{type(exc).__name__}: {exc}"}))
    finally:
        if guard is not None:
            try:
                audit = guard.audit()
                print(json.dumps({"guard_audit": audit}, allow_nan=False))
                out = local_path(args.out)
                if getattr(args, "_score_output_created", False):
                    checked_path(out / "GUARD_AUDIT.json", guard).write_text(
                        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            except (OSError, ValueError, RuntimeError, TypeError, AttributeError) as exc:
                print(json.dumps({"status": "refused", "error": f"guard audit failed: {exc}"}))
                rc = 2
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
