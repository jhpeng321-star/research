"""CLI for the standard WM-S offline evaluation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from .data import WORKSPACE, DataError, iter_states, json_default, recorded_for
from .metrics import HORIZONS_S, car_metrics, paired_difference, traffic_metrics
from .models import MODEL_SPECS, ModelUnavailable, load_model


ARMS = ("native", "pi")


def _available_arms(record: dict) -> list[str]:
    """Arms that have a recorded trajectory in this state (order preserved)."""
    recorded = record.get("recorded")
    if isinstance(recorded, dict):
        return [arm for arm in ARMS if recorded.get(arm)]
    out = []
    for arm in ARMS:
        try:
            recorded_for(record, arm)
        except DataError:
            continue
        out.append(arm)
    return out


def _inside_workspace(path: Path) -> Path:
    # Anchor relative arguments at this checkout, never at an external target.
    # Resolve real paths on both root and target before the containment check:
    # a lexical abspath/relative_to check accepts a junction or symlink that
    # redirects an input read or output write outside the workspace.
    root = Path(os.path.realpath(str(WORKSPACE)))
    target = path if path.is_absolute() else root / path
    resolved = Path(os.path.realpath(str(target)))
    try:
        common = os.path.commonpath([str(root), str(resolved)])
    except ValueError:
        common = None
    if common is None or os.path.normcase(common) != os.path.normcase(str(root)):
        raise ValueError(f"path must be inside workspace: {resolved}")
    return resolved


def _plot(out_dir: Path, traffic: dict) -> str | None:
    """Write a dependency-free SVG speed plot when traffic data exists."""
    points = []
    for key, block in traffic.items():
        speed = block.get("sections", {}).get("merge", {}).get("mean_speed")
        if speed is not None:
            points.append((float(key), float(speed)))
    if not points:
        return None
    width, height = 640, 320
    vmax = max(1.0, max(y for _, y in points))
    coords = [(50 + x * 560 / max(1, len(points) - 1), 270 - y * 220 / vmax)
              for x, y in points]
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
           '<rect width="100%" height="100%" fill="white"/><text x="50" y="25">'
           'merge mean speed</text><polyline fill="none" stroke="#174a7e" stroke-width="2" '
           f'points="{poly}"/></svg>')
    path = out_dir / "plots" / "merge_mean_speed.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return str(path.relative_to(out_dir))


def run(args) -> int:
    states_path = _inside_workspace(Path(args.states))
    out_dir = _inside_workspace(Path(args.out))
    if out_dir.exists():
        raise ValueError(f"refusing to overwrite existing output: {out_dir}")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    physics = None
    if args.physics:
        physics_path = _inside_workspace(Path(args.physics))
        physics = json.loads(physics_path.read_text(encoding="utf-8"))
        if isinstance(physics, dict) and isinstance(physics.get("physics"), dict):
            physics = physics["physics"]
    predictor, flags = load_model(args.model, args.plugin, args.ckpt,
                                  entries=bool(getattr(args, "entries", False)),
                                  entries_v2=bool(getattr(args, "entries_v2", False)),
                                  physics=physics)
    out_dir.mkdir(parents=True)
    results = []
    for index, record in enumerate(iter_states(states_path)):
        state = dict(record)
        t0_s = float(record.get("t0_s", state.get("time_s", 0.0)))
        horizon_s = float(record.get("horizon_s", args.horizon))
        controllers = dict(record.get("cav_controllers", {}))
        controllers["cav_mode"] = getattr(args, "cav_mode", "teacher")
        seed = int(record.get("seed", args.seed))
        free_speed = float(record.get("v0_mps", 13.89))
        arms = _available_arms(record)
        if not arms:
            raise DataError(
                f"state {record.get('key', index)!r} has no recorded arm to score")
        recorded_arms = {arm: recorded_for(record, arm) for arm in arms}
        predicted_arms = {
            arm: predictor(state, dict(controllers, arm=arm), horizon_s, seed)
            for arm in arms
        }
        traffic_pred = {
            arm: traffic_metrics(predicted_arms[arm], t0_s=t0_s, free_speed_mps=free_speed)
            for arm in arms
        }
        traffic_rec = {
            arm: traffic_metrics(recorded_arms[arm], t0_s=t0_s, free_speed_mps=free_speed)
            for arm in arms
        }
        traffic = {"predicted": traffic_pred, "recorded": traffic_rec}
        if ARMS[0] in arms and ARMS[1] in arms:
            traffic["paired_predicted_pi_minus_native"] = paired_difference(
                traffic_pred["pi"], traffic_pred["native"])
            traffic["paired_recorded_pi_minus_native"] = paired_difference(
                traffic_rec["pi"], traffic_rec["native"])
            traffic["paired_difference_error"] = paired_difference(
                paired_difference(traffic_pred["pi"], traffic_pred["native"]),
                paired_difference(traffic_rec["pi"], traffic_rec["native"]))
        item = {
            "index": index, "world": record.get("world", f"state_{index}"), "t0_s": t0_s,
            "seed": int(record.get("seed", args.seed)), "model": args.model,
            "arms": arms, "paired": (ARMS[0] in arms and ARMS[1] in arms),
            "car": {
                arm: car_metrics(predicted_arms[arm], recorded_arms[arm],
                                 t0_s=t0_s, horizons=HORIZONS_S)
                for arm in arms
            },
            "traffic": traffic,
        }
        results.append(item)
    summary = {
        "schema": "wms_eval/1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": args.model, "model_flags": flags.__dict__, "physics": physics,
        "cav_mode": getattr(args, "cav_mode", "teacher"),
        "n_states": len(results),
        "n_paired": sum(1 for item in results if item["paired"]),
        "states": str(states_path), "results": results,
    }
    first_arm = "pi" if (results and "pi" in results[0]["traffic"]["predicted"]) else "native"
    plot = _plot(out_dir, results[0]["traffic"]["predicted"].get(first_arm, {})
                 if results else {})
    summary["plots"] = [plot] if plot else []
    (out_dir / "EVAL.json").write_text(json.dumps(summary, indent=2, default=json_default), encoding="utf-8")
    lines = [f"# WM-S evaluation: {args.model}", "", f"States: {len(results)}", "",
              "This is an offline, identity-matched evaluation. Native and PI arms use the "
              "same state, seed, section, and 30 s window operations.", ""]
    if plot:
        lines.append(f"Plot: `{plot}`")
    (out_dir / "EVAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"model": args.model, "states": len(results), "out": str(out_dir)}))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="wms_eval")
    sub = root.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run")
    run_p.add_argument("--model", required=True, choices=MODEL_SPECS)
    run_p.add_argument("--states", required=True)
    run_p.add_argument("--out", required=True)
    run_p.add_argument("--plugin", help="local module:object implementing predict_states")
    run_p.add_argument("--ckpt", help="WM-S wm_s/1 checkpoint for --model wms")
    run_p.add_argument("--physics", help="JSON dict (or TRAIN_RECEIPT.json) of the "
                                         "PhysicsConfig the wms checkpoint was trained with")
    run_p.add_argument("--horizon", type=float, default=300.0)
    run_p.add_argument("--cav-mode", choices=("teacher", "closed_loop"),
                       default="teacher",
                       help="wms CAV behavior: recorded-arm teacher (default) or execute controller code from predicted states")
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--entries", action="store_true",
                       help="observed-counter vehicle entries (RSU-observable t0 "
                            "counters; default OFF). Applies to the wms spec and to "
                            "the pure_physics/A11 baselines for a fair comparison.")
    run_p.add_argument("--entries-v2", action="store_true",
                       help="observed-HISTORY vehicle entries: recency-weighted rate "
                            "with trend, gamma inter-arrival process and empirical "
                            "entry lane/speed/gap/class, all from t <= t0 (default "
                            "OFF; implies --entries).")
    run_p.set_defaults(func=run)
    return root


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # ``c1``/``seen``/``materialize``/``c1sets``/``r2set`` own their argument
    # surface; dispatch before the ``run``-only parser so the subcommands can
    # evolve independently.
    if argv and argv[0] in ("c1", "seen", "materialize", "c1sets", "r2set", "traffic",
                            "horizon", "ranking"):
        from . import c1, c1sets, seen
        from . import materialize, r2set, traffic_eval
        from . import horizon_report
        from . import ranking

        command = {"c1": c1.main, "seen": seen.main, "materialize": materialize.main,
                   "c1sets": c1sets.main, "r2set": r2set.main,
                   "traffic": traffic_eval.main,
                   "horizon": horizon_report.main,
                   "ranking": ranking.main}[argv[0]]
        return command(argv[1:])
    if argv and argv[0] in ("materialise", "driver"):
        from . import driver

        return driver.main(argv)
    args = parser().parse_args(argv)
    try:
        return args.func(args)
    except (DataError, ModelUnavailable, ValueError, OSError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 2
