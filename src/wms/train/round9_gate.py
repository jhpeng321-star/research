"""Fail-closed round-9 stage-2C smoke verdict, shared by the launcher and tests."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


POSITIVE = ("merge_pivot_tau", "merge_pivot_w", "cuda_peak_mb")
# Owner decision 2026-10-22: the objective is teacher-forced + F2 only. The recorded-matched
# on-policy merge objective is retired (weight 0, not in loss/backward); its rules are replaced
# by the flag below. Zero mass is checked INSIDE the loss, i.e. after the counted exclusion.
OBJECTIVE_FLAG = "merge_pivot_on_objective_disabled"
COUNTERS = (
    "merge_pivot_tf_positive_mass_events", "merge_pivot_tf_zero_mass_events",
    "merge_pivot_tf_scored_seconds", "merge_pivot_tf_scored_cars",
    "merge_pivot_tf_excluded_motion", "merge_pivot_grad_norm",
)


def finite(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def read_rows(path):
    # Blank/malformed/non-object rows are evidence failures, not ignorable noise.
    text = Path(path).read_text(encoding="utf-8")
    if "Traceback" in text:
        raise ValueError("Traceback in train log")
    rows = [json.loads(line) for line in text.splitlines()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("train log rows must be objects")
    return rows


def check_log(train_log: Path, smoke_log: Path, *, train_exit_code: int = 0) -> dict:
    errors, metrics = [], {}
    if train_exit_code != 0:
        errors.append(f"training exited {train_exit_code}")
    try:
        if "Traceback" in Path(smoke_log).read_text(encoding="utf-8"):
            errors.append("Traceback in smoke log")
    except (OSError, UnicodeError) as exc:
        errors.append(f"unreadable smoke log: {exc}")
    try:
        rows = read_rows(train_log)
    except (OSError, ValueError) as exc:
        errors.append(f"unreadable train log: {exc}")
        return {"ok": False, "errors": errors, "metrics": metrics}

    metrics["optimizer_steps"] = len(rows)
    if len(rows) < 4:
        errors.append("need at least 4 optimizer steps")
    valid = True
    for index, row in enumerate(rows, 1):
        for key in ("step", "loss", "stage_s", OBJECTIVE_FLAG, *POSITIVE, *COUNTERS):
            value = row.get(key)
            if not finite(value):
                errors.append(f"row {index}: {key} missing/null/non-finite/non-numeric")
                valid = False
            elif key in (*POSITIVE, "stage_s") and value <= 0:
                errors.append(f"row {index}: {key} must be positive")
            elif key in COUNTERS and value < 0:
                errors.append(f"row {index}: {key} must be nonnegative")
        if finite(row.get(OBJECTIVE_FLAG)) and row[OBJECTIVE_FLAG] != 1:
            errors.append(f"row {index}: {OBJECTIVE_FLAG} must be 1 (on-policy objective retired)")
        step = row.get("step")
        if type(step) is not int or step != index:
            errors.append(f"row {index}: step must be consecutive integer {index}")
    if valid and rows:
        for key in COUNTERS:
            metrics[key] = sum(row[key] for row in rows)
            if not finite(metrics[key]):
                errors.append(f"non-finite total {key}")
        for key in ("merge_pivot_tf_positive_mass_events", "merge_pivot_tf_scored_cars",
                    "merge_pivot_grad_norm"):
            if metrics[key] <= 0:
                errors.append(f"total {key} must be >0")
        for key in ("merge_pivot_tf_zero_mass_events",):
            if metrics[key] != 0:
                errors.append(f"total {key} must be zero")
        for stage in (120, 300):
            if sum(row["stage_s"] == stage for row in rows) < 2:
                errors.append(f"need at least 2 optimizer steps at stage_s={stage}")
            scored = sum(row["merge_pivot_tf_scored_seconds"] for row in rows
                         if row["stage_s"] >= stage)
            metrics[f"scored_s_{stage}"] = scored
            if not finite(scored) or scored <= 0:
                errors.append(f"scored seconds at stage_s>={stage} must be finite and >0")
        metrics["cuda_peak_mb"] = max(row["cuda_peak_mb"] for row in rows)
        # Informational (never a pass/fail input): counted-exclusion totals by class over the smoke rows.
        keys = sorted({key for row in rows for key in row if key.startswith("merge_pivot_support_")
                       and finite(row[key])})
        metrics["support_totals"] = {key: sum(row.get(key, 0) for row in rows if finite(row.get(key, 0)))
                                     for key in keys}
        cars = metrics["merge_pivot_tf_scored_cars"] + metrics["merge_pivot_tf_excluded_motion"]
        metrics["tf_excluded_motion_rate"] = (metrics["merge_pivot_tf_excluded_motion"] / cars
                                              if cars else None)
    return {"ok": not errors, "errors": errors, "metrics": metrics}


def check_checkpoints(smoke_dir: Path, rows: list[dict]) -> list[dict]:
    """Actually reload each saved model, including all pivot weights, on CPU.

    The existing wm_s/1 format is weights-only. This verifies serialization after
    real optimizer updates; it does not claim to resume Adam's internal state.
    """
    import torch

    from wms.model.checkpoint import load_checkpoint

    reports = []
    for name, stage in (("wm_s_stage120.pt", 120), ("wm_s_stage300.pt", 300), ("wm_s.pt", None)):
        path = smoke_dir / name
        payload = torch.load(path, map_location="cpu", weights_only=True)
        loaded = load_checkpoint(path, device="cpu", strict=True)
        weights = loaded.model.state_dict()
        saved = payload["model"]
        if weights.keys() != saved.keys() or not all(
                torch.isfinite(value).all() and torch.equal(value, saved[key])
                for key, value in weights.items()):
            raise ValueError(f"{name}: weights missing, non-finite or changed on reload")
        if not any(key.startswith("merge_pivot.") for key in weights):
            raise ValueError(f"{name}: no merge_pivot weights")
        # The flat checkpoint config omits default-valued model switches.
        if (loaded.model.spec.merge_pivot_state_st is not True
                or not loaded.model.physics.ramp_lane_end_stop):
            raise ValueError(f"{name}: state ST / lane-end stop not enabled")
        if (loaded.legacy_config.get("batch_size") != 4
                or tuple(loaded.legacy_config.get("batch_size_per_stage") or ()) != (4, 4)):
            raise ValueError(f"{name}: both smoke stages must use batch 4")
        if stage is not None:
            last_step = max(row["step"] for row in rows if row["stage_s"] == stage)
            if payload.get("stage_s") != stage or payload.get("step") != last_step:
                raise ValueError(f"{name}: checkpoint stage/step mismatch")
        elif payload.get("history") != rows:
            raise ValueError(f"{name}: checkpoint history does not match smoke log")
        reports.append({"file": name, "sha256": loaded.sha256, "reloaded": True})
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--smoke-log", type=Path, required=True)
    parser.add_argument("--train-exit-code", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    train_log = args.smoke_dir / "train_log.jsonl"
    result = check_log(train_log, args.smoke_log, train_exit_code=args.train_exit_code)
    if result["ok"]:
        try:
            result["checkpoints"] = check_checkpoints(args.smoke_dir, read_rows(train_log))
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"checkpoint reload failed: {exc}")
    try:
        with args.report.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        print(f"smoke: bad report: {exc}")
        return 3
    print("smoke: ok" if result["ok"] else "smoke: bad " + "; ".join(result["errors"]))
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
