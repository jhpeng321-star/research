"""Fail-closed round-12C smoke verdict: physics-only core calibration of round 12.

Round 12C (S-R12C) = round 12 with EVERY learned acceleration emission disabled -- ``model.temporal_accel_off``
(R13F), the new ``model.residual_accel_off`` and ``model.follower_courtesy_off`` (BRAKE_DIAG's 'physics only
(residual and courtesy off)' arm as a training run) -- and the trainable set restricted to the driver encoder
(physical mode) + the shared IDM/MOBIL physics parameters: ``driver_encoder.head`` and
``driver_encoder.centre_logits`` only.  Single 120 s stage, 30 steps, batch 4, val 4 windows, driver prior OFF.
Checks, all fail-closed (round-12 checks adapted, then the round-12C contract):

* log: the round-9/10 merge census with merge_pivot frozen restricted to the SINGLE 120 s stage (>= 2 steps,
  no other stage, consecutive steps, positive tau/w/peak memory, tf zero-mass events = 0, scored seconds > 0);
  the round-10b/11 courtesy checks unchanged (the term still runs inside loss.follower_courtesy; it trains a
  frozen head, so it must stay positive but cannot move anything); fix-B cut-in term absent; fix-D telemetry;
  driver prior OFF (weight 0): ``driver_prior_means`` / ``driver_prior`` absent from every step;
* R12C log: the trainer's ``WM-S optimizer groups`` line parsed from the smoke log must list EXACTLY the five
  driver-encoder tensors (head + centre_logits) as trainable, with the frozen residual/temporal/courtesy
  prefixes matched and no unmatched freeze prefix;
* init report: identical to round 12 (physical centre init, residual accel re-init, fresh zero courtesy head,
  only the driver centre/bounds + courtesy head tensors new, source sha256 = the init checkpoint);
* checkpoints (stage 120 + final; there is no stage 300): strict reload, weights finite and unchanged,
  ``temporal_accel_off`` / ``residual_accel_off`` / ``follower_courtesy_off`` all on after reload, physical
  driver mode with the constant bounds, version-2 courtesy head with rollout_grad off, merge_pivot + state ST +
  lane-end stop, batch 4 single stage, stage/step/history consistent with the log; vs the init checkpoint every
  frozen-prefix tensor and merge_pivot bitwise equal and the changed set inside driver_encoder.head /
  driver_encoder.centre_logits (anything else FAILS, not informational); the re-initialised residual
  acceleration row 0 and the zero-init courtesy head are still exactly zero (emissions off, nets frozen); the
  saved config's ``driver_prior_weight`` = 0; the learned centre strictly inside the physical intervals; at the
  final checkpoint the driver head rows 0..4 and the learned centre have all moved (the core calibrated).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .round9_gate import OBJECTIVE_FLAG, POSITIVE, finite, read_rows
from .round10_gate import ROUND10_COUNTERS
from .round10b_gate import HEAD_LAST_WEIGHT, HEAD_PREFIX
from .round11_gate import head_range_errors, physics_accel_bounds
from .round12_gate import (ACCEL_HEAD, DRIVER_HEAD, DRIVER_ROWS, NEW_KEYS, PRIOR_MODES,
                           centre_physical, check_init_report, courtesy_log_errors,
                           cutin_absent_errors, prior_absent_errors, prior_weight, replay_errors)

#: the intended round-12C trainable set (everything else must stay bitwise equal to the init)
TRAINABLE_PREFIXES_12C = ("driver_encoder.head", "driver_encoder.centre_logits")
EXPECTED_TRAINABLE_TENSORS = ("driver_encoder.centre_logits", "driver_encoder.head.0.bias",
                              "driver_encoder.head.0.weight", "driver_encoder.head.2.bias",
                              "driver_encoder.head.2.weight")
REQUIRED = ("step", "loss", "stage_s", OBJECTIVE_FLAG, *POSITIVE, *ROUND10_COUNTERS)
STAGE_S = 120.0
OPTIMIZER_LINE = re.compile(r"^WM-S optimizer groups: (\{.*\})\s*$", re.MULTILINE)


def merge_log_errors_single_stage(rows: list[dict], metrics: dict) -> list[str]:
    """Round-10 merge census for the single 120 s stage (merge_pivot frozen: zero grad norm allowed)."""
    errors = []
    metrics["optimizer_steps"] = len(rows)
    if len(rows) < 2:
        errors.append("need at least 2 optimizer steps")
    valid = True
    for index, row in enumerate(rows, 1):
        for key in REQUIRED:
            value = row.get(key)
            if not finite(value):
                errors.append(f"row {index}: {key} missing/null/non-finite/non-numeric")
                valid = False
            elif key in (*POSITIVE, "stage_s") and value <= 0:
                errors.append(f"row {index}: {key} must be positive")
            elif key in ROUND10_COUNTERS and value < 0:
                errors.append(f"row {index}: {key} must be nonnegative")
        if finite(row.get(OBJECTIVE_FLAG)) and row[OBJECTIVE_FLAG] != 1:
            errors.append(f"row {index}: {OBJECTIVE_FLAG} must be 1 (on-policy objective retired)")
        if finite(row.get("stage_s")) and row["stage_s"] != STAGE_S:
            errors.append(f"row {index}: stage_s {row['stage_s']!r} != {STAGE_S!r} (single-stage schedule)")
        step = row.get("step")
        if type(step) is not int or step != index:
            errors.append(f"row {index}: step must be consecutive integer {index}")
    if valid and rows:
        for key in ROUND10_COUNTERS:
            metrics[key] = sum(row[key] for row in rows)
            if not finite(metrics[key]):
                errors.append(f"non-finite total {key}")
        for key in ("merge_pivot_tf_positive_mass_events", "merge_pivot_tf_scored_cars"):
            if metrics[key] <= 0:
                errors.append(f"total {key} must be >0")
        if metrics["merge_pivot_tf_zero_mass_events"] != 0:
            errors.append("total merge_pivot_tf_zero_mass_events must be zero")
        scored = sum(row["merge_pivot_tf_scored_seconds"] for row in rows)
        metrics["scored_s_120"] = scored
        if not finite(scored) or scored <= 0:
            errors.append("scored seconds must be finite and >0")
        metrics["cuda_peak_mb"] = max(row["cuda_peak_mb"] for row in rows)
    return errors


def optimizer_report_errors(smoke_log: Path, freeze_prefixes, metrics: dict) -> list[str]:
    """The trainable tensor list printed by the trainer must be exactly the round-12C set."""
    errors = []
    try:
        text = Path(smoke_log).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        return [f"unreadable smoke log: {exc}"]
    match = OPTIMIZER_LINE.search(text)
    if match is None:
        return ["smoke log has no 'WM-S optimizer groups' line"]
    try:
        report = json.loads(match.group(1))
    except ValueError as exc:
        return [f"optimizer groups line is not JSON: {exc}"]
    names = sorted(report.get("trainable_tensor_names") or ())
    metrics["trainable_tensor_names"] = names
    if names != sorted(EXPECTED_TRAINABLE_TENSORS):
        errors.append(f"trainable tensors {names} != the round-12C set {sorted(EXPECTED_TRAINABLE_TENSORS)}")
    frozen = list(report.get("freeze_prefixes") or ())
    for prefix in ("residual_net", "temporal", HEAD_PREFIX.rstrip(".")):
        if prefix not in frozen:
            errors.append(f"freeze_prefixes misses '{prefix}' (emission nets must be frozen)")
    if report.get("unmatched_freeze_prefixes"):
        errors.append(f"unmatched freeze prefixes: {report['unmatched_freeze_prefixes']}")
    metrics["optimizer_groups"] = report.get("groups")
    return errors


def check_log(train_log: Path, smoke_log: Path, *, train_exit_code: int = 0,
              accel_bounds: tuple[float, float] | None = None, prior_weight: float | None = None,
              freeze_prefixes=()) -> dict:
    """Round-12C log contract (single stage, prior off, physics-only trainable set)."""
    if accel_bounds is None:
        from wms.physics.config import PhysicsConfig

        accel_bounds = (PhysicsConfig().accel_min_mps2, PhysicsConfig().accel_max_mps2)
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
    errors.extend(merge_log_errors_single_stage(rows, metrics))
    errors.extend(courtesy_log_errors(rows, metrics))
    errors.extend(head_range_errors(rows, accel_bounds))
    metrics["head_range_bounds"] = [float(accel_bounds[0]), float(accel_bounds[1])]
    errors.extend(cutin_absent_errors(rows))
    errors.extend(replay_errors(rows, metrics))
    if prior_weight is not None and not (finite(prior_weight) and prior_weight >= 0):
        errors.append(f"bad driver prior weight {prior_weight!r}")
    elif prior_weight == 0:
        errors.extend(prior_absent_errors(rows))
    else:
        errors.append(f"round 12C runs with the driver prior OFF (weight 0), got {prior_weight!r}")
    errors.extend(optimizer_report_errors(smoke_log, freeze_prefixes, metrics))
    return {"ok": not errors, "errors": errors, "metrics": metrics}


def check_checkpoints(smoke_dir: Path, rows: list[dict], init_checkpoint: Path, freeze_prefixes,
                      init_report: dict) -> list[dict]:
    """Reload stage-120 and final strictly (CPU) and compare with the round-9 init checkpoint."""
    import torch

    from wms.model.checkpoint import load_checkpoint

    before = torch.load(init_checkpoint, map_location="cpu", weights_only=True)["model"]
    frozen = tuple(freeze_prefixes)
    centre_init = torch.tensor(init_report["driver_param_init"]["centre_logits"], dtype=torch.float32)
    reports = []
    for name, stage in (("wm_s_stage120.pt", STAGE_S), ("wm_s.pt", None)):
        path = smoke_dir / name
        payload = torch.load(path, map_location="cpu", weights_only=True)
        loaded = load_checkpoint(path, device="cpu", strict=True)
        weights = loaded.model.state_dict()
        saved = payload["model"]
        if weights.keys() != saved.keys() or not all(
                torch.isfinite(value).all() and torch.equal(value, saved[key])
                for key, value in weights.items()):
            raise ValueError(f"{name}: weights missing, non-finite or changed on reload")
        model = loaded.model
        spec = model.spec
        problems = []
        for flag in ("temporal_accel_off", "residual_accel_off", "follower_courtesy_off"):
            if getattr(spec, flag, None) is not True or getattr(model, flag, None) is not True:
                problems.append(f"{flag} is not on after reload")
        if problems:
            raise ValueError(f"{name}: " + "; ".join(problems))
        if (spec.driver_param_mode != "physical"
                or getattr(model.driver_encoder, "driver_param_mode", None) != "physical"):
            raise ValueError(f"{name}: driver_param_mode is not physical after reload")
        try:
            centre_phys = centre_physical(saved, model.physics.speed_limit_mps)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from None
        saved_prior = loaded.legacy_config.get("driver_prior_weight", 0.0)
        if not finite(saved_prior) or float(saved_prior) != 0.0:
            raise ValueError(f"{name}: saved driver_prior_weight {saved_prior!r} != 0.0 (prior must be off)")
        head_module = model.follower_courtesy
        if head_module is None or model.follower_courtesy_rollout_grad:
            raise ValueError(f"{name}: courtesy head missing or rollout_grad on after reload")
        if int(getattr(head_module, "version", 1)) != 2:
            raise ValueError(f"{name}: courtesy head is not version 2")
        if not any(key.startswith("merge_pivot.") for key in weights):
            raise ValueError(f"{name}: no merge_pivot weights")
        if model.spec.merge_pivot_state_st is not True or not model.physics.ramp_lane_end_stop:
            raise ValueError(f"{name}: state ST / lane-end stop not enabled")
        if (loaded.legacy_config.get("batch_size") != 4
                or tuple(loaded.legacy_config.get("batch_size_per_stage") or ()) != (4,)):
            raise ValueError(f"{name}: the single smoke stage must use batch 4")
        if stage is not None:
            steps = [row["step"] for row in rows if row.get("stage_s") == stage]
            if not steps or payload.get("stage_s") != stage or payload.get("step") != max(steps):
                raise ValueError(f"{name}: checkpoint stage/step mismatch")
        elif payload.get("history") != rows:
            raise ValueError(f"{name}: checkpoint history does not match smoke log")
        if set(saved) - set(NEW_KEYS) != set(before) or not set(NEW_KEYS) <= set(saved) or set(before) & set(NEW_KEYS):
            raise ValueError(f"{name}: tensor set differs from init + {list(NEW_KEYS)}")
        changed = [key for key in before if not torch.equal(saved[key], before[key])]
        bad = [key for key in changed if key.startswith(frozen) or key.startswith("merge_pivot")]
        if bad:
            raise ValueError(f"{name}: tensors that must stay frozen changed: {sorted(bad)[:5]}")
        outside = [key for key in changed if not key.startswith(TRAINABLE_PREFIXES_12C)]
        if outside:
            raise ValueError(f"{name}: tensors changed outside the driver encoder / physics centre: {sorted(outside)}")
        acc_w, acc_b = saved[ACCEL_HEAD + ".weight"], saved[ACCEL_HEAD + ".bias"]
        if float(acc_w[0].abs().sum() + acc_b[0].abs()) != 0.0:
            raise ValueError(f"{name}: {ACCEL_HEAD} row 0 must stay exactly 0 (residual emission off, net frozen)")
        if float(saved[HEAD_LAST_WEIGHT].abs().sum()) != 0.0:
            raise ValueError(f"{name}: courtesy head must stay exactly 0 (offset off, head frozen)")
        drv_w, drv_b = saved[DRIVER_HEAD + ".weight"], saved[DRIVER_HEAD + ".bias"]
        rows_moved = [bool(drv_w[r].abs().sum() + drv_b[r].abs() > 0) for r in DRIVER_ROWS]
        centre = saved["driver_encoder.centre_logits"]
        centre_moved = centre.shape != centre_init.shape or not torch.equal(centre.float(), centre_init)
        if stage is None:
            if not centre_moved:
                raise ValueError(f"{name}: driver_encoder.centre_logits equals the init centre (core never trained)")
            if not all(rows_moved):
                raise ValueError(f"{name}: {DRIVER_HEAD} rows {[r for r, m in zip(DRIVER_ROWS, rows_moved) if not m]} "
                                 "still exactly zero (centre-initialised rows never trained)")
        reports.append({"file": name, "sha256": loaded.sha256, "reloaded": True, "changed_tensors": len(changed),
                        "centre_moved": centre_moved, "driver_head_rows_moved": rows_moved,
                        "residual_accel_row0_zero": True, "courtesy_head_zero": True,
                        "centre_logits": [float(x) for x in centre], "centre_physical": centre_phys,
                        "driver_prior_weight": saved_prior})
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--smoke-log", type=Path, required=True)
    parser.add_argument("--train-exit-code", type=int, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expect-prior", choices=tuple(PRIOR_MODES), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    train_log = args.smoke_dir / "train_log.jsonl"
    try:
        bounds = physics_accel_bounds(args.config)
        weight = prior_weight(args.config, args.expect_prior)
        from wms.config.load import load_train_config

        freeze = load_train_config(args.config).optim.freeze_prefixes
    except Exception as exc:
        print(f"smoke: bad config {args.config}: {exc}")
        return 3
    result = check_log(train_log, args.smoke_log, train_exit_code=args.train_exit_code, accel_bounds=bounds,
                       prior_weight=weight, freeze_prefixes=freeze)
    result["prior"] = {"mode": args.expect_prior, "effective_weight": weight,
                       "override": PRIOR_MODES[args.expect_prior]}
    if result["ok"]:
        try:
            import torch

            from wms.model.checkpoint import sha256_file

            init = json.loads((args.smoke_dir / "INIT_FROM.json").read_text(encoding="utf-8"))
            init_keys = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)["model"].keys()
            check_init_report(init, init_keys=init_keys, init_sha256=sha256_file(args.init_checkpoint))
            result["checkpoints"] = check_checkpoints(args.smoke_dir, read_rows(train_log),
                                                      args.init_checkpoint, freeze, init)
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"checkpoint/init check failed: {exc}")
    try:
        with args.report.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        print(f"smoke: bad report: {exc}")
        return 3
    print("smoke: ok (round 12C)" if result["ok"] else "smoke: bad " + "; ".join(result["errors"]))
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
