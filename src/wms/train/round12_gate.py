"""Fail-closed round-12 smoke verdict (round 9 merge law frozen + base calibration + round-11 courtesy head).

Round 12 = round 9 (merge_pivot FROZEN) + the base calibration (physical driver settings started at the learned
centre, re-initialised residual acceleration read-out, filled replay presence) + the round-11 contrast courtesy head
(zero init); fix B (cut-in reaction loss) is DROPPED.  Checks, all fail-closed:

* log: the round-9/10 merge census with merge_pivot frozen (TF zero-mass events = 0, finite losses, positive
  tau/w/peak memory, consecutive steps, >= 2 steps per stage, scored seconds > 0; the pivot gradient may be zero);
  the round-10b/11 courtesy checks (loss finite/nonnegative, rows > 0 in total, loss > 0 on some step, contrast
  controls > 0 in total, contrast shift finite) and the round-11 head-range rule with the config's plant clamp;
  the cut-in reaction term must be ABSENT (any ``cutin_reaction*`` value other than an exact zero fails);
* R12 log: fix D telemetry ``follower_courtesy_replay_filled_extra_rows`` (integer >= 0) on every step with courtesy
  rows (its total is a metric, not a requirement); driver prior (owner switch ``--expect-prior`` = config / off /
  weak -> effective ``loss.driver_prior.weight`` = config value / 0.0 / 0.01): with weight > 0
  ``driver_prior_means`` on every step with finite accel / decel / time_headway / min_gap / v0_ratio strictly inside
  the physical open intervals; with weight 0 (off) the prior is never computed, so ``driver_prior_means`` and
  ``driver_prior`` must be ABSENT from every step;
* init report: clean, the only new tensors are the driver-encoder centre/bounds and the six courtesy head tensors,
  zero courtesy head, centre init (head rows 0..4 zeroed), residual acceleration row 0 re-initialised, every
  init-checkpoint tensor loaded, source sha256 = the init checkpoint;
* checkpoints (stage 120, stage 300, final): strict reload, weights finite and unchanged, physical driver mode with
  bounds equal to the constants, version-2 head with rollout_grad off, merge_pivot + state ST + lane-end stop, batch 4,
  stage/step/history consistent with the log; vs the init checkpoint every frozen-prefix tensor and every
  merge_pivot tensor is bitwise equal (the config also freezes the two gate scalars ``gate_*``) and the tensor set is
  init + the new tensors; the saved config's ``driver_prior_weight`` equals the effective prior weight; the learned
  centre mapped through physical_lower/upper lies strictly inside the physical open intervals (every prior mode); at
  the final checkpoint the courtesy head, the learned centre, residual acceleration row 0 and driver-encoder head
  rows 0..4 have all moved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .round9_gate import OBJECTIVE_FLAG, POSITIVE, finite, read_rows
from .round10_gate import ROUND10_COUNTERS
from .round10b_gate import COURTESY_COUNTS, COURTESY_FINITE, HEAD_LAST_WEIGHT, HEAD_PREFIX
from .round11_gate import CONTRAST_FINITE, head_range_errors, physics_accel_bounds

CUTIN_PREFIX = "cutin_reaction"
REPLAY_KEY = "follower_courtesy_replay_filled_extra_rows"
PRIOR_KEY = "driver_prior_means"
#: open physical intervals of the logged driver prior means (wms.model.driver_encoder.PHYSICAL_LOWER/UPPER;
#: v0 as a ratio of the speed limit)
PRIOR_BOUNDS = {"v0_ratio": (0.6, 1.4), "time_headway": (0.4, 3.0), "min_gap": (0.5, 5.0),
                "accel": (0.3, 4.0), "decel": (0.8, 6.0)}
DRIVER_NEW = ("driver_encoder.centre_logits", "driver_encoder.physical_lower", "driver_encoder.physical_upper")
HEAD_NEW = tuple(f"{HEAD_PREFIX}net.{i}.{kind}" for i in (0, 2, 4) for kind in ("weight", "bias"))
NEW_KEYS = tuple(sorted((*DRIVER_NEW, *HEAD_NEW)))
DRIVER_HEAD = "driver_encoder.head.2"
DRIVER_ROWS = [0, 1, 2, 3, 4]
ACCEL_HEAD = "residual_net.accel.2"
#: intended trainable set (informational: changes outside it are reported, frozen changes fail).  The two gate
#: scalars (gate_g_max_logit, gate_tau_raw) are FROZEN by the round-12 config (freeze prefix "gate_").
TRAINABLE_PREFIXES = ("driver_encoder.head", "driver_encoder.centre_logits", "residual_net", "temporal",
                      HEAD_PREFIX)
#: R12_PRIOR owner switch -> override of loss.driver_prior.weight (config = no override)
PRIOR_MODES = {"config": None, "off": 0.0, "weak": 0.01}
REQUIRED = ("step", "loss", "stage_s", OBJECTIVE_FLAG, *POSITIVE, *ROUND10_COUNTERS)


def merge_log_errors(rows: list[dict], metrics: dict) -> list[str]:
    """Round-10 merge census WITHOUT the fix-B part (merge_pivot frozen: its gradient norm may be zero)."""
    errors = []
    metrics["optimizer_steps"] = len(rows)
    if len(rows) < 4:
        errors.append("need at least 4 optimizer steps")
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
        for stage in (120, 300):
            if sum(row["stage_s"] == stage for row in rows) < 2:
                errors.append(f"need at least 2 optimizer steps at stage_s={stage}")
            scored = sum(row["merge_pivot_tf_scored_seconds"] for row in rows if row["stage_s"] >= stage)
            metrics[f"scored_s_{stage}"] = scored
            if not finite(scored) or scored <= 0:
                errors.append(f"scored seconds at stage_s>={stage} must be finite and >0")
        metrics["cuda_peak_mb"] = max(row["cuda_peak_mb"] for row in rows)
    return errors


def courtesy_log_errors(rows: list[dict], metrics: dict) -> list[str]:
    """Round-10b courtesy loss checks + round-11 contrast checks."""
    errors = []
    valid = True
    for index, row in enumerate(rows, 1):
        for key in (*COURTESY_FINITE, *COURTESY_COUNTS):
            value = row.get(key)
            if not finite(value):
                errors.append(f"row {index}: {key} missing/null/non-finite/non-numeric")
                valid = False
            elif value < 0:
                errors.append(f"row {index}: {key} must be nonnegative")
        if row.get("follower_courtesy_n_rows"):
            for key in CONTRAST_FINITE:
                if not finite(row.get(key)):
                    errors.append(f"row {index}: {key} missing/null/non-finite/non-numeric")
                    valid = False
    if valid and rows:
        metrics["follower_courtesy_n_rows"] = sum(row["follower_courtesy_n_rows"] for row in rows)
        metrics["follower_courtesy_shared_replay"] = sum(int(row.get("follower_courtesy_shared_replay") or 0)
                                                         for row in rows)
        metrics["follower_courtesy_last"] = rows[-1]["follower_courtesy"]
        metrics["contrast_n_controls"] = sum(int(row.get("follower_courtesy_contrast_n_controls") or 0)
                                             for row in rows)
        if metrics["follower_courtesy_n_rows"] <= 0:
            errors.append("total follower_courtesy_n_rows must be >0")
        if sum(row["follower_courtesy"] for row in rows) <= 0:
            errors.append("follower courtesy loss must be positive on at least one step")
        if metrics["contrast_n_controls"] <= 0:
            errors.append("total follower_courtesy_contrast_n_controls must be >0")
    return errors


def cutin_absent_errors(rows: list[dict]) -> list[str]:
    """Fix B is dropped: no row may carry a cut-in reaction loss or event count other than an exact zero."""
    errors = []
    for index, row in enumerate(rows, 1):
        for key, value in row.items():
            if key.startswith(CUTIN_PREFIX) and not (finite(value) and value == 0):
                errors.append(f"row {index}: {key} = {value!r} (cut-in reaction term must be absent)")
    return errors


def replay_errors(rows: list[dict], metrics: dict) -> list[str]:
    """Fix D telemetry on every step with courtesy rows (its total may be zero: reported only)."""
    errors, total, valid = [], 0, True
    for index, row in enumerate(rows, 1):
        if not row.get("follower_courtesy_n_rows"):
            continue
        value = row.get(REPLAY_KEY)
        if not finite(value) or value < 0 or value != int(value):
            errors.append(f"row {index}: {REPLAY_KEY} missing/non-finite/negative/non-integer (fix D inactive)")
            valid = False
        else:
            total += int(value)
    metrics["replay_filled_extra_rows_total"] = total if valid else None
    return errors


def prior_weight(config: Path, mode: str) -> float:
    """Effective ``loss.driver_prior.weight`` of the run: the config value with the R12_PRIOR override applied."""
    from wms.config.load import load_train_config

    if mode not in PRIOR_MODES:
        raise ValueError(f"prior mode must be one of {tuple(PRIOR_MODES)}: {mode!r}")
    override = PRIOR_MODES[mode]
    overrides = () if override is None else (f"loss.driver_prior.weight={override}",)
    return float(load_train_config(config, overrides).loss.driver_prior.weight)


def prior_absent_errors(rows: list[dict]) -> list[str]:
    """Prior off (weight 0): the prior is never computed, so no step may log it."""
    return [f"row {index}: {key} logged although the driver prior is off (weight 0)"
            for index, row in enumerate(rows, 1) for key in (PRIOR_KEY, "driver_prior") if key in row]


def prior_errors(rows: list[dict], metrics: dict) -> list[str]:
    """``driver_prior_means`` on every step, every mean finite and strictly inside its physical interval."""
    errors = []
    for index, row in enumerate(rows, 1):
        means = row.get(PRIOR_KEY)
        if not isinstance(means, dict):
            errors.append(f"row {index}: {PRIOR_KEY} missing or not an object")
            continue
        for key, (low, high) in PRIOR_BOUNDS.items():
            value = means.get(key)
            if not finite(value):
                errors.append(f"row {index}: {PRIOR_KEY}.{key} missing/non-finite")
            elif not low < value < high:
                errors.append(f"row {index}: {PRIOR_KEY}.{key} {value!r} outside ({low}, {high})")
    if rows:
        for label, row in (("first", rows[0]), ("last", rows[-1])):
            means = row.get(PRIOR_KEY)
            metrics[f"driver_prior_means_{label}"] = (
                {key: (means.get(key) if finite(means.get(key)) else None) for key in PRIOR_BOUNDS}
                if isinstance(means, dict) else None)
    return errors


def check_log(train_log: Path, smoke_log: Path, *, train_exit_code: int = 0,
              accel_bounds: tuple[float, float] | None = None, prior_weight: float | None = None) -> dict:
    """``accel_bounds`` = the plant clamp (default: :class:`PhysicsConfig` defaults; the CLI reads the config's).

    ``prior_weight`` = the effective driver prior weight (None = the round-12 default, a positive weight): > 0
    requires ``driver_prior_means`` on every step, 0 requires the prior telemetry to be absent.
    """
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
    errors.extend(merge_log_errors(rows, metrics))
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
        errors.extend(prior_errors(rows, metrics))
    return {"ok": not errors, "errors": errors, "metrics": metrics}


def check_init_report(init: dict, *, init_keys=None, init_sha256: str | None = None) -> None:
    """Round-12 INIT_FROM.json contract (round-9 checkpoint -> physical encoder + zero courtesy head)."""
    if init.get("shape_mismatch") or init.get("unused_from_checkpoint") or init.get("expanded"):
        raise ValueError(f"init report not clean (shape_mismatch/unused/expanded): "
                         f"{init.get('shape_mismatch')} {init.get('unused_from_checkpoint')} {init.get('expanded')}")
    for key in ("shape_mismatch", "unused_from_checkpoint", "expanded", "new_in_model", "loaded"):
        if not isinstance(init.get(key), list):
            raise ValueError(f"init report: {key} missing or not a list")
    if sorted(init["new_in_model"]) != list(NEW_KEYS):
        raise ValueError(f"init report: new_in_model {sorted(init['new_in_model'])} != {list(NEW_KEYS)}")
    if init.get("follower_courtesy_init") != "zero":
        raise ValueError(f"init report: follower_courtesy_init {init.get('follower_courtesy_init')!r} != 'zero'")
    centre = init.get("driver_param_init")
    if (not isinstance(centre, dict) or centre.get("mode") != "centre" or centre.get("zeroed_rows") != DRIVER_ROWS
            or centre.get("tensor") != DRIVER_HEAD):
        raise ValueError(f"init report: driver_param_init must be centre with zeroed rows 0..4: {centre}")
    logits = centre.get("centre_logits")
    if not isinstance(logits, list) or len(logits) != len(DRIVER_ROWS) or not all(finite(x) for x in logits):
        raise ValueError(f"init report: driver_param_init.centre_logits must be 5 finite floats: {logits}")
    accel = init.get("residual_accel_reinit")
    if not isinstance(accel, dict) or accel.get("zeroed_rows") != [0] or accel.get("tensor") != ACCEL_HEAD:
        raise ValueError(f"init report: residual_accel_reinit must zero {ACCEL_HEAD} row 0: {accel}")
    if init_keys is not None and sorted(init["loaded"]) != sorted(init_keys):
        missing = sorted(set(init_keys) - set(init["loaded"]))
        raise ValueError(f"init report: loaded tensors != init checkpoint tensors (missing {missing[:5]})")
    if init_sha256 is not None and init.get("sha256") != init_sha256:
        raise ValueError(f"init report: source sha256 {init.get('sha256')} != init checkpoint {init_sha256}")


def _physical_bounds(physics):
    import torch

    from wms.model.driver_encoder import PHYSICAL_LOWER, PHYSICAL_UPPER

    lower, upper = torch.tensor(PHYSICAL_LOWER), torch.tensor(PHYSICAL_UPPER)
    lower[0] *= physics.speed_limit_mps
    upper[0] *= physics.speed_limit_mps
    return lower, upper


def centre_physical(saved: dict, speed_limit_mps: float) -> dict:
    """The learned centre mapped through physical_lower/upper (PRIOR_BOUNDS keys, v0 as a speed-limit ratio).

    Raises unless every value is finite and strictly inside its physical open interval (all prior modes).
    """
    import torch

    lower, upper = saved["driver_encoder.physical_lower"], saved["driver_encoder.physical_upper"]
    values = lower + (upper - lower) * torch.sigmoid(saved["driver_encoder.centre_logits"].to(lower))
    if not (bool(torch.isfinite(values).all()) and bool((values > lower).all()) and bool((values < upper).all())):
        raise ValueError(f"learned centre {values.tolist()} not strictly inside the physical intervals "
                         f"({lower.tolist()}, {upper.tolist()})")
    v0, t, s0, a, b = (float(x) for x in values)
    return {"v0_ratio": v0 / float(speed_limit_mps), "time_headway": t, "min_gap": s0, "accel": a, "decel": b}


def check_checkpoints(smoke_dir: Path, rows: list[dict], init_checkpoint: Path, freeze_prefixes,
                      init_report: dict, prior_weight: float | None = None) -> list[dict]:
    """Reload each saved model strictly (CPU) and compare it with the round-9 init checkpoint.

    ``prior_weight`` (when given) must equal the saved config's ``driver_prior_weight`` (absent = 0).
    """
    import torch

    from wms.model.checkpoint import load_checkpoint

    before = torch.load(init_checkpoint, map_location="cpu", weights_only=True)["model"]
    frozen = tuple(freeze_prefixes)
    centre_init = torch.tensor(init_report["driver_param_init"]["centre_logits"], dtype=torch.float32)
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
        model = loaded.model
        if (model.spec.driver_param_mode != "physical"
                or getattr(model.driver_encoder, "driver_param_mode", None) != "physical"):
            raise ValueError(f"{name}: driver_param_mode is not physical after reload")
        lower, upper = _physical_bounds(model.physics)
        if not (torch.equal(saved["driver_encoder.physical_lower"], lower)
                and torch.equal(saved["driver_encoder.physical_upper"], upper)):
            raise ValueError(f"{name}: physical_lower/upper differ from the PHYSICAL_LOWER/UPPER constants")
        try:
            centre_phys = centre_physical(saved, model.physics.speed_limit_mps)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from None
        saved_prior = loaded.legacy_config.get("driver_prior_weight", 0.0)
        if prior_weight is not None and (not finite(saved_prior) or float(saved_prior) != float(prior_weight)):
            raise ValueError(f"{name}: saved driver_prior_weight {saved_prior!r} != effective {prior_weight!r}")
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
                or tuple(loaded.legacy_config.get("batch_size_per_stage") or ()) != (4, 4)):
            raise ValueError(f"{name}: both smoke stages must use batch 4")
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
        other = [key for key in changed if not key.startswith(TRAINABLE_PREFIXES)]
        drv_w, drv_b = saved[DRIVER_HEAD + ".weight"], saved[DRIVER_HEAD + ".bias"]
        rows_moved = [bool(drv_w[r].abs().sum() + drv_b[r].abs() > 0) for r in DRIVER_ROWS]
        acc_w, acc_b = saved[ACCEL_HEAD + ".weight"], saved[ACCEL_HEAD + ".bias"]
        accel_moved = bool(acc_w[0].abs().sum() + acc_b[0].abs() > 0)
        head_moved = bool(saved[HEAD_LAST_WEIGHT].abs().sum() > 0)
        centre = saved["driver_encoder.centre_logits"]
        centre_moved = centre.shape != centre_init.shape or not torch.equal(centre.float(), centre_init)
        if stage is None:
            if not head_moved:
                raise ValueError(f"{name}: courtesy head read-out is still exactly zero (head never trained)")
            if not centre_moved:
                raise ValueError(f"{name}: driver_encoder.centre_logits equals the init centre (centre never trained)")
            if not accel_moved:
                raise ValueError(f"{name}: {ACCEL_HEAD} row 0 is still exactly zero (re-initialised head never trained)")
            if not all(rows_moved):
                raise ValueError(f"{name}: {DRIVER_HEAD} rows {[r for r, m in zip(DRIVER_ROWS, rows_moved) if not m]} "
                                 "still exactly zero (centre-initialised rows never trained)")
        reports.append({"file": name, "sha256": loaded.sha256, "reloaded": True, "changed_tensors": len(changed),
                        "head_moved": head_moved, "centre_moved": centre_moved, "accel_row0_moved": accel_moved,
                        "driver_head_rows_moved": rows_moved,
                        "centre_logits": [float(x) for x in centre], "centre_physical": centre_phys,
                        "driver_prior_weight": saved_prior,
                        "changed_outside_trainable_set": sorted(other)})
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
    except Exception as exc:
        print(f"smoke: bad config {args.config}: {exc}")
        return 3
    result = check_log(train_log, args.smoke_log, train_exit_code=args.train_exit_code, accel_bounds=bounds,
                       prior_weight=weight)
    result["prior"] = {"mode": args.expect_prior, "effective_weight": weight,
                       "override": PRIOR_MODES[args.expect_prior]}
    if result["ok"]:
        try:
            import torch

            from wms.config.load import load_train_config
            from wms.model.checkpoint import sha256_file

            freeze = load_train_config(args.config).optim.freeze_prefixes
            init = json.loads((args.smoke_dir / "INIT_FROM.json").read_text(encoding="utf-8"))
            init_keys = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)["model"].keys()
            check_init_report(init, init_keys=init_keys, init_sha256=sha256_file(args.init_checkpoint))
            result["checkpoints"] = check_checkpoints(args.smoke_dir, read_rows(train_log),
                                                     args.init_checkpoint, freeze, init, prior_weight=weight)
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
    print("smoke: ok" if result["ok"] else "smoke: bad " + "; ".join(result["errors"]))
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
