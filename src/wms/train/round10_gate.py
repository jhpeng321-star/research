"""Fail-closed round-10 smoke verdict (round 9 plus fix B), shared by the launcher and tests.

Differences from the round-9 verdict: merge_pivot is FROZEN bitwise in round 10 (no pivot gradient is required),
the cut-in reaction loss must have recorded events and finite parts, and the only tensors that may change are the
temporal emitters and the two gate scalars.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .round9_gate import COUNTERS, OBJECTIVE_FLAG, POSITIVE, finite, read_rows

CUTIN_FINITE = ("cutin_reaction", "cutin_reaction_event_loss", "cutin_reaction_control_loss")
CUTIN_COUNTS = ("cutin_reaction_n_events", "cutin_reaction_n_event_rows", "cutin_reaction_n_control_rows")
TRAINABLE_PREFIXES = ("temporal", "gate_g_max_logit", "gate_tau_raw")
#: merge_pivot is frozen in round 10, so its gradient norm is exactly zero and is not required to be positive
ROUND10_COUNTERS = tuple(key for key in COUNTERS if key != "merge_pivot_grad_norm")
REQUIRED = ("step", "loss", "stage_s", OBJECTIVE_FLAG, *POSITIVE, *ROUND10_COUNTERS, *CUTIN_FINITE, *CUTIN_COUNTS)


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
        for key in REQUIRED:
            value = row.get(key)
            if not finite(value):
                errors.append(f"row {index}: {key} missing/null/non-finite/non-numeric")
                valid = False
            elif key in (*POSITIVE, "stage_s") and value <= 0:
                errors.append(f"row {index}: {key} must be positive")
            elif key in (*ROUND10_COUNTERS, *CUTIN_FINITE, *CUTIN_COUNTS) and value < 0:
                errors.append(f"row {index}: {key} must be nonnegative")
        if finite(row.get(OBJECTIVE_FLAG)) and row[OBJECTIVE_FLAG] != 1:
            errors.append(f"row {index}: {OBJECTIVE_FLAG} must be 1 (on-policy objective retired)")
        step = row.get("step")
        if type(step) is not int or step != index:
            errors.append(f"row {index}: step must be consecutive integer {index}")
    if valid and rows:
        for key in (*ROUND10_COUNTERS, *CUTIN_COUNTS):
            metrics[key] = sum(row[key] for row in rows)
            if not finite(metrics[key]):
                errors.append(f"non-finite total {key}")
        for key in ("merge_pivot_tf_positive_mass_events", "merge_pivot_tf_scored_cars",
                    "cutin_reaction_n_events", "cutin_reaction_n_event_rows", "cutin_reaction_n_control_rows"):
            if metrics[key] <= 0:
                errors.append(f"total {key} must be >0")
        if metrics["merge_pivot_tf_zero_mass_events"] != 0:
            errors.append("total merge_pivot_tf_zero_mass_events must be zero")
        if sum(row["cutin_reaction"] for row in rows) <= 0:
            errors.append("cut-in reaction loss must be positive on at least one step")
        for stage in (120, 300):
            if sum(row["stage_s"] == stage for row in rows) < 2:
                errors.append(f"need at least 2 optimizer steps at stage_s={stage}")
            scored = sum(row["merge_pivot_tf_scored_seconds"] for row in rows if row["stage_s"] >= stage)
            metrics[f"scored_s_{stage}"] = scored
            if not finite(scored) or scored <= 0:
                errors.append(f"scored seconds at stage_s>={stage} must be finite and >0")
        metrics["cuda_peak_mb"] = max(row["cuda_peak_mb"] for row in rows)
        metrics["cutin_reaction_last"] = rows[-1]["cutin_reaction"]
    return {"ok": not errors, "errors": errors, "metrics": metrics}


def check_checkpoints(smoke_dir: Path, rows: list[dict], init_checkpoint: Path, freeze_prefixes) -> list[dict]:
    """Reload each saved model (strict, CPU) and compare it with the round-9 init checkpoint.

    Frozen tensors and every merge_pivot tensor must be bitwise equal to the init; only temporal.* and the two
    gate scalars may differ, and at least one of them must have moved in the final checkpoint.
    """
    import torch

    from wms.model.checkpoint import load_checkpoint

    before = torch.load(init_checkpoint, map_location="cpu", weights_only=True)["model"]
    frozen = tuple(freeze_prefixes)
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
        if saved.keys() != before.keys():
            raise ValueError(f"{name}: tensor set differs from the init checkpoint")
        changed = [key for key in saved if not torch.equal(saved[key], before[key])]
        bad = [key for key in changed if key.startswith(frozen) or key.startswith("merge_pivot")]
        # Informational: changed tensors outside the intended trainable set that no freeze prefix covers.
        other = [key for key in changed if not key.startswith(TRAINABLE_PREFIXES) and key not in bad]
        if bad:
            raise ValueError(f"{name}: tensors that must stay frozen changed: {sorted(bad)[:5]}")
        if stage is None and not changed:
            raise ValueError(f"{name}: no trainable tensor moved")
        reports.append({"file": name, "sha256": loaded.sha256, "reloaded": True, "changed_tensors": len(changed),
                        "changed_outside_trainable_set": sorted(other)})
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--smoke-log", type=Path, required=True)
    parser.add_argument("--train-exit-code", type=int, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    train_log = args.smoke_dir / "train_log.jsonl"
    result = check_log(train_log, args.smoke_log, train_exit_code=args.train_exit_code)
    if result["ok"]:
        try:
            from wms.config.load import load_train_config

            freeze = load_train_config(args.config).optim.freeze_prefixes
            init = json.loads((args.smoke_dir / "INIT_FROM.json").read_text(encoding="utf-8"))
            if init["new_in_model"] or init["shape_mismatch"] or init["unused_from_checkpoint"]:
                raise ValueError(f"init report not clean: {init}")
            result["checkpoints"] = check_checkpoints(args.smoke_dir, read_rows(train_log),
                                                     args.init_checkpoint, freeze)
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
