"""Fail-closed round-10b smoke verdict (round 10 + the learned follower courtesy), shared by the launcher and tests.

On top of the round-10 verdict (round-9 merge checks with merge_pivot frozen, fix-B cut-in census and finite parts):

* the courtesy loss is finite on every row, positive on at least one, and has courtesy rows (``follower_courtesy_n_rows``);
* the init report is clean except for the NEW zero-initialised courtesy head tensors (they are absent from round 9);
* every saved checkpoint reloads strictly with the head enabled; frozen tensors and merge_pivot are bitwise equal to the
  round-9 init; only temporal.*, the two gate scalars and the head may differ; the head's zero last layer has moved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .round9_gate import finite, read_rows
from .round10_gate import TRAINABLE_PREFIXES as ROUND10_TRAINABLE
from .round10_gate import check_log as round10_check_log

HEAD_PREFIX = "follower_courtesy."
COURTESY_FINITE = ("follower_courtesy", "follower_courtesy_loss_raw")
COURTESY_COUNTS = ("follower_courtesy_n_rows",)
TRAINABLE_PREFIXES = (*ROUND10_TRAINABLE, HEAD_PREFIX)
#: the zero-initialised read-out layer of the head (FollowerCourtesyHead.net[-1])
HEAD_LAST_WEIGHT = HEAD_PREFIX + "net.4.weight"


def check_log(train_log: Path, smoke_log: Path, *, train_exit_code: int = 0) -> dict:
    result = round10_check_log(train_log, smoke_log, train_exit_code=train_exit_code)
    errors, metrics = result["errors"], result["metrics"]
    try:
        rows = read_rows(train_log)
    except (OSError, ValueError):
        return result
    valid = True
    for index, row in enumerate(rows, 1):
        for key in (*COURTESY_FINITE, *COURTESY_COUNTS):
            value = row.get(key)
            if not finite(value):
                errors.append(f"row {index}: {key} missing/null/non-finite/non-numeric")
                valid = False
            elif value < 0:
                errors.append(f"row {index}: {key} must be nonnegative")
    if valid and rows:
        metrics["follower_courtesy_n_rows"] = sum(row["follower_courtesy_n_rows"] for row in rows)
        metrics["follower_courtesy_shared_replay"] = sum(int(row.get("follower_courtesy_shared_replay") or 0)
                                                         for row in rows)
        metrics["follower_courtesy_last"] = rows[-1]["follower_courtesy"]
        if metrics["follower_courtesy_n_rows"] <= 0:
            errors.append("total follower_courtesy_n_rows must be >0")
        if sum(row["follower_courtesy"] for row in rows) <= 0:
            errors.append("follower courtesy loss must be positive on at least one step")
    result["ok"] = not errors
    return result


def check_init_report(init: dict) -> None:
    new = init["new_in_model"]
    if not new or not all(key.startswith(HEAD_PREFIX) for key in new):
        raise ValueError(f"init report: new tensors must be exactly the courtesy head: {new}")
    if init["shape_mismatch"] or init["unused_from_checkpoint"]:
        raise ValueError(f"init report not clean: {init}")


def check_checkpoints(smoke_dir: Path, rows: list[dict], init_checkpoint: Path, freeze_prefixes) -> list[dict]:
    """Reload each saved model strictly (CPU) and compare it with the round-9 init checkpoint."""
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
        if loaded.model.follower_courtesy is None or loaded.model.follower_courtesy_rollout_grad:
            raise ValueError(f"{name}: courtesy head missing or rollout_grad on after reload")
        if not any(key.startswith("merge_pivot.") for key in weights):
            raise ValueError(f"{name}: no merge_pivot weights")
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
        head = {key for key in saved if key.startswith(HEAD_PREFIX)}
        if set(saved) - head != set(before) or set(before) & head:
            raise ValueError(f"{name}: tensor set differs from init + courtesy head")
        changed = [key for key in before if not torch.equal(saved[key], before[key])]
        bad = [key for key in changed if key.startswith(frozen) or key.startswith("merge_pivot")]
        other = [key for key in changed if not key.startswith(TRAINABLE_PREFIXES) and key not in bad]
        if bad:
            raise ValueError(f"{name}: tensors that must stay frozen changed: {sorted(bad)[:5]}")
        head_moved = bool(saved[HEAD_LAST_WEIGHT].abs().sum() > 0)
        if stage is None and not changed:
            raise ValueError(f"{name}: no round-10 trainable tensor moved")
        if stage is None and not head_moved:
            raise ValueError(f"{name}: courtesy head read-out is still exactly zero (head never trained)")
        reports.append({"file": name, "sha256": loaded.sha256, "reloaded": True, "changed_tensors": len(changed),
                        "head_moved": head_moved, "changed_outside_trainable_set": sorted(other)})
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
            check_init_report(json.loads((args.smoke_dir / "INIT_FROM.json").read_text(encoding="utf-8")))
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
