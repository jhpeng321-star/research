"""Fail-closed round-11 smoke verdict (round 10b + the structural courtesy fix), shared by the launcher and tests.

On top of the round-10b verdict (round-10 merge/fix-B checks; courtesy loss finite/positive with rows):

* the R11 contrast term ran: ``follower_courtesy_contrast_n_controls`` finite and > 0 in total, contrast shift finite;
* fix B ran with the partnered-control exclusion (``cutin_reaction_n_partnered_controls_excluded`` finite, >= 0);
* the init report is clean: no shape mismatch / unused tensor; the only tensor that may be ``expanded`` is the head's
  first layer (round-10b 10-input head -> version-2 13-input head); the only new tensors are head tensors, and the
  recorded ``follower_courtesy_init`` mode equals the launcher's expectation (``checkpoint`` = round-10b head, the
  default; ``zero`` = fresh zero-init head);
* every saved checkpoint reloads strictly with a VERSION-2 head and rollout_grad off; frozen tensors and merge_pivot are
  bitwise equal to the init; only temporal.*, the two gate scalars and the head may differ; the head moved relative to
  its (zero-expanded) init.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .round9_gate import finite, read_rows
from .round10b_gate import HEAD_PREFIX, TRAINABLE_PREFIXES
from .round10b_gate import check_log as round10b_check_log

HEAD_FIRST_WEIGHT = HEAD_PREFIX + "net.0.weight"
CONTRAST_FINITE = ("follower_courtesy_contrast_n_controls", "follower_courtesy_contrast_shift_mean")
EXCLUDED_KEY = "cutin_reaction_n_partnered_controls_excluded"
INIT_MODES = ("checkpoint", "zero")


def check_log(train_log: Path, smoke_log: Path, *, train_exit_code: int = 0) -> dict:
    result = round10b_check_log(train_log, smoke_log, train_exit_code=train_exit_code)
    errors, metrics = result["errors"], result["metrics"]
    try:
        rows = read_rows(train_log)
    except (OSError, ValueError):
        return result
    valid = True
    for index, row in enumerate(rows, 1):
        # A step without courtesy rows has no contrast stats (row count 0 is checked by the round-10b part).
        if row.get("follower_courtesy_n_rows"):
            for key in CONTRAST_FINITE:
                if not finite(row.get(key)):
                    errors.append(f"row {index}: {key} missing/null/non-finite/non-numeric")
                    valid = False
        value = row.get(EXCLUDED_KEY)
        if not finite(value) or value < 0:
            errors.append(f"row {index}: {EXCLUDED_KEY} missing/non-finite/negative")
            valid = False
    if valid and rows:
        metrics["contrast_n_controls"] = sum(int(row.get("follower_courtesy_contrast_n_controls") or 0)
                                             for row in rows)
        metrics["partnered_controls_excluded"] = sum(int(row[EXCLUDED_KEY]) for row in rows)
        if metrics["contrast_n_controls"] <= 0:
            errors.append("total follower_courtesy_contrast_n_controls must be >0")
    result["ok"] = not errors
    return result


def check_init_report(init: dict, expect: str) -> None:
    if expect not in INIT_MODES:
        raise ValueError(f"expected courtesy init mode must be one of {INIT_MODES}: {expect!r}")
    if init.get("shape_mismatch") or init.get("unused_from_checkpoint"):
        raise ValueError(f"init report not clean: {init}")
    new = list(init.get("new_in_model") or [])
    if not all(key.startswith(HEAD_PREFIX) for key in new):
        raise ValueError(f"init report: new tensors must be courtesy head tensors only: {new}")
    expanded = list(init.get("expanded") or [])
    if any(key != HEAD_FIRST_WEIGHT for key in expanded):
        raise ValueError(f"init report: only the head's first layer may be expanded: {expanded}")
    mode = init.get("follower_courtesy_init")
    if mode != expect:
        raise ValueError(f"init report: follower_courtesy_init {mode!r} != expected {expect!r}")
    if expect == "zero" and not new:
        raise ValueError("init report: zero-init head requested but the head was loaded")
    if expect == "checkpoint" and (new or expanded != [HEAD_FIRST_WEIGHT]):
        raise ValueError("init report: round-10b head must be loaded and zero-expanded to version 2")


def check_checkpoints(smoke_dir: Path, rows: list[dict], init_checkpoint: Path, freeze_prefixes) -> list[dict]:
    """Reload each saved model strictly (CPU) and compare it with the (zero-expanded) init checkpoint."""
    import torch

    from wms.model.checkpoint import load_checkpoint
    from wms.model.follower_courtesy import expand_head_inputs

    raw = torch.load(init_checkpoint, map_location="cpu", weights_only=True)["model"]
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
        head_module = loaded.model.follower_courtesy
        if head_module is None or loaded.model.follower_courtesy_rollout_grad:
            raise ValueError(f"{name}: courtesy head missing or rollout_grad on after reload")
        if int(getattr(head_module, "version", 1)) != 2:
            raise ValueError(f"{name}: courtesy head is not version 2")
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
        before = expand_head_inputs(raw, saved)
        head = {key for key in saved if key.startswith(HEAD_PREFIX)}
        init_head = {key for key in before if key.startswith(HEAD_PREFIX)}
        if set(saved) - head != set(before) - init_head or (init_head and init_head != head):
            raise ValueError(f"{name}: tensor set differs from init + courtesy head")
        changed = [key for key in before if key not in head and not torch.equal(saved[key], before[key])]
        bad = [key for key in changed if key.startswith(frozen) or key.startswith("merge_pivot")]
        other = [key for key in changed if not key.startswith(TRAINABLE_PREFIXES) and key not in bad]
        if bad:
            raise ValueError(f"{name}: tensors that must stay frozen changed: {sorted(bad)[:5]}")
        if init_head:
            head_moved = any(not torch.equal(saved[key], before[key]) for key in head)
        else:
            head_moved = bool(saved[HEAD_PREFIX + "net.4.weight"].abs().sum() > 0)
        if stage is None and not changed:
            raise ValueError(f"{name}: no round-10 trainable tensor moved")
        if stage is None and not head_moved:
            raise ValueError(f"{name}: courtesy head did not move from its init (head never trained)")
        reports.append({"file": name, "sha256": loaded.sha256, "reloaded": True, "changed_tensors": len(changed),
                        "head_moved": head_moved, "head_from_init": bool(init_head),
                        "changed_outside_trainable_set": sorted(other)})
    return reports


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-dir", type=Path, required=True)
    parser.add_argument("--smoke-log", type=Path, required=True)
    parser.add_argument("--train-exit-code", type=int, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expect-courtesy-init", choices=INIT_MODES, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    train_log = args.smoke_dir / "train_log.jsonl"
    result = check_log(train_log, args.smoke_log, train_exit_code=args.train_exit_code)
    if result["ok"]:
        try:
            from wms.config.load import load_train_config

            freeze = load_train_config(args.config).optim.freeze_prefixes
            check_init_report(json.loads((args.smoke_dir / "INIT_FROM.json").read_text(encoding="utf-8")),
                              args.expect_courtesy_init)
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
