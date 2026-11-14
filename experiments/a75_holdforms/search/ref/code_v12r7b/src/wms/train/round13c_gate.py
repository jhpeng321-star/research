"""Fail-closed S-R13C gate: R12 objective census, as-is frozen calibrated core."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from . import round12_gate

TRAINABLE = ("residual_net.", "follower_courtesy.")


def optimizer_errors(text, expected):
    match = re.search(r"^WM-S optimizer groups: (\{.*\})\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError("optimizer trainable list missing")
    report = json.loads(match.group(1))
    names = report.get("trainable_tensor_names", [])
    if sorted(names) != sorted(expected) or not names:
        raise ValueError("optimizer trainable list differs from configured model")
    if any(not n.startswith(TRAINABLE) or n.startswith("residual_net.lane") for n in names):
        raise ValueError("trainable list includes core/physics/pivot/temporal/lane tensors")
    if not any(n.startswith("residual_net.") for n in names) or not any(
            n.startswith("follower_courtesy.") for n in names):
        raise ValueError("residual or courtesy path missing from optimizer")
    if report.get("unmatched_freeze_prefixes"):
        raise ValueError("unmatched freeze prefixes")
    return names


def check_artifacts(args):
    import torch
    from wms.config.load import load_train_config
    from wms.model.checkpoint import load_checkpoint, sha256_file
    from wms.train.round13f_gate import check_r13f_checkpoints

    cfg = load_train_config(args.config)
    if cfg.model.driver_param_init != "checkpoint" or cfg.model.residual_accel_reinit:
        raise ValueError("round-12 init transforms must be OFF")
    before = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)["model"]
    init = json.loads((args.smoke_dir / "INIT_FROM.json").read_text())
    courtesy = sorted(n for n in before if n.startswith("follower_courtesy."))
    if (init.get("sha256") != sha256_file(args.init_checkpoint)
            or sorted(init.get("new_in_model", [])) != courtesy
            or sorted(init.get("loaded", [])) != sorted(set(before) - set(courtesy))
            or init.get("follower_courtesy_init") != "zero"
            or any(init.get(k) for k in ("expanded", "shape_mismatch", "unused_from_checkpoint",
                                         "driver_param_init", "residual_accel_reinit", "temporal_accel_reinit"))):
        raise ValueError("init is not as-is S-R12C + fresh zero courtesy")
    reports = []
    rows = round12_gate.read_rows(args.smoke_dir / "train_log.jsonl")
    for name, stage in (("wm_s_stage120.pt", 120), ("wm_s_stage300.pt", 300), ("wm_s.pt", None)):
        path = args.smoke_dir / name
        payload = torch.load(path, map_location="cpu", weights_only=True)
        loaded = load_checkpoint(path, device="cpu", strict=True)
        model, saved = loaded.model, payload["model"]
        weights = model.state_dict()
        if set(weights) != set(before) or any(not torch.isfinite(v).all()
                or not torch.equal(v, saved[n]) for n, v in weights.items()):
            raise ValueError(f"{name}: strict reload changed weights or tensor set")
        changed = [n for n in before if not torch.equal(before[n], saved[n])]
        if any(not n.startswith(TRAINABLE) or n.startswith(tuple(cfg.optim.freeze_prefixes))
               for n in changed):
            raise ValueError(f"{name}: frozen core/physics/other weights changed")
        expected = [n for n, _ in model.named_parameters()
                    if not n.startswith(tuple(cfg.optim.freeze_prefixes))]
        names = optimizer_errors(args.smoke_log.read_text(), expected)
        if (model.spec.residual_accel_off or model.spec.follower_courtesy_off
                or model.spec.driver_param_mode != "physical" or not model.spec.merge_pivot_state_st
                or not model.physics.ramp_lane_end_stop or loaded.legacy_config.get("driver_prior_weight", 0) != 0
                or tuple(loaded.legacy_config.get("batch_size_per_stage", ())) != (4, 4)):
            raise ValueError(f"{name}: model contract mismatch")
        if stage is None:
            if payload.get("history") != rows:
                raise ValueError("final history mismatch")
            for prefix in ("residual_net.accel.", "follower_courtesy."):
                if not any(n.startswith(prefix) for n in changed):
                    raise ValueError(f"{prefix} never trained")
        elif payload.get("stage_s") != stage or payload.get("step") != max(
                r["step"] for r in rows if r["stage_s"] == stage):
            raise ValueError(f"{name}: stage/step mismatch")
        # Exercise the actual emission branch, not merely the serialized switch.
        kin = torch.zeros(1, 3, 7)
        kin[..., 0] = torch.tensor([0., 100., 40.])
        kin[..., 2] = 1.0
        kin[..., 4] = 10.0
        kin[..., 6] = 5.0
        state = {"veh_kin": kin, "present": torch.ones(1, 3, dtype=torch.bool),
                 "is_cav": torch.zeros(1, 3, dtype=torch.bool)}
        model.diag_emit_temporal = True
        from wms.model.history import LONG_HISTORY_F
        history = torch.zeros(1, 3, model.driver_history_len, LONG_HISTORY_F)
        history_mask = torch.zeros(history.shape[:-1], dtype=torch.bool)
        with torch.no_grad():
            out = model(state, k_steps=1, driver_history=history, driver_history_mask=history_mask)
        emission = out.get("temporal_accel_pregate_mps2")
        if emission is None or torch.count_nonzero(emission).item() != 0:
            raise ValueError(f"{name}: temporal emission is not exactly zero")
        reports.append({"file": name, "sha256": loaded.sha256, "core_bitwise_frozen": True,
                        "changed_tensors": changed, "trainable_tensor_names": names,
                        "temporal_emission": 0.0})
    return reports, check_r13f_checkpoints(args.smoke_dir, 4.525)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("smoke-dir", "smoke-log", "init-checkpoint", "config", "report"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--train-exit-code", type=int, required=True)
    args = p.parse_args(argv)
    result = round12_gate.check_log(args.smoke_dir / "train_log.jsonl", args.smoke_log,
                                   train_exit_code=args.train_exit_code, prior_weight=0)
    if result["ok"]:
        try:
            result["checkpoints"], result["repair_a"] = check_artifacts(args)
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(str(exc))
    with args.report.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print("smoke: ok (round 13C); temporal emission = 0" if result["ok"]
          else "smoke: bad " + "; ".join(result["errors"]))
    return 0 if result["ok"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
