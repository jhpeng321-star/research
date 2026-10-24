"""R13E config/preflight and CPU-only, fail-closed artifact verdicts.

This module never trains. The shell owns the explicitly requested launch. A uses
the R12C merge census but no courtesy forward/prior; B uses the R13C/R12 census
and repair-A rules. Frozen comparisons always use original, untransformed bytes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

from wms.config.load import apply_overrides, load_raw, load_train_config
from . import round12_gate as r12
from .round12c_gate import EXPECTED_TRAINABLE_TENSORS, merge_log_errors_single_stage
from .round9_gate import read_rows

ROOT = Path(__file__).resolve().parents[3]
CONFIGS = {"a": ROOT / "configs/c4_wm_v10round13e.toml",
           "b": ROOT / "configs/c4_wm_v10round13e_stage_b.toml"}
EMISSIONS = ("temporal_accel_off", "residual_accel_off", "follower_courtesy_off")
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1",
             "truth", "vtype")
OPTIMIZER = re.compile(r"^WM-S optimizer groups: (\{.*\})\s*$", re.MULTILINE)
REMOTE_ROOT = Path("/root/autodl-tmp/c4/v5")
# Registered S-R12C FINAL (13C's registration and operator start script).
R12C_SHA256 = "c76c338dea6355f85cfaee63cdb9077fb129ed25db0ab582d77eda2961822091"
STOP_RULE = {"baseline": "S-R12C", "paired_same_panel": True,
             "full_road_lane1_hdv_a_lt_minus3_share_ratio_gt": 2.0,
             "all_lane1_onestep_mae_delta_gt_mps2": 0.1,
             "measurement_error": "STOP", "signal": "owned production pidfd SIGTERM only"}
REQUIRED_SOURCE = ("configs/c4_wm_v10round13e.toml", "configs/c4_wm_v10round13e_stage_b.toml",
                   "scripts/round13e_gate.sh", "src/wms/train/round13e_gate.py",
                   "tests/test_r13e_plan.py", "configs/c4_wm_v10round13c.toml",
                   "scripts/round13c_waiter.py", "tests/test_r13e_emergency.py",
                   "tests/test_r13e_secure_gap.py",
                   "tools/r10b_checks/run_onestep.py", "tools/r12_checks/midrun_check.py",
                   "tools/r12_checks/base_calibration.py",
                   "tools/r13e_checks/offline.py", "tools/r13e_checks/bit_identity.py",
                   "tools/r10b_checks/common.py", "tools/r10b_checks/analyze.py",
                   "tools/r10b_checks/make_headzero.py", "tools/r10_attribution.py",
                   "tools/r10b_courtesy_offline.py", "tools/wm_eval_common.py",
                   "src/wms/physics/emergency.py", "src/wms/model/merge_pivot.py",
                   "tests/test_r13e_offline.py", "tests/test_round12_gate_verdict.py",
                   "tests/test_r10b_courtesy.py", "tests/test_r11_courtesy.py", "tests/__init__.py",
                   "pyproject.toml")
BASELINE_COMMIT = "eb75010c6a2a7ef66eebc3282e03f50e37d4b202"
BASELINE_DIR = ".r13e_baseline"
BASELINE_HELPERS = ("tests/test_r10b_courtesy.py", "tests/test_r11_courtesy.py")


def operational_root():
    """Runtime access outside the repo is enabled only in the exact archive location."""
    if ROOT.parent == REMOTE_ROOT and re.fullmatch(r"repo_r13e_[0-9a-f]{8}", ROOT.name):
        return REMOTE_ROOT
    return ROOT


def workspace_path(value, *, exists=True):
    path = Path(value)
    candidate = path if path.is_absolute() else ROOT / path
    candidate = Path(os.path.abspath(candidate))  # lexical checks precede filesystem resolution
    if any(token in str(candidate).lower() for token in FORBIDDEN):
        raise ValueError(f"restricted path: {value}")
    if not candidate.is_relative_to(operational_root()):
        raise ValueError(f"path must stay inside workspace: {value}")
    resolved = candidate.resolve()
    if any(token in str(p).lower() for p in (candidate, resolved) for token in FORBIDDEN):
        raise ValueError(f"restricted path: {value}")
    if not resolved.is_relative_to(operational_root()):
        raise ValueError(f"path must stay inside workspace: {value}")
    if exists and not resolved.exists():
        raise FileNotFoundError(f"missing required asset: {resolved}")
    return resolved


def read_json(path):
    def invalid(value):
        raise ValueError(f"non-finite JSON constant: {value}")
    return json.loads(workspace_path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference_13c(overrides):
    """Use the local 13C config, or its local Git object before the pending merge.

    No checkout/export or file write: the reference recipe is merged in memory.
    The only permissible parent is the same round-2 file used by 13C itself.
    """
    path = ROOT / "configs/c4_wm_v10round13c.toml"
    if path.is_file():
        return load_train_config(path, overrides)
    from wms.config.schema import deep_merge
    from wms.config.train import TrainConfig
    from wms.core import toml
    source = subprocess.run(["git", "show", "c4/v10r13c:configs/c4_wm_v10round13c.toml"],
                            cwd=ROOT, check=True, capture_output=True, text=True).stdout
    raw = toml.loads(source)
    require(raw.pop("extends", None) == "c4_wm_v10round2.toml", "unexpected 13C reference parent")
    return TrainConfig.from_dict(apply_overrides(
        deep_merge(load_raw(ROOT / "configs/c4_wm_v10round2.toml"), raw), overrides))


def check_configs():
    """Resolve both complete recipes, including the real 16-file extends chains."""
    a, b = (load_train_config(CONFIGS[s]) for s in ("a", "b"))
    expected_a = load_train_config(ROOT / "configs/c4_wm_v10round12c.toml", [
        "name=WM-S-C4-V10ROUND13E", "model.init_from=runs/c4_wm_v10_round12c/wm_s.pt",
        "model.driver_param_init=checkpoint", "model.residual_accel_reinit=false",
        "model.follower_courtesy_init_zero=false", "physics.emergency_braking=true",
        "physics.secure_gap_veto=true",
        "loss.driver_prior.weight=0.0", "loss.follower_courtesy.weight=0.0",
        "loss.follower_courtesy.contrast=false",
        "runtime.out_dir=runs/c4_wm_v10_round13e_a"])
    expected_b = reference_13c([
        "name=WM-S-C4-V10ROUND13E-STAGE-B",
        "model.init_from=runs/c4_wm_v10_round13e_a/wm_s.pt",
        "physics.emergency_braking=true", "physics.secure_gap_veto=true",
        "runtime.out_dir=runs/c4_wm_v10_round13e_b"])
    require(a == expected_a, "stage A differs from the conservative R12C recipe")
    require(b == expected_b, "stage B differs from the EXACT13C recipe")
    for cfg in (a, b):
        require(cfg.physics.emergency_braking is True, "emergency braking must be ON")
        require(cfg.physics.secure_gap_veto is True, "secure gap veto must be ON")
        require(cfg.model.driver_param_init == "checkpoint" and not cfg.model.residual_accel_reinit,
                "centre/reset init transforms must be OFF")
        require(cfg.loss.driver_prior.weight == 0, "driver prior must be OFF")
    require(all(getattr(a.model, k) is True for k in EMISSIONS), "A emissions must all be OFF")
    require(a.loss.follower_courtesy.weight == 0 and not a.model.follower_courtesy_init_zero,
            "A courtesy loss/init must be OFF")
    require(a.curriculum.stage_steps == (30,) and a.curriculum.unroll_s == (120.,),
            "A must be 30 steps at 120 seconds")
    require(b.curriculum.stage_steps == (40, 60) and b.curriculum.unroll_s == (120., 300.),
            "B must retain the 13C schedule")
    return {"a": a, "b": b}


def same_bytes(left, right):
    import torch
    return (left.dtype == right.dtype and left.shape == right.shape
            and torch.equal(left.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
                            right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)))


def strict_checkpoint(path):
    import torch
    from wms.model.checkpoint import load_checkpoint
    path = workspace_path(path)
    # The loader may read these implicit siblings. Guard resolved paths first.
    for name in ("physics.json", "TRAIN_RECEIPT.json", "lane_calib.json"):
        workspace_path(path.with_name(name), exists=False)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("format") == "wm_s/1", "not a wm_s/1 checkpoint")
    loaded = load_checkpoint(path, device="cpu", strict=True)
    weights, stored = loaded.model.state_dict(), payload["model"]
    require(bool(weights) and set(weights) == set(stored), "strict reload tensor set mismatch")
    require(all(torch.isfinite(v).all() and same_bytes(v, stored[n]) for n, v in weights.items()),
            "strict reload changed tensor bytes/dtypes or contains non-finite weights")
    require(not getattr(loaded.model, "lane_rate_calib", None)
            and not getattr(loaded.model, "lane_v5_merge_calib", None),
            "unexpected checkpoint calibration transform")
    return payload, loaded


def check_offline(report_path, checkpoint):
    """Validate the producer's evidence, not a four-field completion assertion.

    Recompute one-step statistics from the adjacent NPZ evidence and paired
    rollout estimates from all 96 state/seed rows (48 bootstrap clusters).
    No provenance path in the report is followed to a dataset.
    """
    from tools.r13e_checks import offline
    from wms.model.checkpoint import sha256_file
    import numpy as np

    report_path, checkpoint = workspace_path(report_path), workspace_path(checkpoint)
    doc = read_json(report_path)
    require(isinstance(doc, dict) and doc.get("status") == "complete", "offline report is not complete")
    require(doc.get("strict_exact_weights") is True, "offline strict_exact_weights must be true")
    require(doc.get("checkpoint_sha256") == sha256_file(checkpoint) == R12C_SHA256,
             "offline report does not identify the exact S-R12C init checkpoint")
    require(workspace_path(doc.get("checkpoint", "")) == checkpoint,
             "offline checkpoint path mismatch")
    for key, expected in {"states": 48, "seeds_per_state": 2, "horizon_s": 120.,
                          "dt_s": .5, "device": "cpu", "linux_nice": 19,
                          "arms": offline.ARMS, "definitions": offline.DEFINITIONS,
                          "override_only": ["emergency_braking", "secure_gap_veto"],
                          "emissions": dict.fromkeys(EMISSIONS, True)}.items():
        require(doc.get(key) == expected, f"offline contract mismatch: {key}")
    require(all(type(value) is bool for arm in doc["arms"].values() for value in arm.values())
            and all(value is True for value in doc["emissions"].values()),
            "offline physics switches/emissions must be booleans")
    for key in ("workers", "threads"):
        require(type(doc.get(key)) is int and 1 <= doc[key] <= 4, f"offline CPU limit: {key}")
    roots, hashes = doc.get("train_roots"), doc.get("train_manifest_sha256")
    require(isinstance(roots, list) and roots and isinstance(hashes, list)
            and len(roots) == len(hashes), "offline TRAIN manifest provenance missing")
    for path in [doc.get("config", ""), *roots]:
        require(isinstance(path, str) and path, "offline source path missing")
        workspace_path(path, exists=False)  # spelling and symlink guard; never open data
    require(len(set(roots)) == len(roots), "offline duplicate TRAIN roots")
    for value in [doc.get("state_dict_sha256"), doc.get("states_sha256"), *hashes]:
        require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
                "offline provenance digest missing")
    payload, loaded = strict_checkpoint(checkpoint)
    require(offline.state_digest(payload["model"]) == doc["state_dict_sha256"],
            "offline tensor bytes do not match S-R12C")
    require(doc.get("base_physics") == asdict(loaded.model.physics)
            and all(getattr(loaded.model, key) is True for key in EMISSIONS),
            "offline base physics/emissions mismatch")
    require(not loaded.model.physics.emergency_braking and not loaded.model.physics.secure_gap_veto,
            "offline baseline switches must be OFF")

    windows = doc.get("one_step_windows")
    require(isinstance(windows, list) and len(windows) >= 2
            and all(isinstance(w, str) and w for w in windows)
            and len(set(windows)) == len(windows), "offline one-step windows missing/duplicated")
    for window in windows:
        offline.guard_text(window)
    # Guard ALL evidence filenames before opening the first table.
    paths = {arm: workspace_path(report_path.parent / f"onestep_{arm}.npz") for arm in offline.ARMS}
    from tools.r10b_checks import run_onestep as producer
    cols = producer.COLS + producer.BASE_COLS
    packed = {}
    for arm, path in paths.items():
        with np.load(path, allow_pickle=False) as data:
            require(set(data.files) == {"rows", "cols"} and list(data["cols"]) == cols,
                    "offline one-step evidence columns mismatch")
            rows = data["rows"]
            require(rows.ndim == 2 and rows.shape[1] == len(cols) and len(rows) >= 5,
                    "offline one-step evidence empty/malformed")
            require(set(rows[:, cols.index("win")]) == set(range(len(windows))),
                    "offline one-step evidence windows mismatch")
            require(np.isfinite(rows[:, [cols.index(c) for c in ("fin", "rec")]]).all(),
                    "offline nonfinite one-step evidence")
            packed[arm] = rows
    fixed = [i for i, c in enumerate(cols) if c not in ("fin", "pre", "off")]
    require(all(np.array_equal(packed["off"][:, fixed], rows[:, fixed], equal_nan=True)
                for rows in packed.values()), "offline one-step arms are not paired")
    require(doc.get("one_step") == offline.json_safe(offline.analyze_onestep(packed, cols)),
            "offline one-step statistics/paired CIs differ from evidence")
    for arm in (*offline.ARMS, *offline.CONTRASTS):
        for cls in offline.CLASSES:
            for metric in ("bias", "mae"):
                require(doc["one_step"][arm][cls][metric]["ci_available"] is True,
                        f"offline one-step evidence insufficient: {arm}/{cls}/{metric}")
    check_offline_rollouts(doc.get("rollouts"), offline)
    return doc


def check_offline_rollouts(rollouts, offline):
    import numpy as np

    require(isinstance(rollouts, dict), "offline rollout evidence missing")
    rows = rollouts.get("rows")
    require(isinstance(rows, list) and len(rows) == 96, "offline needs 48 states x 2 seeds")
    metrics = {"hdv_rows", "finite_hdv_transitions", "a_lt_minus3_rows", "a_lt_minus3_share",
               "ttc_eligible_rows", "ttc_lt_1_5_rows", "ttc_lt_1_5_share", "overlap_pair_steps",
               "invalid_hdv_rows", "bad_acceleration_rows", "blowup", "stuck_vehicle_steps",
               "stuck_final_vehicles", "stuck_max_wait_s", "stuck_mean_ramp_in_zone",
               "delay_total_veh_s", "merges_per_run", "hdv_insertion_events",
               "post_insertion_truncated_events"}
    for region in ("upstream", "merge", "downstream"):
        metrics.update((f"delay_{region}_veh_s", f"exposure_{region}_veh_s"))
    for who in ("inserter", "follower"):
        metrics.update(f"hdv_{who}_post_{suffix}" for suffix in
                       ("events", "hard_brake_events", "finite_rows", "hard_brake_rows",
                        "hard_brake_events_share", "hard_brake_rows_share"))
    identities, snapshots, seen = {}, set(), set()
    for row in rows:
        state, replicate = row.get("state"), row.get("replicate")
        require(type(state) is int and 0 <= state < 48 and type(replicate) is int
                and replicate in (0, 1) and (state, replicate) not in seen,
                "offline duplicate/invalid state-seed pairing")
        seen.add((state, replicate))
        require(row.get("seed") == offline.RNG_SEED + 2 * state + replicate,
                "offline seed pairing mismatch")
        require(type(row.get("vehicles")) is int and row["vehicles"] > 0,
                "offline rollout vehicle roster missing")
        t0 = row.get("t0_s")
        require(type(t0) in (int, float) and math.isfinite(t0), "offline state time missing")
        identity = (row.get("world") or row.get("key"), t0)
        require(isinstance(identity[0], str) and identity[0], "offline snapshot identity missing")
        offline.guard_text(identity[0])
        if state in identities:
            require(identities[state] == (identity, row["vehicles"]), "offline seed rosters differ")
        else:
            require(identity not in snapshots, "offline duplicate snapshot")
            snapshots.add(identity)
            identities[state] = (identity, row["vehicles"])
        for arm in offline.ARMS:
            values = row.get(arm)
            require(isinstance(values, dict) and set(values) == metrics,
                    f"offline rollout metrics incomplete: {arm}")
            for key, value in values.items():
                nullable = key.startswith("hdv_") and key.endswith("_share")
                require((nullable and value is None) or
                        (type(value) in (int, float, bool) and math.isfinite(value) and value >= 0),
                        f"offline invalid rollout metric: {key}")
                if (not key.endswith(("_share", "_veh_s"))
                        and key not in ("blowup", "stuck_max_wait_s", "stuck_mean_ramp_in_zone")):
                    require(type(value) is int, f"offline rollout count must be integer: {key}")
            require(type(values["blowup"]) is bool
                    and values["blowup"] == bool(values["invalid_hdv_rows"] or values["bad_acceleration_rows"]),
                    "offline blowup evidence inconsistent")
            require(0 < values["finite_hdv_transitions"] <= 240 * row["vehicles"]
                    and 0 < values["ttc_eligible_rows"] <= values["hdv_rows"] <= 240 * row["vehicles"],
                    "offline rollout exposure missing/invalid")
            shares = [("a_lt_minus3_rows", "finite_hdv_transitions", "a_lt_minus3_share"),
                      ("ttc_lt_1_5_rows", "ttc_eligible_rows", "ttc_lt_1_5_share")]
            for who in ("inserter", "follower"):
                p = f"hdv_{who}_post_"
                shares += [(p + "hard_brake_events", p + "events", p + "hard_brake_events_share"),
                           (p + "hard_brake_rows", p + "finite_rows", p + "hard_brake_rows_share")]
            for numerator, denominator, share in shares:
                n, d = values[numerator], values[denominator]
                require(type(n) is int and type(d) is int and 0 <= n <= d
                        and values[share] == (n / d if d else None), "offline rollout share/count mismatch")
            require(math.isclose(values["delay_total_veh_s"], sum(values[f"delay_{r}_veh_s"]
                    for r in ("upstream", "merge", "downstream"))), "offline delay regions mismatch")
    groups = np.array([r["state"] for r in rows])
    summary, estimates = {}, {}
    for arm in (*offline.ARMS, *offline.CONTRASTS):
        summary[arm] = {}
        for metric in sorted(metrics):
            later, earlier = offline.CONTRASTS.get(arm, (arm, arm))
            def number(value):
                return float("nan") if value is None else float(value)
            values = [number(r[later][metric]) - number(r[earlier][metric])
                      if arm in offline.CONTRASTS else number(r[arm][metric]) for r in rows]
            key = tuple(None if not math.isfinite(v) else v for v in values)
            if key not in estimates:
                estimates[key] = offline._estimate(values, groups)
            summary[arm][metric] = estimates[key]
    require(rollouts.get("state_cluster_run_mean") == offline.json_safe(summary),
            "offline rollout paired snapshot-cluster CIs differ from evidence")
    totals = {}
    for arm in offline.ARMS:
        total = {key: sum(r[arm][key] for r in rows) for key in
                 ("hdv_rows", "finite_hdv_transitions", "a_lt_minus3_rows", "ttc_eligible_rows",
                  "ttc_lt_1_5_rows", "overlap_pair_steps", "invalid_hdv_rows", "bad_acceleration_rows")}
        total.update(blowup_runs=sum(r[arm]["blowup"] for r in rows),
                     blowup_states=len({r["state"] for r in rows if r[arm]["blowup"]}), states=48, runs=96)
        total["a_lt_minus3_share"] = total["a_lt_minus3_rows"] / total["finite_hdv_transitions"]
        total["ttc_lt_1_5_share"] = total["ttc_lt_1_5_rows"] / total["ttc_eligible_rows"]
        totals[arm] = total
    require(rollouts.get("totals") == totals, "offline rollout totals differ from evidence")


def preflight(root, checkpoint, offline_report):
    check_configs()
    root = workspace_path(root)
    require(root.is_dir(), "run root must be a directory")
    for name in ("v5_native", "genrand_plain", "genrand_branch", "pairs300"):
        workspace_path(root / "data" / name, exists=False)
    checkpoint = workspace_path(checkpoint)
    require(checkpoint.name == "wm_s.pt", "init must be S-R12C FINAL wm_s.pt")
    from wms.model.checkpoint import sha256_file
    require(sha256_file(checkpoint) == R12C_SHA256, "init is not registered S-R12C FINAL SHA256")
    check_offline(offline_report, checkpoint)
    payload, loaded = strict_checkpoint(checkpoint)
    require(payload.get("history") and len(payload["history"]) == 30
            and all(r.get("stage_s") == 120 and type(r.get("step")) is int and r["step"] == i
                    for i, r in enumerate(payload["history"], 1))
            and tuple(loaded.legacy_config.get("stage_steps", ())) == (30,)
            and tuple(loaded.legacy_config.get("unroll_curriculum_s", ())) == (120.,),
            "init must be the complete 30-step S-R12C final")
    require(all(getattr(loaded.model.spec, k) is True for k in EMISSIONS)
            and loaded.model.spec.driver_param_mode == "physical", "init is not S-R12C physics-only")
    from wms.cli.assemble import build_model, build_physics
    r12c_physics = build_physics(load_train_config(ROOT / "configs/c4_wm_v10round12c.toml"))
    require(asdict(loaded.model.physics) == asdict(r12c_physics),
            "S-R12C inference physics differs from its registered recipe")
    from wms.train.settings import TrainSettings
    for cfg in check_configs().values():
        TrainSettings.from_config(cfg)  # refuse incomplete cross-owner physics/config integration
    for stage, cfg in check_configs().items():
        check_emissions(build_model(cfg, build_physics(cfg)), stage)
    artifacts = [root / ".round13e_gate.lock", root / "runs/r13e_gate_core_tests",
                 root / "smoke_r13e.complete.json"]
    for stage in ("a", "b"):
        artifacts += [root / f"runs/smoke_r13e_{stage}", root / f"smoke_r13e_{stage}.log",
                      root / f"smoke_r13e_{stage}.gate.json", root / f"production_r13e_{stage}.log",
                      root / f"production_r13e_{stage}.gate.json"]
    artifacts += [root / "runs/c4_wm_v10_round13e_a",
                  root / "runs/c4_wm_v10_round13e_b", root / "chk_r13e"]
    for path in artifacts:
        workspace_path(path, exists=False)
        require(not path.exists() and not path.is_symlink(), f"prior R13E artifact exists: {path}")
    return {"ok": True, "offline": "complete", "strict_exact_weights": True}


def allowed_trainables(model, cfg, stage):
    names = sorted(n for n, _ in model.named_parameters() if not n.startswith(cfg.optim.freeze_prefixes))
    if stage == "a":
        require(names == sorted(EXPECTED_TRAINABLE_TENSORS), "A trainable set is not exactly core-only")
    else:
        require(names and all(n.startswith(("residual_net.", "follower_courtesy."))
                                 and not n.startswith("residual_net.lane") for n in names),
                "B trainable set includes core/physics/pivot/temporal/lane")
        require(all(any(n.startswith(p) for n in names)
                    for p in ("residual_net.accel.", "follower_courtesy.")), "B emission optimizer missing")
    return names


def check_optimizer(text, model, cfg, stage):
    expected = allowed_trainables(model, cfg, stage)
    matches = OPTIMIZER.findall(text)
    require(bool(matches), "optimizer trainable list missing")
    for match in matches:
        doc = json.loads(match)
        require(sorted(doc.get("trainable_tensor_names", [])) == expected,
                "optimizer trainable list differs from exact configured set")
        require(doc.get("freeze_prefixes") == list(cfg.optim.freeze_prefixes), "optimizer freeze list mismatch")
        require(not doc.get("unmatched_freeze_prefixes"), "unmatched freeze prefixes")
        # Verify group membership/count and LR rule, not just the printed names.
        buckets = {}
        for name in expected:
            prefixes = [p for p in cfg.optim.lr_scale if name.startswith(p)]
            prefix = max(prefixes, key=len) if prefixes else None
            buckets[prefix] = buckets.get(prefix, 0) + 1
        groups = doc.get("groups", [])
        require(len(groups) == len(buckets), "optimizer group count mismatch")
        require({g.get("prefix") for g in groups} == set(buckets), "optimizer group prefixes mismatch")
        for group in groups:
            prefix = group["prefix"]
            scale = cfg.optim.lr_scale.get(prefix, 1.)
            require(group.get("tensors") == buckets[prefix] and group.get("lr_scale") == scale
                    and group.get("lr") == cfg.optim.lr * scale,
                    "optimizer group membership/LR mismatch")
    return expected


def check_init(report, before, checkpoint, stage):
    from wms.model.checkpoint import sha256_file
    courtesy = sorted(n for n in before if n.startswith("follower_courtesy.")) if stage == "b" else []
    require(report.get("sha256") == sha256_file(checkpoint), "init source SHA mismatch")
    require(sorted(report.get("loaded", [])) == sorted(set(before) - set(courtesy))
            and sorted(report.get("new_in_model", [])) == courtesy, "init is not as-is source weights")
    for key in ("expanded", "shape_mismatch", "unused_from_checkpoint"):
        require(report.get(key) == [], f"unclean init: {key}")
    for key in ("driver_param_init", "residual_accel_reinit", "temporal_accel_reinit"):
        require(key not in report, f"init transform is forbidden: {key}")
    require(report.get("follower_courtesy_init") == ("zero" if stage == "b" else "checkpoint"),
            "courtesy init mode mismatch")


def check_log(rows, text, stage, *, production=False, train_exit_code=0):
    errors, metrics = [], {}
    if train_exit_code != 0:
        errors.append(f"training exited {train_exit_code}")
    if "Traceback" in text:
        errors.append("Traceback in smoke log")
    expected_stages = ([120] * (30 if production else 2) if stage == "a" else
                       [120] * (40 if production else 2) + [300] * (60 if production else 2))
    if [r.get("stage_s") for r in rows] != expected_stages:
        errors.append("wrong/empty/incomplete stage schedule")
    if stage == "a":
        errors.extend(merge_log_errors_single_stage(rows, metrics))
        for row in rows:
            if any(k.startswith("follower_courtesy") for k in row):
                errors.append("A courtesy telemetry present although forward/loss must be OFF")
    else:
        from .round11_gate import head_range_errors
        from wms.cli.assemble import build_physics
        physics = build_physics(load_train_config(CONFIGS[stage]))
        errors.extend(r12.merge_log_errors(rows, metrics))
        errors.extend(r12.courtesy_log_errors(rows, metrics))
        errors.extend(head_range_errors(rows, (physics.accel_min_mps2, physics.accel_max_mps2)))
        errors.extend(r12.replay_errors(rows, metrics))
    errors.extend(r12.prior_absent_errors(rows))
    errors.extend(r12.cutin_absent_errors(rows))

    def finite_tree(value):
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite_tree(v) for v in value.values())
        if isinstance(value, list):
            return all(finite_tree(v) for v in value)
        return True
    if not finite_tree(rows):
        errors.append("non-finite metric/loss")
    return {"ok": not errors, "errors": errors, "metrics": metrics}


def check_emissions(model, stage):
    """Exercise reloaded emission paths on a synthetic CPU state (no dataset)."""
    import torch
    from wms.model.history import LONG_HISTORY_F
    kin = torch.zeros(1, 3, 7)
    kin[..., 0] = torch.tensor([0., 100., 40.])
    kin[..., 2] = 1.
    kin[:, 2, 2:4] = torch.tensor([0., 1.])  # A ramp partner as well as lane-1 HDVs.
    kin[..., 4], kin[..., 6] = 10., 5.
    history = torch.zeros(1, 3, model.driver_history_len, LONG_HISTORY_F)
    model.diag_emit_temporal = True  # Read-only diagnostic supplied by the 13C merge.
    with torch.no_grad():
        out = model({"veh_kin": kin, "present": torch.ones(1, 3, dtype=torch.bool),
                     "is_cav": torch.zeros(1, 3, dtype=torch.bool)}, k_steps=1,
                    driver_history=history,
                    driver_history_mask=torch.zeros(history.shape[:-1], dtype=torch.bool))
    for name in ("physics_accel_mps2", "accel_mps2", "raw_residual_mps2", "courtesy_accel_mps2"):
        require(name in out and torch.isfinite(out[name]).all(), f"missing/non-finite emission: {name}")
    if stage == "a":
        require(all(torch.count_nonzero(out[k]).item() == 0
                    for k in ("raw_residual_mps2", "courtesy_accel_mps2")), "A emission is not exactly zero")
    require("temporal_accel_pregate_mps2" in out,
            "missing mandatory 13C temporal emission diagnostic (merge 13C sources)")
    require(torch.count_nonzero(out["temporal_accel_pregate_mps2"]).item() == 0,
            "temporal emission is not exactly zero")


def check_checkpoints(directory, rows, text, checkpoint, cfg, stage, *, production=False):
    from wms.cli.assemble import build_physics
    before_payload, _ = strict_checkpoint(checkpoint)
    before = before_payload["model"]
    check_init(read_json(directory / "INIT_FROM.json"), before, checkpoint, stage)
    files = [("wm_s_stage120.pt", 120), ("wm_s.pt", None)]
    if stage == "b":
        files.insert(1, ("wm_s_stage300.pt", 300))
    reports = []
    for name, horizon in files:
        payload, loaded = strict_checkpoint(directory / name)
        model, saved = loaded.model, payload["model"]
        require(set(saved) == set(before), f"{name}: tensor set differs from init")
        names = check_optimizer(text, model, cfg, stage)
        changed = [n for n in before if not same_bytes(before[n], saved[n])]
        allowed = set(names)
        require(all(n in allowed for n in changed), f"{name}: frozen tensor bytes changed")
        require(asdict(model.physics) == asdict(build_physics(cfg)), f"{name}: physics contract mismatch")
        require(model.physics.emergency_braking is True, f"{name}: emergency braking OFF")
        require(model.physics.secure_gap_veto is True, f"{name}: secure gap veto OFF")
        for flag in EMISSIONS:
            expected = stage == "a" or flag == "temporal_accel_off"
            require(getattr(model.spec, flag) is expected and getattr(model, flag) is expected,
                    f"{name}: {flag} mismatch")
        require(model.spec.driver_param_mode == "physical" and model.spec.merge_pivot_state_st
                and model.follower_courtesy is not None and model.follower_courtesy.version == 2
                and not model.follower_courtesy_rollout_grad, f"{name}: model contract mismatch")
        flat = loaded.legacy_config
        # Compare the whole resolved recipe, allowing only the registered smoke/runtime
        # overrides. This rejects lookalike checkpoints with altered losses/model/LRs.
        overrides = [f"model.init_from={workspace_path(checkpoint)}",
                     "loss.loss_share_logging=false", "loss.brake_follow_cav_weight=2.0",
                     "optim.batch_size=4", "runtime.device=cuda"]
        if not production:
            overrides += ["curriculum.stage_steps=[2]" if stage == "a" else
                          "curriculum.stage_steps=[2,2]", "selection.val_max_windows=4",
                          "curriculum.max_hours=1.5"]
            if stage == "b":
                overrides += ["optim.batch_size_per_stage=[4,4]"]
        expected_flat = load_train_config(CONFIGS[stage], overrides).to_legacy_flat()
        require(Path(flat.get("out_dir", "")).resolve() == directory.resolve(),
                f"{name}: saved output directory identity mismatch")
        for key, value in expected_flat.items():
            if key in ("out_dir", "physics"):
                continue  # output is checked separately; physics is compared above.
            actual = flat.get(key)
            require(json.dumps(actual, sort_keys=True) == json.dumps(value, sort_keys=True),
                    f"{name}: saved config identity mismatch: {key}")
        require(flat.get("driver_param_init", "checkpoint") == "checkpoint"
                and flat.get("residual_accel_reinit", False) is False
                and flat.get("follower_courtesy_init_zero", False) is (stage == "b")
                and flat.get("driver_prior_weight", 0) == 0
                and flat.get("follower_courtesy_weight", 0) == cfg.loss.follower_courtesy.weight,
                f"{name}: saved loss/init contract mismatch")
        batches = cfg.optim.batch_size_per_stage if production else ((4,) if stage == "a" else (4, 4))
        steps = cfg.curriculum.stage_steps if production else ((2,) if stage == "a" else (2, 2))
        require(tuple(flat.get("batch_size_per_stage", ())) == batches
                and tuple(flat.get("stage_steps", ())) == steps,
                f"{name}: saved schedule mismatch")
        if horizon is None:
            require(payload.get("history") == rows, f"{name}: final history mismatch")
            prefixes = ("driver_encoder.head.", "driver_encoder.centre_logits") if stage == "a" else (
                "residual_net.accel.", "follower_courtesy.")
            require(all(any(n.startswith(p) for n in changed) for p in prefixes),
                    f"{name}: intended trainable paths never changed")
        else:
            require(payload.get("stage_s") == horizon
                    and payload.get("step") == max(r["step"] for r in rows if r["stage_s"] == horizon),
                    f"{name}: stage/step mismatch")
        r12.centre_physical(saved, model.physics.speed_limit_mps)
        if stage == "b":
            require(model.spec.follower_courtesy_state_grad is True
                    and model.spec.follower_courtesy_intent == "pivot"
                    and flat.get("follower_courtesy_absolute_weight") == 4.525,
                    f"{name}: repair-A contract mismatch")
        check_emissions(model, stage)
        reports.append({"file": name, "sha256": loaded.sha256, "frozen_bitwise": True,
                        "strict_exact_weights": True, "emergency_braking": True,
                        "secure_gap_veto": True, "temporal_emission": 0.0,
                        "trainable_tensor_names": names, "changed_tensors": changed})
    return reports


def verdict(args):
    cfg = check_configs()[args.stage]
    directory = workspace_path(args.smoke_dir)
    checkpoint = workspace_path(args.init_checkpoint)
    log = workspace_path(args.smoke_log)
    rows = read_rows(directory / "train_log.jsonl")
    text = log.read_text(encoding="utf-8")
    result = check_log(rows, text, args.stage, production=args.production,
                       train_exit_code=args.train_exit_code)
    if result["ok"]:
        try:
            result["checkpoints"] = check_checkpoints(directory, rows, text, checkpoint, cfg, args.stage,
                                                        production=args.production)
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(str(exc))
    return result


def digest(path):
    from hashlib import sha256
    with workspace_path(path).open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def write_once(path, doc):
    with workspace_path(path, exists=False).open("x", encoding="utf-8") as stream:
        json.dump(doc, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_name_allowed(name, *, baseline=False):
    """Lexical source allowlist; call before Git archive opens any blob bytes."""
    from pathlib import PurePosixPath
    path = PurePosixPath(name)
    if (not name or "\\" in name or path.is_absolute() or ".." in path.parts
            or path.as_posix() != name or any(t in name.lower() for t in FORBIDDEN)):
        return False
    if name.startswith("src/wms/") and path.suffix in (".py", ".json"):
        return True
    if baseline:
        return name in BASELINE_HELPERS
    return (name == "pyproject.toml" or name in REQUIRED_SOURCE
            or (name.startswith("configs/") and path.suffix == ".toml")
            or (name.startswith("tools/") and path.suffix == ".py"))


def committed_sources(sha, *, baseline=False):
    # ls-tree reads names/modes only, never restricted blob contents.
    raw = subprocess.check_output(["git", "ls-tree", "-rz", "--full-tree", sha], cwd=ROOT)
    names = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        info, name = entry.split(b"\t", 1)
        name = name.decode("utf-8")
        if not source_name_allowed(name, baseline=baseline):
            continue
        mode, kind, _oid = info.split()
        require(mode in (b"100644", b"100755") and kind == b"blob",
                f"non-regular committed source: {name}")
        names.append(name)
    required = BASELINE_HELPERS + ("src/wms/__init__.py",) if baseline else REQUIRED_SOURCE
    require(set(required) <= set(names), "missing committed launch dependencies")
    return sorted(names)


def export_sources(sha, names, target):
    """Export ONLY prevalidated regular source members; never archive broad trees."""
    import hashlib
    import tarfile
    require(names and len(set(names)) == len(names)
            and all(source_name_allowed(n) for n in names), "invalid source allowlist")
    selected, files = set(names), {}
    proc = subprocess.Popen(["git", "archive", "--format=tar", sha, "--", *names],
                            cwd=ROOT, stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            for member in archive:
                if member.isdir():
                    require(any(n.startswith(member.name.rstrip("/") + "/") for n in selected),
                            "unexpected archive directory")
                    continue
                require(member.name in selected and member.name not in files and member.isfile(),
                        f"unexpected archive entry: {member.name}")
                path = target / member.name
                require(path.resolve().is_relative_to(target), "unsafe archive member")
                path.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream, path.open("xb") as out:
                    h = hashlib.sha256()
                    for block in iter(lambda: stream.read(1 << 20), b""):
                        out.write(block)
                        h.update(block)
                files[member.name] = h.hexdigest()
                path.chmod(member.mode & 0o777)
        require(proc.wait() == 0, "git archive failed")
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
            proc.wait()
    require(set(files) == selected, "archive missing launch dependencies")
    return files


def prepare():
    """Future operator operation: export committed source only, never a worktree copy.

    Deliberately unavailable on a local workstation. No training/process inspection.
    Restricted archive members are not extracted or opened.
    """
    require(sys.platform == "linux" and ROOT.is_relative_to(REMOTE_ROOT.parent),
            "prepare must be run on the deployment host under /root/autodl-tmp/c4")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(re.fullmatch(r"[0-9a-f]{40}", sha), "invalid source commit")
    names = committed_sources(sha)
    baseline = committed_sources(BASELINE_COMMIT, baseline=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    require(not any(source_name_allowed(n.decode("utf-8")) for n in untracked.split(b"\0") if n),
            "uncommitted source dependency: commit tested sources before preparing")
    # Refuse a stale commit when the tested launch sources are still pending.
    subprocess.run(["git", "diff", "--quiet", "--no-ext-diff", "--no-textconv", sha, "--", *names],
                   cwd=ROOT, check=True)
    target = REMOTE_ROOT / f"repo_r13e_{sha[:8]}"
    require(not target.exists() and not target.is_symlink(), "prepared repo already exists")
    # Verification precedes mkdir; no overwrite/retry of a partial archive.
    require(REMOTE_ROOT.is_dir(), "deployment parent is missing")
    target.mkdir()
    files = export_sources(sha, names, target)
    baseline_files = export_sources(BASELINE_COMMIT, baseline, target / BASELINE_DIR)
    # The marker is inside a new archive, not part of this source checkout.
    with (target / "R13E_SOURCE.json").open("x", encoding="utf-8") as stream:
        json.dump({"commit": sha, "files": files,
                   "baseline": {"commit": BASELINE_COMMIT, "directory": BASELINE_DIR,
                                "files": baseline_files}}, stream, indent=2)
    return {"ok": True, "repo": str(target), "commit": sha}


def check_source():
    require(operational_root() == REMOTE_ROOT, "launch requires the prepared repo_r13e_<sha8> copy")
    doc = read_json(ROOT / "R13E_SOURCE.json")
    sha = doc.get("commit", "")
    require(re.fullmatch(r"[0-9a-f]{40}", sha)
            and ROOT.name == f"repo_r13e_{sha[:8]}", "archive source identity mismatch")
    files = doc.get("files", {})
    require(isinstance(files, dict) and all(source_name_allowed(n) for n in files),
            "invalid prepared source allowlist")
    require(all(name in files for name in REQUIRED_SOURCE), "missing committed launch dependencies")
    actual = {"pyproject.toml"}
    for directory in ("src", "configs", "scripts", "tests", "tools"):
        top = ROOT / directory
        require(not top.is_symlink(), "symlink addition to prepared source")
        workspace_path(top)
        for parent, directories, names in os.walk(ROOT / directory, followlinks=False):
            for name in (*directories, *names):
                candidate = Path(parent) / name
                require(not any(t in str(candidate).lower() for t in FORBIDDEN),
                        "restricted addition to prepared source")
                require(not candidate.is_symlink(), "symlink addition to prepared source")
            actual.update((Path(parent) / name).relative_to(ROOT).as_posix() for name in names)
    require(actual == set(files), "prepared source file set changed")
    for name, expected in files.items():
        path = workspace_path(ROOT / name)
        require(path.is_relative_to(ROOT) and not path.is_symlink() and digest(path) == expected,
                 f"prepared source drift: {name}")
    check_baseline(doc.get("baseline"))
    return doc


def check_baseline(doc):
    require(isinstance(doc, dict) and doc.get("commit") == BASELINE_COMMIT
            and doc.get("directory") == BASELINE_DIR, "missing pinned 13C baseline export")
    files = doc.get("files")
    require(isinstance(files, dict) and set(BASELINE_HELPERS + ("src/wms/__init__.py",)) <= set(files)
            and all(source_name_allowed(n, baseline=True) for n in files), "invalid 13C baseline allowlist")
    require(not (ROOT / BASELINE_DIR).is_symlink(), "symlink 13C baseline export")
    base, actual = workspace_path(ROOT / BASELINE_DIR), set()
    require(base.is_relative_to(ROOT), "13C baseline must stay inside prepared source")
    for parent, directories, names in os.walk(base, followlinks=False):
        for name in (*directories, *names):
            candidate = Path(parent) / name
            require(not candidate.is_symlink() and not any(t in name.lower() for t in FORBIDDEN),
                    "unsafe 13C baseline addition")
        actual.update((Path(parent) / n).relative_to(base).as_posix() for n in names)
    require(actual == set(files), "13C baseline source file set changed")
    # Validate every name and file set before opening even the first blob.
    for name, expected in files.items():
        require(digest(base / name) == expected, f"13C baseline source drift: {name}")
    return base


def check_registration(root, checkpoint, offline_report):
    source = check_source()
    root, checkpoint = workspace_path(root), workspace_path(checkpoint)
    require(root == REMOTE_ROOT, "registered run root mismatch")
    require(checkpoint == root / "runs/c4_wm_v10_round12c/wm_s.pt",
            "registered S-R12C final path mismatch")
    require(digest(checkpoint) == R12C_SHA256, "registered S-R12C FINAL SHA256 mismatch")
    check_offline(offline_report, checkpoint)
    doc = read_json(root / "registration_r13e.json")
    expected = {"registration": "S-R13E preregistered", "commit": source["commit"],
                "source_manifest_sha256": digest(ROOT / "R13E_SOURCE.json"),
                "config_sha256": {s: digest(p) for s, p in CONFIGS.items()},
                "init_sha256": R12C_SHA256, "offline_sha256": digest(offline_report),
                "stop_rule": STOP_RULE, "check_dir": "chk_r13e",
                "outputs": {s: f"runs/c4_wm_v10_round13e_{s}" for s in ("a", "b")}}
    require(all(doc.get(k) == v for k, v in expected.items()), "R13E preregistration identity/rule mismatch")
    for path in (root / "diag_brake/d11_run5.py", root / "chk_r10b/states192.json.gz"):
        workspace_path(path)
    return expected


def artifact_verdict(root, stage, checkpoint, *, production=False):
    stem = f"production_r13e_{stage}" if production else f"smoke_r13e_{stage}"
    directory = root / "runs" / (f"c4_wm_v10_round13e_{stage}" if production else stem)
    result = verdict(argparse.Namespace(stage=stage, production=production, smoke_dir=directory,
                                       smoke_log=root / f"{stem}.log", init_checkpoint=checkpoint,
                                       train_exit_code=0))
    require(result["ok"], f"{stem} failed revalidation: {result['errors']}")
    return result


def production_ready(root, checkpoint, offline_report, stage):
    """Recompute evidence, never trust a mutable 'ok' receipt alone."""
    registration = check_registration(root, checkpoint, offline_report)
    root, checkpoint = workspace_path(root), workspace_path(checkpoint)
    smoke_a = root / "runs/smoke_r13e_a/wm_s.pt"
    artifact_verdict(root, "a", checkpoint)
    artifact_verdict(root, "b", smoke_a)
    receipt = read_json(root / "smoke_r13e.complete.json")
    require(receipt == {"registration": registration,
                        "a_sha256": digest(smoke_a),
                        "b_sha256": digest(root / "runs/smoke_r13e_b/wm_s.pt")},
            "both-stage smoke receipt identity mismatch")
    if stage == "b":
        # The provisional B smoke bootstraps from A smoke. Re-smoke B separately
        # against the actual A FINAL before the explicit B production command.
        a_final = root / "runs/c4_wm_v10_round13e_a/wm_s.pt"
        artifact_verdict(root, "a", checkpoint, production=True)
        for mode in ("midrun", "postrun"):
            result = read_json(root / f"chk_r13e/a/{mode}/RESULT.json")
            require(result.get("decision") == "CONTINUE"
                    and result.get("registration_sha256") == digest(root / "registration_r13e.json")
                    and result.get("checkpoints", {}).get("r10b", {}).get("sha256")
                    == digest(root / "runs/c4_wm_v10_round13e_a" /
                              ("wm_s_stage120.pt" if mode == "midrun" else "wm_s.pt")),
                    f"A {mode} check missing/failed or checkpoint identity mismatch")
        return {"ok": True, "init_checkpoint": str(a_final)}
    return {"ok": True, "init_checkpoint": str(checkpoint)}


def make_waiter(r13c, args):
    """Reuse 13C's measurement math and pinned-process signaling, with R13E paths."""
    stage = args.stage

    class Waiter(r13c.Waiter):
        def __init__(self):
            self.repo, self.root, self.mode = ROOT, workspace_path(args.root), args.mode
            self.run = self.root / f"runs/c4_wm_v10_round13e_{stage}"
            self.process = r13c.ProductionProcess(args.pid, self.run, args.start_ticks)
            self.out = self.root / f"chk_r13e/{stage}/{self.mode}"
            self.work = self.root / f"chk_r13e/{stage}/{self.mode}_work"
            self.commands = []

        def wait_checkpoint(self):
            # execute() has already claimed its mode and created the output dir.
            # A startup/import/claim failure cannot acknowledge an armed monitor.
            write_once(args.ready, {"pid": args.pid, "start_ticks": args.start_ticks,
                                    "stage": stage, "mode": args.mode})
            return super().wait_checkpoint()

        def snapshot(self, src, name):
            src = workspace_path(src)
            for sibling in ("physics.json", "lane_calib.json"):
                workspace_path(src.with_name(sibling), exists=False)
            checkpoint, info = super().snapshot(src, name)
            if name == "baseline":
                require(info["sha256"] == R12C_SHA256, "midrun baseline is not registered S-R12C FINAL")
            return checkpoint, info

        def command(self, script, *values):
            script = workspace_path(script)
            # 13C's reader overrides emission flags from the checkpoint; the config
            # supplies R13E's two physics switches for both A and B.
            values = tuple(CONFIGS[stage].name if v == "c4_wm_v10round13c.toml" else v
                           for v in values)
            if Path(script).name == "run_onestep.py":
                # This legacy reader builds physics from the config rather than
                # the snapshot. Keep the NEW flags OFF for the as-trained controls.
                values += tuple(v for arm in ("r12c", "r9")
                                for flag in ("emergency_braking", "secure_gap_veto")
                                for v in ("--arm-set", f"{arm}=physics.{flag}=false"))
            # Keep 13C's measurement math and process ownership, but do not inherit
            # its 8-worker/nice15 execution policy. This covers BOTH diagnostics.
            values = list(map(str, values))
            for i, value in enumerate(values):
                if value == "--workers":
                    require(i + 1 < len(values), "missing diagnostic worker count")
                    values[i + 1] = str(min(4, max(1, int(values[i + 1]))))
                elif value.startswith("--workers="):
                    values[i] = f"--workers={min(4, max(1, int(value.split('=', 1)[1])))}"
            command = ["nice", "-n", "19", sys.executable, "-B", str(script), *values]
            self.commands.append(command)
            env = os.environ.copy()
            env.update({k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
            env.update(CUDA_VISIBLE_DEVICES="", WMS_JAM_WORKERS="4", PYTHONDONTWRITEBYTECODE="1",
                       PYTHONPATH=os.pathsep.join(map(str, (self.repo / "src", self.repo,
                                                   self.repo / "tools", self.repo / "tools/r10b_checks"))))
            log = workspace_path(self.work / f"command_{len(self.commands):02d}.log", exists=False)
            with log.open("x", encoding="utf-8") as stream:
                proc = subprocess.run(command, cwd=self.repo, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT, timeout=4 * 3600, check=False)
            require(proc.returncode == 0, f"measurement command exit {proc.returncode}: {log}")

        def measure(self, checkpoint):
            result = super().measure(checkpoint)
            result["labels"]["r10b"] = f"R13E {stage.upper()} candidate"
            result["registration"] = read_json(self.root / "registration_r13e.json")
            result["registration_sha256"] = digest(self.root / "registration_r13e.json")
            return result

        def execute(self):
            code = super().execute()
            report = workspace_path(self.out / "RESULT.md")
            report.write_text(report.read_text(encoding="utf-8").replace(
                "R13C", f"R13E stage {stage.upper()}"), encoding="utf-8")
            return code

    return Waiter()


def wait_registered(args):
    """Inherited midrun errors fail closed; only exact owned PID/start/--out can stop."""
    import importlib.util
    check_source()
    path = workspace_path(ROOT / "scripts/round13c_waiter.py")
    spec = importlib.util.spec_from_file_location("r13e_registered_waiter", path)
    r13c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r13c)
    # An imported operational dependency must match the registered rule too.
    require(not r13c.stop_reasons(.02, .01, .5, .4)
            and len(r13c.stop_reasons(.02001, .01, .50001, .4)) == 2,
            "13C paired stop-rule implementation mismatch")
    waiter = make_waiter(r13c, args)
    return waiter.execute()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config")
    sub.add_parser("prepare")
    for command in ("registered", "production-ready", "smoke-complete"):
        cmd = sub.add_parser(command)
        for flag in ("root", "init-checkpoint", "offline-report"):
            cmd.add_argument("--" + flag, required=True)
        if command == "production-ready":
            cmd.add_argument("--stage", choices=("a", "b"), required=True)
    waiter = sub.add_parser("waiter")
    for flag in ("root", "ready"):
        waiter.add_argument("--" + flag, required=True)
    waiter.add_argument("--pid", required=True, type=int)
    waiter.add_argument("--start-ticks", required=True, type=int)
    waiter.add_argument("--stage", required=True, choices=("a", "b"))
    waiter.add_argument("--mode", required=True, choices=("midrun", "postrun"))
    pre = sub.add_parser("preflight")
    for flag in ("root", "init-checkpoint", "offline-report"):
        pre.add_argument("--" + flag, required=True)
    check = sub.add_parser("verdict")
    check.add_argument("--stage", choices=("a", "b"), required=True)
    check.add_argument("--production", action="store_true")
    check.add_argument("--train-exit-code", type=int, required=True)
    for flag in ("smoke-dir", "smoke-log", "init-checkpoint", "report"):
        check.add_argument("--" + flag, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "config":
            check_configs()
            print("R13E stage A/B config contracts: ok")
            return 0
        if args.command == "prepare":
            result = prepare()
        elif args.command == "waiter":
            return wait_registered(args)
        elif args.command == "registered":
            result = {"ok": True, "registration": check_registration(
                args.root, args.init_checkpoint, args.offline_report)}
        elif args.command == "production-ready":
            result = production_ready(args.root, args.init_checkpoint, args.offline_report, args.stage)
        elif args.command == "smoke-complete":
            registration = check_registration(args.root, args.init_checkpoint, args.offline_report)
            root = workspace_path(args.root)
            a_final = root / "runs/smoke_r13e_a/wm_s.pt"
            artifact_verdict(root, "a", workspace_path(args.init_checkpoint))
            artifact_verdict(root, "b", a_final)
            write_once(root / "smoke_r13e.complete.json", {
                "registration": registration, "a_sha256": digest(a_final),
                "b_sha256": digest(root / "runs/smoke_r13e_b/wm_s.pt")})
            result = {"ok": True}
        elif args.command == "preflight":
            result = preflight(args.root, args.init_checkpoint, args.offline_report)
        else:
            # Guard the report before any other artifact reads.
            report = workspace_path(args.report, exists=False)
            try:
                result = verdict(args)
            except Exception as exc:
                result = {"ok": False, "errors": [str(exc)]}
            with report.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(result, allow_nan=False))
        return 0 if result["ok"] else 3
    except Exception as exc:
        print(f"R13E gate: bad: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
