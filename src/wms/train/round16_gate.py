"""R16 (S-RAMPSPEED) config / init / offline-evidence checks, CPU only and fail closed (never trains).

Patterned on ``round14_gate``: the R16 recipe is the R14 stage A / stage B recipe plus EXACTLY the
registered R16 differences per stage:

* A: ``model.ramp_speed_adapt_learnable = true``, ``optim.lr_scale."ramp_speed_adapt" = 1000`` (the
  shared-scalar scale of ``merge_yield``), init = the R14 stage-B FINAL, new name / output;
* B: ``model.ramp_speed_adapt_learnable = true``, ``"ramp_speed_adapt"`` appended to
  ``optim.freeze_prefixes`` (stage-A scalars frozen like ``merge_yield``), init = R16 A FINAL.

Init: A loads the R14 B FINAL as-is; the three ramp tensors are the ONLY new tensors and hold the
TRAIN-fitted initialization (``src/wms/model/ramp_speed_adapt_init.json``).  B loads R16 A as-is.
Offline evidence: ``R16_OFFLINE_REPORT.json`` (13E b2 FINAL + R14 physics, arms R14 / R16, file hashes,
deterministic re-analysis).  There is NO launch path: R16 is preparation only; a launch needs the
R14 B FINAL, an owner-approved registration and a launch layer patterned on ``round14_gate``.

  python -m wms.train.round16_gate config
  python -m wms.train.round16_gate offline --report R16_OFFLINE_REPORT.json
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

from wms.config.load import load_train_config
from . import round14_gate as r14

ROOT = r14.ROOT
TAG, RUN, NAME = "r16", "c4_wm_v10_round16", "WM-S-C4-V10ROUND16"
CONFIGS = {"a": ROOT / "configs/c4_wm_v10round16.toml",
           "b": ROOT / "configs/c4_wm_v10round16_stage_b.toml"}
REFERENCE = dict(r14.CONFIGS)
INIT_REL = "runs/c4_wm_v10_round14_b/wm_s.pt"
RAMP_TENSORS = ("ramp_speed_adapt.log_headway_scale", "ramp_speed_adapt.log_lookahead",
                "ramp_speed_adapt.strength_logit")
RAMP_LR_SCALE = 1000.0
#: 13E stage-B FINAL (resumed b2) used by the offline evidence (inference only).
OFFLINE_CKPT_SHA256 = "952ec059de5a9c0285d2327bcea57c836fd5d736c9975592ffbb932a506e84c8"
OFFLINE_ARMS = ("R14", "R16")
require = r14.require


def expected_config(stage):
    """R14 stage config + exactly the registered R16 differences."""
    init = INIT_REL if stage == "a" else f"runs/{RUN}_a/wm_s.pt"
    name = NAME if stage == "a" else f"{NAME}-STAGE-B"
    ref = load_train_config(REFERENCE[stage], [f"name={name}", f"model.init_from={init}",
                                               "model.ramp_speed_adapt_learnable=true",
                                               f"runtime.out_dir=runs/{RUN}_{stage}"])
    if stage == "a":
        optim = replace(ref.optim, lr_scale={**ref.optim.lr_scale, "ramp_speed_adapt": RAMP_LR_SCALE})
    else:
        optim = replace(ref.optim, freeze_prefixes=(*ref.optim.freeze_prefixes, "ramp_speed_adapt"))
    return replace(ref, optim=optim), load_train_config(CONFIGS[stage])


def _normalise(cfg):
    """Order-insensitive lr_scale mapping (TOML table order is not part of the recipe)."""
    return replace(cfg, optim=replace(cfg.optim, lr_scale=dict(sorted(cfg.optim.lr_scale.items()))))


def check_configs():
    out = {}
    for stage in ("a", "b"):
        expected, cfg = expected_config(stage)
        require(_normalise(cfg) == _normalise(expected),
                f"R16 stage {stage.upper()} differs from R14 + the registered R16 changes")
        require(cfg.model.ramp_speed_adapt_learnable is True, "R16 ramp speed adaptation must be ON")
        require(cfg.physics.lane_change_secure_gap is True and cfg.physics.emergency_onset == "driver_b"
                and cfg.physics.emergency_braking is True and cfg.physics.secure_gap_veto is True
                and cfg.model.mainline_merge_yield_learnable is True, "R14 / 13E switches must be ON")
        require(math.isclose(sum(s.mix_ratio for s in cfg.data.c4_sources), 1.0, abs_tol=1e-9),
                "source shares must sum to 1")
        out[stage] = cfg
    a, b = out["a"], out["b"]
    require(not any(p.startswith("ramp_speed_adapt") for p in a.optim.freeze_prefixes)
            and a.optim.lr_scale.get("ramp_speed_adapt") == RAMP_LR_SCALE, "A must train the ramp scalars")
    require("ramp_speed_adapt" in b.optim.freeze_prefixes, "B must freeze the stage-A ramp scalars")
    require(a.curriculum.stage_steps == (30,) and b.curriculum.stage_steps == (40, 60),
            "R16 keeps the R14 schedules")
    return out


def check_ramp_init(model):
    """A freshly built / init-loaded R16 A model holds exactly the TRAIN-fitted initialization."""
    from wms.model.ramp_speed_adapt import load_init
    layer = getattr(model, "ramp_speed_adapt", None)
    require(layer is not None and model.ramp_speed_adapt_learnable, "ramp speed adaptation missing")
    init = load_init()
    for name, value in (("lookahead_m", layer.lookahead_m), ("strength", layer.strength),
                        ("headway_scale", layer.headway_scale)):
        require(math.isclose(float(value.detach()), init[name], rel_tol=1e-6), f"ramp init drift: {name}")


def check_init(report, before, checkpoint, stage):
    """A: R14 B FINAL as-is + exactly the three ramp tensors new; B: R16 A FINAL as-is (courtesy as R14)."""
    from wms.model.checkpoint import sha256_file
    courtesy = sorted(n for n in before if n.startswith("follower_courtesy.")) if stage == "b" else []
    new = sorted(RAMP_TENSORS) if stage == "a" else []
    require(stage == "b" or not any(n.startswith("ramp_speed_adapt.") for n in before),
            "A init source must be a pre-R16 core (no ramp tensors)")
    require(stage == "a" or set(RAMP_TENSORS) <= set(before), "B init source lacks the R16 A ramp tensors")
    require(report.get("sha256") == sha256_file(checkpoint), "init source SHA mismatch")
    require(sorted(report.get("loaded", [])) == sorted(set(before) - set(courtesy))
            and sorted(report.get("new_in_model", [])) == sorted(courtesy + new),
            "init is not as-is source weights")
    for key in ("expanded", "shape_mismatch", "unused_from_checkpoint"):
        require(report.get(key) == [], f"unclean init: {key}")
    for key in ("driver_param_init", "residual_accel_reinit", "temporal_accel_reinit"):
        require(key not in report, f"init transform is forbidden: {key}")


def check_offline(report_path):
    """R16 offline evidence: listed file hashes and the deterministic panel re-analysis must match."""
    sys.path.insert(0, str(ROOT / "tools/r16_checks"))
    import r16_panel_ana
    import numpy as np
    report_path = Path(report_path)
    doc = r14.read_json(report_path)
    require(doc.get("status") == "complete" and doc.get("checkpoint_sha256") == OFFLINE_CKPT_SHA256,
            "R16 offline report incomplete or not on the 13E b2 FINAL")
    require(tuple(doc.get("arms", ())) == OFFLINE_ARMS, "R16 offline arms mismatch")
    base = report_path.parent
    files = doc.get("files")
    require(isinstance(files, dict) and files, "R16 offline evidence file list missing")
    for name, sha in files.items():
        require(isinstance(name, str) and ".." not in Path(name).parts and not Path(name).is_absolute(),
                f"unsafe evidence name: {name}")
        require(r14.digest(base / name) == sha, f"R16 offline evidence drift: {name}")
    for key, spec in doc.get("analyses", {}).items():
        again = r16_panel_ana.d2_analysis(str(base / spec["dir"]), list(OFFLINE_ARMS),
                                          np.random.default_rng(16))
        again = r16_panel_ana.ana_r14.clean(again)
        require(r14.same_tree(json.loads(json.dumps(again, allow_nan=False)), spec["d2"]),
                f"R16 offline D2 analysis differs from evidence: {key}")
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config")
    off = sub.add_parser("offline")
    off.add_argument("--report", required=True)
    sub.add_parser("launch")
    args = parser.parse_args(argv)
    if args.command == "config":
        check_configs()
        print(json.dumps({"ok": True, "stages": ["a", "b"]}))
    elif args.command == "offline":
        check_offline(args.report)
        print(json.dumps({"ok": True}))
    else:
        raise SystemExit("R16 is preparation only: no launch path (needs R14 B FINAL + owner registration)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
