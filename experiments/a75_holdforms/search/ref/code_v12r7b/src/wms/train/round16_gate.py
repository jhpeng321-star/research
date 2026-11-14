"""R16 (S-RAMPSPEED) config / init / offline / smoke + production verdicts, CPU only and fail closed.

Patterned on ``round14_gate``: the 13E verdict machinery (``round13e_gate``: strict checkpoint load,
optimizer trainable set, frozen-bytes, schedule, log finiteness, emissions, saved-recipe identity) is
re-pointed at the R16 recipe by :func:`install`.  R16 = the R14 stage A / stage B recipe plus EXACTLY:

* A: ``model.ramp_speed_adapt_learnable = true`` and ``optim.lr_scale."ramp_speed_adapt" = 1000`` (the
  shared-scalar scale of ``merge_yield``); init = the 13E stage-B FINAL (b2), exactly as R14 A, so
  R16 A and R14 A differ ONLY in the ramp term (R16 runs in parallel with R14, on gpuB);
* B: ``model.ramp_speed_adapt_learnable = true`` and ``"ramp_speed_adapt"`` appended to
  ``optim.freeze_prefixes`` (stage-A scalars frozen like ``merge_yield``); init = R16 A FINAL.

Init: A loads the 13E b2 FINAL as-is; the three ramp tensors are the ONLY new tensors and hold the
TRAIN-fitted initialization (``src/wms/model/ramp_speed_adapt_init.json``); A must change them.
B loads R16 A as-is (courtesy fresh-zeroed, as R14 B) and keeps them bit-frozen.

Root-agnostic: the run root is ``R16_ROOT`` (default ``/root/autodl-tmp/c4/v5``, identical on gpuR and
gpuB); runtime paths are accepted only inside it when the gate runs from the prepared
``<root>/repo_r16_<sha8>`` copy (else inside the repository).  The R14 stop-rule waiters (paired
d11 checks against gpuR-only state files) are NOT part of R16: post-hoc checks are run separately.

  python -m wms.train.round16_gate config
  python -m wms.train.round16_gate preflight --root R --offline-report R/diag_r16/R16_OFFLINE_REPORT.json
  python -m wms.train.round16_gate verdict --stage a --run-dir D --log L --train-exit-code 0 \
      --init-checkpoint CK [--production] --report OUT.json
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import re
import sys

from wms.config.load import load_train_config
from . import round13e_gate as g
from . import round14_gate as r14

ROOT = g.ROOT
REMOTE_ROOT = Path(os.environ.get("R16_ROOT", "/root/autodl-tmp/c4/v5"))
TAG, RUN, NAME = "r16", "c4_wm_v10_round16", "WM-S-C4-V10ROUND16"
CONFIGS = {"a": ROOT / "configs/c4_wm_v10round16.toml",
           "b": ROOT / "configs/c4_wm_v10round16_stage_b.toml"}
REFERENCE = dict(r14.CONFIGS)
INIT_REL = "runs/c4_wm_v10_round13e_b2/wm_s.pt"
#: 13E stage-B FINAL (resumed b2) = R14 A init = the offline-evidence checkpoint.
INIT_SHA256 = "952ec059de5a9c0285d2327bcea57c836fd5d736c9975592ffbb932a506e84c8"
OFFLINE_CKPT_SHA256 = INIT_SHA256
RAMP_TENSORS = ("ramp_speed_adapt.log_headway_scale", "ramp_speed_adapt.log_lookahead",
                "ramp_speed_adapt.strength_logit")
RAMP_LR_SCALE = 1000.0
OFFLINE_ARMS = ("R14", "R16")
FLAG_KEYS = ("emergency_braking", "secure_gap_veto", "lane_change_secure_gap", "emergency_onset")
_YIELD = tuple(g.YIELD_TENSORS)
require, read_json = g.require, g.read_json


def digest(path):
    """sha256 of a file outside or inside the workspace; restricted path tokens are refused."""
    import hashlib
    path = Path(os.path.abspath(path))
    require(not any(t in str(path).lower() for t in g.FORBIDDEN), f"restricted path: {path}")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def operational_root():
    if ROOT.parent == REMOTE_ROOT and re.fullmatch(r"repo_r16_[0-9a-f]{8}", ROOT.name):
        return REMOTE_ROOT
    return ROOT


def install():
    """Point the generic 13E verdict code at the R16 recipe (idempotent)."""
    g.VARIANTS.clear()
    g.VARIANTS["main"] = {"tag": TAG, "run": RUN, "secure_gap_veto": True, "name": NAME,
                          "configs": dict(CONFIGS)}
    g.set_variant("main")
    g.FLAG_KEYS = FLAG_KEYS
    g.operational_root = operational_root
    g.check_configs = check_configs
    g.check_init = check_init
    # A trains exactly the 13E calibration set + the learned yield + the three ramp scalars; the
    # A checkpoint tensor set is the init's plus the ramp tensors (the yield tensors are in b2).
    g.YIELD_TENSORS = (*_YIELD, *RAMP_TENSORS)
    return g


# ------------------------------------------------------------------ recipe
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
    require(a.model.init_from == INIT_REL, "A init must be the 13E stage-B FINAL (b2)")
    require(not any(p.startswith("ramp_speed_adapt") for p in a.optim.freeze_prefixes)
            and a.optim.lr_scale.get("ramp_speed_adapt") == RAMP_LR_SCALE, "A must train the ramp scalars")
    require("ramp_speed_adapt" in b.optim.freeze_prefixes, "B must freeze the stage-A ramp scalars")
    require(a.curriculum.stage_steps == (30,) and b.curriculum.stage_steps == (40, 60),
            "R16 keeps the R14 schedules")
    require(all(getattr(a.model, k) is True for k in g.EMISSIONS), "A emissions must all be OFF")
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
    """A: 13E b2 as-is + exactly the three ramp tensors new; B: R16 A as-is (courtesy as R14)."""
    from wms.model.checkpoint import sha256_file
    courtesy = sorted(n for n in before if n.startswith("follower_courtesy.")) if stage == "b" else []
    new = sorted(RAMP_TENSORS) if stage == "a" else []
    require(stage == "b" or not any(n.startswith("ramp_speed_adapt.") for n in before),
            "A init source must be a pre-R16 core (no ramp tensors)")
    require(stage == "a" or set(RAMP_TENSORS) <= set(before), "B init source lacks the R16 A ramp tensors")
    require(set(_YIELD) <= set(before), "init source lacks the learned yield tensors")
    require(report.get("sha256") == sha256_file(checkpoint), "init source SHA mismatch")
    require(sorted(report.get("loaded", [])) == sorted(set(before) - set(courtesy))
            and sorted(report.get("new_in_model", [])) == sorted(courtesy + new),
            "init is not as-is source weights")
    for key in ("expanded", "shape_mismatch", "unused_from_checkpoint"):
        require(report.get(key) == [], f"unclean init: {key}")
    for key in ("driver_param_init", "residual_accel_reinit", "temporal_accel_reinit"):
        require(key not in report, f"init transform is forbidden: {key}")
    require(report.get("follower_courtesy_init") == ("zero" if stage == "b" else "checkpoint"),
            "courtesy init mode mismatch")


def check_ramp_tensors(directory, init_checkpoint, stage):
    """A: the ramp scalars start at the TRAIN init, are trained and stay finite; B: bit-frozen."""
    import torch
    files = ["wm_s_stage120.pt", "wm_s.pt"] + (["wm_s_stage300.pt"] if stage == "b" else [])
    before = torch.load(init_checkpoint, map_location="cpu", weights_only=True)["model"]
    out = {}
    for name in files:
        saved = torch.load(Path(directory) / name, map_location="cpu", weights_only=True)["model"]
        require(all(n in saved and torch.isfinite(saved[n]).all() for n in RAMP_TENSORS),
                f"{name}: ramp tensors missing / non-finite")
        if stage == "b":
            require(all(g.same_bytes(before[n], saved[n]) for n in RAMP_TENSORS),
                    f"{name}: stage-B ramp tensors changed (must be frozen)")
        out[name] = {n: float(saved[n]) for n in RAMP_TENSORS}
    if stage == "a":
        from wms.model.ramp_speed_adapt import LearnedRampSpeedAdapt
        init = LearnedRampSpeedAdapt().state_dict()
        final = torch.load(Path(directory) / "wm_s.pt", map_location="cpu", weights_only=True)["model"]
        require(any(not g.same_bytes(init[n.split(".", 1)[1]], final[n]) for n in RAMP_TENSORS),
                "stage-A ramp scalars never changed (not trained)")
    return out


# ------------------------------------------------------------------ evidence / preflight
def check_offline(report_path):
    """R16 offline evidence: listed file hashes and the deterministic D2 re-analysis must match."""
    sys.path.insert(0, str(ROOT / "tools/r16_checks"))
    import numpy as np
    import r16_panel_ana
    report_path = Path(report_path)
    doc = json.loads(report_path.read_text(encoding="utf-8"))
    require(doc.get("status") == "complete" and doc.get("checkpoint_sha256") == OFFLINE_CKPT_SHA256,
            "R16 offline report incomplete or not on the 13E b2 FINAL")
    require(tuple(doc.get("arms", ())) == OFFLINE_ARMS, "R16 offline arms mismatch")
    base = report_path.parent
    files = doc.get("files")
    require(isinstance(files, dict) and files, "R16 offline evidence file list missing")
    for name, sha in files.items():
        require(isinstance(name, str) and ".." not in Path(name).parts and not Path(name).is_absolute(),
                f"unsafe evidence name: {name}")
        require(digest(base / name) == sha, f"R16 offline evidence drift: {name}")
    for key, spec in doc.get("analyses", {}).items():
        again = r16_panel_ana.ana_r14.clean(r16_panel_ana.d2_analysis(
            str(base / spec["dir"]), list(OFFLINE_ARMS), np.random.default_rng(16)))
        require(r14.same_tree(json.loads(json.dumps(again, allow_nan=False)), spec["d2"]),
                f"R16 offline D2 analysis differs from evidence: {key}")
    require({"panel120", "panel300"} <= set(doc.get("analyses", {})), "R16 offline analyses missing")
    return doc


def preflight(root, offline_report):
    install()
    check_configs()
    root = Path(root)
    from wms.data.c4_episodes import MANIFEST_NAME
    from wms.model.checkpoint import sha256_file
    for name in r14.V5:
        require((root / "data" / name / MANIFEST_NAME).is_file(), f"missing data manifest: {name}")
    for name, sha in r14.LOWPEN.items():
        require(digest(root / r14.LOWPEN_DIR / name / MANIFEST_NAME) == sha,
                f"low-share pack manifest drift: {name}")
    init = root / INIT_REL
    require(sha256_file(init) == INIT_SHA256, "init is not the 13E stage-B FINAL (b2)")
    require((init.with_name("physics.json")).is_file(), "init physics.json missing")
    check_offline(offline_report)
    from wms.cli.assemble import build_model, build_physics
    from wms.train.settings import TrainSettings
    for stage, cfg in check_configs().items():
        TrainSettings.from_config(cfg)
        model = build_model(cfg, build_physics(cfg))
        g.check_emissions(model, stage)
        if stage == "a":
            check_ramp_init(model)
    return {"ok": True, "init_sha256": INIT_SHA256, "offline_sha256": digest(offline_report)}


def verdict(stage, run_dir, log, exit_code, init_checkpoint, production=False):
    install()
    result = g.verdict(argparse.Namespace(stage=stage, production=production, smoke_dir=Path(run_dir),
                                          smoke_log=Path(log), init_checkpoint=Path(init_checkpoint),
                                          train_exit_code=int(exit_code)))
    if result["ok"]:
        try:
            result["ramp_tensors"] = check_ramp_tensors(run_dir, init_checkpoint, stage)
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(str(exc))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config")
    off = sub.add_parser("offline")
    off.add_argument("--report", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--root", required=True)
    pre.add_argument("--offline-report", required=True)
    ver = sub.add_parser("verdict")
    ver.add_argument("--stage", choices=("a", "b"), required=True)
    ver.add_argument("--run-dir", required=True)
    ver.add_argument("--log", required=True)
    ver.add_argument("--train-exit-code", type=int, required=True)
    ver.add_argument("--init-checkpoint", required=True)
    ver.add_argument("--production", action="store_true")
    ver.add_argument("--report", required=True)
    sub.add_parser("launch")
    args = parser.parse_args(argv)
    if args.command == "config":
        install()
        check_configs()
        print(json.dumps({"ok": True, "stages": ["a", "b"]}))
    elif args.command == "offline":
        check_offline(args.report)
        print(json.dumps({"ok": True}))
    elif args.command == "preflight":
        print(json.dumps(preflight(args.root, args.offline_report)))
    elif args.command == "verdict":
        result = verdict(args.stage, args.run_dir, args.log, args.train_exit_code, args.init_checkpoint,
                         args.production)
        report = Path(args.report)
        require(not report.exists(), f"verdict report exists: {report}")
        report.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        print(json.dumps({"ok": result["ok"], "errors": result["errors"]}))
        return 0 if result["ok"] else 3
    else:
        raise SystemExit("use scripts/round16_gpuB.sh (smoke / launch-a / launch-b / chain)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
