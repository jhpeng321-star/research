"""R14 config/preflight/registration and CPU-only, fail-closed verdicts (never trains).

Layered on the R13E gate (``round13e_gate``): its generic log/checkpoint/optimizer/emission
verdicts run unchanged against the R14 recipe after :func:`install` re-points the module
globals (tag, run, configs, flag keys, source allowlist).  R14-specific parts:

* recipe: the 13E stage A / stage B configs plus EXACTLY three registered differences per stage:
  ``physics.lane_change_secure_gap = true`` (D1), the seven-source data mix with the S-LOWPEN
  packs (D3) and ``data.replay.low_share_oversample = 1.69`` (D3); new names / init / outputs;
* init: the 13E stage-B FINAL (A) whose own 13E midrun AND postrun checks said CONTINUE;
  stage B starts from R14 A FINAL;
* stop rule: 13E's paired thresholds, reference = 13E stage-B FINAL evaluated with the SAME
  R14 switches (emergency, gs, learned yield, lane-change secure gap) in the same invocation;
* lesson from 13E: B production needs A's production verdict and A's MIDRUN CONTINUE only.
  A's slow postrun check runs in parallel with B; B's own midrun check requires A's postrun
  RESULT to be CONTINUE before it measures (fail closed: anything else stops B).
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from wms.config.load import load_train_config
from . import round13e_gate as g

ROOT, REMOTE_ROOT = g.ROOT, g.REMOTE_ROOT
TAG, RUN, NAME = "r14", "c4_wm_v10_round14", "WM-S-C4-V10ROUND14"
CONFIGS = {"a": ROOT / "configs/c4_wm_v10round14.toml",
           "b": ROOT / "configs/c4_wm_v10round14_stage_b.toml"}
REFERENCE = {"a": ROOT / "configs/c4_wm_v10round13e.toml",
             "b": ROOT / "configs/c4_wm_v10round13e_stage_b.toml"}
INIT_REL = "runs/c4_wm_v10_round13e_b/wm_s.pt"
SOURCE_MANIFEST = "R14_SOURCE.json"
LOWPEN_DIR = "data/lowpen_20261025"
#: S-LOWPEN-20261025 RECEIPT.md manifest sha256 (packs copied unchanged).
LOWPEN = {"lowpen_native": "cb9b32aaa4b5cb37e4d873ed63d49d1ff666035cfc982acc421a0a4ed764fb01",
          "lowpen_genrand_plain": "d33d9e5471f4e08f35c15993379fbc6464bfcda243b049a0d6549c90e8358249",
          "lowpen_genrand_branch": "fdd037b37de33d9b3e88f20d79475f41bd010a67b6de40d9c50e4f7ca0915cf9"}
V5 = ("v5_native", "genrand_plain", "genrand_branch", "pairs300")
#: (name, path, mix_ratio, paired) of the registered R14 mix (R14_DESIGN.md D3 derivation).
SOURCES = (("native_v5", "v5_native", 0.4694, False), ("lowpen_native", "lowpen_native", 0.0306, False),
           ("genrand_plain", "genrand_plain", 0.2382, False),
           ("lowpen_genrand_plain", "lowpen_genrand_plain", 0.0118, False),
           ("genrand_branch", "genrand_branch", 0.0824, False),
           ("lowpen_genrand_branch", "lowpen_genrand_branch", 0.0176, False),
           ("pairs300", "pairs300", 0.15, True))
LOW_SHARE = {"low_share_oversample": 1.69, "low_share_max": 0.15, "low_share_min_stage_s": 120.0}
FLAG_KEYS = ("emergency_braking", "secure_gap_veto", "lane_change_secure_gap")
STOP_RULE = {"baseline": "S-R13E stage-B FINAL", "paired_same_panel": True,
             "baseline_flags": "same_as_candidate: physics.emergency_braking, physics.secure_gap_veto, "
                               "physics.lane_change_secure_gap, model.mainline_merge_yield_learnable",
             "full_road_lane1_hdv_a_lt_minus3_share_ratio_gt": 2.0,
             "all_lane1_onestep_mae_delta_gt_mps2": 0.1,
             "measurement_error": "STOP", "signal": "owned production pidfd SIGTERM only",
             "stage_gate": "B needs A production verdict + A midrun CONTINUE; A postrun runs in "
                           "parallel and must be CONTINUE before B's midrun measures (else STOP B)"}
R14_SOURCE = ("configs/c4_wm_v10round14.toml", "configs/c4_wm_v10round14_stage_b.toml",
              "scripts/round14_gate.sh", "src/wms/train/round14_gate.py",
              "tests/test_r14_lane_secure_gap.py", "tests/test_r14_lowshare.py", "tests/test_r14_gate.py",
              "tools/r14_checks/ana_r14.py", "tools/r14_checks/mech.py", "tools/r14_checks/roll_r14.py",
              "tools/r14_checks/sanity_roll.py", "tools/r14_checks/offline_report.py")
REQUIRED_SOURCE = tuple(dict.fromkeys(g.REQUIRED_SOURCE + R14_SOURCE))
BASELINE_COMMIT = "65cc714e0e73c532583c61f1bbf0ea7d1d6f6b98"   # S-R13E launch code
BASELINE_DIR = ".r14_baseline"
OFFLINE_ARMS = ("13E", "13EL")
#: 13E stage-B production schedule (13C recipe): 40 steps at 120 s, then 60 at 300 s.
R13E_B_HISTORY = [120.0] * 40 + [300.0] * 60


def operational_root():
    if ROOT.parent == REMOTE_ROOT and re.fullmatch(r"repo_r14_[0-9a-f]{8}", ROOT.name):
        return REMOTE_ROOT
    return ROOT


def install():
    """Point the generic 13E verdict code at the R14 recipe (idempotent)."""
    g.VARIANTS.clear()
    g.VARIANTS["main"] = {"tag": TAG, "run": RUN, "secure_gap_veto": True, "name": NAME,
                          "configs": dict(CONFIGS)}
    g.set_variant("main")
    g.FLAG_KEYS = FLAG_KEYS
    g.REQUIRED_SOURCE = REQUIRED_SOURCE
    g.BASELINE_COMMIT, g.BASELINE_DIR = BASELINE_COMMIT, BASELINE_DIR
    g.STOP_RULE = STOP_RULE
    g.operational_root = operational_root
    g.check_configs = check_configs
    g.check_init = check_init
    g.check_source = check_source
    return g


require, read_json, digest, write_once = g.require, g.read_json, g.digest, g.write_once


def workspace_path(value, *, exists=True):
    return g.workspace_path(value, exists=exists)


# ------------------------------------------------------------------ recipe
def expected_config(stage):
    """13E stage config + exactly the registered R14 differences."""
    init = INIT_REL if stage == "a" else f"runs/{RUN}_a/wm_s.pt"
    name = NAME if stage == "a" else f"{NAME}-STAGE-B"
    ref = load_train_config(REFERENCE[stage], [f"name={name}", f"model.init_from={init}",
                                               "physics.lane_change_secure_gap=true",
                                               f"runtime.out_dir=runs/{RUN}_{stage}"])
    cfg = load_train_config(CONFIGS[stage])
    sources = tuple(replace(ref.data.c4_sources[0], name=n, path=p, mix_ratio=r, paired=pr)
                    for n, p, r, pr in SOURCES)
    data = replace(ref.data, c4_sources=sources, replay=replace(ref.data.replay, **LOW_SHARE))
    return replace(ref, data=data), cfg


def check_configs():
    out = {}
    for stage in ("a", "b"):
        expected, cfg = expected_config(stage)
        require(cfg == expected, f"R14 stage {stage.upper()} differs from 13E + the registered R14 changes")
        require(cfg.physics.lane_change_secure_gap is True and cfg.physics.lane_learnable is True,
                "R14 D1 lane-change secure gap must be ON (lane_learnable)")
        require(cfg.physics.emergency_braking is True and cfg.physics.secure_gap_veto is True
                and cfg.model.mainline_merge_yield_learnable is True, "13E switches must be ON")
        require(math.isclose(sum(s.mix_ratio for s in cfg.data.c4_sources), 1.0, abs_tol=1e-9),
                "R14 source shares must sum to 1")
        require(cfg.data.replay.enabled and cfg.data.replay.low_share_oversample == 1.69,
                "R14 D3 low-share replay weight missing")
        out[stage] = cfg
    a, b = out["a"], out["b"]
    require(a.curriculum.stage_steps == (30,) and a.curriculum.unroll_s == (120.,),
            "A must be 30 steps at 120 seconds")
    require(b.curriculum.stage_steps == (40, 60) and b.curriculum.unroll_s == (120., 300.),
            "B must retain the 13C schedule")
    require(all(getattr(a.model, k) is True for k in g.EMISSIONS), "A emissions must all be OFF")
    require("merge_yield" in b.optim.freeze_prefixes and a.optim.lr_scale.get("merge_yield") == 1000.0,
            "A trains / B freezes the learned yield scalars")
    return out


def check_init(report, before, checkpoint, stage):
    """A: 13E B FINAL as-is (yield + courtesy load); B: R14 A FINAL, courtesy fresh-zeroed (13C)."""
    from wms.model.checkpoint import sha256_file
    courtesy = sorted(n for n in before if n.startswith("follower_courtesy.")) if stage == "b" else []
    require(set(g.YIELD_TENSORS) <= set(before), "init source lacks the learned yield tensors")
    require(report.get("sha256") == sha256_file(checkpoint), "init source SHA mismatch")
    require(sorted(report.get("loaded", [])) == sorted(set(before) - set(courtesy))
            and sorted(report.get("new_in_model", [])) == sorted(courtesy),
            "init is not as-is source weights")
    for key in ("expanded", "shape_mismatch", "unused_from_checkpoint"):
        require(report.get(key) == [], f"unclean init: {key}")
    for key in ("driver_param_init", "residual_accel_reinit", "temporal_accel_reinit"):
        require(key not in report, f"init transform is forbidden: {key}")
    require(report.get("follower_courtesy_init") == ("zero" if stage == "b" else "checkpoint"),
            "courtesy init mode mismatch")


# ------------------------------------------------------------------ evidence
def check_13e_b(root, checkpoint):
    """The 13E stage-B FINAL must exist, be complete, and have passed BOTH 13E B checks."""
    root, checkpoint = workspace_path(root), workspace_path(checkpoint)
    require(checkpoint == root / INIT_REL, "init must be the 13E stage-B FINAL")
    workspace_path(root / "runs/c4_wm_v10_round13e_b/RUN_SUMMARY.json")
    sha = digest(checkpoint)
    for mode in ("midrun", "postrun"):
        result = read_json(root / f"chk_r13e/b/{mode}/RESULT.json")
        require(result.get("decision") == "CONTINUE", f"13E B {mode} check is not CONTINUE")
    post = read_json(root / "chk_r13e/b/postrun/RESULT.json")
    require(post.get("checkpoints", {}).get("r10b", {}).get("sha256") == sha,
            "13E B postrun did not measure this FINAL")
    payload, loaded = g.strict_checkpoint(checkpoint)
    require([r.get("stage_s") for r in payload.get("history") or []] == R13E_B_HISTORY,
            "init is not the complete 100-step 13E stage-B FINAL")
    from wms.cli.assemble import build_physics
    require(asdict(loaded.model.physics) == asdict(build_physics(load_train_config(REFERENCE["b"]))),
            "init inference physics differs from the 13E stage-B recipe")
    return sha


def check_offline(report_path):
    """R14 offline evidence: every listed file hash and the deterministic re-analysis must match."""
    sys.path.insert(0, str(ROOT / "tools/r14_checks"))
    import ana_r14
    report_path = workspace_path(report_path)
    doc = read_json(report_path)
    require(doc.get("status") == "complete" and doc.get("checkpoint_sha256") == g.R12C_SHA256,
            "R14 offline report incomplete or not on the S-R12C checkpoint")
    require(tuple(doc.get("arms", ())) == OFFLINE_ARMS, "R14 offline arms mismatch")
    base = report_path.parent
    files = doc.get("files")
    require(isinstance(files, dict) and files, "R14 offline evidence file list missing")
    for name, sha in files.items():
        require(isinstance(name, str) and ".." not in Path(name).parts and not Path(name).is_absolute(),
                f"unsafe evidence name: {name}")
        require(digest(base / name) == sha, f"R14 offline evidence drift: {name}")
    for key, spec in doc.get("analyses", {}).items():
        again = ana_r14.analyze(str(workspace_path(base / spec["dir"])), spec["horizon_s"], list(OFFLINE_ARMS),
                                spec["tags"])
        require(json.loads(json.dumps(again, allow_nan=False)) == spec["result"],
                f"R14 offline analysis differs from evidence: {key}")
    require({"panel120", "pairs300"} <= set(doc.get("analyses", {})), "R14 offline analyses missing")
    return doc


# ------------------------------------------------------------------ source
def prepare():
    require(sys.platform == "linux" and ROOT.is_relative_to(REMOTE_ROOT.parent),
            "prepare must be run on the deployment host under /root/autodl-tmp/c4")
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    require(re.fullmatch(r"[0-9a-f]{40}", sha), "invalid source commit")
    names = g.committed_sources(sha)
    baseline = g.committed_sources(BASELINE_COMMIT, baseline=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    require(not any(g.source_name_allowed(n.decode("utf-8")) for n in untracked.split(b"\0") if n),
            "uncommitted source dependency: commit tested sources before preparing")
    subprocess.run(["git", "diff", "--quiet", "--no-ext-diff", "--no-textconv", sha, "--", *names],
                   cwd=ROOT, check=True)
    target = REMOTE_ROOT / f"repo_r14_{sha[:8]}"
    require(not target.exists() and not target.is_symlink(), "prepared repo already exists")
    target.mkdir()
    files = g.export_sources(sha, names, target)
    baseline_files = g.export_sources(BASELINE_COMMIT, baseline, target / BASELINE_DIR)
    with (target / SOURCE_MANIFEST).open("x", encoding="utf-8") as stream:
        json.dump({"commit": sha, "files": files,
                   "baseline": {"commit": BASELINE_COMMIT, "directory": BASELINE_DIR,
                                "files": baseline_files}}, stream, indent=2)
    return {"ok": True, "repo": str(target), "commit": sha}


def check_source():
    require(operational_root() == REMOTE_ROOT, "launch requires the prepared repo_r14_<sha8> copy")
    doc = read_json(ROOT / SOURCE_MANIFEST)
    sha = doc.get("commit", "")
    require(re.fullmatch(r"[0-9a-f]{40}", sha) and ROOT.name == f"repo_r14_{sha[:8]}",
            "archive source identity mismatch")
    files = doc.get("files", {})
    require(isinstance(files, dict) and all(g.source_name_allowed(n) for n in files)
            and all(n in files for n in REQUIRED_SOURCE), "invalid prepared source allowlist")
    actual = {"pyproject.toml"}
    for directory in ("src", "configs", "scripts", "tests", "tools"):
        top = ROOT / directory
        require(not top.is_symlink(), "symlink addition to prepared source")
        for parent, directories, names in os.walk(top, followlinks=False):
            for name in (*directories, *names):
                candidate = Path(parent) / name
                require(not candidate.is_symlink()
                        and not any(t in str(candidate).lower() for t in g.FORBIDDEN),
                        "unsafe addition to prepared source")
            actual.update((Path(parent) / n).relative_to(ROOT).as_posix() for n in names
                          if "__pycache__" not in Path(parent).parts)
    require(actual == set(files), "prepared source file set changed")
    for name, expected in files.items():
        require(digest(ROOT / name) == expected, f"prepared source drift: {name}")
    g.check_baseline(doc.get("baseline"))
    return doc


def registration_expected(root, checkpoint, offline_report):
    source = check_source()
    root = workspace_path(root)
    require(root == REMOTE_ROOT, "registered run root mismatch")
    init_sha = check_13e_b(root, checkpoint)
    check_offline(offline_report)
    return {"registration": "S-R14 preregistered", "commit": source["commit"],
            "source_manifest_sha256": digest(ROOT / SOURCE_MANIFEST),
            "config_sha256": {s: digest(p) for s, p in CONFIGS.items()},
            "init": INIT_REL, "init_sha256": init_sha, "offline_sha256": digest(offline_report),
            "lowpen_manifest_sha256": LOWPEN, "stop_rule": STOP_RULE, "check_dir": f"chk_{TAG}",
            "outputs": {s: f"runs/{RUN}_{s}" for s in ("a", "b")}}


def check_registration(root, checkpoint, offline_report):
    expected = registration_expected(root, checkpoint, offline_report)
    doc = read_json(workspace_path(root) / f"registration_{TAG}.json")
    require(all(doc.get(k) == v for k, v in expected.items()), "R14 preregistration identity/rule mismatch")
    for path in (REMOTE_ROOT / "diag_brake/d11_run5.py", REMOTE_ROOT / "chk_r10b/states192.json.gz"):
        workspace_path(path)
    return expected


def check_data(root):
    from wms.data.c4_episodes import MANIFEST_NAME
    root = workspace_path(root)
    for name in V5:
        workspace_path(root / "data" / name / MANIFEST_NAME)
    for name, sha in LOWPEN.items():
        require(digest(root / LOWPEN_DIR / name / MANIFEST_NAME) == sha, f"low-share pack manifest drift: {name}")


def preflight(root, checkpoint, offline_report):
    check_configs()
    root = workspace_path(root)
    check_data(root)
    check_13e_b(root, checkpoint)
    check_offline(offline_report)
    from wms.cli.assemble import build_model, build_physics
    from wms.train.settings import TrainSettings
    for stage, cfg in check_configs().items():
        TrainSettings.from_config(cfg)
        model = build_model(cfg, build_physics(cfg))
        g.check_emissions(model, stage)
    artifacts = [root / f".round{TAG[1:]}_gate.lock", root / f"runs/{TAG}_gate_core_tests",
                 root / f"smoke_{TAG}.complete.json", root / f"chk_{TAG}",
                 root / f"runs/{RUN}_a", root / f"runs/{RUN}_b"]
    for stage in ("a", "b"):
        artifacts += [root / f"runs/smoke_{TAG}_{stage}", root / f"smoke_{TAG}_{stage}.log",
                      root / f"smoke_{TAG}_{stage}.gate.json", root / f"production_{TAG}_{stage}.log"]
    for path in artifacts:
        workspace_path(path, exists=False)
        require(not path.exists() and not path.is_symlink(), f"prior R14 artifact exists: {path}")
    return {"ok": True}


def production_ready(root, checkpoint, offline_report, stage):
    """B: A production verdict + A MIDRUN CONTINUE (A postrun is NOT awaited; it gates B's midrun)."""
    registration = check_registration(root, checkpoint, offline_report)
    root, checkpoint = workspace_path(root), workspace_path(checkpoint)
    smoke_a = root / f"runs/smoke_{TAG}_a/wm_s.pt"
    g.artifact_verdict(root, "a", checkpoint)
    g.artifact_verdict(root, "b", smoke_a)
    receipt = read_json(root / f"smoke_{TAG}.complete.json")
    require(receipt == {"registration": registration, "a_sha256": digest(smoke_a),
                        "b_sha256": digest(root / f"runs/smoke_{TAG}_b/wm_s.pt")},
            "both-stage smoke receipt identity mismatch")
    if stage == "b":
        g.artifact_verdict(root, "a", checkpoint, production=True)
        result = read_json(root / f"chk_{TAG}/a/midrun/RESULT.json")
        require(result.get("decision") == "CONTINUE"
                and result.get("registration_sha256") == digest(root / f"registration_{TAG}.json")
                and result.get("checkpoints", {}).get("r10b", {}).get("sha256")
                == digest(root / f"runs/{RUN}_a/wm_s_stage120.pt"),
                "A midrun check missing/failed or checkpoint identity mismatch")
        return {"ok": True, "init_checkpoint": str(root / f"runs/{RUN}_a/wm_s.pt")}
    return {"ok": True, "init_checkpoint": str(checkpoint)}


# ------------------------------------------------------------------ waiter
A_POSTRUN_WAIT_S = 6 * 3600


def a_postrun_continue(root):
    """B midrun precondition: A's parallel postrun finished with CONTINUE (polls, fail closed)."""
    path = root / f"chk_{TAG}/a/postrun/RESULT.json"
    deadline = time.time() + A_POSTRUN_WAIT_S
    while not path.is_file():
        require(time.time() < deadline, "A postrun RESULT never appeared (fail closed)")
        time.sleep(30)
    result = read_json(path)
    require(result.get("decision") == "CONTINUE", f"A postrun decision {result.get('decision')} (stop B)")
    require(result.get("checkpoints", {}).get("r10b", {}).get("sha256") == digest(root / f"runs/{RUN}_a/wm_s.pt"),
            "A postrun did not measure A FINAL")
    return result


def make_waiter(r13c, args):
    stage = args.stage
    install()
    base_waiter = g.make_waiter(r13c, args)
    Base = type(base_waiter)

    class Waiter(Base):
        def measure(self, checkpoint):
            if stage == "b" and self.mode == "midrun":
                self.a_postrun = a_postrun_continue(self.root)
            return measure_r14(self, r13c, checkpoint)

        def snapshot(self, src, name):
            src = workspace_path(src)
            for sibling in ("physics.json", "lane_calib.json"):
                workspace_path(src.with_name(sibling), exists=False)
            checkpoint, info = r13c.Waiter.snapshot(self, src, name)
            if name == "candidate":
                self.candidate_snapshot = checkpoint
            if name == "baseline":
                require(info["sha256"] == read_json(self.root / f"registration_{TAG}.json")["init_sha256"],
                        "baseline is not the registered 13E stage-B FINAL")
                require(getattr(self, "candidate_snapshot", None) is not None,
                        "candidate must be snapshotted before the same-flags reference")
                checkpoint, reference = g.derive_same_flags_reference(
                    checkpoint, self.candidate_snapshot,
                    workspace_path(self.work / "baseline_same_flags", exists=False))
                info = {**info, "same_flags_reference": reference}
            return checkpoint, info

        def command(self, script, *values):
            if Path(str(script)).name == "run_onestep.py":
                values += ("--arm-set", "r9=physics.lane_change_secure_gap=false")
            return Base.command(self, script, *values)

        def execute(self):
            code = r13c.Waiter.execute(self)
            report = workspace_path(self.out / "RESULT.md")
            report.write_text(report.read_text(encoding="utf-8").replace("R13C", f"R14 stage {stage.upper()}")
                              .replace("S-R12C", "13E-B FINAL (same R14 switches)"), encoding="utf-8")
            return code

    return _rebind(Waiter, base_waiter)


def _rebind(cls, instance):
    instance.__class__ = cls
    return instance


def measure_r14(self, r13c, checkpoint):
    """13C measurement math with the 13E stage-B FINAL (same switches) as the paired reference."""
    self.work.mkdir(exist_ok=False)
    panel, provenance = r13c.train_panel(self.root, self.work)
    candidate, candidate_info = self.snapshot(checkpoint, "candidate")
    baseline, baseline_info = self.snapshot(self.root / INIT_REL, "baseline")
    r9, r9_info = self.snapshot(self.root / "runs/c4_wm_v10_round9/wm_s_stage300.pt", "r9")
    if any(r9_info["emissions"].values()):
        raise ValueError("unexpected round9 stored emission flags")
    checkpoints = {"r10b": candidate_info, "r12c": baseline_info, "r9": r9_info}
    tools = self.repo / "tools"
    variables = ["--var", f"r10b={candidate}", "--var", f"r12c={baseline}", "--var", f"r9={r9}"]
    if self.mode == "postrun":
        hz = self.work / "headzero" / "wm_s.pt"
        hz.parent.mkdir()
        receipt = self.out / "headzero_receipt.json"
        self.command(tools / "r10b_checks/make_headzero.py", candidate, hz, receipt)
        report = read_json(receipt)
        if not report.get("all_other_tensors_equal") or not report.get("zeroed_are_zero"):
            raise ValueError("headzero verification failed")
        variables += ["--var", f"headzero={hz}"]
    overrides = {"r10b": candidate_info["emissions"], "r12c": baseline_info["emissions"],
                 "r9": {"driver_param_mode": "legacy", "driver_param_init": "checkpoint",
                        "residual_accel_reinit": False, **dict.fromkeys(r13c.EMISSIONS, False)}}
    if self.mode == "postrun":
        overrides["headzero"] = candidate_info["emissions"]
    arm_sets = []
    for arm, settings in overrides.items():
        for key, value in settings.items():
            arm_sets += ["--arm-set", f"{arm}=model.{key}={str(value).lower()}"]
    limits = ["--limit", "24", "--stride-eps", "10"] if self.mode == "midrun" else ["--stride-eps", "4"]
    os_dir = self.work / "os"
    self.command(tools / "r10b_checks/run_onestep.py", "--repo", self.repo,
                 "--config", "c4_wm_v10round13c.toml", *variables, *arm_sets,
                 "--replay-present", "filled", "--base-cols", "--horizon-s", "120",
                 "--roots", *(self.root / "data" / s for s in r13c.SOURCES),
                 "--workers", r13c.WORKERS, *limits, "--out", os_dir)
    done = read_json(os_dir / "ONESTEP_DONE.json")
    if done.get("replay_present") != "filled" or not done.get("base_cols") or not done.get("windows"):
        raise ValueError("one-step completion/presence/columns missing")
    if self.mode == "midrun" and done["windows"] != 24:
        raise ValueError("midrun requires exactly 24 one-step windows")
    mae = r13c.paired_mae(os_dir, ["r10b", "r12c", "r9"] + (["headzero"] if self.mode == "postrun" else []))
    roll = self.work / "roll"
    self.command(self.root / "diag_brake/d11_run5.py", "--repo", self.repo,
                 "--states", panel, "--limit", "24", "--k", "1", "--steps", "120",
                 "--workers", r13c.WORKERS, "--var", f"r10b={candidate}",
                 "--var", f"r12c={baseline}", "--var", f"r9={r9}", "--out", roll)
    done = read_json(roll / "ROLLOUT_DONE.json")
    if (done.get("n_states"), done.get("k"), done.get("steps")) != (24, 1, 120):
        raise ValueError("invalid rollout completion receipt")
    brakes = r13c.hard_brakes(roll, ("r10b", "r12c", "r9"))
    reasons = r13c.stop_reasons(brakes["r10b"]["share"], brakes["r12c"]["share"],
                                mae["r10b"]["mae"], mae["r12c"]["mae"])
    registration = workspace_path(self.root / f"registration_{TAG}.json")
    return {"decision": "STOP" if reasons else "CONTINUE", "reasons": reasons,
            "labels": {"r10b": f"R14 {self.stage_label} candidate",
                       "r12c": "13E-B FINAL (same R14 switches)", "r9": "as-trained round9"},
            "checkpoints": checkpoints, "provenance": provenance, "one_step": mae,
            "one_step_windows": done.get("windows") or read_json(os_dir / "ONESTEP_DONE.json")["windows"],
            "full_road_60s_hard_brakes": brakes,
            "a_postrun": getattr(self, "a_postrun", {}).get("decision"),
            "registration": read_json(registration), "registration_sha256": digest(registration)}


def wait_registered(args):
    import importlib.util
    check_source()
    path = workspace_path(ROOT / "scripts/round13c_waiter.py")
    spec = importlib.util.spec_from_file_location("r14_registered_waiter", path)
    r13c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(r13c)
    require(not r13c.stop_reasons(.02, .01, .5, .4)
            and len(r13c.stop_reasons(.02001, .01, .50001, .4)) == 2,
            "13C paired stop-rule implementation mismatch")
    waiter = make_waiter(r13c, args)
    waiter.stage_label = args.stage.upper()
    return waiter.execute()


def main(argv=None):
    install()
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("config")
    sub.add_parser("prepare")
    for command in ("registered", "registration-template", "production-ready", "smoke-complete", "preflight"):
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
            print("R14 stage A/B config contracts: ok")
            return 0
        if args.command == "prepare":
            result = prepare()
        elif args.command == "waiter":
            return wait_registered(args)
        elif args.command == "registered":
            result = {"ok": True, "registration": check_registration(
                args.root, args.init_checkpoint, args.offline_report)}
        elif args.command == "registration-template":
            # Prints the exact document an operator must write as registration_r14.json.
            result = {"ok": True, "registration": registration_expected(
                args.root, args.init_checkpoint, args.offline_report)}
        elif args.command == "production-ready":
            result = production_ready(args.root, args.init_checkpoint, args.offline_report, args.stage)
        elif args.command == "smoke-complete":
            registration = check_registration(args.root, args.init_checkpoint, args.offline_report)
            root = workspace_path(args.root)
            a_final = root / f"runs/smoke_{TAG}_a/wm_s.pt"
            g.artifact_verdict(root, "a", workspace_path(args.init_checkpoint))
            g.artifact_verdict(root, "b", a_final)
            write_once(root / f"smoke_{TAG}.complete.json", {
                "registration": registration, "a_sha256": digest(a_final),
                "b_sha256": digest(root / f"runs/smoke_{TAG}_b/wm_s.pt")})
            result = {"ok": True}
        elif args.command == "preflight":
            result = preflight(args.root, args.init_checkpoint, args.offline_report)
        else:
            report = workspace_path(args.report, exists=False)
            try:
                result = g.verdict(args)
            except Exception as exc:
                result = {"ok": False, "errors": [str(exc)]}
            with report.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
        print(json.dumps(result, allow_nan=False))
        return 0 if result["ok"] else 3
    except Exception as exc:
        print(f"R14 gate: bad: {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
