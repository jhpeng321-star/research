"""Three separately registered, guarded TRAIN stages.

This is an execution entry point, not an authorization or a source of SHA pins.
Each invocation needs an externally reviewed policy and its raw SHA256.  In
particular a later stage never discovers or registers its own input files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from pathlib import Path

TOOL_DIR = Path(__file__).absolute().parent
WM_DIR = TOOL_DIR.parent
WORKTREE = WM_DIR.parents[2]
SCHEMA = "wm_n1_trainstage_launcher/1"
PLAN_SCHEMA = "wm_n1_trainstage_plan/1"
STAGES = ("metadata-plan", "train-sidecars", "cpu40")
GUARD_FILES = ("sitecustomize.py", "train_stage_guard.py", "run_train_stage.py")
COMMON_CODE = (
    "train/__init__.py", "train/config.py", "train/dataset.py",
    "train/trainer.py", "train/cli.py", "train/label_isolation.py",
    "train/identity_sidecar.py", "features/outcomes.py",
    "model/__init__.py", "model/constants.py",
    "model/net.py", "model/ensemble.py", "model/rollout.py", "model/losses.py",
)
CPU_CODE = (
    "train/idea_a.py", "train/contrast_support.py", "train/g4_real.py",
    "model/outcome_norm.py", "evalh/__init__.py", "evalh/seals.py",
    "evalh/cal_split.py", "arrival/__init__.py", "arrival/head.py",
    "arrival/sampler.py", "hdv/__init__.py", "hdv/config.py", "hdv/net.py",
    "hdv/input_schema.py", "ctrl/__init__.py", "ctrl/config.py",
    "ctrl/mover.py", "ctrl/wall_hook.py", "ctrl/wall_v22.py",
    "corr/__init__.py", "corr/head.py", "corr/ood.py",
    "corr/state_token.py", "corr/summary.py", "features/__init__.py",
    "features/idea_a.py",
)
PACK_NAMES = ("manifest.json", "INDEX.json")
SHARD_COMMON_NAMES = ("DONE.json", "slot_vid.json", "slot_index.npy")
SHARD_OUTCOME_NAME = "outcomes.parquet"
IDENTITY_NAMES = ("TRAIN_IDENTITY_RECEIPT.json", "identities.npy",
                  "global_positions.npy", "vehicle_map.json")
LABEL_NAMES = ("TRAIN_ISOLATION_RECEIPT.json", "train_outcomes.parquet")
PACK_ARRAYS = ("veh_step", "veh_mask", "plan", "wall", "wall_mask", "counter",
               "accel_next", "target_mask", "lane_next", "arrivals", "veh_kin",
               "hdv_base", "cav_plan_ch", "event_tok", "event_mask")
CANONICAL_PACK = Path("D:/rd-staging/wmn1_feat/pack_v3")
CANONICAL_SHARDS = Path("D:/rd-staging/wmn1_feat/shards_v3")
SANDBOX_ROOT = Path("E:/tmp/wmn1_train_runs_20261002")
CPU = {"member": 1, "seed": 101, "device": "cpu", "k_max": 192,
       "batch_size": 8, "epochs": 1, "steps_per_epoch": 40, "log_every": 1}
BATCH_CHUNKS = {"chunk1": "chunk1", "chunk2": "chunk2",
                "chunk3": "chunk3", "chunk4": "chunk4",
                "chunk1_fix27": "chunk1_fix27",
                "truearms_chunk2": "chunk2", "truearms_chunk3": "chunk3",
                "walls_pilot_chunk4": "chunk4"}


class LauncherRefusal(PermissionError):
    """A policy or stage input fails closed before training."""


def require(ok, message):
    if not ok:
        raise LauncherRefusal(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def exact_path(value, *, exists=None):
    require(isinstance(value, str) and value and value == value.strip(),
            "absolute literal path required")
    raw = Path(value)
    require(raw.is_absolute() and ".." not in raw.parts,
            "path must be absolute, literal and traversal-free")
    for part in (raw, *raw.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        require(not stat.S_ISLNK(info.st_mode) and
                not (getattr(info, "st_file_attributes", 0) & 0x400),
                "reparse/symlink path refused")
    if exists == "file":
        require(raw.is_file(), "required regular file missing")
    elif exists == "dir":
        require(raw.is_dir(), "required directory missing")
    elif exists == "fresh":
        require(not raw.exists(), "output must be new")
    return raw


def under(path, root):
    return path == root or root in path.parents


def forbidden_component(path):
    return any(re.search(r"holdout|calibration|cal[_. -]*split|chunk[_. -]*5|j4b|inc1|sealed",
                         str(part), re.IGNORECASE) for part in path.parts)


def policy_inputs(policy):
    stage = policy.get("stage")
    require(stage in STAGES, "unknown stage")
    inputs = policy.get("stage_inputs")
    require(isinstance(inputs, dict), "stage_inputs object required")
    common = {"pack"}
    extra = {"metadata-plan": {"plan_out"},
             "train-sidecars": {"shards_root", "plan_receipt", "selected_shards",
                                "positive_shards",
                                "identity_out", "sidecar_out"},
             "cpu40": {"plan_receipt", "identity_dir",
                       "sidecar_dir", "cpu_out", "num_workers"}}[stage]
    synthetic = inputs.get("synthetic_test_mode", False)
    require(type(synthetic) is bool, "invalid synthetic_test_mode")
    optional = ({"synthetic_test_mode", "synthetic_root"} if synthetic else
                ({"synthetic_test_mode"} if "synthetic_test_mode" in inputs else set()))
    require(set(inputs) == common | extra | optional, "stage_inputs keys differ from frozen stage")
    pack = exact_path(inputs["pack"], exists="dir")
    sandbox = exact_path(policy.get("sandbox"), exists="fresh")
    require(sandbox.drive.lower() == "e:" and sandbox.parent == SANDBOX_ROOT,
            "sandbox must be a fresh direct child of the reserved E: root")
    require(not any(under(sandbox, root) or under(root, sandbox)
                    for root in (WORKTREE, TOOL_DIR, Path(sys.prefix))),
            "sandbox overlaps code/runtime")
    if synthetic:
        synth = exact_path(inputs["synthetic_root"], exists="dir")
        require(synth.drive.lower() == "e:" and synth.parent == SANDBOX_ROOT
                and pack != synth and under(pack, synth)
                and not any(under(synth, root) or under(root, synth)
                            for root in (WORKTREE, Path(sys.prefix), sandbox,
                                         CANONICAL_PACK, CANONICAL_SHARDS)),
                "synthetic fixture must be a narrow disjoint E root")
    else:
        require(pack == CANONICAL_PACK, "canonical pack_v3 required")
    if stage == "train-sidecars":
        shards = exact_path(inputs["shards_root"], exists="dir")
        require(under(shards, synth) if synthetic else shards == CANONICAL_SHARDS,
                "canonical TRAIN shards root required")
    outputs = []
    for key in ("plan_out", "identity_out", "sidecar_out", "cpu_out"):
        if key in inputs:
            target = exact_path(inputs[key], exists="fresh")
            require(target.parent == sandbox, "stage output must be direct fresh sandbox child")
            outputs.append(target)
    require(len(outputs) == len(set(outputs)), "stage outputs must be distinct")
    for key in ("plan_receipt",):
        if key in inputs:
            target = exact_path(inputs[key], exists="file")
            require(target.name == "PLAN_RECEIPT.json" and target.drive.lower() == "e:",
                    "exact E plan receipt required")
    for key in ("identity_dir", "sidecar_dir"):
        if key in inputs:
            target = exact_path(inputs[key], exists="dir")
            require(target.drive.lower() == "e:", "exact E sidecar directory required")
    if "num_workers" in inputs:
        require(type(inputs["num_workers"]) is int and inputs["num_workers"] in (0, 2),
                "frozen worker count is 0 or 2")
    if "selected_shards" in inputs:
        shards = inputs["selected_shards"]
        require(isinstance(shards, list) and shards and len(shards) == len(set(shards))
                and all(isinstance(x, str) and len(Path(x).parts) == 3
                        and ".." not in Path(x).parts and not forbidden_component(Path(x))
                        for x in shards), "invalid selected TRAIN shards")
        positive = inputs["positive_shards"]
        require(isinstance(positive, list) and all(isinstance(x, str) for x in positive),
                "positive outcome shards must be TRAIN paths")
        positive_set = set(positive)
        require(len(positive) == len(positive_set)
                and positive == [x for x in shards if x in positive_set],
                "positive outcome shards must be an ordered selected TRAIN subset")
    return stage, inputs


def _expected_data(stage, inputs, selected):
    pack = Path(inputs["pack"])
    paths = {str(pack / name) for name in PACK_NAMES}
    if stage == "train-sidecars":
        paths.add(inputs["plan_receipt"])
        root = Path(inputs["shards_root"])
        paths |= {str(root.joinpath(*rel.split("/")) / name)
                  for rel in selected for name in SHARD_COMMON_NAMES}
        paths |= {str(root.joinpath(*rel.split("/")) / SHARD_OUTCOME_NAME)
                  for rel in inputs["positive_shards"]}
    elif stage == "cpu40":
        paths.add(inputs["plan_receipt"])
        paths |= {str(pack / (name + ".npy")) for name in PACK_ARRAYS}
        paths |= {str(Path(inputs["identity_dir"]) / name) for name in IDENTITY_NAMES}
        paths |= {str(Path(inputs["sidecar_dir"]) / name) for name in LABEL_NAMES}
    return paths


def load_policy(path, raw_sha):
    source = exact_path(path, exists="file")
    require(under(source, WORKTREE) and source.suffix == ".json", "policy must be a worktree JSON file")
    require(isinstance(raw_sha, str) and re.fullmatch(r"[0-9a-f]{64}", raw_sha),
            "externally reviewed raw policy SHA256 required")
    raw_policy = source.read_bytes()
    require(hashlib.sha256(raw_policy).hexdigest() == raw_sha,
            "reviewed policy SHA mismatch")
    payload = json.loads(raw_policy)
    require(isinstance(payload, dict) and payload.get("version") == 1
            and set(payload) in ({"version", "stage", "sandbox", "stage_inputs", "files",
                                  "interpreter", "pythonpath"},
                                 {"version", "stage", "sandbox", "stage_inputs", "files",
                                  "interpreter", "pythonpath", "helpers"}),
            "unknown policy schema")
    paths = payload.get("stage_inputs")
    require(isinstance(paths, dict), "stage inputs required")
    if "sandbox" in payload:
        payload["sandbox"] = str(Path(payload["sandbox"]))
    for key in ("pack", "shards_root", "plan_receipt", "identity_dir",
                "sidecar_dir", "plan_out", "identity_out", "sidecar_out", "cpu_out",
                "synthetic_root"):
        if key in paths:
            paths[key] = str(Path(paths[key]))
    files = payload.get("files")
    require(isinstance(files, dict) and files, "exact file pins required")
    payload["files"] = {str(Path(p)): record for p, record in files.items()}
    require(len(payload["files"]) == len(files), "duplicate normalized file path")
    payload["pythonpath"] = [str(Path(p)) for p in payload.get("pythonpath", [])]
    if isinstance(payload.get("interpreter"), dict) and "path" in payload["interpreter"]:
        payload["interpreter"]["path"] = str(Path(payload["interpreter"]["path"]))
    stage, inputs = policy_inputs(payload)
    require(payload.get("helpers", {}) == {}, "external helpers are outside TRAIN stages")
    files = payload["files"]
    seen = set()
    for name, record in files.items():
        file = exact_path(name, exists="file")
        require(str(file) not in seen and isinstance(record, dict)
                and set(record) == {"role", "sha256"}, "invalid file registration")
        seen.add(str(file))
        require(record["role"] in ("code", "config", "data")
                and isinstance(record["sha256"], str)
                and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]),
                "invalid role or externally pinned SHA")
        if record["role"] == "code":
            require(file.suffix == ".py" and under(file, WORKTREE),
                    "code registration outside this worktree")
        elif record["role"] == "data":
            require(not forbidden_component(file), "protected data registration")
        else:
            require(file == WM_DIR / "REGISTRATION_WM_N1_AMENDMENT_6_STATE_TOKEN.md"
                    and stage == "cpu40", "only the frozen A6 document is a config")
    required_code = ({str(TOOL_DIR / name) for name in GUARD_FILES} |
                     {str(WM_DIR / name) for name in COMMON_CODE} |
                     ({str(WM_DIR / name) for name in CPU_CODE}
                      if stage == "cpu40" else set()))
    if stage == "cpu40":
        required_code.add(str(WM_DIR.parent / "wdpaths.py"))
    if stage == "metadata-plan" and inputs.get("synthetic_test_mode") is True:
        required_code.add(str(TOOL_DIR / "tests" / "synthetic_startup_probe.py"))
    require(required_code == {p for p, r in files.items() if r["role"] == "code"},
            "code pins differ from exact reviewed project/guard source closure")
    if stage == "cpu40":
        a6 = str(WM_DIR / "REGISTRATION_WM_N1_AMENDMENT_6_STATE_TOKEN.md")
        require(files.get(a6, {}).get("role") == "config",
                "frozen A6 document needs external config pin")
    if stage != "metadata-plan":
        require(str(Path(inputs["plan_receipt"])) in files
                and files[str(Path(inputs["plan_receipt"]))]["role"] == "data",
                "external plan receipt pin required")
    interp = payload.get("interpreter")
    require(isinstance(interp, dict) and set(interp) == {"path", "sha256"}
            and interp["path"] == sys.executable
            and re.fullmatch(r"[0-9a-f]{64}", interp["sha256"]),
            "reviewed interpreter pin required")
    pythonpath = payload.get("pythonpath")
    require(isinstance(pythonpath, list) and len(pythonpath) == len(set(pythonpath))
            and str(TOOL_DIR) in pythonpath and str(WM_DIR) in pythonpath
            and all(under(exact_path(p, exists="dir"), WORKTREE) for p in pythonpath),
            "reviewed project import paths required")
    if stage == "metadata-plan" and inputs.get("synthetic_test_mode") is True:
        require(str(TOOL_DIR / "tests") in pythonpath,
                "synthetic startup probe needs its exact reviewed import path")
    return payload


def verify_nondatapins(policy):
    for path, record in policy["files"].items():
        if record["role"] != "data":
            require(digest(path) == record["sha256"], "reviewed code/config drift")
    require(digest(policy["interpreter"]["path"]) ==
            policy["interpreter"]["sha256"], "interpreter drift")


def verify_data_registrations(policy):
    stage, inputs = policy_inputs(policy)
    selected = inputs.get("selected_shards", [])
    expected = _expected_data(stage, inputs, selected)
    actual = {p for p, r in policy["files"].items() if r["role"] == "data"}
    require(actual == expected, "stage data pins differ from exact allowed files")


def plan_and_selection(policy):
    """Use a pinned prior receipt; only the guarded child runs `_plan`."""
    stage, inputs = policy_inputs(policy)
    if stage == "metadata-plan":
        return None, []
    receipt_path = Path(inputs["plan_receipt"])
    require(digest(receipt_path) == policy["files"][str(receipt_path)]["sha256"],
            "external plan receipt pin drift")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    require(isinstance(receipt, dict) and set(receipt) ==
            {"schema", "pack", "plan", "selection_sha256"}
            and receipt["schema"] == PLAN_SCHEMA and receipt["pack"] == inputs["pack"],
            "unknown or unbound plan receipt")
    plan = receipt["plan"]
    require(isinstance(plan, dict) and set(plan) ==
            {"manifest_sha256", "index_sha256", "k_max", "n_items", "selection"}
            and plan["k_max"] == 192 and type(plan["n_items"]) is int
            and plan["n_items"] > 0 and isinstance(plan["selection"], list)
            and plan["selection"], "invalid K=192 TRAIN plan")
    sha = policy["files"]
    require(plan["manifest_sha256"] == sha[str(Path(inputs["pack"]) / "manifest.json")]["sha256"]
            and plan["index_sha256"] == sha[str(Path(inputs["pack"]) / "INDEX.json")]["sha256"]
            and receipt["selection_sha256"] == canonical_sha(plan),
            "plan metadata/selection SHA differs from external pins")
    keys = {"position", "tick", "plan_row", "key", "batch", "chunk", "world", "arm",
            "split", "t0_s", "tick_start", "tick_end", "n_ticks", "dec_start",
            "dec_end", "n_outcome_rows", "input_sha256", "shard_rel"}
    selected, positions, identities = [], [], set()
    for item in plan["selection"]:
        require(isinstance(item, dict) and set(item) == keys and item["split"] == "TRAIN",
                "unrecognized/non-TRAIN plan selection")
        for name in ("batch", "chunk", "world", "arm"):
            value = item[name]
            require(isinstance(value, str) and value and value not in (".", "..")
                    and not re.search(r"[\\/:\x00-\x1f]", value)
                    and not forbidden_component(Path(value)), "protected plan identity")
        require(BATCH_CHUNKS.get(item["batch"]) == item["chunk"],
                "unknown/inconsistent TRAIN batch/chunk")
        require(item["key"] ==
                f'{item["batch"]}/{item["world"]}__{item["arm"]}',
                "plan key drift")
        rel = item["shard_rel"]
        require(rel in (f'{item["batch"]}/{item["world"]}/{item["arm"]}',
                        f'{item["chunk"]}/{item["world"]}/{item["arm"]}')
                and not forbidden_component(Path(rel)), "plan shard identity drift")
        for name in ("position", "tick", "plan_row", "tick_start", "tick_end",
                     "n_ticks", "dec_start", "dec_end", "n_outcome_rows"):
            require(type(item[name]) is int and item[name] >= 0, "invalid plan index")
        require(item["position"] < plan["n_items"]
                and item["tick_start"] <= item["tick"] < item["tick_end"]
                and item["dec_start"] <= item["plan_row"] < item["dec_end"]
                and item["tick_end"] == item["tick_start"] + item["n_ticks"]
                and type(item["t0_s"]) in (int, float)
                and math.isfinite(item["t0_s"])
                and re.fullmatch(r"[0-9a-f]{64}", item["input_sha256"]),
                "invalid TRAIN ranges/hash")
        identity = (item["batch"], item["world"], item["arm"])
        require(identity not in identities, "duplicate TRAIN identity")
        identities.add(identity)
        positions.append(item["position"])
        selected.append(rel)
    require(positions == sorted(set(positions)) and len(selected) == len(set(selected)),
            "TRAIN selection duplicated or out of order")
    if stage == "train-sidecars":
        require(selected == inputs["selected_shards"],
                "reviewed selected_shards differ from pinned plan")
        require([item["shard_rel"] for item in plan["selection"]
                 if item["n_outcome_rows"] > 0] == inputs["positive_shards"],
                "positive outcome shards differ from pinned plan")
    return plan, selected


def validate_stage_data(policy, plan, selected):
    stage, inputs = policy_inputs(policy)
    files = policy["files"]
    expected = _expected_data(stage, inputs, selected)
    actual = {p for p, r in files.items() if r["role"] == "data"}
    require(actual == expected, "stage data pins differ from exact allowed files")
    require(all(not forbidden_component(Path(p)) for p in actual), "protected data path")
    if stage == "train-sidecars":
        # Only a stat of the exact, plan-selected zero-row leaf: no outcome
        # content is opened or hashed, and it is never registered as data.
        root = Path(inputs["shards_root"])
        for item in plan["selection"]:
            if item["n_outcome_rows"] == 0:
                leaf = root.joinpath(*item["shard_rel"].split("/")) / SHARD_OUTCOME_NAME
                require(not os.path.lexists(leaf),
                        "zero-row TRAIN shard has an unexpected outcomes file")
    for path in sorted(actual):
        exact_path(path, exists="file")
    # All source files, including the three startup files, are externally
    # pinned.  No missing code hash is silently filled from current bytes.
    if plan is not None:
        require(plan["manifest_sha256"] == files[str(Path(inputs["pack"]) / "manifest.json")]["sha256"]
                and plan["index_sha256"] == files[str(Path(inputs["pack"]) / "INDEX.json")]["sha256"],
                "metadata SHA differs from registered pins")
    verified = {}
    for path in sorted(actual):
        got = digest(path)
        require(got == files[path]["sha256"], "selected registered data SHA drift")
        verified[path] = got
    return verified


def child_spec(policy, plan, verified):
    stage, inputs = policy_inputs(policy)
    sandbox = policy["sandbox"]
    e_roots = [] if not inputs.get("synthetic_test_mode") else [inputs["synthetic_root"]]
    return {"version": 1, "stage": stage, "stage_inputs": inputs,
            "selection_sha256": (None if plan is None else canonical_sha(plan)),
            "verified_data": verified,
            "sandbox": sandbox, "sandbox_tmp": str(Path(sandbox) / "tmp"),
            "roots": [str(WORKTREE)], "e_input_roots": e_roots,
            "files": policy["files"], "helpers": policy.get("helpers", {}),
            "pythonpath": policy["pythonpath"], "interpreter": policy["interpreter"],
            "threads": 4}


def child_env(spec, spec_path, spec_sha):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("WM_N1_")}
    for key in ("PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(key, None)
    env.update({"PYTHONPATH": os.pathsep.join(spec["pythonpath"]),
                "WM_TRAINSTAGE_REQUIRE_GUARD": "1",
                "WM_TRAINSTAGE_POLICY_SPEC_PATH": str(spec_path),
                "WM_TRAINSTAGE_POLICY_SHA256": spec_sha,
                "WD_FORBID_D": "1", "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                "PYTEST_ADDOPTS": "", "TMP": spec["sandbox_tmp"],
                "TEMP": spec["sandbox_tmp"], "TMPDIR": spec["sandbox_tmp"],
                "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4"})
    if spec["stage"] == "cpu40":
        env["WM_N1_CANONICAL_PACK"] = spec["stage_inputs"]["pack"]
        env["WM_N1_TRAIN_IDENTITIES"] = spec["stage_inputs"]["identity_dir"]
        env["WM_N1_TRAIN_OUTCOMES"] = spec["stage_inputs"]["sidecar_dir"]
    return env


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def _write_spec(path, spec):
    raw = json.dumps(spec, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True, allow_nan=False).encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def run_stage_child():
    guard = sys.modules.get("train_stage_guard")
    require(guard is not None, "sitecustomize guard absent")
    guard.assert_startup(strict_pid=True)
    guard.ensure_required_wrappers()
    from train import identity_sidecar as ids
    inputs = guard._POLICY.spec["stage_inputs"]
    stage = guard._POLICY.stage
    plan = ids._plan(inputs["pack"])
    require(plan["k_max"] == 192 and plan["selection"]
            and all(s["split"] == "TRAIN" for s in plan["selection"]),
            "guarded child refuses non-K192/non-TRAIN plan")
    if stage != "metadata-plan":
        require(canonical_sha(plan) == guard._POLICY.spec["selection_sha256"],
                "child plan changed from parent")
    if stage == "metadata-plan":
        out = Path(inputs["plan_out"])
        out.mkdir(exist_ok=False)
        _write_new(out / "PLAN_RECEIPT.json",
                   {"schema": PLAN_SCHEMA, "pack": inputs["pack"], "plan": plan,
                    "selection_sha256": canonical_sha(plan)})
        summary = {"plan_receipt": str(out / "PLAN_RECEIPT.json"),
                   "n_selected": len(plan["selection"]), "k_max": plan["k_max"]}
        if inputs.get("synthetic_test_mode") is True:
            from synthetic_startup_probe import probe_inherited_startup
            summary["startup_probe"] = probe_inherited_startup()
    elif stage == "train-sidecars":
        require([s["shard_rel"] for s in plan["selection"]] == inputs["selected_shards"],
                "TRAIN selection drift")
        require([s["shard_rel"] for s in plan["selection"] if s["n_outcome_rows"] > 0]
                == inputs["positive_shards"], "TRAIN outcome presence drift")
        def registered_sha(path):
            record = guard._POLICY.files.get(guard._lexnorm(str(path)))
            require(isinstance(record, dict) and record.get("role") == "data"
                    and isinstance(record.get("sha256"), str),
                    "selected TRAIN source lacks exact registered pin")
            return record["sha256"]
        root = Path(inputs["shards_root"])
        for sel in plan["selection"]:
            shard = root.joinpath(*sel["shard_rel"].split("/"))
            done_path = ids._file(shard, "DONE.json")
            require(ids._sha(done_path) == registered_sha(done_path),
                    "DONE changed from registered pin")
            done = ids._json(done_path)
            require(done.get("schema") == "wm_n1_features_index/1"
                    and ids._relative_shard(done) == sel["shard_rel"],
                    "DONE schema/path mismatch")
            for key in ("batch", "chunk", "world", "arm", "split", "t0_s",
                        "n_ticks", "n_outcome_rows", "input_sha256"):
                require(done.get(key) == sel[key], "DONE selected identity mismatch")
            outputs = done.get("outputs")
            require(isinstance(outputs, dict), "DONE outputs missing")
            for name, key in (("slot_index.npy", "slot_index"),
                              ("slot_vid.json", "slot_vid.json")):
                path = ids._file(shard, name)
                got = ids._sha(path)
                require(got == outputs.get(key) and got == registered_sha(path),
                        "DONE source output hash mismatch")
            outcome = shard / SHARD_OUTCOME_NAME
            if sel["n_outcome_rows"] == 0:
                require(SHARD_OUTCOME_NAME not in outputs,
                        "zero-row DONE falsely records an outcomes file")
                require(not os.path.lexists(outcome),
                        "zero-row TRAIN shard has an unexpected outcomes file")
            else:
                expected_sha = registered_sha(outcome)
                outcome_sha = outputs.get(SHARD_OUTCOME_NAME)
                require(isinstance(outcome_sha, str)
                        and re.fullmatch(r"[0-9a-f]{64}", outcome_sha)
                        and outputs[SHARD_OUTCOME_NAME] == expected_sha,
                        "positive-row DONE outcome SHA differs from registered pin")
                require(ids._sha(ids._file(shard, SHARD_OUTCOME_NAME)) == expected_sha,
                        "positive-row TRAIN outcome hash mismatch")
        identity_receipt = ids.build_identity_sidecar(inputs["pack"], inputs["shards_root"],
                                                       inputs["identity_out"])
        from train import label_isolation
        label_receipt = label_isolation.build_sidecar(inputs["pack"], inputs["shards_root"],
                                                       inputs["sidecar_out"])
        summary = {"identity_receipt": str(Path(inputs["identity_out"]) /
                                           "TRAIN_IDENTITY_RECEIPT.json"),
                   "label_receipt": str(Path(inputs["sidecar_out"]) /
                                        "TRAIN_ISOLATION_RECEIPT.json"),
                   "n_vehicles": identity_receipt["n_vehicles"],
                   "n_label_rows": label_receipt["sidecar"]["n_rows"]}
    elif stage == "cpu40":
        require(plan["k_max"] == 192 and os.environ.get("WM_N1_TRAIN_IDENTITIES") ==
                inputs["identity_dir"] and os.environ.get("WM_N1_TRAIN_OUTCOMES") ==
                inputs["sidecar_dir"], "CPU sidecar binding drift")
        from evalh import seals
        seals.require_frozen_amendment_6()
        # Verify both receipts and exact TRAIN row/identity coverage before the
        # first dataframe or packed array read.
        ids.load_identity_sidecar(inputs["pack"], inputs["identity_dir"])
        from train import label_isolation
        label_isolation._validate(inputs["pack"], inputs["sidecar_dir"])
        import torch
        torch.set_num_threads(4)
        from train import idea_a
        out = Path(inputs["cpu_out"])
        out.mkdir(exist_ok=False)
        frozen = idea_a.IdeaATrainConfig(member=CPU["member"],
                                        out_dir=str(out), ensemble_k=5,
                                        epochs=CPU["epochs"],
                                        steps_per_epoch=CPU["steps_per_epoch"],
                                        k_max=CPU["k_max"],
                                        num_workers=inputs["num_workers"],
                                        log_every=CPU["log_every"])
        require(frozen.batch_size == CPU["batch_size"] and
                frozen.device == CPU["device"] and frozen.state_global_token is True
                and frozen.quick_tune is False and frozen.allow_incomplete_losses is False
                and frozen.subset_frac == 1.0 and frozen.epoch_frac == 1.0,
                "canonical Idea A defaults differ from frozen CPU contract")
        argv = ["--member", "1", "--seed", "101", "--corpus", inputs["pack"],
                "--out", str(out), "--device", "cpu", "--ensemble-k", "5",
                "--epochs", "1", "--steps-per-epoch", "40", "--k-max", "192",
                "--num-workers", str(inputs["num_workers"]), "--log-every", "1"]
        require(idea_a.main(argv) == 0, "canonical Idea A CLI failed")
        receipt = json.loads((out / "TRAIN_RECEIPT.json").read_text(encoding="utf-8"))
        require(receipt.get("member") == 1 and receipt.get("seed") == 101
                and receipt.get("steps") == 40 and receipt.get("epochs") == 1
                and receipt.get("num_workers") == inputs["num_workers"]
                and receipt.get("state_global_token") is True
                and receipt.get("quick_tune") is False
                and receipt.get("allow_incomplete_losses") is False
                and set(receipt.get("active_loss_terms", [])) ==
                    set(receipt.get("required_loss_terms", []))
                and len(receipt.get("required_loss_terms", [])) == 5,
                "canonical Idea A receipt failed frozen CPU contract")
        summary = {"member": 1, "seed": 101, "steps_requested": 40,
                   "device": "cpu", "num_workers": inputs["num_workers"],
                   "train_receipt": str(out / "TRAIN_RECEIPT.json")}
    else:
        raise LauncherRefusal("unknown guarded stage")
    return summary


_CHILD = ("import json, os, sys, traceback\n"
          "try:\n"
          " import train_stage_guard as guard\n"
          " guard.assert_startup(strict_pid=True)\n"
          " guard.ensure_required_wrappers()\n"
          " import run_train_stage\n"
          " result = run_train_stage.run_stage_child()\n"
          " with open(sys.argv[1], 'x', encoding='utf-8') as stream:\n"
          "  json.dump(result, stream, sort_keys=True, indent=2)\n"
          "except BaseException as exc:\n"
          " sys.stderr.write('[wm-trainstage] child refused: %s: %s\\n' % (type(exc).__name__, exc))\n"
          " traceback.print_exc(file=sys.stderr)\n"
          " sys.stderr.flush()\n"
          " os._exit(87)\n")


def launch(policy):
    verify_data_registrations(policy)
    verify_nondatapins(policy)
    plan, selected = plan_and_selection(policy)
    verified = validate_stage_data(policy, plan, selected)
    spec = child_spec(policy, plan, verified)
    sandbox = Path(spec["sandbox"])
    sandbox.mkdir(exist_ok=False)
    (sandbox / "tmp").mkdir(exist_ok=False)
    (sandbox / "receipts").mkdir(exist_ok=False)
    spec_path = sandbox / "child_spec.json"
    spec_sha = _write_spec(spec_path, spec)
    _write_new(sandbox / "receipts" / "STARTED.json",
               {"schema": SCHEMA, "stage": spec["stage"],
                "selection_sha256": spec["selection_sha256"],
                "policy_spec_sha256": spec_sha})
    receipt_path = sandbox / "receipts" / "RESULT.json"
    log_path = sandbox / "child.log"
    try:
        with log_path.open("x", encoding="utf-8") as log:
            proc = subprocess.run([policy["interpreter"]["path"], "-c", _CHILD,
                                   str(receipt_path)], cwd=sandbox,
                                  env=child_env(spec, spec_path, spec_sha),
                                  stdout=log, stderr=subprocess.STDOUT, check=False)
        if proc.returncode:
            raise LauncherRefusal(f"guarded child exited {proc.returncode}")
        require(receipt_path.is_file(), "guarded child produced no result receipt")
        _write_new(sandbox / "receipts" / "SUCCEEDED.json",
                   {"schema": SCHEMA, "stage": spec["stage"],
                    "result_sha256": digest(receipt_path)})
        return sandbox
    except BaseException as exc:
        _write_new(sandbox / "receipts" / "FAILED.json",
                   {"schema": SCHEMA, "stage": spec["stage"],
                    "error_type": type(exc).__name__, "message": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-sha256", required=True)
    args = parser.parse_args(argv)
    policy = load_policy(args.policy, args.policy_sha256)
    sandbox = launch(policy)
    print(str(sandbox))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
