"""Prepare a NEW hashed arm request from MAIN14; never install a waiter or start a phase.

Run later on gpuR after committed code is staged on all three hosts. The CPU contract
is supplied from S-NEWCPU's completed setup: explicit paths, no recursive discovery.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/c4")]
import loop14 as loop  # noqa: E402
from loop14_fast import FLAGS  # noqa: E402


def verify_cpu_contract(contract, code_sha):
    code = """import hashlib,json,pathlib,sys
item=json.loads(sys.argv[1]); root=pathlib.Path(item['host']['code_root'])
source=json.loads((root/'FASTLOOP_SOURCE.json').read_text())
assert source['release'] and source['code_sha']==item['code_sha'], 'CPU runtime is not the exact release'
for name,expected in source['files'].items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
assert (root/'scenarios').is_dir() and (root/'scenarios').resolve()==pathlib.Path(item['fixture']).resolve()
for name in item['roots']: assert (pathlib.Path(name)/'C4_EPISODES_MANIFEST.json').is_file(),name
import torch,numpy,traci
print(json.dumps({'source_verified':True,'torch':torch.__version__,'numpy':numpy.__version__}))
"""
    results = {}
    for box in ("cpuP", "cpuQ"):
        h = contract["hosts"][box]
        payload = {"host": h, "code_sha": code_sha, "fixture": contract["scenario_fixture"][box], "roots": contract["check_roots"]}
        argv = [*h["ssh"], shlex.join([h["python"], "-B", "-c", code, json.dumps(payload)])]
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=300, check=False)
        if proc.returncode:
            raise loop.Refused(f"{box} CPU contract not ready: {proc.stderr[-800:]}")
        results[box] = json.loads(proc.stdout)
    return results


def prepare(main_arm, contract, out, deadline, enable, retry_reserve=False):
    original_arm = loop.read_json(main_arm)
    installed = main_arm.parent / "WAITER_INSTALLED.json"
    if installed.exists():
        waiter = loop.read_json(installed)
        if loop.fast_module().process_token(waiter["pid"]) == str(waiter["start_ticks"]):
            raise loop.Refused("old MAIN14 waiter is still live; owner must gracefully disarm it first")
    cfg = copy.deepcopy(loop.read_json(original_arm["config"]))
    if (Path(cfg["loop_root"]) / loop.STATE).exists():
        raise loop.Refused("MAIN14 already initialised: cannot reuse its registered seeds in a new loop")
    reg = copy.deepcopy(loop.read_json(original_arm["registration"]))
    original_reg = copy.deepcopy(reg)
    source = loop.read_json(ROOT / "FASTLOOP_SOURCE.json")
    if not source.get("release"):
        raise loop.Refused("re-arm preparation requires a clean committed --release export")
    if out.exists():
        raise loop.Refused("new arm directory already exists")
    for value in loop._strings(contract):
        loop.guard(value)
        if "FILL_FROM_" in value:
            raise loop.Refused("CPU contract still contains an unresolved setup/model path")
    options = loop.read_json(ROOT / "configs/c4_fastloop_options.json")
    for flag in FLAGS:
        options[flag] = bool(enable)
    options.update(check_roots=contract["check_roots"], check_env=contract["check_env"],
                   verdict_candidates=contract["verdict_candidates"], base_models=contract["base_models"])
    cfg["fastloop"] = options
    for box in ("cpuP", "cpuQ"):
        h = contract["hosts"][box]
        if h["ssh"][-1] != box or h["rsync_target"] != box:
            raise loop.Refused("CPU contract must use exact cpuP/cpuQ aliases")
        if h["code_root"] != "/home/ubuntu/c4-fastloop-sfastloop":
            raise loop.Refused("CPU contract must use the isolated fastloop runtime")
        cfg["hosts"][box] = h
    cfg["hosts"]["cpu"] = copy.deepcopy(cfg["hosts"]["cpuP"])
    cfg["hosts"]["gpu"]["code_root"] = str(ROOT)
    cfg["collect"]["work_root"] = "/home/ubuntu/c4-fastloop-sfastloop/work"
    cfg["collect"]["workers"] = 7
    cfg["loop_root"] = str(out / "loop")
    cfg["registration"] = str(out / "REGISTRATION.json")
    cfg["policy"]["config"] = "configs/c4_policy14_main14_nativelane.toml"
    args = cfg["policy"]["extra_args"]
    args[args.index("--gate-panel") + 1] = str(out / "gate_panel.json")
    args[args.index("--horizon-gate") + 1] = str(out / "horizon_gate_current.json")
    cfg["policy"]["allowed_roots"].append(str(out))
    loop.fast_module().validate(cfg, loop.Refused)
    cpu_verified = verify_cpu_contract(contract, source["code_sha"])
    # Primary budgets remain the registered MAIN14 caps. Retry reserve is NEVER inferred:
    # this command only accepts a separately amended, acknowledged registration supplied later.
    if contract.get("registration"):
        reg = loop.read_json(contract["registration"])
        if reg["id"] != cfg["loop_id"]:
            raise loop.Refused("amendment id changed")
        for key in ("seed_block", "exclude", "stop"):
            if reg[key] != original_reg[key]:
                raise loop.Refused(f"fastloop amendment cannot change {key}")
        for key, value in original_reg["budget"].items():
            if key not in ("sim_s_per_cycle", "sim_s_total") and reg["budget"][key] != value:
                raise loop.Refused(f"fastloop amendment cannot change budget.{key}")
    if retry_reserve:
        if not enable:
            raise loop.Refused("retry reserve requires explicit --enable")
        primary = loop.collect_sim_s(cfg)
        reg["budget"]["sim_s_per_cycle"] = 2 * primary
        reg["budget"]["sim_s_total"] = 2 * primary * int(reg["budget"]["max_cycles"])
        reg["sources"]["retry_reserve_sim_s_per_cycle"] = primary
    reg["sources"].update(code_sha=source["code_sha"], fastloop=True,
                          fastloop_options=options, predecessor_arm_sha256=loop.sha256(main_arm))
    reg["sources"]["base_rule"] = "13E/14b/16: no new HDV reaction FAIL vs 13E; best strict TTC ratio; newer ties"
    reg["sources"]["config_sha256"] = hashlib.sha256((json.dumps(cfg, indent=1) + "\n").encode()).hexdigest()
    when = datetime.fromisoformat(deadline)
    if when.tzinfo is None:
        raise loop.Refused("deadline needs explicit timezone")
    out.mkdir(parents=True, exist_ok=False)
    cfg_path = out / "config.json"
    cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
    Path(cfg["registration"]).write_text(json.dumps(reg, indent=1) + "\n")
    # Preserve the current arm's frozen update-10 panel, without modifying its owner.
    old_cfg = loop.read_json(original_arm["config"])
    old_args = old_cfg["policy"]["extra_args"]
    import shutil
    shutil.copy2(old_args[old_args.index("--gate-panel") + 1], out / "gate_panel.json")
    arm = {**original_arm, "code_sha": source["code_sha"], "source_files": source["files"],
           "config": str(cfg_path), "config_sha256": loop.sha256(cfg_path),
           "registration": cfg["registration"], "registration_sha256": loop.sha256(cfg["registration"]),
           "verdict_deadline": deadline, "gate_panel_sha256": loop.sha256(out / "gate_panel.json")}
    loop.write_atomic(out / "ARM_REQUEST.json", arm)
    loop.write_atomic(out / "CPU_CONTRACT_VERIFIED.json", cpu_verified)
    loop.check_registration(cfg)
    return {"prepared": str(out / "ARM_REQUEST.json"), "armed": False, "launched": False,
            "source_commit": source["code_sha"], "options": options}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--main-arm", required=True, type=Path)
    p.add_argument("--cpu-contract", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--deadline", default="2026-10-26T18:00:00+08:00")
    p.add_argument("--enable", action="store_true", help="explicitly enable fastloop options in the NEW request")
    p.add_argument("--retry-reserve", action="store_true", help="propose one full-cycle-equivalent retry reserve; register amendment before arming")
    args = p.parse_args()
    print(json.dumps(prepare(args.main_arm, loop.read_json(args.cpu_contract), args.out, args.deadline, args.enable, args.retry_reserve)))
