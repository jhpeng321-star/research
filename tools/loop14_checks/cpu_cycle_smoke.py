#!/usr/bin/env python3
"""LOOP14 miniature full cycle on a CPU laptop (mechanics fixture, NO scientific use).

    python tools/loop14_checks/cpu_cycle_smoke.py --python <interpreter with torch+onnxruntime+traci> \\
        --wm <dir with S-R12C wm_s.pt, physics.json, policy14_native_returns.npz> \\
        --panel-root <converted TRAIN source for the check panel> --out runs/loop14_smoke/<new>

Drives ``scripts/c4/loop14.py`` through init + all five phases with real commands: policy14
smoke (2 CPU updates), SUMO 1.27.1 collection of 2 exposed development worlds (4107-4108,
registered here as a mechanics fixture) with one 300 s POLICY/B pair each, conversion (val 0.5)
+ real returns, a 2-step 10 s WM fine-tune of ``c4_wm_loop14_ft.toml`` (data overridden to the
collected pairs), and the acceptance check on 2 panel windows.  Writes ``SMOKE.json``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FT_TEST_SETS = [
    'data.c4_base="loop14_pairs_plain"',
    'data.c4_sources=[{name="loop14_pairs_plain", path="loop14_pairs", mix_ratio=0.5, window_stride_s=10.0, '
    'use_val=true}, {name="loop14_pairs", path="loop14_pairs", mix_ratio=0.5, window_stride_s=10.0, '
    'use_val=true, paired=true}]',
    "curriculum.unroll_s=[10.0]", "curriculum.stage_steps=[2]", "curriculum.truncation_steps_per_stage=[20]",
    "curriculum.max_hours=0.25", "optim.lr_per_stage=[3.0e-5]", "optim.batch_size_per_stage=[2]",
    "selection.val_max_windows=2"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--wm", required=True)
    ap.add_argument("--panel-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    wm = Path(a.wm).resolve()
    reg = {"id": "LOOP14-CPU-SMOKE", "status": "registered", "split": "TRAIN",
           "acknowledged_by": "CPU mechanics fixture: exposed development seeds 4107-4108, no scientific use",
           "seed_block": [4107, 4108], "exclude": [],
           "budget": {"max_cycles": 1, "policy_updates_per_cycle": 2, "sim_s_per_cycle": 2000,
                      "sim_s_total": 2000, "gpu_h_policy_per_cycle": 0.5, "gpu_h_wm_per_cycle": 0.5,
                      "wm_steps_per_cycle": 2},
           "stop": {"consecutive_wm_rejects": 2, "max_branch_incidents_per_cycle": 0}}
    (out / "REGISTRATION.json").write_text(json.dumps(reg, indent=1), encoding="utf-8")
    cfg = {"loop_id": "LOOP14-CPU-SMOKE", "loop_root": str(out / "loop"),
           "registration": str(out / "REGISTRATION.json"),
           "hosts": {"gpu": {"python": a.python, "code_root": str(ROOT)},
                     "cpu": {"python": a.python, "code_root": str(ROOT)}},
           "guards": {"min_free_gb": 2.0, "gpu_query": [a.python, "-c", "pass"]},
           "policy": {"config": "configs/c4_policy14.toml", "updates_per_cycle": 2, "seed": 965140,
                      "device": "cpu", "smoke": True, "threads": 2, "timeout_h": 0.5,
                      "initial_real_returns": str(wm / "policy14_native_returns.npz")},
           "collect": {"work_root": str(out / "cpu"), "worlds_per_cycle": 2, "workers": 2, "warmup": 60,
                       "branch_first": 120, "branch_cadence": 60, "branch_points": 1, "branch_horizon": 300,
                       "demands": "2400", "penetrations": "0.2", "pen_weights": "1", "population": "textbook",
                       "timeout_h": 0.5},
           "convert": {"val_frac": 0.5, "mb_per_episode": 8},
           "wm": {"config": "configs/c4_wm_loop14_ft.toml", "initial_wm": str(wm / "wm_s.pt"),
                  "initial_wm_config": "configs/c4_wm_v10round13e.toml", "device": "cpu", "timeout_h": 0.5,
                  "extra_set": FT_TEST_SETS},
           "check": {"roots": [str(Path(a.panel_root).resolve())], "onestep_windows": 2, "rollout_windows": 2,
                     "pair_groups": 2, "stride_eps": 1, "workers": 1, "timeout_h": 0.5}}
    cfg_path = out / "loop.json"
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    loop = [a.python, "-B", str(ROOT / "scripts/c4/loop14.py")]
    steps = []
    for cmd in (["init"], ["plan"], ["next"], ["next"], ["next"], ["next"], ["next"], ["status"]):
        t0 = time.time()
        proc = subprocess.run([*loop, *cmd, "--config", str(cfg_path)], cwd=ROOT, capture_output=True,
                              text=True, check=False)
        steps.append({"cmd": cmd[0], "exit": proc.returncode, "wall_s": round(time.time() - t0, 1),
                      "stdout_tail": proc.stdout.strip().splitlines()[-1:] if proc.stdout.strip() else [],
                      "stderr_tail": proc.stderr.strip().splitlines()[-2:]})
        print(json.dumps(steps[-1]), flush=True)
        if proc.returncode not in (0, 4):
            break
    state = json.loads((out / "loop" / "LOOP14_STATE.json").read_text(encoding="utf-8"))
    ledger = [json.loads(x) for x in (out / "loop" / "cycle_ledger.jsonl").read_text(encoding="utf-8").splitlines()]
    smoke = {"schema": "loop14_cpu_smoke/1", "scientific_use": False, "steps": steps,
             "phases": {p: v["status"] for p, v in state["cycles"][0]["phases"].items()},
             "receipts": {p: v.get("receipt") for p, v in state["cycles"][0]["phases"].items()},
             "accepted_wm": state["accepted_wm"], "policy": state["policy"], "stopped": state["stopped"],
             "sim_s_charged": state["sim_s_charged"], "ledger_events": [e.get("event") for e in ledger]}
    (out / "SMOKE.json").write_text(json.dumps(smoke, indent=1), encoding="utf-8")
    ok = all(s["exit"] in (0, 4) for s in steps) and all(v == "done" for v in smoke["phases"].values())
    print("SMOKE", "PASS" if ok else "FAIL", out / "SMOKE.json")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
