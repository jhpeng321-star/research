"""gpuR-only, conditional MAIN14 re-arm. No process signals; STOP_ARM is the only disarm."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from datetime import datetime
from pathlib import Path

V5 = Path("/root/autodl-tmp/c4/v5")
ARM = V5 / "main14_arm"
CODE = V5 / "repo_main14"
PY = "/root/miniconda3/bin/python"
CPU_PY = "/home/ubuntu/c4/venv/bin/python"
DEADLINE = datetime.fromisoformat("2026-10-26T09:00:00+08:00").timestamp()
BAD = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth", "vtype")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def identity(pid):
    try:
        f = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return None if f[0] == "Z" else f[19]
    except FileNotFoundError:
        return None


def atomic(path, value):
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(p)


def cpu(code, data=None):
    import base64
    payload = base64.b64encode(code.encode()).decode()
    cmd = f'{CPU_PY} -c "import base64;exec(base64.b64decode(\'{payload}\'))"'
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "cpuX", cmd],
                       input=data, capture_output=True, timeout=180, check=False)
    if p.returncode:
        raise RuntimeError(p.stderr.decode() + p.stdout.decode())
    return p.stdout


def extract(bundle, root):
    assert root.parent.is_dir()
    root.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as t:
        for m in t:
            assert m.isfile() and not any(b in m.name.lower() for b in BAD)
            assert (root / m.name).resolve().is_relative_to(root.resolve())
            t.extract(m, root)


def verify(root, manifest):
    for name, expected in manifest.items():
        assert not any(b in name.lower() for b in BAD)
        assert sha((root / name).read_bytes()) == expected, name


def eligible(decision):
    gap = decision["absolute_gap_reduction_ci95"]
    contrast = decision["vacate_contrast_per_cav_h_ci95"]
    sign = (contrast["on"][0] > 0) - (contrast["on"][0] < 0) == (contrast["sumo"][0] > 0) - (contrast["sumo"][0] < 0)
    return (decision.get("adopt") is True and decision.get("adopt_scientific") is True
            and (sign or gap[1] > 0)
            and decision["native_counts"]["counts_move_toward_sumo"] is True
            and decision["native_counts"]["n"] == 33
            and decision["qualification"]["complete"] is True
            and not decision["qualification"]["pass_to_fail"]
            and all(v["bit_identical"] for v in decision["parity"]["synthetic_bit_identity"].values())
            and decision["decision_time"] < DEADLINE)


def run(argv, cwd, env, log):
    with log.open("ab") as stream:
        p = subprocess.run(list(map(str, argv)), cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                           stdout=stream, stderr=subprocess.STDOUT, timeout=1800, check=False)
    assert p.returncode == 0, f"prearm command failed: {log} rc={p.returncode}"


def execute(package):
    assert V5.is_dir() and Path(PY).exists(), "gpuR runtime required"
    assert time.time() < DEADLINE
    req = json.loads((package / "request.json").read_text())
    decision_bytes = cpu("from pathlib import Path;import sys;sys.stdout.buffer.write((Path.home()/'nativelane/DECISION.json').read_bytes())")
    decision = json.loads(decision_bytes)
    assert eligible(decision), "registered adoption rule not met"
    old_arm_bytes = (ARM / "ARMED.json").read_bytes()
    assert sha(old_arm_bytes) == req["expected_old_arm_sha256"]
    old = json.loads(old_arm_bytes)
    assert sha(Path(old["registration"]).read_bytes()) == req["expected_old_registration_sha256"]
    assert sha(Path(old["config"]).read_bytes()) == req["expected_old_config_sha256"]
    old_cfg = json.loads(Path(old["config"]).read_text())
    owner = req["expected_old_waiter"]
    assert identity(owner["pid"]) == owner["start_ticks"], "old waiter changed"
    assert not any((ARM / n).exists() for n in ("LAUNCHED.json", "WAIT_OWNER.json", "EXPIRED.json", "STOP_ARM"))
    assert not (Path(old_cfg["loop_root"]) / "LOOP14_STATE.json").exists()
    assert shutil.disk_usage(V5).free > 1.5 * 1024**3
    assert sha(Path(old_cfg["wm"]["initial_wm"]).read_bytes()) == "952ec059de5a9c0285d2327bcea57c836fd5d736c9975592ffbb932a506e84c8"
    bundle = (package / "source.tar.gz").read_bytes()
    assert sha(bundle) == req["source_bundle_sha256"]
    candidate = ARM / ("nativelane_candidate_" + req["code_sha"][:12])
    extract(bundle, candidate)
    verify(candidate, req["source_files"])
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("candidate_main14", candidate / "scripts/c4/main14_autolaunch.py")
    sys.path.insert(0, str(candidate / "scripts/c4"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    env = module.environment(old_cfg, ARM)
    env.update(PYTHONPATH=f"{candidate}/src:{candidate}", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    # Tests and real-WM two-update smoke complete BEFORE touching STOP_ARM.
    run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", *["tests/unit/" + t for t in module.TESTS]],
        candidate, env, candidate / "prearm_tests.log")
    smoke = candidate / "prearm_smoke"
    run([PY, "-m", "wms.control.policy14_cli", "--config", "configs/c4_policy14_main14_nativelane.toml",
         "--wm-ckpt", old_cfg["wm"]["initial_wm"], "--wm-config", old_cfg["wm"]["initial_wm_config"],
         "--real-returns", old_cfg["policy"]["initial_real_returns"], "--device", "cpu", "--compact",
         "--smoke", "--max-updates", "2", "--output-dir", smoke], candidate, env, candidate / "prearm_smoke.log")
    summary = json.loads((smoke / "RUN_SUMMARY.json").read_text())
    assert len(summary.get("updates", [])) == 2, "two smoke updates required"
    assert summary.get("ramp_merge", {}).get("native_lane_change") is True, "smoke flag missing"
    assert time.time() < DEADLINE - 300, "insufficient time for verified disarm/re-arm before 09:00"
    assert identity(owner["pid"]) == owner["start_ticks"]
    assert sha((ARM / "ARMED.json").read_bytes()) == req["expected_old_arm_sha256"]
    assert not (ARM / "LAUNCHED.json").exists()
    panel = (ARM / "gate_panel.json").read_bytes()
    assert sha(panel) == "3eaa69a170c17302660d12dd4d7fddb4657e43fa334eece17844dd0fa6c0411e"
    (ARM / "STOP_ARM").touch()
    until = min(time.time() + 180, DEADLINE)
    while identity(owner["pid"]) == owner["start_ticks"] and time.time() < until:
        time.sleep(2)
    assert identity(owner["pid"]) != owner["start_ticks"], "documented disarm not yet complete"
    archive = V5 / "main14_arm_before_nativelane"
    assert not archive.exists()
    # Retain the prior source installation as well as its arm, config and registration.
    with tarfile.open(ARM / "prior_source.tar.gz", "w:gz") as t:
        for name, expected in old["source_files"].items():
            assert not any(b in name.lower() for b in BAD)
            data = (CODE / name).read_bytes()
            assert sha(data) == expected
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            t.addfile(ti, io.BytesIO(data))
    shutil.copyfile(old["registration"], ARM / "previous_REGISTRATION.json")
    shutil.copyfile(old["config"], ARM / "previous_config.json")
    ARM.rename(archive)
    ARM.mkdir()
    (ARM / "gate_panel.json").write_bytes(panel)
    extract(bundle, CODE)
    verify(CODE, req["source_files"])
    cpu_extract = '''
import io,tarfile,sys,hashlib,json
from pathlib import Path
p=Path('/home/ubuntu/c4/repo-main14'); assert p.is_dir()
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as t:
 for m in t:
  assert m.isfile() and (p/m.name).resolve().is_relative_to(p.resolve())
  assert not any(x in m.name.lower() for x in ('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth','vtype'))
  t.extract(m,p)
print('staged')
'''
    cpu(cpu_extract, bundle)
    cpu("import json,hashlib,sys;from pathlib import Path;m=json.load(sys.stdin);p=Path('/home/ubuntu/c4/repo-main14');assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==s for n,s in m.items());print('verified')", json.dumps(req["source_files"]).encode())
    Path(old["registration"]).write_text(req["registration_text"])
    Path(old["config"]).write_text(req["config_text"])
    assert sha(req["registration_text"].encode()) == req["registration_sha256"]
    assert sha(req["config_text"].encode()) == req["config_sha256"]
    new = {**old, "code_sha": req["code_sha"], "source_files": req["source_files"],
           "registration_sha256": req["registration_sha256"], "config_sha256": req["config_sha256"],
           "gate_panel_sha256": sha(panel), "native_lane_decision_sha256": sha(decision_bytes)}
    atomic(ARM / "ARMED.json", new)
    atomic(CODE / "MAIN14_SOURCE.json", {"commit": req["code_sha"], "files": req["source_files"]})
    atomic(ARM / "NATIVELANE_DECISION.json", decision)
    atomic(ARM / "PREARM_VERIFIED.json", {"code_sha": req["code_sha"], "tests": "PASS", "real_wm_smoke_updates": 2,
           "registration_sha256": req["registration_sha256"], "config_sha256": req["config_sha256"],
           "gate_panel_sha256": sha(panel), "previous_installation": str(archive), "time": time.time()})
    argv = [PY, str(CODE / "scripts/c4/main14_autolaunch.py"), "--arm", str(ARM / "ARMED.json")]
    run([*argv, "--check-only"], CODE, {**env, "PYTHONPATH": f"{CODE}/src:{CODE}"}, ARM / "check_only.log")
    assert time.time() < DEADLINE
    with (V5 / "main14_autolaunch.log").open("ab") as log:
        p = subprocess.Popen(["setsid", "nohup", *argv], cwd=CODE, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1"})
    time.sleep(3)
    assert p.poll() is None and (ARM / "waiter.pid").read_text().strip() == str(p.pid)
    fields = Path(f"/proc/{p.pid}/stat").read_text().rsplit(")", 1)[1].split()
    receipt = {"pid": p.pid, "session": int(fields[3]), "ppid": int(fields[1]), "start_ticks": fields[19],
               "code_sha": req["code_sha"], "registration_sha256": req["registration_sha256"],
               "config_sha256": req["config_sha256"], "native_lane_change": True, "time": time.time()}
    atomic(ARM / "WAITER_INSTALLED.json", receipt)
    atomic(ARM / "NATIVELANE_ADOPTED.json", receipt)
    cpu("import sys;from pathlib import Path;(Path.home()/'nativelane/ADOPTED.json').write_bytes(sys.stdin.buffer.read())", json.dumps(receipt, indent=1).encode())
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    args = ap.parse_args()
    try:
        execute(Path(args.package))
    except Exception as error:
        # Before disarm this does not alter the armed installation. After disarm,
        # preserve the evidence and stop for owner review; never retry silently.
        cpu("import sys;from pathlib import Path;(Path.home()/'nativelane/ADOPTION_FAILED.json').write_bytes(sys.stdin.buffer.read())",
            json.dumps({"error": repr(error), "time": time.time()}).encode())
        raise
