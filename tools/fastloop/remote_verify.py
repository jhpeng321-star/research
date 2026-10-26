"""Export current allowlisted code into OWN remote scratch; test or time, never arm.

Run from the fastloop worktree. Only code/config/unit-test Git objects and explicit
working changes under those prefixes are included. No simulator inputs are exported.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
BLOCK = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth", "vtype")
PREFIXES = ("src/", "traffic_shepherd/", "configs/", "scripts/", "tools/", "tests/", "output/")


def bundle(release=False):
    if release:
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
            raise RuntimeError("--release requires a clean committed worktree")
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT).decode().splitlines()
    files = {}
    for name in names:
        if any(t in name.lower() for t in BLOCK):
            continue
        if not (name.startswith(PREFIXES) or name in ("pyproject.toml", "tests/test_speed2_flags.py")):
            continue
        if Path(name).suffix not in (".py", ".toml", ".json"):
            continue
        if name.startswith("output/") and not name.endswith(".py"):
            continue
        path = ROOT / name
        files[name] = path.read_bytes() if path.is_file() else subprocess.check_output(["git", "show", ":" + name], cwd=ROOT)
    manifest = {"code_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                "release": release, "files": {n: hashlib.sha256(data).hexdigest() for n, data in files.items()}}
    files["FASTLOOP_SOURCE.json"] = json.dumps(manifest).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def base_bundle():
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "9cc39f15", "--", "src"], cwd=ROOT).decode().splitlines()
    names = [n for n in names if not any(t in n.lower() for t in BLOCK)]
    blobs = subprocess.check_output(["git", "cat-file", "--batch"], cwd=ROOT,
                input="".join("9cc39f15:" + n + "\n" for n in names).encode())
    source = io.BytesIO(blobs)
    files = {}
    for name in names:
        header = source.readline().split()
        files[name] = source.read(int(header[2]))
        assert source.read(1) == b"\n"
    # A read-only diagnostic interface needed by the comparison harness; no legacy computation imports it.
    files["src/wms/core/speed_flags.py"] = (ROOT / "src/wms/core/speed_flags.py").read_bytes()
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return out.getvalue()


def remote(host, script, data=None, timeout=1800):
    payload = base64.b64encode(script.encode()).decode()
    py = "/home/ubuntu/miniconda3/bin/python" if host in ("cpuP", "cpuQ") else "/root/miniconda3/bin/python"
    command = shlex.join([py, "-c", f"import base64;exec(base64.b64decode('{payload}'))"])
    if host in ("cpuP", "cpuQ"):
        command = shlex.join(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command])
        host = "gpuR"
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
                          input=data, capture_output=True, timeout=timeout, check=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("stage", "test", "measure", "dry-run", "bitoff"))
    p.add_argument("--host", choices=("gpuR", "gpuB", "cpuP", "cpuQ"), default="gpuB")
    p.add_argument("--release", action="store_true")
    p.add_argument("--scenario-fixture", help="explicit tested fixture directory on CPU; link only, never open assignments")
    args = p.parse_args()
    cpu = args.host in ("cpuP", "cpuQ")
    if cpu and args.action != "stage":
        raise ValueError("CPU deployment supports stage only; no SUMO is launched")
    if cpu and args.release and not args.scenario_fixture:
        raise ValueError("CPU --release needs --scenario-fixture from S-NEWCPU")
    if args.scenario_fixture and any(t in args.scenario_fixture.lower() for t in BLOCK):
        raise ValueError("restricted fixture path")
    scratch = "/home/ubuntu/c4-fastloop-sfastloop" if cpu else "/root/autodl-tmp/c4-fastloop-sfastloop"
    dry_out = scratch + "/dry_run_" + str(time.time_ns())
    if args.action in ("stage", "bitoff"):
        destination = scratch if args.action == "stage" else scratch + "/flagoff_base"
        script = f"""
import io,pathlib,sys,tarfile,json
root=pathlib.Path({destination!r}); assert root.parent.is_dir()
assert not (root/'arm/waiter.pid').exists(), 'refuse mutation after fastloop waiter installation'
root.mkdir(exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as archive:
 for item in archive:
  assert item.isfile() and (root/item.name).resolve().is_relative_to(root.resolve())
  archive.extract(item,root,filter='data')
fixture={args.scenario_fixture!r}
if fixture:
 src=pathlib.Path(fixture); assert src.is_dir()
 dst=root/'scenarios'
 if not dst.exists(): dst.symlink_to(src,target_is_directory=True)
 assert dst.resolve()==src.resolve()
if (root/'FASTLOOP_SOURCE.json').is_file():
 import hashlib
 manifest=json.loads((root/'FASTLOOP_SOURCE.json').read_text())
 for rel,expected in manifest['files'].items(): assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected,rel
print(json.dumps({{'staged':str(root)}}))
"""
        result = remote(args.host, script, bundle(args.release) if args.action == "stage" else base_bundle())
        if args.action == "bitoff" and result.returncode == 0:
            script = f"""
import os,pathlib,subprocess,json
root=pathlib.Path({scratch!r}); py='/root/miniconda3/bin/python'
env={{**os.environ,'PYTHONPATH':str(root/'flagoff_base/src')+':'+str(root),
 'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}}
subprocess.run([py,'tools/speed2_verify.py','--save',str(root/'FLAG_OFF_BASE.pt')],cwd=root,env=env,check=True)
env['PYTHONPATH']=str(root/'src')+':'+str(root)
p=subprocess.run([py,'tools/speed2_verify.py','--compare',str(root/'FLAG_OFF_BASE.pt')],cwd=root,env=env,capture_output=True,text=True)
print(p.stdout); print(p.stderr)
print('RECEIPT_JSON='+json.dumps({{'base':'9cc39f15','status':'PASS' if p.returncode==0 else 'FAIL','comparison':p.stdout.strip()}}))
raise SystemExit(p.returncode)
"""
            result = remote(args.host, script)
    else:
        argv = (["-m", "pytest", "-q", "-p", "no:cacheprovider", "--basetemp", scratch + "/pytest_tmp",
                 "tests/unit/test_fastloop.py", "tests/unit/test_loop14.py", "tests/unit/test_main14.py",
                 "tests/unit/test_native_lane.py", "tests/unit/test_native_adoption.py",
                 "tests/unit/test_policy14_integration.py", "tests/unit/test_policy14_critic.py",
                 "tests/unit/test_policy14_lanes.py", "tests/unit/test_loop14c.py", "tests/test_speed2_flags.py",
                 "tests/unit/test_coopw.py", "tests/unit/test_sumocredit.py", "tests/unit/test_sumocredit_kcheck.py",
                 "tests/unit/test_gate_panel.py", "tests/unit/test_policy14e.py", "tests/unit/test_policy14_r13e.py",
                 "tests/unit/train/test_gpu_memory_cap.py"]
                if args.action == "test" else ["tools/speed2_measure.py", "--case", "policy", "--device", "cuda",
                    "--native-ramp-merge", "wm_law", "--native-cav", "head", "--native-lane-change",
                    "--output", scratch + "/POLICY_TIMING.json"])
        if args.action == "dry-run":
            argv = ["tools/fastloop/dry_run.py", "--out", dry_out]
        script = f"""
import os,subprocess,json,pathlib,time
root=pathlib.Path({scratch!r}); assert root.is_dir()
if hasattr(os,'sched_getaffinity'): os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
if hasattr(os,'nice') and os.nice(0)<19: os.nice(19-os.nice(0))
env={{**os.environ,'PYTHONPATH':str(root/'src')+':'+str(root),'PYTHONDONTWRITEBYTECODE':'1',
 'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}}
if {args.action!r} in ('test','dry-run'): env['CUDA_VISIBLE_DEVICES']=''
child=subprocess.Popen(['/root/miniconda3/bin/python',*{argv!r}],cwd=root,env=env)
peak=0
while child.poll() is None:
 if {args.action!r}=='measure':
  query=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
  for line in query.stdout.splitlines():
   fields=[s.strip() for s in line.split(',')]
   if fields[0]==str(child.pid) and fields[1].isdigit(): peak=max(peak,int(fields[1]))
 time.sleep(.2)
if {args.action!r}=='measure' and (root/'POLICY_TIMING.json').is_file():
 doc=json.loads((root/'POLICY_TIMING.json').read_text()); doc['owned_nvidia_peak_mib']=peak
 assert peak<=4096,'owned GPU use exceeded cap'
 (root/'POLICY_TIMING.json').write_text(json.dumps(doc))
 print('RECEIPT_JSON='+ (root/'POLICY_TIMING.json').read_text())
if {args.action!r}=='dry-run' and pathlib.Path({dry_out!r}+'/DRY_RUN.json').is_file():
 print('RECEIPT_JSON='+ pathlib.Path({dry_out!r}+'/DRY_RUN.json').read_text())
raise SystemExit(child.returncode)
"""
        result = remote(args.host, script)
    print(result.stdout.decode(errors="replace"))
    print(result.stderr.decode(errors="replace"))
    if args.action == "test":
        doc = {"host": args.host, "status": "PASS" if result.returncode == 0 else "FAIL",
               "cuda_visible_devices": "", "stdout": result.stdout.decode(errors="replace"),
               "stderr": result.stderr.decode(errors="replace"), "new_sumo_sim_s": 0}
        (ROOT / "tools/fastloop/UNIT_TESTS.json").write_text(json.dumps(doc, indent=1) + "\n")
    if args.action in ("measure", "dry-run", "bitoff") and result.returncode == 0:
        text = result.stdout.decode().split("RECEIPT_JSON=", 1)[1]
        filename = {"measure": "POLICY_TIMING.json", "dry-run": "DRY_RUN.json", "bitoff": "FLAG_OFF.json"}[args.action]
        target = ROOT / "tools/fastloop" / filename
        target.write_text(json.dumps(json.loads(text), indent=1) + "\n")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
