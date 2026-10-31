"""Create-once owned cpuQ LC continuation, independent of completed dispatches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/home/ubuntu/rbfit20261101_addendum")
OLD = Path("/home/ubuntu/rbfit20261031")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("smoke", "fit", "status"))
    args = parser.parse_args()
    if args.action == "status":
        for name in ("lane_directional_smoke_v3", "lane_directional_v3"):
            path = ROOT/(name+"_process.json")
            if path.is_file():
                receipt = json.loads(path.read_text())
                pid = receipt.get("pid")
                proc = Path("/proc")/str(pid)
                print(json.dumps({"name": name, "pid": pid, "exists": proc.exists(),
                    "cmdline": (proc/"cmdline").read_bytes().decode().replace("\0", " ") if proc.exists() else "",
                    "complete": (ROOT/name/"validation.json").is_file()}))
        return
    if not ROOT.is_dir() or not OLD.is_dir() or os.getloadavg()[0] > min(16, os.cpu_count()-2):
        raise RuntimeError("Existing-parent/free-load admission failed")
    smoke = args.action == "smoke"
    name = "lane_directional_smoke_v3" if smoke else "lane_directional_v3"
    receipt_path, output = ROOT/(name+"_process.json"), ROOT/name
    if receipt_path.exists() or output.exists():
        raise RuntimeError("Existing dispatch/output: inspect and preserve; never retry automatically")
    if not smoke:
        report_path = ROOT/"lane_directional_smoke_v3/validation.json"
        if not report_path.is_file():
            raise RuntimeError("Verified smoke required")
        report = json.loads(report_path.read_text())
        if report["n_fit_intervals"] <= 0:
            raise RuntimeError("Empty smoke")
    paths = [ROOT/"tools/rbfit"/n for n in (
        "lane_directional.py", "lane_fit.py", "idm_fit.py", "dispatch_directional.py")]
    paths += [ROOT/"src/wms/physics"/n for n in (
        "lc2013.py", "lc2013_geometry.py", "config.py", "mobil.py", "neighbours.py", "batched.py")]
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if manifest["src/wms/physics/lc2013.py"] != "1bdded8e9f213372db946079b20c0cbac677120eeb228baf91d468644ac37206":
        raise RuntimeError("Physical source changed: register a new version")
    command = ["nice", "-n", "19", sys.executable, "tools/rbfit/lane_directional.py",
        "--root", "/home/ubuntu/wmdata20261031/views", "--idm-pooled",
        str(OLD/"pooled_full_v5/idm_pooled.json"), "--out", str(output),
        "--workers", "1" if smoke else "2", "--sobol-power", "6"]
    if smoke:
        command += ["--worlds", "969400"]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH=str(ROOT/"src"))
    with receipt_path.open("x") as stream:
        json.dump({"state": "claimed", "source_sha256": manifest, "argv": command}, stream)
    with (ROOT/(name+".log")).open("xb") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    proc_start = (Path("/proc")/str(process.pid)/"stat").read_text().split()[21]
    data = {"state": "dispatched", "pid": process.pid, "proc_start_ticks": proc_start,
        "timestamp_unix": time.time(), "argv": command, "source_sha256": manifest,
        "load": os.getloadavg(), "gpu": False, "automatic_retry": False}
    receipt_path.write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
