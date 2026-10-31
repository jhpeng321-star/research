"""Owned create-once raw-physical LC least squares on completed v4 caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/home/ubuntu/rbfit20261101_lc_boundaries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("smoke","fit","status"))
    args = parser.parse_args()
    if args.action == "status":
        for name in ("lane_raw_smoke_v5","lane_raw_v5"):
            path = ROOT/(name+"_process.json")
            if path.is_file():
                receipt = json.loads(path.read_text())
                proc = Path("/proc")/str(receipt.get("pid"))
                print(json.dumps({"name":name,"pid":receipt.get("pid"),"exists":proc.exists(),
                    "cmdline":(proc/"cmdline").read_bytes().decode().replace("\0"," ") if proc.exists() else "",
                    "complete":(ROOT/name/"validation.json").is_file()}))
        return
    if not ROOT.is_dir() or os.getloadavg()[0] > min(16,os.cpu_count()-2):
        raise RuntimeError("Expected-parent/free-load admission failed")
    smoke = args.action == "smoke"
    name = "lane_raw_smoke_v5" if smoke else "lane_raw_v5"
    cache = ROOT/("lane_boundaries_smoke_v4" if smoke else "lane_boundaries_v4")
    if not (cache/"VERIFIED.json").is_file():
        raise RuntimeError("Completed independently verified boundary cache required")
    if not smoke and not (ROOT/"lane_raw_smoke_v5/VERIFIED.json").is_file():
        raise RuntimeError("Verified raw-physics smoke required")
    output,receipt_path = ROOT/name,ROOT/(name+"_process.json")
    if output.exists() or receipt_path.exists():
        raise RuntimeError("Existing dispatch/output; preserve and inspect")
    paths = [ROOT/n for n in ("lane_raw_fit.py","dispatch_raw.py")]
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    command = ["nice","-n","19",sys.executable,"lane_raw_fit.py","--cache",str(cache),"--out",str(output)]
    environment = dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
    with receipt_path.open("x") as stream:
        json.dump({"state":"claimed","source_sha256":manifest,"argv":command},stream)
    with (ROOT/(name+".log")).open("xb") as log:
        process = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
    data = {"state":"dispatched","pid":process.pid,"timestamp_unix":time.time(),"argv":command,
        "proc_start_ticks":(Path("/proc")/str(process.pid)/"stat").read_text().split()[21],
        "source_sha256":manifest,"cache_receipt_sha256":hashlib.sha256((cache/"VERIFIED.json").read_bytes()).hexdigest(),
        "load":os.getloadavg(),"gpu":False,"automatic_retry":False}
    receipt_path.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
