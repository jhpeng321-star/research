"""Source-versioned CPU-only create-once LC continuation through owner freeze."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

BASE = Path("/home/ubuntu/rbfit20261101_addendum")
OLD = Path("/home/ubuntu/rbfit20261031")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("action",choices=("prepare","repair-bootstrap","fit","raw","status"))
    args = parser.parse_args()
    root = args.root
    if root.parent != Path("/home/ubuntu") or not re.fullmatch(r"rbfit20261101_lc_source_[0-9a-f]{12}",root.name):
        raise ValueError("Expected own versioned cpuQ root")
    if args.action == "prepare":
        if not root.parent.is_dir() or not BASE.is_dir():
            raise ValueError("Expected existing parents")
        root.mkdir(exist_ok=False)
        (root/"src/wms/physics").mkdir(parents=True)
        (root/"src/wms/__init__.py").symlink_to(BASE/"src/wms/__init__.py")
        (root/"src/wms/core").symlink_to(BASE/"src/wms/core",target_is_directory=True)
        for name in ("__init__","accel","config","idm","merge_cal","mobil","neighbours",
                     "projection","state","merge_v6","batched","secure_gap"):
            (root/f"src/wms/physics/{name}.py").symlink_to(BASE/f"src/wms/physics/{name}.py")
        for name in ("idm_fit","lane_fit"):
            (root/f"{name}.py").symlink_to(BASE/f"tools/rbfit/{name}.py")
        print(json.dumps({"created":str(root)}))
        return
    if args.action == "repair-bootstrap":
        # Add only the independently identified missing frozen dependency to an
        # existing own pre-dispatch overlay. No source overwrite or job retry.
        if not (root/"src/wms/physics").is_dir() or (root/"lane_fit_process.json").exists():
            raise RuntimeError("Bootstrap repair requires undispatched own root")
        destination = root/"src/wms/physics/secure_gap.py"
        destination.symlink_to(BASE/"src/wms/physics/secure_gap.py")
        print(json.dumps({"added_frozen_dependency":str(destination)}))
        return
    if args.action == "status":
        for name in ("lane_fit","lane_raw"):
            receipt = root/(name+"_process.json")
            if receipt.is_file():
                r = json.loads(receipt.read_text())
                proc = Path("/proc")/str(r.get("pid"))
                ticks = (proc/"stat").read_text().split()[21] if proc.exists() else None
                print(json.dumps({"name":name,"pid":r.get("pid"),"exists":proc.exists(),
                    "same_incarnation":ticks==r.get("proc_start_ticks"),
                    "complete":(root/name/"validation.json").is_file(),
                    "verified":(root/name/"VERIFIED.json").is_file()}))
        return
    if not root.is_dir() or os.getloadavg()[0] > min(16,os.cpu_count()-2):
        raise RuntimeError("Expected-parent/free-load admission failed")
    name = "lane_fit" if args.action == "fit" else "lane_raw"
    receipt,output = root/(name+"_process.json"),root/name
    if receipt.exists() or output.exists():
        raise RuntimeError("Existing dispatch/output; no automatic retry")
    version = json.loads((root/"source_version.json").read_text())
    for relative,expected in version["source_sha256"].items():
        if hashlib.sha256((root/relative).read_bytes()).hexdigest()!=expected:
            raise RuntimeError("Source-version hash mismatch before dispatch")
    if args.action == "fit":
        command = ["lane_directional.py","--root","/home/ubuntu/wmdata20261031/views",
            "--idm-pooled",str(OLD/"pooled_full_v5/idm_pooled.json"),"--out",str(output),
            "--support-json",str(root/"support.json"),"--snapshot-version","v2","--workers","2"]
    else:
        if not (root/"lane_fit/VERIFIED.json").is_file():
            raise RuntimeError("Independently verified completed physical cache required")
        command = ["lane_raw_fit.py","--cache",str(root/"lane_fit"),"--out",str(output)]
    source_paths = [root/command[0],root/"dispatch_lc_version.py",root/"support.json",
                    root/"src/wms/physics/lc2013.py",root/"src/wms/physics/lc2013_geometry.py"]
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    command = ["nice","-n","19",sys.executable,*command]
    environment = dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",PYTHONPATH=str(root/"src"))
    with receipt.open("x") as stream:
        json.dump({"state":"claimed","source_sha256":hashes,"argv":command},stream)
    with (root/(name+".log")).open("xb") as log:
        process = subprocess.Popen(command,cwd=root,env=environment,stdout=log,
                                   stderr=subprocess.STDOUT,start_new_session=True)
    r = {"state":"dispatched","pid":process.pid,"argv":command,"timestamp_unix":time.time(),
        "proc_start_ticks":(Path("/proc")/str(process.pid)/"stat").read_text().split()[21],
        "source_sha256":hashes,"source_version":version,"load":os.getloadavg(),"gpu":False,"automatic_retry":False}
    receipt.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r))


if __name__ == "__main__":
    main()
