"""Read/verify only explicitly named own LC jobs; never signal any process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--name",required=True)
    parser.add_argument("action",choices=("inspect","verify"))
    args = parser.parse_args()
    root,name = args.root,args.name
    if root.parent != Path("/home/ubuntu") or not re.fullmatch(r"rbfit20261101_(lc_boundaries|lc_source_[0-9a-f]{12})",root.name):
        raise ValueError("Only explicitly owned continuation roots")
    if name not in ("lane_boundaries_v4","lane_raw_v5","lane_fit","lane_raw"):
        raise ValueError("Only explicitly owned job names")
    path = root/name
    receipt_path = root/(name+"_process.json")
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    proc = Path("/proc")/str(receipt.get("pid")) if receipt else None
    exists = proc.exists() if proc else False
    ticks = (proc/"stat").read_text().split()[21] if exists else None
    complete = all((path/name).is_file() for name in ("validation.json","provenance.json",
        "lane_events.json","lc2013_population.json","lc2013_posteriors.json","lc2013_shrinkage.json"))
    verified = (path/"VERIFIED.json").is_file()
    result = {"root":str(root),"name":name,"claimed":receipt is not None,"pid":receipt.get("pid") if receipt else None,
        "running":exists and ticks==receipt.get("proc_start_ticks"),"pid_reused":exists and ticks!=receipt.get("proc_start_ticks"),
        "complete":complete,"verified":verified,
        "observation_worlds":sum((path/f"observations_{w}.npz").is_file() for w in range(969400,969432))}
    result["frozen_bootstrap_sha256"] = {name:hashlib.sha256((root/"src/wms/physics"/name).read_bytes()).hexdigest()
        for name in ("config.py","secure_gap.py","mobil.py","neighbours.py","batched.py")}
    if args.action == "verify" and complete and not verified:
        claim = root/(name+"_verification_claim.json")
        with claim.open("x") as stream:
            json.dump({"state":"claimed","timestamp_unix":time.time()},stream)
        environment = dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1")
        command = ["nice","-n","19",sys.executable,str(root/"verify_directional.py"),
                   "--fit",str(path),"--out",str(path/"VERIFIED.json")]
        run = subprocess.run(command,env=environment,capture_output=True,text=True)
        claim.write_text(json.dumps({"state":"PASS" if run.returncode==0 else "FAILED",
            "returncode":run.returncode,"stdout":run.stdout,"stderr":run.stderr},indent=2)+"\n")
        if run.returncode:
            raise RuntimeError(f"Verification failed; evidence preserved: {run.stderr}")
        result["verified"] = True
    print(json.dumps(result))


if __name__ == "__main__":
    main()
