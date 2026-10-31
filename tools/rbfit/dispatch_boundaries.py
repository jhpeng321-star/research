"""Create-once source-domain LC fit; preserves and reuses completed v3 decisions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path("/home/ubuntu/rbfit20261101_lc_boundaries")
PREVIOUS = Path("/home/ubuntu/rbfit20261101_addendum")
OLD = Path("/home/ubuntu/rbfit20261031")


def prepare():
    import numpy as np
    from scipy.stats import qmc
    if not ROOT.parent.is_dir() or not PREVIOUS.is_dir() or not OLD.is_dir():
        raise RuntimeError("Expected existing parents")
    ROOT.mkdir(exist_ok=False)
    # Only own frozen source files; no broad data/source traversal.
    for name in ("idm_fit.py", "lane_fit.py"):
        shutil.copyfile(PREVIOUS/"tools/rbfit"/name,ROOT/name)
    shutil.copyfile(Path(__file__),ROOT/"dispatch_boundaries.py")
    (ROOT/"src").symlink_to(PREVIOUS/"src",target_is_directory=True)
    prior = json.loads((PREVIOUS/"lane_directional_smoke_v3/lc2013_population.json").read_text())
    old = np.array(prior["support"])
    # SUMO source domains: coop [0,1], other behavioral weights nonnegative,
    # assertiveness strictly positive. Zero weights are documented model faces,
    # not hand-selected adopted settings. ML determines every mixture mass.
    points = qmc.Sobol(5,scramble=True,seed=20261031).random_base2(6)
    interior = np.exp(4*points-2)
    interior[:,1] = points[:,1]
    faces = []
    for coordinate, count in ((3,64),(2,16),(1,16),(0,16)):
        face = interior[:count].copy()
        face[:,coordinate] = 0
        if coordinate == 2:
            face[:,3] = 0  # speedGain=0 also disables keep-right source threshold
        faces.append(face)
    support = np.concatenate((old[old[:,1] <= 1],interior,*faces))
    # Preserve the source-default row first for diagnostic comparability.
    _, first = np.unique(support,axis=0,return_index=True)
    support = support[np.sort(first)]
    artifact = {"schema":"wm_phys.lc2013_search_support.v1","support":support.tolist(),
        "note":"Fixed numerical design, source-domain boundary faces; no action-outcome selection",
        "source_reference":"rb-shared/lcspec/spec.md:97-111,261-283",
        "source_defaults_index":0}
    with (ROOT/"support_v4.json").open("x") as stream:
        json.dump(artifact,stream,indent=2)
        stream.write("\n")
    print(json.dumps({"created":str(ROOT),"physical_profiles":len(support)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","smoke","fit","status"))
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
        return
    if args.action == "status":
        for name in ("lane_boundaries_smoke_v4","lane_boundaries_v4"):
            path = ROOT/(name+"_process.json")
            if path.is_file():
                receipt = json.loads(path.read_text())
                proc = Path("/proc")/str(receipt.get("pid"))
                print(json.dumps({"name":name,"pid":receipt.get("pid"),"exists":proc.exists(),
                    "cmdline":(proc/"cmdline").read_bytes().decode().replace("\0"," ") if proc.exists() else "",
                    "complete":(ROOT/name/"validation.json").is_file()}))
        return
    if not ROOT.is_dir() or os.getloadavg()[0] > min(16,os.cpu_count()-2):
        raise RuntimeError("Existing-parent/free-load admission failed")
    smoke = args.action == "smoke"
    name = "lane_boundaries_smoke_v4" if smoke else "lane_boundaries_v4"
    receipt_path, output = ROOT/(name+"_process.json"),ROOT/name
    if receipt_path.exists() or output.exists():
        raise RuntimeError("Existing dispatch/output; preserve and inspect")
    previous_name = "lane_directional_smoke_v3" if smoke else "lane_directional_v3"
    previous = PREVIOUS/previous_name
    if not (previous/"VERIFIED.json").is_file():
        raise RuntimeError("Completed independently verified cache required")
    if not smoke and not (ROOT/"lane_boundaries_smoke_v4/VERIFIED.json").is_file():
        raise RuntimeError("Verified boundary smoke required")
    paths = [ROOT/n for n in ("lane_directional.py","idm_fit.py","lane_fit.py","dispatch_boundaries.py","support_v4.json")]
    paths += [ROOT/"src/wms/physics"/n for n in ("lc2013.py","lc2013_geometry.py","config.py")]
    manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    command = ["nice","-n","19",sys.executable,"lane_directional.py","--root",
        "/home/ubuntu/wmdata20261031/views","--idm-pooled",str(OLD/"pooled_full_v5/idm_pooled.json"),
        "--out",str(output),"--workers","1" if smoke else "2","--support-json",str(ROOT/"support_v4.json"),
        "--reuse",str(previous)]
    if smoke:
        command += ["--worlds","969400"]
    environment = dict(os.environ,CUDA_VISIBLE_DEVICES="",OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",PYTHONPATH=str(ROOT/"src"))
    with receipt_path.open("x") as stream:
        json.dump({"state":"claimed","source_sha256":manifest,"argv":command},stream)
    with (ROOT/(name+".log")).open("xb") as log:
        process = subprocess.Popen(command,cwd=ROOT,env=environment,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True)
    data = {"state":"dispatched","pid":process.pid,
        "proc_start_ticks":(Path("/proc")/str(process.pid)/"stat").read_text().split()[21],
        "timestamp_unix":time.time(),"argv":command,"source_sha256":manifest,
        "load":os.getloadavg(),"gpu":False,"automatic_retry":False}
    receipt_path.write_text(json.dumps(data,indent=2)+"\n")
    print(json.dumps(data))


if __name__ == "__main__":
    main()
