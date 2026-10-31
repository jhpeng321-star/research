"""Freeze explicit committed LANE source and prepare an own fresh cpuQ root."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
LANE = REPO.parent/"c4-rb-lane"
REMOTE_PYTHON = "/home/ubuntu/miniconda3/bin/python"
BOOTSTRAP = "/home/ubuntu/rbfit20261101_addendum/tools/rbfit/dispatch_lc_version.py"
PHYSICS = ("src/wms/physics/lc2013.py","src/wms/physics/lc2013_geometry.py")


def run(argv, **kwargs):
    return subprocess.check_output(argv,**kwargs)


def source_identity(commit=None):
    commit = commit or run(["git","-C",str(LANE),"log","-1","--format=%H","--",*PHYSICS]).decode().strip()
    files = {path:run(["git","-C",str(LANE),"show",f"{commit}:{path}"]) for path in PHYSICS}
    digest = hashlib.sha256(b"".join(files[path] for path in PHYSICS)).hexdigest()
    return commit,digest,files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-commit")
    parser.add_argument("--prepare",action="store_true")
    args = parser.parse_args()
    commit,digest,files = source_identity(args.lane_commit)
    # Fail explicitly if a new port adds dependencies outside our frozen overlay.
    allowed = {"config","mobil","neighbours","lc2013_geometry","batched"}
    for payload in files.values():
        for node in ast.walk(ast.parse(payload.decode())):
            if isinstance(node,ast.ImportFrom) and node.level:
                if node.level!=1 or node.module not in allowed:
                    raise RuntimeError(f"New LC dependency requires manual reconciliation: {node.module}")
    key = digest[:12]
    parent = REPO/"artifacts/rbfit/source_versions"
    if not parent.is_dir():
        raise ValueError("Existing local source-version parent required")
    bundle = parent/key
    bundle.mkdir(exist_ok=False)
    (bundle/"src/wms/physics").mkdir(parents=True)
    for path,payload in files.items():
        (bundle/path).write_bytes(payload)
    own_files = {name:REPO/"tools/rbfit"/name for name in (
        "lane_directional.py","lane_raw_fit.py","verify_directional.py","dispatch_lc_version.py")}
    own_files.update({"test_rbfit_directional.py":REPO/"tests/unit/test_rbfit_directional.py",
        "motion_fit.py":REPO/"src/wms/physics/motion_fit.py",
        "support.json":REPO/"artifacts/rbfit/receipts/support_v4.json"})
    for name,path in own_files.items():
        (bundle/name).write_bytes(path.read_bytes())
    payload_paths = [*files,*own_files]
    version = {"schema":"wm_phys.lc_source_version.v1","lane_commit":commit,"physical_source_digest":digest,
        "fit_parent_commit":run(["git","-C",str(REPO),"rev-parse","HEAD"]).decode().strip(),
        "fit_source_identity":"exact source_sha256 bytes; parent commit is not a clean-source assertion",
        "source_sha256":{path:hashlib.sha256((bundle/path).read_bytes()).hexdigest() for path in payload_paths},
        "snapshot_version":"v2","worlds":list(range(969400,969432)),"adopted":False}
    (bundle/"source_version.json").write_text(json.dumps(version,indent=2)+"\n")
    remote = f"/home/ubuntu/rbfit20261101_lc_source_{key}"
    if args.prepare:
        run(["scp",str(bundle/"dispatch_lc_version.py"),f"cpuQ:{BOOTSTRAP}"])
        print(run(["ssh","cpuQ",f"{REMOTE_PYTHON} {BOOTSTRAP} --root {remote} prepare"]).decode().strip())
        run(["scp",*[str(bundle/path) for path in files],f"cpuQ:{remote}/src/wms/physics/"])
        run(["scp",*[str(bundle/name) for name in (*own_files,"source_version.json")],f"cpuQ:{remote}/"])
    print(json.dumps({"key":key,"bundle":str(bundle),"remote_root":remote,"lane_commit":commit,"prepared":args.prepare}))


if __name__ == "__main__":
    main()
