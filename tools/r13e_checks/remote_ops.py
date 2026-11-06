"""Narrow, auditable gpuR transport for the authorized CPU-only R13E study."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "/root/autodl-tmp/c4/v5"
REPO = REMOTE + "/repo_r13e_8c7ff0a1"
WORK = REMOTE + "/diag_emerg"


def remote(source):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "gpuR",
         "env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
         "nice -n 19 /root/miniconda3/bin/python -"], input=source, text=True, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inspect", "deploy", "script"))
    parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    if args.action == "script":
        path = (ROOT / args.path).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("remote script must be in this worktree")
        return remote(path.read_text(encoding="utf-8"))
    if args.action == "inspect":
        return remote(f'''
import json,os
from pathlib import Path
root=Path({REMOTE!r})
paths=[root,root/'runs/c4_wm_v10_round12c/wm_s.pt',root/'diag_emerg',root/'repo_r13e_8c7ff0a1',root/'chk_r10b/states192.json.gz',root/'data/v5_native/C4_EPISODES_MANIFEST.json']
print(json.dumps({{str(p):p.exists() for p in paths}},indent=2))
print('cpu_count',os.cpu_count(),'nice',os.getpriority(os.PRIO_PROCESS,0))
''')
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "c4/v10r13e":
        raise ValueError("wrong branch")
    archive = ROOT / "r13e_8c7ff0a1_runtime.tar"
    if archive.exists():
        raise ValueError("local archive already exists")
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", "8c7ff0a1", "src", "configs", "tools", "pyproject.toml"], cwd=ROOT, text=True).splitlines()
    forbidden = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "hetdata_truth", "vtype")
    names = [p for p in names if not any(t in p.lower() for t in forbidden)
             and (p.startswith(("src/", "configs/", "tools/r10b_checks/", "tools/r12_checks/", "tools/r13e_checks/"))
                  or (p.startswith("tools/") and p.count("/") == 1 and p.endswith(".py"))
                  or p == "pyproject.toml")]
    subprocess.run(["git", "archive", "--format=tar", f"--output={archive}", "8c7ff0a1", *names], cwd=ROOT, check=True)
    remote(f'''
from pathlib import Path
root=Path({REMOTE!r}); assert root.is_dir()
work=Path({WORK!r}); work.mkdir(exist_ok=True)
assert not Path({REPO!r}).exists(), 'deployment already exists; inspect before reuse'
assert not (work/'r13e_8c7ff0a1_runtime.tar').exists()
assert __import__('shutil').disk_usage(root).free > {archive.stat().st_size * 3 + 100_000_000}, 'insufficient disk headroom'
''')
    subprocess.run(["scp", str(archive), f"gpuR:{WORK}/r13e_8c7ff0a1_runtime.tar"], check=True)
    remote(f'''
import hashlib,json,subprocess
from pathlib import Path
archive=Path({WORK!r})/'r13e_8c7ff0a1_runtime.tar'
repo=Path({REPO!r}); repo.mkdir()
subprocess.run(['tar','-xf',str(archive),'-C',str(repo)],check=True)
print(json.dumps({{'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'repo':str(repo)}}))
''')


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        sys.exit(exc.returncode)
