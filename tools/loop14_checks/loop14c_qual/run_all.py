"""EXPLORATORY LOOP14C qualification-style WM runs (scratch; registered runs untouched).  Own PIDs only."""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
REPO = "E:/agent-work/c4-loop14c"
WORKERS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
VARIANTS = sys.argv[2].split(",") if len(sys.argv) > 2 else ["off", "wmlaw", "wmlaw_head"]
KS = range(6)
REVERSE = len(sys.argv) > 3 and sys.argv[3] == "reverse"
env = {**os.environ, "C4_ALLOWED_ROOTS": str(HERE).replace("\\", "/"), "PYTHONIOENCODING": "utf-8",
       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
common = ["--wm", "ck/wm_s.pt", "--policy", "policy/v8_u600.pt", "--repo", REPO]
tasks = []
pjobs = sorted((HERE / "data/pulse/jobs").glob("s*_t300"))
cjobs = [j for j in sorted((HERE / "data/coop/jobs").glob("s*_t300")) if (j / "traj__A__noop.npz").exists()]
for k in KS:
    for v in VARIANTS:
        for j in pjobs:
            tasks.append((f"pulse_{v}_k{k}_{j.name}", ["launch.py", v, REPO, "tools/qual_wm_pulse.py",
                          "--out", f"out/{v}/pulse", "--source", "data/pulse/src/all", "--jobs", str(j),
                          *common, "--horizon", "30", "--future-seed", str(k), "--tag", f"k{k}"]))
        for j in cjobs:
            suf = "" if k == 0 else f"_k{k}"
            tasks.append((f"c9_{v}_k{k}_{j.name}", ["launch.py", v, REPO, "tools/wm_coop_k.py",
                          "--out", f"out/{v}/c9", "--source", "data/coop/src/all", "--jobs", str(j),
                          "--variant", "as_is", "--tests", "A", "--horizon", "120", "--threads", "1",
                          "--future-seed", str(k), "--suffix", suf, *common]))
# off / wmlaw for every sample first (the requested check), the extra R1 head variant last
tasks.sort(key=lambda t: t[0].split("_")[1] == "wmlaw" and t[0].startswith(("pulse_wmlaw_head", "c9_wmlaw_head")))
if REVERSE:
    tasks = tasks[::-1]
(HERE / "done").mkdir(exist_ok=True)
(HERE / "logs").mkdir(exist_ok=True)


def run(task):
    name, argv = task
    marker = HERE / "done" / name
    if marker.exists():
        return name, "cached"
    if REVERSE and (HERE / "logs" / f"{name}.log").exists():
        return name, "claimed"
    with open(HERE / "logs" / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call([PY, *argv], cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT,
                             creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0))
    if rc == 0:
        marker.touch()
    return name, rc


start = time.time()
print(f"{len(tasks)} tasks, {WORKERS} workers", flush=True)
with ThreadPoolExecutor(WORKERS) as pool:
    for i, (name, rc) in enumerate(pool.map(run, tasks), 1):
        print(f"{time.time() - start:7.0f}s {i}/{len(tasks)} {name} {rc}", flush=True)
