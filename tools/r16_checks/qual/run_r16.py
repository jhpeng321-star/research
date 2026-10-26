"""EXPLORATORY S-RAMPSPEED qualification-style WM runs on cpuX (scratch; registered runs untouched).
Own PIDs only, nice 19, <= 8 workers.  usage: python run_r16.py REPO [WORKERS] [VARIANTS]"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
REPO = sys.argv[1]
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
if not 1 <= WORKERS <= 8:
    raise SystemExit("workers 1..8")
VARIANTS = sys.argv[3].split(",") if len(sys.argv) > 3 else ["r14", "r16"]
KS = range(6)
env = {**os.environ, "C4_ALLOWED_ROOTS": str(HERE), "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
common = ["--wm", "ck/wm_s.pt", "--policy", "policy/v8_u600.pt", "--repo", REPO]
tasks = []
pjobs = sorted((HERE / "data/pulse/jobs").glob("s*_t300"))
cjobs = [j for j in sorted((HERE / "data/coop/jobs").glob("s*_t300")) if (j / "traj__A__noop.npz").exists()]
for k in KS:
    for v in VARIANTS:
        for j in pjobs:
            tasks.append((f"pulse_{v}_k{k}_{j.name}", ["launch_r16.py", v, REPO, "tools/qual_wm_pulse.py",
                          "--out", f"out/{v}/pulse", "--source", "data/pulse/src/all", "--jobs", str(j),
                          *common, "--horizon", "30", "--future-seed", str(k), "--tag", f"k{k}"]))
        for j in cjobs:
            suf = "" if k == 0 else f"_k{k}"
            tasks.append((f"c9_{v}_k{k}_{j.name}", ["launch_r16.py", v, REPO, "tools/wm_coop_k.py",
                          "--out", f"out/{v}/c9", "--source", "data/coop/src/all", "--jobs", str(j),
                          "--variant", "as_is", "--tests", "A", "--horizon", "120", "--threads", "1",
                          "--future-seed", str(k), "--suffix", suf, *common]))
(HERE / "done").mkdir(exist_ok=True)
(HERE / "logs").mkdir(exist_ok=True)
os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))


def run(task):
    name, argv = task
    if (HERE / "done" / name).exists():
        return name, "cached"
    with open(HERE / "logs" / f"{name}.log", "w", encoding="utf-8") as log:
        rc = subprocess.call([PY, *argv], cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
    if rc == 0:
        (HERE / "done" / name).touch()
    return name, rc


start = time.time()
print(f"{len(tasks)} tasks, {WORKERS} workers", flush=True)
with ThreadPoolExecutor(WORKERS) as pool:
    for i, (name, rc) in enumerate(pool.map(run, tasks), 1):
        print(f"{time.time() - start:8.0f}s {i}/{len(tasks)} {name} {rc}", flush=True)
(HERE / "ALL_DONE").write_text(time.strftime("%F %T"))
