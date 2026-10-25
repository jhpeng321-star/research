"""Detached, resumable, 12-worker CPU measurement and registered decision driver."""
from __future__ import annotations

import concurrent.futures as F
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path.home() / "nativelane"
REPO = ROOT / "repo"
PY = sys.executable
WM = Path.home() / "mappodiag/wm_13e_b2.pt"
WM_SHA = "952ec059de5a9c0285d2327bcea57c836fd5d736c9975592ffbb932a506e84c8"
DEADLINE = datetime.fromisoformat("2026-10-26T09:00:00+08:00").timestamp()
ENV = {**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}:{Path.home()}/mappodiag/site",
       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
       "CUDA_VISIBLE_DEVICES": "", "WMS_SKIP_PIVOT_DIAG": "1", "WMS_UNIT_SEED": "1",
       "C4_ALLOWED_ROOTS": ":".join(str(Path.home() / x) for x in
           ("nativelane", "mappodiag", "coopmeas", "c4", "actqual", "c1refresh"))}


def write(path, doc):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.replace(path)


def run(name, argv, on=False, timeout=18000, measurement=True):
    marker = ROOT / "done" / (name + ".json")
    if marker.exists():
        return json.loads(marker.read_text())["rc"]
    if (measurement and time.time() >= DEADLINE) or (ROOT / "STOP").exists():
        return 124
    with (ROOT / "logs" / (name + ".log")).open("ab") as log:
        proc = subprocess.Popen(list(map(str, argv)), cwd=REPO,
                                env={**ENV, "NATIVE_LANE": str(int(on))},
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        stat = Path(f"/proc/{proc.pid}/stat").read_text().rsplit(")", 1)[1].split()
        record = {"pid": proc.pid, "start_ticks": stat[19], "argv": argv,
                  "started": datetime.now().astimezone().isoformat()}
        write(ROOT / "logs" / (name + ".pid.json"), record)
        rc = proc.wait(timeout=timeout)
    print(json.dumps({"task": name, "rc": rc, "t": time.time()}), flush=True)
    if rc == 0:
        write(marker, {**record, "rc": rc, "finished": time.time()})
    return rc


def checks():
    tests = ["test_native_lane.py", "test_loop14c.py", "test_coopw.py", "test_sumocredit.py",
             "test_sumocredit_kcheck.py", "test_gate_panel.py", "test_loop14.py", "test_main14.py",
             "test_policy14_integration.py", "test_policy14_critic.py", "test_policy14_lanes.py",
             "test_policy14e.py", "test_policy14_r13e.py", "test_c4_nativecav.py"]
    assert run("unit_tests", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                              *["tests/unit/" + t for t in tests]]) == 0
    for suffix, flags in (("legacy", []), ("loop14c", ["0,wm_law"])):
        for name, repo in (("base", ROOT / "repo_off"), ("changed", REPO)):
            assert run(f"bitid_{suffix}_{name}", [PY, REPO / "tools/loop14_checks/loop14c_bitid.py",
                        repo, ROOT / f"bitid_{suffix}_{name}.pt", *flags]) == 0
    import torch
    bitid = {}
    for suffix in ("legacy", "loop14c"):
        a = torch.load(ROOT / f"bitid_{suffix}_base.pt", weights_only=True)
        b = torch.load(ROOT / f"bitid_{suffix}_changed.pt", weights_only=True)
        assert a.keys() == b.keys()
        for key in a:
            assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape, key
            assert torch.equal(a[key].reshape(-1).view(torch.uint8), b[key].reshape(-1).view(torch.uint8)), key
        bitid[suffix] = {"tensors": len(a), "bit_identical": True}
    write(ROOT / "CHECKS.json", {"passed": True, "bitid": bitid, "wm_sha256": WM_SHA})


def mappo():
    common = ["--repo", REPO, "--wm", WM, "--wm-config", "configs/c4_wm_v10round13e_stage_b.toml",
              "--mappo", Path.home() / "coopmeas/mappo_v5_final.pt", "--samples", "4", "--workers", "6"]
    # Read the completed predecessor's unit names, rather than select starts on outcomes.
    prior = Path.home() / "mappodiag"
    for cohort, horizon, source, pad in (("coop", 60, "src_bsrc", 64), ("c", 300, "src_cbase", 300)):
        units = []
        for p in sorted((prior / f"wm_{cohort}").glob("w*_t*.json")):
            d = json.loads(p.read_text())
            assert "error" not in d
            seed = int(d["world"][1:7])
            assert not 964000 <= seed <= 964199
            units.append(f"{d['world']}@{int(d['t0'])}")
        assert len(units) == (33 if horizon == 60 else 12)
        for on in (False, True):
            tag = "on" if on else "off"
            argv = [PY, "tools/nativelane/launch.py", "mappo", prior / "wm_mappo.py",
                    "--source", prior / source, "--units", ",".join(units), "--horizon", str(horizon),
                    "--pad-max", str(pad), "--out", ROOT / f"{tag}/wm_{cohort}", *common]
            assert run(f"mappo_{cohort}_{tag}", argv, on=on) == 0
    for tag in ("off", "on"):
        argv = [PY, Path.home() / "mappodiag/analyze_b.py", "--wm-coop", ROOT / tag / "wm_coop",
                "--wm-c", ROOT / tag / "wm_c", "--coopmeas-jobs", Path.home() / "coopmeas/run/jobs",
                "--src", prior / "src_bsrc", "--src-c", prior / "src_cbase",
                "--c-results", prior / "C_RESULTS.json", "--c-branches", prior / "out",
                "--thresholds", Path.home() / "coopmeas/thresholds.json", "--out", ROOT / f"{tag}/B_RESULTS.json"]
        assert run(f"analyze_{tag}", argv) == 0


def qual(workers=6):
    tasks = []
    for k in range(6):
        for suite in ("pulse", "coop"):
            for job in sorted((Path.home() / f"c4/runs/{suite}/jobs").glob("s*_t300")):
                if suite == "coop" and not (job / "traj__A__noop.npz").exists():
                    continue
                assert not 964000 <= int(job.name[1:7]) <= 964199
                for on in (False, True):
                    tag = "on" if on else "off"
                    tool = "qual_wm_pulse.py" if suite == "pulse" else "wm_coop_k.py"
                    argv = [PY, "tools/nativelane/launch.py", "qual", Path.home() / "actqual/tools" / tool,
                            "--out", ROOT / tag / ("pulse" if suite == "pulse" else "c9"),
                            "--source", Path.home() / f"c4/runs/{suite}/src/all", "--jobs", job,
                            "--repo", REPO, "--wm", WM, "--policy", Path.home() / "c4/coopcl/repo/_ckpt/v8_u600.pt",
                            "--threads", "1", "--future-seed", str(k)]
                    if suite == "pulse":
                        argv += ["--horizon", "30", "--tag", f"k{k}"]
                    else:
                        argv += ["--variant", "as_is", "--tests", "A", "--horizon", "120",
                                 "--suffix", "" if k == 0 else f"_k{k}"]
                    tasks.append((f"qual_{suite}_{tag}_k{k}_{job.name}", argv, on))
    with F.ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda x: run(*x), tasks))
    write(ROOT / "QUAL_TASKS.json", {"expected": len(tasks), "complete": all(rc == 0 for rc in results),
                                     "failed": [tasks[i][0] for i, rc in enumerate(results) if rc]})


def main():
    assert ROOT.is_dir() and REPO.is_dir()
    os.nice(19)
    for name in ("logs", "done", "off", "on"):
        (ROOT / name).mkdir(exist_ok=True)
    assert hashlib.sha256(WM.read_bytes()).hexdigest() == WM_SHA
    write(ROOT / "DRIVER.json", {"pid": os.getpid(), "sid": os.getsid(0), "started": time.time(),
                                "workers_mappo": 6, "workers_qual": 6, "deadline": DEADLINE})
    checks()
    with F.ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(mappo), pool.submit(qual)]
        failures = []
        for f in futures:
            try:
                f.result()
            except Exception as exc:
                failures.append(repr(exc))
                print(repr(exc), flush=True)
    write(ROOT / "MEASUREMENTS.json", {"finished": time.time(), "errors": failures})
    rc = run("decision", [PY, "tools/nativelane/analyze.py"], measurement=False)
    if rc:
        write(ROOT / "DECISION.json", {"adopt": False, "reason": "analysis incomplete/failed", "rc": rc})


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write(ROOT / "DRIVER_FAILED.json", {"error": repr(exc), "t": time.time()})
        raise
