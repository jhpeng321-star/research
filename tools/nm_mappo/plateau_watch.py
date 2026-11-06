"""S-MAPPOWZL plateau stop rule (MAIN, REGISTRATION 2026-11-07 04:2x, registered before it3) - box watcher.

Rule: train at least 48 iterations.  From iteration 48, every 6 iterations (48, 54, 60, ...), take
the OLS slope of ``mean_episode_return`` over the last 24 iterations with a 95% t-interval
(df = 22); STOP at the first check where the CI includes 0 or is negative (lower bound <= 0).
On STOP: bind the latest snapshot (that iteration's export) as the MAPPO_LANE final.

Read-only on ``log.jsonl``; the only write into the run is ``production/STOP_REQUESTED`` (the
trainer's clean-stop file), dropped the moment the checked iteration's ``iter`` row appears, i.e.
after its last PPO minibatch and before the trainer's loop-top stop check, so the run ends with
that iteration checkpointed/exported and nothing of the next iteration simulated.  Every check is
recorded (``PLATEAU_CHECK_<it>.json``); a stop also writes ``PLATEAU_STOP.json``.  After the stop
(``EXIT_*`` of the production supervisor) it runs the registered post steps itself: ledger
reconcile, sanity_post (10,800 sim-s precharged) on that export, and the bind receipt.  If the run
completes without a stop, it does nothing more (the launch chain handles it168).

    python plateau_watch.py RUN_ROOT [--poll-s 0.05] [--no-post]
    python plateau_watch.py --replay LOG.jsonl          # rule on an existing log (no writes)
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

MIN_IT, EVERY, WINDOW = 48, 6, 24
T975 = {22: 2.073873067904015}          # Student t 0.975 quantile, df = WINDOW - 2
RULE_ID = "mappowzl-plateau-v1 (min 48, every 6, OLS slope of mean_episode_return over last 24, 95% t CI lo <= 0 -> stop)"


def ols_ci(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    sse = sum((y - a - b * x) ** 2 for x, y in zip(xs, ys))
    se = math.sqrt(sse / (n - 2) / sxx)
    t = T975[n - 2]
    return b, b - t * se, b + t * se


def check(iters: dict, k: int) -> dict | None:
    """Rule at iteration ``k`` given {iteration: mean_episode_return}; None if not a check point."""
    if k < MIN_IT or k % EVERY:
        return None
    xs = list(range(k - WINDOW + 1, k + 1))
    if any(x not in iters for x in xs):
        raise ValueError(f"missing iterations in the window ending {k}")
    slope, lo, hi = ols_ci(xs, [iters[x] for x in xs])
    return {"iteration": k, "window": [xs[0], xs[-1]], "slope": slope, "ci95": [lo, hi], "stop": lo <= 0.0,
            "rule": RULE_ID}


def iter_rows(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("type") == "iter":
                out[int(r["iteration"])] = float(r["mean_episode_return"])
    return out


def replay(path: Path) -> dict | None:
    iters = iter_rows(path)
    for k in sorted(iters):
        res = check(iters, k)
        if res and res["stop"]:
            return res
    return None


def post(root: Path, k: int) -> None:
    py = "/home/ubuntu/miniconda3/bin/python"
    code, run = root / "code", root / "run"
    sh = "/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo"
    env = {"PYTHONPATH": f"{code / 'src'}:{code}", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "SUMO_HOME": sh, "PATH": f"{sh}/bin:/home/ubuntu/miniconda3/bin:/usr/bin:/bin", "HOME": "/home/ubuntu"}
    status = json.loads((run / "production/status.json").read_text())
    actual = float(status["sumo_seconds_simulated"]) + 900.0 * float(status.get("failed_episodes", 0))
    with (root / "ledger/CHARGES.jsonl").open("a") as fh:
        fh.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "id": "S-MAPPOWZL",
                             "phase": "mappo_wz_lane", "reconcile": "production",
                             "status": f"plateau stop at it{k} ({RULE_ID})", "iterations": k,
                             "actual_sim_s": actual, "precharged_sim_s": 6955200.0,
                             "release_sim_s": 6955200.0 - actual}) + "\n")
    export = run / "production" / f"policy_{k:05d}.pt"
    panel = json.loads((run / "SANITY_PANEL.json").read_text())
    rec = run / "SANITY_POST_REGISTERED.json"
    rec.open("x").write(json.dumps({"panel_sha256": panel["panel_sha256"], "budget_sim_s": 10800.0,
                                    "stage": "sanity_post", "policy": str(export)}))
    if subprocess.call([py, str(root / "ledger/charge.py"), "sanity_post", "10800",
                        f"12 worlds x 1 arm (lane MAPPO it{k}) x (300+600) s"], env=env) != 0:
        return
    with open(root / "sanity_post.log", "x") as fh:
        subprocess.call([py, "-u", "tools/nm_mappo/sanity_eval.py", "--config", "configs/c4_mappo_lane_wz3to2.toml",
                         "--policy", f"wzl={export}", "--skip-native", "--prior-rows", str(root / "sanity_pre/rows.jsonl"),
                         "--out", str(root / "sanity_post"), "--registered", str(rec), "--workers", "12"],
                        cwd=code, env=env, stdout=fh, stderr=subprocess.STDOUT)
    subprocess.call([py, "tools/nm_mappo/launch.py", "bind", "--run-dir", str(run), "--iteration", str(k)],
                    cwd=code, env=env)


def watch(root: Path, poll_s: float, do_post: bool) -> int:
    prod = root / "run" / "production"
    log = prod / "log.jsonl"
    iters = iter_rows(log) if log.exists() else {}
    seen, offset, buf, last_exit_check = set(iters), (log.stat().st_size if log.exists() else 0), "", 0.0
    while True:
        if time.time() - last_exit_check > 10:
            last_exit_check = time.time()
            if any(prod.glob("EXIT_production_*.json")) and not (root / "run/PLATEAU_STOP.json").exists():
                return 0                                 # completed / stopped otherwise: nothing to do
        size = log.stat().st_size if log.exists() else 0
        if size > offset:                                # parse only the appended bytes
            with log.open("rb") as fh:
                fh.seek(offset)
                buf += fh.read(size - offset).decode("utf-8", "replace")
            offset = size
            *lines, buf = buf.split(chr(10))
            for line in lines:
                if '"type": "iter"' in line:
                    r = json.loads(line)
                    iters[int(r["iteration"])] = float(r["mean_episode_return"])
        for k in sorted(set(iters) - seen):
            seen.add(k)
            res = check(iters, k)
            if res is None:
                continue
            if res["stop"]:
                (prod / "STOP_REQUESTED").open("a").write(f"plateau rule stop at it{k}\n")
                res["stop_written_unix"] = time.time()
                (root / "run/PLATEAU_STOP.json").write_text(json.dumps(res, indent=1))
                while not any(prod.glob("EXIT_production_*.json")):
                    time.sleep(10)
                if do_post:
                    post(root, k)
                return 0
            (root / "run" / f"PLATEAU_CHECK_{k:05d}.json").write_text(json.dumps(res, indent=1))
        time.sleep(poll_s)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("--replay", default=None)
    ap.add_argument("--poll-s", type=float, default=0.05)
    ap.add_argument("--no-post", action="store_true")
    a = ap.parse_args(argv)
    if a.replay:
        print(json.dumps(replay(Path(a.replay))))
        return 0
    return watch(Path(a.root), a.poll_s, not a.no_post)


if __name__ == "__main__":
    sys.exit(main())
