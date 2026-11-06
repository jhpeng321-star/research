"""Stop a running MAPPO production cleanly AT an iteration boundary (box-side, own run only).

    python tools/nm_mappo/stop_at_boundary.py RUN_DIR [--poll-s 0.05] [--timeout-s 3600]

The trainer appends the iteration's ``episode`` rows to ``log.jsonl`` right AFTER the last PPO
minibatch and BEFORE writing the checkpoint; it checks ``STOP_REQUESTED`` at the top of its loop
before collecting the next iteration.  Writing the stop file as soon as new episode rows appear
therefore stops the run with the just-finished iteration checkpointed and (if the write lands
before the loop-top check) no simulation of the next iteration at all.  If the write lands later,
the in-flight collection is abandoned within ~1 s; ``STOP_AT_BOUNDARY.json`` records the exact
timestamps so the (small) abandoned simulation can be charged conservatively.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--poll-s", type=float, default=0.05)
    ap.add_argument("--timeout-s", type=float, default=3600.0)
    a = ap.parse_args(argv)
    prod = Path(a.run_dir) / "production"
    log = prod / "log.jsonl"
    start = log.stat().st_size
    t0 = time.time()
    while time.time() - t0 < a.timeout_s:
        size = log.stat().st_size
        if size > start:
            with log.open("rb") as fh:
                fh.seek(start)
                new = fh.read().decode("utf-8", "replace")
            if '"type": "episode"' in new:
                t_seen = time.time()
                (prod / "STOP_REQUESTED").open("a").write("stop_at_boundary: compute amendment\n")
                t_stop = time.time()
                status = json.loads((prod / "status.json").read_text()) if (prod / "status.json").exists() else {}
                rec = {"t_episode_rows_seen": t_seen, "t_stop_written": t_stop,
                       "status_iteration_at_stop": status.get("iteration"),
                       "status_mtime": (prod / "status.json").stat().st_mtime if (prod / "status.json").exists() else None}
                (Path(a.run_dir) / f"STOP_AT_BOUNDARY_{int(t_stop)}.json").write_text(json.dumps(rec, indent=1))
                print(json.dumps(rec))
                return 0
            start = size if '"type": "failure"' in new else start
        time.sleep(a.poll_s)
    print("timeout: no iteration boundary seen")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
