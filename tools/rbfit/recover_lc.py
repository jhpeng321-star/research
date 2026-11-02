"""Publish one completed, previously receipted LC job; never dispatch work."""
import argparse
from datetime import datetime
import json

from watch_lc import STATE_PATH, acquire_singleton, inspect, publish, save_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", required=True)
    parser.add_argument("--name", choices=("lane_fit", "lane_raw"), default="lane_fit")
    args = parser.parse_args()
    release = acquire_singleton()
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        matches = [job for job in state["jobs"]
                   if job.get("key") == args.key and job["name"] == args.name]
        if len(matches) != 1:
            raise ValueError("Require exactly one existing owned job receipt")
        job = matches[0]
        status = inspect(job)
        if status["running"] or not status["claimed"] or not status["complete"]:
            raise RuntimeError(f"Only completed, exited owned jobs can be recovered: {status}")
        if not status["verified"]:
            status = inspect(job, "verify")
        recovery = {"timestamp": datetime.now().astimezone().isoformat(),
                    "status": status, "previous_blocker": job.get("blocked")}
        job.setdefault("recoveries", []).append(recovery)
        job["last_status"] = status
        job.pop("blocked", None)
        save_state(state)
        if not job.get("published_commit"):
            publish(job, state)
            save_state(state)
        print(json.dumps({"status": "PASS", "publication": job["publication"],
                          "commit": job["published_commit"], "new_dispatches": 0}))
    finally:
        release()


if __name__ == "__main__":
    main()
