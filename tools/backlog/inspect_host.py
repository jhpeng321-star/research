"""S-BACKLOG host-side helpers: inspect loop state / pair datasets (read-only)."""
import json
import sys
from pathlib import Path

LOOP = Path("/root/autodl-tmp/c4/restart20261026/arm/loop")


def state_pairs():
    s = json.loads((LOOP / "LOOP14_STATE.json").read_text())
    def walk(d, p=""):
        for k, v in (d.items() if isinstance(d, dict) else []):
            kk = f"{p}.{k}" if p else k
            if isinstance(v, dict):
                walk(v, kk)
            elif any(t in kk.lower() for t in ("pair", "data", "cycle", "host")):
                print(kk, "=", str(v)[:140])
    walk(s)


def dataset_info(name):
    root = LOOP / "data" / name
    m = json.loads((root / "C4_EPISODES_MANIFEST.json").read_text())
    worlds = sorted({e["world"] for e in m["episodes"]})
    have_q = sum(1 for e in m["episodes"] if "queue_arrays" in e)
    tapes = sorted({e["source"].split("/worlds/")[1].split("/")[0] for e in m["episodes"]
                    if "/worlds/" in e.get("source", "")})
    print(json.dumps({"name": name, "episodes": len(m["episodes"]), "worlds": len(worlds),
                      "world_ids": [w.split("m")[0] for w in worlds],
                      "with_queue_arrays": have_q, "tape_worlds": tapes}, indent=1))


if __name__ == "__main__":
    {"state": state_pairs, "ds": dataset_info}[sys.argv[1]](*sys.argv[2:])
