"""Amendment 77 (S-MAPPO20C) training phase ledger = the A65 ledger (mappo_p60c/ledger/charge.py) with the A77 names.

    python charge.py <stage> <sim_s> "<basis>"      precharge a stage BEFORE it starts (refuses a repeat / the cap)
    python charge.py --reconcile <stage> <actual_sim_s> "<status>"
"""
import json
import sys
import time
from pathlib import Path

R = Path("/home/ubuntu/a77")
CAP = 3_560_000
LEDGER = R / "ledger/CHARGES.jsonl"


def rows():
    return [json.loads(x) for x in LEDGER.read_text().splitlines() if x.strip()] if LEDGER.exists() else []


def write(doc):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as f:
        f.write(json.dumps(doc) + "\n")
        f.flush()


def charge(stage, sim, basis):
    rs = rows()
    if any(r.get("stage") == stage for r in rs):
        raise SystemExit(f"stage {stage} already charged")
    total = sum(r.get("precharge_sim_s", 0.0) for r in rs) + sim
    if total > CAP:
        raise SystemExit(f"cap exceeded: {total}")
    write({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "id": "S-MAPPO20C", "phase": "mappo_p20c_a77",
           "stage": stage, "precharge_sim_s": sim, "cumulative_precharged_sim_s": total, "cap_sim_s": CAP,
           "basis": basis})
    print(json.dumps({"charged": stage, "sim_s": sim, "cumulative": total}))


def reconcile(stage, actual, status):
    write({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "id": "S-MAPPO20C", "phase": "mappo_p20c_a77",
           "reconcile": stage, "status": status, "actual_sim_s": actual})


if __name__ == "__main__":
    if sys.argv[1] == "--reconcile":
        reconcile(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        charge(sys.argv[1], float(sys.argv[2]), sys.argv[3])
