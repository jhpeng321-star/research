"""S-BACKLOG (cpuP/cpuQ): survey recorded pair branch tapes under
/home/ubuntu/c4/main14/restart20261026/work/*/  (read-only).

Prints, per cycle dir (c00/c01/c02): each branches/policy/<job> with its tape
count and the base episode's requests.rou.xml presence.  Read-only; never opens
sealed/ or restricted paths."""
import json
import sys
from pathlib import Path

WORK = Path("/home/ubuntu/c4/main14/restart20261026/work")
BAD = ("holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1", "sealed")

for wdir in sorted(WORK.iterdir()):
    if not wdir.is_dir() or any(b in str(wdir) for b in BAD):
        continue
    for cyc in sorted(p for p in wdir.rglob("c0[2]") if p.is_dir()):
        br = cyc / "collect" / "branches" / "policy"
        eps = cyc / "collect" / "episodes" / "policy"
        if not br.is_dir():
            continue
        jobs = sorted(p for p in br.iterdir() if p.is_dir())
        tapes = sum(1 for _ in br.glob("*/*"))
        req = sum(1 for _ in eps.glob("*/scenario/requests.rou.xml"))
        print(f"{wdir.name}/{cyc.name} jobs={len(jobs)} tapes={tapes} "
              f"base_req={req} summaries={(cyc / 'artifact' / 'summaries').is_dir()}")
