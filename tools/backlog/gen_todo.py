"""S-BACKLOG: list the task ids still to run at a short horizon on this box (loop first, then E3).

Excludes tasks that already have a 300 s row in rows2 and tasks currently owned by the kept long
workers (given as shard indexes of the original 16-way split).  Prints one task id per line."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("C4_ALLOWED_ROOTS", str(Path.home() / "backlog"))
import cqr_rows as C  # noqa: E402

box, parity = sys.argv[1], int(sys.argv[2])
kept = {int(x) for x in sys.argv[3].split(",") if x}
loop = [t for n, t in enumerate(C.loop_tasks()) if n % 2 == parity]
tasks = loop + C.e3_tasks(box)
done = set()
for f in (Path.home() / "backlog/rows2").glob("rows_*.jsonl"):
    done |= {json.loads(l)["task"] for l in f.read_text().splitlines() if l.strip()}
owned = {C.task_id(t) for n, t in enumerate(tasks) if n % 16 in kept and n < len(loop)}
for t in tasks:
    tid = C.task_id(t)
    if tid not in done and tid not in owned:
        print(tid)
