"""tools/compare_runs.py — priority-ordered run comparison (user steer 2026-08-05).

Ranks eval dirs by the user's priority: TTC uncompensated > fairness > delay.
  1. ttc-regressed seed count (policy ttc > noop ttc) — lower wins
  2. domination count (slack rule, tools/domination_scoreboard.py)
  3. win-rule count (tools/win_scorecard.py)

Usage: python tools/compare_runs.py runs/eval/<dir1> [dir2 ...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SEED_RE = re.compile(r"seed(\d+)")


def _seed_of(path: Path) -> int:
    return int(_SEED_RE.search(path.name).group(1))


def stats(eval_dir: Path) -> dict:
    policy, noop = {}, {}
    for p in sorted(eval_dir.glob("*.seed*.json")):
        if p.name in ("summary.json", "scoreboard.json", "wincard.json"):
            continue
        row = json.loads(p.read_text(encoding="utf-8"))
        seed = _seed_of(p)
        (noop if p.name.startswith("noop.") else policy)[seed] = row

    n_dom = n_win = n_ttc_worse = 0
    for s, p in policy.items():
        n = noop.get(s)
        if not n:
            continue
        d_ttc = p.get("ttc_events", 0) - n.get("ttc_events", 0)
        d_g = p["fair_gap"] - n["fair_gap"]
        d_loss = p["loss"] - n["loss"]
        d_coll = p.get("collisions", 0) - n.get("collisions", 0)
        if d_ttc > 0:
            n_ttc_worse += 1
        if d_g < -0.1 and d_loss <= 0.5 and d_coll <= 0 and d_ttc <= 0:
            n_dom += 1
        if d_ttc < 0 and d_g <= 0.1 and d_loss <= 0.5 and d_coll <= 0:
            n_win += 1
    return {
        "dir": eval_dir.name,
        "n": len(policy),
        "ttc_worse": n_ttc_worse,
        "domination": n_dom,
        "win": n_win,
    }


def main() -> None:
    rows = [stats(Path(a)) for a in sys.argv[1:]]
    rows.sort(key=lambda r: (r["ttc_worse"], -r["domination"], -r["win"]))
    print(f"{'rank':>4} {'ttc_worse':>9} {'domination':>10} {'win':>5}  dir")
    for i, r in enumerate(rows, 1):
        print(f"{i:>4} {r['ttc_worse']:>9} {r['domination']:>10} {r['win']:>5}  {r['dir']}")


if __name__ == "__main__":
    main()
