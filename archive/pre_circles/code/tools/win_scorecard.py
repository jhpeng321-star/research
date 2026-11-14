"""tools/win_scorecard.py — user win-rule scorecard (2026-08-04).

Per-seed WIN iff, vs the paired no_op baseline at the SAME ruler:
    policy ttc_events <  noop ttc_events      (safety strictly better)
    policy fair_gap  <= noop fair_gap + 0.1   (fairness better or ~equal)
    policy loss      <= noop loss + 0.5       (delay not regressing a lot)
    policy collisions <= noop collisions

This is the user's research win definition; the canonical selection metric
remains tools/domination_scoreboard.py (strict-gap rule). Usage:

    python tools/win_scorecard.py runs/eval/<eval_dir> [more dirs...]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_SEED_RE = re.compile(r"seed(\d+)")
TTC_RULE = "strict"  # "strict": policy < noop; "slack": policy <= noop
G_SLACK = 0.1
LOSS_SLACK = 0.5


def _seed_of(path: Path) -> int:
    return int(_SEED_RE.search(path.name).group(1))


def score(eval_dir: Path) -> dict:
    policy, noop = {}, {}
    for p in sorted(eval_dir.glob("*.seed*.json")):
        if p.name in ("summary.json", "scoreboard.json", "wincard.json"):
            continue
        row = json.loads(p.read_text(encoding="utf-8"))
        seed = _seed_of(p)
        if p.name.startswith("noop."):
            noop[seed] = row
        else:
            policy[seed] = row

    per_seed = []
    for s in sorted(policy):
        n = noop.get(s)
        if not n:
            continue
        p = policy[s]
        d_ttc = p.get("ttc_events", 0) - n.get("ttc_events", 0)
        d_g = p["fair_gap"] - n["fair_gap"]
        d_loss = p["loss"] - n["loss"]
        d_coll = p.get("collisions", 0) - n.get("collisions", 0)
        ttc_ok = d_ttc < 0 if TTC_RULE == "strict" else d_ttc <= 0
        win = ttc_ok and d_g <= G_SLACK and d_loss <= LOSS_SLACK and d_coll <= 0
        per_seed.append(
            {
                "seed": s,
                "win": bool(win),
                "d_ttc": d_ttc,
                "d_g": round(d_g, 3),
                "d_loss": round(d_loss, 3),
                "d_collisions": d_coll,
                "ttc_ok": bool(ttc_ok),
            }
        )

    out = {
        "eval_dir": str(eval_dir),
        "rule": f"ttc {'< 0' if TTC_RULE == 'strict' else '<= 0'}; d_g <= +{G_SLACK}; d_loss <= +{LOSS_SLACK}; d_coll <= 0",
        "n": len(per_seed),
        "n_win": sum(1 for r in per_seed if r["win"]),
        "n_ttc_better": sum(1 for r in per_seed if r["d_ttc"] < 0),
        "n_ttc_worse": sum(1 for r in per_seed if r["d_ttc"] > 0),
        "per_seed": per_seed,
    }
    (eval_dir / "wincard.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    for arg in sys.argv[1:]:
        res = score(Path(arg))
        fails = [r for r in res["per_seed"] if not r["win"]]
        print(f"{res['eval_dir']}")
        print(
            f"  WIN {res['n_win']}/{res['n']}  (rule: {res['rule']})  "
            f"ttc better {res['n_ttc_better']}, worse {res['n_ttc_worse']}"
        )
        for r in fails:
            why = []
            if not r["ttc_ok"]:
                why.append(f"ttc+{r['d_ttc']}")
            if r["d_g"] > G_SLACK:
                why.append(f"G+{r['d_g']:.1f}")
            if r["d_loss"] > LOSS_SLACK:
                why.append(f"loss+{r['d_loss']:.1f}")
            if r["d_collisions"] > 0:
                why.append(f"coll+{r['d_collisions']}")
            print(f"    miss {r['seed']}: {', '.join(why)}")


if __name__ == "__main__":
    main()
