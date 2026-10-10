"""tools/domination_scoreboard.py — per-seed domination scoreboard (slack rule).

A seed is DOMINATED iff, vs the paired no_op baseline:
    policy fair_gap <  noop fair_gap - 0.1s   (strict margin: d_g < -g_margin)
    policy loss     <= noop loss + 0.5s       (slack: d_loss <= loss_slack)
    policy collisions <= noop collisions
    policy ttc_events <= noop ttc_events

Single source of truth for checkpoint selection (replaces TB reward and the
<= gate, which coast passes trivially). Usage:

    python tools/domination_scoreboard.py runs/eval/fair_safe_v9_i40_dt05_holdout101_150
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# Easy no_op seeds (PLAN_V9_IMPROVE_PASS §1.2): no_op already near-fair,
# nothing to win. Reported separately; ties there are the accepted ceiling.
STRUCTURAL_SEEDS = [106, 117, 119, 120, 129, 136, 137, 143, 146, 147, 148]

_SEED_RE = re.compile(r"seed(\d+)")


def _seed_of(path: Path) -> int:
    return int(_SEED_RE.search(path.name).group(1))


def _load_rows(eval_dir: Path):
    policy, noop = {}, {}
    for p in sorted(eval_dir.glob("*.seed*.json")):
        if p.name == "summary.json" or p.name == "scoreboard.json":
            continue
        row = json.loads(p.read_text(encoding="utf-8"))
        seed = _seed_of(p)
        if p.name.startswith("noop."):
            noop[seed] = row
        else:
            if seed in policy:
                raise SystemExit(
                    f"multiple policy files for seed {seed} in {eval_dir}; "
                    "score one checkpoint tag per directory"
                )
            policy[seed] = row
    return policy, noop


def _fail_cause(d_g, d_loss, d_coll, d_ttc, g_margin, loss_slack):
    causes = []
    if d_g >= -g_margin:
        causes.append("G")
    if d_loss > loss_slack:
        causes.append("loss")
    if d_coll > 0 or d_ttc > 0:
        causes.append("safety")
    return "+".join(causes) if causes else "none"


def score_eval_dir(eval_dir, noop_dir=None, eps=1e-9, g_margin=0.1, loss_slack=0.5) -> dict:
    eval_dir = Path(eval_dir)
    noop_dir = Path(noop_dir) if noop_dir else eval_dir
    policy, noop = _load_rows(eval_dir)
    if noop_dir != eval_dir:
        _, noop_extra = _load_rows(noop_dir)
        noop = {**noop_extra, **noop}
    seeds = sorted(set(policy) & set(noop))
    if not seeds:
        raise SystemExit(f"no paired seeds found in {eval_dir}")

    per_seed = []
    for s in seeds:
        r, n = policy[s], noop[s]
        d_g = float(r["fair_gap"]) - float(n["fair_gap"])
        d_loss = float(r["loss"]) - float(n["loss"])
        d_coll = int(r["collisions"]) - int(n["collisions"])
        d_ttc = int(r["ttc_events"]) - int(n["ttc_events"])
        dominate = d_g < -g_margin and d_loss <= loss_slack and d_coll <= 0 and d_ttc <= 0
        per_seed.append(
            {
                "seed": s,
                "dominate": dominate,
                "d_g": d_g,
                "d_loss": d_loss,
                "d_collisions": d_coll,
                "d_ttc": d_ttc,
                "fail_cause": "none"
                if dominate
                else _fail_cause(d_g, d_loss, d_coll, d_ttc, g_margin, loss_slack),
            }
        )

    winnable = [r for r in per_seed if r["seed"] not in STRUCTURAL_SEEDS]
    structural = [r for r in per_seed if r["seed"] in STRUCTURAL_SEEDS]
    return {
        "eval_dir": str(eval_dir),
        "n": len(per_seed),
        "n_dominate": sum(r["dominate"] for r in per_seed),
        "n_winnable": len(winnable),
        "n_winnable_dominate": sum(r["dominate"] for r in winnable),
        "structural": {
            "n": len(structural),
            "n_dominate": sum(r["dominate"] for r in structural),
            "n_tie_g": sum(
                1
                for r in structural
                if r["d_g"] > -g_margin
                and r["d_loss"] <= loss_slack
                and r["d_collisions"] <= 0
                and r["d_ttc"] <= 0
            ),
            "n_harmed": sum(1 for r in structural if r["fail_cause"] != "none"),
        },
        "per_seed": per_seed,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("eval_dir", type=Path)
    ap.add_argument("--noop-dir", type=Path, default=None)
    ap.add_argument("--g-margin", type=float, default=0.1)
    ap.add_argument("--loss-slack", type=float, default=0.5)
    args = ap.parse_args()
    result = score_eval_dir(
        args.eval_dir, noop_dir=args.noop_dir, g_margin=args.g_margin, loss_slack=args.loss_slack
    )
    out = Path(args.eval_dir) / "scoreboard.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    s = result["structural"]
    print(
        f"dominate {result['n_dominate']}/{result['n']}  "
        f"(g_margin={args.g_margin}, loss_slack={args.loss_slack}; "
        f"winnable {result['n_winnable_dominate']}/{result['n_winnable']}; "
        f"structural: {s['n_dominate']} dominated, {s['n_tie_g']} ties, "
        f"{s['n_harmed']} harmed)"
    )
    fails = [r for r in result["per_seed"] if not r["dominate"]]
    for r in fails:
        print(f"  seed {r['seed']}: {r['fail_cause']} dG={r['d_g']:+.2f} dL={r['d_loss']:+.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
