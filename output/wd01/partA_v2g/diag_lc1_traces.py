"""WD-01-PA2G-LC1 per-event MOBIL traces for the worst windows.

For each of the top disc-surplus windows (results/lc1/worst_windows.json,
top 12): accepted predicted DISCRETIONARY decisions whose vehicle shows
NO truth lane change within +-5 s (predicted-but-not-true), with the full
re-derived MOBIL term vector and the vehicle's truth trajectory beside
it (retro-fixed model rows, +-8 s at 0.5 s).

TRAIN only; zero SUMO.  Output: results/lc1/traces.json + traces.md
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
for _p in (str(WD01), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import wdpaths  # noqa: E402
DATA_ROOT = Path(wdpaths.RESEARCH_DENSE_ROOT)

from partA_v2g import eval_a as EA  # noqa: E402
from partA_v2g import eval_a2 as EA2  # noqa: E402

LC = HERE / "results" / "lc1"
N_WINDOWS = 12
MAX_EVENTS_PER_WINDOW = 5


def main() -> None:
    EA.MAPPER = EA.ModelMapper()
    EA.OPENED_RUN_FILES.clear()
    EA.TRAIN_WORLD_NAMES.clear()
    plan = json.loads(EA2.PLAN.read_text(encoding="utf-8"))
    meta = {w["world"]: w for w in plan["worlds"]}
    worst = json.loads((LC / "worst_windows.json").read_text(
        encoding="utf-8"))[:N_WINDOWS]
    # cohort lookup
    cohorts = {}
    for line in open(LC / "pred_states.jsonl", encoding="utf-8"):
        r = json.loads(line)
        cohorts[(r["world"], r["w"])] = set(r["vids"])
    # accepted decisions per (world, w)
    dec = defaultdict(list)
    for line in open(LC / "pred_decisions_accepted.jsonl",
                     encoding="utf-8"):
        d = json.loads(line)
        dec[(d["world"], d["w"])].append(d)
    # truth events per world
    te = defaultdict(list)
    for line in open(LC / "truth_events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        te[e["world"]].append(e)

    opened = set()
    EA.TRAIN_WORLD_NAMES.update(meta)
    out = []
    logs = {}
    for ww in worst:
        world, w = ww["world"], ww["w"]
        entry = dict(meta[world])
        assert entry.get("split") == "TRAIN"
        entry["chunk"] = 4
        entry["native_run"] = DATA_ROOT / "runs" / \
            "campaign_stage1_chunk4" / f"{world}__native_base" / "run" / \
            "physics_steps.jsonl"
        if world not in logs:
            logs[world] = EA.NativeLog(entry)
            opened.update(EA.OPENED_RUN_FILES)
            EA.OPENED_RUN_FILES.clear()
        log = logs[world]
        coh = cohorts[(world, w)]
        cand = [d for d in dec.get((world, w), [])
                if d["kind"] == "disc" and d["vid"] in coh and
                w < d["t"] <= w + 120 + 1e-9]
        # predicted-but-not-true: no truth event of this vid within +-5 s
        t_ev = te.get(world, [])
        sel = []
        for d in cand:
            near = [e for e in t_ev if e["vid"] == d["vid"] and
                    abs(e["t"] - d["t"]) <= 5.0 + 1e-9]
            if not near:
                sel.append(d)
        # prioritise repeat offenders
        cnt = defaultdict(int)
        for d in sel:
            cnt[d["vid"]] += 1
        sel.sort(key=lambda d: (-cnt[d["vid"]], d["t"]))
        events = []
        for d in sel[:MAX_EVENTS_PER_WINDOW]:
            j = int(round((d["t"] - log.t0) / 0.5))
            traj = []
            for jj in range(j - 16, j + 17):
                row = log.rows.get(jj)
                st = row.get(d["vid"]) if row else None
                if st is None:
                    continue
                traj.append(dict(t=round(log.t0 + jj * 0.5, 1),
                                 lane=int(st[0]), s=round(st[1], 1),
                                 v=round(st[2], 1)))
            # truth local context at the decision tick
            trow = log.rows.get(j, {})
            same = [(st[1], st[2]) for v2, st in trow.items()
                    if v2 != d["vid"] and st[0] == d["src"]]
            ahead = sorted(ds for ds, _v in same if ds > d["s"])
            events.append(dict(
                decision=d,
                n_pred_only_this_vid=cnt[d["vid"]],
                truth_gap_ahead=round(ahead[0] - d["s"], 1)
                if ahead else None,
                truth_trajectory=traj))
        out.append(dict(world=world, w=w, disc_pred=ww["disc_pred"],
                        disc_true=ww["disc_true"],
                        n_pred_only=len(sel), events=events))
        print(f"{world} w={w}: pred_only disc decisions {len(sel)} "
              f"(dumped {len(events)})", flush=True)
    (LC / "traces.json").write_text(json.dumps(out, indent=1),
                                    encoding="utf-8")
    lines = ["# WD-01-PA2G-LC1 MOBIL traces (worst windows, "
             "predicted-but-not-true)", ""]
    for blk in out:
        lines.append(f"## {blk['world']} w={blk['w']} "
                     f"(disc pred {blk['disc_pred']} vs true "
                     f"{blk['disc_true']}; pred-only decisions "
                     f"{blk['n_pred_only']})")
        for ev in blk["events"]:
            d = ev["decision"]
            lines.append("")
            lines.append(f"### {d['vid']} t={d['t']} "
                         f"{d['src']}->{d['tgt']} s={d['s']} "
                         f"(pred-only x{ev['n_pred_only_this_vid']}, "
                         f"truth gap ahead "
                         f"{ev['truth_gap_ahead']} m)")
            lines.append(f"- gain {d.get('gain', float('nan')):.3f} vs "
                         f"eff {d.get('eff', float('nan')):.3f} "
                         f"(thr {d.get('threshold', float('nan')):.2f}, "
                         f"bias {d.get('bias', 0):.2f}, p "
                         f"{d.get('politeness', float('nan')):.2f})")
            lines.append(f"- da_ego {d.get('a_ego_new', float('nan')):.3f}"
                         f" - {d.get('a_ego_old', float('nan')):.3f}; "
                         f"d_new_foll {d.get('a_new_after', 0):.3f}"
                         f" - {d.get('a_new_before', 0):.3f}; "
                         f"d_old_foll {d.get('a_old_after', 0):.3f}"
                         f" - {d.get('a_old_before', 0):.3f}")
            lines.append(f"- gap_lead {d['gap_lead']:.1f} m, "
                         f"gap_foll {d['gap_foll']:.1f} m, "
                         f"v_ego {d['v_ego']:.1f}, v_lead "
                         f"{d['v_lead']:.1f}, v_foll {d['v_foll']:.1f}")
            lines.append("- truth trajectory (retro-fixed model rows):")
            lines.append("")
            lines.append("  | t | lane | s | v |")
            lines.append("  |---|---|---|---|")
            for tr in ev["truth_trajectory"]:
                mark = " <-decision" if abs(tr["t"] - d["t"]) < 1e-6 else ""
                lines.append(f"  | {tr['t']} | {tr['lane']} | "
                             f"{tr['s']} | {tr['v']} |{mark}")
        lines.append("")
    (LC / "traces.md").write_text("\n".join(lines), encoding="utf-8")
    EA.OPENED_RUN_FILES.update(opened)
    EA.assert_train_only()
    (LC / "traces_opened_run_files.json").write_text(
        json.dumps(sorted(opened), indent=1), encoding="utf-8")
    print("DONE traces ->", LC / "traces.md")


if __name__ == "__main__":
    main()
