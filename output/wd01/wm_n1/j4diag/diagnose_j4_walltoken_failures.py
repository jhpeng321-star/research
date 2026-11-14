"""Diagnose the 18 J4 wall runs that fail wall_tokens_v04 at a NON-terminal tick.

For every run: wall episodes from wall_events.jsonl (request / terminal ts,
members), one streaming pass over run/physics_steps.jsonl (mechanics rows
only: vehicles / cav_ids / departed_ids / lane_id), then for each member the
tick timeline vs the wall's active window [request, terminal]:
  absent_from_vehicles  - member gone from the physics row (left network)
  no_cav_report         - member present but its vid is not in that row's
                          cav_ids (junction ':' lane report gap)
  non_cav_member        - member vid never in any cav_ids row (data bug)
Read-only over the runs; no evaluate/suffix/dispatcher/insertion files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

WM_N1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WM_N1))

RUNS = [
    "w8658m2600p10t360__w-lo-early",
    "w8664m2600p10t360__w-lo-early",
    "w8667m2700p10t420__w-lo-early",
    "w8669m2700p10t360__w-mid-early",
    "w8669m2700p10t360__w-mid-late",
    "w8670m2700p10t300__w-lo-late",
    "w8682m2400p20t240__w-lo-late",
    "w8684m2700p20t360__w-lo-early",
    "w8686m2600p20t240__w-mid-late",
    "w8692m2600p40t300__w-lo-late",
    "w8694m2600p40t240__w-lo-late",
    "w8697m2700p40t240__w-lo-late",
    "w8698m2600p40t300__w-lo-late",
    "w8700m2600p40t420__w-mid-late",
    "w8703m2600p40t540__w-lo-early",
    "w8704m2700p40t480__w-mid-late",
    "w8705m2700p40t300__w-lo-early",
    "w8705m2700p40t300__w-lo-late",
]
RUNS_ROOT = Path("E:/research-dense/runs/campaign_walls_pilot_chunk4")
EPS = 1e-9


def episodes(run_dir: Path) -> list[dict]:
    path = run_dir / "run" / "wall_events.jsonl"
    eps: dict[str, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            wid = str(ev.get("wall_id"))
            ep = eps.setdefault(wid, dict(wall_id=wid, request=None,
                                          assembled=None, terminal=None,
                                          members=()))
            kind = ev.get("kind")
            ts = float(ev["timestamp_s"])
            if kind == "wall_request":
                ep["request"] = ts
                ep["members"] = tuple(str(m) for m in ev.get("members", ()))
            elif kind == "wall_event":
                e = ev.get("event")
                if e == "assembly_start":
                    ep["assembly_start"] = ts
                elif e == "assembled":
                    ep["assembled"] = ts
                elif e in ("released", "safety_abort"):
                    ep["terminal"] = (e, ts, ev.get("reason"),
                                      ev.get("original_reason"))
    return [eps[k] for k in sorted(eps)]


def diagnose(run: str) -> dict:
    run_dir = RUNS_ROOT / run
    eps = episodes(run_dir)
    members = {m for ep in eps for m in ep["members"]}
    phys = run_dir / "run" / "physics_steps.jsonl"
    ever_cav: set[str] = set()
    first_issue: dict[str, dict] = {}
    last_in_veh: dict[str, float] = {m: None for m in members}
    departed: dict[str, float] = {m: None for m in members}
    junction_ticks: dict[str, list] = {m: [] for m in members}
    n_rows = 0
    with open(phys, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = float(r["time_s"])
            n_rows += 1
            vehs = r.get("vehicles") or {}
            cav = {str(x) for x in r.get("cav_ids", ())}
            ever_cav |= cav
            for vid in (str(x) for x in r.get("departed_ids", ())):
                if vid in departed and departed[vid] is None:
                    departed[vid] = t
            for m in members:
                vv = vehs.get(m)
                if vv is not None:
                    last_in_veh[m] = t
                    lid = vv.get("lane_id")
                    if lid is None or ":" in str(lid):
                        junction_ticks[m].append(t)
            for ep in eps:
                req = ep["request"]
                term = ep["terminal"][1] if ep["terminal"] else None
                if req is None or t + EPS < req:
                    continue
                if term is not None and t > term + EPS:
                    continue
                for m in ep["members"]:
                    if m in first_issue:
                        continue
                    if m not in vehs:
                        first_issue[m] = dict(
                            run=run, wall=ep["wall_id"], tick=t,
                            member=m, category="absent_from_vehicles",
                            request=req, terminal=term,
                            in_row_cav_report=m in cav,
                            ever_cav=m in ever_cav,
                            on_junction_lane=False)
                    elif m not in ever_cav:
                        lid = (vehs.get(m) or {}).get("lane_id")
                        first_issue[m] = dict(
                            run=run, wall=ep["wall_id"], tick=t,
                            member=m,
                            category="present_but_never_cav_reported",
                            request=req, terminal=term,
                            in_row_cav_report=m in cav,
                            ever_cav=False,
                            lane_id=str(lid),
                            on_junction_lane=lid is None
                            or ":" in str(lid))
    # member absent while wall active at ALL: verify vs terminal
    out = dict(run=run, n_rows=n_rows, episodes=[], members={},
               departed=departed, first_issue=first_issue)
    for ep in eps:
        out["episodes"].append(dict(
            wall_id=ep["wall_id"], request_s=ep["request"],
            assembled_s=ep["assembled"],
            terminal=(ep["terminal"][0], ep["terminal"][1],
                      ep["terminal"][2], ep["terminal"][3])
            if ep["terminal"] else None, members=list(ep["members"])))
    for m in members:
        out["members"][m] = dict(
            last_in_vehicles_s=last_in_veh[m], departed_at_s=departed[m],
            ever_in_cav_ids=m in ever_cav,
            junction_lane_ticks=len(junction_ticks[m]),
            first_junction_tick=junction_ticks[m][0]
            if junction_ticks[m] else None)
    return out


def main() -> None:
    results = [diagnose(r) for r in RUNS]
    out_path = WM_N1 / "J4_WALLTOKEN_DIAGNOSIS_RAW.json"
    out_path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for res in results:
        print(f"== {res['run']} rows={res['n_rows']}")
        for ep in res["episodes"]:
            print(f"   wall {ep['wall_id']} req={ep['request_s']} "
                  f"asm={ep['assembled_s']} term={ep['terminal']} "
                  f"members={ep['members']}")
        for m, info in sorted(res["members"].items()):
            print(f"   member {m}: last_in_veh={info['last_in_vehicles_s']} "
                  f"departed={info['departed_at_s']} ever_cav="
                  f"{info['ever_in_cav_ids']} junc_ticks="
                  f"{info['junction_lane_ticks']}")
        for m, fi in sorted(res["first_issue"].items()):
            print(f"   FIRST-ISSUE {json.dumps(fi, default=str)}")


if __name__ == "__main__":
    main()
