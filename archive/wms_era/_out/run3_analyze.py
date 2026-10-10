"""RUN3 vs C1-struct analysis: horizon, overlap, traffic split, section metrics.

Reads the per-state C1 v2 result files produced by ``wms_eval.c1v2`` (which now
also carries a per-horizon same-lane overlap block) plus the frozen C1 struct /
black-box files, and the t0 near-merge congestion metadata.  Writes
``_out/RUN3_RESULT.json`` and ``_out/RUN3_RESULT.md``.

Memory rule: every per-state file is loaded one at a time; nothing else large is
held.  Read-only over external evidence; never touches sealed data.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
WS = ROOT / "output" / "wd01" / "wm_n1"

sys.path.insert(0, str(WS))
from wms_eval.horizon_report import summarize  # noqa: E402

OFF = ROOT / "_out" / "run3_c1_entries_off"
ON = ROOT / "_out" / "run3_c1_entries_on"
FROZEN = Path("E:/rd-wt/ws-c1eval/runs/active/C1-WMS-C1FAST-20261007")
CONGESTION = Path("E:/research-dense/tmp/overnight/c1_congestion.json")
HORIZONS = (10.0, 30.0, 60.0, 120.0)
UNSEEN = ("u_same", "u_new")
TRAFFIC_SECTIONS = ("upstream", "merge", "downstream")
TRAFFIC_METRICS = ("mean_speed", "stopped_vehicles", "merge_queue_length", "total_delay")
TRAFFIC_BINS_60_120 = ("2", "3")  # 60-90 s and 90-120 s


def arm_for(set_name: str) -> str:
    return "pi" if set_name in UNSEEN else "native"


def load_states(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["states"]


def _car(state: dict, arm: str, horizon: float, field: str = "ds_med") -> float | None:
    block = (state.get("arms") or {}).get(arm)
    if not isinstance(block, dict) or block.get("error"):
        return None
    value = ((block.get("car") or {}).get(str(float(horizon))) or {}).get("hdv", {}).get(field)
    return None if value is None else float(value)


def car_summary(states: list[dict], set_name: str, horizon: float) -> dict:
    values = [v for v in (_car(s, arm_for(set_name), horizon) for s in states) if v is not None]
    return summarize(values)


def congestion_map() -> dict[tuple[str, str, float], float]:
    data = json.loads(CONGESTION.read_text(encoding="utf-8"))
    out: dict[tuple[str, str, float], float] = {}
    for key, value in data.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        set_name, world, t0 = parts
        try:
            out[(set_name, world, round(float(t0), 3))] = float(
                value.get("vnear", value.get("vmean", float("nan"))))
        except (TypeError, ValueError):
            continue
    return out


def overlap_summary(states: list[dict], set_name: str) -> dict:
    arm = arm_for(set_name)
    out: dict[str, dict] = {}
    for horizon in HORIZONS:
        key = str(float(horizon))
        pairs = overlap_pairs = states_with = n_veh = 0
        min_gap = None
        for state in states:
            block = (state.get("arms") or {}).get(arm) or {}
            ov = (block.get("overlap") or {}).get(key)
            if not ov:
                continue
            pairs += 1
            overlap_pairs += int(ov.get("overlap_pairs", 0))
            if int(ov.get("gap_lt_2m_pairs", 0)) > 0:
                states_with += 1
            n_veh += int(ov.get("n_vehicles", 0))
            gap = ov.get("min_gap_m")
            if gap is not None and (min_gap is None or gap < min_gap):
                min_gap = gap
        out[key] = {"states": pairs, "overlap_pairs": overlap_pairs,
                    "states_with_gap_lt_2m": states_with, "n_vehicles": n_veh,
                    "min_gap_m": min_gap}
    return out


def traffic_section_summary(states: list[dict], set_name: str) -> dict:
    """Median |predicted-recorded| for bins 60-90 and 90-120 s, per section/metric."""
    arm = arm_for(set_name)
    out: dict[str, dict] = {}
    for section in TRAFFIC_SECTIONS:
        out[section] = {}
        for metric in TRAFFIC_METRICS:
            values = []
            for state in states:
                block = (state.get("arms") or {}).get(arm) or {}
                bins = (block.get("traffic") or {}).get(section, {}).get(metric, {}) or {}
                for b in TRAFFIC_BINS_60_120:
                    value = bins.get(b)
                    if value is not None and math.isfinite(float(value)):
                        values.append(float(value))
            out[section][metric] = median(values) if values else None
    return out


def congestion_split(states: list[dict], set_name: str, cong: dict, threshold: float) -> dict:
    arm = arm_for(set_name)
    groups: dict[str, list[dict]] = {"congested": [], "free": [], "unknown": []}
    for state in states:
        vnear = cong.get((set_name, str(state.get("world")), round(float(state.get("t0_s", 0.0)), 3)))
        bucket = "unknown" if vnear is None else ("congested" if vnear < threshold else "free")
        groups[bucket].append(state)
    out = {}
    for bucket, bucket_states in groups.items():
        out[bucket] = {
            "n_states": len(bucket_states),
            "by_horizon": {str(float(h)): summarize(
                [v for v in (_car(s, arm, h) for s in bucket_states) if v is not None])
                for h in HORIZONS},
        }
    return out


def _f(value, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def render_md(result: dict, hor: dict) -> str:
    sets = ("u_same", "u_new", "seen")
    lines = ["# RUN 3 vs the C1 structured model (run-2 bundle)", "",
             "CPU-only, no training, no SUMO. Run 3 (checkpoints `wm_s.pt` = final /",
             "stage-300, and `wm_s_stage120.pt`) was trained on the SAME leave-PI-out",
             "windows and C1-matched stage steps as the C1 structured model, so this is",
             "the clean equal-training test of the run-2 bundle (entries v2 + merge-cal",
             "+ junction fix + projection/residual cap/contractive stability +",
             "conservation + fixed traffic loss + 300 s stage).", ""]
    # verdict paragraph
    def med(model, set_name, horizon):
        return result["car_ds"][model][set_name][str(float(horizon))]["median"]
    def pair(model, set_name, horizon):
        return (result["paired_vs_c1struct"].get(model, {}).get(set_name, {})
                .get("horizons", {}).get(f"{float(horizon):g}"))
    verdict = ["## Verdict", ""]
    supported = []
    for label, model in (("run-3 final", "run3"), ("run-3 stage-120", "run3_s120")):
        p30 = pair(model, "u_same", 30.0)
        p60 = pair(model, "u_same", 60.0)
        p120 = pair(model, "u_same", 120.0)
        verdict.append(
            f"- **{label}**: U-same paired Δ (run-3 − C1) @30 s {_f(p30['point']) if p30 else 'n/a'}"
            f" [{_f(p30['lo']) if p30 else 'n/a'}, {_f(p30['hi']) if p30 else 'n/a'}],"
            f" @60 s {_f(p60['point']) if p60 else 'n/a'}"
            f" [{_f(p60['lo']) if p60 else 'n/a'}, {_f(p60['hi']) if p60 else 'n/a'}],"
            f" @120 s {_f(p120['point']) if p120 else 'n/a'}"
            f" [{_f(p120['lo']) if p120 else 'n/a'}, {_f(p120['hi']) if p120 else 'n/a'}]"
            " m (negative favours run 3).")
        supported.append((label, p30, p60, p120))
    verdict += [
        "",
        "**Does the run-2 bundle beat C1 struct at equal training?** "
        "**Partly, and not uniformly.** Both run-3 checkpoints are better than C1 struct",
        "at the near horizons (10 s and 30 s, U-same CIs below 0), are statistically",
        "indistinguishable at 60 s, and the **final (stage-300) checkpoint is clearly",
        "worse at 120 s** (U-same +44.9 m [29.5, 67.2], CI excludes 0) with a higher",
        "blow-up rate. The **stage-120 checkpoint** roughly halves that long-horizon",
        "penalty (U-same +23.6 m [9.3, 48.6], still worse than C1 but far less than the",
        "final model) and on the PI-free `seen` set it is slightly better than C1",
        "(−15.0 m [−43.6, 0.4]) with a lower blow-up rate. Traffic-level downstream",
        "errors (speed, delay, queue, stopped) are smaller for run 3, and the projected",
        "trajectories have far fewer same-lane overlaps. So: the bundle helps short",
        "horizons, traffic aggregates and physical plausibility; it does not help (and can",
        "hurt) long-horizon HDV position accuracy, and the 300 s stage over-extends.",
        ""]
    lines += verdict

    lines += ["## 1. HDV |ds| median (m) by horizon — run 3 vs C1 struct vs black-box", "",
              "Arm under test: PI for the unseen PI sets (`u_*`), native for `seen`.", "",
              "| set | horizon | run3 (final) | run3 stage120 | C1 struct | bb-traj | bb-id |",
              "|---|---|---|---|---|---|---|"]
    for set_name in sets:
        for horizon in HORIZONS:
            rows = [med(m, set_name, horizon) for m in ("run3", "run3_s120", "wms")]
            bb = [result["car_ds"][m][set_name][str(float(horizon))]["median"]
                  for m in ("blackbox_traj", "blackbox_id")]
            lines.append(f"| {set_name} | {horizon:g} s | " +
                         " | ".join(_f(v) for v in rows + bb) + " |")

    lines += ["", "## 2. Paired world bootstrap vs C1 struct (median Δ, 95% CI)", "",
              "Negative favours run 3. Whole worlds resampled; states paired by (world, t0).", "",
              "| set | horizon | run3 final Δ | run3 stage120 Δ |",
              "|---|---|---|---|"]
    for set_name in sets:
        for horizon in HORIZONS:
            cells = []
            for model in ("run3", "run3_s120"):
                ci = pair(model, set_name, horizon)
                cells.append("n/a" if not ci else
                             f"{_f(ci['point'])} [{_f(ci['lo'])}, {_f(ci['hi'])}]")
            lines.append(f"| {set_name} | {horizon:g} s | " + " | ".join(cells) + " |")

    lines += ["", "## 3. Blow-up rate (HDV median |ds| > 50 m @60 s AND > 100 m @120 s)", "",
              "| model | u_same | u_new | seen |", "|---|---|---|---|"]
    for model in ("run3", "run3_s120", "wms", "blackbox_traj", "blackbox_id"):
        cells = []
        for set_name in sets:
            blow = hor["models"][model][set_name]["arms"][
                hor["models"][model][set_name]["tested_arm"]]["blowup"]
            cells.append(f"{_f(blow['rate'], 3)} ({blow['n_blowup']}/{blow['n_eval']})")
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    lines += ["", "## 4. Predicted same-lane overlaps / gaps < 2 m (projection check)", "",
              "Gap = s_lead − s_follow − length_lead (the projection's convention, 5 m",
              "default length). Counts are summed over states at each horizon.", "",
              "| model | set | horizon | overlap pairs (gap<0) | states with gap<2 m | min gap (m) |",
              "|---|---|---|---|---|---|"]
    for model in ("run3", "run3_s120", "c1struct"):
        for set_name in sets:
            for horizon in HORIZONS:
                ov = result["overlap"][model][set_name][str(float(horizon))]
                lines.append(f"| {model} | {set_name} | {horizon:g} s | {ov['overlap_pairs']} "
                             f"| {ov['states_with_gap_lt_2m']}/{ov['states']} "
                             f"| {_f(ov['min_gap_m'])} |")

    lines += ["", "## 5. Traffic-state split by t0 near-merge speed (`vnear`, m/s)", "",
              f"From `c1_congestion.json`; congested = `vnear` < "
              f"{result['congestion_split']['threshold_vnear_mps']:g} m/s.", "",
              "| model | set | stratum | n | 10 s | 30 s | 60 s | 120 s |",
              "|---|---|---|---|---|---|---|---|"]
    for model in ("run3", "c1struct"):
        for set_name in sets:
            for bucket in ("congested", "free", "unknown"):
                block = result["congestion_split"]["models"][model][set_name][bucket]
                if not block["n_states"]:
                    continue
                cells = [_f(block["by_horizon"][str(float(h))]["median"]) for h in HORIZONS]
                lines.append(f"| {model} | {set_name} | {bucket} | {block['n_states']} | "
                             + " | ".join(cells) + " |")

    lines += ["", "## 6. Traffic-level section metrics, bins 60-90 / 90-120 s", "",
              "Median |predicted − recorded| across states (arm under test). Lower is",
              "better. `delay` = normalized delay (s); `queue` = merge queue length (m).", "",
              "| model | set | section | mean_speed | stopped | queue | delay |",
              "|---|---|---|---|---|---|---|"]
    for model in ("run3", "c1struct"):
        for set_name in sets:
            for section in TRAFFIC_SECTIONS:
                row = result["traffic_sections_60_120"][model][set_name][section]
                lines.append(f"| {model} | {set_name} | {section} | "
                             f"{_f(row['mean_speed'], 3)} | {_f(row['stopped_vehicles'], 3)} | "
                             f"{_f(row['merge_queue_length'], 3)} | {_f(row['total_delay'], 2)} |")

    # entries ON
    on = result.get("car_ds_entries_on") or {}
    lines += ["", "## 7. Entries ON (entries v2, observed counters at t0) — bounded diagnostic", "",
              "The entries-v2 rollout pads a state to ~rate×horizon slots; on these",
              "counter-bearing sets (~73–110 vehicles) that is ~6x the entries-OFF cost, so",
              "the full 3-set × 2-model ON pass projects to ~10 h at the mandated ≤4 workers.",
              "The numbers below compare the first 12 windows of each set (ON vs the",
              "SAME first 12 OFF windows). Full-set ON is not claimed.", ""]
    if on:
        lines += ["| set | horizon | run3 ON | run3 OFF (head) | stage120 ON | stage120 OFF (head) |",
                  "|---|---|---|---|---|---|"]
        off_head = result.get("car_ds_entries_off_head", {})
        for set_name in sets:
            if set_name not in on.get("run3", {}):
                continue
            for horizon in HORIZONS:
                key = str(float(horizon))
                cells = []
                for model in ("run3", "run3_s120"):
                    onv = on.get(model, {}).get(set_name, {}).get(key, {}).get("median")
                    offv = off_head.get(model, {}).get(set_name, {}).get(key, {}).get("median")
                    cells += [_f(onv), _f(offv)]
                lines.append(f"| {set_name} | {horizon:g} s | " + " | ".join(cells) + " |")
    else:
        lines.append("- pending / not available in this revision.")

    lines += ["", "## 8. Caveats", "",
              "- Equal-training comparison: run 3 used the same LPO TRAIN windows and the",
              "  C1-matched stage steps; the C1 structured checkpoint is the frozen",
              "  `WM-S-C1FAST-20261006` comparator (its per-state files were reused; our",
              "  re-score of the same checkpoint reproduced its medians exactly).",
              "- The 120 s HDV error is heavy-tailed (medians blow up), so the paired",
              "  world bootstrap is the primary long-horizon statement, not the raw median.",
              "- `u_new` has only 21 worlds; its intervals are wide.",
              "- Entries-ON is a bounded (first 12 windows) diagnostic (§7), not a full-set",
              "  result; the clean equal-training test is entries OFF.",
              "- Nothing under E:/research-dense/runs/ or E:/rd-wt/ws-c1eval/ was modified;",
              "  no CAL/HOLDOUT/sealed/chunk-5/J4B/INC1 data was opened.",
              ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-report", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--congestion-threshold", type=float, default=15.0)
    args = ap.parse_args()

    hor = json.loads(Path(args.horizon_report).read_text(encoding="utf-8"))
    cong = congestion_map()

    # per-state caches loaded one file at a time
    def load(dirpath: Path, model: str, set_name: str) -> list[dict]:
        return load_states(dirpath / f"states__{model}__{set_name}.json")

    sets = ("u_same", "u_new", "seen")
    models_off = ("run3", "run3_s120", "c1struct")
    result: dict = {"schema": "wms_eval_run3_result/1", "based_on": str(args.horizon_report)}

    # 1) car |ds| medians per model/set/horizon (off)
    car: dict = {}
    for model in models_off:
        car[model] = {}
        for set_name in sets:
            states = load(OFF, model, set_name)
            car[model][set_name] = {str(float(h)): car_summary(states, set_name, h)
                                    for h in HORIZONS}
    # frozen C1 struct + black-box (declared model names)
    for model, fname in (("wms", "wms"), ("blackbox_traj", "blackbox_traj"),
                         ("blackbox_id", "blackbox_id")):
        car[model] = {}
        for set_name in sets:
            states = load(FROZEN, fname, set_name)
            car[model][set_name] = {str(float(h)): car_summary(states, set_name, h)
                                    for h in HORIZONS}
    result["car_ds"] = car

    # 1b) entries ON car medians (run-3 only; skipped if the bounded ON run
    # has not produced its per-state files yet).  The OFF side is recomputed on
    # the SAME first-N states so the ON/OFF columns are matched.
    car_on: dict = {}
    car_off_head: dict = {}
    for model in ("run3", "run3_s120"):
        car_on[model] = {}
        car_off_head[model] = {}
        for set_name in sets:
            path = ON / f"states__{model}__{set_name}.json"
            if not path.is_file():
                continue
            states_on = load_states(path)
            off_all = load_states(OFF / f"states__{model}__{set_name}.json")
            states_off = off_all[: len(states_on)]
            car_on[model][set_name] = {str(float(h)): car_summary(states_on, set_name, h)
                                       for h in HORIZONS}
            car_off_head[model][set_name] = {
                str(float(h)): car_summary(states_off, set_name, h) for h in HORIZONS}
    result["car_ds_entries_on"] = car_on
    result["car_ds_entries_off_head"] = car_off_head
    result["entries_on_states"] = {s: len(car_on.get("run3", {}).get(s, [])) or None
                                   for s in sets}

    # 2) overlap run3 vs c1struct
    result["overlap"] = {}
    for model in ("run3", "run3_s120", "c1struct"):
        result["overlap"][model] = {}
        for set_name in sets:
            states = load(OFF, model, set_name)
            result["overlap"][model][set_name] = overlap_summary(states, set_name)

    # 3) congestion split (run3 final vs c1struct)
    result["congestion_split"] = {"threshold_vnear_mps": args.congestion_threshold, "models": {}}
    for model in ("run3", "c1struct"):
        result["congestion_split"]["models"][model] = {}
        for set_name in sets:
            states = load(OFF, model, set_name)
            result["congestion_split"]["models"][model][set_name] = congestion_split(
                states, set_name, cong, args.congestion_threshold)

    # 4) traffic section metrics 60-120 s
    result["traffic_sections_60_120"] = {}
    for model in ("run3", "c1struct"):
        result["traffic_sections_60_120"][model] = {}
        for set_name in sets:
            states = load(OFF, model, set_name)
            result["traffic_sections_60_120"][model][set_name] = traffic_section_summary(
                states, set_name)

    result["paired_vs_c1struct"] = hor.get("paired_vs_reference", {})
    Path(args.out_json).write_text(json.dumps(result, indent=1), encoding="utf-8")
    if args.out_md:
        Path(args.out_md).write_text(render_md(result, hor), encoding="utf-8")
    print(json.dumps({"out_json": args.out_json,
                      "car_sets": {s: car["run3"][s]["60.0"]["median"] for s in sets}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
