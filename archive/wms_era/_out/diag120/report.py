"""Emit _out/DIAG120.json and _out/DIAG120.md from the analysis artifacts."""
from __future__ import annotations

import json
from pathlib import Path

D = Path(__file__).resolve().parent
OUT = D.parent


def rd(name):
    return json.loads((D / name).read_text(encoding="utf-8"))


stored = rd("stored_summary.json")
a1 = rd("analysis.json")
a2 = rd("analysis2.json")
ap = rd("params_analysis.json")

doc: dict = {
    "schema": "diag120/1",
    "question": "Why do WM-S HDV predictions blow up at 60-120 s?",
    "blow_up_definition": "HDV median |ds| > 50 m at 60 s or > 100 m at 120 s; "
                          "unseen u_same set, PI arm; rollouts re-run and verified against stored scores",
    "integrity": a1["integrity"],
    "stored_blow_rates": {},
    "mechanisms": {},
    "growth": {},
    "attribution": {},
    "oldhist_params": ap,
    "run3_entries": a2["task5"],
}

for model in ("c1_struct", "oldhist", "run3", "run3_s120", "blackbox_id",
              "blackbox_traj", "keep_speed", "pure_physics"):
    doc["stored_blow_rates"][model] = {
        s: {"n": v.get("n"), "n_blow": v.get("n_blow"), "frac": v.get("blow_frac"),
            "arm": v.get("arm")}
        for s, v in stored[model].items() if "n" in v}

doc["mechanisms"] = a2["task1"]
doc["growth"] = a2["task2"]
doc["attribution"] = {
    m: {k: v for k, v in a1["task3"][m].items() if k != "detail_top"}
    for m in a1["task3"]}
doc["attribution_structure"] = {
    "c1_struct": {"top_carriers": 200, "upstream_t0": 180, "merge_t0": 20,
                  "downstream_t0": 0, "lane_changers": 158, "median_onset_delay_s": 1.0},
    "oldhist": {"top_carriers": 200, "upstream_t0": 195, "merge_t0": 5,
                "downstream_t0": 0, "lane_changers": 181, "median_onset_delay_s": 1.5},
}

(OUT / "DIAG120.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")

m = doc["mechanisms"]
g = doc["growth"]
att = doc["attribution"]
st = doc["stored_blow_rates"]


def pct(x):
    return "n/a" if x is None else f"{100*x:.1f}%"


def f1(x):
    return "n/a" if x is None else f"{x:.2f}"


def f0(x):
    return "n/a" if x is None else f"{x:.0f}"


lines = []
lines += [
    "# DIAG120 - why WM-S HDV predictions blow up at 60-120 s",
    "",
    "Offline, CPU-only re-rollout diagnosis of the WM-S (C1-structured), OLDHIST "
    "(per-driver physics) and RUN-3 world models on the shifted v2 unseen set "
    "(`u_same`, PI arm). Blow-up = HDV median |ds| > 50 m at 60 s or > 100 m at "
    "120 s. Every re-rolled state was checked against the stored shift scores "
    "(0 disagreements), so the diagnosis runs on the same predictions that were scored.",
    "",
    "## Table 1 - blow-up rate by model (stored scores)",
    "",
    "| model | set | arm | states | blow-ups | rate |",
    "|---|---|---|---|---|---|",
]
for model in ("c1_struct", "oldhist", "run3", "run3_s120", "blackbox_id",
              "blackbox_traj", "keep_speed", "pure_physics"):
    for s in ("u_same", "u_new", "seen"):
        row = st[model].get(s)
        if row:
            lines.append(f"| {model} | {s} | {row['arm']} | {row['n']} | "
                         f"{row['n_blow']} | {pct(row['frac'])} |")
lines += [
    "",
    "Baselines bracket the problem: constant speed (`keep_speed`) and the residual-free "
    "`pure_physics` blow up on ~90% of unseen states, so a large part of the 120 s error "
    "is simply any sustained speed error integrating over the horizon. The structured "
    "models roughly halve that but still fail on 41-56% of u_same states (RUN-3 `run3` "
    "55.7%, `run3_s120` 51.4%).",
    "",
    "> **RUN-3 note.** RUN-3 blow-up rates and the entries comparison below come from its "
    "stored scores. Its per-state re-rollout could not be reproduced with the workspace "
    "loader: the stored `run3` scores require the projection/bounded-residual/merge-cal "
    "rollout switches (`projection_enabled`, `residual_cap_mps2=1.5`, `merge_cal=true`) "
    "that `wms_eval.wms_plugin` does not wire in, so a naive reload gives ds@60 of "
    "900/1438/1776 m where the stored scores are 5.0/11.3/13.7 m (even with the "
    "FeasibilityProjection monkey-patched in: 44.5/79.4/96.5 m). Tasks 1-3 are therefore "
    "derived from the two re-rollout-verified models (c1_struct, oldhist); RUN-3 is used "
    "for rates (Table 1) and the entries ablation (Table 6) only.",
    "",
    "## Table 2 - mechanism mix, blow-up vs non-blow-up (Task 1)",
    "",
    "`speed_bias` = time-mean merge (pred - recorded) mean speed, m/s (>0 = model drives "
    "too fast where real traffic slows). `missed_peak`/`phantom_peak` = largest "
    "recorded-minus-predicted / predicted-minus-recorded simultaneous stopped-vehicle gap "
    "in merge or upstream. Classes: missed (real jam the model under-forms), phantom "
    "(model invents a jam), fast-drift (systematic under-braking), drift (rest).",
    "",
    "| model | group | n | missed | phantom | fast-drift | drift | speed_bias m/s | missed_peak | phantom_peak |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for model in ("c1_struct", "oldhist"):
    for grp in ("blow", "nonblow"):
        e = m[model][grp]
        c = e["counts"]
        lines.append(f"| {model} | {grp} | {e['n']} | {c.get('missed',0)} | "
                     f"{c.get('phantom',0)} | {c.get('fast-drift',0)} | {c.get('drift',0)} | "
                     f"{f1(e['mean_speed_bias'])} | {f1(e['mean_missed_peak'])} | "
                     f"{f1(e['mean_phantom_peak'])} |")
lines += [
    "",
    f"Blow-ups are dominated by *missed jams* and systematic fast drift, not phantom jams. "
    f"Mean merge speed bias is {f1(m['c1_struct']['blow']['mean_speed_bias'])} m/s for "
    f"c1_struct blow-ups vs {f1(m['c1_struct']['nonblow']['mean_speed_bias'])} m/s for "
    f"non-blow-ups ({f1(m['oldhist']['blow']['mean_speed_bias'])} vs "
    f"{f1(m['oldhist']['nonblow']['mean_speed_bias'])} for oldhist). Across blow-ups the "
    f"speed bias correlates with ds@120 at r="
    f"{f1(m['c1_struct']['corr_bias_ds120_blow'])} (c1_struct) and "
    f"r={f1(m['oldhist']['corr_bias_ds120_blow'])} (oldhist). Phantom jam is a minority "
    f"({m['c1_struct']['blow']['counts'].get('phantom',0)}/57 c1_struct, "
    f"{m['oldhist']['blow']['counts'].get('phantom',0)}/72 oldhist).",
    "",
    "### Signed section bias (mean over states)",
    "",
    "| model | group | upstream bias m/s | merge bias m/s | downstream bias m/s |",
    "|---|---|---|---|---|",
    "| c1_struct | blow | +1.18 | +2.64 | -5.92 |",
    "| c1_struct | non | +0.11 | +1.18 | -5.06 |",
    "| oldhist | blow | +1.85 | +4.82 | -0.35 |",
    "| oldhist | non | +0.09 | +1.85 | -0.35 |",
]
lines += [
    "",
    "The blow-up signal is concentrated **upstream of the merge**: predicted upstream speed "
    "is 1.2-1.9 m/s too high on blow-ups versus ~0.1 m/s on non-blow-ups. The model fails to "
    "carry a recorded upstream queue/deceleration.",
    "",
    "## Table 3 - error-growth shape (Task 2)",
    "",
    "| model | blow-ups | median t(ds>20) | t(ds>50) | t(ds>100) | largest 5 s rise | where | recorded speed drop at jump | occurs during recorded braking |",
    "|---|---|---|---|---|---|---|---|---|",
]
for model in ("c1_struct", "oldhist"):
    e = a1["task2"][model]
    jj = g[model]
    lines.append(
        f"| {model} | {jj['n']} | {f0(e['t_cross20']['median'])} s | "
        f"{f0(e['t_cross50']['median'])} s | {f0(e['t_cross100']['median'])} s | "
        f"{f0(jj['median_rise_5s'])} m | t~{f0(jj['median_jump_t'])} s | "
        f"{f1(jj['median_rec_speed_drop'])} m/s | {pct(jj['coincides_missed_braking_frac'])} |")
sl = a1["task2"]
lines += [
    "",
    f"Growth is gradual then accelerating: the 30-60 s slope is "
    f"{f1(sl['c1_struct']['slope_30_60']['median'])} m/s (c1_struct) / "
    f"{f1(sl['oldhist']['slope_30_60']['median'])} m/s (oldhist) but the 60-90 s slope rises "
    f"to {f1(sl['c1_struct']['slope_60_90']['median'])} / "
    f"{f1(sl['oldhist']['slope_60_90']['median'])} m/s, and the single largest 5 s rise "
    f"({f0(g['c1_struct']['median_rise_5s'])}-{f0(g['oldhist']['median_rise_5s'])} m) lands "
    f"at t={f0(g['c1_struct']['median_jump_t'])}-{f0(g['oldhist']['median_jump_t'])} s, i.e. "
    f"near the end of the 120 s horizon and not at t0. In ~53% of blow-ups that late jump "
    f"coincides with a recorded braking event (recorded merge speed dropping ~1.8 m/s); the "
    f"model under-reacts and the position error accelerates. There is no single event at t0 "
    f"that starts it - it is compounded under-braking. CAV commands cannot be the trigger: "
    f"this harness teacher-forces every CAV from the recorded arm, so only HDVs are modelled.",
    "",
    "## Table 4 - attribution of the 20 worst blow-ups (Task 3)",
    "",
    "| model | HDVs/state | with onset | in merge at t0 | lane changers/state | top-10 carriers upstream at t0 | median onset delay vs leader |",
    "|---|---|---|---|---|---|---|",
]
s4 = doc["attribution_structure"]
for model in ("c1_struct", "oldhist"):
    e = att[model]
    s = s4[model]
    lines.append(
        f"| {model} | {f0(e['n_veh_mean'])} | {f0(e['n_onset_mean'])} | "
        f"{f0(e['n_merge_t0_mean'])} | {f0(e['n_lane_changers_mean'])} | "
        f"{s['upstream_t0']}/{s['top_carriers']} | {f1(s['median_onset_delay_s'])} s |")
lines += [
    "",
    f"In the 20 worst blow-ups there are ~{f0(att['c1_struct']['n_veh_mean'])}-"
    f"{f0(att['oldhist']['n_veh_mean'])} modelled HDVs, of which ~"
    f"{f0(att['c1_struct']['n_onset_mean'])}-{f0(att['oldhist']['n_onset_mean'])} have "
    f"crossed 10 m by 120 s. The error is **not carried at the merge**: 90-97% of the top-10 "
    f"carriers are already upstream (s<0, up to ~2 km back) at t0. Onset propagates backward "
    f"through the string: a vehicle crosses 10 m about "
    f"{f1(s4['c1_struct']['median_onset_delay_s'])}-"
    f"{f1(s4['oldhist']['median_onset_delay_s'])} s after its recorded-lane leader "
    f"({pct(a1['task3']['c1_struct']['prop_frac_positive'])} of leader/follower pairs have "
    f"the follower later). No entrants are involved (0 of the 20 worst contain an entrant), "
    f"and ~79-90% of top carriers perform at least one recorded lane change during the "
    f"horizon (merge-related weaving), consistent with the string being excited by local "
    f"lane changes too.",
    "",
    "## Table 5 - OLDHIST recovered per-driver IDM parameters (Task 4)",
    "",
    "Classes on the 20 worst + 15 non-blow-up states: starter = diverges before its leader, "
    "propagator = diverges after its leader, non = never crosses 10 m. Stability test = IDM "
    "low-speed sufficient condition a >= s0 / T^2 (Treiber & Kesting 2013; arXiv:2506.05909 Eq. 4).",
    "",
    "| class | n | median T (s) | median accel | median min_gap (m) | frac satisfying a>=s0/T^2 | median margin |",
    "|---|---|---|---|---|---|---|",
]
for c in ("starter", "propagator", "non"):
    p = ap["params"][c]
    s = ap["stability"][c]
    lines.append(
        f"| {c} | {s['n']} | {f1(p['time_headway']['median'])} | "
        f"{f1(p['accel']['median'])} | {f1(p['min_gap']['median'])} | "
        f"{pct(s['frac_stable'])} | {f1(s['median_margin'])} |")
lines += [
    "",
    f"Recovered parameters do **not** separate wave starters from non-starters: the encoder "
    f"pins accel at ~{f1(ap['params']['starter']['accel']['median'])} for every vehicle and "
    f"the stable fraction is essentially identical across classes "
    f"({pct(ap['stability']['starter']['frac_stable'])} starter, "
    f"{pct(ap['stability']['propagator']['frac_stable'])} propagator, "
    f"{pct(ap['stability']['non']['frac_stable'])} non). The recovered T and s0 vary "
    f"(T p10-p90 {f1(ap['params']['starter']['time_headway']['p10'])}-"
    f"{f1(ap['params']['starter']['time_headway']['p90'])} s) but the ~43% that violate the "
    f"low-speed stability condition do so equally in all classes, so per-driver string "
    f"instability is not the discriminator - the failure is a shared systematic under-braking.",
    "",
    "## Table 6 - RUN-3 entries off vs on (Task 5)",
    "",
    "Matched to the worlds present in both arms.",
    "",
    "| model | set | common states | blow-ups off | blow-ups on | mean ds@120 off | mean ds@120 on |",
    "|---|---|---|---|---|---|---|",
]
for key, v in a2["task5"].items():
    model, name = key.split("/")
    lines.append(f"| {model} | {name} | {v['common']} | {v['blow_off']} | {v['blow_on']} | "
                 f"{f0(v['mean_ds120_off'])} | {f0(v['mean_ds120_on'])} |")
lines += [
    "",
    "Turning observed-counter entries ON does not remove the blow-ups (the count is "
    "unchanged or slightly higher, and mean ds@120 changes by <30%). Entry dynamics are not "
    "the dominant cause, matching the 0-entrant attribution above (task 5 is on a 12-52 state "
    "overlap, so it is directional, not decisive).",
    "",
    "## Conclusions",
    "",
    f"1. **No phantom jam.** Predicted congestion does not exceed recorded congestion in "
    f"general (phantom only {m['c1_struct']['blow']['counts'].get('phantom',0)}/57 c1_struct, "
    f"{m['oldhist']['blow']['counts'].get('phantom',0)}/72 oldhist blow-ups). The models do "
    f"form queues; they form them too weakly and too late.",
    f"2. **Dominant mechanism = missed jam + systematic under-braking (fast drift).** Predicted "
    f"traffic is too fast in the upstream and merge sections (blow-up mean bias "
    f"+1.2/+2.6 m/s c1_struct, +1.9/+4.8 m/s oldhist; non-blow-ups ~+0.1/+1.2). The bias "
    f"integrates into the position error (corr(bias, ds@120) = "
    f"{f1(m['c1_struct']['corr_bias_ds120_blow'])} c1_struct, "
    f"{f1(m['oldhist']['corr_bias_ds120_blow'])} oldhist). It is a per-step deceleration "
    f"deficit, not an invented jam.",
    "3. **Growth is gradual then accelerating, with a late (90-110 s) jump, not an onset at "
    "t0.** Median crossing times: 20 m at ~57-67 s, 50 m at ~84-89 s, 100 m at ~96-99 s; the "
    "largest 5 s rise lands at t~107-108 s and coincides with a recorded braking event in "
    "~53% of cases. Fixing only the first 30 s cannot fix the 120 s error.",
    "4. **Error is carried upstream and propagates backward** (90-97% of top-10 carriers are "
    "upstream at t0; follower onset ~1.0-1.5 s after its leader). The merge zone itself is "
    "not where the error lives.",
    "5. **Lane changes are present but not entrants.** ~79-90% of top carriers change lanes "
    "during the horizon; 0 of the 20 worst contain an entrant, and RUN-3 entries-on does not "
    "reduce blow-ups. The disturbance source is recorded upstream queueing plus local lane "
    "changes, which the model under-propagates.",
    "6. **Per-driver parameters do not explain it.** OLDHIST's recovered IDM params are "
    "near-identical for wave starters and non-starters; the low-speed string-stability "
    "condition fails for ~43% of all vehicles regardless of class.",
    "",
    "## What each mechanism implies (fix directions, not yet evaluated)",
    "",
    "- **Under-braking / missed jam** -> increase deceleration fidelity and sensitise the "
    "residual to a closing/hard-braking leader (loss weighting on decelerating HDVs, an "
    "asymmetric speed-error loss, or a physics decel floor matched to the recorded "
    "distribution). This is the primary lever.",
    "- **Backward wave propagation deficit** -> the residual must transfer a leader's braking "
    "to followers within 1-2 s; add a string-propagation training term (short-horizon "
    "multistep loss on followers after a leader decelerates).",
    "- **Upstream queue carry** -> boundary/observation coverage upstream matters; the error "
    "origin is 1-2 km behind the merge, so upstream state coverage or a boundary queue term "
    "is needed.",
    "- **Entries** -> not a priority; entries-on does not reduce blow-ups.",
    "- **Per-driver params** -> not the discriminator; further driver-encoder work is unlikely "
    "to fix long-horizon blow-up.",
    "",
    "## Reproduce / artifacts",
    "",
    "- Rollouts: `_out/diag120/roll_struct_u_same.jsonl`, `roll_oldhist_u_same.jsonl` "
    "(4 workers, 0.5 s resolution, verified 0/140 flag disagreements vs stored scores).",
    "- Scripts: `_out/diag120/{summarize,rollouts,analyze,analyze2,extract_params,analyze_params}.py`.",
    "- Analysis: `_out/diag120/{stored_summary,analysis,analysis2,params_analysis}.json`, "
    "`_out/DIAG120.json`.",
    "",
]
(OUT / "DIAG120.md").write_text("\n".join(lines), encoding="utf-8")
print("wrote", OUT / "DIAG120.md", OUT / "DIAG120.json")
