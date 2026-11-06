"""Markdown report of a merged qualification run (WMQUAL_BASELINE.md / WM_QUAL_REPORT.md).

    python -m tools.wzqual.report --run FINAL_DIR --out REPORT.md --title "..." [--compare OTHER_FINAL]

Reads GATE_VERDICT.json + WM_QUAL_METRICS.json; every primary criterion is listed with value,
threshold, status and the distance to the threshold; ``--compare`` adds a column for another run
(e.g. the WM2 baseline next to the fine-tuned model).  Numbers only, no interpretation is invented.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _f(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        if not math.isfinite(float(x)):
            return "n/a"
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    return str(x)


def _ci(c):
    if not c or c[0] is None:
        return ""
    return f" [{_f(c[0])}, {_f(c[1])}]"


def criteria_rows(m: dict, v: dict) -> list:
    d, a = m.get("distributional", {}), m.get("action_response", {})
    crit = v.get("criteria", {})
    cov = m.get("role_coverage", {})
    rows = []

    def add(cid, value, thr, gap):
        rows.append((cid, value, thr, crit.get(cid, {}).get("status", "n/a"), gap))

    add("C0_role_coverage", f"{_f(cov.get('fraction_T_candidates_supported'))} (roles {', '.join(cov.get('supported_roles', []))})",
        ">= 0.80; hold + lane_to", None)
    rd = d.get("rate_diff_pooled")
    add("D1_breakdown_rate", f"WM {_f((d.get('breakdown_rate_wm') or {}).get('300'))} vs real "
        f"{_f((d.get('breakdown_rate_truth') or {}).get('300'))}; diff {_f(rd)}{_ci(d.get('rate_diff_pooled_ci'))}",
        "abs(diff) <= 0.10 pooled, <= 0.20 per cell (n>=12)", None if rd is None else abs(rd) - 0.10)
    w1 = d.get("onset_w1_s")
    add("D2_onset", f"W1 {_f(w1, 1)} s (real events {d.get('n_truth_events')})", "<= 45 s",
        None if w1 is None else w1 - 45.0)
    q, qw = d.get("queue_rel_mean_err"), d.get("queue_norm_w1")
    add("D3_queue", f"rel err {_f(q)}; W1/SD {_f(qw)}", "abs(rel) <= 0.25 and W1/SD <= 0.50",
        None if q is None else abs(q) - 0.25)
    t = d.get("throughput_rel_err")
    add("D4_throughput", f"rel err {_f(t)}", "abs(rel) <= 0.08", None if t is None else abs(t) - 0.08)
    cr = d.get("zone_crps") or {}
    add("D5_zone_crps", "; ".join(f"{h} s WM {_f(cr.get(h, {}).get('wm'))} vs keep-speed {_f(cr.get(h, {}).get('keep_speed'))}"
                                  for h in ("120", "300")), "WM <= keep-speed at 120 and 300 s", None)
    w = a.get("within_moment_spearman_T")
    add("A1_within_moment_spearman_T", f"{_f(w)}{_ci(a.get('within_moment_spearman_T_ci'))} (moments {a.get('n_moments_ranked')})",
        ">= 0.30, CI low > 0", None if w is None else 0.30 - w)
    p = a.get("pooled_spearman_T")
    add("A2_pooled_spearman_T", f"{_f(p)}{_ci(a.get('pooled_spearman_T_ci'))} (merge map 0.103)",
        ">= 0.30, CI low > 0.10", None if p is None else 0.30 - p)
    pp = a.get("precision_positive_T") or {}
    need3 = None if pp.get("base_rate") is None else max(0.60, pp["base_rate"] + 0.10)
    add("A3_precision_positive_T", f"{_f(pp.get('precision'))} of {pp.get('n_pred_pos')} WM-positive "
        f"(base rate {_f(pp.get('base_rate'))}; merge map 0.407)", f">= {_f(need3)}",
        None if pp.get("precision") is None or need3 is None else need3 - pp["precision"])
    pk = a.get("precision_at_k_S") or {}
    need4 = None if pk.get("base_rate") is None else max(0.5, min(0.7, 1.3 * pk["base_rate"]))
    add("A4_precision_at_k_S", f"{_f(pk.get('precision'))} at k={pk.get('k')} of {pk.get('n')} "
        f"(base {_f(pk.get('base_rate'))})", f">= {_f(need4)}",
        None if pk.get("precision") is None or need4 is None else need4 - pk["precision"])
    rg = a.get("regret") or {}
    add("A5_regret", f"WM/random regret {_f(rg.get('ratio_wm_to_random'))}; WM gain {_f(rg.get('wm_gain'), 1)}"
        f"{_ci(rg.get('wm_gain_ci'))} veh-s; acted {_f(rg.get('acted_frac'))}",
        "ratio <= 0.50 and gain CI low > 0", None if rg.get("ratio_wm_to_random") is None
        else rg["ratio_wm_to_random"] - 0.5)
    ks = (a.get("keep_speed") or {}).get("pooled_spearman_T")
    add("A6_beats_keep_speed", f"WM {_f(p)} vs keep-speed {_f(ks)}", "WM > keep-speed",
        None if p is None or ks is None else ks - p)
    return rows


def render(run: Path, title: str, compare: Path | None = None) -> str:
    m = json.loads((run / "WM_QUAL_METRICS.json").read_text())
    v = json.loads((run / "GATE_VERDICT.json").read_text())
    rows = criteria_rows(m, v)
    cmp_rows = None
    if compare is not None:
        cm = json.loads((compare / "WM_QUAL_METRICS.json").read_text())
        cv = json.loads((compare / "GATE_VERDICT.json").read_text())
        cmp_rows = {r[0]: r for r in criteria_rows(cm, cv)}
    L = [f"# {title}", "",
         f"Verdict: **{v.get('verdict')}** (integrity failures: {v.get('integrity_failures') or 'none'}). "
         f"Gate spec sha {str(v.get('gate_spec_sha256'))[:12]}; metrics sha {str(v.get('metrics_sha256'))[:12]}.",
         f"Model: backend `{m.get('backend')}`, checkpoint {m.get('checkpoint')} (sha {str(m.get('checkpoint_sha256'))[:12]}), "
         f"physics {m.get('physics_source')} (physics sha {str(m.get('physics_sha256'))[:12]}); samples K={m.get('samples')}.",
         "", "## Primary criteria", "",
         "| criterion | value | pass rule | status | gap to threshold |" + (" compare |" if cmp_rows else ""),
         "|---|---|---|---|---|" + ("---|" if cmp_rows else "")]
    for cid, val, thr, st, gap in rows:
        g = "" if gap is None else f"{gap:+.3f}"
        extra = ""
        if cmp_rows:
            c = cmp_rows.get(cid)
            extra = f" {c[1]} ({c[3]}) |" if c else " |"
        L.append(f"| {cid} | {val} | {thr} | {st} | {g} |{extra}")
    d = m.get("distributional", {})
    if d:
        L += ["", "## Distributional detail", "",
              f"Starts {d.get('n_starts')} (not broken {d.get('n_starts_not_broken')}, broken at start "
              f"{d.get('n_broken_at_start')}); rule boundary {_f(d.get('rule_boundary_mps'), 2)} m/s; "
              f"backend predicts entrants: {d.get('backend_models_entries')}.", "",
              "| cell | n | real P(bd<=300 s) | WM | diff |", "|---|---|---|---|---|"]
        for c, r in (d.get("rate_diff_by_cell") or {}).items():
            L.append(f"| {c} | {r.get('n')} | {_f(r.get('truth'))} | {_f(r.get('wm'))} | {_f(r.get('diff'))} |")
        L += ["", f"Breakdown by 60/120/300 s - real {d.get('breakdown_rate_truth')}, WM {d.get('breakdown_rate_wm')}, "
              f"keep-speed {d.get('breakdown_rate_keep_speed')}.",
              f"Queue: rel err {_f(d.get('queue_rel_mean_err'))}, closing-lane {_f(d.get('closing_queue_rel_mean_err'))}, "
              f"extent {_f(d.get('extent_rel_mean_err'))}. Throughput rel err WM {_f(d.get('throughput_rel_err'))}, "
              f"keep-speed {_f(d.get('throughput_rel_err_keep_speed'))}."]
        sec = d.get("secondary", {})
        L += [f"Secondary: Brier(300) WM {_f(sec.get('brier_300'))} vs keep-speed {_f(sec.get('brier_300_keep_speed'))}; "
              f"AUC {_f(sec.get('auc_300'))}; recovery of broken starts {sec.get('recovery')}.",
              f"Lane-change rates per veh-h {sec.get('lane_change_rates')}."]
    a = m.get("action_response", {})
    if a:
        L += ["", "## Action response detail", "",
              f"Candidates scored {a.get('n_candidates')}; pooled Spearman S {_f(a.get('pooled_spearman_S'))}; "
              f"strict precision@k (S lower bound > 0) {a.get('precision_at_k_S_strict')}.",
              f"Keep-speed reference: {a.get('keep_speed')}.", "", "| family | n | Spearman T | precision WM-positive |",
              "|---|---|---|---|"]
        for fam, r in (a.get("by_family") or {}).items():
            L.append(f"| {fam} | {r.get('n')} | {_f(r.get('spearman_T'))} | {_f(r.get('precision_positive_T'))} |")
        for name, sec in (m.get("secondary_action_response") or {}).items():
            sm = sec.get("metrics", {})
            L += ["", f"Secondary `{name}`: pooled Spearman T {_f(sm.get('pooled_spearman_T'))}, within-moment "
                  f"{_f(sm.get('within_moment_spearman_T'))}, precision {sm.get('precision_positive_T')}, "
                  f"precision@k S {sm.get('precision_at_k_S')}."]
    se = m.get("secondary_errors")
    if se:
        L += ["", "## Secondary car-level errors (pooled per shard)", "", "```", json.dumps(se, indent=1)[:6000], "```"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="WM-D qualification report")
    ap.add_argument("--compare", default=None)
    args = ap.parse_args(argv)
    Path(args.out).write_text(render(Path(args.run), args.title, Path(args.compare) if args.compare else None),
                              encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
