"""Append REPORT_V04A.md sections: %-of-mean errors, v0.3.2 comparison,
report strata, always-reported checklist, disclosures.

Offline report rendering from artefacts already written by the ONE CAL
read (eval_v04a.json per results dir) + the TRAIN mean-target scale
(train_target_means_v04a.json).  No CAL file is opened; no model is
refitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WD01 = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[1]))

import wdpaths  # noqa: E402

RES_ROOT = Path(wdpaths.RESEARCH_DENSE_ROOT) / "output" / "partB_results"
DIRS = {"strict": "v0.4a_strict", "detector": "v0.4a_detector",
        "strict_noaux": "v0.4a_strict_noaux"}
TAUS = ("30", "60", "120", "300", "570")


def load(name):
    return json.loads((RES_ROOT / DIRS[name] / "eval_v04a.json")
                      .read_text(encoding="utf-8"))


def main() -> int:
    means = json.loads(
        (HERE / "train_target_means_v04a.json").read_text(encoding="utf-8"))
    ev = {k: load(k) for k in DIRS}
    scale = {k: ("all" if k != "strict_noaux" else "primary")
             for k in DIRS}

    def mean_tau(name, tau):
        key = scale[name]
        return means["per_tau"][tau][f"mean_{key}"]

    out = []
    out.append("\n## Appendix A: errors as % of the mean target\n")
    out.append(
        "Scale: TRAIN mean remaining-delay target per tau over the TRAIN "
        "rows of the loaded dataset (the climatology comparator "
        "population: all TRAIN rows for strict/detector, non-auxiliary "
        "rows for the ablation; "
        "`partB_v0data/v04a_results/train_target_means_v04a.json`). "
        "CAL files were not re-opened for this section.\n")
    for name in DIRS:
        res = ev[name]
        out.append(f"\n### {res['label']}\n")
        out.append("| tau | MAE model (s) | % of mean target | CRPS model (s) "
                   "| % of mean target | mean target (s) |")
        out.append("|---|---|---|---|---|---|")
        for row in res["a1"]["per_tau"]:
            tau = f"{row['tau']:.0f}"
            mt = mean_tau(name, tau)
            out.append(
                f"| {tau} | {row['mae_model']:.1f} "
                f"| {100.0 * row['mae_model'] / mt:.1f}% "
                f"| {row['crps_model']:.1f} "
                f"| {100.0 * row['crps_model'] / mt:.1f}% | {mt:.1f} |")

    out.append("\n## Appendix B: comparison to v0.3.2_strict (reference)\n")
    s = ev["strict"]
    out.append(
        "v0.3.2_strict numbers are from ITS OWN single CAL read on the "
        "v0.3.2 root (`partB_results/v0.3.2_strict/eval_report.md`, 10959 "
        "CAL rows); v0.4a is on the new TRUE-policy root (7542/7173/6435/"
        "4221/900 CAL rows per tau). The two CAL sets differ (chunk-4 "
        "worlds added; native arms restored), so this is a like-for-like "
        "family comparison, NOT a paired difference.\n")
    out.append("")
    out.append("| quantity | v0.3.2_strict | v0.4a strict (HEADLINE) |")
    out.append("|---|---|---|")
    out.append("| native reference | random per-CAV SWITCHING mix "
               "(WM-72 caveat) | restored TRUE fleet-native arms |")
    out.append(f"| overall CRPS model vs climatology | 977.0 vs 1911.4 "
               f"(coverage 0.800, in band) | "
               f"{s['a1']['crps_overall']:.1f} vs "
               f"{s['a1']['crps_climatology_overall']:.1f} "
               f"(coverage {s['a1']['coverage_overall']:.3f}, "
               f"{'in' if s['a1']['verdict'] == 'pass' else 'OUT of'} "
               f"band) |")
    out.append("| A2 vs ridge, gating 30/60/120 | PASS (FAIL at 300/570 "
               "reported) | PASS (strict: 300 PASS, 570 FAIL reported) |")
    out.append("| A1 overall | (no formal A1; coverage in band) | "
               f"{s['a1']['verdict']} (coverage 0.880 above the 0.85 "
               "upper band edge) |")
    out.append("| contrast evidence | damping vs switching-mix 'native' "
               "at tau=570 only: sign 0.76, Spearman 0.030 | full A3: "
               "sign accuracy PASS at all gating taus, but dV CRPS does "
               "not beat the zero-contrast / TRAIN-cell-mean baselines "
               "at 60/120/300 -> A3 fail |")
    out.append("")
    out.append(
        "Reading: v0.4a improves per-tau sharpness (CRPS below v0.3.2 at "
        "30-300 s on a harder, larger CAL set) and keeps A2-gating, but "
        "(i) the conformal 10-90% band is too WIDE on CAL (coverage "
        "0.880 > 0.85: A1 fail) and (ii) the learned contrast distribution "
        "is not yet sharp enough to beat a zero contrast on |dY|>median "
        "pairs: the TRUE damping-vs-native and PI-vs-native gaps are "
        "smaller and noisier than the old switching-mix contrasts "
        "(PB-RESCORE1 premise). VERDICT per registration s.6: FAIL "
        "(strict, detector, ablation all FAIL; A3(ii) is the binding "
        "constraint).")

    out.append("\n## Appendix C: demand / penetration breakdowns (report "
               "strata only)\n")
    out.append("Coverage and CRPS by demand and penetration, strict "
               "(HEADLINE); detector and ablation strata are in their "
               "`eval_v04a.json` (`a1.by_demand`, `a1.by_penetration`).\n")
    out.append("| stratum | n | coverage | CRPS model | CRPS climatology |")
    out.append("|---|---|---|---|---|")
    for col, label in (("by_demand", "demand"), ("by_penetration",
                                                 "penetration")):
        for v, d in sorted(s["a1"][col].items(), key=lambda kv: float(kv[0])):
            out.append(f"| {label}={v} | {d['n']} | {d['coverage']:.3f} "
                       f"| {d['crps_model']:.1f} "
                       f"| {d['crps_climatology']:.1f} |")

    out.append("\n## Appendix D: always-reported items (registration s.6)\n")
    out.append("")
    out.append("- per-policy coverage: per-variant section tables above.")
    out.append("- switching-native and fixed-native contrasts: A3 tables "
               "above (reported, not gating).")
    out.append("- auxiliary-data ablation: strict_noaux variant above "
               "(removing aux rows changes A1 coverage 0.880 -> 0.879 and "
               "leaves every verdict FAIL: the failure is not an "
               "auxiliary-data artefact).")
    out.append("- comonotone dV sensitivity: `CRPS dV comonotone` column "
               "of every A3 table (same pass/fail pattern as "
               "independent).")
    out.append("- demand / penetration strata: Appendix C.")
    out.append("- TRAIN-OOB versions of A1-A3 (selection-free of CAL, "
               "in-sample for the conformal shifts): `TRAIN-OOB verdict` "
               "lines above; full tables in each `eval_v04a.json` "
               "(`train_oob`).")

    out.append("\n## Appendix E: disclosures (executor + process)\n")
    out.append("")
    out.append("- Executor change: the original executor (Sonnet) was "
               "stopped by owner decision 2026-09-30 mid dry-run; this "
               "resume was executed by GLM Flash (ZCode) in worktree "
               "`E:/rd-wt/wd01v04c`, branch `zcode/wd01v04c` from "
               "`15def5f`, after a runtime-limit resume of the session.")
    out.append("- Owner-stop interruption: the first dry-run (2026-09-30 "
               "17:03 local, PID 6816) was interrupted by the owner stop "
               "(laptop taken out) BEFORE any CAL read; 0 "
               "`CAL_READ_ONCE_v04a.json` existed then and now before "
               "this resume (searched wd01v04a, wd01v04b, wd01v04c and "
               "`partB_results`). See "
               "`partB_v0data/v04a_results/INTERRUPTED_DRYRUN_20260930.txt`.")
    out.append("- ONE CAL read: performed exactly once in this resume "
               "(lock `CAL_READ_ONCE_v04a.json` written before the first "
               "CAL row; started/finished per the lock; no refit after "
               "it).")
    out.append("- MemoryError failed attempts: the first strict/detector "
               "fit launches and the first backup launch died at start "
               "with MemoryError (receipts in "
               "`partB_v0data/v04a_results/*.err`, logs under "
               "`logs/attempt1_memoryerror/`); the half-created result "
               "dirs were refitted in place (attempt 2), all verified.")
    out.append("- `git reset -q HEAD -- <paths>` was run once by mistake "
               "in the previous session to unstage paths (plus "
               "`git rm --cached` for staged `__pycache__`); nothing was "
               "discarded. Disclosed because the rules forbid git reset "
               "in any form (also CHANGES_V04A.md item 8).")
    out.append("- CAL exposure (from the registration): chunk-2/3 CAL was "
               "read once each by v0.3.2 and RESCORE1; chunk-4 CAL is "
               "also C4L1's Part A look set. This run is the disclosed "
               "third read of chunk-2/3 CAL.")
    out.append("- Memory gates: before the dry-run 13.68 GB free RAM / "
               "22.0 GB free VM; before the CAL read 16.65 GB / 32.52 GB "
               "(both >= 12 / >= 8 GB; readings logged in "
               "`dryrun_resume.ram.log` / `cal_read.ram.log`). Runs were "
               "detached, BelowNormal priority, <= 4 workers, "
               "WD_FORBID_D=1, zero SUMO/TraCI.")

    report = WD01 / "REPORT_V04A.md"
    with report.open("a", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print(f"appended {len(out)} lines to {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
