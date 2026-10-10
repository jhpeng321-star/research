"""Markdown rendering of the WD-01-PB-V04A acceptance results."""
from __future__ import annotations

PREMISE = ("Ideal-sensing premise: lane bins come from the TRUE state; the "
           "reconstruction gap is measured separately (Part A). Inputs "
           "carry no penetration, pre_fix, speed_mode or world strings; "
           "t0 is unrecoverable (ctx col 1 = 0). Wall tokens are empty "
           "(v0.4b adds walls).")


def _f(x, nd=1):
    try:
        if x is None or x != x:
            return "nan"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _yn(b):
    return "PASS" if b else "fail"


def _table(head, rows):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


def _a1_lines(a1: dict, title: str) -> list[str]:
    v = a1["verdict"]
    L = [f"### {title}", ""]
    L += _table(["tau", "n", "CRPS model", "CRPS climatology",
                 "beats clim", "MAE(median)", "10-90 coverage"],
                [[f"{r['tau']:.0f}", r["n"], _f(r["crps_model"]),
                  _f(r["crps_climatology"]),
                  "yes" if r["crps_model"] < r["crps_climatology"] else "NO",
                  _f(r["mae_model"]), _f(r["coverage"], 3)]
                 for r in a1["per_tau"]])
    L += ["", f"Overall 10-90 coverage {_f(a1['coverage_overall'], 3)} "
              f"(band {v['band'][0]}-{v['band'][1]}: "
              f"{'in' if v['coverage_in_band'] else 'OUT'}); "
              f"A1 = {_yn(v['passes'])}", ""]
    return L


def _a2_lines(a2: dict, table: list[dict], title: str) -> list[str]:
    L = [f"### {title}", ""]
    L += _table(["tau", "n", "MAE model", "MAE ridge", "CRPS model",
                 "CRPS ridge", "verdict", "gating"],
                [[f"{r['tau']:.0f}", r["n"], _f(r["mae_model"]),
                  _f(r["mae_ridge"]), _f(r["crps_model"]),
                  _f(r["crps_ridge"]), r["result"],
                  "gating" if r["tau"] in (30.0, 60.0, 120.0)
                  else "reported"] for r in table])
    L += ["", f"A2-gating (tau 30, 60, 120 all PASS): "
              f"{_yn(a2['gating_pass'])}", ""]
    return L


def _a3_lines(records: list[dict], a3: dict, title: str,
              contrasts=None) -> list[str]:
    L = [f"### {title}", ""]
    head = ["contrast", "tau", "pairs (worlds)", "eligible |dY|>med",
            "sign acc [90% CI]", "(i)", "CRPS dV", "CRPS zero",
            "CRPS cell-mean", "(ii)", "CRPS dV comonotone",
            "tau pass"]
    rows = []
    for r in sorted(records, key=lambda x: (x["contrast"], x["tau"])):
        if contrasts and r["contrast"] not in contrasts:
            continue
        rows.append([r["contrast"], f"{r['tau']:.0f}",
                     f"{r['n_pairs']} ({r['n_worlds']})",
                     f"{r['n_eligible']} (thr {_f(r['train_median_abs_dY'])})",
                     f"{_f(r['sign_acc'], 3)} [{_f(r['sign_ci_lo'], 3)}, "
                     f"{_f(r['sign_ci_hi'], 3)}]", _yn(r["pass_i"]),
                     _f(r["crps_dv"]), _f(r["crps_zero"]),
                     _f(r["crps_cell_mean"]), _yn(r["pass_ii"]),
                     _f(r["crps_dv_comonotone"]), _yn(r["pass_tau"])])
    L += _table(head, rows)
    L += [""]
    for c, v in a3.items():
        L.append(f"- {c}: A3 (i AND ii at >= 2 of 3 taus in 60/120/300) "
                 f"= {_yn(v['passes'])} "
                 f"({v['n_taus_pass']}/{v['n_taus_evaluated']} taus)")
    L.append("")
    return L


def render_variant_report(res: dict) -> str:
    v = res["verdict"]
    L = [f"# eval_report (WD-01-PB-V04A): {res['label']}", "",
         f"**VERDICT: {v['verdict']}**  (A1 {_yn(v['a1'])}, A2-gating "
         f"{_yn(v['a2_gating'])}, A3 contrasts passing "
         f"{v['a3_contrasts_passing']}/2: {v['a3_by_contrast']})", "",
         PREMISE, "",
         f"Data MANIFEST sha256 `{res['data_manifest_sha256']}`; CAL "
         f"scoring rows {res['n_cal_rows']} (auxiliary sources excluded); "
         f"CAL pairs {res['n_cal_pairs']}, TRAIN pairs "
         f"{res['n_train_pairs']}. tau variants shipped: "
         f"{res['tau_variants']}.", "",
         "## A1 calibration (CAL)", ""]
    L += _a1_lines(res["a1"], "A1 per tau")
    L += ["Per-policy coverage (always reported):", ""]
    L += _table(["policy", "n", "10-90 coverage", "CRPS model",
                 "CRPS climatology"],
                [[p, d["n"], _f(d["coverage"], 3), _f(d["crps_model"]),
                  _f(d["crps_climatology"])]
                 for p, d in res["a1"]["by_policy"].items()])
    L += ["", "Strata (report strata only, never model inputs):", ""]
    for key, title in (("by_demand", "demand (veh/h)"),
                       ("by_penetration", "penetration")):
        L += _table([title, "n", "coverage", "CRPS model", "CRPS clim"],
                    [[k, d["n"], _f(d["coverage"], 3), _f(d["crps_model"]),
                      _f(d["crps_climatology"])]
                     for k, d in res["a1"][key].items()])
        L.append("")
    L += ["## A2 vs ridge (CAL; HARD, a tie is a TIE)", ""]
    L += _a2_lines(res["a2"], res["a2_table"], "A2 per tau")
    L += ["## A3 contrast skill (CAL)", ""]
    L += _a3_lines([r for r in res["a3_records"]
                    if r["contrast"] in ("damping-native", "pi-native")],
                   res["a3"], "Gating contrasts")
    L += _a3_lines([r for r in res["a3_records"]
                    if r["contrast"] not in ("damping-native",
                                             "pi-native")],
                   {}, "Reported (non-gating) contrasts: switching-native, "
                       "fixed-native")
    t = res["train_oob"]
    L += ["## TRAIN-OOB versions (selection-free of CAL; in-sample for the "
          "conformal shifts)", "", t["note"], "",
          f"TRAIN-OOB verdict: **{t['verdict']['verdict']}**", ""]
    L += _a1_lines(dict(per_tau=t["a1"]["per_tau"],
                        coverage_overall=t["a1"]["coverage_overall"],
                        verdict=t["a1"]["verdict"]), "A1 (TRAIN-OOB)")
    L += _a2_lines(t["a2"], t["a2_table"], "A2 (TRAIN-OOB)")
    L += _a3_lines(t["a3_records"], t["a3"], "A3 (TRAIN-OOB)")
    return "\n".join(L) + "\n"


def render_report(results: dict) -> str:
    L = ["# REPORT_V04A: Part B value model v0.4a on TRUE policy "
         "references (no walls)", "",
         "Registration: `output/wd01/REGISTRATION_PARTB_V04A.md` (LF "
         "sha256 `38a6fa406ca9d598fdaae986263f4fa81ee15dfa8c632414cc5f28dbe5bdee00`). "
         "Acceptance rule fixed BEFORE running; ONE CAL read; no refit "
         "afterwards.", "", PREMISE, "",
         "## Verdict table", ""]
    rows = []
    for name, res in results.items():
        v = res["verdict"]
        a3 = res["a3"]
        rows.append([res["label"], v["verdict"], _yn(v["a1"]),
                     _yn(v["a2_gating"]),
                     _yn(a3["damping-native"]["passes"]),
                     _yn(a3["pi-native"]["passes"])])
    L += _table(["variant", "VERDICT", "A1", "A2 gating (30/60/120)",
                 "A3 damping-native", "A3 pi-native"], rows)
    L += [""]
    for name, res in results.items():
        L += [f"## {res['label']}", "",
              f"VERDICT **{res['verdict']['verdict']}**", ""]
        L += _a1_lines(res["a1"], "A1")
        L += _a2_lines(res["a2"], res["a2_table"], "A2")
        L += _a3_lines(res["a3_records"], res["a3"], "A3 (all contrasts)")
        L += ["Per-policy coverage:", ""]
        L += _table(["policy", "n", "coverage"],
                    [[p, d["n"], _f(d["coverage"], 3)]
                     for p, d in res["a1"]["by_policy"].items()])
        L += ["", f"TRAIN-OOB verdict (labelled, not out-of-sample): "
                  f"{res['train_oob']['verdict']['verdict']}", ""]
    return "\n".join(L) + "\n"
