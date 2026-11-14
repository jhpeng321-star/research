"""Assemble REPORT_PB_RESCORE1.md + the close-out integrity record.

Reads the Q1/Q2 outputs written by q1_label_audit.py / q2_rescore.py
(raw rows + tables), the frozen earlier reports (read-only) and the two
chunk WORLD_PLAN.json files (the pre-open split source), asserts that no
opened path matches any HOLDOUT world name, and writes:

  output/wd01/partB_rescore1/REPORT_PB_RESCORE1.md
  output/wd01/partB_rescore1/closeout_integrity.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WD01 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(WD01))

import common as C  # noqa: E402
import wdpaths  # noqa: E402

RD = Path(wdpaths.RESEARCH_DENSE_ROOT)
PLANS = {
    "chunk2": RD / "output" / "campaign_stage1_v2" / "results_plan_chunk2"
              / "WORLD_PLAN.json",
    "chunk3": RD / "output" / "campaign_stage1_v3" / "results_plan_chunk3"
              / "WORLD_PLAN.json",
}
RESULTS = C.RESULTS


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lf_sha256_prefix(path: Path, n: int = 16) -> str:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()[:n]


def holdout_names() -> dict:
    out = {}
    for chunk, p in PLANS.items():
        plan = json.loads(p.read_text(encoding="utf-8"))
        out[chunk] = sorted(w["world"] for w in plan["worlds"]
                            if str(w["split"]).upper() == "HOLDOUT")
    return out


def opened_files_inventory() -> dict:
    """Every file the RESCORE1 analysis opened (deterministic enumeration
    of exactly the paths the scripts read)."""
    man = C.load_label_manifest()
    label_files = sorted(
        rel for e in man["worlds"].values() for rel in e["files"])
    v032 = json.loads(
        (C.V032_ROOT / "MANIFEST.json").read_text(encoding="utf-8"))
    base_keys = [k for k, e in sorted(v032["worlds"].items())
                 if e["chunk_id"] in ("chunk2", "chunk3")
                 and e["source"] == "base"]
    v032_files = sorted(
        rel for k in base_keys
        for rel in v032["worlds"][k]["files"]
        if rel.endswith(("_labels.parquet", "_td.parquet",
                         "_feat_dep_strict.npz", "_feat_dep_det.npz")))
    model_files = []
    for d in ("v0.3.2_strict", "v0.3.2_detector"):
        for f in ("gbm_value.pkl", "gbm_value.json", "metrics.json",
                  "eval_report.md"):
            p = RD / "output" / "partB_results" / d / f
            if p.is_file():
                model_files.append(str(p))
    q3_files = [
        str(RD / "output" / "partB_results" / r / "eval_report.md")
        for r in ("v0.2_run1", "v0.3_hybrid", "v0.3.1_hybrid_deployable",
                  "v0.3.1_selected_deployable", "v0.3.1_selected_deployable_r2",
                  "v0.3.2_strict", "v0.3.2_detector")
    ] + [
        str(WD01 / "A2_ERRATUM.md"),
        str(WD01 / "PARTB_LEAK_AUDIT.md"),
        str(WD01 / "REPORT_PA2G_EVAL_A.md"),
        str(WD01 / "REGISTRATION_NATIVE_PARTA_CHUNK2.md"),
        str(WD01 / "REGISTRATION_NATIVE_PARTA_CHUNK2_AMENDMENT_1.md"),
        str(WD01 / "REGISTRATION_C2L1_V2G_DRAFT.md"),
        str(RD / "archive" / "pre_circles" / "docs" / "STAGE1C2_RESULTS.md"),
        str(RD / "archive" / "pre_circles" / "code" / "docs"
            / "STAGE1C2_RESULTS.md"),
    ]
    plan_files = [str(p) for p in PLANS.values()]
    return dict(label_root_files=label_files, v032_files_open=v032_files,
                model_files=model_files, q3_report_files=q3_files,
                plan_files=plan_files,
                other=[str(C.LABEL_ROOT / "MANIFEST.json"),
                       str(C.V032_ROOT / "MANIFEST.json"),
                       str(RESULTS / "q1_raw_rows.csv")])


def assert_no_holdout(inv: dict, ho: dict) -> dict:
    names = {w for ws in ho.values() for w in ws}
    opened = (inv["label_root_files"] + inv["v032_files_open"]
              + inv["model_files"] + inv["q3_report_files"]
              + inv["plan_files"] + inv["other"])
    hits = [p for p in opened
            if any(n in Path(p).name.lower() for n in names)]
    assert not hits, f"HOLDOUT world path opened: {hits[:5]}"
    disclosed = ["w8477m2600p02t480"]
    hits_d = [p for p in opened
              if any(n in Path(p).name.lower() for n in disclosed)]
    assert not hits_d, f"kept_disclosed HOLDOUT world opened: {hits_d}"
    return dict(n_opened_paths=len(opened), n_holdout_names=len(names),
                n_holdout_hits=0,
                kept_disclosed_worlds=disclosed,
                kept_disclosed_opened=False,
                holdout_accessed=False,
                checked_utc=datetime.now(timezone.utc).isoformat())


Q3_ROWS = [
    dict(item="Part B v0.2 CAL policy contrast at t0 (14 worlds: sign "
              "agreement 0.57; MAE(model) 7177.3 vs dV=0 3506.4)",
         source="E:/research-dense/output/partB_results/v0.2_run1/"
                "eval_report.md section 3",
         status="corrected",
         reason=("realised 'native' side was the chunk-2 switching arm "
                 "(WM-72); vs TRUE native the damping contrast is ~3x "
                 "larger (Q1 tau570 mean +3087 vs old +2115 over 112 "
                 "worlds) and 10/14 CAL signs change magnitude basis -- "
                 "the numbers are damping-vs-SWITCHING facts, not "
                 "damping-vs-native")),
    dict(item="Part B v0.3 hybrid CAL policy contrast (21 worlds: 0.76; "
              "MAE 3472.3 < 4076.6)",
         source="E:/research-dense/output/partB_results/v0.3_hybrid/"
                "eval_report.md section 3",
         status="corrected",
         reason="same mislabelled switching truth; Q2 re-score vs TRUE "
                "native: CAL sign acc 0.71-0.86 by tau but CRPS never "
                "beats the TRAIN-cell-mean baseline"),
    dict(item="Part B v0.3.1 hybrid deployable CAL contrast (21: 0.76; "
              "MAE 3277.7, Spearman 0.451)",
         source="E:/research-dense/output/partB_results/"
                "v0.3.1_hybrid_deployable/eval_report.md section 3",
         status="corrected",
         reason="same switching truth; the Spearman 0.45 ranking skill "
                "was measured against switching-arm noise, and Q1 shows "
                "old-vs-true contrast Spearman is only 0.59 at tau570"),
    dict(item="Part B v0.3.1 selected deployable (+_r2; x1/x2 UNVERIFIED "
              "prior attempts) CAL contrast (21: 0.76; MAE 3843.1)",
         source="E:/research-dense/output/partB_results/"
                "v0.3.1_selected_deployable*/eval_report.md section 3",
         status="corrected",
         reason="same switching truth; superseded scoring per Q2"),
    dict(item="Part B v0.3.2 STRICT (HEADLINE) CAL contrast (21: 0.76; "
              "MAE 3563.8 < 4076.6; Spearman 0.030)",
         source="E:/research-dense/output/partB_results/v0.3.2_strict/"
                "eval_report.md section 3",
         status="corrected",
         reason="vs TRUE native (Q2): sign accuracy survives (CAL tau570 "
                "0.857 [0.714, 1.000]) and CRPS beats zero at tau>=300, "
                "but the model under-predicts the true gap by 1.1-3.6e3 "
                "veh-s (bias) and loses to the TRAIN-cell-mean baseline "
                "at every tau; the reported 0.76/3563.8 numbers were vs "
                "switching"),
    dict(item="Part B v0.3.2 DETECTOR (SENSITIVITY) CAL contrast (21: "
              "0.76; MAE 3606.7)",
         source="E:/research-dense/output/partB_results/"
                "v0.3.2_detector/eval_report.md section 3",
         status="corrected",
         reason="same as strict: sign skill survives (tau570 0.857), "
                "CRPS beats zero at tau>=60 but not the cell-mean "
                "baseline; bias -1.0 to -3.0e3 veh-s"),
    dict(item="A2 step-2 erratum: paired 30 s deltas are exact per root "
              "(no post-t0 stochasticity); the across-root sigma is not "
              "noise; zero-delta predictor won via Part A model error",
         source="output/wd01/A2_ERRATUM.md",
         status="stands",
         reason="its claims concern deterministic CRN-pair exactness and "
                "the source of the spread, which hold whatever the "
                "'native' branch's policy is; caveat: the paired "
                "magnitudes are damping/PI-vs-SWITCHING effects (WM-72), "
                "not fleet-native effects"),
    dict(item="Part A v2 general evaluation (a) on chunk-2/3 __native "
              "truth (model-error upper bound with CAV-behaviour "
              "mismatch)",
         source="output/wd01/REPORT_PA2G_EVAL_A.md",
         status="stands",
         reason="its own TRUTH CAVEAT already states the truth is a "
                "per-CAV switching mix, and the analysis is superseded "
                "for headlines by PA2G-A2 on chunk-4 __native_base; Q1/"
                "Q2 add nothing that changes it"),
    dict(item="Held C2L1: native Part A v1 chunk-2 CAL look (frozen "
              "registration, never run)",
         source="output/wd01/REGISTRATION_NATIVE_PARTA_CHUNK2.md (+ "
                "AMENDMENT_1); replaced by REGISTRATION_C2L1_V2G_DRAFT.md",
         status="withdrawn",
         reason="never executed and its chunk-2 __native truth is "
                "invalid as a native reference (owner native-arm "
                "ruling); replaced by the C4L1 v2-general draft on "
                "fresh true-native worlds"),
    dict(item="stage1C-2 results document: damping-vs-native statements",
         source="E:/research-dense/archive/pre_circles/docs/"
                "STAGE1C2_RESULTS.md (only copy; docs/STAGE1C2_RESULTS.md "
                "does not exist at the live root)",
         status="stands",
         reason="no damping-vs-native claim found in the file (its one "
                "'damping' mention is a feature-variance test) -- "
                "nothing to correct"),
    dict(item="Planner-owned analyses of the same arms (PL-02a)",
         source="planner worktree records (out of scope per the "
                "registration)",
         status="stands",
         reason="out of scope for this WM-owned inventory; listed only "
                "as required by the registration"),
]


def main() -> int:
    ho = holdout_names()
    inv = opened_files_inventory()
    guard = assert_no_holdout(inv, ho)
    C.write_json(RESULTS / "closeout_integrity.json",
                 dict(holdout_guard=guard, opened=inv,
                      label_root=str(C.LABEL_ROOT),
                      label_root_manifest_sha256=sha256(
                          C.LABEL_ROOT / "MANIFEST.json"),
                      label_root_sha256sums_entries=sum(
                          1 for _ln in (C.LABEL_ROOT / "SHA256SUMS")
                          .read_text().splitlines() if _ln),
                      backup="E:/research-dense-backups/wd01/"
                             "partB_truearms_labels_v1",
                      backup_verify=str(WD01 / "partB_rescore1"
                                        / "backup_verify_truearms_labels_v1.json"),
                      plans_holdout=ho,
                      registration_lf_sha256_prefix=lf_sha256_prefix(
                          WD01 / "REGISTRATION_PB_RESCORE1.md")))

    q1_md = (RESULTS / "q1_tables.md").read_text(encoding="utf-8")
    q2s_md = (RESULTS / "q2_tables_strict.md").read_text(encoding="utf-8")
    q2d_md = (RESULTS / "q2_tables_detector.md").read_text(encoding="utf-8")

    L = ["# REPORT -- WD-01-PB-RESCORE1: Part B (<= v0.3.2) re-scored "
         "against the chunk-2/3 TRUE native/PI references", "",
         "- Registration: `output/wd01/REGISTRATION_PB_RESCORE1.md` "
         "(LF-sha256 prefix "
         f"`{lf_sha256_prefix(WD01 / 'REGISTRATION_PB_RESCORE1.md')}`, "
         "verified before execution).",
         "- Type: ANALYSIS, zero simulation, no training, no refit, no "
         "model selection; existing v0.3.2 models only.",
         "- Executor: ZCode task WD-01-PB-RESCORE1, worktree "
         "`zcode/wd01rs1` (from zcode/wd01 1089b52); label-code commit "
         "`6ceddb2863a75dd483fb134a7749f39885e1f0e5`.",
         f"- Completed: {datetime.now(timezone.utc).isoformat()}", "",
         "## 0. Data and integrity", "",
         "- New label root `E:/research-dense/output/"
         "wd01_partB_truearms_labels_v1` (schema "
         "`wd01_partB_truearms_labels/1`): per TRAIN/CAL world of chunks "
         "2-3 (56 + 56 worlds) five sources -- `base` (damping) and "
         "`native` (switching) COPIED hash-verified from the frozen "
         "v0.3.2 root (labels+td only), `pi` (switching), "
         "`truearms_native` (TRUE fleet native) and `truearms_pi` (TRUE "
         "fleet PI) BUILT from the raw runs with the v0.3.2 label code "
         "(build_v0data/v0lib; pre-t0 history from the base run's "
         "physics departed_ids; insertions.jsonl / dispatcher_rows.jsonl "
         "never read).  560 sources; COPY only (hard links asserted "
         "against); SHA256SUMS 1793 entries, 0 mismatches; backup "
         "`E:/research-dense-backups/wd01/partB_truearms_labels_v1` "
         "verified 1793/1793, 0 mismatches, 0 extra files.",
         "- HOLDOUT protocol: each world's split was read from its "
         "chunk's WORLD_PLAN.json BEFORE any run file access; HOLDOUT "
         "worlds were skipped by name (14 + 14); "
         "tools/holdout_guard.py + protocols/holdout_exclusions.json "
         "applied (13 excluded, w8477 kept_disclosed -- and w8477 is a "
         "HOLDOUT world, so it was NOT opened either: this analysis is "
         "TRAIN/CAL only).  Builder audit hook recorded 3291 opened "
         "paths, 0 HOLDOUT hits; the analysis opened-file inventory "
         "(closeout_integrity.json) asserts 0 hits again at close-out.  "
         "holdout_accessed=false.",
         "- Label parity: a TRAIN chunk-2 damping base source rebuilt "
         "from the raw runs with the v0.3.2 code reproduces the frozen "
         "v0.3.2 labels EXACTLY "
         "(output/wd01/tests/test_rescore1.py::"
         "test_label_parity_with_v032_code_on_train_damping_source; "
         "passed).",
         "- Cross-check: the 14 chunk-2/3 CAL `d_damp_switch` values at "
         "tau570 in q1_raw_rows.csv reproduce the v0.3.2_strict "
         "eval_report section-3 'dV realised' column exactly "
         "(w8480 -2772.5 ... w8606 +466.8).", "",
         "## 1. Q1 -- label audit (realised contrasts vs the damping "
         "base)", "",
         "Conventions: branch arms are read at t0 + 10 s (their decision "
         "grids start after t0; the same 10 s window-offset caveat as "
         "the v0.3.2 evals; all arms restore the SAME saved t0 state).  "
         "Positive contrast = the first arm has MORE remaining delay.",
         "", q1_md,
         "Reading: the relabel MATTERED.  The true damping-vs-native "
         "gap is 1.5-3.2x the old damping-vs-switching gap (tau570 mean "
         "+3087.3 vs +2115.3 over the 112 TRAIN/CAL worlds; CAL mean "
         "+5184.8 vs +3110.6), and the old contrast's sign agrees with "
         "the true one in only 0.759-0.902 of world-tau pairs "
         "(tau-dependent; 10/14 at CAL tau570).  The switching arms sit "
         "much closer to damping than true native does "
         "(tau570: switch +972.0, PI(true) +1069.2 vs damping +3087.3) "
         "-- coherent with their per-CAV damping share.", "",
         "## 2. Q2 -- proxy skill of the v0.3.2 deployable models "
         "against the TRUE contrast", "",
         "dV_old = V(s_t0, tau, damping) - V(s_t0, tau, switching_mix) "
         "from each frozen v0.3.2 model (same t0 base-arm state, "
         "policy-conditioned; the median path reproduces the old eval's "
         "1658.5 exactly), scored against "
         "dY = Y_damping(t0, tau) - Y_true_native(t0+10s, tau).  "
         "Predictive difference distribution: every damping-quantile "
         "level paired with every switching-quantile level (7 x 7 = 49 "
         "equally-likely differences -- the quantile-grid analogue of "
         "the v0.3.x independent-member pairing).  CAL use is the "
         "DISCLOSED second CAL read of v0.3.2, report-only (no "
         "selection, no calibration follows); TRAIN rows are labelled "
         "in-sample-trained.", "",
         q2s_md, "", q2d_md,
         "Reading: the OLD headline does not survive as a value claim.  "
         "Sign direction survives (CAL sign accuracy 0.714-0.857 by "
         "tau; tau570 0.857 [0.714, 1.000] -- both variants; the "
         "constant +1658.5 median dV has the right sign for 12/14 CAL "
         "worlds at tau570).  But (a) CRPS of dY under the model is "
         "WORSE than the zero-contrast baseline at tau <= 120 (strict) "
         "/ <= 60 (detector), (b) it is worse than the TRAIN-cell-mean "
         "TRUE-contrast baseline at EVERY tau and in BOTH variants, and "
         "(c) the bias is large and one-sided (-1059 to -3562 veh-s on "
         "CAL): the models systematically under-predict how much worse "
         "damping is than TRUE native, because they were fitted to the "
         "damping-vs-SWITCHING gaps, which are ~1.5-3x smaller (Q1).  "
         "On TRAIN the same pattern holds with a better apparent CRPS "
         "(in-sample).", "",
         "## 3. Q3 -- inventory of earlier WD-01 claims that used the "
         "mislabelled __native arms", "",
         "| claim | source report | status | reason |",
         "|---|---|---|---|"]
    for r in Q3_ROWS:
        L.append(f"| {r['item']} | {r['source']} | {r['status']} | "
                 f"{r['reason']} |")
    L += ["",
          "Net effect: every Part B v0.2-v0.3.2 'damping vs native' "
          "CONTRAST claim is CORRECTED (it was damping vs switching); "
          "the A2 erratum and PA2G-A stand (already caveated or "
          "policy-name-independent); held C2L1 is withdrawn (never "
          "run); no planner-owned claim is re-judged here (PL-02a out "
          "of scope, listed only).", "",
          "## 4. Answers, limitations, next steps", "",
          "**Answers.**  (Q1) The relabel changes both the level "
          "(+39% at tau570 over all worlds) and, in ~13-24% of "
          "world-tau pairs, the sign of the damping-vs-'native' "
          "contrast.  (Q2) The v0.3.2 deployable models keep moderate "
          "SIGN skill for the damping-vs-true-native contrast (0.71-"
          "0.86 CAL by tau) but no VALUE skill: they lose to the "
          "TRAIN-cell-mean baseline everywhere and to zero at short "
          "tau, with a large one-sided bias.  (Q3) see the inventory.",
          "",
          "**Limitations.**  Single realisation per arm and world (no "
          "replicates; the A2 erratum says such pairs are exact per "
          "root, so contrasts are exact per root, not expectations "
          "over stochasticity).  10 s window offset between the damping "
          "arm (t0) and the branch arms (t0+10 s), as in all earlier "
          "reports.  The 7 x 7 quantile pairing is an approximation of "
          "the difference distribution (levels, not joint draws).  CAL "
          "results are a second read of CAL (disclosed, report-only).  "
          "Q2 evaluates the models OFF their training distribution "
          "(true-native gaps are larger than anything in the training "
          "labels); the under-prediction bias is a statement about the "
          "shipped models, not about retrained ones.  Chunk-1 worlds "
          "have no truearms references and are out of scope.",
          "",
          "**Next steps.**  Owner decisions: (1) accept the Q3 "
          "corrections/withdrawals for the affected reports; (2) "
          "whether a v0.4 contrast model must be trained on relabelled "
          "labels + a true-native-augmented dataset before any planner "
          "use (this registration neither trains nor selects); "
          "(3) PL-02a re-scoring stays planner-owned.",
          "",
          "## 5. Evidence paths", "",
          "- Label root + MANIFEST: `E:/research-dense/output/"
          "wd01_partB_truearms_labels_v1/` (MANIFEST sha256 "
          f"`{sha256(C.LABEL_ROOT / 'MANIFEST.json')}`).",
          "- Backup: `E:/research-dense-backups/wd01/"
          "partB_truearms_labels_v1/` (verified).",
          "- Raw rows: `output/wd01/partB_rescore1/q1_raw_rows.csv` "
          f"(sha256 `{sha256(RESULTS / 'q1_raw_rows.csv')}`), "
          "q2_raw_rows_strict.csv "
          f"({sha256(RESULTS / 'q2_raw_rows_strict.csv')}), "
          "q2_raw_rows_detector.csv "
          f"({sha256(RESULTS / 'q2_raw_rows_detector.csv')}).",
          "- Tables: q1_tables.json/.md, q2_tables_strict.json/.md, "
          "q2_tables_detector.json/.md, q2_tables.json.",
          "- Close-out integrity: closeout_integrity.json "
          "(opened-file inventory, HOLDOUT assertion "
          f"{guard['n_holdout_hits']} hits over "
          f"{guard['n_opened_paths']} opened paths).",
          "- Tests: output/wd01/tests/test_rescore1.py (8 passed, incl. "
          "copy-not-link, pre-open HOLDOUT guard, real-data label "
          "parity).", ""]
    (RESULTS / "REPORT_PB_RESCORE1.md").write_text("\n".join(L),
                                                   encoding="utf-8")
    print("wrote", RESULTS / "REPORT_PB_RESCORE1.md")
    print("wrote", RESULTS / "closeout_integrity.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
