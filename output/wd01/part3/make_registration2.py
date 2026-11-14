"""Generate REGISTRATION_P3_v2.md (supersedes v1) from the run2 weights,
validation gate results, and val_selection.json; record sha256 + UTC.
MUST run BEFORE evaluate.py (which verifies the v2 record)."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    gate = json.loads((HERE / "results" / "val_gate_run2.json").read_text(
        encoding="utf-8"))["gate"]
    val = json.loads((HERE / "results" / "val_selection.json").read_text(
        encoding="utf-8"))
    weights = sorted(f.name for f in
                     (HERE / "weights" / "run2").glob("member*.pt"))
    w_lines = "\n".join(
        f"- `weights/run2/{n}` sha256 `{sha(HERE / 'weights' / 'run2' / n)}`"
        for n in weights)
    entries = "\n".join(
        f"- member {e['member']} stage `{e['stage']}`: val speed RMSE "
        f"{e['val_speed_rmse']:.4f} m/s" for e in val["entries"])

    reg = f"""# WD-01 Part 3 - REGISTRATION v2 (supersedes REGISTRATION_P3.md; frozen BEFORE the single holdout evaluation)

- Written: {datetime.now(timezone.utc).isoformat()} (UTC).  sha256 of this
  file is recorded in results/registration_p3_v2_record.json BEFORE
  evaluate.py runs; evaluate.py (v2) verifies it and aborts on mismatch
  or if a previous holdout_eval.json exists (one-shot discipline).
- v1 (sha256 3c83b2dbe280e8f7e02275eeb11764ec821c41ab448d75d09e51b70b0efc9b01)
  is superseded WITHOUT any holdout use (the holdout was never scored
  under v1 - the run was stopped for broken validation rollouts).
- ZERO SIMULATION.  Pinned inputs (verified at evaluation start and end):
  part-1 surviving checkpoint
  ded83907b8d218574d3f5359c332408f17167da055574afbe29eaecb59d4a8f5;
  p2C MANIFEST.json
  2439004d528a860c701bfb147afcda99ab3f864a29e6187be721db81cc82dd27.

## 1. Why v2 (justified from validation + the p2C erratum only)

1. The p2C data card's TTC rates were wrong (front/behind inverted; see
   partA_data/DATA_CARD.md ERRATUM): true follower->leader TTC<3 s
   exposure is 2.31 per 1000 vehicle-steps and TTC<1.5 s is ~0 at the
   0.5 s sampling.  v1's S rule 'recall >= 0.90 for any TTC<1.5 s' is
   therefore degenerate (a handful of windows).  v2 S rule: window-level
   any TTC<3 s AND any hard brake (a < -3 m/s^2); TTC<1.5 s metrics are
   still computed and reported, just not gated.
2. Per-H truth base rates on VALIDATION (reported first, as required):
   see the val gate numbers in section 4 and
   results/val_gate_run2.json.
3. No other registered rule changed.  C2, C3, D, U, the protocol, and
   the recommendation rule are identical to v1.

## 2. Protocol (unchanged from v1)

- Holdout: the 5 p2C holdout roots; windows t0 in {{120, 150, ..., 480}} s
  (13 per root, 65 windows); horizons 30/60/90/120 s as prefixes of ONE
  120 s rollout per (window, variant) at the 0.5 s step.
- Variants: ensemble mean, each member, B1 persistence (with births, no
  lane changes), B2 IDM 'normal' (no lane changes), B3 IDM oracle with
  true vType params (SUPERVISION used by the oracle ONLY, labelled),
  B4 part-1 v3 checkpoint at 1 s (RAW integrator, action-blind).
- Full vehicle set: t0 vehicles + logged births inserted at logged
  (tick, lane, s, v); exits at lane end; truth matched by vehicle id.
- Metrics C2/C3/S/D/U as defined in REGISTRATION_P3.md section 2 with
  the S substitution above.

## 3. Registered PASS rules (v2, verbatim)

A horizon H passes for the ensemble iff ALL of:

1. C2: predicted collisions <= 1 per 1000 vehicle-steps;
2. C3: model speed RMSE <= min(B1 speed RMSE, B2 speed RMSE) at H;
3. S: window-level recall >= 0.90 for any TTC<3 s AND any hard brake
   (a < -3 m/s^2), with precision strictly above the corresponding
   truth base rate.

part_a_horizon_s = the LONGEST H passing all three; otherwise the first
failing criterion is reported.  Frozen as written; changes are reported
DEVIATIONS.

## 4. Validation sanity gate (measured, BEFORE this registration)

results/val_gate_run2.json (28 validation windows, H=60 s):
- ensemble speed RMSE {gate['ens_speed_rmse_h60']} m/s vs B1
  {gate['B1_speed_rmse_h60']} and B2 {gate['B2_speed_rmse_h60']}
  -> beats both: {gate['beat_B1'] and gate['beat_B2']}
- ensemble predicted collisions {gate['ens_collisions_per_1000']} per
  1000 vehicle-steps (<= 1 required): {gate['collisions_ok']}
- GATE PASS: {gate['PASS']}
- truth base rates on validation windows: any TTC<3 s
  {gate['ttc3_true_rate']}, any hard brake {gate['hb_true_rate']};
  ensemble recall TTC<3 s {gate['ttc3_recall']}, hard brake
  {gate['hb_recall']}.

Validation selection log (results/val_selection.json):
{entries}

## 5. Trained weights registered for this evaluation (run2)

{w_lines}

All weight files are state_dict + plain metadata (weights_only=True
verified) and are copied to BACKUPS_ROOT/wd01/part3/ (default
E:/research-dense-backups/wd01/part3/) with
SHA256SUMS.txt by backup.py, which verifies the copies.  The stopped
run's weights are preserved in weights/run1_stopped/ with recorded
sha256 (results/run1_weights_sha256*.txt).
"""
    (HERE / "REGISTRATION_P3_v2.md").write_text(reg, encoding="utf-8")
    s = sha(HERE / "REGISTRATION_P3_v2.md")
    rec = dict(file=str(HERE / "REGISTRATION_P3_v2.md"), sha256=s,
               recorded_utc=datetime.now(timezone.utc).isoformat(),
               supersedes="REGISTRATION_P3.md "
                          "(3c83b2dbe280e8f7e02275eeb11764ec821c41ab448d75d"
                          "09e51b70b0efc9b01)",
               note="recorded BEFORE the single holdout evaluation")
    (HERE / "results" / "registration_p3_v2_record.json").write_text(
        json.dumps(rec, indent=1), encoding="utf-8")
    print("REGISTRATION_P3_v2.md written; sha256", s)


if __name__ == "__main__":
    main()
