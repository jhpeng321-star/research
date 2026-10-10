"""Generate REGISTRATION_P3.md from the trained weights + validation log,
then record its sha256 + UTC in results/registration_p3_record.json.
MUST run BEFORE evaluate.py (the evaluator aborts without the record)."""
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
    val = json.loads((HERE / "results" / "val_selection.json")
                     .read_text(encoding="utf-8"))
    weights = sorted(f.name for f in (HERE / "weights").glob("member*.pt"))
    w_lines = "\n".join(f"- `{n}` sha256 `{sha(HERE / 'weights' / n)}`"
                        for n in weights)
    entries = "\n".join(
        f"- member {e['member']} stage `{e['stage']}`: val speed RMSE "
        f"{e['val_speed_rmse']:.4f} m/s" for e in val["entries"])
    reg = f"""# WD-01 Part 3 - registration BEFORE the single holdout evaluation

- Written: {datetime.now(timezone.utc).isoformat()} (UTC).  This file is
  frozen and hashed; results/registration_p3_record.json carries the
  sha256 recorded BEFORE evaluate.py runs.  evaluate.py aborts if the
  file changes afterwards, and aborts if a previous holdout_eval.json
  exists (one-shot discipline).
- Zero simulation.  Data: p2C dataset, MANIFEST sha256 pinned
  (2439004d528a860c701bfb147afcda99ab3f864a29e6187be721db81cc82dd27),
  verified at evaluate.py start; surviving part-1 checkpoint sha256
  ded83907b8d218574d3f5359c332408f17167da055574afbe29eaecb59d4a8f5,
  verified at start and end of every phase.

## 1. Registered evaluation protocol (frozen)

- Holdout roots: the 5 p2C roots whose runs are holdout in the v3
  build_tracks_manifest (cv20_7007m2700? no - the actual list is loaded
  from MANIFEST.json and printed in the results; the validation roots
  cv20_7005m2000, cv20_7006m2000, cv30_7305m2700, seed_7535m2400 and all
  17 train-minus-val roots are excluded from this evaluation).
- Windows: t0 in {{120, 150, ..., 480}} s (13 per root, 65 windows);
  horizons H in {{30, 60, 90, 120}} s, all prefixes of ONE 120 s rollout
  per (window, variant) at the 0.5 s step.
- Variants: the ensemble (mean of members), each member individually,
  B1 constant-speed persistence (with births, no lane changes), B2 pure
  IDM with the 'normal' vType (no lane changes), B3 IDM oracle with each
  vehicle's true vType params from type_label (SUPERVISION used here by
  the oracle ONLY - clearly labelled), B4 the part-1 v3 surviving
  checkpoint at its native 1 s step (RAW integrator, action-blind with
  the action one-hot fixed to the native arm, births inserted).
- Full vehicle set: initial t0 vehicles plus logged births inserted at
  their logged tick/lane/s/v; exits when s passes the lane end; truth
  matching by vehicle id.

## 2. Metrics (frozen definitions)

- C2: predicted per-lane order inversions after a step (adjacent
  pre-step pairs whose order inverted or touched), per 1000 predicted
  vehicle-steps; truth base ~0 by construction.  Cap violations counted
  (v > 1.2 x lane limit or v < 0).  NO pass-through repair anywhere.
- C3: matched-vehicle speed RMSE and position RMSE (RMSE of per-window
  RMSEs over the 65 windows), fraction of matched vehicle-steps with
  predicted lane == true lane (junction truth rows mapped to the
  downstream lane), lane-change events per window predicted vs truth.
- S (window level): recall/precision of any TTC<1.5 s, any TTC<3 s, any
  hard brake (a < -3 m/s^2), with the truth BASE RATES reported;
  Pearson/Spearman and bias (pred - truth) of TTC<3 exposure,
  hard-brake seconds and min TTC (min TTC compared over windows where
  both are finite).  TTC = same-lane follower->leader, gap>0,
  closing>0, at the 0.5 s step (1 s for B4).
- D (delay proxy): sum over vehicles of time present in the window,
  pred vs truth: bias, Pearson, Spearman.
- U (ensemble spread): fraction of windows where truth TTC<3 exposure
  and delay proxy lie within the member [min, max] and within
  mean +/- 2 sd.

## 3. Registered PASS rules for part_a_horizon_s (verbatim)

A horizon H passes for the ensemble iff ALL of:

1. C2: predicted collisions <= 1 per 1000 vehicle-steps;
2. C3: model speed RMSE <= min(B1 speed RMSE, B2 speed RMSE) at H;
3. S: recall >= 0.90 for any-TTC<1.5 s AND any hard brake, with
   precision strictly above the corresponding truth base rate.

part_a_horizon_s = the LONGEST H passing all three; otherwise the first
failing criterion is reported per horizon.  These rules are frozen as
written here; any change is a reported DEVIATION.

## 4. Validation-informed selections (carve-out and log)

- Validation roots (deterministic stratified carve-out of 4 train
  roots): cv20_7005m2000, cv20_7006m2000, cv30_7305m2700,
  seed_7535m2400.  Holdout untouched before this registration.
- Selection log (results/val_selection.json):
{entries}
- Recipe after selection: {val['selection_rule']}
- Declared reductions (CPU budget): <=20000 phase-1 samples/root,
  4 phase-1 epochs, phase-2 curriculum (10,200)/(40,120)/(120,60)
  iterations, interaction slots recomputed per step without gradient,
  lanes teacher-forced during phase-2 training (lane head evaluated
  greedily), z frozen per window from the t0 truth history (newborns
  from the learned prior), residual authority reduced to +/-2 m/s^2
  after a saturation diagnostic on the pilot.

## 5. Trained weights registered for this evaluation

{w_lines}

All weight files are state_dict + plain metadata; torch.load with
weights_only=True verified.  Copies + SHA256SUMS.txt land in
BACKUPS_ROOT/wd01/part3/ (default E:/research-dense-backups/wd01/part3/;
backup.py verifies them).
"""
    (HERE / "REGISTRATION_P3.md").write_text(reg, encoding="utf-8")
    s = sha(HERE / "REGISTRATION_P3.md")
    rec = dict(file=str(HERE / "REGISTRATION_P3.md"), sha256=s,
               recorded_utc=datetime.now(timezone.utc).isoformat(),
               note="recorded BEFORE the single holdout evaluation; "
                    "evaluate.py verifies this hash and refuses to run "
                    "on mismatch")
    (HERE / "results" / "registration_p3_record.json").write_text(
        json.dumps(rec, indent=1), encoding="utf-8")
    print("REGISTRATION_P3.md written; sha256", s)


if __name__ == "__main__":
    main()
