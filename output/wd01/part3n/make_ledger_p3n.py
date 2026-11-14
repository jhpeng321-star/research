"""WD-01 part 3n: assemble LEDGER_DRAFT_P3N.json (the part's own DRAFT
record; the SHARED ledger output/wd01/LEDGER_DRAFT_WD01_DEVIATIONS.json
is NOT edited by part 3n -- the validation-look count here is a draft
for the owner to merge)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"


def load(path):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() \
        else None


def main():
    diag = load(RES / "diag_p3n.json")
    fit = load(RES / "fit_p3n.json")
    gate_iters = {}
    for k in range(0, 5):
        g = load(RES / f"gate_train_internal_iter{k}.json")
        if g is not None:
            gate_iters[f"iter{k}"] = dict(
                iteration=k,
                pass_=g["gate_v2"]["H30"]["v2_PASS"],
                failing=g["gate_v2"]["H30"]["v2_failing"],
                lc_pred=g["gate_v2"]["H30"]["lc_pred"],
                lc_true=g["gate_v2"]["H30"]["lc_true"],
                lc_dev=g["gate_v2"]["H30"]["lc_deviation"],
                ttc3_cov=g["gate_v2"]["H30"]["ttc3_cov_frac"],
                delay_phys_cov=g["gate_v2"]["H30"][
                    "delay_phys_cov_frac"],
                crps=g["gate_v2"]["H30"]["crps_phys"],
                speed=g["gate_v2"]["H30"]["speed_rmse"],
                json=f"gate_train_internal_iter{k}.json")
    frozen_cal = load(RES / "calibration_p3n.json.FROZEN.sha256")
    frozen_fit = load(RES / "fit_p3n.json.FROZEN.sha256")
    calib = load(RES / "calibration_p3n.json")
    val = load(RES / "gate_v2_p3n.json")

    looks = []
    if val is not None:
        g30 = val["gate_v2"]["L6c@H30"]
        g60 = val["gate_v2"]["L6c@H60"]
        o30 = val["old_gate"]["L6c@H30"]
        o60 = val["old_gate"]["L6c@H60"]
        looks.append(dict(
            task="p3n (this part)",
            look_number="8 (7 previous looks in WD-01-DEV-2)",
            gate="gate v2 + old",
            verdict=("PASS" if g30["v2_PASS"] else
                     "FAIL: " + ",".join(g30["v2_failing"])),
            detail=dict(
                H30=dict(failing=g30["v2_failing"],
                         lc_dev=g30["lc_deviation"],
                         ttc3_cov=g30["ttc3_cov_frac"],
                         delay_phys_cov=g30["delay_phys_cov_frac"],
                         crps=g30["crps_phys"],
                         old_pass=o30["old_pass"]),
                H60=dict(failing=g60["v2_failing"],
                         lc_dev=g60["lc_deviation"],
                         ttc3_cov=g60["ttc3_cov_frac"],
                         delay_phys_cov=g60["delay_phys_cov_frac"],
                         crps=g60["crps_phys"],
                         old_pass=o60["old_pass"]),
                sharpness_climatology=val.get(
                    "sharpness_climatology"))))

    ledger = dict(
        part="3n",
        date="2026-09-27",
        title="gate-v2 (b) closure attempt for the part 3m model: "
              "LC-by-type diagnosis, joint lc_scale x end_force_s refit "
              "with the LC-BY-TYPE loss, TRAIN-internal 2-fold gate "
              "iteration, freeze, ONE validation look",
        holdout_accessed=False,
        holdout_note="no holdout root loaded, no registration hash "
                     "created or passed; holdout_block authoritative",
        simulation=False,
        engine_change="NONE (part3/engine.py untouched in part 3n; "
                      "end_force_s is already a merge_params knob)",
        data_root="runs used WD01_DATA_ROOT=E:/rd-wt/wd01/output/wd01 "
                  "read-only; never written",
        split=dict(
            fit_roots=["cv17", "cv19_4107", "cv19_4108",
                       "cv20_7001m2000"],
            gate_fold_A=list(["cv20_7002m2000", "cv20_7003m2000"]),
            gate_fold_B=list(["cv20_7004m2000", "cv20_7007m2700"]),
            gate_windows=28,
            protocol="2-fold: scored fold's spread factors fitted on "
                     "the other fold only; frozen factors for the "
                     "single look fitted on the FULL calibration split"),
        diagnosis=diag,
        fit=dict(
            selected=fit["selected"] if fit else None,
            members=fit["members"] if fit else None,
            grids=fit["grids"] if fit else None,
            loss=fit["loss"] if fit else None) if fit else None,
        train_internal_gate_iterations=gate_iters,
        freeze=dict(
            calibration=calib,
            frozen_calibration_utc=frozen_cal["frozen_utc"] if
            frozen_cal else None,
            sha256_calibration=frozen_cal["sha256"][:16] + "..." if
            frozen_cal else None,
            frozen_fit_utc=frozen_fit["frozen_utc"] if frozen_fit
            else None,
            order="TRAIN-internal gate pass -> freeze -> the single "
                  "validation look; eval_p3n.py refuses without "
                  "matching sidecars") if calib else None,
        validation_look=looks[0] if looks else dict(
            taken=False,
            note="no validation look taken in part 3n (TRAIN-internal "
                 "gate did not pass within the iteration budget)"),
        validation_looks_used=len(looks),
        shared_ledger_note="WD-01-DEV-2 in the SHARED "
                           "LEDGER_DRAFT_WD01_DEVIATIONS.json was NOT "
                           "edited; this draft records look #8 for the "
                           "owner to merge",
        not_done=[
            "holdout evaluation (blocked until the owner opens it)",
            "second validation look (forbidden in this task regardless "
            "of the look-8 verdict)",
            "sampled-vType posterior re-fit (p3d's; untouched)",
            "idm_sampled stale-llen defect (p3k item) recorded, not "
            "fixed"],
    )
    out = HERE / "LEDGER_DRAFT_P3N.json"
    out.write_text(json.dumps(ledger, indent=1, default=str),
                   encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
