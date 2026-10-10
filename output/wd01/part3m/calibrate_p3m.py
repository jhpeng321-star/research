"""WD-01 part 3m TASK C: ensemble spread recalibration on the SAME
TRAIN-internal CALIBRATION split as part 3l (the next 4 sorted TRAIN
roots after the fit roots; validation never used), for the L6 members.

Identical mechanism A to part 3l: per (output, horizon) a factor
s_out >= 1 widening each window's member ensemble around its MEAN
(x'_i = mean + s*(x_i - mean); the ensemble MEAN is exactly invariant);
s_out is the smallest grid factor whose widened 10-90 % band covers the
truth in >= 80 % of the calibration windows.  Outputs: contract physics
delay and TTC<3 exposure at H in {30, 60} (the gate horizons).

FREEZE PROTOCOL (brief): the resulting factors are FROZEN BEFORE the
validation run -- calibration_p3m.json is written together with a
sidecar calibration_p3m.FROZEN.sha256 containing the file's sha256 and
the freeze UTC timestamp; eval_p3m.py REFUSES to run on validation
unless the sidecar matches the file.

ZERO simulation; holdout blocked in root_arrays (never passed).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("WD_DEVICE", "cpu")    # single process, shared box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "part3k"))
import decompose_p3k as D  # noqa: E402
import engine as E  # noqa: E402
import p3m_lib as M  # noqa: E402
import p3lib as P3  # noqa: E402

T_START = time.time()
RAW_DIR = HERE / "results" / "raw"
HORIZONS = (30, 60)              # the part 3m gate horizons
K_TICKS = 120                    # 60 s rollout covers H <= 60 exactly


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def below_normal_priority():
    if sys.platform == "win32":
        try:
            import ctypes
            h = ctypes.windll.kernel32.GetCurrentProcess()
            return bool(ctypes.windll.kernel32.SetPriorityClass(h, 0x4000))
        except Exception:
            return False
    return False


def member_configs():
    fit = M.L3.load_fit(HERE / "results" / "fit_p3m.json")
    members = fit["members"]
    assert len(members) == 5, len(members)
    out = []
    for mi, mem in enumerate(members):
        out.append((mi, int(mem["seed"]), float(mem["lc_scale"]),
                    dict(mem["merge_params"]),
                    float(mem.get("dawdle_scale", 1.0))))
    return out


def rollout_metrics(root, t0, seed, lc, mp, dw, dev):
    """One 60 s idm_merge rollout; per-H contract physics delay + TTC<3
    exposures with the SAME functions the gate uses."""
    r = E.Rollout(root, float(t0), kind="idm_merge", member_seed=seed,
                  lc_scale=lc, merge_params=dict(mp), dawdle_scale=dw,
                  dev=dev).run(n_ticks=K_TICKS)
    ra, k0 = r.ra, r.k0
    snaps = D.snaps_to_dicts(r, K_TICKS)
    births = D.births_from_engine(r, K_TICKS)
    rows = D.truth_rows(ra, k0, K_TICKS)
    lc_t = D.lc_ticks_from_ra(ra)
    bstates = D.birth_states_from_ra(ra, k0, K_TICKS)
    out = {}
    for H in HORIZONS:
        kh = int(round(H / 0.5))
        dp, dt_, np_, nt_ = D.phys_delay_sums(
            snaps, births, rows, lc_t, bstates, k0, kh)
        m = E.window_metrics(r.snapshots, ra, k0, H)
        out[H] = dict(delay_phys_pred=dp, delay_phys_true=dt_,
                      pred_ttc3_exposure=m["pred_ttc3_exposure"],
                      true_ttc3_exposure=m["true_ttc3_exposure"])
    return out


def main():
    import torch
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sys.stdout = D.Tee(HERE / "results" / "stdout_calibrate.txt")

    print("priority below-normal:", below_normal_priority())
    dev = P3.device()
    sp = M.L3.splits_3l()
    cal = sp["calibration"]
    assert not (set(cal) & set(sp["fit"])), "calibration leaks fit roots"
    assert not (set(cal) & set(sp["val"])), "calibration leaks validation"
    assert not (set(cal) & set(sp["holdout"])), "calibration leaks holdout"
    configs = member_configs()
    print("=== WD-01 part 3m TASK C: ensemble spread recalibration "
          "(TRAIN-internal) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never passed).")
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"calibration roots (TRAIN, held out of every fit): {cal}; "
          f"t0 {M.L3.CAL_T0S} -> {len(cal) * len(M.L3.CAL_T0S)} windows")
    print(f"fit roots (NOT rolled here): {sp['fit']}; validation "
          f"{sp['val']} NOT used here; holdout blocked")
    print("L6 member configs (from fit_p3m.json):")
    for mi, seed, lc, mp, dw in configs:
        print(f"  member {mi}: seed={seed} lc_scale={lc} merge_params="
              f"{mp} dawdle_scale={dw}")

    raw_path = RAW_DIR / "raw_calibration_p3m.jsonl"
    rows = []
    if raw_path.exists():
        rows = [json.loads(ln) for ln in
                raw_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    if rows:
        log(f"reusing {len(rows)} calibration rows already in {raw_path} "
            f"(no re-roll)")
    else:
        wins = [(r_, t0) for r_ in cal for t0 in M.L3.CAL_T0S]
        with open(raw_path, "w", encoding="utf-8") as rf:
            n_rows = 0
            for mi, seed, lc, mp, dw in configs:
                log(f"rolling L6 member {mi} (seed {seed}, dawdle {dw}) "
                    f"...")
                for root in cal:
                    ra = P3.root_arrays(root)
                    for t0 in M.L3.CAL_T0S:
                        per = rollout_metrics(root, float(t0), seed, lc,
                                              mp, dw, dev)
                        for H, stats in per.items():
                            row = dict(variant="L6", split="calibration",
                                       member=mi, root=root, t0=int(t0),
                                       H=H, **stats)
                            rf.write(json.dumps(row, default=str) + "\n")
                            n_rows += 1
                rf.flush()
        log(f"wrote {n_rows} calibration rows to {raw_path}")
        rows = [json.loads(ln) for ln in
                raw_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()]

    # ---- group per (H, root, t0): member -> row
    by = {}
    for r in rows:
        by.setdefault((r["H"], r["root"], r["t0"]), {})[r["member"]] = r
    outputs = (("delay_phys", "delay_phys_pred", "delay_phys_true"),
               ("ttc3", "pred_ttc3_exposure", "true_ttc3_exposure"))
    calib = dict(s_out={}, coverage={}, crps={}, meta=dict(
        part="3m", task="C", calibration_roots=cal,
        windows=len({(r["root"], r["t0"]) for r in rows}),
        members=len(configs),
        holdout_accessed=False, simulation=False,
        validation_used=False,
        note=("mechanism A = mean-centred widening (mean exactly "
              "invariant), the part 3l convention; s_out = smallest "
              "grid factor reaching 80 % coverage on the calibration "
              "windows; FROZEN before the validation run")))

    print("\n== spread calibration per (output, H) on the TRAIN "
          "calibration split ==")
    for name, pk, tk in outputs:
        for H in HORIZONS:
            memb = [by[(h, r_, t0)] for (h, r_, t0) in by if h == H]
            memb.sort(key=lambda w_: (w_[min(w_)]["root"],
                                      w_[min(w_)]["t0"]))
            samples = [np.array([float(w_[mi][pk]) for mi in
                                 sorted(w_)])
                       for w_ in memb]
            truths = [float(w_[min(w_)][tk]) for w_ in memb]
            raw_cov = M.L3.coverage_frac(samples, truths)
            s_out = M.L3.s_out_for(samples, truths)
            entry = dict(raw_coverage=round(raw_cov, 4), s_out=s_out,
                         target=M.L3.COVERAGE_TARGET)
            if s_out is not None:
                wid = [M.L3.widen_ensemble(x, s_out, "mean")
                       for x in samples]
                entry["coverage_mean_centred"] = round(
                    M.L3.coverage_frac(wid, truths), 4)
                entry["crps_raw_mean"] = round(float(np.mean(
                    [M.L3.crps_ensemble(x, y) for x, y in
                     zip(samples, truths)])), 3)
                entry["crps_widened_mean"] = round(float(np.mean(
                    [M.L3.crps_ensemble(x, y) for x, y in
                     zip(wid, truths)])), 3)
                entry["mean_shift_max"] = round(float(max(
                    abs(float(np.mean(w)) - float(np.mean(x)))
                    for w, x in zip(wid, samples))), 6)
            calib["s_out"].setdefault(name, {})[str(H)] = s_out
            calib["coverage"][f"{name}@H{H}"] = entry
            print(f"  {name}@H{H}: raw_cov={entry['raw_coverage']} "
                  f"s_out={s_out} "
                  f"cov_mean={entry.get('coverage_mean_centred')} "
                  f"crps {entry.get('crps_raw_mean')}->"
                  f"{entry.get('crps_widened_mean')} "
                  f"mean_shift={entry.get('mean_shift_max')}")

    # ---- FREEZE: write the calibration file, then its sha256 sidecar
    cal_text = json.dumps(calib, indent=1, default=str)
    cal_path = HERE / "results" / "calibration_p3m.json"
    cal_path.write_text(cal_text, encoding="utf-8")
    frozen = dict(sha256=sha256_text(cal_text),
                  frozen_utc=datetime.now(timezone.utc).isoformat(),
                  file=str(cal_path),
                  note="eval_p3m.py refuses validation unless this "
                       "hash matches calibration_p3m.json")
    (HERE / "results" / "calibration_p3m.FROZEN.sha256").write_text(
        json.dumps(frozen, indent=1), encoding="utf-8")
    log(f"FROZEN {cal_path}")
    log(f"  sha256={frozen['sha256']}")
    log(f"  frozen_utc={frozen['frozen_utc']}")
    print("\nDONE part 3m TASK C calibration (factors frozen before "
          "validation).")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
