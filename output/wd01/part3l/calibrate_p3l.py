"""WD-01 part 3l TASK C: ensemble spread calibration on a TRAIN-internal
CALIBRATION split (holdout of the fit roots; validation never used).

Split: the 4 calibration roots are the NEXT 4 sorted TRAIN roots after
the 4 fit roots -- disjoint from every fit in part 3l; validation and
holdout are never loaded here.

Members: the five L5 members from results/fit_p3l.json (member 0 =
main fit; members 1-4 = wider draws incl. dawdle width).  Each of the
4 x 7 calibration windows is rolled with all 5 members (K=240 ticks =
120 s); per horizon H in {30, 60, 90, 120} the CONTRACT PHYSICS DELAY
and the TTC<3 s exposure are computed per member with the SAME
functions the ladder evaluation uses (engine.window_metrics,
decompose.phys_delay_sums).

Mechanism A (applied at gate time as rung L5c): per (output, horizon)
a factor s_out >= 1 widening each window's member ensemble around its
ensemble MEAN -- x'_i = mean + s*(x_i - mean) -- so the ensemble MEAN
is EXACTLY invariant; s_out is the smallest grid factor whose widened
10-90 % band covers the truth in >= 80 % of the CALIBRATION windows.
The brief's median-centred variant is computed alongside as a check
(coverage + mean shift), not applied.

Mechanism B (already built into the L5 member ensemble by fit_p3l.py):
wider parameter draws; its RAW calibration coverage is reported
alongside so the two mechanisms can be compared before scaling.

ZERO simulation; holdout blocked in root_arrays (never passed).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("WD_DEVICE", "cpu")    # single process, shared box

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "part3k"))
import decompose_p3k as D  # noqa: E402
import engine as E  # noqa: E402
import p3l_lib as L  # noqa: E402
import p3lib as P3  # noqa: E402

T_START = time.time()
RAW_DIR = HERE / "results" / "raw"
HORIZONS = L.HORIZONS


def log(m):
    print(f"[{time.time()-T_START:6.1f}s] {m}", flush=True)


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
    fit = L.load_fit(HERE / "results" / "fit_p3l.json")
    members = fit["members"]
    assert len(members) == 5, len(members)
    out = []
    for mi, mem in enumerate(members):
        out.append((mi, int(mem["seed"]), float(mem["lc_scale"]),
                    dict(mem["merge_params"]),
                    float(mem.get("dawdle_scale", 1.0))))
    return out


def rollout_metrics(root, t0, seed, lc, mp, dw, dev):
    """One 120 s idm_merge rollout; per-H contract physics delay and
    window metrics (the SAME call pattern as the part 3k ladder eval)."""
    r = E.Rollout(root, float(t0), kind="idm_merge", member_seed=seed,
                  lc_scale=lc, merge_params=mp, dawdle_scale=dw,
                  dev=dev).run()
    ra, k0 = r.ra, r.k0
    snaps = D.snaps_to_dicts(r, P3.ROLL_TICKS)
    births = D.births_from_engine(r, P3.ROLL_TICKS)
    rows = D.truth_rows(ra, k0, P3.ROLL_TICKS)
    lc_t = D.lc_ticks_from_ra(ra)
    bstates = D.birth_states_from_ra(ra, k0, P3.ROLL_TICKS)
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


def cov_block(samples_by_win, truths, label, s_grid=L.S_GRID):
    """Coverage / CRPS for one output at raw and at each widening
    centre; returns raw coverage + the s_out search result."""
    raw = L.coverage_frac(samples_by_win, truths)
    s = L.s_out_for(samples_by_win, truths, grid=s_grid)
    return raw, s


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
    sp = L.splits_3l()
    cal = sp["calibration"]
    assert not (set(cal) & set(sp["fit"])), "calibration leaks fit roots"
    assert not (set(cal) & set(sp["val"])), "calibration leaks validation"
    configs = member_configs()
    print("=== WD-01 part 3l TASK C: ensemble spread calibration "
          "(TRAIN-internal) ===")
    print("ZERO simulation; holdout blocked in root_arrays (never passed).")
    print(f"device={dev} torch_threads={torch.get_num_threads()}")
    print(f"calibration roots (TRAIN, held out of the fit): {cal}; "
          f"t0 {L.CAL_T0S} -> {len(cal) * len(L.CAL_T0S)} windows")
    print(f"fit roots (NOT rolled here): {sp['fit']}; validation "
          f"{sp['val']} NOT used; holdout blocked")
    print("member configs (from fit_p3l.json):")
    for mi, seed, lc, mp, dw in configs:
        print(f"  member {mi}: seed={seed} lc_scale={lc} merge_params="
              f"{mp} dawdle_scale={dw}")

    raw_path = RAW_DIR / "raw_calibration.jsonl"
    rows = []
    if raw_path.exists():
        rows = [json.loads(ln) for ln in
                raw_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
    if not rows:
        wins = [(r_, t0) for r_ in cal for t0 in L.CAL_T0S]
        with open(raw_path, "w", encoding="utf-8") as rf:
            n_rows = 0
            for mi, seed, lc, mp, dw in configs:
                log(f"rolling L5 member {mi} (seed {seed}) ...")
                for root in cal:
                    ra = P3.root_arrays(root)
                    for t0 in L.CAL_T0S:
                        per = rollout_metrics(root, float(t0), seed, lc,
                                              mp, dw, dev)
                        for H, stats in per.items():
                            row = dict(variant="L5", split="calibration",
                                       member=mi, root=root, t0=int(t0),
                                       H=H, **stats)
                            rf.write(json.dumps(row, default=str) + "\n")
                            n_rows += 1
                rf.flush()
        log(f"wrote {n_rows} calibration rows to {raw_path}")
        rows = [json.loads(ln) for ln in
                raw_path.read_text(encoding="utf-8").splitlines()
                if ln.strip()]
        log(f"loaded {len(rows)} calibration rows back from {raw_path}")
    else:
        log(f"reusing {len(rows)} calibration rows already in {raw_path} "
            f"(no re-roll)")

    # ---- group per (output, H): window -> member values
    by = {}
    for r in rows:
        by.setdefault((r["H"], r["root"], r["t0"]), {})[r["member"]] = r
    outputs = (("delay_phys", "delay_phys_pred", "delay_phys_true"),
               ("ttc3", "pred_ttc3_exposure", "true_ttc3_exposure"))
    calib = dict(s_out={}, coverage={}, crps={}, meta=dict(
        part="3l", task="C", calibration_roots=cal,
        windows=len({(r["root"], r["t0"]) for r in rows}),
        members=len(configs),
        holdout_accessed=False, simulation=False,
        note=("mechanism A = mean-centred widening (mean exactly "
              "invariant); median-centred variant reported as a check; "
              "mechanism B = the wider L5 member draws themselves")))
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
            raw_cov, s_out = cov_block(samples, truths, name)
            entry = dict(raw_coverage=round(raw_cov, 4),
                         s_out=s_out,
                         target=L.COVERAGE_TARGET)
            if s_out is not None:
                wid_mean = [L.widen_ensemble(x, s_out, "mean")
                            for x in samples]
                wid_med = [L.widen_ensemble(x, s_out, "median")
                           for x in samples]
                entry["coverage_mean_centred"] = round(
                    L.coverage_frac(wid_mean, truths), 4)
                entry["coverage_median_centred"] = round(
                    L.coverage_frac(wid_med, truths), 4)
                entry["crps_raw_mean"] = round(float(np.mean(
                    [L.crps_ensemble(x, y) for x, y in
                     zip(samples, truths)])), 3)
                entry["crps_widened_mean"] = round(float(np.mean(
                    [L.crps_ensemble(x, y) for x, y in
                     zip(wid_mean, truths)])), 3)
                entry["mean_shift_max"] = round(float(max(
                    abs(float(np.mean(w)) - float(np.mean(x)))
                    for w, x in zip(wid_mean, samples))), 6)
                entry["mean_shift_median_centred_max"] = round(float(max(
                    abs(float(np.mean(w)) - float(np.mean(x)))
                    for w, x in zip(wid_med, samples))), 3)
            calib["s_out"].setdefault(name, {})[str(H)] = s_out
            calib["coverage"][f"{name}@H{H}"] = entry
            print(f"  {name}@H{H}: raw_cov={entry['raw_coverage']} "
                  f"s_out={s_out} "
                  f"cov_mean={entry.get('coverage_mean_centred')} "
                  f"cov_median={entry.get('coverage_median_centred')} "
                  f"crps {entry.get('crps_raw_mean')}->"
                  f"{entry.get('crps_widened_mean')} "
                  f"mean_shift={entry.get('mean_shift_max')}")

    (HERE / "results" / "calibration_p3l.json").write_text(
        json.dumps(calib, indent=1, default=str), encoding="utf-8")
    log(f"wrote {HERE / 'results' / 'calibration_p3l.json'}")
    print("\nDONE part 3l TASK C calibration.")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout


if __name__ == "__main__":
    main()
