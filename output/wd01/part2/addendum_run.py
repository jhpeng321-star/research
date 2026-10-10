"""WD-01 part 2A addendum run: corrected RAW overlap, re-scored RAW C2, and
contract-level window safety diagnostics (REGISTRATION_ADDENDUM.md).

Zero simulation: deterministic re-run of the part-1 T2 rollouts on the same
states caches, same surviving checkpoint, same integrators and event rules
(all imported from part-1 / source, never edited).  21 holdout runs x 7
start ticks x {RAW, CONSTRAINED}, 120-step rollouts, windows H in
{30, 60, 90, 120} as prefixes of the same rollout.

Corrected RAW overlap (addendum section 2): pairs adjacent in the d-order at
t within a lane group whose order is inverted at t+1 (d_follower <=
d_leader); the follower is flagged; rate per 1000 vehicle-steps.  Corrected
RAW C2: zero vehicle-steps above 1.2 x lane-group train max AND corrected
overlap rate <= 1/1000.  CONSTRAINED C2 is unchanged (integrator flags).

Window diagnostics (addendum section 3, DIAGNOSTICS ONLY): per
(run x t0 x H x variant), predicted vs truth over the frozen t0 vehicle set;
truth leaders from the full truth snapshot (part-1 truth-event convention),
predicted leaders from rollout states.  No pass/fail.

Outputs (results/):
  overlap_addendum.json / .csv   corrected overlap rates + C2 verdicts per H
  deviation_check.json           re-run vs part-1 stored aggregates
  diag_summary.json              pooled window diagnostics per H x variant
  diag_per_eval.csv              per (run, t0, variant, H) truth/pred row
  addendum_run_meta.json         commands, wall clock, sha256 receipts
"""
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import wd2lib as W  # noqa: E402

T_START = time.time()
H_MAX = 120
VARIANTS = ("RAW", "CONSTRAINED")
EXPECT_ADDENDUM_SHA256 = ("52e95ac3ba391a016155344d7deab43165830856c6e213b4f"
                          "21ca0edb680259a")


def log(msg):
    print(f"[{time.time() - T_START:7.1f}s] {msg}", flush=True)


def main():
    # ---------------- registration + pinned-checkpoint guards -------------
    add_path = HERE / "REGISTRATION_ADDENDUM.md"
    add_sha = W.sha256_file(add_path)
    assert add_sha == EXPECT_ADDENDUM_SHA256, \
        f"REGISTRATION_ADDENDUM.md sha256 {add_sha} != pinned expectation"
    rec = W.load_json(HERE / "results" / "registration_addendum_record.json")
    assert rec["sha256"] == add_sha, \
        "registration record disagrees with the addendum file on disk"
    print(f"addendum guard ok ({add_sha[:16]}..., "
          f"registered {rec['recorded_utc']})")

    L1 = W.part1_lib()
    T2 = W.part1_t2()
    ckpt_sha = W.sha256_file(W.SURV_CKPT)
    assert ckpt_sha == W.SURV_CKPT_SHA256, \
        f"surviving checkpoint sha256 {ckpt_sha} != pinned"
    print(f"checkpoint guard ok ({ckpt_sha[:16]}...)")

    dev = W.device()
    w = W.part1_wm()
    tmod = L1.train_mod()
    ck = torch.load(L1.SURV_CKPT, map_location="cpu", weights_only=False)
    model = tmod.DynamicsMLP(in_dim=ck["arch"]["in_dim"],
                             hidden=tuple(ck["arch"]["hidden"]))
    model.load_state_dict(ck["model_state"])
    model = model.to(dev).eval()
    x_mean = np.asarray(ck["x_mean"], dtype=np.float32)
    x_std = np.asarray(ck["x_std"], dtype=np.float32)

    cap_base = T2.train_group_caps()
    vcap = [1.2 * cap_base[g] for g in range(4)]
    p1_summary = W.load_json(L1.RESULTS / "t2_summary.json")
    assert {str(g): vcap[g] for g in range(4)} == p1_summary["v_cap"], \
        "v_cap differs from part-1 scoring"
    log(f"v_cap ok {[round(c, 2) for c in vcap]}; device {dev}")

    tmap = W.load_json(L1.RESULTS / "tag_run_map.json")
    holdout = [r for r in tmap["rows"] if r["split"] == "holdout"]
    assert len(holdout) == 21

    fwd = [0.0, 0]                    # model-only seconds, vehicle-steps
    truth_stats = {}                  # (tag, t0) -> per-step truth counts
    pred_stats = {}                   # (tag, t0, variant) -> rollout counts

    for r in holdout:
        tag, aid = r["tag"], r["action_idx"]
        states = L1.cache_to_states(L1.RESULTS / "states_cache" / "holdout" /
                                    f"{tag}.npz")
        for t0 in W.START_TICKS:
            snap = states[t0]
            ids = sorted(snap)
            frozen = set(ids)
            d0 = np.array([snap[i][0] for i in ids], dtype=np.float64)
            v0 = np.array([snap[i][1] for i in ids], dtype=np.float64)
            g0 = np.array([snap[i][2] for i in ids], dtype=np.int64)
            n0 = len(ids)

            # ---- truth side.  full_* counts follow the part-1 truth-event
            # convention (every vehicle of the FULL truth snapshot); frz_*
            # counts restrict the scored vehicles to the frozen t0 set while
            # leaders still come from the full snapshot (addendum section 3).
            full_ttc3 = np.zeros(H_MAX + 1, dtype=np.int64)
            full_brake = np.zeros(H_MAX + 1, dtype=np.int64)
            frz_ttc3 = np.zeros(H_MAX + 1, dtype=np.int64)
            frz_ttc15 = np.zeros(H_MAX + 1, dtype=np.int64)
            frz_brake = np.zeros(H_MAX + 1, dtype=np.int64)
            frz_minttc = np.full(H_MAX + 1, np.nan)
            n_cmp = np.zeros(H_MAX + 1, dtype=np.int64)
            for h in range(1, H_MAX + 1):
                sn = states.get(t0 + h)
                if not sn:
                    continue
                pv = states.get(t0 + h - 1)
                ids_h = sorted(sn)
                d_h = np.array([sn[i][0] for i in ids_h], dtype=np.float64)
                v_h = np.array([sn[i][1] for i in ids_h], dtype=np.float64)
                g_h = np.array([sn[i][2] for i in ids_h], dtype=np.int64)
                lt15, lt3, _, ttc = W.closing_ttc(d_h, v_h, g_h, w)
                full_ttc3[h] = len(lt3)
                frz_idx = [i for i, vid in enumerate(ids_h) if vid in frozen]
                frz_here = {ids_h[i] for i in frz_idx}
                frz_ttc3[h] = len(frz_here & {ids_h[i] for i in lt3})
                frz_ttc15[h] = len(frz_here & {ids_h[i] for i in lt15})
                vals = ttc[frz_idx]
                if np.isfinite(vals).any():
                    frz_minttc[h] = float(np.nanmin(vals))
                if pv:
                    full_brake[h] = sum(
                        1 for vid in ids_h
                        if vid in pv and sn[vid][1] - pv[vid][1] < -3.0)
                    frz_brake[h] = sum(
                        1 for vid in frozen
                        if vid in sn and vid in pv
                        and sn[vid][1] - pv[vid][1] < -3.0)
                n_cmp[h] = sum(1 for i, vid in enumerate(ids)
                               if vid in sn and sn[vid][2] == g0[i])
            truth_stats[(tag, t0)] = dict(
                n0=n0, full_ttc3=full_ttc3, full_brake=full_brake,
                frz_ttc3=frz_ttc3, frz_ttc15=frz_ttc15, frz_brake=frz_brake,
                frz_minttc=frz_minttc, n_cmp=n_cmp)

            # ---- one autoregressive rollout per variant (part-1 Roll)
            for variant in VARIANTS:
                roll = T2.Roll(d0, v0, g0)
                for _ in range(H_MAX):
                    roll.model_step(model, x_mean, x_std, w, aid, dev,
                                    variant, vcap, fwd)
                vcap_arr = np.asarray(vcap, dtype=np.float64)[g0]
                p = dict(ttc3=np.zeros(H_MAX + 1, dtype=np.int64),
                         ttc15=np.zeros(H_MAX + 1, dtype=np.int64),
                         brake=np.zeros(H_MAX + 1, dtype=np.int64),
                         ovl=np.zeros(H_MAX + 1, dtype=np.int64),
                         ovl_def=np.zeros(H_MAX + 1, dtype=np.int64),
                         capviol=np.zeros(H_MAX + 1, dtype=np.int64),
                         neg=np.zeros(H_MAX + 1, dtype=np.int64),
                         minttc=np.full(H_MAX + 1, np.nan),
                         vmax=np.zeros(H_MAX + 1))
                for h in range(1, H_MAX + 1):
                    d, v = roll.d[h], roll.v[h]
                    lt15, lt3, m, _ = W.closing_ttc(d, v, g0, w)
                    p["ttc3"][h] = len(lt3)
                    p["ttc15"][h] = len(lt15)
                    p["minttc"][h] = m
                    p["brake"][h] = int((v - roll.v[h - 1] < -3.0).sum())
                    p["capviol"][h] = int((v > vcap_arr).sum())
                    p["neg"][h] = int((v < 0).sum())
                    p["vmax"][h] = float(v.max())
                    if variant == "RAW":
                        # corrected counter: order at t (roll.d[h-1]) vs t+1
                        p["ovl"][h] = int(W.corrected_passthrough(
                            roll.d[h - 1], roll.d[h], g0).sum())
                        # defective part-1 counter, reproduced to show the
                        # stored zero is an artefact (always false by
                        # construction)
                        p["ovl_def"][h] = int(T2.raw_passthrough(
                            d, g0).sum())
                    else:
                        p["ovl"][h] = int(roll.ov[h].sum())
                pred_stats[(tag, t0, variant)] = p
        log(f"{tag} done")

    # =================== aggregate per (variant, H) =======================
    evals = sorted({k[:2] for k in truth_stats})
    assert len(evals) == 147
    horizons = W.HORIZONS

    overlap = {}
    deviation = {}
    diag_rows = []
    diag_summary = {}
    for variant in VARIANTS:
        for H in horizons:
            n_steps = sum(truth_stats[e]["n0"] for e in evals) * H
            n_cmp_steps = sum(int(truth_stats[e]["n_cmp"][1:H + 1].sum())
                              for e in evals)
            ovl = sum(int(pred_stats[(e[0], e[1], variant)]["ovl"]
                          [1:H + 1].sum()) for e in evals)
            ovl_def = sum(int(pred_stats[(e[0], e[1], variant)]["ovl_def"]
                              [1:H + 1].sum()) for e in evals)
            capviol = sum(int(pred_stats[(e[0], e[1], variant)]["capviol"]
                              [1:H + 1].sum()) for e in evals)
            neg = sum(int(pred_stats[(e[0], e[1], variant)]["neg"]
                          [1:H + 1].sum()) for e in evals)
            vmax = max(float(pred_stats[(e[0], e[1], variant)]["vmax"]
                             [1:H + 1].max()) for e in evals)
            rate = ovl / max(n_steps, 1) * 1000.0
            p1v = p1_summary["verdicts"][f"{variant}@H{H}"]
            overlap[f"{variant}@H{H}"] = dict(
                n_evals=len(evals), vehicle_steps=n_steps,
                overlap_events=ovl, overlap_per_1000=round(rate, 6),
                overlap_events_defective_counter=ovl_def,
                capviol_steps=capviol, negspeed_steps=neg,
                max_speed_mps=vmax,
                part1_overlap_events=0 if variant == "RAW"
                else p1v["overlap_events"],
                part1_overlap_per_1000=0.0 if variant == "RAW"
                else p1v["overlap_per_1000"],
                C2_corrected_pass=bool(capviol == 0 and rate <= 1.0)
                if variant == "RAW" else bool(p1v["C2_runaway_pass"]))

            # ---- deviation check vs part-1 stored aggregates
            ttc_t = sum(int(truth_stats[e]["full_ttc3"][1:H + 1].sum())
                        for e in evals)
            brk_t = sum(int(truth_stats[e]["full_brake"][1:H + 1].sum())
                        for e in evals)
            ttc_p = sum(int(pred_stats[(e[0], e[1], variant)]["ttc3"]
                            [1:H + 1].sum()) for e in evals)
            brk_p = sum(int(pred_stats[(e[0], e[1], variant)]["brake"]
                            [1:H + 1].sum()) for e in evals)
            got = dict(vehicle_steps=n_steps,
                       comparable_vehicle_steps=n_cmp_steps,
                       capviol_steps=capviol, negspeed_steps=neg,
                       max_speed_mps=vmax, ttc_truth=ttc_t, ttc_pred=ttc_p,
                       brake_truth=brk_t, brake_pred=brk_p)
            exp = dict(vehicle_steps=p1v["vehicle_steps"],
                       comparable_vehicle_steps=p1v["comparable_vehicle_steps"],
                       capviol_steps=p1v["capviol_steps"],
                       negspeed_steps=p1v["negspeed_steps"],
                       max_speed_mps=p1v["max_speed_mps"],
                       ttc_truth=p1v["ttc_truth"], ttc_pred=p1v["ttc_pred"],
                       brake_truth=p1v["brake_truth"],
                       brake_pred=p1v["brake_pred"])
            diffs = {}
            for k in got:
                if isinstance(got[k], float):
                    if abs(got[k] - exp[k]) > 1e-9:
                        diffs[k] = dict(recomputed=got[k], part1=exp[k])
                elif got[k] != exp[k]:
                    diffs[k] = dict(recomputed=got[k], part1=exp[k])
            # The registered exception: part-1's RAW overlap column is the
            # defective counter's output.  Reproduce it (must be 0), then
            # supersede it with the corrected count (reported in overlap).
            if variant == "RAW":
                if ovl_def != p1v["overlap_events"]:
                    diffs["overlap_events_defective_recomputed"] = dict(
                        recomputed=ovl_def, part1=p1v["overlap_events"])
            elif ovl != p1v["overlap_events"]:
                diffs["overlap_events"] = dict(recomputed=ovl,
                                               part1=p1v["overlap_events"])
            # Second measured part-1 property (C1 scope, not re-registered
            # here): part-1 pooled its event universes into dicts keyed by
            # vehicle id (t2_run.py: pt/tt_/pb/tb_ comprehensions), so its
            # stored event totals count DISTINCT VEHICLES per window, not
            # vehicle-seconds.  Our recomputed per-step totals are larger by
            # the per-vehicle event persistence; classify as explained.
            explained = {k: diffs.pop(k) for k in
                         ("ttc_truth", "ttc_pred", "brake_truth", "brake_pred")
                         if k in diffs}
            deviation[f"{variant}@H{H}"] = dict(
                matches=not diffs, unexpected_diffs=diffs or None,
                explained_diffs=explained or None,
                explanation=("ttc/brake pools: part-1 dict-collapse counts "
                             "distinct vehicles, recomputed counts are "
                             "vehicle-seconds (addendum REPORT section 4.2)"
                             if explained else None))

            # ---- window diagnostics rows for this (variant, H)
            for (tg, tt) in evals:
                tr = truth_stats[(tg, tt)]
                pr = pred_stats[(tg, tt, variant)]
                sl = slice(1, H + 1)
                t_m = tr["frz_minttc"][sl]
                p_m = pr["minttc"][sl]
                row = dict(
                    run=tg, t0=tt, variant=variant, H=H, n0=tr["n0"],
                    truth_ttc_lt3_exposure_s=float(tr["frz_ttc3"][sl].sum()),
                    pred_ttc_lt3_exposure_s=float(pr["ttc3"][sl].sum()),
                    truth_ttc_lt1p5_exposure_s=float(
                        tr["frz_ttc15"][sl].sum()),
                    pred_ttc_lt1p5_exposure_s=float(pr["ttc15"][sl].sum()),
                    truth_hard_brake_s=float(tr["frz_brake"][sl].sum()),
                    pred_hard_brake_s=float(pr["brake"][sl].sum()),
                    truth_min_ttc_s=(None if not np.isfinite(t_m).any()
                                     else float(np.nanmin(t_m))),
                    pred_min_ttc_s=(None if not np.isfinite(p_m).any()
                                    else float(np.nanmin(p_m))))
                row["truth_any_ttc_lt1p5"] = \
                    row["truth_ttc_lt1p5_exposure_s"] > 0
                row["pred_any_ttc_lt1p5"] = \
                    row["pred_ttc_lt1p5_exposure_s"] > 0
                row["truth_any_ttc_lt3"] = \
                    row["truth_ttc_lt3_exposure_s"] > 0
                row["pred_any_ttc_lt3"] = \
                    row["pred_ttc_lt3_exposure_s"] > 0
                row["truth_any_hard_brake"] = row["truth_hard_brake_s"] > 0
                row["pred_any_hard_brake"] = row["pred_hard_brake_s"] > 0
                diag_rows.append(row)

    for variant in VARIANTS:
        for H in horizons:
            rows = [r for r in diag_rows
                    if r["variant"] == variant and r["H"] == H]
            diag_summary[f"{variant}@H{H}"] = W.detection_table(rows)

    # =================== write results ====================================
    out = HERE / "results"
    out.mkdir(exist_ok=True)

    W.dump_json(out / "overlap_addendum.json", dict(
        experiment="WD-01 part 2A addendum (corrected RAW overlap + C2)",
        registration_addendum_sha256=add_sha,
        checkpoint=W.SURV_CKPT, checkpoint_sha256=W.SURV_CKPT_SHA256,
        device=str(dev), horizons=list(horizons),
        n_evals_per_H=len(evals),
        corrected_overlap_definition=(
            "pairs adjacent in the d-order at t within a lane group whose "
            "order is inverted at t+1 (d_follower <= d_leader); follower "
            "flagged; rate per 1000 vehicle-steps"),
        corrected_raw_C2=(
            "zero vehicle-steps above 1.2 x lane-group train max AND "
            "corrected overlap rate <= 1 per 1000 vehicle-steps"),
        constrained_C2="unchanged from part-1 REGISTRATION.md section 6",
        verdicts=overlap))

    with (out / "overlap_addendum.csv").open("w", newline="",
                                             encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["variant", "H", "part1_overlap_events",
                     "part1_overlap_per_1000", "overlap_events_corrected",
                     "overlap_per_1000_corrected", "vehicle_steps",
                     "capviol_steps", "max_speed_mps", "C2_corrected_pass"])
        for variant in VARIANTS:
            for H in horizons:
                v = overlap[f"{variant}@H{H}"]
                wr.writerow([variant, H, v["part1_overlap_events"],
                             v["part1_overlap_per_1000"],
                             v["overlap_events"], v["overlap_per_1000"],
                             v["vehicle_steps"], v["capviol_steps"],
                             round(v["max_speed_mps"], 4),
                             v["C2_corrected_pass"]])

    W.dump_json(out / "deviation_check.json", dict(
        note=("re-run aggregates vs part-1 t2_summary.json verdicts. "
              "unexpected_diffs must be empty: every step-indexed aggregate "
              "(vehicle_steps, comparable_vehicle_steps, capviol, negspeed, "
              "max_speed, CONSTRAINED overlap events, RAW defective-geo "
              "reproduction) reproduces part-1 exactly. explained_diffs "
              "carry the two registered/measured exceptions: (1) part-1 RAW "
              "overlap column = defective counter output, superseded by the "
              "corrected count in overlap_addendum.json; (2) part-1 pooled "
              "its C1 event universes into dicts keyed by vehicle id "
              "(t2_run.py pt/tt_/pb/tb_ comprehensions), so its stored "
              "ttc/brake totals count distinct vehicles per window, not "
              "vehicle-seconds - the recomputed totals here are per-step "
              "vehicle-seconds."),
        by_variant_H=deviation))

    W.dump_json(out / "diag_summary.json", dict(
        experiment="WD-01 part 2A window-level safety diagnostics",
        registration_addendum_sha256=add_sha,
        vehicle_set=("frozen t0 set on both sides; truth leaders from the "
                     "full truth snapshot (part-1 truth-event convention), "
                     "predicted leaders from rollout states"),
        diagnostics_only=True, n_evals_per_H=len(evals),
        per_H_variant=diag_summary))

    cols = ["run", "t0", "variant", "H", "n0",
            "truth_ttc_lt3_exposure_s", "pred_ttc_lt3_exposure_s",
            "truth_ttc_lt1p5_exposure_s", "pred_ttc_lt1p5_exposure_s",
            "truth_hard_brake_s", "pred_hard_brake_s",
            "truth_min_ttc_s", "pred_min_ttc_s",
            "truth_any_ttc_lt1p5", "pred_any_ttc_lt1p5",
            "truth_any_ttc_lt3", "pred_any_ttc_lt3",
            "truth_any_hard_brake", "pred_any_hard_brake"]
    with (out / "diag_per_eval.csv").open("w", newline="",
                                          encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        for row in diag_rows:
            wr.writerow({c: ("" if row[c] is None else row[c])
                         for c in cols})

    meta = dict(
        commands=[sys.executable + " " + str(Path(sys.argv[0]).name)
                  + "   (env: WD_DEVICE=cpu)"],
        python=sys.version.split()[0],
        torch=torch.__version__, numpy=np.__version__,
        registration_addendum_sha256=add_sha,
        checkpoint=W.SURV_CKPT,
        checkpoint_start_sha256=ckpt_sha,
        checkpoint_end_sha256=W.sha256_file(W.SURV_CKPT),
        device=str(dev), wall_seconds=round(time.time() - T_START, 1),
        model_forward_seconds=round(fwd[0], 2),
        model_forward_vehicle_steps=fwd[1],
        throughput_vehicle_steps_per_s=(fwd[1] / fwd[0]) if fwd[0] else None,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        deviations=[k for k, v in deviation.items() if not v["matches"]])
    W.dump_json(out / "addendum_run_meta.json", meta)

    show = {k: dict(overlap_per_1000=v["overlap_per_1000"],
                    overlap_events=v["overlap_events"],
                    C2_corrected_pass=v["C2_corrected_pass"])
            for k, v in overlap.items()}
    print(json.dumps(dict(verdicts=show, wall_s=meta["wall_seconds"]),
                    indent=1))
    bad = meta["deviations"]
    print("ADDENDUM RUN DONE" if not bad
          else f"ADDENDUM RUN DONE WITH DEVIATIONS: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
