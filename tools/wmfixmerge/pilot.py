"""S-WMFIX-PILOT: pre-registered k-step panel-descent pilot (amended gate).

Authorization: MAIN REVIEW DECISION 2026-10-31 10:40 (option (b) of
WMFIX_FEAS_BLOCKED.md), registered in C4-V7FT-20261017/REGISTRATION.md. The
FD-agreement sub-gate of the real TRAIN feasibility is unsatisfiable for this
objective class; this pilot replaces it with a panel-descent measurement.

Protocol (pre-registered, unchanged here): from the zero-head entry-path
step_000000 (fresh WM2 weights, conserve_queue + learned admission ON,
merge-response head at its zero initialization), run 30 Adam steps on the 12
FIT groups with the registered training lr/schedule; measure frozen-rerun noise
as 3 extra full-panel reruns of step_000000; evaluate the full 12x2x{120,300} s
TRAIN panel paired-J MAE at steps 0/10/20/30. PASS = best of steps 10-30 beats
the step-0 baseline at BOTH horizons by strictly more than 2x the frozen-rerun
sample SD, all gradients finite, frozen tensors bitwise retained. PASS
continues the registered curriculum in place within the <= 6 h budget (pilot
time and steps count); FAIL stops training and keeps every artifact.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time

from tools.wmfixmerge.data import WM2_SHA, digest, implementation_hashes, install_guard, safe

PILOT_STEPS = 30
EVAL_STEPS = (10, 20, 30)
RERUNS = 3
HORIZONS = (120, 300)


def decide(step0_runs, step_evals, horizons=HORIZONS, multiplier=2.0,
           finite_gradients=True, frozen_bitwise=True):
    """Pure decision rule; ``step0_runs[h]`` lists the 1+RERUNS step-0 panels."""
    baseline = {h: statistics.fmean(step0_runs[h]) for h in horizons}
    sd = {h: statistics.stdev(step0_runs[h]) if len(step0_runs[h]) > 1 else 0.0
          for h in horizons}
    best = {h: min(step_evals[s][h] for s in step_evals) for h in horizons}
    improvement = {h: baseline[h] - best[h] for h in horizons}
    passed = bool(finite_gradients and frozen_bitwise and
                  all(improvement[h] > multiplier * sd[h] for h in horizons))
    return dict(baseline=baseline, sample_sd=sd, best_of_steps=best,
                improvement=improvement, threshold={h: multiplier * sd[h] for h in horizons},
                finite_gradients=finite_gradients, frozen_bitwise=frozen_bitwise,
                passed=passed)


def panel(rt, horizons=HORIZONS):
    """Same measurement as feasibility.panel_gate 'initial': paired-J MAE."""
    import numpy as np
    import torch
    began = time.monotonic()
    values = {h: [] for h in horizons}
    per_group = {h: {} for h in horizons}
    for h in horizons:
        for group in range(len(rt.source.groups)):
            for arm in (0, 1):
                with torch.no_grad():
                    predicted, recorded, _, _ = rt.rollout(group, h, arm_index=arm)
                contrast = float(((predicted[1, 0] - predicted[0, 0]) -
                                  (recorded[1, 0] - recorded[0, 0])))
                values[h].append(contrast)
                per_group[h].setdefault(group, []).append(contrast)
        if len(values[h]) != 2 * len(rt.source.groups):
            raise ValueError("incomplete pilot panel")
    if rt.device.type == "cuda":
        torch.cuda.empty_cache()
    return {str(h): dict(paired_J_MAE=float(np.mean(np.abs(values[h]))),
                         per_group={g: float(np.mean(np.abs(v)))
                                    for g, v in per_group[h].items()},
                         rows=len(values[h])) for h in horizons
            }, dict(seconds=round(time.monotonic() - began, 1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "config", "checkpoint", "controllers", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--schedule", default="tools/wmfixmerge/train_config.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=None)
    args = parser.parse_args(argv)
    install_guard()
    schedule = json.loads(safe(args.schedule).read_text())
    if schedule["max_seconds"] > 21600 or schedule["stages"] != [[60, 30], [120, 30], [300, 60]] \
            or schedule["checkpoint_every_updates"] != 1:
        raise ValueError("schedule differs from registered six-hour curriculum")
    if digest(args.checkpoint) != WM2_SHA:
        raise ValueError("base is not registered WM2")
    out = safe(args.out)
    if not out.is_dir() or any(out.iterdir()):
        raise ValueError("out must be an existing empty directory; never duplicate/resume by guess")
    start = time.monotonic()
    with (out / "RUN.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), started=time.time(), device=args.device,
                       pilot="S-WMFIX-PILOT option (b), MAIN 10:40",
                       schedule_sha256=digest(args.schedule)), stream)
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(schedule["seed"])
    if args.device == "cuda" and args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    from tools.wmfixmerge.runtime import Runtime
    from tools.wmfixmerge.train import export
    from wms.model.merge_response import freeze_except_response
    from wms.train.merge_response import objective_terms, paired_objective
    rt = Runtime(args.prepared, args.config, args.checkpoint, args.controllers, args.device)
    names = freeze_except_response(rt.model)
    optimizer = torch.optim.Adam([p for p in rt.model.parameters() if p.requires_grad],
                                 lr=schedule["lr"])
    original = torch.load(safe(args.checkpoint), map_location="cpu", weights_only=True)
    frozen = {n: v.detach().cpu().clone() for n, v in rt.model.state_dict().items()
              if not n.startswith("merge_response.")}
    from dataclasses import asdict
    receipt = dict(optimizer_names=names, prepared_sha256=digest(args.prepared),
                   base_sha256=WM2_SHA, implementation_hashes=implementation_hashes(),
                   authorization="MAIN REVIEW DECISION 2026-10-31 10:40 (option b)",
                   amended_blocked_feasibility_sha256=
                   "65771d0290d75df530110cfe91f3ac65f09424ed2b5318db94c35bee0627f7b9",
                   flags=dict(merge_response=True, conserve_queue=True, learned_admission=True))
    resolved = asdict(rt.config)
    resolved["model"]["merge_response"] = True
    with (out / "candidate_config.json").open("x") as stream:
        json.dump(resolved, stream, indent=2)
    step = 0
    latest = export(rt.model, original, frozen, optimizer, out, step, receipt)

    def record_panel(tag, values, meta):
        with (out / f"PANEL_{tag}.json").open("x") as stream:
            json.dump(dict(step=step, panel=values, **meta), stream, indent=2, allow_nan=False)
        return {h: values[str(h)]["paired_J_MAE"] for h in HORIZONS}

    step0_runs = {h: [] for h in HORIZONS}
    for repeat in range(1 + RERUNS):
        values, meta = panel(rt)
        mae = record_panel(f"step0_r{repeat}", values, meta)
        for h in HORIZONS:
            step0_runs[h].append(mae[h])
    step_evals = {0: {h: statistics.fmean(step0_runs[h]) for h in HORIZONS}}

    finite_steps = 0
    cap_stop = False
    timings = []
    try:
        for step in range(1, PILOT_STEPS + 1):
            reserve = max(timings, default=300.)
            if time.monotonic() - start + reserve >= schedule["max_seconds"]:
                cap_stop = True
                break
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            predicted, recorded, refs, roll_receipt = rt.rollout(
                (step - 1) % len(rt.source.groups), schedule["stages"][0][0],
                arm_index=(step - 1) // len(rt.source.groups))
            loss = paired_objective(predicted, recorded, refs, rt.scales,
                                    anchor_scales=rt.anchor_scales)
            terms = objective_terms(predicted, recorded, refs, rt.scales,
                                    anchor_scales=rt.anchor_scales)
            if step == 1:
                with (out / "STEP0_TERMS.json").open("x") as stream:
                    json.dump(terms, stream, indent=2, allow_nan=False)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite paired objective")
            loss.backward()
            gradients = [p.grad for p in rt.model.parameters()
                         if p.requires_grad and p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError("nonfinite or absent correction gradients")
            optimizer.step()
            finite_steps = step
            latest = export(rt.model, original, frozen, optimizer, out, step, receipt)
            row = dict(step=step, horizon=schedule["stages"][0][0], loss=float(loss.detach()),
                       terms=terms, seconds=time.monotonic() - begin,
                       elapsed=time.monotonic() - start, rollout=roll_receipt,
                       checkpoint=str(latest),
                       source_contrast_error=((predicted[:, 2] - predicted[0, 2]) -
                                              (recorded[:, 2] - recorded[0, 2])).detach().cpu().tolist())
            with (out / f"update_{step:06d}.json").open("x") as stream:
                json.dump(row, stream, allow_nan=False)
            print(json.dumps(dict(step=step, loss=row["loss"], seconds=row["seconds"])), flush=True)
            if step in EVAL_STEPS:
                values, meta = panel(rt)
                step_evals[step] = record_panel(f"step{step}", values, meta)
            timings.append(row["seconds"])
        frozen_ok = all(torch.equal(v, rt.model.state_dict()[n].detach().cpu())
                        for n, v in frozen.items())
        if step_evals.keys() >= {0, *EVAL_STEPS}:
            decision = decide(step0_runs, {s: e for s, e in step_evals.items() if s},
                              finite_gradients=finite_steps == PILOT_STEPS,
                              frozen_bitwise=frozen_ok)
        else:  # cap stop or partial panels: fail closed
            decision = dict(passed=False, blocker="pilot did not reach step 30",
                            baseline={}, sample_sd={}, improvement={}, threshold={})
        decision.update(protocol=dict(pilot_steps=PILOT_STEPS, eval_steps=[0, *EVAL_STEPS],
                                      reruns=RERUNS, horizons=list(HORIZONS),
                                      noise="sample SD across 1+3 step-0 full panels",
                                      baseline="mean of the 4 step-0 panels"),
                        step0_runs={str(h): step0_runs[h] for h in HORIZONS},
                        step_evals={str(s): e for s, e in step_evals.items()},
                        completed_steps=finite_steps, cap_stop=cap_stop,
                        elapsed_s=time.monotonic() - start)
        with (out / "PILOT.json").open("x") as stream:
            json.dump(decision, stream, indent=2, allow_nan=False)
        print(json.dumps(decision, indent=2, allow_nan=False), flush=True)
        if not decision["passed"]:
            return 2
    except (RuntimeError, ValueError) as exc:
        with (out / "PILOT.json").open("x") as stream:
            json.dump(dict(passed=False, status="FAIL", blocker=f"{type(exc).__name__}: {exc}",
                           completed_steps=finite_steps, elapsed_s=time.monotonic() - start),
                      stream, indent=2, allow_nan=False)
        raise

    # PASS: continue the registered curriculum in place; budget counts the pilot.
    stopped = False
    for horizon, updates in schedule["stages"][1:]:
        for _ in range(updates):
            reserve = max(timings, default=300.)
            if time.monotonic() - start + reserve >= schedule["max_seconds"]:
                stopped = True
                break
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            predicted, recorded, refs, roll_receipt = rt.rollout(
                step % len(rt.source.groups), horizon,
                arm_index=step // len(rt.source.groups))
            loss = paired_objective(predicted, recorded, refs, rt.scales,
                                    anchor_scales=rt.anchor_scales)
            terms = objective_terms(predicted, recorded, refs, rt.scales,
                                    anchor_scales=rt.anchor_scales)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite paired objective")
            loss.backward()
            gradients = [p.grad for p in rt.model.parameters()
                         if p.requires_grad and p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError("nonfinite or absent correction gradients")
            optimizer.step()
            step += 1
            latest = export(rt.model, original, frozen, optimizer, out, step, receipt)
            row = dict(step=step, horizon=horizon, loss=float(loss.detach()), terms=terms,
                       seconds=time.monotonic() - begin, elapsed=time.monotonic() - start,
                       rollout=roll_receipt, checkpoint=str(latest),
                       source_contrast_error=((predicted[:, 2] - predicted[0, 2]) -
                                              (recorded[:, 2] - recorded[0, 2])).detach().cpu().tolist())
            with (out / f"update_{step:06d}.json").open("x") as stream:
                json.dump(row, stream, allow_nan=False)
            print(json.dumps(dict(step=step, horizon=horizon, loss=row["loss"],
                                  seconds=row["seconds"])), flush=True)
            timings.append(row["seconds"])
        if stopped:
            break
    with (out / "FINAL.json").open("x") as stream:
        json.dump(dict(status="COMPLETE", step=step, checkpoint=str(latest),
                       sha256=digest(latest), elapsed=time.monotonic() - start,
                       selection="final completed update", cap_stop=stopped,
                       pilot_passed=True), stream, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
