"""S-WMFIX-PILOT continuation after the 11:52 determinism amendment (MAIN).

PANEL_step0_r1 reproduced r0 exactly (all panel values bitwise-equal; only
wallclock meta differs), so frozen-rerun noise is 0 by determinism. MAIN
ADDENDUM 2026-10-31 11:52: skip r2/r3 and go straight to training steps 1-30;
the step-10/20/30 full panels are evaluated OFF the critical path on gpuR from
immutable exported checkpoints (same code/inputs/CRN, hash-verified relay) and
relayed back as PANEL_step{10,20,30}.json; this process polls for them,
assembles PILOT.json with the unchanged pre-registered decide(), and on PASS
continues the registered curriculum in place under a fresh six-hour cap
(addendum item 3: the killed process' 21,570 s cap had <= 3 h of real training
left after the projected PASS; every pilot step is still reported).
"""
from __future__ import annotations

import argparse
import json
import os
import time

from tools.wmfixmerge.data import WM2_SHA, digest, implementation_hashes, install_guard, safe
from tools.wmfixmerge.pilot import EVAL_STEPS, HORIZONS, PILOT_STEPS, decide

PANEL_WAIT_S = 4 * 3600


def step0_panel_runs(out):
    """Load every complete PANEL_step0_r*.json; all must be identical (json only)."""
    out = safe(out)
    runs = {h: [] for h in HORIZONS}
    files = sorted(out.glob("PANEL_step0_r*.json"))
    if len(files) < 2:
        raise ValueError("fewer than two step-0 panels; determinism unproven")
    docs = [json.loads(p.read_text()) for p in files]
    for doc in docs:
        row = validate_step_panel(doc, 0)
        for h in HORIZONS:
            runs[h].append(row[h])
    for h in HORIZONS:
        if len(set(runs[h])) != 1:
            raise ValueError(f"step-0 panels disagree at {h} s; determinism unproven")
    return runs, files


def validate_step_panel(doc, step):
    """Schema-check one panel receipt; returns {horizon: paired_J_MAE}."""
    panel = doc.get("panel")
    if not isinstance(panel, dict) or doc.get("step") != step:
        raise ValueError("panel receipt is not for this step")
    out = {}
    for h in HORIZONS:
        row = panel.get(str(h))
        if not isinstance(row, dict) or not isinstance(row.get("per_group"), dict) \
                or row.get("rows") != 2 * len(row["per_group"]) or len(row["per_group"]) != 12:
            raise ValueError(f"panel receipt incomplete at {h} s")
        out[h] = float(row["paired_J_MAE"])
    return out


def wait_for_step_panels(out, deadline):
    """Poll the out dir for the relayed PANEL_step{10,20,30}.json receipts."""
    out = safe(out)
    while True:
        evals, missing = {}, []
        for tag in EVAL_STEPS:
            path = out / f"PANEL_step{tag}.json"
            if path.is_file():
                try:
                    evals[tag] = validate_step_panel(json.loads(path.read_text()), tag)
                except (ValueError, KeyError, json.JSONDecodeError):
                    missing.append(tag)  # partially relayed; retry next poll
            else:
                missing.append(tag)
        if not missing:
            return evals
        if time.monotonic() > deadline:
            raise TimeoutError("relayed step panels missing: " + ",".join(map(str, missing)))
        time.sleep(60)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "config", "checkpoint", "controllers", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--schedule", default="tools/wmfixmerge/train_config.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=None)
    parser.add_argument("--superseded-pids", default="", help="gpuA pilot PIDs stopped per MAIN 11:52")
    args = parser.parse_args(argv)
    install_guard()
    schedule = json.loads(safe(args.schedule).read_text())
    if schedule["max_seconds"] > 21600 or schedule["stages"] != [[60, 30], [120, 30], [300, 60]] \
            or schedule["checkpoint_every_updates"] != 1:
        raise ValueError("schedule differs from registered six-hour curriculum")
    if digest(args.checkpoint) != WM2_SHA:
        raise ValueError("base is not registered WM2")
    out = safe(args.out)
    step0_runs, step0_files = step0_panel_runs(out)
    for refused in ("CONT_RUN.json", "PILOT.json", "FINAL.json", "update_000001.json",
                    "step_000001.pt", "PANEL_step0_r0_mismatch"):
        if (out / refused).exists():
            raise ValueError("continuation artifacts already exist: " + refused)
    if "pilot" not in json.loads((out / "RUN.json").read_text()).get("pilot", ""):
        raise ValueError("out dir is not a registered S-WMFIX-PILOT run")
    start = time.monotonic()
    with (out / "CONT_RUN.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), started=time.time(), device=args.device,
                       superseded_pids=[p for p in args.superseded_pids.split(",") if p],
                       authorization="MAIN ADDENDUM 2026-10-31 11:52 (r1==r0 exactly; "
                                     "noise 0 by determinism; r2/r3 skipped; step panels "
                                     "on gpuR; fresh 6 h cap)",
                       step0_panels=[p.name for p in step0_files],
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
    payload = torch.load(safe(out / "step_000000.pt"), map_location="cpu", weights_only=True)
    if payload["wmfix"]["base_sha256"] != WM2_SHA or payload["wmfix"].get("step") != 0:
        raise ValueError("step_000000 is not the exported zero-head entry-path start")
    rt.model.load_state_dict(payload["model"], strict=True)
    for name, value in payload["model"].items():
        if not name.startswith("merge_response.") and not torch.equal(value, original["model"][name]):
            raise ValueError("frozen WM2 tensor changed in step_000000: " + name)
    optimizer.load_state_dict(torch.load(safe(out / "optimizer_000000.pt"),
                                         map_location="cpu", weights_only=True))
    frozen = {n: v.detach().cpu().clone() for n, v in rt.model.state_dict().items()
              if not n.startswith("merge_response.")}
    receipt = dict(optimizer_names=names, prepared_sha256=digest(args.prepared),
                   base_sha256=WM2_SHA, implementation_hashes=implementation_hashes(),
                   authorization="MAIN ADDENDUM 2026-10-31 11:52 continuation of "
                                 "S-WMFIX-PILOT (MAIN REVIEW DECISION 10:40 option b)",
                   flags=dict(merge_response=True, conserve_queue=True, learned_admission=True))
    latest = out / "step_000000.pt"

    step = 0
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
            timings.append(row["seconds"])
        frozen_ok = all(torch.equal(v, rt.model.state_dict()[n].detach().cpu())
                        for n, v in frozen.items())
        step_evals = wait_for_step_panels(out, time.monotonic() + PANEL_WAIT_S)
        decision = decide(step0_runs, step_evals,
                          finite_gradients=finite_steps == PILOT_STEPS, frozen_bitwise=frozen_ok)
        gpu_r_baseline = None
        if (out / "PANEL_step0_gpuR.json").is_file():
            try:
                gpu_r_baseline = validate_step_panel(
                    json.loads((out / "PANEL_step0_gpuR.json").read_text()), 0)
            except (ValueError, KeyError, json.JSONDecodeError):
                gpu_r_baseline = None
        decision.update(protocol=dict(pilot_steps=PILOT_STEPS, eval_steps=[0, *EVAL_STEPS],
                                      step0_panels=len(step0_files), horizons=list(HORIZONS),
                                      noise="0 by determinism (all step-0 panels identical); "
                                            "MAIN ADDENDUM 11:52",
                                      baseline="mean of the identical step-0 panels",
                                      step_panel_producer="gpuR relay (hash-verified)"),
                        determinism=dict(r0_equals_r1=True, r2_r3_skipped=True,
                                         superseded_pids=args.superseded_pids),
                        gpuR_step0_baseline=gpu_r_baseline,
                        step_evals={str(s): e for s, e in step_evals.items()},
                        completed_steps=finite_steps, cap_stop=cap_stop,
                        elapsed_s=time.monotonic() - start)
        with (out / "PILOT.json").open("x") as stream:
            json.dump(decision, stream, indent=2, allow_nan=False)
        print(json.dumps(decision, indent=2, allow_nan=False), flush=True)
        if not decision["passed"]:
            return 2
    except (RuntimeError, ValueError, TimeoutError) as exc:
        with (out / "PILOT.json").open("x") as stream:
            json.dump(dict(passed=False, status="FAIL", blocker=f"{type(exc).__name__}: {exc}",
                           completed_steps=finite_steps, elapsed_s=time.monotonic() - start),
                      stream, indent=2, allow_nan=False)
        raise

    # PASS: continue the registered curriculum in place under the fresh cap.
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
                       sha256=digest(latest), elapsed_s=time.monotonic() - start,
                       selection="final completed update", cap_stop=stopped,
                       pilot_passed=True), stream, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
