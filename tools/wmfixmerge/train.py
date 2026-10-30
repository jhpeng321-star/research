"""Frozen-everything-else fine-tune. MAIN must explicitly enable both default-OFF flags."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

from tools.wmfixmerge.data import WM2_SHA, digest, implementation_hashes, install_guard, safe


def check_gates(args):
    feasibility = json.loads(safe(args.feasibility).read_text())
    if feasibility.get("mode") != "real_train" or feasibility.get("launch_ready") is not True \
            or feasibility.get("source_clock_match") is not True or feasibility.get("passed") is not True:
        raise ValueError("real TRAIN feasibility/source-clock gates are not PASS")
    for key in ("prepared", "config", "checkpoint", "controllers", "corrected_report"):
        if feasibility.get(key + "_sha256") != digest(getattr(args, key)):
            raise ValueError("feasibility input changed: " + key)
    if feasibility.get("implementation_hashes") != implementation_hashes():
        raise ValueError("implementation changed after CPU feasibility")
    review = json.loads(safe(args.review).read_text())
    if review.get("reviewer") != "Astra" or review.get("status") != "APPROVED" \
            or review.get("feasibility_sha256") != digest(args.feasibility):
        raise ValueError("independent Astra review not bound to passing real feasibility")
    return feasibility


def export(model, original, frozen, optimizer, out, step, receipt):
    import torch
    for name, value in frozen.items():
        if not torch.equal(value, model.state_dict()[name].detach().cpu()):
            raise ValueError("frozen WM2 tensor changed: " + name)
    payload = dict(original)
    payload["model"] = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
    payload["config"] = dict(original["config"], merge_response=True, physics=asdict(model.physics))
    payload["wmfix"] = dict(receipt, step=step, base_sha256=WM2_SHA)
    # Immutable completed-update checkpoints; interrupted writes never masquerade as completed.
    path = out / f"step_{step:06d}.pt"
    if path.exists() or (out / f"optimizer_{step:06d}.pt").exists():
        raise FileExistsError("completed update already exists")
    with path.with_suffix(".pt.partial").open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(path.with_suffix(".pt.partial"), path)
    with (out / f"optimizer_{step:06d}.pt").open("xb") as stream:
        torch.save(optimizer.state_dict(), stream)
    from wms.model.checkpoint import load_checkpoint
    loaded = load_checkpoint(path).model
    if loaded.merge_response is None or any(not torch.equal(v, loaded.state_dict()[n])
                                            for n, v in payload["model"].items()):
        raise ValueError("canonical reload failed exact tensor parity")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "config", "checkpoint", "controllers", "feasibility", "corrected-report", "review", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--schedule", default="tools/wmfixmerge/train_config.json")
    parser.add_argument("--enable-merge-response", action="store_true")
    parser.add_argument("--conserve-queue", action="store_true")
    parser.add_argument("--learned-admission", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    install_guard()
    if not args.enable_merge_response or not args.conserve_queue or not args.learned_admission:
        parser.error("all three default-OFF flags must be explicitly enabled")
    check_gates(args)  # Before torch/CUDA, model, or source opening.
    schedule = json.loads(safe(args.schedule).read_text())
    if schedule["max_seconds"] > 21600 or schedule["stages"] != [[60, 30], [120, 30], [300, 60]]:
        raise ValueError("schedule differs from registered six-hour curriculum")
    out = safe(args.out)
    if not out.is_dir() or any(out.iterdir()):
        raise ValueError("out must be an existing empty directory; never duplicate/resume by guess")
    # Exclusive persistent run receipt blocks a second invocation even after interruption.
    start = time.monotonic()
    with (out / "RUN.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), started=time.time(), device=args.device,
                       feasibility_sha256=digest(args.feasibility), schedule_sha256=digest(args.schedule)), stream)
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(schedule["seed"])
    from tools.wmfixmerge.runtime import Runtime
    from wms.model.merge_response import freeze_except_response
    from wms.train.merge_response import objective_terms, paired_objective
    rt = Runtime(args.prepared, args.config, args.checkpoint, args.controllers, args.device)
    names = freeze_except_response(rt.model)
    optimizer = torch.optim.Adam([p for p in rt.model.parameters() if p.requires_grad], lr=schedule["lr"])
    original = torch.load(safe(args.checkpoint), map_location="cpu", weights_only=True)
    frozen = {n: v.detach().cpu().clone() for n, v in rt.model.state_dict().items()
              if not n.startswith("merge_response.")}
    receipt = dict(optimizer_names=names, prepared_sha256=digest(args.prepared),
                   feasibility_sha256=digest(args.feasibility), review_sha256=digest(args.review),
                   implementation_hashes=implementation_hashes(),
                   flags=dict(merge_response=True, conserve_queue=True, learned_admission=True))
    # A matching nested config is supplied to legacy scorers/accept.py; unknown tensors must reject.
    resolved = asdict(rt.config)
    resolved["model"]["merge_response"] = True
    with (out / "candidate_config.json").open("x") as stream:
        json.dump(resolved, stream, indent=2)
    step = 0
    latest = export(rt.model, original, frozen, optimizer, out, step, receipt)
    timings = []
    stopped = False
    for horizon, updates in schedule["stages"]:
        for _ in range(updates):
            # Reserve measured worst update + export time: do not start work past the cap.
            reserve = max(timings, default=300.)
            if time.monotonic() - start + reserve >= schedule["max_seconds"]:
                stopped = True
                break
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            predicted, recorded, refs, roll_receipt = rt.rollout(step % len(rt.source.groups), horizon,
                arm_index=step // len(rt.source.groups))
            loss = paired_objective(predicted, recorded, refs, rt.scales,
                                    anchor_scales=rt.anchor_scales)
            terms = objective_terms(predicted, recorded, refs, rt.scales,
                                    anchor_scales=rt.anchor_scales)
            if step == 0:
                # WMFIX_REVIEW item 7: the measured gradient-term shares at step0
                # are part of the run receipt; the paired term must dominate.
                with (out / "STEP0_TERMS.json").open("x") as stream:
                    json.dump(terms, stream, indent=2, allow_nan=False)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite paired objective")
            loss.backward()
            gradients = [p.grad for p in rt.model.parameters() if p.requires_grad and p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError("nonfinite or absent correction gradients")
            optimizer.step()
            step += 1
            latest = export(rt.model, original, frozen, optimizer, out, step, receipt)
            row = dict(step=step, horizon=horizon, loss=float(loss.detach()), terms=terms,
                       seconds=time.monotonic() - begin,
                       elapsed=time.monotonic() - start, rollout=roll_receipt, checkpoint=str(latest),
                       source_contrast_error=((predicted[:, 2] - predicted[0, 2]) -
                                              (recorded[:, 2] - recorded[0, 2])).detach().cpu().tolist())
            with (out / f"update_{step:06d}.json").open("x") as stream:
                json.dump(row, stream, allow_nan=False)
            print(json.dumps(row), flush=True)
            timings.append(row["seconds"])
        if stopped:
            break
    with (out / "FINAL.json").open("x") as stream:
        json.dump(dict(status="COMPLETE" if step else "NO_UPDATES", step=step,
                       checkpoint=str(latest), sha256=digest(latest), elapsed=time.monotonic() - start,
                       selection="final completed update", cap_stop=stopped), stream, indent=2)
    return 0 if step else 2


if __name__ == "__main__":
    raise SystemExit(main())
