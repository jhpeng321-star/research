"""MAIN 15:25 v2-DATA authorization; exact v1 objective and unchanged decide()."""
from __future__ import annotations

import json
import math
import os
import subprocess
import time

from tools.wmfixmerge.data import WM2_SHA, digest, implementation_hashes, safe

PREPARED_SHA = "84751dc7b0c9c27795810575f45a716b1af587f5dd6c886b93b453f4697419cc"
CONTRACT_SHA = "e67850f3eaac8595f35eea90e478c3a265dc5e0a250a11a0531127a3a7948155"
SELECTION_SEEDS = tuple(range(969432, 969440))
EVAL_STEPS = (10, 20, 30)


def check_binding(args):
    """An explicit, immutable authorization route, never an FD receipt override."""
    if args.smooth_events:
        raise ValueError("v2-DATA requires smooth-events OFF")
    if not args.harness_contract or digest(args.harness_contract) != CONTRACT_SHA:
        raise ValueError("v2-DATA requires the registered HARNESS_CONTRACT_V2")
    if digest(args.prepared) != PREPARED_SHA:
        raise ValueError("v2-DATA requires the registered 52-world TRAIN_PREPARED_V2")
    if digest(args.checkpoint) != WM2_SHA:
        raise ValueError("base is not registered WM2")
    return dict(authorization="MAIN REVIEW 2026-10-31 15:25 v2-DATA; 10:40/11:52 gate",
                prepared_sha256=PREPARED_SHA, harness_contract_sha256=CONTRACT_SHA,
                base_sha256=WM2_SHA, smooth_events=False)


def fit_groups(groups, labels):
    seeds = {r["pair_key"]: int(r["seed"]) for r in labels}
    selected = [i for i, g in enumerate(groups) if seeds[g["pair_key"]] not in SELECTION_SEEDS]
    if len(groups) != 52 or len(selected) != 44:
        raise ValueError("expected 52 panel worlds and 44 fitting worlds (MAIN 15:00)")
    return selected


def decide_panels(baseline, candidates, finite_gradients, frozen_bitwise):
    from tools.wmfixmerge.pilot import decide
    if set(candidates) != set(EVAL_STEPS):
        raise ValueError("all step-10/20/30 full panels are required")
    if not all(math.isfinite(v) for row in [baseline, *candidates.values()] for v in row.values()):
        raise ValueError("nonfinite full-panel score")
    decision = decide({h: [v, v] for h, v in baseline.items()}, candidates,
                      finite_gradients=finite_gradients, frozen_bitwise=frozen_bitwise)
    decision["step_evals"] = candidates
    decision["noise_note"] = "SD=0: measured identical frozen reruns, MAIN 11:52"
    return decision


class PanelQueue:
    """Bounded same-gpuA shards; grow concurrency only with measured free memory.

    Reserve 5 GiB per new inference process (4 GiB allocator plus context), at
    most four concurrent shards. No peer process is signalled or reconfigured.
    """
    def __init__(self, panels):
        self.panels = panels
        self.pending = []
        self.done = set()

    def submit(self, name, candidate, groups):
        self.pending.append((name, candidate, groups))

    def tick(self):
        for name in list(self.panels.running):
            code = self.panels.poll(name)
            if code is not None:
                if code:
                    raise RuntimeError(f"panel {name} failed: exit {code}")
                self.done.add(name)
        # One uninitialised process at a time: otherwise the same free bytes
        # could be reserved repeatedly before CUDA contexts have materialised.
        active = len(self.panels.running)
        if self.pending and active < 4:
            free = int(subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                text=True).strip()) if self.panels.device == "cuda" else 999999
            if free >= 5120 * (active + 1):
                name, candidate, groups = self.pending.pop(0)
                self.panels.launch(name, candidate, groups)

    def drain(self):
        while self.pending or self.panels.running:
            self.tick()
            time.sleep(5)


def run(args):
    import torch
    from dataclasses import asdict
    from tools.wmfixmerge.pilot_v2 import Panels, assemble
    from tools.wmfixmerge.runtime import Runtime
    from tools.wmfixmerge.train import export
    from wms.model.merge_response import freeze_except_response
    from wms.train.merge_response import objective_terms, paired_objective

    binding = check_binding(args)
    schedule = json.loads(safe(args.schedule).read_text())
    if schedule["max_seconds"] > 21600 or schedule["stages"] != [[60, 30], [120, 30], [300, 60]] \
            or schedule["checkpoint_every_updates"] != 1:
        raise ValueError("schedule differs from registered six-hour curriculum")
    out = safe(args.out)
    if not out.is_dir() or any(out.iterdir()):
        raise ValueError("out must be an existing empty directory")
    started = time.monotonic()

    def write(name, doc):
        with (out / name).open("x") as stream:
            json.dump(doc, stream, indent=2, allow_nan=False)

    write("RUN.json", dict(pid=os.getpid(), started=time.time(), **binding))
    torch.set_num_threads(2)
    torch.manual_seed(schedule["seed"])
    if args.device == "cuda" and args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    rt = Runtime(args.prepared, args.config, args.checkpoint, args.controllers, args.device)
    if rt.smooth_events or rt.merge_refine != (0, 0):
        raise AssertionError("data-only route changed dynamics/objective")
    fitting = fit_groups(rt.source.groups, rt.doc["labels"])
    names = freeze_except_response(rt.model)
    optimizer = torch.optim.Adam([p for p in rt.model.parameters() if p.requires_grad], lr=schedule["lr"])
    original = torch.load(safe(args.checkpoint), map_location="cpu", weights_only=True)
    frozen = {n: v.detach().cpu().clone() for n, v in rt.model.state_dict().items()
              if not n.startswith("merge_response.")}
    receipt = dict(**binding, optimizer_names=names, implementation_hashes=implementation_hashes(),
                   driver_hashes={n: digest(safe(__file__).parent / n) for n in
                                  ("pilot_v2.py", "data_pilot.py", "pilot.py", "panel_eval_v2.py")},
                   schedule_sha256=digest(args.schedule), fitting_groups=fitting,
                   selection_seeds=list(SELECTION_SEEDS),
                   flags=dict(merge_response=True, conserve_queue=True, learned_admission=True,
                              smooth_events=False, merge_bases=0, ramp_bases=0),
                   full_panel=dict(groups=52, arms=[0, 1], horizons=[120, 300], steps=[0, 10, 20, 30]))
    config = asdict(rt.config)
    config["model"]["merge_response"] = True
    write("candidate_config.json", config)
    write("PROTOCOL.json", receipt)
    latest = export(rt.model, original, frozen, optimizer, out, 0, receipt)
    panels = Panels(out, ["--prepared", args.prepared, "--config", args.config,
                         "--checkpoint", args.checkpoint, "--controllers", args.controllers],
                    0.08 if args.device == "cuda" else None, args.device)
    queue = PanelQueue(panels)
    shards = [list(range(i * 13, (i + 1) * 13)) for i in range(4)]

    def submit(step, checkpoint):
        for i, groups in enumerate(shards):
            queue.submit(f"step{step}_full_s{i}", checkpoint, groups)

    submit(0, latest)
    timings = []
    step = 0

    def update(horizon):
        nonlocal step, latest
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        predicted, recorded, refs, roll_receipt = rt.rollout(
            fitting[step % len(fitting)], horizon, arm_index=step // len(fitting))
        loss = paired_objective(predicted, recorded, refs, rt.scales, anchor_scales=rt.anchor_scales)
        terms = objective_terms(predicted, recorded, refs, rt.scales, anchor_scales=rt.anchor_scales)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite paired objective")
        loss.backward()
        grads = [p.grad for p in rt.model.parameters() if p.requires_grad and p.grad is not None]
        if not grads or not all(torch.isfinite(g).all() for g in grads):
            raise ValueError("nonfinite/absent gradient")
        optimizer.step()
        step += 1
        latest = export(rt.model, original, frozen, optimizer, out, step, receipt)
        timings.append(time.monotonic() - begin)
        write(f"update_{step:06d}.json", dict(step=step, horizon=horizon, loss=float(loss.detach()),
              terms=terms, finite_gradients=True, frozen_bitwise=True, seconds=timings[-1],
              elapsed=time.monotonic() - started, rollout=roll_receipt, checkpoint=str(latest)))
        print(json.dumps(dict(step=step, horizon=horizon, loss=float(loss.detach()))), flush=True)
        queue.tick()

    try:
        for _ in range(30):
            if time.monotonic() - started + max(timings, default=300) >= schedule["max_seconds"]:
                raise RuntimeError("budget exhausted before the 30-step pilot completed")
            update(60)
            if step in EVAL_STEPS:
                submit(step, latest)
        queue.drain()
        scores = {}
        for s in (0, *EVAL_STEPS):
            panel = assemble([panels.read(f"step{s}_full_s{i}")["rows"] for i in range(4)], range(52))
            write(f"FULL_PANEL_STEP{s}.json", panel)
            scores[s] = {h: panel[str(h)]["paired_J_MAE"] for h in (120, 300)}
        frozen_ok = all(torch.equal(v, rt.model.state_dict()[n].detach().cpu()) for n, v in frozen.items())
        decision = decide_panels(scores[0], {s: scores[s] for s in EVAL_STEPS}, step == 30, frozen_ok)
        decision.update(completed_steps=step, protocol=receipt, elapsed_s=time.monotonic() - started)
        write("PILOT_V2.json", decision)
        print(json.dumps(decision, allow_nan=False), flush=True)
        if not decision["passed"]:
            return 2
        # Same v1 continuation: optimizer/weights at step 30, no selection on verdict panels.
        stopped = False
        for horizon, count in schedule["stages"][1:]:
            for _ in range(count):
                if time.monotonic() - started + max(timings, default=300) >= schedule["max_seconds"]:
                    stopped = True
                    break
                update(horizon)
            if stopped:
                break
        write("FINAL.json", dict(status="COMPLETE", step=step, checkpoint=str(latest), sha256=digest(latest),
                                 elapsed_s=time.monotonic() - started, cap_stop=stopped,
                                 pilot_passed=True, pilot="v2-DATA"))
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        if not (out / "PILOT_V2.json").exists():
            write("PILOT_V2.json", dict(passed=False, status="BLOCKED", blocker=str(exc), completed_steps=step))
        raise
