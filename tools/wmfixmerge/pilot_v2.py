"""S-WMFIX-PILOT v2: 52-world smooth-events pilot (MAIN 13:50 TRACK 1, owner go).

Pre-registered protocol. Registered changes vs the failed 12-world pilot v1
(WMFIX_PILOT_FAIL.md, decide() unchanged in semantics):

1. DATA: the combined TRAIN corpus of the 12 registered FIT worlds + the 40
   S-WMDATA worlds (role wmdata_fit; merge_corpus.py root), one prepared doc.
2. TRAINING-ONLY smoothed admission/crossings (--smooth-events, S-WMFIX-C
   fallback 1; c4/wmfixc merged). Evaluation NEVER sets the flag: every panel
   below measures the exact integer path. Finer merge bases stay OFF (v3
   candidate).

Measurements: 30 Adam steps (registered lr 3e-4, stage-0 60 s horizon,
group-then-arm cycling over the 52 groups, merge_response.* only); a 4-group
mini panel (arms 0,1; horizons 120/300 s; seeds 973002/969400/969405/969410)
every 5 steps evaluated by same-host subprocesses (panel_eval_v2, exact
integer path) on the just-exported immutable checkpoint; early stop after 2 consecutive completed mini-panel
regressions vs the running best; best step = argmin pooled mini MAE (earliest
tie). DECISION: the FULL 52-group x arms(0,1) x {120,300} s panel paired-J MAE
at the chosen step must be STRICTLY better than the same panel at step 0 at
BOTH horizons (noise is 0 by the measured r1==r0 determinism of this exact
code path, PILOT.json v1), with finite gradients and frozen-bitwise retained.
PASS continues the registered curriculum in place under the <= 6 h budget;
FAIL stops and keeps every artifact.

Gates before any step: the smooth-events real TRAIN feasibility receipt must
PASS (FD agreement re-measured on the smoothed objective) and bind this tree,
this prepared doc and the registered base checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

from tools.wmfixmerge.data import WM2_SHA, digest, implementation_hashes, install_guard, safe

PILOT_STEPS = 30
MINI_EVERY = 5
MINI_SEEDS = (973002, 969400, 969405, 969410)
HORIZONS = (120, 300)
ARMS = (0, 1)
MINI_INFLIGHT = 3
FULL_SHARDS = 4
PANEL_MEMORY_FRACTION = 0.15


def pooled(values):
    """Mini metric: mean of the per-horizon paired-J MAEs."""
    return statistics.fmean(values[h] for h in HORIZONS)


def mini_update(completed, best):
    """(best, regression_streak) from ordered completed mini panels.

    ``completed`` is a list of (step, values) chronologically ordered;
    a panel regresses when its pooled metric exceeds the running best.
    """
    streak = 0
    for step, values in sorted(completed, key=lambda item: item[0]):
        metric = pooled(values)
        if best is None or metric < best[1] - 1e-12:
            best = (step, metric)
            streak = 0
        elif metric > best[1] + 1e-12:
            streak += 1
        else:
            streak = 0
    return best, streak


def choose_step(completed):
    """Best step = argmin pooled mini MAE over completed panels (earliest tie)."""
    if not completed:
        raise ValueError("no completed mini panels")
    best = None
    for step, values in sorted(completed, key=lambda item: item[0]):
        metric = pooled(values)
        if best is None or metric < best[1] - 1e-12:
            best = (step, metric)
    return best[0]


def assemble(shard_rows, expected_groups):
    """Full-panel MAE per horizon from shard rows; fails closed on any gap."""
    rows = [row for shard in shard_rows for row in shard]
    panel = {}
    for h in HORIZONS:
        part = [r for r in rows if r["horizon"] == h]
        seen = {r["group"] for r in part}
        if seen != set(expected_groups) or len(part) != len(expected_groups) * len(ARMS):
            raise ValueError(f"incomplete full panel at {h} s: {len(part)} rows, "
                             f"{len(seen)} groups")
        panel[str(h)] = dict(
            paired_J_MAE=statistics.fmean(abs(r["contrast"]) for r in part),
            per_group={str(g): statistics.fmean(abs(r["contrast"]) for r in part
                                                if r["group"] == g)
                       for g in expected_groups},
            rows=len(part))
    return panel


def decide_v2(step0, chosen, finite_gradients=True, frozen_bitwise=True,
              horizons=HORIZONS):
    """PASS = the chosen step strictly beats step 0 at BOTH horizons."""
    improvement = {h: step0[h] - chosen[h] for h in horizons}
    passed = bool(finite_gradients and frozen_bitwise and
                  all(improvement[h] > 0 for h in horizons))
    return dict(baseline=step0, chosen=chosen, improvement=improvement,
                finite_gradients=finite_gradients, frozen_bitwise=frozen_bitwise,
                noise_sd=0.0,
                noise_note="deterministic CRN rollouts; r1==r0 measured exactly on this "
                           "code path (PILOT.json v1, 2026-10-31 11:52)",
                passed=passed)


class Panels:
    """gpuR-native panel subprocess driver (evaluation never uses smoothing)."""

    def __init__(self, out, argv_common, memory_fraction, device="cuda"):
        self.out = out
        self.argv_common = argv_common
        self.memory_fraction = memory_fraction
        self.device = device
        self.running = {}

    def launch(self, name, candidate, groups):
        target = self.out / f"SHARD_{name}.json"
        if target.exists():
            raise FileExistsError("shard receipt exists: " + target.name)
        cmd = [sys.executable, "-m", "tools.wmfixmerge.panel_eval_v2",
               *self.argv_common, "--candidate", str(candidate),
               "--out", str(self.out), "--name", name,
               "--groups", ",".join(str(g) for g in groups),
               "--device", self.device]
        if self.memory_fraction:
            cmd += ["--memory-fraction", str(self.memory_fraction)]
        log = (self.out / f"SHARD_{name}.log").open("w")
        self.running[name] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                               env=os.environ, cwd=os.getcwd()), log)
        return name

    def poll(self, name):
        proc, log = self.running.get(name, (None, None))
        if proc is None:
            return None
        code = proc.poll()
        if code is None:
            return None
        log.close()
        del self.running[name]
        return code

    def wait(self, name):
        proc, log = self.running[name]
        code = proc.wait()
        log.close()
        del self.running[name]
        return code

    def read(self, name):
        return json.loads(safe(self.out / f"SHARD_{name}.json").read_text())


def mini_values(shard):
    return {h: statistics.fmean(abs(r["contrast"]) for r in shard["rows"] if r["horizon"] == h)
            for h in HORIZONS}


def harvest(panels, mini_completed, block=False):
    """Collect finished mini panels (fail-closed on a nonzero exit)."""
    for name in [n for n in list(panels.running) if n.startswith("mini_")]:
        code = panels.wait(name) if block else panels.poll(name)
        if code is None:
            continue
        if code != 0:
            raise RuntimeError(f"mini panel subprocess failed: {name} exit {code}")
        shard = panels.read(name)
        mini_completed.append((shard["step"], mini_values(shard)))


def check_feasibility(path, prepared):
    doc = json.loads(safe(path).read_text())
    if doc.get("mode") != "real_train" or doc.get("status") != "PASS" \
            or doc.get("launch_ready") is not True:
        raise ValueError("smooth-events real TRAIN feasibility is not PASS")
    if doc.get("smooth_events") is not True or doc.get("finite_difference_ok") is not True:
        raise ValueError("feasibility does not bind a passing FD check on the smoothed objective")
    if doc.get("prepared_sha256") != digest(prepared):
        raise ValueError("feasibility binds a different prepared doc")
    if doc.get("implementation_hashes") != implementation_hashes():
        raise ValueError("implementation changed after the feasibility receipt")
    if doc.get("checkpoint_sha256") != WM2_SHA:
        raise ValueError("feasibility binds a different base checkpoint")
    return doc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared", "config", "checkpoint", "controllers", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--feasibility")
    parser.add_argument("--registered-data-pilot", action="store_true",
                        help="MAIN 15:25 data-only route: amended 30-step panel gate, smoothing OFF")
    parser.add_argument("--harness-contract")
    parser.add_argument("--data-recovery", help="explicit hash-bound infrastructure recovery receipt")
    parser.add_argument("--data-recovery-sha256")
    parser.add_argument("--schedule", default="tools/wmfixmerge/train_config.json")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--memory-fraction", type=float, default=None)
    parser.add_argument("--smooth-events", action="store_true",
                        help="training-only smoothed admission (required for the registered v2)")
    args = parser.parse_args(argv)
    install_guard()
    if args.registered_data_pilot:
        from tools.wmfixmerge.data_pilot import check_binding, run
        check_binding(args)
        return run(args)
    if not args.smooth_events or not args.feasibility:
        parser.error("the registered PILOT v2 runs with --smooth-events (training only)")
    schedule = json.loads(safe(args.schedule).read_text())
    if schedule["max_seconds"] > 21600 or schedule["stages"] != [[60, 30], [120, 30], [300, 60]] \
            or schedule["checkpoint_every_updates"] != 1:
        raise ValueError("schedule differs from registered six-hour curriculum")
    if digest(args.checkpoint) != WM2_SHA:
        raise ValueError("base is not registered WM2")
    feasibility = check_feasibility(args.feasibility, args.prepared)
    out = safe(args.out)
    if not out.is_dir() or any(out.iterdir()):
        raise ValueError("out must be an existing empty directory; never duplicate/resume by guess")
    start = time.monotonic()
    with (out / "RUN.json").open("x") as stream:
        json.dump(dict(pid=os.getpid(), started=time.time(), device=args.device,
                       pilot="S-WMFIX-PILOT v2, MAIN 13:50 TRACK 1 (owner go)",
                       schedule_sha256=digest(args.schedule),
                       feasibility_sha256=digest(args.feasibility)), stream)
    import torch
    torch.set_num_threads(2)
    torch.manual_seed(schedule["seed"])
    if args.device == "cuda" and args.memory_fraction:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    from tools.wmfixmerge.runtime import Runtime
    from tools.wmfixmerge.train import export
    from wms.model.merge_response import freeze_except_response
    from wms.train.merge_response import SMOOTH_GRADIENT_COLUMNS, objective_terms, paired_objective
    rt = Runtime(args.prepared, args.config, args.checkpoint, args.controllers, args.device,
                 smooth_events=True)
    n_groups = len(rt.source.groups)
    labels_by_seed = {}
    for row in rt.doc["labels"]:
        labels_by_seed.setdefault(int(row["seed"]), row["pair_key"])
    pair_to_group = {g["pair_key"]: i for i, g in enumerate(rt.source.groups)}
    mini_groups = []
    for seed in MINI_SEEDS:
        key = labels_by_seed.get(seed)
        if key is None or pair_to_group.get(key) is None:
            raise ValueError(f"mini seed {seed} not in the combined corpus")
        mini_groups.append(pair_to_group[key])
    all_groups = list(range(n_groups))
    shard_bounds = [round(i * n_groups / FULL_SHARDS) for i in range(FULL_SHARDS + 1)]
    shard_map = [all_groups[shard_bounds[i]:shard_bounds[i + 1]] for i in range(FULL_SHARDS)]
    if any(not s for s in shard_map):
        raise ValueError("empty full-panel shard")
    names = freeze_except_response(rt.model)
    optimizer = torch.optim.Adam([p for p in rt.model.parameters() if p.requires_grad],
                                 lr=schedule["lr"])
    original = torch.load(safe(args.checkpoint), map_location="cpu", weights_only=True)
    frozen = {n: v.detach().cpu().clone() for n, v in rt.model.state_dict().items()
              if not n.startswith("merge_response.")}
    from dataclasses import asdict
    receipt = dict(optimizer_names=names, prepared_sha256=digest(args.prepared),
                   feasibility_sha256=digest(args.feasibility), base_sha256=WM2_SHA,
                   implementation_hashes=implementation_hashes(),
                   authorization="MAIN ADDENDUM 13:50 TRACK 1 (owner go); two registered "
                                 "changes: 52-world combined TRAIN corpus + training-only "
                                 "smoothed admission",
                   flags=dict(merge_response=True, conserve_queue=True, learned_admission=True,
                              smooth_events=True, merge_bases=0, ramp_bases=0,
                              gradient_columns=list(SMOOTH_GRADIENT_COLUMNS)),
                   mini=dict(seeds=list(MINI_SEEDS), groups=mini_groups, every=MINI_EVERY,
                             arms=list(ARMS), horizons=list(HORIZONS)),
                   full_panel=dict(groups=n_groups, arms=list(ARMS), horizons=list(HORIZONS),
                                   shards=[list(s) for s in shard_map]))
    resolved = asdict(rt.config)
    resolved["model"]["merge_response"] = True
    with (out / "candidate_config.json").open("x") as stream:
        json.dump(resolved, stream, indent=2)
    step = 0
    latest = export(rt.model, original, frozen, optimizer, out, step, receipt)
    with (out / "PROTOCOL.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)

    panels = Panels(out, ["--prepared", args.prepared, "--config", args.config,
                          "--checkpoint", args.checkpoint, "--controllers", args.controllers],
                    PANEL_MEMORY_FRACTION if args.device == "cuda" else None, args.device)
    for i in range(FULL_SHARDS):
        panels.launch(f"step0_full_s{i}", latest, shard_map[i])

    mini_completed = []
    finite_steps = 0
    cap_stop = False
    early_stop = False
    timings = []
    try:
        for step in range(1, PILOT_STEPS + 1):
            reserve = max(timings, default=300.)
            if time.monotonic() - start + reserve >= schedule["max_seconds"]:
                cap_stop = True
                break
            best, streak = mini_update(mini_completed, None)
            if len(mini_completed) >= 2 and streak >= 2:
                early_stop = True
                break
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            predicted, recorded, refs, roll_receipt = rt.rollout(
                (step - 1) % n_groups, schedule["stages"][0][0],
                arm_index=(step - 1) // n_groups)
            loss = paired_objective(predicted, recorded, refs, rt.scales,
                                    anchor_scales=rt.anchor_scales,
                                    gradient_columns=SMOOTH_GRADIENT_COLUMNS)
            terms = objective_terms(predicted, recorded, refs, rt.scales,
                                    anchor_scales=rt.anchor_scales,
                                    gradient_columns=SMOOTH_GRADIENT_COLUMNS)
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
            # Harvest finished mini panels without blocking (full step-0 shards
            # are never polled here; full_panel() waits on them explicitly).
            harvest(panels, mini_completed, block=False)
            if step % MINI_EVERY == 0:
                while sum(n.startswith("mini_") for n in panels.running) >= MINI_INFLIGHT:
                    time.sleep(2.)
                    harvest(panels, mini_completed, block=False)
                panels.launch(f"mini_step{step:06d}", latest, mini_groups)
        # Drain outstanding mini panels (and step-0 full shards keep running).
        harvest(panels, mini_completed, block=True)
        mini_completed.sort(key=lambda item: item[0])
        with (out / "MINI_PANELS.json").open("x") as stream:
            json.dump([dict(step=s, values={str(h): v[h] for h in HORIZONS},
                            pooled=pooled(v)) for s, v in mini_completed],
                      stream, indent=2, allow_nan=False)
        chosen_step = choose_step(mini_completed)
        chosen_checkpoint = out / f"step_{chosen_step:06d}.pt"

        def full_panel(tag, base):
            for i in range(FULL_SHARDS):
                name = f"{tag}_s{i}"
                if name not in panels.running and not (out / f"SHARD_{name}.json").exists():
                    panels.launch(name, base, shard_map[i])
            for i in range(FULL_SHARDS):
                name = f"{tag}_s{i}"
                if name in panels.running and panels.wait(name) != 0:
                    raise RuntimeError(f"full panel shard failed: {name}")
            shard_rows = [panels.read(f"{tag}_s{i}")["rows"] for i in range(FULL_SHARDS)]
            return assemble(shard_rows, all_groups)

        step0_panel = full_panel("step0_full", out / "step_000000.pt")
        chosen_panel = full_panel(f"chosen{chosen_step:06d}", chosen_checkpoint)
        with (out / "FULL_PANEL_STEP0.json").open("x") as stream:
            json.dump(step0_panel, stream, indent=2, allow_nan=False)
        with (out / f"FULL_PANEL_STEP{chosen_step}.json").open("x") as stream:
            json.dump(chosen_panel, stream, indent=2, allow_nan=False)
        frozen_ok = all(torch.equal(v, rt.model.state_dict()[n].detach().cpu())
                        for n, v in frozen.items())
        decision = decide_v2({h: step0_panel[str(h)]["paired_J_MAE"] for h in HORIZONS},
                             {h: chosen_panel[str(h)]["paired_J_MAE"] for h in HORIZONS},
                             finite_gradients=finite_steps > 0, frozen_bitwise=frozen_ok)
        decision.update(protocol=receipt, chosen_step=chosen_step,
                        chosen_checkpoint=str(chosen_checkpoint),
                        chosen_sha256=digest(chosen_checkpoint),
                        mini_panels=[dict(step=s, pooled=pooled(v)) for s, v in mini_completed],
                        completed_steps=finite_steps, cap_stop=cap_stop, early_stop=early_stop,
                        n_groups=n_groups, elapsed_s=time.monotonic() - start)
        with (out / "PILOT_V2.json").open("x") as stream:
            json.dump(decision, stream, indent=2, allow_nan=False)
        print(json.dumps(decision, indent=2, allow_nan=False), flush=True)
        if not decision["passed"]:
            return 2
    except (RuntimeError, ValueError, OSError) as exc:
        with (out / "PILOT_V2.json").open("x") as stream:
            json.dump(dict(passed=False, status="FAIL",
                           blocker=f"{type(exc).__name__}: {exc}",
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
                step % n_groups, horizon, arm_index=step // n_groups)
            loss = paired_objective(predicted, recorded, refs, rt.scales,
                                    anchor_scales=rt.anchor_scales,
                                    gradient_columns=SMOOTH_GRADIENT_COLUMNS)
            terms = objective_terms(predicted, recorded, refs, rt.scales,
                                    anchor_scales=rt.anchor_scales,
                                    gradient_columns=SMOOTH_GRADIENT_COLUMNS)
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
                       pilot_passed=True, pilot="v2"), stream, indent=2, allow_nan=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
