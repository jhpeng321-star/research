"""WMFIX candidate-side four-arm SPEEDHEAD quartet (B producer GPU side).

Runs the frozen base protocol (SPEEDHEAD wm_gpu/wm_gpu_host, base receipt
wm_gpu_host_0230) with the trained WMFIX candidate: the WM checkpoint carries
the merge_response head (loaded from its candidate config), and the entry
process runs conserve_queue + learned admission. Worlds, exact t300 starts,
futures plans, lane draws, entry-slot RNG and the four CAV controllers are the
frozen base ones; only the WM and its entry path differ. Every task writes an
immutable JSON receipt; provenance must equal the base WM_PREPARED provenance.

Staged on the GPU host next to the SPEEDHEAD deployment; never a verdict
substitute by itself: tools/wmfixmerge/producer_b.py reduces this output with
the base tasks and the recorded real values into the wmfix_speedhead/1 panel.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

OWN = Path("/root/autodl-tmp/c4/speedhead20261030")
FIX = Path("/root/autodl-tmp/c4/setupaudit20261030/fix")
TREE = Path(__file__).resolve().parents[2]
OUT = OWN / "wmfix_candidate_quartet"
ARMS = ("ctrl0", "mappo", "pi", "circles")
ORDER = (7, 4, 3, 5, 0, 1, 2, 6)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_frozen():
    """Import the frozen wm_worker (and its FIX runtime) under a private name."""
    spec = importlib.util.spec_from_file_location("wmfix_frozen_worker",
                                                  OWN / "tools/speedhead/wm_worker.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def switch_to_candidate_tree():
    """After the frozen helpers are imported, resolve wms.* from the WMFIX tree.

    The frozen modules already hold their own class/module references; purging
    the cached FIX wms package makes every later import (model, entries,
    admission, mix) come from the candidate implementation.
    """
    sys.path[:0] = [str(TREE / "src"), str(TREE), str(TREE / "tools")]
    for name in [k for k in list(sys.modules) if k == "wms" or k.startswith("wms.")]:
        del sys.modules[name]


def move(value, device):
    import torch
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: move(v, device) for k, v in value.items()}
    return value


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default=str(OUT), help="pre-created candidate-specific output directory")
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()
    OUT = Path(args.out)
    for label, path in (("candidate", args.candidate), ("config", args.config)):
        if not Path(path).is_file():
            raise SystemExit("missing " + label + ": " + path)
    if not OUT.is_dir():
        raise SystemExit("output directory must be pre-created: " + str(OUT))
    lock = (OUT / "WORKER.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import torch
    torch.cuda.set_per_process_memory_fraction(8 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    frozen = load_frozen()
    expected = json.loads((OWN / "WM_PREPARED.json").read_text())
    switch_to_candidate_tree()

    from tools.wmcombined.e3_score import load_for_scoring
    _, _, model = load_for_scoring(args.config, args.candidate)
    if getattr(model, "merge_response", None) is None:
        raise SystemExit("candidate model has no merge_response head")
    payload = torch.load(frozen.CANDIDATE, map_location="cpu", weights_only=False)
    quartet_cfg = replace(frozen.DreamerConfig(**payload["config"]), horizon=600, entrant_learned_mix=True)
    model.cav_exec_cap = quartet_cfg.cav_exec_cap
    model = model.to("cuda").eval()

    # The frozen four CAV controllers, exactly as the base quartet built them.
    # wm_gpu imports wm_worker by its real name; register the already-loaded
    # private instance so it is not executed twice. Loading wm_gpu may cache
    # tree-side wms modules; the candidate-first order is re-asserted after.
    sys.modules["wm_worker"] = frozen
    frozen_controllers = importlib.util.spec_from_file_location(
        "wmfix_quartet_controllers", OWN / "tools/speedhead/wm_gpu.py")
    gpu = importlib.util.module_from_spec(frozen_controllers)
    sys.modules[frozen_controllers.name] = gpu
    frozen_controllers.loader.exec_module(gpu)
    switch_to_candidate_tree()

    cp, ce, _, _ = frozen.load_policy_components(frozen.CANDIDATE, device="cuda")
    torch.nn.init.zeros_(cp.residual_head.weight)
    torch.nn.init.zeros_(cp.residual_head.bias)
    cp.eval()
    ce = ce.encoder if isinstance(ce, frozen.GroupSignal) else ce
    ce.eval()
    binding = json.loads((OWN / "EXPORT_BOUND.json").read_text())
    if digest(Path(binding["path"])) != binding["sha256"]:
        raise SystemExit("MAPPO export changed")
    mp = frozen.load_policy_fn(binding["path"], device="cuda")
    mp.policy.eval()
    mp.group_encoder.eval()
    if not mp.native_lanes or mp.has_lane_head:
        raise SystemExit("unexpected MAPPO lane head")
    from wms.sim_bridge import load_controller_dependencies
    deps = load_controller_dependencies(OWN / "e3final/runtime")

    torch.manual_seed(965140)
    batches, provenance = frozen.make_starts(model)
    if provenance != expected["provenance"]:
        raise SystemExit("start provenance differs from the frozen base panel")

    from wms.control import dreamer as candidate_dreamer
    from wms.control.entrant_mix import make_sampler
    from wms.model.learned_admission import LearnedAdmission

    identity = dict(schema="wmfix_candidate_quartet/1", K=8, horizon=600,
                    candidate_sha256=digest(args.candidate), candidate_config_sha256=digest(args.config),
                    base_wm_sha256=expected["wm_sha256"], provenance=provenance,
                    flags=dict(merge_response=True, conserve_queue=True, learned_admission=True),
                    arms=list(ARMS), seed=965140, device="cuda", memory_limit_gib=8,
                    frozen_worker_sha256=digest(OWN / "tools/speedhead/wm_worker.py"),
                    frozen_gpu_sha256=digest(OWN / "tools/speedhead/wm_gpu.py"))
    if (OUT / "IDENTITY.json").exists():
        if json.loads((OUT / "IDENTITY.json").read_text()) != identity:
            raise SystemExit("IDENTITY changed; reconcile the existing output directory")
    else:
        with (OUT / "IDENTITY.json").open("x") as stream:
            json.dump(identity, stream, indent=1)

    native = frozen.native_fn("head", a_max=1.2)
    for bi in ORDER:
        for fi in range(8):
            target = OUT / (f"canary_{bi}_{fi}.json" if args.canary else f"w{bi}_k{fi}.json")
            if target.exists():
                continue
            if (OUT / "STOP").exists():
                raise SystemExit("cooperative stop")
            begin = time.monotonic()
            torch.manual_seed(965140)
            b = frozen.with_entry_slots(copy.deepcopy(batches[bi]), 600, 1920,
                torch.Generator().manual_seed(1_000_003 * 191), queue_state=True)
            b.state = move(b.state, "cuda")
            b.rollout_kwargs = move(b.rollout_kwargs, "cuda")
            plan = frozen.SampledFutures(frozen.FuturesConfig.from_trainer(quartet_cfg), 190).plan(
                [model], b.state, b.rollout_kwargs, 1200)[0]
            initial = frozen.expand(b.state, 4)
            frozen.pair_merge_streams(initial, torch.zeros(4, dtype=torch.long, device="cuda"), 1, 190 + fi)
            initial["exit_s_m"] = torch.full_like(initial["s_m"], frozen.road_end_exit_bound(model.physics))
            kw = frozen.expand({**b.rollout_kwargs, **plan[fi]}, 4)
            lanes = frozen.lane_uniform_draws([model], 1, 1200, initial["s_m"].shape[1],
                190 + 1000003 * fi if fi else 190, device="cuda")
            if lanes is not None:
                kw["lane_uniform"] = frozen.expand(lanes, 4)
            process = candidate_dreamer._entry_process(b.entry_spec, [0] * 4,
                initial["s_m"].shape[1], quartet_cfg.entry_insert_rule, None,
                couple_draws=True, sample_index=fi)
            mix = make_sampler(model, initial, kw, 190 + 1000003 * fi, [0] * 4)
            process.cfg = replace(process.cfg, conserve_queue=True,
                                  learned_admission=LearnedAdmission(mix))
            entries = frozen.Entries(process)
            kw["entries"] = entries
            kw["entrant_driver_sampler"] = mix
            cb = gpu.controllers(model, cp, ce, mp, native, initial["s_m"].shape[1],
                                 (deps.config, deps.circles_type, deps.direct_target, deps.neighbor_type,
                                  deps.frame_type, deps.pi_type, deps.u_report_type, deps.pi_params_type))
            with torch.no_grad():
                roll = model.rollout(initial, n_steps=10 if args.canary else 1200, cav_controller=cb, **kw)
                if not torch.isfinite(roll["speeds"]).all():
                    raise SystemExit("nonfinite rollout")
                score = {} if args.canary else frozen.costs(roll, entries, b.entry_spec["k0"])
            if not args.canary:
                import numpy as np
                if not all(np.isfinite(score[h][k]).all() for h in ("120", "300", "600")
                           for k in ("J", "real_J")):
                    raise SystemExit("nonfinite costs")
            task = dict(start=bi, future=fi, arms=list(ARMS), metadata=b.metadata[0], costs=score,
                        seconds=time.monotonic() - begin,
                        mix_assigned=kw["entrant_driver_sampler"].assigned,
                        peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                        peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                        slots=initial["s_m"].shape[1], steps=10 if args.canary else 1200)
            with target.open("x") as stream:
                json.dump(task, stream, allow_nan=False, indent=1)
                stream.flush()
                os.fsync(stream.fileno())
            print("DONE", bi, fi, json.dumps({k: task[k] for k in ("seconds", "peak_reserved_mib")}), flush=True)
            del roll, entries, process, kw, cb, b, initial, plan, mix
            gc.collect()
            torch.cuda.empty_cache()
            if args.canary:
                return
    with (OUT / "COMPLETE.json").open("x") as stream:
        json.dump(dict(tasks=64, arms=list(ARMS),
                       hashes={f"w{b}_k{k}.json": digest(OUT / f"w{b}_k{k}.json")
                               for b in range(8) for k in range(8)},
                       identity=identity), stream, indent=1)


if __name__ == "__main__":
    main()
