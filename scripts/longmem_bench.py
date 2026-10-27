"""Single-thread CPU-only imagination benchmark; each point has a fresh process.

Run with PYTHONPATH=src python scripts/longmem_bench.py. Synthetic starts and
nonlinear frozen WM only; no dataset, simulator, GPU, or loop entrypoint.
"""
import argparse
from contextlib import nullcontext
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def measure(args):
    import torch
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("longmem_fixture", root / "tests/unit/test_longmem.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    trainer = fixture.trainer(args.micro, args.segment, args.offload, horizon=args.horizon,
                              batch=args.batch or (1 if args.real_wm else 8),
                              slots=args.slots or (4 if args.real_wm else 8),
                              width=args.width, real_world=args.real_wm)
    # Exclude lazy optimizer/import initialization from the incremental update peak.
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    started = time.perf_counter()
    context = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                      profile_memory=True) if args.tensor_memory else nullcontext())
    with context as profile:
        result = trainer.update()
    seconds = time.perf_counter() - started
    tensor_peak = None
    if args.tensor_memory:
        live = high = 0
        for event in sorted(profile.profiler.kineto_results.events(), key=lambda e: e.start_ns()):
            if event.name() == "[memory]":
                live += event.nbytes()
                high = max(high, live)
        tensor_peak = round(high / 2**20, 4)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(json.dumps(dict(horizon_s=args.horizon, micro=args.micro, segment=args.segment,
                          offload=args.offload, peak_rss_mib=round(peak, 2),
                          incremental_peak_mib=round(peak-rss_before, 2),
                          seconds=round(seconds, 4), actor_loss=result["actor_loss"],
                          torch=torch.__version__, threads=1, real_wm=args.real_wm,
                          tensor_peak_mib=tensor_peak, batch=trainer.config.batch_size,
                          slots=args.slots or (4 if args.real_wm else 8), width=args.width)), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--horizon", type=int)
    ap.add_argument("--micro", type=int, default=0)
    ap.add_argument("--segment", type=int, default=0)
    ap.add_argument("--offload", action="store_true")
    ap.add_argument("--real-wm", action="store_true")
    ap.add_argument("--tensor-memory", action="store_true")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--slots", type=int)
    ap.add_argument("--width", type=int, default=256)
    args = ap.parse_args()
    if args.horizon:
        measure(args)
        return
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for horizon in (30, 60, 120, 300):
        settings = (((0, 0), (1, 8), (1, 16)) if args.real_wm or args.batch == 1 else
                    ((0, 0), (2, 0), (2, 8), (1, 8)))
        for micro, segment in settings:
            cmd = [sys.executable, __file__, "--horizon", str(horizon), "--micro", str(micro),
                   "--segment", str(segment)]
            if micro:
                cmd.append("--offload")
            if args.real_wm:
                cmd.append("--real-wm")
            if args.tensor_memory:
                cmd.append("--tensor-memory")
            for key in ("batch", "slots", "width"):
                if getattr(args, key) is not None:
                    cmd.extend(["--" + key, str(getattr(args, key))])
            subprocess.run(cmd, env=env, check=True)


if __name__ == "__main__":
    main()
