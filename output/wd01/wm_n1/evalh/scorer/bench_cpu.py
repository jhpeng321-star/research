"""CPU compute probe for the WM-N1 CAL-tune scorer (ANALYSIS ONLY, no data).

Builds a RANDOM-INIT Idea A member-1 model (``train.idea_a`` + ``model``
defaults: ``k_max=192``, A6 ``state_global_token`` ON) and a GENERATED
synthetic ``K=192`` world, then times, at torch default threads and at one
thread:

* one Idea A ``d1`` rollout of 20 steps (the registered 10 s correction
  window) and one of 1140 steps (570 s);
* the P5 correction head on each rollout (the summary window is sliced back
  to 20 steps, so the head cost is horizon-independent);
* the Part A ``v2g`` baseline rollout of the same lengths.

It NEVER loads the real checkpoint, the real pack, or any TRAIN/CAL data:
every tensor is generated in-process.  The timings feed the E5 registration
compute section only; they are not efficacy evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
WM_N1 = os.path.dirname(os.path.dirname(HERE))
if WM_N1 not in sys.path:
    sys.path.insert(0, WM_N1)

#: registered training/correction summary window (model.constants).
CORRECTION_WINDOW_STEPS = 20
#: 570 s at dt 0.5 s (the longest registered tau).
HORIZON_570_STEPS = 1140


def build_random_member(seed: int = 101):
    """Random-init Idea A member-1 bundle at the registered architecture.

    Mirrors ``wm_api.checkpoint._build_member``: the learned HDV / correction /
    state-token modules are cast to float64 (the ``rollout2`` engine's internal
    math dtype); the arrival head stays float32 (arrivals are off here).
    """
    from train.idea_a import IdeaATrainConfig, build_idea_a_bundle

    cfg = IdeaATrainConfig(member=1, k_max=192, state_global_token=True)
    torch.manual_seed(int(seed))
    bundle = build_idea_a_bundle(cfg)
    for name in ("hdv", "corr", "state_enc"):
        module = getattr(bundle, name, None)
        if isinstance(module, torch.nn.Module):
            module.to(torch.float64)
    return bundle, cfg


def make_world_batch(k: int = 192, seed: int = 7) -> dict:
    """A generated K-slot world batch; no file is read."""
    import torch.nn.functional as F

    from model import constants as C

    g = torch.Generator().manual_seed(int(seed))
    slots = torch.arange(k, dtype=torch.float32)
    lane_idx = slots.long() % C.N_LANES
    one_hot = F.one_hot(lane_idx, C.N_LANES).float()
    s0 = (slots * 20.0).unsqueeze(0)
    v0 = (22.0 - 0.01 * slots).clamp(min=5.0).unsqueeze(0)
    a0 = (0.2 * torch.sin(slots * 0.7)).unsqueeze(0)
    length0 = torch.full((1, k), float(C.NOMINAL_LENGTH_M))
    kin = torch.stack([s0,
                       one_hot[..., 0].unsqueeze(0),
                       one_hot[..., 1].unsqueeze(0),
                       one_hot[..., 2].unsqueeze(0),
                       v0, a0, length0], dim=-1)
    is_cav = torch.zeros(1, k, dtype=torch.bool)
    is_cav[:, :max(1, k // 16)] = True
    present = torch.ones(1, k, dtype=torch.bool)
    counter = torch.randint(0, 4, (1, C.N_ENTRANCES, C.COUNTER_BINS),
                            generator=g)
    return {
        "veh_kin": kin,
        "veh_mask": present,
        "is_cav": is_cav,
        "wall_group": torch.full((1, k), -1, dtype=torch.long),
        "slot_id": torch.arange(k).unsqueeze(0).contiguous(),
        "counter": counter,
        "integrator": "d1",
        "residual_enabled": True,
        "arrivals_enabled": False,
        "t0_s": 0.0,
        "params": None,
    }


def _synthetic_rows(n: int = 1):
    from evalh.protocol import OutcomeRow
    from model import constants as C

    return [OutcomeRow(world="SYNTH", t0_s=0.0, decision_t_s=0.0,
                       tau_s=float(C.TAUS[0]), policy="native",
                       y_delay_s=0.0) for _ in range(int(n))]


def _time_capture(fn, reps: int, warmup: int = 1):
    """Time ``fn``; return ``(stats, last_output)`` (output reused)."""
    times = []
    last = None
    for r in range(int(reps) + int(warmup)):
        t0 = time.perf_counter()
        last = fn()
        dt = time.perf_counter() - t0
        if r >= int(warmup):
            times.append(dt)
    if not times:
        times = [0.0]
    stats = {"median_s": float(statistics.median(times)),
             "min_s": float(min(times)), "max_s": float(max(times)),
             "n": len(times), "times_s": [float(x) for x in times]}
    return stats, last


def _time(fn, reps: int, warmup: int = 1) -> dict:
    return _time_capture(fn, reps, warmup)[0]


def _correction_fn(bundle, rollouts, rows):
    from evalh.idea_a import IdeaAVariantPredictor

    predictor = IdeaAVariantPredictor(
        bundle, rollout_fn=lambda r, **kw: rollouts)
    return lambda: predictor.predict_quantiles(rows)


def run_suite(bundle, batch, *, threads: int, reps_short: int, reps_long: int,
              steps_short: int = CORRECTION_WINDOW_STEPS,
              steps_long: int = HORIZON_570_STEPS) -> dict:
    from inc1eval.rollout_baseline import rollout_v2g_baseline
    from rollout2.engine import rollout2, world_from_batch

    torch.set_num_threads(int(threads))
    world = world_from_batch(batch)
    rows = _synthetic_rows(1)

    def d1(n):
        return rollout2(bundle, batch, n_steps=int(n), n_samples=1, seed=0)

    def v2g(n):
        return rollout_v2g_baseline(bundle, world, n_steps=int(n), seed=0)

    # the long rollout is measured once (its fixed cost dominates the warmup);
    # its output is reused for the correction-head timing so it runs only once.
    d1_short_stats, roll_short = _time_capture(
        lambda: d1(steps_short), reps_short, warmup=1)
    d1_long_stats, roll_long = _time_capture(
        lambda: d1(steps_long), reps_long, warmup=0)
    v2g_short_stats = _time(lambda: v2g(steps_short), reps_short, warmup=1)
    v2g_long_stats = _time(lambda: v2g(steps_long), reps_long, warmup=0)

    return {
        "threads": int(threads),
        "steps_short": int(steps_short),
        "steps_long": int(steps_long),
        "d1_rollout_short": d1_short_stats,
        "d1_rollout_long": d1_long_stats,
        "correction_head_short":
            _time(_correction_fn(bundle, roll_short, rows), reps_short,
                  warmup=1),
        "correction_head_long":
            _time(_correction_fn(bundle, roll_long, rows), reps_short,
                  warmup=1),
        "v2g_rollout_short": v2g_short_stats,
        "v2g_rollout_long": v2g_long_stats,
    }


def _n_params(bundle) -> int:
    total = 0
    for name in ("hdv", "arrival", "corr", "state_enc", "ctrl"):
        mod = getattr(bundle, name, None)
        if isinstance(mod, torch.nn.Module):
            total += sum(p.numel() for p in mod.parameters())
    return int(total)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, action="append", default=None,
                    help="repeatable; default torch default and 1")
    ap.add_argument("--reps-short", type=int, default=3)
    ap.add_argument("--reps-long", type=int, default=1)
    ap.add_argument("--steps-short", type=int, default=CORRECTION_WINDOW_STEPS)
    ap.add_argument("--steps-long", type=int, default=HORIZON_570_STEPS)
    ap.add_argument("--k", type=int, default=192)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    threads_list = args.threads or [torch.get_num_threads(), 1]
    print(json.dumps({"event": "start", "torch": torch.__version__,
                      "cuda": torch.cuda.is_available(),
                      "default_threads": torch.get_num_threads(),
                      "threads": threads_list, "k": args.k,
                      "steps_short": args.steps_short,
                      "steps_long": args.steps_long,
                      "reps_short": args.reps_short,
                      "reps_long": args.reps_long}, sort_keys=True),
          flush=True)

    bundle, cfg = build_random_member(args.seed)
    batch = make_world_batch(k=args.k, seed=args.seed + 1)
    n_params = _n_params(bundle)

    suites = []
    for threads in threads_list:
        rec = run_suite(bundle, batch, threads=threads,
                        reps_short=args.reps_short, reps_long=args.reps_long,
                        steps_short=args.steps_short,
                        steps_long=args.steps_long)
        suites.append(rec)
        print(json.dumps({"event": "suite", **rec}, sort_keys=True),
              flush=True)

    out = {
        "schema": "wm_n1_score_caltune_cpu_bench/1",
        "analysis_only": True,
        "random_init": True,
        "real_checkpoint_loaded": False,
        "real_data_read": False,
        "torch": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "default_threads": int(torch.get_num_threads()),
        "k": int(args.k),
        "state_global_token": bool(cfg.state_global_token),
        "k_max": int(cfg.k_max) if cfg.k_max is not None else None,
        "n_params": n_params,
        "suites": suites,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
    print(json.dumps({"event": "done", **out}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
