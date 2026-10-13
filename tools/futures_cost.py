"""CPU cost of sampled futures: one DreamerTrainer.update() with the real WM-S architecture
(random weights unless --wm-ckpt) on a synthetic start batch; prints JSON (time, peak working set).

    python tools/futures_cost.py --samples 1 [--accel-noise] [--no-checkpoint] [--ensemble 1]
Run each setting in its own process: the peak working set is per process.
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--accel-noise", action="store_true")
    ap.add_argument("--no-checkpoint", action="store_true")
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--slots", type=int, default=48)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--wm-ckpt")
    args = ap.parse_args()
    torch.set_num_threads(1)

    from wms.control.bridge import make_cav_controller
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.dreamer import DreamerConfig, DreamerTrainer
    from wms.control.lanes import LaneRules
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
    from wms.control.reward import team_reward
    from wms.control.safety import safe_accel
    from wms.control.start_states import StartBatch
    from wms.control.train_cli import GroupSignal

    if args.wm_ckpt:
        from wms.eval.backend import load_backend
        models = [load_backend(args.wm_ckpt, device="cpu").model for _ in range(args.ensemble)]
    else:
        from wms.model.spec import ModelSpec
        from wms.model.wms import WmSModel
        models = []
        for e in range(args.ensemble):
            torch.manual_seed(e)
            models.append(WmSModel(ModelSpec(rollout_mode="hybrid")))
    k = args.slots
    gen = torch.Generator().manual_seed(0)
    s = (-1500 + 30 * torch.arange(k).float()).unsqueeze(0)
    state = {"s_m": s, "v_mps": 18 + 4 * torch.rand(1, k, generator=gen),
             "a_mps2": torch.zeros(1, k), "lane": (torch.arange(k) % 3).unsqueeze(0),
             "present": torch.ones(1, k, dtype=torch.bool),
             "is_cav": (torch.arange(k) % 4 == 0).unsqueeze(0), "length_m": torch.full((1, k), 5.0)}

    class Starts:
        def sample(self, batch_size, device="cpu"):
            return StartBatch({key: v.expand(batch_size, -1).clone() for key, v in state.items()}, {})

    cfg = DreamerConfig(batch_size=args.batch, horizon=args.horizon, checkpoint_every=0, validation_every=0,
                        checkpoint_imagination=not args.no_checkpoint, future_samples=args.samples,
                        future_accel_noise=args.accel_noise)
    trainer = DreamerTrainer(
        models, CavPolicy(hidden_dim=32), GroupSignal(RsuGroupEncoder(hidden_dim=32)),
        TeamCritic(hidden_dim=32), Starts(), observation_builder=build_observations,
        reward_fn=team_reward, safety=safe_accel, controller_factory=make_cav_controller,
        config=cfg, lane_rules=LaneRules(),
        counterfactual=CounterfactualConfig(enabled=True, n_individual=2, n_group=1, n_flip=1,
                                            group_max_size=4))
    times = []
    for _ in range(args.repeats + 1):            # first update is a warm-up
        started = time.perf_counter()
        metrics = trainer.update()
        times.append(time.perf_counter() - started)
    import psutil
    mem = psutil.Process().memory_info()
    peak = getattr(mem, "peak_wset", mem.rss)
    print(json.dumps({"samples": args.samples, "accel_noise": args.accel_noise,
                      "checkpoint_imagination": not args.no_checkpoint, "ensemble": args.ensemble,
                      "update_s": min(times[1:]), "peak_mb": peak / 2 ** 20,
                      "rows": metrics.get("cf_rows"), "disagreement": metrics["ensemble_disagreement"]}))


if __name__ == "__main__":
    main()
