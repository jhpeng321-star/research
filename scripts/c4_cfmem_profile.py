"""Peak tensor memory / time of one C4 Dreamer update on CPU (CFMEM_NOTES.md).

Peak = the maximum, during one ``trainer.update()``, of the live bytes of every CPU tensor
storage CREATED inside the update (``--method dispatch``, default): a ``TorchDispatchMode``
registers each new output storage of every ATen op (forward, checkpoint recomputation and
backward) and a ``weakref.finalize`` on it (PyTorch preserves the Python storage object while
the StorageImpl lives), so freed storages are subtracted when they die.  This is the CPU
analogue of CUDA ``max_memory_allocated() - memory_allocated()`` at entry (no caching-allocator
rounding).  ``--method profiler`` instead sums ``torch.profiler`` allocator events (same
numbers on small runs; too memory-hungry for large runs).  ``--no-profile`` times only.

    python scripts/c4_cfmem_profile.py --wm-ckpt PATH --batch 8 --horizon 15 --max-starts 2         --n-group 1 --n-flip 1 [--checkpoint-imagination on|off] [--updates 1]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class LiveBytes(TorchDispatchMode):
    """Running and peak bytes of CPU storages created by ops inside the mode."""

    def __init__(self):
        super().__init__()
        self.live, self.current, self.peak = {}, 0, 0

    def _free(self, key):
        self.current -= self.live.pop(key, 0)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for tensor in tree_flatten(out)[0]:
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu":
                storage = tensor.untyped_storage()
                key = storage.data_ptr()
                if key and key not in self.live:
                    self.live[key] = storage.nbytes()
                    self.current += storage.nbytes()
                    self.peak = max(self.peak, self.current)
                    weakref.finalize(storage, self._free, key)
        return out


def peak_of(prof):
    # Self (exclusive) allocator bytes of every op plus the free-standing [memory] events.
    events = [(e.time_range.start, e.self_cpu_memory_usage) for e in prof.events()
              if e.self_cpu_memory_usage]
    events.sort(key=lambda item: item[0])
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/c4_policy.toml")
    p.add_argument("--wm-ckpt", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--max-starts", type=int, default=2)
    p.add_argument("--n-group", type=int, default=None)
    p.add_argument("--n-flip", type=int, default=None)
    p.add_argument("--checkpoint-imagination", choices=("on", "off"), default=None)
    p.add_argument("--no-counterfactual", action="store_true")
    p.add_argument("--updates", type=int, default=2, help="measured updates (after 1 warm-up)")
    p.add_argument("--no-profile", action="store_true", help="time only")
    p.add_argument("--method", choices=("dispatch", "profiler"), default="dispatch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    args = p.parse_args(argv)

    from wms.control.contract import RewardWeights
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.dreamer import DreamerConfig, DreamerTrainer
    from wms.control.lanes import LaneRules
    from wms.control import train_cli as cli

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    raw = cli._read_config(args.config)
    trainer_cfg = dict(raw.get("trainer", {}))
    if args.checkpoint_imagination is not None:
        trainer_cfg["checkpoint_imagination"] = args.checkpoint_imagination == "on"
    cfg = replace(DreamerConfig(**trainer_cfg), batch_size=args.batch, horizon=args.horizon,
                  checkpoint_every=0, validation_every=0)
    cf_raw = dict(raw.get("counterfactual", {}))
    cf_raw["max_starts"] = args.max_starts
    if args.n_group is not None:
        cf_raw["n_group"] = args.n_group
    if args.n_flip is not None:
        cf_raw["n_flip"] = args.n_flip
    if args.no_counterfactual:
        cf_raw["enabled"] = False
    counterfactual = CounterfactualConfig(**cf_raw)
    lane_section = dict(raw.get("lanes", {}))
    lane_rules = LaneRules.from_mapping(lane_section) if lane_section.pop("enabled", True) else None
    components = cli._load_components()
    models = cli._load_world_models([args.wm_ckpt], "cpu")
    train, _ = cli._load_starts(raw.get("data", {}), models, 0)
    trainer = DreamerTrainer(models, components.pop("policy"), components.pop("group_encoder"),
                             components.pop("critic"), train, **components, config=cfg,
                             reward_weights=RewardWeights(**raw.get("reward", {})), device="cpu",
                             lane_rules=lane_rules, counterfactual=counterfactual)
    trainer.update_count = 1500     # full cooperation curriculum (all row kinds weighted)
    trainer.update()                 # warm-up (lazy init, caches)
    results = []
    for _ in range(args.updates):
        if args.no_profile:
            started = time.perf_counter()
            metrics = trainer.update()
            results.append({"s_per_update": time.perf_counter() - started,
                            "rows": metrics.get("cf_rows", 0) + args.batch})
            continue
        if args.method == "dispatch":
            tracker = LiveBytes()
            with tracker:
                metrics = trainer.update()
            peak = tracker.peak
        else:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                        profile_memory=True) as prof:
                metrics = trainer.update()
            peak = peak_of(prof)
        results.append({"peak_bytes": peak, "rows": metrics.get("cf_rows", 0) + args.batch,
                        "wall_time_s_profiled": metrics["wall_time_s"]})
    k = int(train.sample(1).state["s_m"].shape[-1])
    print(json.dumps({"args": vars(args), "slots_K": k, "results": results}))


if __name__ == "__main__":
    main()
