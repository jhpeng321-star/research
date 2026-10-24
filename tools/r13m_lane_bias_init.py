"""R13M init values of the per-lane constant temporal emission (CPU, no training, no SUMO).

The R13M variant keeps the MEAN of the learned temporal acceleration emission and removes its state/history-driven
fluctuation: the emission becomes a per-lane constant b[lane] (``model.temporal_accel_lane_bias``).  Its init is
measured, not set by hand.  The correction is gate(k) x (attention residual + emission), so the bias that
preserves the per-lane MEAN gated contribution of the emission is the gate-weighted mean

    b[lane] = sum(gate(k) * e) / sum(gate(k))

of the PRE-GATE temporal emission e (row 0 of the temporal MLP, the tensor ``temporal_accel_off`` / BRAKE_DIAG
``zerotemp`` zero) of the init checkpoint, per step-start lane, over the learned on-road HDV rows of free rollouts
from real TRAIN windows, with the run's init transforms applied (the trainer's ``init_from`` path).  The rollouts
run without gradient through the real unroll runner.  The unweighted mean is reported too.

    python tools/r13m_lane_bias_init.py --config configs/c4_wm_v10round13f.toml --set model.temporal_accel_off=false \
        --ckpt <init.pt> --roots <TRAIN roots> --out docs/r13m_lane_bias_init.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from torch.utils.data import ConcatDataset, DataLoader  # noqa: E402

from wms.cli.assemble import build_model, build_physics  # noqa: E402
from wms.config.load import load_train_config  # noqa: E402
from wms.data.c4_episodes import build_c4_datasets  # noqa: E402
from wms.data.collate import collate  # noqa: E402
from wms.physics.state import state_kin  # noqa: E402
from wms.train.trainer import WmSTrainer  # noqa: E402


class Accumulator:
    """Per-lane sums of the pre-gate emission and of the gated emission over learned on-road rows."""

    def __init__(self, model):
        self.model = model
        self.sum = torch.zeros(3, dtype=torch.float64)
        self.gated = torch.zeros(3, dtype=torch.float64)
        self.sq = torch.zeros(3, dtype=torch.float64)
        self.n = torch.zeros(3, dtype=torch.float64)
        self.gate_sum = torch.zeros(3, dtype=torch.float64)
        self._orig = model.forward

    def __call__(self, state, **kwargs):
        out = self._orig(state, **kwargs)
        with torch.no_grad():
            kin = state_kin(state).float()
            on_road = kin[..., 1:4].sum(-1) > 0
            lane = kin[..., 1:4].argmax(-1)
            learned = out["hdv_mask"].bool() & on_road
            emission = out["temporal_accel_pregate_mps2"].double()
            gate = self.model.gate(torch.as_tensor(kwargs.get("k_steps", 0))).double()
            gated = emission * (gate if gate.ndim else gate.expand_as(emission))
            for ln in range(3):
                m = learned & (lane == ln)
                self.sum[ln] += emission[m].sum()
                self.sq[ln] += emission[m].square().sum()
                self.gated[ln] += gated[m].sum()
                self.gate_sum[ln] += (gate if gate.ndim else gate.expand_as(emission))[m].sum()
                self.n[ln] += m.sum()
        return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/c4_wm_v10round13f.toml")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--roots", nargs="+", required=True)
    p.add_argument("--all-roles-as-train", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--n-batches", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--unroll-s", type=float, default=60.0)
    p.add_argument("--stride-s", type=float, default=30.0)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", action="append", default=[])
    args = p.parse_args(argv)
    cfg = load_train_config(ROOT / args.config, [f"model.init_from={args.ckpt}", "runtime.device=cpu",
                                                 f"runtime.threads={args.threads}", *args.set])
    if cfg.model.temporal_accel_off or cfg.model.temporal_accel_lane_bias:
        raise SystemExit("measure the DIRECT emission: run with temporal_accel_off / lane_bias false")
    torch.set_num_threads(args.threads)
    torch.manual_seed(cfg.runtime.seed)
    chunk = max(1, round(args.unroll_s / 0.5))
    datasets = []
    for root in args.roots:
        manifest = json.loads((Path(root) / "C4_EPISODES_MANIFEST.json").read_text(encoding="utf-8"))
        role_of = {e["world"]: "train" for e in manifest["episodes"]} if args.all_roles_as_train else None
        datasets.append(build_c4_datasets(
            root, horizon_steps=chunk, history_s=cfg.data.history_s, history_stride_s=cfg.data.history_stride_s,
            entries_v2=bool(cfg.data.entries_v2), stride_s=args.stride_s, junction_fix=cfg.data.junction_fix,
            command_known=True, merge_pivot_history=True, **({"role_of": role_of} if role_of else {}))["train"])
    physics = build_physics(cfg)
    out_dir = Path(tempfile.mkdtemp(prefix="r13m_init_", dir=Path(args.out).resolve().parent))
    trainer = WmSTrainer(cfg, build_model(cfg, physics), ConcatDataset(datasets), val_dataset=None, out_dir=out_dir)
    model = trainer.model
    model.diag_emit_temporal = True
    acc = Accumulator(model)
    model.forward = acc
    model.eval()
    loader = DataLoader(trainer.dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed))
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.n_batches:
                break
            trainer.forward_backward(batch, unroll_steps=chunk, backward=False, truncation_steps=chunk)
            print(f"batch {i}: n {acc.n.tolist()} mean {(acc.sum / acc.n.clamp_min(1)).tolist()}", flush=True)
    n = acc.n.clamp_min(1)
    mean = (acc.sum / n).tolist()
    sd = (acc.sq / n - (acc.sum / n).square()).clamp_min(0).sqrt().tolist()
    res = {"rule": "temporal_lane_bias_init[lane] = gate-weighted mean sum(g*e)/sum(g) of the pre-gate temporal "
                   "emission e of the init checkpoint (run init transforms applied) over learned on-road HDV rows of "
                   "free rollouts, by step-start lane (preserves the mean gated contribution)",
           "config": args.config, "overrides": args.set, "ckpt": str(args.ckpt), "roots": args.roots,
           "init_report": trainer.init_report, "n_batches": args.n_batches, "batch_size": args.batch_size,
           "unroll_s": args.unroll_s, "sampler_seed": args.seed, "n_rows": acc.n.tolist(),
           "pregate_mean": mean, "pregate_sd": sd, "gated_mean": (acc.gated / n).tolist(),
           "mean_gate": (acc.gate_sum / n).tolist(),
           "gate_weighted_mean": (acc.gated / acc.gate_sum.clamp_min(1e-12)).tolist()}
    res["temporal_lane_bias_init"] = [float(f"{x:.4g}") for x in res["gate_weighted_mean"]]
    Path(args.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("n_rows", "pregate_mean", "pregate_sd", "gated_mean", "mean_gate",
                                          "temporal_lane_bias_init")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
