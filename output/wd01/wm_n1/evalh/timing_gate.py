"""WM-N1 GPU timing gate (design WM_N1_DESIGN.md section 7).

The design (lines 229-231) requires a **1-h gate on the rented GPU**
(throughput, memory, a loss decrease on a 2 % TRAIN subset) that extrapolates
the full cost before the owner approves the run.  The budget being tested is
the design's estimate (lines 224-227): "15 epochs x 5 members ~ 25-50 GPU-h" +
"rollout fine-tuning +30 %" + "evaluation about 2 GPU-h", i.e. an upper bound
of 50 GPU-h -- the ``BUDGET_HOURS_MAX`` below.

This module measures training-step throughput and rollout-sample throughput for
a member config and extrapolates the full-plan wall time.  It is runnable on
CPU (``--device cpu --tiny``) for the tests; the gate writes
``TIMING_GATE.json`` with a pass/fail verdict.

Only 1 compute worker, default threads pinned to 1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Design WM_N1_DESIGN.md:229-231 (the gate) and 224-227 (the budget).
BUDGET_HOURS_MAX = 50.0          # design:226 "25-50 GPU-h" upper bound
GATE_LIMIT_HOURS = 1.0           # design:230 "a 1-h gate"
ROLLOUT_FRACTION = 0.30          # design:227 "rollout fine-tuning +30 %"
EVAL_HOURS = 2.0                 # design:227 "evaluation about 2 GPU-h"
GATE_QUOTE = ("Timing gate first: a 1-h gate on the rented GPU (throughput, "
              "memory, a loss decrease on a 2 % TRAIN subset). It extrapolates "
              "the full cost, and the owner approves before the full run.")

# K_max is a FORMAT parameter (amendment 5, "K = 192"): gate scenes are sized
# at the format K -- 128 for the frozen v2 format, 192 for v3.  Mirrors
# features/model constants (the contract test pins those); kept local so this
# Toy-proxy module stays import-light.
N_VEHICLES_BY_FORMAT = {"v2": 128, "v3": 192}


def n_vehicles_for_format(format_name: str) -> int:
    """Registered slot width of a gate format (v2 -> 128, v3 -> 192)."""
    try:
        return N_VEHICLES_BY_FORMAT[format_name]
    except KeyError:
        raise ValueError(
            f"unknown format {format_name!r}; registered: "
            f"{sorted(N_VEHICLES_BY_FORMAT)}") from None


@dataclass
class MemberConfig:
    """One member's shape and the full-plan schedule to extrapolate.

    ``n_vehicles`` is the slot width K of the measured scene: v2 default
    128; a v3 (amendment 5) gate passes ``n_vehicles=192`` or is built with
    ``make_config(format="v3")``.
    """

    name: str = "member"
    n_members: int = 5
    epochs: int = 15
    steps_per_epoch: int = 2000
    batch_size: int = 8
    n_vehicles: int = 128
    d_in: int = 16
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_quantiles: int = 19
    rollout_horizon_steps: int = 20
    lr: float = 1e-3
    seed: int = 0

    def to_json(self) -> dict:
        return asdict(self)


def make_config(tiny: bool = False, format: str | None = None,
                **overrides) -> MemberConfig:
    cfg = MemberConfig()
    if tiny:
        cfg = MemberConfig(n_members=5, epochs=2, steps_per_epoch=4,
                           batch_size=2, n_vehicles=8, d_model=16, n_layers=1,
                           n_heads=2, rollout_horizon_steps=4)
    elif format is not None and "n_vehicles" not in overrides:
        overrides["n_vehicles"] = n_vehicles_for_format(format)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _torch(device: str):
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(0)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("cuda requested but not available")
    return torch


def _build_model(cfg: MemberConfig, device: str):
    torch = _torch(device)
    import torch.nn as nn

    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Linear(cfg.d_in, cfg.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model, nhead=cfg.n_heads,
                dim_feedforward=2 * cfg.d_model, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            self.step_head = nn.Linear(cfg.d_model, 1)
            self.out_head = nn.Linear(cfg.d_model, cfg.n_quantiles)

        def forward(self, x, mask=None):
            h = self.encoder(self.embed(x))
            pooled = h.mean(dim=1)
            return self.step_head(h), self.out_head(pooled)

    return Toy().to(device)


def measure_throughput(cfg: MemberConfig, device: str, *,
                       warmup: int = 2, timed_steps: int = 5) -> dict:
    """Training steps/s and rollout scene-steps/s on the given device."""
    torch = _torch(device)
    model = _build_model(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    def batch():
        x = torch.randn(cfg.batch_size, cfg.n_vehicles, cfg.d_in,
                        generator=gen).to(device)
        y = torch.randn(cfg.batch_size, cfg.n_vehicles, 1,
                        generator=gen).to(device)
        return x, y

    model.train()
    for _ in range(warmup):
        x, y = batch()
        opt.zero_grad()
        step, _ = model(x)
        loss = torch.nn.functional.mse_loss(step, y)
        loss.backward()
        opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(timed_steps):
        x, y = batch()
        opt.zero_grad()
        step, _ = model(x)
        loss = torch.nn.functional.mse_loss(step, y)
        loss.backward()
        opt.step()
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    steps_per_sec = timed_steps / train_s if train_s > 0 else float("nan")

    model.eval()
    with torch.no_grad():
        x, _ = batch()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        h = x
        for _ in range(cfg.rollout_horizon_steps):
            h = h + 0.01 * model(h)[0]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        roll_s = time.perf_counter() - t1
    scene_steps = cfg.batch_size * cfg.rollout_horizon_steps
    rollout_per_sec = scene_steps / roll_s if roll_s > 0 else float("nan")

    peak_mb = float("nan")
    try:
        import resource
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:                                # pragma: no cover
        try:
            import tracemalloc
            cur, peak = tracemalloc.get_traced_memory()
            peak_mb = peak / 1e6
        except Exception:
            pass
    return dict(steps_per_sec=float(steps_per_sec),
                rollout_scene_steps_per_sec=float(rollout_per_sec),
                rollout_samples_per_sec=float(rollout_per_sec),
                train_seconds=float(train_s), rollout_seconds=float(roll_s),
                peak_memory_mb=peak_mb,
                device=device, n_params=int(sum(
                    p.numel() for p in model.parameters())))


def loss_decrease_check(cfg: MemberConfig, device: str, *,
                        steps: int = 30) -> dict:
    """Overfit a fixed tiny target: the loss must strictly decrease."""
    torch = _torch(device)
    model = _build_model(cfg, device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + 1)
    x = torch.randn(cfg.batch_size, cfg.n_vehicles, cfg.d_in,
                    generator=gen).to(device)
    target = torch.randn(cfg.batch_size, cfg.n_vehicles, 1,
                         generator=gen).to(device)
    first = last = float("nan")
    for i in range(steps):
        opt.zero_grad()
        step, _ = model(x)
        loss = torch.nn.functional.mse_loss(step, target)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss.item())
        last = float(loss.item())
    return dict(first_loss=first, last_loss=last,
                decreased=bool(last < first), steps=int(steps))


def extrapolate(cfg: MemberConfig, throughput: dict) -> dict:
    """Extrapolate the full-plan wall time from the measured throughput."""
    total_steps = cfg.n_members * cfg.epochs * cfg.steps_per_epoch
    sps = throughput["steps_per_sec"]
    if not (sps and np.isfinite(sps) and sps > 0):
        return dict(train_hours=float("inf"), rollout_hours=float("inf"),
                    eval_hours=EVAL_HOURS, total_hours=float("inf"),
                    total_steps=int(total_steps), feasible=False)
    train_hours = total_steps / sps / 3600.0
    rollout_hours = train_hours * ROLLOUT_FRACTION
    total_hours = train_hours + rollout_hours + EVAL_HOURS
    return dict(train_hours=float(train_hours),
                rollout_hours=float(rollout_hours),
                eval_hours=EVAL_HOURS, total_hours=float(total_hours),
                total_steps=int(total_steps), feasible=True)


def run_gate(cfg: MemberConfig, *, device: str = "cpu", tiny: bool = False,
             budget_hours_max: float = BUDGET_HOURS_MAX,
             gate_limit_hours: float = GATE_LIMIT_HOURS,
             warmup: int = 2, timed_steps: int = 5,
             loss_steps: int = 30) -> dict:
    t0 = time.perf_counter()
    throughput = measure_throughput(cfg, device, warmup=warmup,
                                    timed_steps=timed_steps)
    loss = loss_decrease_check(cfg, device, steps=loss_steps)
    gate_seconds = time.perf_counter() - t0
    extra = extrapolate(cfg, throughput)
    within_budget = bool(extra["feasible"]
                         and extra["total_hours"] <= budget_hours_max)
    within_gate = bool(gate_seconds <= gate_limit_hours * 3600.0)
    passed = bool(within_budget and within_gate and loss["decreased"])
    return dict(
        schema="wm_n1_timing_gate/1",
        generated_utc=datetime.now(timezone.utc)
        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        device=device, tiny=bool(tiny), config=cfg.to_json(),
        throughput=throughput, loss_decrease=loss,
        extrapolation=extra,
        thresholds=dict(budget_hours_max=budget_hours_max,
                        gate_limit_hours=gate_limit_hours,
                        rollout_fraction=ROLLOUT_FRACTION, eval_hours=EVAL_HOURS,
                        citation="WM_N1_DESIGN.md:226-227,229-231"),
        gate_seconds=float(gate_seconds),
        checks=dict(within_budget=within_budget, within_gate=within_gate,
                    loss_decreased=bool(loss["decreased"])),
        passed=passed, quote=GATE_QUOTE)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalh.timing_gate")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--config", default=None, help="JSON member config")
    p.add_argument("--budget-hours-max", type=float, default=BUDGET_HOURS_MAX)
    p.add_argument("--gate-limit-hours", type=float, default=GATE_LIMIT_HOURS)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--timed-steps", type=int, default=5)
    p.add_argument("--loss-steps", type=int, default=30)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    overrides = json.loads(Path(args.config).read_text(encoding="utf-8")) \
        if args.config else {}
    cfg = make_config(tiny=args.tiny, **overrides)
    result = run_gate(cfg, device=args.device, tiny=args.tiny,
                      budget_hours_max=args.budget_hours_max,
                      gate_limit_hours=args.gate_limit_hours,
                      warmup=args.warmup, timed_steps=args.timed_steps,
                      loss_steps=args.loss_steps)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"timing_gate device={args.device} tiny={args.tiny} "
          f"steps/sec={result['throughput']['steps_per_sec']:.2f} "
          f"rollout/sec={result['throughput']['rollout_samples_per_sec']:.2f}")
    print(f"  extrapolated total={result['extrapolation']['total_hours']:.2f} "
          f"GPU-h vs budget {args.budget_hours_max:.1f} GPU-h "
          f"(WM_N1_DESIGN.md:226)")
    print(f"  gate={result['gate_seconds']:.2f}s "
          f"loss_decreased={result['loss_decrease']['decreased']} "
          f"PASS={result['passed']}")
    print(f"  wrote {out}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
