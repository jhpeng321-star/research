"""Resumable executed-acceleration DAgger in WM2, with candidate-A checkpoints."""
from __future__ import annotations

import json
import random
import time
from dataclasses import replace

import numpy as np
import torch
import torch.nn.functional as F

from wms.control.dreamer import DreamerConfig
from wms.control.policy14 import Policy14Trainer, make_controller_factory
from wms.control.train_cli import GroupSignal, _read_config

from .law import RADIUS, execute, inverse
from .represent import parser
from .runtime import (Feedback, check_seed, configure_execution, detach_inputs, load_assets, load_training_starts,
                      residual, rollout, safe_path, sha256, zones)


def make_trainer(model, starts, raw, *, device="cpu", teacher=None):
    from wms.control.contract import RewardWeights
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.objective14 import authoritative_reward
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.real_returns14 import ScaledTeamCritic
    from wms.control.safety import safe_accel
    # Save the *future candidate-A phase* config, not a fictitious RL update.
    # Real-return regression is inactive during distillation, so its provider is
    # unnecessary here. The export reinstates the configured real_return_mix.
    cfg = DreamerConfig(**raw["trainer"])
    policy = CavPolicy(lane_head=False, residual_on_native=True, accel_max=RADIUS)
    encoder = RsuGroupEncoder()
    if teacher is not None:
        source = teacher.policy.state_dict()
        for name in ("own_encoder", "report_encoder", "attention"):
            module = getattr(policy, name)
            prefix = name + "."
            module.load_state_dict({k[len(prefix):]: v for k, v in source.items() if k.startswith(prefix)})
        encoder.load_state_dict(teacher.group_encoder.state_dict())
    trainer = Policy14Trainer([model], policy, GroupSignal(encoder),
        ScaledTeamCritic(entry_backlog=True, value_scale=raw["policy14"]["critic_value_scale"]), starts,
        observation_builder=build_observations, reward_fn=authoritative_reward, safety=safe_accel,
        controller_factory=make_controller_factory(native_ramp_merge="wm_law", native_lane_change=True),
        config=replace(cfg, real_return_mix=0.), reward_weights=RewardWeights(**raw["reward"]), device=device,
        counterfactual=CounterfactualConfig(**raw["counterfactual"]))
    return trainer


def cpu_record(row):
    return {"state": {k: v.cpu().clone() for k, v in row["state"].items()},
            "inputs": detach_inputs(row["inputs"]),
            **{k: row[k].detach().cpu().clone() for k in ("mask", "ready", "teacher", "entrant")}}


def executed_loss(trainer, row, law="current", delta=1.):
    device = trainer.device
    state = {k: v.to(device) for k, v in row["state"].items()}
    inputs = detach_inputs(row["inputs"], device)
    action = residual(trainer.policy, trainer.group_encoder, inputs)
    # Newborns between decision ticks receive zero, just as in Feedback.
    action = torch.where(row["ready"].to(device), action, 0.)
    predicted = execute(state, action, law)
    target = row["teacher"].to(device)
    mask = row["mask"].to(device)
    values = F.huber_loss(predicted, target, reduction="none", delta=delta)
    return values[mask].mean() if bool(mask.any()) else predicted.sum() * 0., predicted


@torch.no_grad()
def fit_initial_head(trainer, rows, law):
    """Fit the residual head to inverse-executed teacher labels by least squares.

    This learned initialization crosses the incumbent cap dead zone without a
    hand-set braking bias or a raw-MAPPO-command target. Subsequent optimisation
    uses only Huber loss through the exact execution law.
    """
    features, targets = [], []
    for row in rows:
        state = {k: v.to(trainer.device) for k, v in row["state"].items()}
        captured = []
        handle = trainer.policy.residual_head.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        try:
            residual(trainer.policy, trainer.group_encoder, detach_inputs(row["inputs"], trainer.device))
        finally:
            handle.remove()
        mask = (row["mask"] & row["ready"]).to(trainer.device)
        r = inverse(state, row["teacher"].to(trainer.device), law)
        eps = torch.finfo(r.dtype).eps
        latent = torch.atanh((r / RADIUS).clamp(-1 + eps, 1 - eps))
        features.append(captured[0][mask].double().cpu())
        targets.append(latent[mask].double().cpu())
    x, y = torch.cat(features), torch.cat(targets)
    if not len(y):
        raise ValueError("initial teacher rollouts contain no acting CAVs")
    x = torch.cat((x, torch.ones(len(x), 1, dtype=x.dtype)), -1)
    fitted = torch.linalg.lstsq(x, y[:, None], driver="gelsd").solution[:, 0]
    head = trainer.policy.residual_head
    head.weight.copy_(fitted[:-1][None].to(head.weight))
    head.bias.copy_(fitted[-1:].to(head.bias))
    return {"method": "inverse-execution least squares", "cav_ticks": len(y),
            "latent_mse": float((x @ fitted - y).square().mean())}


@torch.no_grad()
def fit_metrics(trainer, rows, law):
    totals = {name: [0, 0., 0.] for name in ("all", "merge", "ramp", "mainline", "entrants")}
    brake_total = brake_reproduced = 0.
    for row in rows:
        _, prediction = executed_loss(trainer, row, law)
        prediction = prediction.cpu()
        err = prediction - row["teacher"]
        masks = {"all": row["mask"], **{k: v & row["mask"] for k, v in zones(row["state"]).items()},
                 "entrants": row["mask"] & row["entrant"]}
        for name, mask in masks.items():
            totals[name][0] += int(mask.sum())
            totals[name][1] += float(err[mask].abs().sum())
            totals[name][2] += float(err[mask].square().sum())
        ctrl0 = execute(row["state"], torch.zeros_like(row["teacher"]))
        braking = (ctrl0 - row["teacher"]).clamp_min(0.)
        copied = torch.minimum((ctrl0 - prediction).clamp_min(0.), braking)
        brake_total += float(braking[row["mask"]].sum())
        brake_reproduced += float(copied[row["mask"]].sum())
    return {"zones": {name: {"cav_ticks": n, "mae": a/n if n else None,
                            "rmse": (s/n)**.5 if n else None} for name, (n, a, s) in totals.items()},
            "teacher_braking_magnitude_sum": brake_total,
            "fraction_teacher_braking_reproduced": brake_reproduced/brake_total if brake_total else None}


def save_checkpoint(trainer, path, raw, state):
    """Use the actual candidate-A writer; DAgger state is an additive payload key."""
    previous = trainer.config
    try:
        trainer.config = DreamerConfig(**raw["trainer"])
        trainer.save_checkpoint(path)
    finally:
        trainer.config = previous
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["distillation"] = state
    # Reconstituted sampler state belongs to DAgger; an RL phase uses its own
    # selected training windows (the existing loop-refresh guard handles this).
    torch.save(payload, path)


def main(argv=None):
    p = parser(__doc__)
    p.set_defaults(config="configs/c4_distill_mappo.toml")
    p.add_argument("--updates", type=int, default=600, help="total DAgger updates, including resume")
    p.add_argument("--resume")
    args = p.parse_args(argv)
    check_seed(args.seed)
    if args.updates < 1:
        p.error("--updates must be positive")
    execution = configure_execution(args.execution_driver)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    raw = _read_config(args.config)
    spec = raw["distill"]
    if (spec["law"] not in ("current", "cap_relative") or
            min(spec[k] for k in ("horizon_s", "pad_max", "learning_rate", "huber_delta",
                                  "gradient_steps", "replay_frames", "checkpoint_every")) <= 0):
        raise ValueError("invalid distillation settings")
    model, teacher, receipts = load_assets(args.wm, args.wm_config, args.mappo, args.device)
    train, val = load_training_starts(raw, model, args.seed)
    trainer = make_trainer(model, train, raw, device=args.device, teacher=teacher)
    # The training optimizer is distinct from the future RL phase's optimizer,
    # but has identical trainable parameter ordering for an exact full resume.
    optimizer = torch.optim.Adam(trainer.actor_parameters, lr=spec["learning_rate"])
    generator = torch.Generator().manual_seed(args.seed)
    signature = {"wm": receipts, "teacher_sha256": sha256(args.mappo), "seed": args.seed,
                 "resolved_config": raw, "law": spec["law"], "execution": execution,
                 "corpora": train.distill_provenance,
                 "training_worlds": sorted({m.get("world", m.get("world_id")) for m in train.metadata.values()})}
    replay, update, initial = [], 0, None
    if args.resume:
        saved = torch.load(safe_path(args.resume), map_location="cpu", weights_only=True)
        state = saved["distillation"]
        if signature != state["signature"]:
            raise ValueError("DAgger resume provenance/config/seed mismatch")
        # Restore without asking the future RL phase to regress real returns.
        trainer.load_checkpoint(safe_path(args.resume), policy_only=True)
        optimizer.load_state_dict(state["optimizer"])
        train.load_state_dict(state["sampler"])
        generator.set_state(state["generator"])
        from wms.control.dreamer import _restore_rng
        _restore_rng(saved["rng"], trainer.device)
        replay, update, initial = state["replay"], state["updates"], state["initial_head_fit"]
    out = safe_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    for u in range(update, args.updates):
        batch = train.sample(1, device=args.device)
        incoming = []
        # Both distributions are visited each round; teacher labels both. Prior
        # rounds remain in a bounded FIFO replay, persisted for exact restart.
        for drive in ("teacher", "student"):
            rollout(model, batch, Feedback(teacher, trainer.policy, trainer.group_encoder,
                drive=drive, law=spec["law"], record=lambda row: incoming.append(cpu_record(row))),
                horizon_s=spec["horizon_s"], seed=args.seed + u, pad_max=spec["pad_max"])
        replay.extend(incoming)
        replay = replay[-spec["replay_frames"]:]
        if initial is None:
            initial = fit_initial_head(trainer, incoming, spec["law"])
        losses = []
        for _ in range(spec["gradient_steps"]):
            picked = torch.randint(len(replay), (), generator=generator).item()
            loss, _ = executed_loss(trainer, replay[picked], spec["law"], spec["huber_delta"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.actor_parameters, trainer.config.grad_clip,
                                           error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        metrics = dict(update=u+1, huber_loss=sum(losses)/len(losses),
                       **fit_metrics(trainer, incoming, spec["law"]))
        with (out / "updates.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        state = {"schema": "wm_dagger/1", "signature": signature, "updates": u+1,
                 "initial_head_fit": initial, "replay": replay, "optimizer": optimizer.state_dict(),
                 "sampler": train.state_dict(), "generator": generator.get_state()}
        # RL update_count stays zero: imitation updates are a separate clock.
        if (u+1) % spec["checkpoint_every"] == 0 or u+1 == args.updates:
            save_checkpoint(trainer, out / f"policy_{u+1:06d}.pt", raw, state)
        save_checkpoint(trainer, out / "policy_latest.pt", raw, state)
        print(json.dumps({k: v for k, v in metrics.items() if k != "zones"}), flush=True)
    # Development fit is measured on disjoint TRAIN-corpus validation windows.
    dev_rows = []
    rollout(model, val.sample(1, device=args.device), Feedback(teacher, trainer.policy,
        trainer.group_encoder, drive="student", law=spec["law"],
        record=lambda row: dev_rows.append(cpu_record(row))), horizon_s=spec["horizon_s"],
        seed=args.seed, pad_max=spec["pad_max"])
    report = {"schema": "wm_dagger_summary/1", "updates": args.updates, "signature": signature,
              "initial_head_fit": initial, "development_fit": fit_metrics(trainer, dev_rows, spec["law"]),
              "seconds": time.monotonic()-started, "distilled_only": True,
              "checkpoint": str(out / "policy_latest.pt")}
    (out / "DISTILLATION.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
