"""Crash-safe PPO training for the one-step paired IDM option environment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.io import atomic_write_json
from idm.paired_option_dataset import load_dataset
from idm.paired_option_policy import (
    BinaryOptionActorCritic,
    evaluate_binary_policy,
    normalization,
    select_deployment_threshold,
)


SCHEMA = "idm_paired_option_ppo/1"
LOCK_NAME = "active_trainer.lock"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(
                ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
                and int(code.value) == 259
            )
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _acquire_lock(output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / LOCK_NAME
    if path.exists():
        try:
            owner = int(json.loads(path.read_text(encoding="utf-8"))["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            owner = -1
        if _pid_alive(owner):
            raise RuntimeError(f"trainer PID {owner} already owns {output}")
        path.unlink(missing_ok=True)
    path.write_text(
        json.dumps({"pid": os.getpid(), "created_unix_s": time.time()}), encoding="utf-8"
    )
    return path


def _release_lock(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("pid", -1)) == os.getpid():
            path.unlink(missing_ok=True)
    except (OSError, ValueError, json.JSONDecodeError):
        pass


def _torch_save_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _explained_variance(target: torch.Tensor, prediction: torch.Tensor) -> float:
    variance = torch.var(target, unbiased=False)
    if float(variance) <= 1e-12:
        return float("nan")
    return float(1.0 - torch.var(target - prediction, unbiased=False) / variance)


def _flatten(prefix: str, payload: dict, output: dict[str, Any]) -> None:
    for key, value in payload.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            if key == "by_penetration":
                continue
            _flatten(name, value, output)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            output[name] = value


def _write_metrics(output: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_json(output / "metrics.json", rows)
    fields = sorted({key for row in rows for key in row})
    temporary = output / "metrics.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output / "metrics.csv")


def _checkpoint_payload(
    *, model, optimizer, step, iteration, config, dataset_manifest, rng, threshold, metrics
) -> dict:
    return {
        "schema": SCHEMA,
        "step": int(step),
        "iteration": int(iteration),
        "config": config,
        "dataset_sha256": dataset_manifest["dataset_sha256"],
        "records_sha256": dataset_manifest["records_sha256"],
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "numpy_rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "deployment_threshold": float(threshold),
        "metrics": metrics,
    }


def train(args: argparse.Namespace) -> dict:
    output = args.output.resolve()
    lock = _acquire_lock(output)
    try:
        train_rows, manifest = load_dataset(args.dataset.resolve(), "train")
        validation_rows, validation_manifest = load_dataset(
            args.dataset.resolve(), "validation"
        )
        if manifest["dataset_sha256"] != validation_manifest["dataset_sha256"]:
            raise ValueError("training/validation dataset manifest mismatch")
        config = {
            "hidden": int(args.hidden),
            "learning_rate": float(args.learning_rate),
            "batch_size": int(args.batch_size),
            "ppo_epochs": int(args.ppo_epochs),
            "clip_ratio": float(args.clip_ratio),
            "entropy_coefficient": float(args.entropy_coefficient),
            "value_coefficient": float(args.value_coefficient),
            "maximum_gradient_norm": float(args.maximum_gradient_norm),
            "seed": int(args.seed),
        }
        mean, scale = normalization(train_rows)
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        model = BinaryOptionActorCritic(mean, scale, hidden=args.hidden)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        step = 0
        iteration = 0
        metrics: list[dict[str, Any]] = []
        latest = output / "checkpoints" / "latest.pt"

        if args.resume:
            if not latest.exists():
                raise FileNotFoundError(f"--resume requires verified checkpoint {latest}")
            state = torch.load(latest, map_location="cpu", weights_only=False)
            if state.get("schema") != SCHEMA:
                raise ValueError("resume checkpoint schema mismatch")
            if state.get("dataset_sha256") != manifest["dataset_sha256"] or state.get(
                "records_sha256"
            ) != manifest["records_sha256"]:
                raise ValueError("resume checkpoint dataset identity mismatch")
            if state.get("config") != config:
                raise ValueError("resume checkpoint training configuration mismatch")
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            rng.bit_generator.state = state["numpy_rng_state"]
            torch.set_rng_state(state["torch_rng_state"])
            step = int(state["step"])
            iteration = int(state["iteration"])
            metrics = list(state.get("metrics") or [])
        elif latest.exists():
            raise FileExistsError(
                f"fresh training refuses to overwrite {latest}; choose a new output or --resume"
            )

        observations = torch.as_tensor(
            np.asarray([row.observation for row in train_rows], dtype=np.float32)
        )
        activate_rewards = torch.as_tensor(
            [row.activate_reward for row in train_rows], dtype=torch.float32
        )
        checkpoints = sorted({0, *args.checkpoint_steps, int(args.total_steps)})

        def save(name: str, threshold: float, row: dict) -> None:
            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                step=step,
                iteration=iteration,
                config=config,
                dataset_manifest=manifest,
                rng=rng,
                threshold=threshold,
                metrics=metrics,
            )
            _torch_save_atomic(output / "checkpoints" / name, payload)
            _torch_save_atomic(latest, payload)
            atomic_write_json(output / "latest_evaluation.json", row)

        if step == 0 and not metrics:
            threshold, selection = select_deployment_threshold(model, train_rows)
            row = {
                "schema": SCHEMA,
                "step": 0,
                "iteration": 0,
                "sampled_activate_count": 0,
                "sampled_abstain_count": 0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": math.log(2.0),
                "approximate_kl": 0.0,
                "explained_variance": float("nan"),
                "deployment_threshold": threshold,
                "threshold_gate_feasible_on_train": selection["frozen_gate_feasible"],
            }
            _flatten("train", evaluate_binary_policy(model, train_rows, threshold), row)
            _flatten(
                "validation", evaluate_binary_policy(model, validation_rows, threshold), row
            )
            metrics.append(row)
            _write_metrics(output, metrics)
            save("step_000000.pt", threshold, row)

        while step < args.total_steps:
            next_checkpoint = min(value for value in checkpoints if value > step)
            batch_size = min(args.batch_size, next_checkpoint - step)
            indices = rng.integers(0, len(train_rows), size=batch_size)
            batch_obs = observations[indices]
            target_rewards = activate_rewards[indices]
            with torch.no_grad():
                old_logits, old_values = model(batch_obs)
                old_probabilities = torch.softmax(old_logits, dim=-1)
                actions = torch.multinomial(old_probabilities, 1).squeeze(-1)
                old_log_prob = torch.log_softmax(old_logits, dim=-1).gather(
                    1, actions[:, None]
                ).squeeze(1)
                returns = torch.where(actions == 1, target_rewards, torch.zeros_like(target_rewards))
                advantages = returns - old_values
                advantage_std = advantages.std(unbiased=False)
                if float(advantage_std) > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantage_std + 1e-8)

            policy_loss_value = value_loss_value = entropy_value = kl_value = 0.0
            explained = _explained_variance(returns, old_values)
            for _ in range(args.ppo_epochs):
                logits, values = model(batch_obs)
                log_prob = torch.log_softmax(logits, dim=-1).gather(
                    1, actions[:, None]
                ).squeeze(1)
                ratio = torch.exp(log_prob - old_log_prob)
                surrogate = torch.minimum(
                    ratio * advantages,
                    torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
                    * advantages,
                )
                policy_loss = -surrogate.mean()
                value_loss = F.mse_loss(values, returns)
                probabilities = torch.softmax(logits, dim=-1)
                entropy = -(probabilities * torch.log_softmax(logits, dim=-1)).sum(-1).mean()
                loss = (
                    policy_loss
                    + args.value_coefficient * value_loss
                    - args.entropy_coefficient * entropy
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.maximum_gradient_norm)
                optimizer.step()
                with torch.no_grad():
                    kl = (old_log_prob - log_prob).mean()
                policy_loss_value = float(policy_loss.detach())
                value_loss_value = float(value_loss.detach())
                entropy_value = float(entropy.detach())
                kl_value = float(kl)

            step += batch_size
            iteration += 1
            threshold, selection = select_deployment_threshold(model, train_rows)
            train_eval = evaluate_binary_policy(model, train_rows, threshold)
            validation_eval = evaluate_binary_policy(model, validation_rows, threshold)
            row = {
                "schema": SCHEMA,
                "step": int(step),
                "iteration": int(iteration),
                "sampled_activate_count": int((actions == 1).sum()),
                "sampled_abstain_count": int((actions == 0).sum()),
                "sampled_reward_mean": float(returns.mean()),
                "sampled_constraint_failure_count": int(
                    sum(
                        bool(actions[index] == 1 and not train_rows[int(indices[index])].constraints_pass)
                        for index in range(batch_size)
                    )
                ),
                "policy_loss": policy_loss_value,
                "value_loss": value_loss_value,
                "entropy": entropy_value,
                "approximate_kl": kl_value,
                "explained_variance": explained,
                "deployment_threshold": threshold,
                "threshold_gate_feasible_on_train": selection["frozen_gate_feasible"],
            }
            _flatten("train", train_eval, row)
            _flatten("validation", validation_eval, row)
            if any(not math.isfinite(float(row[key])) for key in ("policy_loss", "value_loss", "entropy", "approximate_kl")):
                raise FloatingPointError(f"non-finite learning metric at step {step}")
            metrics.append(row)
            _write_metrics(output, metrics)
            save("latest.pt", threshold, row)
            if step in checkpoints:
                save(f"step_{step:06d}.pt", threshold, row)
            print(
                f"step={step:6d} p_activate={train_eval['mean_activation_probability']:.3f} "
                f"coverage={validation_eval['activation_coverage']:.3f} "
                f"val_reward={validation_eval['mean_reward']:+.4f} "
                f"val_harm={validation_eval['harmful_activation_fraction']:.3f}",
                flush=True,
            )

        final = metrics[-1]
        summary = {
            "schema": SCHEMA,
            "status": "complete",
            "dataset": str(args.dataset.resolve()),
            "dataset_sha256": manifest["dataset_sha256"],
            "total_steps": int(step),
            "iteration": int(iteration),
            "checkpoint": str((output / "checkpoints" / f"step_{step:06d}.pt").resolve()),
            "latest_metrics": final,
            "scientific_scope": "development-only paired contextual-bandit smoke",
            "checkpoint_freeze_authorized": False,
        }
        atomic_write_json(output / "summary.json", summary)
        return summary
    finally:
        _release_lock(lock)


def _steps(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-steps", type=int, default=2000)
    parser.add_argument("--checkpoint-steps", type=_steps, default=[2000, 5000, 10000])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--maximum-gradient-norm", type=float, default=0.5)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.total_steps <= 0 or args.batch_size <= 0 or args.ppo_epochs <= 0:
        parser.error("steps, batch size and PPO epochs must be positive")
    print(json.dumps(train(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
