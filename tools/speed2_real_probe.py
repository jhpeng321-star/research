"""S-SPEED2 trained-checkpoint / TRAIN-state numerical and 300 s rollout probe.

Only the explicitly selected TRAIN manifest entry and its causal <=t0 observations
are read. The exported episode can live in an isolated measurement directory. No
training run or policy process is modified. See SPEED2.md for the measured receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from speed2_measure import diff, flags, summary, timed  # also enforces nice19 / <=2 CPU affinity
import numpy as np
import torch

from wms.data.history import long_history_features
from wms.model.checkpoint import load_checkpoint
from wms.model.residual_net import courtesy_features
from wms.train.inputs import state_from_kin


def checked(path):
    resolved = Path(path).resolve()
    if any(word in str(resolved).lower() for word in
           ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")):
        raise ValueError("restricted path")
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=checked, required=True)
    parser.add_argument("--manifest", type=checked, required=True)
    parser.add_argument("--episode", type=checked, required=True)
    parser.add_argument("--t0-s", type=float, default=300.)
    parser.add_argument("--pad-slots", type=int, default=595)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=checked, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        # Leave room for the CUDA context and the concurrent one-core CPU measurement.
        torch.cuda.set_per_process_memory_fraction(3.0 * 1024 ** 3 / props.total_memory)
    manifest = json.loads(args.manifest.read_text())
    entry = next(row for row in manifest["episodes"] if row["episode"] == args.episode.name)
    assert entry["split_role"].lower() == "train"
    loaded = load_checkpoint(args.checkpoint, device=str(device))
    model = loaded.model.eval()
    dt = float(model.physics.dt_s)
    t0 = round(args.t0_s / dt)
    arrays = {name: np.load(checked(args.episode / f"{name}.npy"), mmap_mode="r", allow_pickle=False)
              for name in ("veh_kin", "veh_mask", "is_cav", "vid")}
    k = arrays["veh_kin"].shape[1]
    assert k <= args.pad_slots and t0 < len(arrays["veh_kin"])
    start = max(0, t0 - round((model.history_s or 5.) / dt))
    kin = torch.tensor(np.array(arrays["veh_kin"][start:t0 + 1]), device=device)
    mask = torch.tensor(np.array(arrays["veh_mask"][start:t0 + 1]), device=device).bool()
    ids = arrays["vid"][start:t0 + 1]
    continuity = torch.tensor(np.array(ids[:-1] == ids[1:]), device=device)
    continuity = continuity & mask[:-1] & mask[1:]
    kwargs = {}
    if model.long_history:
        features, seen = long_history_features(
            kin, mask, continuity, mask[-1], history_s=model.history_s,
            history_stride_s=model.history_stride_s, dt_s=dt)
        kwargs.update(driver_history=torch.nn.functional.pad(
            features[None], (0, 0, 0, 0, 0, args.pad_slots - k)),
            driver_history_mask=torch.nn.functional.pad(seen[None], (0, 0, 0, args.pad_slots - k)))
    initial_kin = torch.nn.functional.pad(kin[-1:,:], (0, 0, 0, args.pad_slots - k))
    present = torch.nn.functional.pad(mask[-1:], (0, args.pad_slots - k))
    state = state_from_kin(initial_kin, present)
    state["is_cav"] = torch.nn.functional.pad(
        torch.tensor(np.array(arrays["is_cav"][t0:t0 + 1]), device=device).bool(),
        (0, args.pad_slots - k))
    with torch.no_grad():
        _, embedding, _ = model.encode_drivers(state, **kwargs)
        courtesy = courtesy_features(initial_kin, present, model.physics)
    receipt = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": loaded.sha256,
               "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
               "episode": entry, "device": str(device), "torch": torch.__version__,
               "t0_s": args.t0_s, "horizon_s": 300., "slots": args.pad_slots,
               "present": int(present.sum()), "residual_accel_off": model.residual_accel_off,
               "relative_denominator_floor": 1e-8}
    results, times = [], []
    for flag in (0, 1):
        with flags(compact_slots=flag):
            def step():
                x = initial_kin.detach().clone().requires_grad_()
                cond = embedding.detach().clone().requires_grad_()
                cour = courtesy.detach().clone().requires_grad_()
                out = model.residual_net(x, present, conditioning=cond, courtesy=cour)
                loss = (out["accel_delta_mps2"].square().sum()
                        + (out["accel_log_sigma"].square() * present).sum()
                        + out["lane_logit_delta"].square().sum()) / present.sum()
                grads = torch.autograd.grad(loss, [x, cond, cour, *model.residual_net.parameters()],
                                            allow_unused=True)
                return ({key: val.detach().cpu() for key, val in out.items()}, loss.detach().cpu(),
                        [None if val is None else val.detach().cpu() for val in grads])
            result, timing = timed(step, repeats=3)
            results.append(result)
            times.append(timing)
    receipt["trained_residual"] = {"off": times[0], "on": times[1],
        "outputs": diff(results[0][0], results[1][0]), "loss": diff(results[0][1], results[1][1]),
        "gradients": diff(results[0][2], results[1][2])}
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"trained_residual": receipt["trained_residual"]}), flush=True)
    results, times = [], []
    for values in ({}, {"pivot_diag_off": 1}, {"pivot_diag_off": 1, "compact_slots": 1}):
        with flags(**values), torch.no_grad():
            def step():
                torch.manual_seed(748)
                roll = model.rollout(state, n_steps=round(300. / dt), **kwargs)
                return ({key: roll[key].cpu() for key in
                         ("positions", "speeds", "accel", "lanes", "masks")},
                        [torch.get_rng_state(), torch.cuda.get_rng_state()]
                        if device.type == "cuda" else [torch.get_rng_state()])
            result, timing = timed(step, repeats=1, warmup=0)
            results.append(result)
            times.append(timing)
        receipt["completed_rollout_regimes"] = len(results)
        args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    receipt["rollout"] = {"off": times[0], "diag_off": times[1], "diag_off_compact": times[2],
        "diagnostics_only_difference": diff(results[0][0], results[1][0]),
        "compact_difference": diff(results[1][0], results[2][0]),
        "diagnostics_rng": diff(results[0][1], results[1][1]),
        "compact_rng": diff(results[1][1], results[2][1])}
    assert summary(receipt["rollout"]["diagnostics_only_difference"])["bit_equal"]
    if device.type == "cuda":
        receipt["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"rollout": receipt["rollout"]}), flush=True)


if __name__ == "__main__":
    main()
