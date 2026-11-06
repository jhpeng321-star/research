"""Compare flag-off tensors/updates against a separately exported legacy source tree.

Run this file once with legacy src first on PYTHONPATH and --save legacy.pt;
then with the patched src and --compare legacy.pt. Inputs are synthetic only.
The same probe checks the original training/rollout, history, controller and
policy update paths, including parameter/input gradients and RNG states.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
if hasattr(os, "nice") and os.nice(0) < 19:
    os.nice(19 - os.nice(0))

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def helpers(name):
    spec = importlib.util.spec_from_file_location("_speed2_verify_helpers", ROOT / "tests" / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def collect():
    from wms.control.bridge import _merge_lane
    from wms.core import speed_flags
    from wms.data.history import long_history_features
    from wms.train.cav_drive import TimelineControllers
    from wms.train.losses.base import LossContext
    from wms.train.losses.merge_pivot import MergePivotConfig, MergePivotTerm
    for name, env in speed_flags.FLAGS.items():
        os.environ[env] = "1" if name == "grad_audit_every" else "0"
    torch.set_num_threads(1)
    torch.manual_seed(810)
    speed = helpers("test_speed2_flags.py")
    net = speed._net()
    inputs = speed._inputs()
    out, gradients = speed._run(net, *inputs)
    result = {"residual_outputs": out, "residual_gradients": gradients}
    m, inp, targets = helpers("test_merge_pivot_training.py")._fixture("hybrid")
    term = MergePivotTerm(MergePivotConfig(tf_weight=1., weight=.1, passage_law="hybrid"))
    term.prepare(targets)
    prep = targets.prepared["merge_pivot"]
    prep["event"].zero_()
    for side in ("f", "l"):
        prep["geometry"][f"gap_{side}"][:, :, 0] = torch.tensor([10., 15., 20., 25.])
    torch.manual_seed(811)
    roll = m.rollout(inp.initial, n_steps=8)
    # The term's target window has four steps; retain the full eight-step rollout as evidence.
    pred = {key: val[:, :4] if torch.is_tensor(val) and val.ndim >= 3 and val.shape[1] == 8 else val
            for key, val in roll.items()}
    loss = term(LossContext(pred=pred, current=inp.initial, targets=targets, c0=0, c1=4)).value
    result["training_loss"] = loss.detach()
    result["training_gradients"] = torch.autograd.grad(
        loss, tuple(p for p in m.parameters() if p.requires_grad), allow_unused=True)
    result["recorded_geometry"] = prep["geometry"]
    result["rollout"] = {key: val for key, val in roll.items() if torch.is_tensor(val)}
    result["rollout_rng"] = torch.get_rng_state()
    history = inputs[0][0:1].expand(7, -1, -1).clone()
    mask = inputs[1][0:1].expand(7, -1)
    target = torch.ones(6, mask.shape[1], dtype=torch.bool)
    result["history"] = long_history_features(history, mask, target, mask[-1], history_s=5.)
    state = speed._state()
    state["is_cav"][:] = True
    result["merge_lane"] = _merge_lane(state, state["is_cav"] & state["present"])
    ids = torch.tensor([[[0, 3, 3, 0, 3, 0], [0, 0, 3, 0, 3, 0]]])
    timeline = dict(controller=ids, acceleration_mps2=torch.full(ids.shape, -.7),
                    target_speed_mps=torch.full(ids.shape, 4.))
    ctrl = TimelineControllers(speed._StubModel(), timeline)
    result["cav_drive"] = torch.stack([ctrl(state) for _ in range(4)])
    D = helpers("unit/test_c4_dreamer.py")
    D._seed(812)
    trainer = D._trainer(horizon=2, ensemble=2, actor_gradient_audit=True,
                         checkpoint_imagination=True, uncertainty_weight=.5,
                         reward_fn=D.hdv_reward, entropy_weight=.1)
    result["policy_metrics"] = [trainer.update() for _ in range(2)]
    for row in result["policy_metrics"]:
        row.pop("wall_time_s")
    for name in ("policy", "group_encoder", "critic", "target_critic", "actor_optimizer", "critic_optimizer"):
        result[name] = getattr(trainer, name).state_dict()
    result["policy_rng"] = torch.get_rng_state()
    return result


def compare(a, b, path="root"):
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype, path
        # Compare storage bytes, including signed zero, infinities and NaN payloads.
        assert torch.equal(a.reshape(-1).contiguous().view(torch.uint8),
                           b.reshape(-1).contiguous().view(torch.uint8)), path
        return 1
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        return sum(compare(a[key], b[key], f"{path}/{key}") for key in a)
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        return sum(compare(x, y, f"{path}/{i}") for i, (x, y) in enumerate(zip(a, b)))
    assert a == b, (path, a, b)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--save", type=Path)
    group.add_argument("--compare", type=Path)
    args = parser.parse_args()
    result = collect()
    if args.save:
        torch.save(result, args.save)
        print("saved legacy snapshot", flush=True)
    else:
        reference = torch.load(args.compare, weights_only=True)
        print(f"flag-off bit identity: {compare(reference, result)} tensors matched", flush=True)


if __name__ == "__main__":
    main()
