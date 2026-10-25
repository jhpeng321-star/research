"""Reproducible, synthetic S-SPEED2 timings/equivalence (no traffic data is read).

Run with PYTHONPATH=src python tools/speed2_measure.py --case all --device cuda
--output <receipt.json>. GPU allocations are capped at 3.5 GiB; CPU affinity is
restricted to two allowed CPUs, with one Torch/BLAS thread and nice 19 on Linux.
Times are synchronized wall times; each comparison resets weights and RNG.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
if hasattr(os, "nice") and os.nice(0) < 19:
    os.nice(19 - os.nice(0))

import torch  # noqa: E402
from wms.core import speed_flags  # noqa: E402
from wms.model import ModelSpec, WmSModel  # noqa: E402
from wms.model.residual_net import HdvConfig, HdvResidualNet  # noqa: E402
from wms.physics.config import PhysicsConfig  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@contextlib.contextmanager
def flags(**values):
    old = {env: os.environ.get(env) for env in speed_flags.FLAGS.values()}
    try:
        for env in old:
            os.environ.pop(env, None)
        for name, value in values.items():
            os.environ[speed_flags.FLAGS[name]] = str(value)
        yield
    finally:
        for env, value in old.items():
            if value is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = value


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    times = []
    result = None
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        result = fn()
        sync()
        times.append(time.perf_counter() - start)
    return result, {"median_s": statistics.median(times), "samples_s": times}


def diff(a, b):
    """Max abs / reference-relative difference; nonfinite patterns must match."""
    if isinstance(a, dict):
        return {key: diff(a[key], b[key]) for key in a}
    if isinstance(a, (list, tuple)):
        rows = [diff(x, y) for x, y in zip(a, b)]
        return {"max_abs": max((x["max_abs"] for x in rows), default=0.),
                "max_rel": max((x["max_rel"] for x in rows), default=0.),
                "bit_equal": all(x["bit_equal"] for x in rows)}
    if a is None:
        assert b is None
        return {"max_abs": 0., "max_rel": 0., "bit_equal": True}
    a, b = torch.as_tensor(a).cpu(), torch.as_tensor(b).cpu()
    assert a.shape == b.shape
    if not a.is_floating_point():
        return {"max_abs": int((a != b).sum()), "max_rel": 0., "bit_equal": torch.equal(a, b)}
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    assert torch.equal(torch.isposinf(a), torch.isposinf(b))
    assert torch.equal(torch.isneginf(a), torch.isneginf(b))
    finite = torch.isfinite(a)
    delta = (a[finite].double() - b[finite].double()).abs()
    return {"max_abs": float(delta.max()) if delta.numel() else 0.,
            "max_rel": float((delta / a[finite].double().abs().clamp_min(1e-8)).max())
            if delta.numel() else 0., "bit_equal": torch.equal(a, b)}


def summary(tree):
    """Reduce a nested difference receipt without obscuring near-zero relative errors."""
    if "max_abs" in tree:
        return tree
    leaves = [summary(value) for value in tree.values()]
    return {"max_abs": max((value["max_abs"] for value in leaves), default=0.),
            "max_rel": max((value["max_rel"] for value in leaves), default=0.),
            "bit_equal": all(value["bit_equal"] for value in leaves)}


def scene(device, k=595, p=100):
    g = torch.Generator().manual_seed(741)
    kin = torch.zeros(1, k, 7)
    # Stable noncontiguous slot packing, about 100 present of 595 slots.
    slots = torch.randperm(k, generator=g)[:p].sort().values
    kin[0, slots, 0] = torch.linspace(-1200., 300., p)
    lanes = torch.arange(p) % 2
    lanes[:2] = 2
    kin[0, slots[:2], 0] = torch.tensor([-5., 22.])
    kin[0, slots, 1:4] = torch.nn.functional.one_hot(lanes, 3).float()
    kin[0, slots, 4] = 8. + torch.rand(p, generator=g) * 7
    kin[0, slots, 6] = 5.
    present = torch.zeros(1, k, dtype=torch.bool)
    present[0, slots] = True
    cav = torch.zeros_like(present)
    cav[0, slots[::5]] = True
    return kin.to(device), present.to(device), cav.to(device)


def model(device):
    torch.manual_seed(742)
    return WmSModel(ModelSpec(rollout_mode="hybrid", lane_model="hazard_v5",
                              lane_decision="sample", merge_decision="first_passage",
                              merge_passage_law="hybrid", merge_w_init=.45),
                    physics=PhysicsConfig()).to(device)


def net_case(device):
    torch.manual_seed(740)
    net = HdvResidualNet(HdvConfig()).to(device)
    kin, present, _ = scene(device)
    results, times = [], []
    for flag in (0, 1):
        with flags(compact_slots=flag):
            # Keep the measured closure simple: gradients include parameters and kinematic input.
            def step():
                net.zero_grad(set_to_none=True)
                x = kin.detach().clone().requires_grad_()
                out = net(x, present)
                loss = (out["accel_delta_mps2"].square().sum()
                        + (out["accel_log_sigma"].square() * present).sum()
                        + out["lane_logit_delta"].square().sum()) / present.sum()
                loss.backward()
                return ({key: val.detach().cpu() for key, val in out.items()}, loss.detach().cpu(),
                        [x.grad.detach().cpu()] + [p.grad.detach().cpu() if p.grad is not None else None
                                                  for p in net.parameters()])
            result, timing = timed(step, repeats=5)
            results.append(result)
            times.append(timing)
    return {"off": times[0], "on": times[1], "outputs": diff(results[0][0], results[1][0]),
            "loss": diff(results[0][1], results[1][1]), "gradients": diff(results[0][2], results[1][2])}


def kernels_case(device):
    from wms.control.bridge import _merge_lane
    from wms.data.history import long_history_features
    from wms.train.cav_drive import TimelineControllers
    from wms.train.inputs import state_from_kin
    kin, present, cav = scene(device)
    state = state_from_kin(kin, present)
    state["is_cav"] = cav
    m = model(device).eval()
    hkin = kin.expand(61, -1, -1).clone()
    mask = present.expand(61, -1)
    cont = torch.ones(60, kin.shape[1], device=device, dtype=torch.bool)
    ids = cav.long().unsqueeze(1) * 3
    timeline = dict(controller=ids, acceleration_mps2=torch.full_like(ids, -1., dtype=torch.float32),
                    target_speed_mps=torch.full_like(ids, 8., dtype=torch.float32))
    out = {}
    for name in ("fast_long_history", "fast_merge_lane", "fast_cav_drive"):
        values, times = [], []
        for flag in (0, 1):
            with flags(**{name: flag}), torch.no_grad():
                ctrl = TimelineControllers(m, timeline)
                def step():
                    if name == "fast_long_history":
                        return long_history_features(hkin, mask, cont, present[0], history_s=30.)
                    if name == "fast_merge_lane":
                        return _merge_lane(state, cav)
                    ctrl.step_index = 0
                    return ctrl(state)
                value, timing = timed(step, repeats=5)
                values.append(value)
                times.append(timing)
        out[name] = {"off": times[0], "on": times[1], "difference": diff(values[0], values[1])}
    return out


def train_case(device, steps):
    from wms.train.losses.compose import build_loss
    from wms.train.settings import TrainSettings
    from wms.train.unroll import UnrollRunner
    kin, present, cav = scene(device)
    rec = kin[:, None].expand(-1, steps + 1, -1, -1).clone()
    rec[..., 0] += torch.arange(steps + 1, device=device)[None, :, None] * .5 * kin[:, None, :, 4]
    batch = dict(states=rec, present=present[:, None].expand(-1, steps + 1, -1),
                 is_cav=cav[:, None].expand(-1, steps + 1, -1))
    settings = TrainSettings(rollout_mode="hybrid", lane_model="hazard_v5",
                             merge_decision="first_passage", merge_passage_law="hybrid",
                             merge_core_tf_weight=.1, truncation_steps=1,
                             driver_grad_across_chunks=True, device=str(device))
    results, times = [], []
    for values in ({}, {"compact_slots": 1, "pivot_diag_off": 1}):
        with flags(**values):
            m = model(device).train()
            runner = UnrollRunner(m, settings, build_loss(settings, physics=m.physics),
                                  projection=None, device=torch.device(device))
            optimizer = torch.optim.Adam(m.parameters(), lr=1e-5)
            def step(optimizer=optimizer, runner=runner, m=m):
                optimizer.zero_grad(set_to_none=True)
                torch.manual_seed(743)
                metrics = runner.run(batch, unroll_steps=steps, backward=True)
                grads = [p.grad.detach().cpu().clone() if p.grad is not None else None
                         for p in m.parameters()]
                optimizer.step()
                return metrics["loss"], grads
            result, timing = timed(step, repeats=1, warmup=0)
            results.append(result)
            times.append(timing)
            del step, m, runner, optimizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return {"physics_steps": steps, "stage_s": steps * .5, "truncation_steps": 1,
            "off": times[0], "on": times[1], "loss": diff(results[0][0], results[1][0]),
            "gradients": diff(results[0][1], results[1][1])}


def replay_case(device):
    from types import SimpleNamespace
    from wms.train.losses.merge_pivot import MergePivotConfig, build_pre_window_history
    m = model(device).eval()
    prefixes, anchors, masks = [], [], []
    lengths, widths = [8, 12, 16, 10], [595, 531, 560, 580]
    for h, k in zip(lengths, widths):
        kin, present, _ = scene(device, k=k)
        rec = kin[:, None].expand(-1, h + 1, -1, -1).clone()
        rec[..., 0] += torch.arange(h + 1, device=device)[None, :, None] * .5 * kin[:, None, :, 4]
        prefixes.append(dict(states=rec[0], present=present.expand(h + 1, -1),
                             is_cav=torch.zeros(h + 1, k, dtype=torch.bool, device=device),
                             **{"continue": torch.ones(h, k, dtype=torch.bool, device=device)}))
        anchors.append(torch.nn.functional.pad(rec[:, -1:], (0, 0, 0, 595 - k)))
        masks.append(torch.nn.functional.pad(present[:, None], (0, 595 - k)))
    inp = SimpleNamespace(states=torch.cat(anchors), present=torch.cat(masks),
                          merge_pivot_prefix=prefixes)
    results, times = [], []
    for flag in (0, 1):
        with flags(batch_replay=flag), torch.no_grad():
            result, timing = timed(lambda: build_pre_window_history(m, inp, MergePivotConfig()),
                                   repeats=3)
            results.append(result)
            times.append(timing)
    # Certification is discrete and must be exact; geometry tolerates GEMM reassociation.
    differences = {str(i): diff(a, b) for i, (a, b) in enumerate(zip(*results))}
    return {"prefix_lengths": lengths, "prefix_widths": widths,
            "off": times[0], "on": times[1], "difference": summary(differences)}


def pivot_case(device):
    from wms.model.merge_pivot import update_state
    c = torch.full((1, 3), -3., device=device)
    live = torch.ones_like(c, dtype=torch.bool)
    tau, w = c.new_tensor(.7), c.new_tensor(.45)
    half = torch.full_like(c, .5)
    torch.manual_seed(747)
    with torch.no_grad():
        carry = update_state(c, tau, live, live, w=w, uniform=half, epsilon_uniform=half)
    t = torch.arange(240, device=device)
    carry["history"] = (-3. + .2 * torch.sin(t / 7.)[None, None]
                        + .1 * torch.arange(3, device=device)[None, :, None])
    carry["maximum"] = carry["history"].amax(-1)
    carry["decision_step"] = torch.full_like(carry["decision_step"], 240)
    carry.pop("cross")
    results, times = [], []
    for diagnostics in (True, False):
        with torch.no_grad():
            result, timing = timed(lambda: update_state(
                c, tau, live, live, carry=carry, w=w, epsilon_uniform=half, diagnostics=diagnostics),
                repeats=3)
            results.append({key: val for key, val in result.items()
                            if key not in ("logS", "event_mass", "cdf")})
            times.append(timing)
    return {"waiting_history_steps": 240, "ramp_slots": 3, "off": times[0], "on": times[1],
            "nondiagnostic_carry_difference": summary(diff(results[0], results[1]))}


def rollout_case(device, steps):
    from wms.train.inputs import state_from_kin
    kin, present, _ = scene(device)
    state = state_from_kin(kin, present)
    state["is_cav"] = torch.zeros_like(present)
    m = model(device).eval()
    results, times = [], []
    for values in ({}, {"pivot_diag_off": 1}, {"pivot_diag_off": 1, "compact_slots": 1}):
        with flags(**values), torch.no_grad():
            def step():
                torch.manual_seed(744)
                roll = m.rollout(state, n_steps=steps)
                return {key: roll[key].cpu() for key in
                        ("positions", "speeds", "accel", "lanes", "masks")}
            result, timing = timed(step, repeats=1, warmup=0)
            results.append(result)
            times.append(timing)
    return {"physics_steps": steps, "horizon_s": steps * .5,
            "off": times[0], "diag_off": times[1], "diag_off_compact": times[2],
            "diagnostics_only_difference": diff(results[0], results[1]),
            "compact_difference": diff(results[1], results[2])}


def policy_case(device, native_ramp_merge="gap6", native_cav="idm"):
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.dreamer import DreamerConfig
    from wms.control.lanes import LaneRules
    from wms.control.objective14 import authoritative_reward
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
    from wms.control.policy14 import Policy14Trainer, make_controller_factory
    from wms.control.safety import safe_accel
    from wms.control.start_states import StartBatch
    from wms.control.train_cli import GroupSignal
    from wms.train.inputs import state_from_kin

    kin, present, cav = scene(device, k=96, p=24)
    initial = state_from_kin(kin, present)
    initial["is_cav"] = cav
    class Sampler:
        def sample(self, batch_size, device="cpu"):
            return StartBatch({key: val.clone() for key, val in initial.items()})
    results, times = [], []
    regimes = ({}, {"grad_audit_every": 10},
               {"pivot_diag_off": 1, "fast_merge_lane": 1, "grad_audit_every": 10},
               {"pivot_diag_off": 1, "compact_slots": 1, "fast_merge_lane": 1, "grad_audit_every": 10})
    for values in regimes:
        with flags(**values):
            m = model(device).eval()
            torch.manual_seed(745)
            trainer = Policy14Trainer([m], CavPolicy(hidden_dim=32, lane_head=True),
                GroupSignal(RsuGroupEncoder(hidden_dim=32)), TeamCritic(hidden_dim=32), Sampler(),
                observation_builder=build_observations, reward_fn=authoritative_reward,
                safety=safe_accel, controller_factory=make_controller_factory(
                    native_ramp_merge=native_ramp_merge),
                lane_rules=LaneRules(consequence_mask=True),
                counterfactual=CounterfactualConfig(enabled=True, n_individual=0, n_group=1,
                                                    n_flip=1, group_state_fraction=0., native_cav=native_cav),
                config=DreamerConfig(batch_size=1, horizon=10, actor_gradient_audit=True,
                                     checkpoint_imagination=True, policy_lane_actions=True,
                                     checkpoint_every=0, validation_every=0), device=str(device))
            trainer.update_count = 1  # update 2: the audit-thinning flag skips this audit
            def step(trainer=trainer):
                torch.manual_seed(746)
                metrics = trainer.update()
                weights = {name: {key: val.detach().cpu().clone() for key, val in
                                 getattr(trainer, name).state_dict().items()}
                           for name in ("policy", "group_encoder", "critic", "target_critic")}
                return metrics, weights
            result, timing = timed(step, repeats=1, warmup=0)
            results.append(result)
            times.append(timing)
            del step, trainer, m
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    selected = ("actor_loss", "critic_loss", "actor_grad_norm", "critic_grad_norm", "returns")
    return {"slots": 96, "present": 24, "control_horizon": 10, "off": times[0],
            "native_ramp_merge": native_ramp_merge, "native_cav": native_cav,
            "audit_thinned": times[1], "exact_flags": times[2], "compact_and_exact": times[3],
            "comparisons": {name: {
                "metrics": diff({key: results[0][0][key] for key in selected},
                                {key: results[i][0][key] for key in selected}),
                "weights": summary(diff(results[0][1], results[i][1]))}
                for i, name in enumerate(("off", "audit_thinned", "exact_flags", "compact_and_exact"))
                if i},
            "counterfactual_rows": results[0][0].get("cf_rows")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", "net", "kernels", "train", "replay", "pivot", "rollout", "policy"),
                        default="all")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--rollout-steps", type=int, default=600)
    parser.add_argument("--native-ramp-merge", choices=("gap6", "wm_law"), default="gap6")
    parser.add_argument("--native-cav", choices=("idm", "head"), default="idm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.device.startswith("cuda"):
        props = torch.cuda.get_device_properties(torch.device(args.device))
        torch.cuda.set_per_process_memory_fraction(3.5 * 1024 ** 3 / props.total_memory)
        torch.cuda.reset_peak_memory_stats()
    receipt = {"torch": torch.__version__, "device": args.device, "synthetic": True,
               "threads": torch.get_num_threads(), "relative_denominator_floor": 1e-8,
               "slots": 595, "present": 100, "pid": os.getpid(),
               "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
               "nice": os.nice(0) if hasattr(os, "nice") else None,
               "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [ROOT / "tools/speed2_measure.py",
                                              *[ROOT / f"src/wms/{name}" for name in
                                                ("core/speed_flags.py", "model/residual_net.py",
                                                 "model/merge_pivot.py", "model/wms.py",
                                                 "train/cav_drive.py", "train/losses/merge_pivot.py",
                                                 "data/history.py", "control/bridge.py", "control/dreamer.py")]]}}
    cases = {"net": lambda: net_case(args.device), "kernels": lambda: kernels_case(args.device),
             "train": lambda: train_case(args.device, args.train_steps),
             "replay": lambda: replay_case(args.device),
             "pivot": lambda: pivot_case(args.device),
             "rollout": lambda: rollout_case(args.device, args.rollout_steps),
             "policy": lambda: policy_case(args.device, args.native_ramp_merge, args.native_cav)}
    for name, run in cases.items():
        if args.case not in ("all", name):
            continue
        receipt[name] = run()
        if args.device.startswith("cuda"):
            receipt["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        print(json.dumps({name: receipt[name]}, allow_nan=False), flush=True)
    return receipt


if __name__ == "__main__":
    main()
