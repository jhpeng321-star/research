"""CPU-capable action-adjoint and from-scratch optimisation readiness diagnostics."""
from __future__ import annotations

import copy

import torch

from .contract import PHYSICS_DT_S, RewardWeights
from .objective14 import authoritative_reward


def missing_history(model, state):
    """Correctly shaped, explicitly unobserved history for synthetic starts."""
    if not model.long_history:
        return {}
    from wms.core.constants import LONG_HISTORY_F
    shape = (*state["present"].shape, model.driver_history_len)
    return {"driver_history": torch.zeros(*shape, LONG_HISTORY_F,
                                          device=state["s_m"].device),
            "driver_history_mask": torch.zeros(shape, dtype=torch.bool,
                                               device=state["s_m"].device)}


def probe_state(device="cpu", *, free=False):
    def tensor(row, dtype=torch.float32):
        return torch.tensor([row], dtype=dtype, device=device)
    if free:
        positions, speeds, lanes, cav = [-120.], [8.], [0], [True]
    else:
        positions, speeds, lanes, cav = [25., 65., 0., -60.], [4., 7., 6., 8.], [2, 1, 1, 0], [True, False, False, False]
    return {"s_m": tensor(positions), "v_mps": tensor(speeds),
            "a_mps2": tensor([0.] * len(cav)), "lane": tensor(lanes, torch.long),
            "length_m": tensor([5.] * len(cav)), "present": tensor([True] * len(cav), torch.bool),
            "is_cav": tensor(cav, torch.bool), "entry_queue": tensor([3. if not free else 0.]).reshape(1)}


def action_cost(model, action, *, free=False, n_steps=4):
    """First one-second CAV impulse, paired stochastic futures, no teacher path."""
    state = probe_state(action.device, free=free)
    tick = 0
    def controller(current):
        nonlocal tick
        a = torch.zeros_like(current["v_mps"]) + (action if tick < 2 else action * 0)
        a = torch.where(current["is_cav"], a, torch.zeros_like(a))
        from .safety import safe_accel
        from wms.model.cav import apply_exec_cap, direct_command_motion
        a = safe_accel(current, a, accel_max=1.5)
        a = apply_exec_cap(current, a, current["is_cav"].bool())
        s, v, a, _ = direct_command_motion(current["s_m"], current["v_mps"],
                                          torch.stack((a, current["v_mps"] + .75), -1),
                                          PHYSICS_DT_S, -3., 1.5)
        tick += 1
        return torch.stack((s, current["lane"].to(v.dtype), v, a), -1)
    torch.manual_seed(965148)
    roll = model.rollout(state, n_steps=n_steps, cav_controller=controller,
                         **missing_history(model, state))
    cost = action * 0
    for t in range(n_steps):
        nxt = {**state, "s_m": roll["positions"][:, t], "v_mps": roll["speeds"][:, t],
               "lane": roll["lanes"][:, t], "present": roll["masks"][:, t]}
        cost = cost + authoritative_reward(state, nxt, roll["accel"][:, t], RewardWeights())["J"].sum()
        state = nxt
    return cost


def _fd(model):
    a = torch.tensor(.2, device=next(model.parameters()).device, requires_grad=True)
    cost = action_cost(model, a)
    grad = float(torch.autograd.grad(cost, a)[0])
    differences = []
    with torch.no_grad():
        for eps in (.04, .02):
            fd = float((action_cost(model, a.detach() + eps)
                        - action_cost(model, a.detach() - eps)) / (2 * eps))
            differences.append({"epsilon_mps2": eps, "finite_difference": fd,
                                "absolute_error": abs(grad - fd),
                                "passed": abs(grad - fd) <= 0.0002 + .05 * abs(fd)})
    if not torch.isfinite(cost) or not all(x["passed"] for x in differences) or abs(grad) < 1e-7:
        raise ValueError(f"WM action derivative failed: grad={grad}, differences={differences}")
    return {"J": float(cost.detach()), "autograd": grad, "differences": differences}


def gradient_probe(model):
    """Exact checkpoint and separately labelled repaired-courtesy wiring diagnostic.

    S-R12C deliberately has an OFF, exactly-zero courtesy output head. It cannot
    demonstrate a nonzero learned courtesy response. A disposable *copy* with a
    seeded nonzero output layer exercises repair A without altering its checkpoint
    or the WM used for training. This is wiring evidence, never fidelity evidence.
    """
    diagnostic = copy.deepcopy(model).eval().requires_grad_(False)
    diagnostic.grad_decay_per_step = 1.0
    diagnostic.checkpoint_forward = False
    exact = _fd(diagnostic)
    head = getattr(diagnostic, "follower_courtesy", None)
    if head is None:
        raise ValueError("courtesy wiring check needs a WM with a follower-courtesy head")
    zero_head = not bool(head.net[-1].weight.abs().any() or head.net[-1].bias.abs().any())
    was_off = bool(diagnostic.follower_courtesy_off)
    diagnostic.follower_courtesy_state_grad = True
    diagnostic.follower_courtesy_intent = "pivot"
    diagnostic.follower_courtesy_off = False
    if zero_head:
        generator = torch.Generator(device=next(diagnostic.parameters()).device).manual_seed(965147)
        with torch.no_grad():
            head.net[-1].weight.copy_(.15 * torch.randn(head.net[-1].weight.shape,
                                      device=head.net[-1].weight.device, generator=generator))
    repaired = _fd(diagnostic)
    no_courtesy = copy.deepcopy(diagnostic)
    no_courtesy.follower_courtesy_off = True
    baseline = _fd(no_courtesy)
    isolated_autograd = repaired["autograd"] - baseline["autograd"]
    isolated = []
    for repair_fd, base_fd in zip(repaired["differences"], baseline["differences"]):
        difference = repair_fd["finite_difference"] - base_fd["finite_difference"]
        passed = abs(isolated_autograd - difference) <= 2e-5 + .1 * abs(difference)
        isolated.append({"epsilon_mps2": repair_fd["epsilon_mps2"],
                         "autograd": isolated_autograd, "finite_difference": difference,
                         "passed": passed})
    if not all(row["passed"] for row in isolated):
        raise ValueError(f"isolated courtesy contribution failed finite differences: {isolated}")
    diagnostic.follower_courtesy_state_grad = False
    a = torch.tensor(.2, device=next(diagnostic.parameters()).device, requires_grad=True)
    stopped = action_cost(diagnostic, a)
    stopped_grad = float(torch.autograd.grad(stopped, a)[0])
    contribution = repaired["autograd"] - stopped_grad
    if abs(contribution) < 1e-7:
        raise ValueError("courtesy adjoint did not contribute to dJ/d(CAV acceleration)")
    return {"exact_checkpoint": exact, "checkpoint_courtesy_off": was_off,
            "checkpoint_courtesy_zero_output_head": zero_head,
            "courtesy_probe": repaired, "courtesy_adjoint_contribution": contribution,
            "isolated_courtesy_fd": isolated,
            "courtesy_probe_injected_output_weights": zero_head,
            "courtesy_probe_training_model_modified": False,
            "gradient_decay": 1.0, "fixed_lane_event_regime_only": True}


def free_road_probe(model):
    """A separate scratch actor must learn to accelerate under exactly -J."""
    from .obs import build_observations
    from .policy import CavPolicy
    from .contract import GROUP_SIGNAL_DIM
    device = next(model.parameters()).device
    torch.manual_seed(965146)
    policy = CavPolicy(hidden_dim=16, lane_head=False, accel_max=1.5).to(device)
    optim = torch.optim.Adam(policy.parameters(), lr=.01)
    obs = build_observations(probe_state(device, free=True))
    group = torch.zeros(1, 1, GROUP_SIGNAL_DIM, device=device)
    first = None
    for step in range(13):
        action = policy(obs, group, deterministic=True)[0][0, 0]
        cost = action_cost(model, action, free=True, n_steps=2)
        if first is None:
            first = {"action_mps2": float(action.detach()), "J": float(cost.detach())}
        if step == 12:
            last = {"action_mps2": float(action.detach()), "J": float(cost.detach())}
            break
        optim.zero_grad(set_to_none=True)
        cost.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 100., error_if_nonfinite=True)
        optim.step()
    passed = last["J"] < first["J"] and last["action_mps2"] > first["action_mps2"] and last["action_mps2"] > .1
    if not passed:
        raise ValueError(f"free-road scratch actor failed: {first} -> {last}")
    return {"passed": passed, "updates": 12, "before": first, "after": last,
            "objective": "negative_unshaped_J", "demonstrations": False}
