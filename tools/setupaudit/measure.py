"""Read-only paired candidate-A objective audit; never calls update or a simulator."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import time

import torch

from wms.control.antiexploit import tail_credit
from wms.control.contract import RewardWeights, OWN_FEATURES
from wms.control.counterfactual import (CounterfactualConfig, baseline_index, build_variants,
                                        decompose, row_coefficients)
from wms.control.coopw import ExplorationConfig, HorizonConfig
from wms.control.dreamer import DreamerConfig
from wms.control.gated300 import SIGMA, gate, install
from wms.control.mappo.eval_bridge import load_policy_fn
from wms.control.objective14 import authoritative_reward
from wms.control.obs import build_observations
from wms.control.policy import load_policy_components
from wms.control.policy14 import Policy14Trainer, make_controller_factory
from wms.control.policy14_cli import load_members
from wms.control.real_returns14 import ScaledTeamCritic
from wms.control.safety import safe_accel
from wms.control.train_cli import GroupSignal, _read_config

WM = "b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436"
MAPPO = "aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(x):
    x = x.detach().float().cpu().reshape(-1)
    return ({"n": x.numel(), "mean": float(x.mean()),
             "q": torch.quantile(x, torch.tensor([0., .01, .1, .5, .9, .99, 1.])).tolist()}
            if x.numel() else {"n": 0})


def credits(trainer, imagined, horizon):
    """Call the actual actor loss, and expose its paired decomposition and gates."""
    weights = trainer.counterfactual.weights(190)
    v = imagined["variants"]
    r = imagined["returns"]
    base = baseline_index(v, r.shape[-1], device=r.device)
    difference = r - r[..., base]
    c = row_coefficients(v, weights, r.shape[-1], imagined["n_base"], device=r.device)
    selected, _ = tail_credit(imagined, difference, c, enabled=True)
    # REINFORCE log probabilities are gradient estimators, not a value function
    # for comparing two different actors. Candidate A has no lane head at all.
    criterion = dict(imagined, lanes=None)
    loss, metrics = trainer.actor_objective(criterion, 1., weights)
    totals = imagined["rewards"].sum(1)
    improvement = totals - totals[..., base]
    gated, on, agreement, median = gate(improvement[:, None], SIGMA.get(horizon, 25.38),
                                       c != 0, gain_only=True)
    pseudo = dict(imagined, returns=totals[:, None])
    long_selected, _ = tail_credit(pseudo, gated, c, enabled=True)
    dense = float((selected.mean((0, 1)) * c).sum())
    j = -totals
    out = dict(J=float(j[:, :imagined["n_base"]].mean()),
               road=float(imagined["components"]["delay"][..., :1].sum(1).mean()),
               source=float(imagined["components"]["source_wait"][..., :1].sum(1).mean()),
               dense_credit=dense, objective_loss=float(loss),
               decomposition=decompose(r.mean((0, 1)), v),
               accrued_decomposition=decompose(totals.mean(0), v),
               coefficients=c.tolist(), row_kind=v.row_kind.tolist(),
               group_members=[g["members"] for g in v.groups],
               gain_only_on=on.tolist(), gate_agreement=agreement.tolist(),
               gate_median=median.tolist(), gate_q10=torch.quantile(improvement, .1, dim=0).tolist(),
               long_credit=float((long_selected.mean((0, 1)) * c).sum()),
               all_J_futures=j[:, 0].tolist(),
               native_J_futures=j[:, v.base_row[0]].tolist(),
               actor_metrics=metrics)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wm", default="/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt")
    p.add_argument("--wm-config", default="/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml")
    p.add_argument("--mappo", default="/root/autodl-tmp/c4/mappores20261028/inputs/mappo_v5_final.pt")
    p.add_argument("--candidate", default="policy_000190.pt")
    p.add_argument("--starts", default="starts.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--actors", nargs="+", default=["CTRL0", "MAPPO", "candidateA"])
    p.add_argument("--n-starts", type=int, default=2)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--noisy", action="store_true")
    p.add_argument("--mix", action="store_true")
    p.add_argument("--legacy-native-bridge", action="store_true")
    a = p.parse_args()
    from inspect_inputs import allowed
    assert all(allowed(x) for x in (a.wm, a.wm_config, a.mappo, a.candidate, a.starts, a.out))
    assert digest(a.wm) == WM and digest(a.mappo) == MAPPO
    torch.set_num_threads(2)
    if a.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(.35)
    raw = _read_config("configs/c4_policy14_candA_longgrad_skip.toml")
    payload = torch.load(a.candidate, map_location="cpu", weights_only=False)
    batches = torch.load(a.starts, map_location=a.device, weights_only=False)[:a.n_starts]
    assert all(allowed(m) for b in batches for m in b.metadata)
    models, receipt = load_members([a.wm], [a.wm_config], a.device)
    cfg = DreamerConfig(**payload["config"])
    if a.mix:
        cfg = replace(cfg, entrant_learned_mix=True)
    if a.probe:
        cfg = replace(cfg, horizon=2)
    results = dict(schema="setupaudit/1", pid=os.getpid(), device=a.device,
                   gpu_fraction=.35 if a.device == "cuda" else None,
                   wm=receipt, candidate_sha256=digest(a.candidate), mappo_sha256=digest(a.mappo),
                   starts_sha256=digest(a.starts), noisy=a.noisy,
                   learned_mix=a.mix, matched_native_background=not a.legacy_native_bridge, rows=[])
    original_rollout = models[0].rollout
    future_receipts = []

    def progress_rollout(*args, **kwargs):
        begin = time.perf_counter()
        result = original_rollout(*args, **kwargs)
        sampler = kwargs.get('entrant_driver_sampler')
        future_receipts.append(dict(seconds=time.perf_counter() - begin,
                                    mix_assigned=None if sampler is None else sampler.assigned))
        print('FUTURE', len(future_receipts), future_receipts[-1], flush=True)
        return result

    models[0].rollout = progress_rollout
    out = Path(a.out)
    assert not out.exists() and out.parent.is_dir()
    print("OWN_PID", os.getpid(), "DEVICE", a.device, flush=True)
    for actor in a.actors:
        for bi, original in enumerate(batches):
            begin = time.perf_counter()
            torch.manual_seed(965140)
            policy, encoder, rules, _ = load_policy_components(a.candidate, device=a.device)
            if actor == "CTRL0":
                torch.nn.init.zeros_(policy.residual_head.weight)
                torch.nn.init.zeros_(policy.residual_head.bias)
            if actor.startswith("MAPPO"):
                executor = load_policy_fn(a.mappo, device=a.device)
                policy, encoder, rules = executor.policy, executor.group_encoder, executor.lane_rules
            else:
                encoder = encoder.encoder if isinstance(encoder, GroupSignal) else encoder
                rules = None
                policy.residual_additive_exploration = True
            native_lane = actor == "MAPPO_native_lanes"
            if native_lane:
                rules = None
            trainer = Policy14Trainer(models, policy.eval(), GroupSignal(encoder).eval(),
                ScaledTeamCritic(entry_backlog=True, value_scale=10000.), None,
                observation_builder=build_observations, reward_fn=authoritative_reward,
                safety=safe_accel,
                controller_factory=make_controller_factory(policy_ramp_merge=bool(rules),
                    native_ramp_merge="wm_law", native_lane_change=True),
                config=replace(cfg, exploit_monitor=not actor.startswith("MAPPO"),
                               native_residual_train_group=not actor.startswith("MAPPO"),
                               native_residual_train_trunk=not actor.startswith("MAPPO")),
                reward_weights=RewardWeights(**raw["reward"]), device=a.device, lane_rules=rules,
                counterfactual=CounterfactualConfig(**raw["counterfactual"]),
                real_return_provider=lambda **kw: None,
                exploration=ExplorationConfig(**raw["exploration"]),
                horizon=None if a.probe else HorizonConfig(**raw["horizon"]))
            trainer._tilt_sampler = None
            if trainer.horizon_curriculum is not None:
                # Same WM-bound 90 s dense stage recorded by the production launch.
                trainer.horizon_curriculum.allowed_stage = 1
            trainer.update_count = payload["update_count"]
            trainer.coop_anchor = payload["coopw"].get("anchor", 0)
            trainer.long_credit_requested = True
            install(trainer, dict(raw["long_credit"], enabled=not a.probe), [WM])
            trainer._training_imagination = True
            delta, reachable, residuals, lane_changes, obs_values, signals, entrants = [], [], [], [], [], [], []
            factory = trainer.controller_factory

            def observed_factory(*args, **kwargs):
                callback = factory(*args, **kwargs)

                def callback_audit(state):
                    from wms.control.alwayscmd import executed, inferred_cap
                    if actor.startswith('MAPPO') and not a.legacy_native_bridge:
                        # The standard absolute bridge uses an inferred IDM cap
                        # and ballistic motion. Supply the SAME exact speedMode-27
                        # cap as candidate A, then integrate semi-implicit Euler.
                        from wms.model.cav import EXEC_CAP_KEY, WM_MERGE_KEY
                        callback_state = {**state, EXEC_CAP_KEY: inferred_cap(state)}
                        command = callback(callback_state)
                        if WM_MERGE_KEY in callback_state:
                            state[WM_MERGE_KEY] = callback_state[WM_MERGE_KEY]
                        applied = torch.maximum(-state['v_mps'] / .5,
                            torch.minimum(command[..., 3], callback_state[EXEC_CAP_KEY]))
                        applied_v = (state['v_mps'] + .5 * applied).clamp_min(0.)
                        command = torch.stack((state['s_m'] + .5 * applied_v,
                                                command[..., 1], applied_v, applied), -1)
                    else:
                        command = callback(state)
                    mask = state["present"].bool() & state["is_cav"].bool() & (state["lane"] >= 0)
                    if actor.startswith('MAPPO') and not a.legacy_native_bridge:
                        # MAPPO's absolute-action bridge must not change the paired
                        # NATIVE background: candidate A executes its non-acting rows
                        # with project_native + Euler, without the absolute bridge's
                        # soft safety/cap. Keep MAPPO's own acting actions and lanes.
                        from wms.control.continuous import project_native
                        native_accel = project_native(state, kwargs['native'](state))
                        native_v = (state['v_mps'] + .5 * native_accel).clamp_min(0.)
                        native_s = state['s_m'] + .5 * native_v
                        native_command = torch.stack((native_s, command[..., 1], native_v,
                                                       native_accel), -1)
                        nonacting = mask & ~kwargs['act_mask']
                        command = torch.where(nonacting[..., None], native_command, command)
                    # Only all-CAV row: actor vs CTRL0 at IDENTICAL geometry, every .5 s.
                    m = mask[0]
                    z = torch.zeros_like(state["v_mps"])
                    ctrl = executed(state, z)
                    low, high = executed(state, z - 1.2), executed(state, z + 1.2)
                    d = command[0, :, 3][m] - ctrl[0][m]
                    delta.append(d.cpu())
                    reachable.append(((command[0, :, 3][m] >= low[0][m] - 1e-5) &
                                      (command[0, :, 3][m] <= high[0][m] + 1e-5)).cpu())
                    lane_changes.append((command[0, :, 1][m] != state["lane"][0][m]).cpu())
                    entrants.append((state.get("entrant", torch.zeros_like(mask))[0][m]).cpu())
                    return command

                if hasattr(callback, "native_lane_mask"):
                    callback_audit.native_lane_mask = callback.native_lane_mask
                return callback_audit

            def policy_hook(module, args, output):
                obs, signal = args[:2]
                m = obs.cav_mask[0]
                obs_values.append(obs.own[0][m].detach().cpu())
                signals.append(signal[0][m].detach().cpu())
                residuals.append(output[0][0][m].detach().cpu())

            trainer.controller_factory = observed_factory
            handle = policy.register_forward_hook(policy_hook)
            batch = trainer._entry_slots(copy.deepcopy(original), torch.Generator().manual_seed(1_000_003 * 191))
            variants = build_variants(batch.state, trainer.counterfactual, torch.Generator().manual_seed(965140))
            print("BEGIN", actor, bi, original.metadata, "slots", batch.state["s_m"].shape,
                  "rows", variants.n_rows, flush=True)
            future_receipts.clear()
            with torch.no_grad():
                imagined = trainer.imagine(batch, deterministic=not a.noisy, variants=variants, future_seed=190)
                dense_h = int(imagined["imagination_horizon_s"])
                row = dict(actor=actor, start=bi, metadata=original.metadata,
                           slots=batch.state["s_m"].shape[-1], n_rows=variants.n_rows,
                           dense_horizon=dense_h, dense=credits(trainer, imagined, dense_h))
                if not a.probe:
                    # Gated300 exposes accrued 300 s rewards as paired differences;
                    # reconstruct suffix returns only for the raw diagnostic decomposition.
                    ld = imagined["gated300_long_difference"]
                    c = row_coefficients(variants, trainer.counterfactual.weights(190),
                                         variants.n_rows, 1, device=ld.device)
                    row["long"] = dict(J=float(imagined["gated300_long_J"][0]),
                        row_J=imagined["gated300_long_J"].tolist(),
                        accrued_decomposition=decompose(ld.mean((0, 1)), variants),
                        median=torch.quantile(ld[:, 0], .5, dim=0).tolist(),
                        q10=torch.quantile(ld[:, 0], .1, dim=0).tolist(),
                        **{k: val for k, val in row["dense"]["actor_metrics"].items()
                           if k.startswith("gated300_")})
                    row["long"]["all_gain_futures"] = ld[:, 0, 0].tolist()
                row["seconds"] = time.perf_counter() - begin
                row['future_receipts'] = copy.deepcopy(future_receipts)
                row["action_delta_vs_ctrl0"] = stats(torch.cat(delta))
                row["within_residual_envelope_share"] = float(torch.cat(reachable).float().mean())
                row["positive_delta_share"] = float((torch.cat(delta) > 1e-5).float().mean())
                row["requested_action"] = stats(torch.cat(residuals))
                row["bridge_lane_change_share"] = float(torch.cat(lane_changes).float().mean())
                row["entrant_cav_tick_share"] = float(torch.cat(entrants).float().mean())
                own = torch.cat(obs_values)
                row["own_nonzero_share"] = dict(zip(OWN_FEATURES, (own != 0).float().mean(0).tolist()))
                row["signal"] = stats(torch.cat(signals))
                results["rows"].append(row)
                out.write_text(json.dumps(results, indent=2) + "\n")
                print("DONE", actor, bi, row["seconds"], row["dense"]["J"],
                      row.get("long", {}).get("J"), row["dense"]["dense_credit"], flush=True)
            handle.remove()
            del imagined, trainer
    print("COMPLETE", a.out, flush=True)


if __name__ == "__main__":
    main()
