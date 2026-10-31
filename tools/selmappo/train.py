"""Fresh baseline training; SUMO requires MAIN approval and peer integration.

Use the existing CPU environment: sequential mechanics, then bounded spawn waves.
"""
import argparse
import json
import os
import time

from .environment import SelectorEnv
from .protocol import (BLOCKS, EPISODES, ITERATIONS, LIBRARY, TIMES,
                        approval_line, audit_registration, digest, safe_path, training_world)
from .integration import evaluation_panel, load as load_integration
from .storage import Budget, append


def worlds(variant, iteration):
    start = BLOCKS[variant][0]+iteration*EPISODES
    return [training_world(start+i, variant) for i in range(EPISODES)]


def episode(model, world, args, budget, *, evaluation=False, smoke=False):
    import torch
    from .learning import Transition
    from .runtime import Backend
    root = safe_path(args.out)/str(world['seed'])
    root.mkdir(exist_ok=False)
    initialized = time.monotonic()
    backend = Backend(world=world, variant=args.variant, registration=args.registration,
                       c1_root=args.c1_root, episode_root=root, budget=budget,
                       integration=args.integration, design_sha=args.design_sha,
                       evaluation=evaluation, smoke=smoke,
                       panel=getattr(args, 'panel_seeds', ()))
    start = time.monotonic()
    initialization_wall = start-initialized
    warmup_tape = backend.digest.hexdigest()
    generator = torch.Generator().manual_seed(world['seed'])
    transitions, total, metrics = [], 0., {}
    try:
        env = SelectorEnv(backend, args.variant)
        for decision in range(len(TIMES)):
            if (safe_path(args.out)/'STOP_REQUESTED').exists():
                raise RuntimeError('cooperative stop')
            obs = env.observation
            actions, logp = model.act(obs, deterministic=evaluation, generator=generator)
            if smoke:
                # Every library hand-over is exercised, irrespective of initial logits.
                actions = {key: decision % len(LIBRARY) for key in obs.keys}
            _, reward, done, components = env.step(actions)
            transitions.append(Transition(obs, actions, logp, reward))
            total += -reward
            for key, value in components.items():
                metrics[key] = metrics.get(key, 0.)+value
            if done != (decision == len(TIMES)-1):
                raise ValueError('selector terminal mismatch')
        append(root/'result.jsonl', dict(world=world, variant=args.variant, evaluation=evaluation,
                                        smoke=smoke, J600=total, **metrics,
                                        warmup_tape_sha256=warmup_tape,
                                        tape_sha256=backend.digest.hexdigest(),
                                        design_sha256=args.design_sha,
                                        initialization_warmup_wall_s=initialization_wall,
                                        control_wall_s=time.monotonic()-start,
                                        timing=backend.timing, actual_sim_s=backend.actual))
        return transitions
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', required=True, choices=('fleet',))
    parser.add_argument('--registration', required=True)
    parser.add_argument('--c1-root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--integration', required=True)
    parser.add_argument('--design-sha', required=True)
    parser.add_argument('--panel')
    parser.add_argument('--panel-sha')
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--continue-from', help='MAIN-authorized exact stopped original build only')
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        raise ValueError('workers must be1..12')
    # Refuse before importing torch, runtime, or creating any simulation directories.
    if not approval_line(args.registration):
        raise PermissionError('S-SELMAPPO approval absent; no SUMO may run')
    if not audit_registration(args.registration)['passed']:
        raise ValueError('intervening seed allocation conflict')
    load_integration(args.integration, args.registration, args.c1_root, args.design_sha)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise ValueError('CPU-only nice19 required')
    import torch
    from .learning import Selector, Updater
    torch.set_num_threads(1)
    torch.manual_seed(0 if args.variant == 'fleet' else 1)
    out = safe_path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    # Exclusive allocation never expires/auto-overwrites, including after failures.
    with (out/'OWNER.json').open('x', encoding='utf-8') as stream:
        json.dump(dict(pid=os.getpid(), variant=args.variant, approval=approval_line(args.registration),
                       registration_sha256=digest(args.registration)), stream)
    if args.continue_from:
        from .continuation import inherit
        inherit(out, args.continue_from, args.registration)
    budget, model = Budget(out), Selector()
    updater, start = Updater(model), time.monotonic()
    optimization_rng = torch.get_rng_state()
    first_smoke = BLOCKS[args.variant][1] if args.continue_from else BLOCKS[args.variant][1]-1
    for seed in range(first_smoke, BLOCKS[args.variant][1]+1):
        episode(model, dict(seed=seed, penetration=.4, demand=2700), args, budget, smoke=True)
    append(out/'status.jsonl', dict(status='MECHANICS_COMPLETE', charged_sim_s=budget.charged,
                                  wall_s=time.monotonic()-start,
                                  successful_episodes=1 if args.continue_from else 2))
    if args.smoke_only:
        append(out/'status.jsonl', dict(status='SMOKE_COMPLETE', charged_sim_s=budget.charged))
        return
    torch.set_rng_state(optimization_rng)
    from .dispatch import wave
    for iteration in range(ITERATIONS):
        if time.monotonic()-start >= 86400 or (out/'STOP_REQUESTED').exists():
            raise RuntimeError('wall limit or cooperative stop')
        begin = time.monotonic()
        episodes = wave(model, worlds(args.variant, iteration), args, budget, start+86400)
        simulation_wall = time.monotonic()-begin
        stats = updater.update(episodes)
        elapsed = time.monotonic()-begin
        append(out/'training.jsonl', dict(iteration=iteration+1, **stats,
                                         simulation_wall_s=simulation_wall, total_wall_s=elapsed,
                                         sim_s_per_hour=EPISODES*900/elapsed*3600,
                                         charged_sim_s=budget.charged))
        # Retain every checkpoint/optimizer/scaler; no rolling deletion.
        torch.save(dict(model=model.state_dict(), actor_opt=updater.actor_opt.state_dict(),
                         critic_opt=updater.critic_opt.state_dict(), scaler=vars(updater.scaler),
                         iteration=iteration+1, protocol='S-C1SWITCH2/v2', library=LIBRARY,
                         design_sha256=args.design_sha, rng=torch.get_rng_state()),
                   out/f'checkpoint_{iteration+1:03d}.pt')
    if not args.panel or not args.panel_sha:
        append(out/'status.jsonl', dict(status='TRAIN_COMPLETE_AWAIT_REGISTERED_PANEL',
                                      charged_sim_s=budget.charged))
        return
    panel = evaluation_panel(args.panel, args.panel_sha, args.registration)
    args.panel_seeds = tuple(w['seed'] for w in panel)
    model.eval()
    for world in panel:
        episode(model, world, args, budget, evaluation=True)


if __name__ == '__main__':
    main()
