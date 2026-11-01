"""Development-only paired futures using the frozen five-arm WM evaluator.

Filter starts BEFORE opening artifacts; preserve full600s/K8 RNG planning but
score an explicitly labelled fixed K subset. No reserved-world input is opened.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def restrict_start_source(code, worlds):
    if not worlds or any(isinstance(w, bool) or not isinstance(w, int) or not 969420 <= w <= 969431 for w in worlds):
        raise ValueError('development panel969420..969431 only')
    marker = "    worlds = json.loads((DATA/'design.json').read_text())['worlds']"
    if code.count(marker) != 1:
        raise ValueError('registered start producer changed')
    return code.replace(marker, marker+f"\n    worlds = [w for w in worlds if w['seed'] in {tuple(worlds)!r}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--world', type=int, required=True)
    p.add_argument('--futures', type=int, default=2)
    p.add_argument('--params', choices=('off', 'learned_v1'), required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--artifact', type=Path)
    args = p.parse_args()
    if args.world not in range(969420, 969432) or not 1 <= args.futures <= 8:
        p.error('explicit development world and K1..8 required')
    root = Path(__file__).resolve().parents[2]
    sys.path[:0] = [str(root), str(root/'src'), str(root/'tools/rbeval')]
    os.environ.update(WMPHYS_VARIANT='FULL', C4_ALLOWED_ROOTS='/root/autodl-tmp/c4:/root/wmcalib:/root/rbeval20261031',
        C4_EXECUTION_PARITY='2', C4_SUMO_LEADER_RULE='1', C4_EXECUTION_DRIVER=str(root/'configs/execparity_cav_idm_v1.json'))
    import numpy as np
    import torch
    from tools.rbint import recipe
    from tools.rbeval import run_wm
    from tools.rbeval.core import reference, load, save_new, digest
    torch.set_num_threads(1)
    assert args.out.parent.is_dir() and not args.out.exists()
    # Keep immutable recipe outside task output, which the registered worker creates.
    source_recipe = root/'configs/wm_phys.toml'
    text = source_recipe.read_text()
    artifact = args.artifact or root/'artifacts/rb2lphys/learned_v1.json'
    if args.params != 'off':
        from wms.physics.learned_parameters import load_parameters
        load_parameters(str(artifact.resolve()))
        text = text.replace('[physics]', '[physics]\nphysics_params = "learned_v1"\nphysics_params_file = '+json.dumps(str(artifact.resolve())))
    config = args.out.parent/(args.out.name+'_recipe.toml')
    with config.open('x') as f:f.write(text)
    original_reference = run_wm.reference
    def dev_reference(key, namespace):
        if key == 'starts':
            ref = load(root/'tools/rbeval/references.json')['starts']
            assert hashlib.sha256(ref['code'].encode()).hexdigest() == ref['code_sha256']
            code = restrict_start_source(ref['code'], [args.world])
            exec(compile(code, str(Path(__file__))+' [development starts]', 'exec'), namespace)
            return hashlib.sha256(code.encode()).hexdigest()
        result = original_reference(key, namespace)
        if key == 'costs':
            original_costs = namespace['costs']
            def costs(roll, entries, initial):
                values = original_costs(roll, entries, initial)
                lanes = torch.stack([s['lane'] for s in entries.states], 1)
                present = torch.stack([s['present'] for s in entries.states], 1).bool()
                born = torch.stack(entries.new_masks, 1).bool()
                change = present & ~born & roll['masks'].bool() & (lanes != roll['lanes'])
                ramp_road = .5*torch.where(present & ~born & (lanes == 2),
                    (1-roll['speeds']/31.29).clamp_min(0), 0).sum(-1)
                for h in (120,300,600):
                    n = 2*h
                    values[str(h)]['lp_lane_changes'] = change[:, :n].sum((1,2)).cpu().tolist()
                    values[str(h)]['lp_ramp_lane_road'] = ramp_road[:, :n].sum(-1).cpu().tolist()
                    source = np.stack(entries.lp_ramp_source, 1)
                    values[str(h)]['lp_ramp_source'] = source[:, :n].sum(-1).tolist()
                values['lp_lane_changes_10s'] = change.sum(-1).reshape(change.shape[0],60,20).sum(-1).cpu().tolist()
                return values
            namespace['costs'] = costs
        return result
    def future_builder(**kwargs):
        kw, entries, receipt = recipe.candidate_future(**kwargs)
        entries.lp_ramp_source = []
        process = entries.process
        next_wait = process._next
        area = np.zeros(process.batch)
        def next_ramp(b, e):
            if e == 1:area[b] += max(0., process.now[b]-process.t_next[b,e])
            return next_wait(b,e)
        process._next = next_ramp
        insert = entries.insert
        def record_insert(state):
            area[:] = 0
            pending = process.pending[:,1].copy()
            result = insert(state)
            entries.lp_ramp_source.append(.5*pending+area.copy())
            return result
        entries.insert = record_insert
        receipt.update(physics_params=args.params, development_only=True, ensemble_size=args.futures)
        return kw, entries, receipt
    def save(path, data):
        if Path(path).name == 'IDENTITY.json':
            data.update(schema='rb2lphys.development/1', K=args.futures,
                planned_K=8, evaluation_worlds=[args.world], development_only=True,
                purpose='paired development comparison; not final qualification')
        if isinstance(data, dict) and data.get('schema') == 'rbeval/1':
            data['schema'] = 'rb2lphys.development/1'
        return save_new(path, data)
    run_wm.reference = dev_reference
    run_wm.SEEDS = (args.world,)
    run_wm.save_new = save
    # Only the worker's K loop is narrowed; the future builder still plans K8.
    run_wm.range = lambda *a: range(args.futures) if a == (8,) else range(*a)
    out = str(args.out)
    sys.argv = [sys.argv[0], '--out', out, '--runtime-root', str(root),
        '--wm-config', str(config), '--model-name', 'LPDEV_'+args.params,
        '--device', args.device, '--memory-gib', '5', '--max-tasks', str(args.futures)]
    sources = [Path(__file__), source_recipe, root/'tools/rbint/recipe.py', root/'tools/rbint/departure.py',
        root/'tools/rbint/observations.py', root/'tools/rbint/motion_posterior.py',
        root/'artifacts/rbfit/shrinkage_v1/idm_population.json',
        root/'artifacts/rbfit/shrinkage_v1/idm_posteriors.json.gz',
        root/'artifacts/rbfit/shrinkage_v1/idm_shrinkage.json',
        root/'docs/paper_results/rbunc/observed_arrivals.json']
    if args.params != 'off':sources.append(artifact)
    run_wm.main(model_loader=recipe.load_candidate, identity_sources=sources, future_builder=future_builder)
    save_new(args.out/'LP_COMPLETE.json', dict(world=args.world,K=args.futures,params=args.params,
        tasks=[dict(file=f'w{args.world}_k{k}.json',sha256=digest(args.out/f'w{args.world}_k{k}.json')) for k in range(args.futures)]))


if __name__ == '__main__':
    main()
