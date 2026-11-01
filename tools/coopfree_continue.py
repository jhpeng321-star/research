"""Separately bound OWNER15:11 phase; never mutate/relaunch the build5 pipeline."""
import concurrent.futures as futures
import copy
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
import torch
from tools.coopfree import backend, runner, storage
from tools.coopfree.policy import SetPointPolicy, gaussian_log_prob
from tools.coopfree.protocol import digest, sha, worlds
from tools.coopfree.selection import select
from tools.coopfree.storage import completed, ledger, read, write
from tools.coopfree_hold import mean_hold, solo_hold, optimize, issued_difference, summary, relocate_completed
from tools.coopfree.credit import credit, next_sigma
from tools.coopfree_learning_checks import learning_check, paired_hold

LEGACY = Path('/home/ubuntu/coopfree20261101_build5')
ROOT = Path('/home/ubuntu/coopfree20261101_owner1511')
CAPS = dict(screen=27360, repeat=13680, train=5494000, dev=288000, discovery=132000)
CHECKS = (3, 6, 9)
WORKER = None


def configure():
    # Process-local accounting extension; protocol and immutable verification stay original.
    storage.CAPS = dict(CAPS)
    original = storage.admission
    if not getattr(original, '_owner1511', False):
        def polite(root, max_load=12.):
            return original(root, max_load=min(max_load, 12.))
        polite._owner1511 = True
        storage.admission = polite
        runner.admission = polite


def verify(root):
    cert = read(root/'CONTINUE_CPU.json')
    assert cert['pass_'] and cert['sim_s'] == 0 and not cert['skips']
    for name, expected in cert['sources'].items():
        assert sha(root/'pytools/tools'/name) == expected, 'continuation source mutation'
    assert sha(root/'CONTINUE_PRIOR.json') == cert['prior_sha256']
    authority = read(root/'CONTINUE_AUTHORITY.json')
    assert authority['total_cap'] == 6000000 and authority['caps'] == CAPS
    assert sha(root/'CONTINUE_DESIGN.md') == cert['design_sha256'] == authority['design_sha256']
    assert authority['approval_line'].startswith('- 15:11 S-COOPFREE BUDGET EXTENSION APPROVED')
    return runner.verify(root)


def load(root, update):
    torch.manual_seed(20261101)
    policy = SetPointPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=.001)
    state = torch.load(root/f'policy_hold_u{update:03}.pt', map_location='cpu', weights_only=True)
    policy.load_state_dict(state['policy'])
    optimizer.load_state_dict(state['optimizer'])
    return policy, optimizer, state['sigma'], state['enlargements']


def hold_key(w, sample):
    # rho/sigma are search metadata; they do not affect a deterministic absolute HOLD.
    return digest(dict(world=w['world'], t0=w['t0'], context=sample['context'],
                       action=sample['action'], window=sample['window']))


def cache(root):
    out = {}
    old = read(root/'HOLD_PLAN_001.json')
    original = read(root/'PLAN_001.json')
    for w, ident in zip(original['windows'], old['baseline_ids']):
        result = completed(root, ident)
        out[hold_key(w, result['sample'])] = ident
    for path in sorted(root.glob('CONTINUE_PLAN_*.json')):
        plan = read(path)
        for w, ident in zip(plan['windows'], plan['baseline_ids']):
            out[hold_key(w, w['hold'])] = ident
    return out


def random_solos(n, generator):
    assert n % 4 == 0
    return sorted(int(i) for i in generator.choice(n, n//4, replace=False))


def make_plan(root, policy, sigma, update, phase):
    screens = [completed(root, f's{w["seed"]}_screen') for w in worlds()]
    priorpath = root/'CONTINUE_PRIOR.json' if (root/'CONTINUE_PRIOR.json').exists() else root/'PRIOR.json'
    prior = read(priorpath) if priorpath.exists() else None
    if prior is not None and phase.get('es'):
        prior = copy.deepcopy(prior)
        prior['coefficients'] = {k:2*v for k,v in prior['coefficients'].items()}
        prior['purpose'] += '; phase-fix doubles signed observable log-priority, keeping ordinary support'
    chosen, selection = select(screens, np.random.default_rng(20261101+update), prior)
    # Sampling is frozen before outcomes; all members remain in every sampled group.
    generator = torch.Generator().manual_seed(20271101+update)
    zero = SetPointPolicy()
    known = cache(root)
    tasks, windows, ids = [], [], []
    exact = random_solos(len(chosen), np.random.default_rng(30261101+update))
    for i, r in enumerate(chosen):
        sample = policy.sample(r['context'], generator, sigma=sigma)
        if phase.get('es'):
            # A global output-bias population perturbation moves the whole group policy.
            eps = torch.randn(2, generator=generator)
            scale = torch.tensor(phase['es_sigma'])
            sample['action'] = (torch.tensor(sample['action'])+eps*scale).tolist()
            sample['es_eps'], sample['es_sigma'] = eps.tolist(), scale.tolist()
        w = dict(world=r['world'], t0=r['t0'], sample=sample)
        w['hold'] = mean_hold(zero, sample)
        key = hold_key(w, w['hold'])
        if key not in known:
            ident = f'cf2_hold_{key[:24]}'
            known[key] = ident
            tasks.append(dict(id=ident, window=w, sample=w['hold']))
        ids.append(known[key])
        tasks.append(dict(id=f'cf2_u{update:03}_w{i:03}_joint', window=w, sample=sample))
        if i in exact:
            for j, member in enumerate(sample['context']['members']):
                tasks.append(dict(id=f'cf2_u{update:03}_w{i:03}_solo{j}', window=w,
                                  sample=solo_hold(w['hold'], sample, member)))
        windows.append(w)
    assert len(windows) >= 150 and len({w['world']['seed'] for w in windows}) >= 12
    return dict(update=update, windows=windows, baseline_ids=ids, exact_indices=exact,
                tasks=tasks, selection=selection, phase=phase,
                solo_selection='simple random39/156 indices without replacement; whole groups; frozen before outcomes',
                cached_holds=sum(completed(root, ident) is not None for ident in set(ids)),
                charge=sum(t['window']['t0']+600 for t in tasks))


def setup_worker(root):
    global WORKER
    configure()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = Path(root)
    binding = verify(root)
    backend.setup(binding)
    WORKER = root, binding


def work(task):
    root, binding = WORKER
    if task.get('kind') == 'dev':
        from tools.coopfree.evaluation import dev_episode
        policy, _, _, _ = load(root, task['update'])
        result = dev_episode(root, binding, task['world'], 'learned', 200+task['update'], policy, None)
    else:
        w = task['window']
        base = completed(root, f's{w["world"]["seed"]}_screen')
        result = runner.branch(root, binding, task['id'], w['world'], w['t0'],
                               base['states'][str(w['t0'])], task['sample'],
                               task['sample']['context']['members'])
    return dict(id=task['id'], actual_sim_s=result['actual_sim_s'])


def drain(root, tasks):
    pending = [t for t in tasks if completed(root, t['id']) is None]
    with futures.ProcessPoolExecutor(max_workers=4, initializer=setup_worker, initargs=(str(root),)) as pool:
        iterator = iter(pending)
        running = {pool.submit(work, t):t for t in [next(iterator, None) for _ in range(4)] if t}
        while running:
            done, _ = futures.wait(running, return_when=futures.FIRST_COMPLETED)
            for future in done:
                task = running.pop(future)
                print(json.dumps(dict(phase='owner1511_physics', **future.result())), flush=True)
                relocate_completed(root)
                following = next(iterator, None)
                if following:
                    running[pool.submit(work, following)] = following


def records(root, plan):
    out = []
    u = plan['update']
    for i, w in enumerate(plan['windows']):
        ident = f'cf2_u{u:03}_w{i:03}_joint'
        joint, hold = completed(root, ident), completed(root, plan['baseline_ids'][i])
        assert joint['sample'] == w['sample'] and joint['prefix_sha256'] == hold['prefix_sha256']
        dg = {str(h):hold['metrics'][str(h)]['team_delay_vs']-joint['metrics'][str(h)]['team_delay_vs'] for h in (120,300,600)}
        changed = any(backend.motion_changed(hold['motions'][t], v) for t, v in joint['motions'].items())
        issued = issued_difference(root, ident, plan['baseline_ids'][i], w['t0']+w['sample']['window'])
        row = dict(world=w['world'], t0=w['t0'], sample=w['sample'], D_G=dg,
                   baseline_id=plan['baseline_ids'][i], joint_id=ident,
                   execution=dict(joint['execution'], motion_changed=changed, issued_changed=issued))
        if i in plan['exact_indices']:
            solos = [completed(root, f'cf2_u{u:03}_w{i:03}_solo{j}') for j in range(len(w['sample']['context']['members']))]
            row['credit'] = {str(h):credit(hold['metrics'][str(h)]['team_delay_vs'],
                joint['metrics'][str(h)]['team_delay_vs'], [s['metrics'][str(h)]['team_delay_vs'] for s in solos]) for h in (120,300,600)}
        out.append(row)
    return out


def optimize_es(policy, optimizer, rows):
    """Hybrid conditional score for body + population ES for shared output bias.

    Both are unbiased for the same convolved search objective; do not double-count
    the bias gradient. Solos remain diagnostics, never a shaped objective.
    """
    class Step:
        def zero_grad(self):
            optimizer.zero_grad()
        def step(self):
            with torch.no_grad():
                estimates = [-(2*r['D_G']['600']/10000.)*torch.tensor(r['sample']['es_eps']) /
                             torch.tensor(r['sample']['es_sigma']) for r in rows]
                policy.mean.bias.grad.copy_(torch.stack(estimates).mean(0))
            optimizer.step()
    conditional = copy.deepcopy(rows)
    for r in conditional:
        s = r['sample']
        # Score the ordinary Gaussian conditional on the population perturbation.
        s['action'] = (torch.tensor(s['action'])-torch.tensor(s['es_eps'])*torch.tensor(s['es_sigma'])).tolist()
    report = optimize(policy, Step(), conditional)
    report['population_es'] = dict(windows=len(rows), additional_sim_s=0,
        estimator='global mean output-bias ES; conditional Gaussian body score; no double-counted bias',
        bias_gradient_norm=float(policy.mean.bias.grad.norm()),
        comparison='same charged windows; fresh pre/post-phase descriptive effect and gradient norms, not causal efficiency proof')
    return report


def train(root, update, phase):
    path = root/f'HOLD_UPDATE_{update:03}.json'
    if path.exists():
        return read(path)
    assert not (root/f'policy_hold_u{update:03}.pt').exists(), 'unreceipted optimizer blocks replay'
    policy, optimizer, sigma, enlargements = load(root, update-1)
    if phase.get('sigma') and update == phase.get('first_update'):
        sigma = phase['sigma']
    planpath = root/f'CONTINUE_PLAN_{update:03}.json'
    plan = read(planpath) if planpath.exists() else make_plan(root, policy, sigma, update, phase)
    if not planpath.exists():
        write(planpath, plan)
    attempts = [dict(id=t['id'], sim_s=t['window']['t0']+600, phase='train') for t in plan['tasks']]
    runner.ensure_reserved(root, attempts)
    drain(root, plan['tasks'])
    rows = records(root, plan)
    report = optimize_es(policy, optimizer, rows) if phase.get('es') else optimize(policy, optimizer, rows)
    adjusted, enlargements = next_sigma(sigma, report['executed_change_share'], enlargements)
    report.update(update=update, baseline='zero-update policy-mean HOLD; W exogenous',
                  sigma_used=sigma, sigma_next=adjusted, enlargements=enlargements,
                  records=rows, selection=plan['selection'], phase=phase, cached_holds=plan['cached_holds'],
                  gradient_method='continuous Gaussian score; objective2DG; shared team credit; W not scored',
                  limitation='exact HOLD Di/CG on simple random25% complete-group subset only', charged=storage.status(root))
    checkpoint = root/f'policy_hold_u{update:03}.pt'
    runner.save_checkpoint(checkpoint, policy, optimizer, update, adjusted, enlargements)
    report['checkpoint_sha256'] = sha(checkpoint)
    write(path, report)
    print(json.dumps(dict(phase='owner1511_update', update=update,
        executed=report['executed_change_share'], D_G=report['D_G'], C_G=report['C_G'])), flush=True)
    return report


def dev(root, update):
    from tools.coopfree.evaluation import paired_summary
    path = root/f'CONTINUE_DEV_{update:03}.json'
    if path.exists():
        return read(path)
    tasks = [dict(kind='dev', id=f'dev_u{200+update:03}_s{w["seed"]}_learned', world=w, update=update) for w in worlds(True)]
    runner.ensure_reserved(root, [dict(id=t['id'], sim_s=1200, phase='dev') for t in tasks])
    drain(root, tasks)
    learned = [completed(root, t['id']) for t in tasks]
    controls = [completed(root, f'dev_u101_s{w["seed"]}_{arm}') for w in worlds(True) for arm in ('mappo','bandit')]
    hold = [completed(root, f'dev_u001_s{w["seed"]}_learned') for w in worlds(True)]
    result = paired_summary(learned+controls)
    result.update(update=update, checkpoint_sha256=sha(root/f'policy_hold_u{update:03}.pt'),
                  paired_hold=paired_hold(learned, hold), learned_rows=learned,
                  baseline_reuse='same24DEV worlds; immutable MAPPO/bandit/zero-HOLD plant receipts reused')
    write(path, result)
    return result


def check(root, update, panel=None):
    reports = [read(root/f'HOLD_UPDATE_{u:03}.json') for u in range(1, update+1)]
    charged = sum(r['sim_s'] for r in ledger(root) if r['phase']=='train')
    result = learning_check(reports, charged, CAPS['train'],
                            paired=panel['paired_hold'] if panel else None, dev=panel)
    write(root/f'CONTINUE_CHECK_{update:03}.json', result)
    return result


def analysis(root, update):
    """Observational detectors plus exact random-subset CG; no false release labels."""
    from tools.coopfree.analysis import tape
    from tools.coopbehav.detectors import detect
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    report = read(root/f'HOLD_UPDATE_{update:03}.json')
    events, x = [], []
    for row in report['records']:
        if 'credit' not in row:
            continue
        path = root/'jobs'/row['joint_id']/'observable.jsonl.gz'
        found = detect(tape(path, row['world']['seed']))
        events.append(dict(joint_id=row['joint_id'], events=found['events'], credit=row['credit'], trace_sha256=sha(path)))
        a = np.asarray(row['sample']['action'])-np.asarray(row['sample']['context']['anchors'])
        x.append([*a.mean(0), *a.std(0), row['sample']['window'], len(a)])
    clusters = []
    if len(x) >= 3 and len(np.unique(x, axis=0)) >= 3:
        z = StandardScaler().fit_transform(x)
        for k in range(2, min(8, len(x)-1, len(np.unique(z, axis=0))-1)+1):
            model = KMeans(n_clusters=k, n_init=10, random_state=20261101).fit(z)
            if 1 < len(set(model.labels_)) < len(z):
                clusters.append(dict(k=k, silhouette=float(silhouette_score(z, model.labels_)), labels=model.labels_.tolist()))
    result = dict(update=update, events=events, clustering=max(clusters, key=lambda c:c['silhouette']) if clusters else None,
                  limitation='TRAIN explored proposals, not deterministic learned DEV behaviour; observational detectors only; HOLD is not a native/policy-command release; exact CG separate')
    write(root/f'CONTINUE_DISCOVERY_{update:03}.json', result)
    return result


def run(root=ROOT):
    import fcntl
    configure()
    torch.set_num_threads(1)
    root = Path(root)
    binding = verify(root)
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    with (root/'supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            backend.setup(binding)
            phase = dict(id='owner1511_score', es=False)
            last = 1
            for update in range(2, 26):
                if (root/'CONTINUE_FIX_AUTHORITY.json').exists():
                    phase = read(root/'CONTINUE_FIX_AUTHORITY.json')['phase']
                try:
                    train(root, update, phase)
                except ValueError as error:
                    if 'whole-batch budget insufficient' not in str(error):
                        raise
                    write(root/'CONTINUE_BUDGET_STOP.json', dict(before_update=update, error=str(error), charged=storage.status(root)))
                    break
                last = update
                if update in CHECKS:
                    panel = dev(root, update)
                    result = check(root, update, panel)
                    analysis(root, update)
                    if result['status'] == 'phase fix required' and not phase.get('es'):
                        share = result['latest']['executed_change_share']
                        sigma = result['latest']['sigma_next']
                        factor = max(1., (1./max(share, 1./156))**.5)
                        proposed = dict(id=f'owner1511_fix_after{update}', es=True, first_update=update+1,
                            sigma=[s*factor for s in sigma], es_sigma=[s*factor for s in sigma],
                            measured_execution_share=share, scale_factor=factor,
                            state_focus='double signed full-EXAM observable log-priority; retain ordinary support; no action templates')
                        write(root/'CONTINUE_FIX_REQUIRED.json', dict(after_update=update, phase=proposed, check=result,
                            requirement='observer must append REGISTRATION line before authorizing new phase; no completed physics replay'))
                        while not (root/'CONTINUE_FIX_AUTHORITY.json').exists():
                            if (root/'STOP.json').exists():
                                raise RuntimeError('durable STOP while awaiting fix registration')
                            time.sleep(30)
                        phase = read(root/'CONTINUE_FIX_AUTHORITY.json')['phase']
            if last not in CHECKS and last > 1:
                dev(root, last)
                analysis(root, last)
            write(root/'CONTINUE_COMPLETE.json', dict(last_update=last, charged=storage.status(root)))
        except BaseException:
            if not (root/'STOP.json').exists():
                write(root/'STOP.json', dict(error=traceback.format_exc(), charged=storage.status(root)))
            raise


if __name__ == '__main__':
    run()
