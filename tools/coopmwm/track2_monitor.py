"""P3: update receipts and budgeted isolated-cooperation diagnostics (CPU only).

Public API: monitor_update(cfg, arm, u, receipts), probe(cfg, u), replicated(cfg, u).
``receipts`` is the T2C metrics mapping, or an envelope with ``metrics`` and optional
JSON ``sources`` (receipt IDs/hashes). Missing measurements are never replaced by zero.
Read everything with::

    python -m tools.coopmwm.track2_monitor status --root <clean1-root>

Integration contract for probe/replicated:
* cfg.run_root is the existing experiment root; cfg.isolated_registration is the
  explicit TRAIN registration (replicates with world_seed/state_set_sha256/
  window_set_sha256). Same registered states/windows are used by B and R.
* cfg.monitor_backend supplies resolve_states(job) -> [(snapshot, bank_receipt)],
  weights(arm, update) -> (current_weights, common_u0_weights),
  reserve(jobs, phase) -> accounting receipt, and run_branch(spec, snapshot).
  reserve MUST atomically precharge the whole wave in the experiment's own ledger;
  run_branch MUST use the coordinator worker permit and qualified full-layer policy
  controller. No runner, checkpoint reader, SUMO process or remote is created here.
* cfg.monitor_dry_run defaults to True. Dry calls expand and quote real full-window
  specs but never reserve/dispatch. Live calls require the above hooks. A failed or
  interrupted live claim is never automatically retried. Completed calls are reusable.
* phase caps come from cfg.budget.phases (early_probes / isolated_replicated);
  optional per-cell caps use the T2C early_probes/isolated_test.sim_s_per_* keys.
  The coordinator remains the sole owner of the <=5M cumulative experiment ledger.

Replicated checks: u6/u12/u24, five paired TRAIN worlds, B/R only. The common u0
cooperation gain is algebraically zero. Each state runs group-new, each solo-new,
and a fresh all-base reference, with all background vehicles at common u0. Reuses
policy_synergy.specs_for and metrics, including captured t0 and CRN; no shorter
look-ahead. Probes at u2/u4 are one run on exactly three registered states per arm
(A/B/R by default, configurable via early_probes.arms). Bootstrap over WORLD
replicates, never over branches/states; u24 B-vs-common-base positive 95% CI and
no safety regression is the criterion, B-R descriptive. Final paired episode
evaluation belongs to T1D/T2C, not this isolated diagnostic.
"""
from copy import deepcopy
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
from statistics import fmean
import threading

from .protocol import digest

METRICS = ('group_credit_wm', 'group_credit_checked', 'dense_ce', 'dense_accuracy',
           'dense_filtered_share', 'dense_checked_accuracy', 'bank_move_ce',
           'layer_kl', 'merge_missing_share', 'reward')
COUNTS = ('decisions', 'credit_rows', 'credit_checked_rows', 'dense_labels',
          'dense_eligible_labels', 'dense_checked_labels', 'augmented_examples', 'weighted_starts')
SAFETY = ('collisions', 'teleports', 'hard_brakes', 'ttc_events')
MONITOR_DEFAULTS = dict(trigger_saturation=[0., 1.], history_updates=3,
                        accuracy_tolerance=1e-6, layer_kl_near_zero=1e-8,
                        action='DIAGNOSE; pause only for safety')
BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='percentile')
_LOCK = threading.RLock()
_FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth')
_TRUTH = re.compile(r'(vtype|driver|vehicle_params)[^/]*truth|truth[^/]*(vtype|driver|vehicle_params)')


def _guard(path):
    text = str(path).replace('\\', '/').lower()
    if any(word in text for word in _FORBIDDEN) or _TRUTH.search(text):
        raise ValueError('REFUSED: restricted monitor path')
    return Path(path)


def _root(cfg):
    root = _guard(cfg['run_root'])
    root = _guard(root.resolve())
    if not root.is_dir():
        raise ValueError('REFUSED: existing experiment run_root required')
    return root


def _path(root, relative):
    p = _guard(root/relative)
    resolved = _guard(p.resolve())
    if root not in resolved.parents:
        raise ValueError('REFUSED: monitor path escapes experiment root')
    return p


def _seal(value, key):
    value = deepcopy({k: v for k, v in value.items() if k != key})
    json.dumps(value, allow_nan=False)
    return dict(value, **{key: digest(value)})


def _read(path, key):
    p = _guard(path)
    _guard(p.resolve())
    value = json.loads(p.read_text(encoding='utf-8'))
    if _seal(value, key) != value:
        raise ValueError('REFUSED: edited monitor receipt: '+p.name)
    return value


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')


def _number(value, name, *, low=None, high=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (low is not None and value < low) or (high is not None and value > high)):
        raise ValueError('REFUSED: invalid '+name)
    return value


def _count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError('REFUSED: invalid count '+name)
    return value


def _safety(value):
    if not isinstance(value, dict) or set(value) != set(SAFETY):
        raise ValueError('REFUSED: all safety counters required')
    return {k: _count(value[k], k) for k in SAFETY}


def checked_metrics(arm, metrics):
    """Canonical T2C fields; None = unmeasured/not applicable, never a fake zero."""
    required = set(METRICS+COUNTS+('families', 'trigger_counts', 'safety', 'safety_stop'))
    if required-set(metrics):
        raise ValueError('REFUSED: missing monitor metrics: '+', '.join(sorted(required-set(metrics))))
    for key in COUNTS:
        _count(metrics[key], key)
    families, triggers = metrics['families'], metrics['trigger_counts']
    if (not isinstance(families, (list, tuple)) or not families
            or any(not isinstance(f, str) or not f for f in families)
            or len(set(families)) != len(families) or 'none' not in families
            or not isinstance(triggers, dict) or set(triggers) != set(families)):
        raise ValueError('REFUSED: every move family including none required')
    if sum(_count(n, f) for f, n in triggers.items()) != metrics['decisions']:
        raise ValueError('REFUSED: inconsistent trigger counts')
    for checked, total in (('credit_checked_rows', 'credit_rows'),
                           ('dense_eligible_labels', 'dense_labels'),
                           ('dense_checked_labels', 'dense_labels')):
        if metrics[checked] > metrics[total]:
            raise ValueError('REFUSED: inconsistent '+checked)
    for key in METRICS:
        value = metrics[key]
        nullable = (arm == 'A' and key != 'reward') or (
            key == 'group_credit_checked' and metrics['credit_checked_rows'] == 0) or (
            key == 'dense_checked_accuracy' and metrics['dense_checked_labels'] == 0) or (
            key in ('dense_ce', 'dense_accuracy') and metrics['dense_eligible_labels'] == 0)
        if value is None and nullable:
            continue
        low = 0 if key in ('dense_ce', 'bank_move_ce', 'layer_kl') or 'share' in key or 'accuracy' in key else None
        high = 1 if 'share' in key or 'accuracy' in key else None
        _number(value, key, low=low, high=high)
    if arm != 'A':
        if any(metrics[k] == 0 for k in ('decisions', 'credit_rows', 'dense_labels')):
            raise ValueError('REFUSED: missing learning component observations')
        filtered = 1-metrics['dense_eligible_labels']/metrics['dense_labels']
        if not math.isclose(metrics['dense_filtered_share'], filtered, abs_tol=1e-12):
            raise ValueError('REFUSED: inconsistent calibrated dense filtered share')
    _safety(metrics['safety'])
    if type(metrics['safety_stop']) is not bool:
        raise ValueError('REFUSED: explicit safety_stop required')


def _settings(cfg):
    settings = dict(MONITOR_DEFAULTS, **cfg.get('learning_monitor', {}))
    # These are diagnostics, not tuneable intervention/behaviour thresholds.
    if settings['history_updates'] != 3 or settings['trigger_saturation'] != [0., 1.]:
        raise ValueError('REFUSED: registered three-update/saturation diagnostics differ')
    for k in ('accuracy_tolerance', 'layer_kl_near_zero'):
        _number(settings[k], k, low=0)
    return settings


def _monitor_row(cfg, arm, update, metrics, history, sources):
    checked_metrics(arm, metrics)
    settings = _settings(cfg)
    decisions = metrics['decisions']
    rates = {f: n/decisions if decisions else None for f, n in metrics['trigger_counts'].items()}
    diagnose = []
    if arm != 'A':
        if 1-rates['none'] in settings['trigger_saturation']:
            diagnose.append('trigger_saturated')
        if metrics['layer_kl'] <= settings['layer_kl_near_zero']:
            diagnose.append('layer_kl_near_zero')
        window = [r['metrics'] for r in history[-2:]]+[metrics]
        if len(window) == 3:
            if all(b['group_credit_wm'] <= a['group_credit_wm'] for a, b in zip(window, window[1:])):
                diagnose.append('group_credit_not_rising_3_updates')
            accuracy = [m['dense_accuracy'] for m in window]
            if all(v is not None for v in accuracy) and max(accuracy)-min(accuracy) <= settings['accuracy_tolerance']:
                diagnose.append('dense_accuracy_flat_3_updates')
        if metrics['dense_eligible_labels'] == 0:
            diagnose.append('no_eligible_dense_labels')
    return _seal(dict(schema='coopmwm.track2.learning_monitor.v1', arm=arm, update=update,
        metrics=metrics, trigger_rates=rates, diagnose=diagnose, sources=sources,
        settings=settings, action='PAUSE_SAFETY' if metrics['safety_stop'] else
        ('DIAGNOSE' if diagnose else 'CONTINUE'),
        previous_monitor_sha256=history[-1]['monitor_sha256'] if history else None), 'monitor_sha256')


def monitor_history(root, arm):
    if arm not in ('A', 'B', 'R'):
        raise ValueError('REFUSED: invalid monitor arm')
    rows = []
    folder = _path(root, 'learning_monitor/'+arm)
    for path in sorted(folder.glob('u*.json')):
        path = _path(root, path.relative_to(root))
        row = _read(path, 'monitor_sha256')
        u = len(rows)+1
        if row['arm'] != arm or row['update'] != u or path.name != f'u{u:03}.json':
            raise ValueError('REFUSED: monitor history must be contiguous')
        expected = _monitor_row(dict(learning_monitor=row['settings']), arm, u, row['metrics'], rows, row['sources'])
        if row != expected:
            raise ValueError('REFUSED: invalid monitor history chain')
        rows.append(row)
    return rows


def monitor_update(cfg, arm, u, receipts):
    """Persist one immutable update; exact duplicate delivery is idempotent."""
    if arm not in ('A', 'B', 'R') or type(u) is not int or not 1 <= u <= 24:
        raise ValueError('REFUSED: invalid monitor arm/update')
    root = _root(cfg)
    metrics = deepcopy(receipts.get('metrics', receipts))
    sources = deepcopy(receipts.get('sources', {})) if 'metrics' in receipts else {}
    with _LOCK:
        history = monitor_history(root, arm)
        if u > len(history)+1:
            raise ValueError('REFUSED: monitor history must be contiguous')
        row = _monitor_row(cfg, arm, u, metrics, history[:u-1], sources)
        if u <= len(history):
            if row != history[u-1]:
                raise ValueError('REFUSED: conflicting duplicate monitor update')
        else:
            _write(_path(root, f'learning_monitor/{arm}/u{u:03}.json'), row)
        _write_status(root)
    return row


def read_monitor(root):
    root = _root(dict(run_root=root))
    with _LOCK:
        rows = {a: monitor_history(root, a) for a in ('A', 'B', 'R')}
        diagnostics = []
        folder = _path(root, 'track2_diagnostics')
        for p in sorted(folder.glob('*_u[0-9][0-9][0-9]*.json')):
            diagnostics.append(_read(_path(root, p.relative_to(root)), 'receipt_sha256'))
        return dict(schema='coopmwm.track2.status.v1',
            updates={a: len(rs) for a, rs in rows.items()},
            latest={a: rs[-1] if rs else None for a, rs in rows.items()}, diagnostics=diagnostics)


def _write_status(root):
    status = read_monitor(root)
    lines = ['# CLEAN1 status', '', 'Learning checks diagnose; only safety requests a pause.', '']
    for arm, row in status['latest'].items():
        if row is None:
            lines.append(f'- {arm}: no update receipt yet.')
            continue
        m = row['metrics']
        lines += [f"- {arm} u{row['update']}: {row['action']}; reward={m['reward']}; "
            f"group WM/checked={m['group_credit_wm']}/{m['group_credit_checked']}; "
            f"dense CE/accuracy={m['dense_ce']}/{m['dense_accuracy']}; "
            f"filtered={m['dense_filtered_share']}; checked accuracy={m['dense_checked_accuracy']}; "
            f"bank CE={m['bank_move_ce']}; layer KL={m['layer_kl']}; merge missing={m['merge_missing_share']}.",
            '  Trigger rates: '+json.dumps(row['trigger_rates'], sort_keys=True)+'.',
            '  Safety: '+json.dumps(m['safety'], sort_keys=True)+'.',
            '  DIAGNOSE: '+(', '.join(row['diagnose']) or 'none')+'.']
    for row in status['diagnostics']:
        lines.append(f"- {row['kind']} u{row['update']}: {row['status']}; dry={row['dry']}; "
                     f"quoted={row['quoted_sim_s']} sim-s; verdict={row.get('verdict', 'descriptive')}.")
    lines += ['', 'Read full receipts: python -m tools.coopmwm.track2_monitor status --root <clean1-root>', '']
    # The single coordinator may call from training and diagnostic threads.
    temp = _path(root, 'CLEAN1_STATUS.md.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        stream.write('\n'.join(lines))
    os.replace(temp, _path(root, 'CLEAN1_STATUS.md'))


def diagnostic_identity(states):
    from .sumo_verify import snapshot_sha
    return dict(state_set_sha256=digest(dict(states=[dict(state_id=r['verification_id'],
        snapshot_sha256=snapshot_sha(s), members=r['chosen_plan']['members']) for s, r in states])),
        window_set_sha256=digest(dict(windows=[r['window'] for _, r in states])))


def isolated_expansion(job, states, new_weights, common_weights):
    """Pure real-machinery expansion; budget is checked against full specs, not estimates."""
    from . import policy_synergy as ps
    from .exact_grade import _seal as branch_seal
    if not states or any(job.get(k) != v for k, v in diagnostic_identity(states).items()):
        raise ValueError('REFUSED: registered diagnostic states/windows differ')
    out, ids = [], set()
    for index, (snapshot, receipt) in enumerate(states):
        if (snapshot['episode'].get('role') != 'train'
                or snapshot['episode']['world']['seed'] != job['world_seed']):
            raise ValueError('REFUSED: diagnostic requires paired TRAIN world')
        group = receipt['chosen_plan']['members']
        if not 2 <= len(group) <= 4 or len(set(group)) != len(group):
            raise ValueError('REFUSED: isolated group requires 2..4 distinct CAVs')
        vid = receipt['verification_id']
        if vid in ids:
            raise ValueError('REFUSED: duplicate diagnostic state')
        ids.add(vid)
        pairs = ps.specs_for(snapshot, receipt, job['update'], job['arm'],
                             new_weights, common_weights, isolated=True)[:-1]
        ref = ps.specs_for(snapshot, receipt, job['update'], job['arm'],
                           common_weights, common_weights, isolated=True)[0]
        pairs.append(ref)
        renamed = []
        for i, (spec, snap) in enumerate(pairs):
            if spec['window'] != receipt['window'] or spec['window'].get('truncated'):
                raise ValueError('REFUSED: changed/truncated full diagnostic window')
            spec = {k: v for k, v in spec.items() if k != 'branch_sha256'}
            spec.update(verification_id=f"{job['id']}_s{index:03}_b{i:02}",
                        branch='reference' if i == len(pairs)-1 else spec['branch'])
            renamed.append((branch_seal(spec, 'branch_sha256'), snap))
        out.append((list(group), renamed))
    return out


def _registration(cfg):
    reg = deepcopy(cfg['isolated_registration'])
    if reg.get('role') != 'train' or reg.get('metric') != 'policy_cooperation_gain_veh_s':
        raise ValueError('REFUSED: explicit TRAIN cooperation registration required')
    rows = reg.get('replicates', [])
    if len(rows) != 5:
        raise ValueError('REFUSED: exactly five registered world replicates required')
    seeds = set()
    for i, row in enumerate(sorted(rows, key=lambda r: r['replicate'])):
        seed = row.get('world_seed')
        if type(row.get('replicate')) is not int or row['replicate'] != i:
            raise ValueError('REFUSED: unique replicate indices 0..4 required')
        if type(seed) is not int or not 986000 <= seed <= 986071 or seed in seeds:
            raise ValueError('REFUSED: distinct registered TRAIN world seeds required')
        seeds.add(seed)
        for k in ('state_set_sha256', 'window_set_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', row.get(k, '')):
                raise ValueError('REFUSED: missing registered '+k)
    return reg, sorted(rows, key=lambda r: r['replicate'])


def _bootstrap(values, update, pair_index, settings):
    seed = settings['seed']*10000+100*update+pair_index
    rng = random.Random(seed)
    n, reps = len(values), settings['reps']
    means = sorted(fmean(values[rng.randrange(n)] for _ in range(n)) for _ in range(reps))
    alpha = (1-settings['level'])/2
    return dict(mean=fmean(values), ci_lo=means[int(alpha*reps)],
        ci_hi=means[min(reps-1, int((1-alpha)*reps))], n=n, level=settings['level'],
        reps=reps, seed=seed)


def _analyse(records, update, settings):
    cells = {(r['arm'], r['replicate']): r for r in records}
    if len(cells) != 10 or len(records) != 10 or set(cells) != {(a, i) for a in ('B', 'R') for i in range(5)}:
        raise ValueError('REFUSED: incomplete paired replicated grid')
    for i in range(5):
        for k in ('world_seed', 'state_set_sha256', 'window_set_sha256'):
            if cells['B', i][k] != cells['R', i][k]:
                raise ValueError('REFUSED: unpaired replicated '+k)
    comparisons = {}
    for pi, (a, b) in enumerate((('B', 'common_base'), ('R', 'common_base'), ('B', 'R'))):
        values = [cells[a, i]['metric']-(0 if b == 'common_base' else cells[b, i]['metric']) for i in range(5)]
        comparisons[a+'-'+b] = _bootstrap(values, update, pi, settings)
    safety = {k: dict(B=sum(cells['B', i]['safety'][k] for i in range(5)),
        common_base=sum(cells['B', i]['reference_safety'][k] for i in range(5))) for k in SAFETY}
    safety_ok = all(v['B'] <= v['common_base'] for v in safety.values())
    positive = comparisons['B-common_base']['ci_lo'] > 0
    return dict(comparisons=comparisons, safety=safety, safety_ok=safety_ok,
        criterion='u24 B-common_base CI95 low > 0 and no safety regression; B-R descriptive',
        verdict=('pass' if positive and safety_ok else 'fail') if update == 24 else 'descriptive')


def _diagnostic(cfg, u, kind):
    schedule = (2, 4) if kind == 'probe' else (6, 12, 24)
    if type(u) is not int:
        raise ValueError('REFUSED: integer diagnostic update required')
    if u not in schedule:
        return dict(kind=kind, update=u, status='not_scheduled', new_sim_s=0)
    root = _root(cfg)
    dry = cfg.get('monitor_dry_run', True)
    if type(dry) is not bool:
        raise ValueError('REFUSED: explicit boolean monitor_dry_run required')
    reg, rows = _registration(cfg)
    registration_sha = digest(reg)
    backend = cfg['monitor_backend']
    arms = list(cfg.get('early_probes', {}).get('arms', ['A', 'B', 'R'])) if kind == 'probe' else ['B', 'R']
    if not arms or len(set(arms)) != len(arms) or any(a not in ('A', 'B', 'R') for a in arms):
        raise ValueError('REFUSED: invalid probe arms')
    if cfg.get('early_probes', {}).get('states', 3) != 3:
        raise ValueError('REFUSED: quick probes require exactly three states')
    bs = dict(BOOTSTRAP, **cfg.get('isolated_test', {}).get('bootstrap', {}))
    if bs != BOOTSTRAP or cfg.get('isolated_test', {}).get('replicates', 5) != 5:
        raise ValueError('REFUSED: registered N=5/bootstrap settings differ')
    phase = 'early_probes' if kind == 'probe' else 'isolated_replicated'
    binding = dict(registration_sha256=registration_sha, arms=arms, bootstrap=bs,
                   criterion_sha256=cfg.get('criterion_sha256'), phase=phase)
    result_path = _path(root, f'track2_diagnostics/{kind}_u{u:03}_{"dry" if dry else "live"}.json')
    claim_path = _path(root, f'track2_diagnostics/claims/{kind}_u{u:03}.json')
    with _LOCK:
        if result_path.exists():
            saved = _read(result_path, 'receipt_sha256')
            if saved['binding'] != binding:
                raise ValueError('REFUSED: diagnostic registration changed after completion')
            return saved
        if not dry and claim_path.exists():
            raise ValueError('REFUSED: interrupted diagnostic claim; never replay')
    prepared, jobs = [], []
    # Resolve and validate ALL cells before reserving the WHOLE wave.
    for arm in arms:
        for row in (rows[:1] if kind == 'probe' else rows):
            job = dict(row, id=f'p3_{kind}_u{u:03}_{arm}_r{row["replicate"]:03}',
                       arm=arm, update=u, phase=phase, registration_sha256=registration_sha)
            states = backend.resolve_states(deepcopy(job))
            if diagnostic_identity(states) != {k: row[k] for k in diagnostic_identity(states)}:
                raise ValueError('REFUSED: registered diagnostic states/windows differ')
            if kind == 'probe':
                if len(states) < 3:
                    raise ValueError('REFUSED: quick probe requires three registered states')
                states = states[:3]
                job.update(diagnostic_identity(states))
            current, common = backend.weights(arm, u)
            expanded = isolated_expansion(job, states, current, common)
            cell_sim_s = sum(s['sim_s'] for _, pairs in expanded for s, _ in pairs)
            cap_key = 'sim_s_per_arm' if kind == 'probe' else 'sim_s_per_replicate'
            cap = cfg.get('early_probes' if kind == 'probe' else 'isolated_test', {}).get(cap_key)
            if cap is not None and cell_sim_s > _count(cap, cap_key):
                raise ValueError('REFUSED: full diagnostic windows exceed cell allowance')
            prepared.append((job, expanded))
            jobs.extend(dict(id=s['verification_id'], sim_s=_count(s['sim_s'], 'branch sim_s'))
                        for _, pairs in expanded for s, _ in pairs)
    quoted = sum(j['sim_s'] for j in jobs)
    phase_cap = _count(cfg['budget']['phases'][phase], phase)
    if quoted > phase_cap:
        raise ValueError('REFUSED: full diagnostic wave exceeds phase allowance')
    quote_sha = digest(jobs)
    accounting, records = None, []
    if not dry:
        with _LOCK:
            _write(claim_path, dict(binding=binding, quoted_sim_s=quoted, quote_sha256=quote_sha))
        accounting = backend.reserve(deepcopy(jobs), phase)
        # Validate accounting is serializable before any dispatch.
        json.dumps(accounting, allow_nan=False)
        from .policy_synergy import metrics as synergy_metrics
        for job, expanded in prepared:
            entries, safety, reference_safety = [], dict.fromkeys(SAFETY, 0), dict.fromkeys(SAFETY, 0)
            for group, pairs in expanded:
                results = []
                for spec, snapshot in pairs:
                    result = backend.run_branch(spec, snapshot)
                    if result.get('status') != 'graded':
                        raise ValueError('REFUSED: ungraded diagnostic; never drop/impute/retry')
                    _number(result.get('J_veh_s'), 'diagnostic J')
                    _safety(result.get('safety'))
                    results.append(result)
                ref = results[-1]
                measure = synergy_metrics(ref['J_veh_s'], results[:-1], group)
                entries.append(dict(group=group, window=pairs[0][0]['window'], results=results, **measure))
                for k in SAFETY:
                    safety[k] += sum(r['safety'][k] for r in results)
                    reference_safety[k] += len(results)*ref['safety'][k]
            records.append(dict(job, metric=fmean(e['policy_cooperation_gain_veh_s'] for e in entries),
                status='graded', safety=safety, reference_safety=reference_safety, entries=entries))
    analysis = _analyse(records, u, bs) if not dry and kind == 'replicated' else {}
    output = _seal(dict(schema='coopmwm.track2.isolated_diagnostic.v1', kind=kind, update=u,
        status='dry_run' if dry else 'complete', dry=dry, binding=binding,
        cells=len(prepared), branches=len(jobs), quoted_sim_s=quoted, quote_sha256=quote_sha,
        new_sim_s=0 if dry else quoted, accounting=accounting, records=records, **analysis), 'receipt_sha256')
    with _LOCK:
        _write(result_path, output)
        _write_status(root)
    return output


def probe(cfg, u):
    """u2/u4 only: single-run isolated probe on three registered states."""
    return _diagnostic(cfg, u, 'probe')


def replicated(cfg, u):
    """u6/u12/u24 only: N=5 B/R paired-world isolated test and bootstrap CIs."""
    return _diagnostic(cfg, u, 'replicated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status',))
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(json.dumps(read_monitor(args.root), indent=2, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
