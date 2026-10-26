"""Fixed-budget CPU fitting on existing real pairs; world-disjoint calibration."""
import copy
import hashlib
import json
from pathlib import Path
import runpy

import numpy as np
import torch

from wms.control.breakdown import (BreakdownHead, HORIZONS, calibrate, event_labels,
    fit_head, qualify_calibration, reliability_diagram, rule_sha256)
from wms.control.paircrit import PairedProvider, TailValue
from wms.control.real_returns14 import NPZRealReturnProvider, ScaledTeamCritic

OWN = Path('/root/autodl-tmp/c4/paircrit20261026')
ROOT = Path('/root/main14_loop')


def metrics(predicted, actual):
    p, y = np.asarray(predicted, float), np.asarray(actual, float)
    def ranks(x):
        return np.array([((x < v).sum() + .5 * ((x == v).sum() - 1)) for v in x])
    rp, ry = ranks(p), ranks(y)
    return {'n': len(y), 'rmse': float(np.mean((p - y) ** 2) ** .5),
        'r2': float(1 - np.mean((p - y) ** 2) / y.var()) if y.var() else None,
        'spearman': float(np.corrcoef(rp, ry)[0, 1]) if rp.std() and ry.std() else None,
        'sign_accuracy': float((np.sign(p) == np.sign(y)).mean()),
        'bias': float((p - y).mean()), 'prediction_sd': float(p.std()), 'actual_sd': float(y.std())}


def subset(states, use):
    return {k: v[use] for k, v in states.items()}


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261026)
    policy = NPZRealReturnProvider(OWN / 'policy_returns.npz')
    native = NPZRealReturnProvider(OWN / 'native_returns.npz')
    sha = policy.provenance['policy_checkpoint']
    pairs = PairedProvider(policy, native, OWN / 'PAIRING.json', sha)
    rule = json.loads((OWN / 'NATIVE_RULE.json').read_text())
    helper = runpy.run_path(str(OWN / 'tools/paircrit/prepare.py'))
    manifest = json.loads((ROOT / 'data/pairs_c00.a1/C4_EPISODES_MANIFEST.json').read_text())
    entries = {(e['pair_key'], e['branch']['kind']): e for e in manifest['episodes']
               if e.get('branch', {}).get('kind') in ('B', 'POLICY')}
    ys, ms, initially = [], [], []
    with np.load(OWN / 'native_traces.npz', allow_pickle=False) as z:
        native_speed, native_times = z['speed'], z['time']
    for i, row in enumerate(pairs.proof['pairs']):
        entry = entries[row['pair_key'], 'POLICY']
        raw = helper['safe'](Path(entry['source']) / 'summary.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['policy_summary_sha256']
        t, v, start_sha = helper['load_trace'](entry)
        assert start_sha == row['observable_start_sha256'], 'native/policy initial arrays differ'
        assert np.array_equal(t, native_times[i])
        y, m, initial = event_labels(np.stack([native_speed[i], v], 1), t, rule)
        ys.append(y)
        ms.append(m)
        initially.append(initial)
    y, mask = torch.tensor(np.array(ys), dtype=torch.float32), torch.tensor(np.array(ms))
    seeds = policy.provenance['seeds']
    fit = np.array([s not in pairs.development_seeds for s in seeds])
    cal_seed = min(pairs.development_seeds)
    cal = seeds == cal_seed
    test = ~fit & ~cal
    states = policy._states
    cost_p, cost_b = -policy._returns, -native._returns
    advantage = cost_p - cost_b
    # Same representation, fixed 200 full-batch updates and optimizer for both fits.
    base = ScaledTeamCritic(entry_backlog=True, value_scale=30000.)
    absolute = copy.deepcopy(base)
    paired = TailValue(base, paired=True)
    for model, target in [(absolute, -cost_p), (paired, advantage)]:
        opt = torch.optim.Adam(model.parameters(), lr=.0003)
        for _ in range(200):
            pred = (model.cost_advantage(subset(states, fit)) if model is paired
                    else model(subset(states, fit)))
            loss = ((pred - target[fit]) / 30000.).square().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
            opt.step()
    old_checkpoint = ROOT / 'policy/c00.a2/policy_latest.pt'
    assert hashlib.sha256(old_checkpoint.read_bytes()).hexdigest() == sha
    old_payload = torch.load(old_checkpoint, map_location='cpu', weights_only=True)
    frozen = ScaledTeamCritic(entry_backlog=True)
    frozen.load_state_dict(old_payload['critic'])
    with torch.no_grad():
        pp = paired.cost_advantage(states).numpy()
        ap = -absolute(states).numpy()
        old = -frozen(states).numpy()
    head = BreakdownHead()
    head.boundary_mps.fill_(rule['boundary_mps'])
    fit_head(head, subset(states, fit), y[fit], mask[fit], updates=400)
    with torch.no_grad():
        before = head(subset(states, test)).numpy()
    calibrate(head, subset(states, cal), y[cal], mask[cal])
    with torch.no_grad():
        probability = head(states).numpy()
    risk_doc = {'schema': 'paircrit_risk/1', 'split': 'TRAIN', 'head': head.state_dict(),
        'rule': rule, 'rule_sha256': rule_sha256(rule), 'policy_checkpoint_sha256': sha,
        'fit_worlds': sorted(set(seeds[fit].tolist())), 'calibration_worlds': [int(cal_seed)],
        'test_worlds': sorted(set(seeds[test].tolist())), 'wm_event_weights': [0., 0., 0.],
        'wm_event_reliability': 'unavailable; no transfer from reward reliability',
        'initially_broken_excluded': True}
    prevalence = (y[fit] * mask[fit]).sum(0).numpy() / mask[fit].sum(0).numpy().clip(1)
    qualification = qualify_calibration(probability[test], y[test].numpy(), mask[test].numpy(),
                                         seeds[test], prevalence)
    risk_doc['calibration_qualified'] = qualification['qualified']
    risk_doc['qualification'] = qualification
    torch.save(risk_doc, OWN / 'breakdown.pt')
    torch.save(paired.state_dict(), OWN / 'paired_critic.pt')
    diagrams, risk_metrics = {}, {}
    yy, mm = y.numpy(), mask.numpy()
    for arm in range(2):
        for j, h in enumerate(HORIZONS):
            name = f'{("native", "policy")[arm]}_{h}'
            d = reliability_diagram(probability[test, arm, j], yy[test, arm, j], mm[test, arm, j])
            diagrams[name] = d
            risk_metrics[name] = {'brier': d['brier'], 'n': int(mm[test, arm, j].sum()),
                'uncalibrated_brier': reliability_diagram(before[:, arm, j], yy[test, arm, j], mm[test, arm, j])['brier']}
    delta_metrics = {}
    for j, h in enumerate(HORIZONS):
        valid = test & mm[:, 0, j] & mm[:, 1, j]
        delta_metrics[str(h)] = metrics(probability[valid, 1, j] - probability[valid, 0, j],
                                       yy[valid, 1, j] - yy[valid, 0, j]) if valid.any() else {'n': 0}
    result = {'schema': 'paircrit_measurement/1', 'new_sumo_sim_s': 0, 'device': 'cpu', 'threads': 4,
        'source_sha256s': {name: hashlib.sha256((OWN / name).read_bytes()).hexdigest() for name in (
            'src/wms/control/paircrit.py', 'src/wms/control/breakdown.py', 'tools/paircrit/measure.py',
            'PAIRING.json', 'policy_returns.npz', 'native_returns.npz')},
        'rule': rule, 'rule_sha256': rule_sha256(rule), 'policy_sha256': sha,
        'fit_worlds': risk_doc['fit_worlds'], 'calibration_worlds': risk_doc['calibration_worlds'],
        'test_worlds': risk_doc['test_worlds'], 'critic_development_worlds': sorted(pairs.development_seeds),
        'critic_updates': 200, 'risk_updates': 400,
        'old_absolute_all': metrics(old, cost_p.numpy()),
        'old_absolute_development': metrics(old[~fit], cost_p.numpy()[~fit]),
        'matched_absolute_development': metrics(ap[~fit], cost_p.numpy()[~fit]),
        'paired_development': metrics(pp[~fit], advantage.numpy()[~fit]),
        'zero_advantage_development': metrics(np.zeros((~fit).sum()), advantage.numpy()[~fit]),
        'zero_advantage_all': metrics(np.zeros(len(seeds)), advantage.numpy()),
        'initially_low_starts': np.asarray(initially).sum(0).tolist(),
        'risk_test': risk_metrics, 'event_difference_test': delta_metrics,
        'probability_difference_identified': False,
        'calibration_qualification': qualification,
        'note': 'single deterministic future per state; observed event-difference skill is not conditional probability-difference skill',
        'wm_event_weights': [0., 0., 0.]}
    (OWN / 'MEASURED.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    (OWN / 'RELIABILITY.json').write_text(json.dumps(diagrams, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
