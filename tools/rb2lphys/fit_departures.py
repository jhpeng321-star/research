"""Bayesian optimisation of three global SUMO departure-mode probabilities.

Conditional recorded-state likelihood/CRPS screen, not a long-run qualification.
Only observable FIT tapes are accepted. Existing joint IDM support is frozen.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr

from tools.rbfit.idm_fit import native_episode, read_episode
from wms.model.entries import ENTRY_S_M
from wms.model.sumo_insertion import insertion_speed
from wms.physics.config import PhysicsConfig
from wms.physics.merge_network import NUMERICAL_EPS, POSITION_EPS

NAMES = ('lane_first_probability', 'mainline_last_probability', 'ramp_last_probability')


def require_fit(worlds):
    if not worlds or any(isinstance(w, bool) or int(w) != w or not 969400 <= w <= 969431 for w in worlds):
        raise ValueError('development-only FIT worlds969400..969431 required')


def component_rows(a, world, population, weights, draws=16):
    """Compare exact insertion forms on post-movement observed surroundings.

    Only basePos births, no internal-junction reappearances. Exclude ambiguous
    same-entrance simultaneous births whose insertion order is unobserved.
    Future frames are outcomes for this conditional diagnosis, never WM inputs.
    """
    kin, mask, vid, times = (a[k] for k in ('veh_kin', 'veh_mask', 'vid', 'time_s'))
    a0, b0, tau0, gap0, _ = PhysicsConfig().idm()
    default = dict(v0=31.29, time_headway=tau0, accel=a0, decel=b0, min_gap=gap0)
    seen, rows = set(), []
    cdf = weights.cumsum()/weights.sum()
    for t in range(len(times)):
        if times[t] > 900:
            break
        k = np.asarray(kin[t])
        active = np.asarray(mask[t], bool) & (k[:, 1:4].sum(-1) > 0)
        ids = vid[t] if vid.ndim == 2 else vid
        new = [i for i in np.flatnonzero(active) if int(ids[i]) not in seen]
        seen.update(int(ids[i]) for i in np.flatnonzero(active))
        if t == 0 or not 240 <= times[t] <= 900:
            continue
        lanes = k[:, 1:4].argmax(-1)
        eligible = [i for i in new if abs(float(k[i, 0])-ENTRY_S_M[lanes[i]]-float(k[i, 6])-POSITION_EPS) < NUMERICAL_EPS]
        ambient = active.copy()
        ambient[new] = False
        for i in eligible:
            entrance = int(lanes[i] == 2)
            if sum((lanes[j] == 2) == bool(entrance) for j in eligible) != 1:
                continue
            geometry = {}
            for lane in ((2,) if entrance else (1, 0)):
                occupants = np.flatnonzero(ambient & (lanes == lane))
                if len(occupants):
                    rear = occupants[np.argmin(k[occupants, 0])]
                    space = float(k[rear, 0])-ENTRY_S_M[lane]
                    gap = float(k[rear, 0]-k[rear, 6])-ENTRY_S_M[lane]-float(k[i, 6])-POSITION_EPS
                    leader = float(k[rear, 4])
                else:
                    space, gap, leader = float('inf'), 1e9, 0.
                geometry[lane] = (space, gap, leader)
            order = sorted(geometry, reverse=True)
            free = max(order, key=lambda ln: geometry[ln][0])
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'{world}:{int(ids[i])}:departure-fit'.encode()).digest()[:8], 'big'))
            indices = np.searchsorted(cdf, (np.arange(draws)+rng.random())/draws)
            params = [dict(zip(('v0', 'time_headway', 'accel', 'decel', 'min_gap'), population[j])) for j in indices]
            if bool(a['is_cav'][t, i]):
                params = [default]*draws
            predictions = []
            for first, last in itertools.product((False, True), repeat=2):
                lane = order[0] if first else free
                _, gap, leader = geometry[lane]
                limit = 22.22 if entrance else 31.29
                samples = []
                for param in params:
                    cap = float(param['v0'])*limit/31.29
                    requested = min(cap, leader if gap < 1e9 else cap) if last else cap
                    value = insertion_speed(requested, gap, leader, {**param, 'v0': cap}, .5)
                    ok = value is not None and (not last or value >= requested)
                    samples.append([float(value)/limit if ok else 0., lane if ok else -1, int(ok)])
                predictions.append(samples)
            rows.append(dict(world=world, time_s=float(times[t]), entrance=entrance,
                actual_speed=float(k[i, 4])/(22.22 if entrance else 31.29), actual_lane=int(lanes[i]),
                predictions=predictions))
    return rows


class ConditionalScore:
    def __init__(self, rows):
        self.rows = rows
        a = np.array([r['predictions'] for r in rows])
        target = np.array([r['actual_speed'] for r in rows])
        self.term1 = np.abs(a[..., 0]-target[:, None, None]).mean(-1)
        self.term2 = np.empty((len(rows), 4, 4))
        for i, j in itertools.product(range(4), repeat=2):
            self.term2[:, i, j] = np.abs(a[:, i, :, 0, None]-a[:, j, None, :, 0]).mean((1, 2))
        self.birth = a[..., 2].mean(-1)
        self.lanes = np.stack([(a[..., 1] == ln).mean(-1) for ln in range(3)], -1)
        self.target_lane = np.eye(3)[[r['actual_lane'] for r in rows]]
        self.entrance = np.array([r['entrance'] for r in rows], bool)
        worlds = np.array([r['world'] for r in rows])
        # Equal world and entrance weighting; no world-specific parameter fit.
        self.weights = np.zeros(len(rows))
        for w in np.unique(worlds):
            for e in (0, 1):
                use = (worlds == w) & (self.entrance == e)
                if use.any():self.weights[use] = 1/use.sum()
        self.weights /= self.weights.sum()

    def metrics(self, x):
        lane = np.where(self.entrance, 0., x[0])
        speed = np.where(self.entrance, x[2], x[1])
        p = np.stack(((1-lane)*(1-speed), (1-lane)*speed, lane*(1-speed), lane*speed), -1)
        speed_crps = (p*self.term1).sum(-1)-.5*np.einsum('ni,nij,nj->n', p, self.term2, p)
        lane_prob = np.einsum('ni,nij->nj', p, self.lanes)
        lane_brier = ((lane_prob-self.target_lane)**2).sum(-1)
        refusal_brier = (1-(p*self.birth).sum(-1))**2
        values = dict(speed_crps_normalized=speed_crps, lane_brier=lane_brier, birth_brier=refusal_brier)
        result = {k: float(v@self.weights) for k, v in values.items()}
        result['objective'] = sum(result.values())
        return result


def bayes_search(score, iterations=48, seed=20261101):
    """Exact RBF GP + expected improvement, fixed budget and deterministic seed."""
    rng = np.random.default_rng(seed)
    xs = [np.array(v, float) for v in itertools.product((0., 1.), repeat=3)]+[np.full(3, .5)]
    ys = [score(x) for x in xs]
    for _ in range(iterations):
        x, y = np.array(xs), np.array(ys)
        scale = max(float(y.std()), np.finfo(float).eps)
        z = (y-y.mean())/scale
        def kernel(a, b):
            return np.exp(-.5*(((a[:, None]-b[None])/0.35)**2).sum(-1))
        k = kernel(x, x)+1e-8*np.eye(len(x))
        chol = np.linalg.cholesky(k)
        candidates = rng.random((2048, 3))
        near = np.clip(x[int(y.argmin())]+rng.normal(0, .08, (512, 3)), 0, 1)
        candidates = np.concatenate((candidates, near))
        cross = kernel(x, candidates)
        alpha = np.linalg.solve(chol.T, np.linalg.solve(chol, z))
        mu = cross.T@alpha
        v = np.linalg.solve(chol, cross)
        sigma = np.sqrt(np.maximum(1-(v*v).sum(0), 1e-12))
        improve = z.min()-mu
        standard = improve/sigma
        ei = improve*ndtr(standard)+sigma*np.exp(-.5*standard**2)/np.sqrt(2*np.pi)
        next_x = candidates[int(ei.argmax())]
        xs.append(next_x)
        ys.append(score(next_x))
    best = int(np.argmin(ys))
    return xs[best], [dict(parameters=x.tolist(), objective=y) for x, y in zip(xs, ys)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=48)
    args = p.parse_args()
    assert args.out.is_dir() and not (args.out/'SEARCH.json').exists()
    root = Path(__file__).resolve().parents[2]
    population_file = root/'artifacts/rbfit/shrinkage_v1/idm_population.json'
    pop = json.loads(population_file.read_text())
    assert pop['parameter_order'] == ['v0', 'time_headway', 'accel', 'decel', 'min_gap']
    support, weights = np.array(pop['support']), np.array(pop['weights'])
    rows, provenance = [], []
    begin = time.monotonic()
    for world in range(969400, 969432):
        a, receipt = read_episode(native_episode(args.root, world), world)
        rows.extend(component_rows(a, world, support, weights))
        provenance.append(receipt)
    # Prefix fit then later development-panel conditional check, both development.
    fit = [r for r in rows if r['time_s'] <= 300]
    dev = [r for r in rows if r['world'] >= 969420 and r['time_s'] > 300]
    fit_score, dev_score = ConditionalScore(fit), ConditionalScore(dev)
    best, trials = bayes_search(lambda x: fit_score.metrics(x)['objective'], args.iterations)
    artifact = dict(schema='wm_phys.learned_parameters.v1', name='learned_v1',
        fit_worlds=list(range(969400, 969432)), selection_worlds=list(range(969420, 969432)),
        departure_parameters=dict(zip(NAMES, best.tolist())),
        fit_interval_s=[240, 300], conditional_check_interval_s=[300, 900],
        frozen_idm_population_sha256=hashlib.sha256(population_file.read_bytes()).hexdigest(),
        rule_forms='SUMO1.27.1 SumoInsertion; max/last speed; free/first lane',
        ownership='IDM/LC populations unchanged; FIT/LANE own those coordinates',
        status='conditional screen only; autonomous development comparison required')
    report = dict(schema='rb2lphys.search.v1', method='RBF Gaussian-process Bayesian optimisation, expected improvement',
        parameters=NAMES, budget=len(trials), fit_rows=len(fit), dev_rows=len(dev), draws=16,
        objective='equal world/entrance mean(speed CRPS/local speed limit + lane Brier + birth Brier)',
        fitted=fit_score.metrics(best), baseline_fit=fit_score.metrics(np.zeros(3)),
        development=dev_score.metrics(best), baseline_development=dev_score.metrics(np.zeros(3)),
        trials=trials, artifact=artifact, provenance=provenance, seconds=time.monotonic()-begin,
        limitations=['Conditional birth-time recall, not insertion precision or long-run traffic fidelity.',
                    'Existing population already saw all32FIT worlds; this is development evidence.'])
    (args.out/'learned_v1.json').write_text(json.dumps(artifact, indent=2)+'\n')
    (args.out/'SEARCH.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('fit_rows','dev_rows','fitted','baseline_fit','development','baseline_development','seconds')}), flush=True)


if __name__ == '__main__':
    main()
