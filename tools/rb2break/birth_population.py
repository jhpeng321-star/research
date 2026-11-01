"""Motion-only incoming HDV population: a birth cohort, not a resident snapshot.

Fits finite mixture weights on frozen joint physical support. No neural module,
traffic-outcome objective, future observation, driver type or assignment input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tools.rb2break.run import DEV
from tools.rbint.motion_posterior import chain_gaps, fit_kernels
from tools.rbint.observations import load_motion_history
from tools.rbeval.core import digest, save_new


def births(start, history):
    times, kin = history['time_s'], history['kin']
    vids, present = history['vid_index'], history['mask'].astype(bool)
    events = {}
    for event in start['past_entries']:
        key = (float(event['time']), int(event['lane']))
        events.setdefault(key, []).append(event)
    seen = set(map(int, vids[0, present[0]]))
    matched, skipped = {}, 0
    for tick in range(1, len(times)):
        for slot in np.flatnonzero(present[tick] & (vids[tick] >= 0)):
            identity = int(vids[tick, slot])
            if identity in seen:
                continue
            seen.add(identity)
            if kin[tick, slot, 1:4].sum() != 1:
                skipped += 1
                continue
            lane = int(kin[tick, slot, 1:4].argmax())
            candidates = events.get((float(times[tick]), lane), [])
            # Match observed insertion clock/lane/position, never source ordinal
            # or regenerated CAV/type assignments. Tolerance is FLOAT32 ulp only.
            position = float(kin[tick, slot, 0])
            tolerance = 2*abs(float(np.spacing(np.float32(position))))
            candidates = [e for e in candidates if abs(float(e['s_m'])-position) <= tolerance]
            if len(candidates) != 1:
                skipped += 1
                continue
            event = candidates[0]
            matched[identity] = dict(time=float(times[tick]), lane=lane, is_cav=bool(event['is_cav']))
    return matched, skipped


def logsumexp(x, axis=-1):
    top = x.max(axis=axis, keepdims=True)
    result = top + np.log(np.exp(x-top).sum(axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis)


def fit_weights(log_likelihood, prior, *, iterations=200):
    weights = np.asarray(prior, dtype=float)/np.sum(prior)
    trace = []
    for iteration in range(iterations):
        with np.errstate(divide='ignore'):
            joint = log_likelihood+np.log(weights)[None]
        evidence = logsumexp(joint)
        new = np.exp(joint-evidence[:, None]).mean(0)
        new /= new.sum()
        trace.append(dict(iteration=iteration, negative_log_likelihood=float(-evidence.mean()),
                          max_weight_change=float(np.abs(new-weights).max())))
        weights = new
        if trace[-1]['max_weight_change'] < 1e-8:
            break
    return weights, trace


def acceleration_crps(predictions, target, weights):
    order = predictions.argsort(axis=1)
    x = np.take_along_axis(predictions, order, axis=1)
    w = np.broadcast_to(weights, x.shape)
    w = np.take_along_axis(w, order, axis=1)
    pair = (w*x*(2*w.cumsum(1)-w-1)).sum(1)
    return (np.abs(predictions-target[:, None])*weights).sum(1)-pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    assert args.out.parent.is_dir() and not args.out.exists()
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    path = root/'artifacts/rbfit/shrinkage_v1/idm_population.json'
    population = json.loads(path.read_bytes())
    support = np.array(population['support'])
    prior = np.array(population['weights']); prior /= prior.sum()
    sigma = float(population['sigma_accel_mps2'])
    extract, predict, kernel_sha = fit_kernels()
    likelihood, validation, audits = [], [], []
    for world in DEV:
        start, history = load_motion_history(world)
        cohort, skipped = births(start, history)
        hdv = {i:b for i,b in cohort.items() if b['lane'] < 2 and not b['is_cav']}
        arrays = dict(time_s=history['time_s'], veh_kin=history['kin'], veh_mask=history['mask'],
                      vid=history['vid_index'], is_cav=~np.isin(history['vid_index'], list(hdv)))
        rows, _, _ = extract(arrays, start=180., stop=300.)
        rows = chain_gaps(rows, arrays)
        counts = dict(fit_drivers=0, validation_drivers=0, fit_rows=0, validation_rows=0)
        for identity, birth in sorted(hdv.items()):
            if birth['time'] > 270:
                continue
            selected = rows[(rows[:, 1] == identity) & (rows[:, 0] >= birth['time'])
                            & (rows[:, 0]+rows[:, 9] <= min(300., birth['time']+60))]
            if not len(selected):
                continue
            assert np.all(selected[:, 0]+selected[:, 9] <= 300.)
            dt = selected[:, 9]
            assert np.all(dt == dt[0])
            prediction = (predict(selected[:, 3, None], selected[:, 5, None],
                selected[:, 6, None], support, float(dt[0]))-selected[:, 3, None])/dt[:, None]
            target = (selected[:, 4]-selected[:, 3])/dt
            if birth['time'] <= 240:
                likelihood.append(-((prediction-target[:, None])**2).sum(0)/(2*sigma*sigma))
                counts['fit_drivers'] += 1
                counts['fit_rows'] += len(target)
            else:
                validation.append((world, prediction, target))
                counts['validation_drivers'] += 1
                counts['validation_rows'] += len(target)
        audits.append(dict(world=world, matched_births=len(cohort), skipped_unmatched=skipped,
                           **counts))
        print('EXTRACTED', world, counts, flush=True)
    assert likelihood and validation, 'birth cohort unavailable: no fit or inference permitted'
    weights, trace = fit_weights(np.stack(likelihood), prior)
    metrics = {}
    for label, value in (('frozen_resident_population', prior), ('birth_cohort_population', weights)):
        per_world = {}
        for world in DEV:
            values = [acceleration_crps(pred, target, value) for s,pred,target in validation if s == world]
            if values:
                per_world[str(world)] = float(np.concatenate(values).mean())
        metrics[label] = dict(mean_world_crps_mps2=float(np.mean(list(per_world.values()))),
                              per_world=per_world)
    improved = (metrics['birth_cohort_population']['mean_world_crps_mps2']
                < metrics['frozen_resident_population']['mean_world_crps_mps2'])
    artifact = dict(schema='wm_phys.entrant_population.v1', parameter_order=population['parameter_order'],
        source_population_sha256=digest(path), support=population['support'], weights=weights.tolist(),
        fit_worlds=list(DEV), fit_birth_interval_s=[180, 240], validation_birth_interval_s=[240, 270],
        conditioning_end_s=300, physical_likelihood='stable-mainline HDV SUMO IDM accelerations; frozen Gaussian scale',
        sigma_accel_mps2=sigma, motion_validation_improved=bool(improved), metrics=metrics,
        kernel_sha256=kernel_sha, source_sha256=digest(__file__),
        caution='birth-disjoint internal motion validation, not independent-world or traffic qualification')
    args.out.mkdir()
    save_new(args.out/'entrant_population.json', artifact)
    save_new(args.out/'FIT.json', dict(audits=audits, trace=trace, metrics=metrics,
        improved=bool(improved), source_sha256=digest(__file__)))
    print(json.dumps(dict(improved=bool(improved), metrics=metrics, drivers=len(likelihood))))


if __name__ == '__main__':
    main()
