"""New fitted-input diagnostic; preserves every completed encoder-input probe."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch

from fit_probe import BASE, digest, install_guard, prepare, run, safe
from wms.control.arrival_fit import LearnedArrivalFit
from wms.control.fitted_hidden import FittedHiddenState, IDM_ORDER, LC_ORDER
from wms.control.hidden_state import plan_hidden_state_futures

FIT = '/home/ubuntu/rbfit20261031/fit_mixture_v4'
LANE = '/home/ubuntu/rbfit20261031/lane_v1'


def observed_ids(world, state):
    """Bind numeric recorder VID by exact observable tape/start correspondence.

    start.vehicle_ids is a route-string list, NOT the numeric VID used by FIT.
    Never use its ordinal as a posterior key.
    """
    root = Path('/home/ubuntu/wmdata20261031/views')
    manifest = safe(root/f'world_{world}.json')
    name = json.loads(manifest.read_text())['native']['episode']
    assert name.startswith(f'w{world}m') and name.endswith('__native')
    assert '/' not in name and '\\' not in name and '..' not in name
    ep = safe(root/'episodes'/name)
    t = np.load(safe(ep/'time_s.npy'), mmap_mode='r', allow_pickle=False)
    ti = int(np.flatnonzero(t == 300)[0])
    mask = np.load(safe(ep/'veh_mask.npy'), mmap_mode='r', allow_pickle=False)[ti].astype(bool)
    kin = np.load(safe(ep/'veh_kin.npy'), mmap_mode='r', allow_pickle=False)[ti][mask]
    vid = np.load(safe(ep/'vid.npy'), mmap_mode='r', allow_pickle=False)[ti][mask]
    n = int(state['present'].sum())
    assert len(vid) == n and len(np.unique(vid)) == n and (vid >= 0).all()
    assert np.array_equal(kin[:, 0].astype(np.float32), state['s_m'][0, :n].numpy())
    assert np.array_equal(kin[:, 4].astype(np.float32), state['v_mps'][0, :n].numpy())
    ids = torch.full_like(state['lane'], -1)
    ids[0, :n] = torch.as_tensor(np.array(vid, copy=True))
    return ids, dict(manifest_sha256=digest(manifest), t0_numeric_vid=vid.tolist(),
                     exact_observable_correspondence=True,
                     vid_sha256=digest(ep/'vid.npy'))


def probe_world(world, out, fit, arrival, lane):
    started = time.monotonic()
    while os.getloadavg()[0] > 24:
        time.sleep(5)
    model, state, history, _, eh, queue, receipt = prepare(world)
    ids, identity = observed_ids(world, state)
    base = dict(future_sampling='hidden_state', seed=20261031, n_samples=8,
        row_keys=[world], fitted_state=fit, vehicle_ids=ids, t0_s=300,
        entrance_histories=[eh], queues=[queue], arrival_fit=arrival)
    modes, driver_sd, requests, source_receipts = {}, {}, {}, {}
    sources = [('both', True, True)]
    if world == 969400:
        sources += [('drivers', True, False), ('arrivals', False, True), ('fixed', False, False)]
    for name, drivers, arrivals in sources:
        kwargs = dict(base, sample_present=drivers, sample_arrivals=arrivals)
        before = torch.get_rng_state().clone()
        futures = plan_hidden_state_futures(model, state, history, **kwargs)
        paired = plan_hidden_state_futures(model, state, history, **kwargs)
        hdv = state['present'] & ~state['is_cav']
        driver_sd[name] = {n: float(torch.stack([f.kwargs['driver_params'][n][hdv]
            for f in futures]).std(0).mean()) for n in IDM_ORDER}
        for f, other in zip(futures, paired):
            for n in f.kwargs['driver_params']:
                assert torch.equal(f.kwargs['driver_params'][n], other.kwargs['driver_params'][n])
            np.testing.assert_array_equal(f.kwargs['entries'].t_next, other.kwargs['entries'].t_next)
            a, b = f.kwargs['entrant_driver_sampler'], other.kwargs['entrant_driver_sampler']
            for entrance in (0, 1):
                for ordinal in (0, 7, 19):
                    assert a.candidate(0, entrance, ordinal) == b.candidate(0, entrance, ordinal)
        assert torch.equal(before, torch.get_rng_state())
        modes[name], requests[name] = [], []
        for f in futures:
            result = run(model, state, history, f, 120)
            modes[name].append(result)
            process = f.kwargs['entries']
            clocks = {'120': process.due_total[0].tolist()}
            for step in range(240, 1200):
                process._emit_due(0)
                if step+1 in (600, 1200):
                    clocks[str((step+1)//2)] = process.due_total[0].tolist()
            requests[name].append(clocks)
        source_receipts[name] = futures[0].receipt
        if name == 'fixed':
            assert len({r['tape_sha256'] for r in modes[name]}) == 1
        print('DONE', world, name, round(time.monotonic()-started, 2), flush=True)
    # Separate LC latent plumbing only: candidate is not adopted by physics here.
    lane_kwargs = dict(base, fitted_state=lane, n_samples=8)
    lf = plan_hidden_state_futures(model, state, history, **lane_kwargs)
    lane_sd = {n:float(torch.stack([f.kwargs['driver_params'][n][hdv] for f in lf]).std(0).mean())
               for n in LC_ORDER}
    real_path = safe(BASE/f's{world}_native/cumulative_120.json')
    real = json.loads(real_path.read_text())
    result = dict(schema='rbunc_fitted_probe/1', world=world, horizon=120, receipt=receipt,
        identity=identity, modes=modes, requests=requests, driver_sd=driver_sd,
        optional_lc_input_sd=lane_sd, source_receipts=source_receipts,
        paired_hidden_inputs_exact=True, global_rng_unchanged=True,
        real={n:real[n] for n in ('team_delay_vs','road_delay_vs','source_wait_vs')},
        real_sha256=digest(real_path), seconds=time.monotonic()-started,
        source_sha256=digest(__file__))
    with (out/f'fitted_w{world}.json').open('x') as stream:
        json.dump(result, stream, indent=1, allow_nan=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--worker', type=int, choices=range(4), required=True)
    args = p.parse_args()
    assert args.out.is_dir()
    os.nice(19)
    torch.set_num_threads(1)
    install_guard()
    fit = FittedHiddenState.from_directory(FIT)
    lane = FittedHiddenState.from_directory(FIT, lane_directory=LANE)
    arrival = LearnedArrivalFit.from_file(args.out/'observed_arrivals.json')
    for world in range(969400+args.worker, 969432, 4):
        assert not (args.out/f'fitted_w{world}.json').exists(), 'preserve finished work'
        probe_world(world, args.out, fit, arrival, lane)


if __name__ == '__main__':
    main()
