"""CPU-only audit of actual frozen runtime bindings before reserved data access."""
import argparse
import copy
import io
import json
import time
import zipfile

import numpy as np

from .audit import ARMS, digest, safe
from .core import features
from .reconstruct import observe_prefix
from .restart import current_names, prefix, road_rectangle


def read_prefixes(path, seed, t0):
    with zipfile.ZipFile(safe(path)) as z:
        p, meta = {}, {}
        for arm in ARMS:
            name = f's{seed}_{arm}'
            with np.load(io.BytesIO(z.read(name+'.npz')), allow_pickle=False) as a:
                tape = {k: a[k] for k in a.files}
            p[arm] = prefix(tape, t0)
            meta[arm] = json.loads(z.read(name+'.json'))
    return p, meta


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prefixes', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    from . import runtime as r
    r.torch.set_num_threads(1)
    start = time.monotonic()
    rt = r.load('cpu')
    ps, meta = read_prefixes(a.prefixes, 969400, 360)
    columns = sorted(set().union(*(current_names(ps[arm]) for arm in ARMS)))
    batches, events = [], []
    for arm in ARMS:
        batch, ev = r.make_batch(ps[arm], columns, rt['model'], meta[arm]['queue300'],
                                 meta[arm]['previous_commands']['360'])
        batches.append(batch)
        events.append(ev)
        # Feature extraction from a packed WM restart must match the direct tape.
        frame = {k: v[0].numpy() for k, v in batch.state.items()
                 if k in ('s_m', 'lane', 'v_mps', 'a_mps2', 'present', 'is_cav')}
        x = features(r.imagined_observation(frame, ev, 360))
        direct = r.imagined_observation(frame, ev, 360)
        assert np.array_equal(x, features(direct))
        # Real projection equality independent of ID packing.
        tape = {k: v for k, v in ps[arm].items() if k != 'names'}
        assert np.allclose(x, features(observe_prefix(tape, 360)), atol=1e-4, rtol=1e-6)

    # Real-runtime replay/entries, repeated independent construction: exact CRN.
    one = r.rollout(rt, batches, events, 360, 0, 'cpu', ps, columns, seconds=2)
    two = r.rollout(rt, batches, events, 360, 0, 'cpu', ps, columns, seconds=2)
    assert one == two, 'same-prefix independent restart is not deterministic'

    # Score the same synthetic birth/exit transition through production cost code.
    torch = r.torch
    before = dict(present=np.array([[1, 0, 1]]*5, bool), v_mps=np.array([[10., 0., 5.]]*5))
    after = dict(present=np.array([[1, 1, 0]]*5, bool), v_mps=np.array([[20., 15., 100.]]*5))
    frame = {k: torch.tensor(v) for k, v in before.items()}
    for k in ('s_m', 'a_mps2', 'lane', 'is_cav'):
        frame[k] = torch.zeros((5, 3))
    rec = type('Rec', (), {})()
    rec.frames = [frame]
    rec.queues = [np.ones((5, 2))*3]
    rec.births = [torch.tensor([[0, 1, 0]]*5, dtype=torch.bool)]
    roll = dict(masks=torch.tensor(after['present'])[:, None], speeds=torch.tensor(after['v_mps'])[:, None])
    scored = r.score(roll, rec, events, 360)
    expected = road_rectangle(before, after)+3.
    assert np.allclose(scored['J'], expected, atol=1e-12)

    # Production terminal path, including births, exits, RSU and the exact horizon
    # boundary. No actual long rollout is needed to test these observation clocks.
    n = 180
    frame = dict(s_m=torch.tensor([[-100., -1000., 1000.]]*5),
                 lane=torch.tensor([[0, 0, 1]]*5), v_mps=torch.full((5, 3), 15.),
                 a_mps2=torch.full((5, 3), 2.), present=torch.tensor([[1, 1, 0]]*5).bool(),
                 is_cav=torch.tensor([[1, 0, 0]]*5).bool())
    rec.frames = [frame]*n
    rec.queues = [np.ones((5, 2))*3]*n
    rec.births = [torch.zeros((5, 3), dtype=torch.bool) for _ in range(n)]
    rec.births[119][:, 1] = True  # birth exactly at the 60s boundary
    synthetic = {k: frame[v][:, None].expand(5, n, 3).clone() for k, v in
                 (('positions', 's_m'), ('lanes', 'lane'), ('speeds', 'v_mps'),
                  ('accel', 'a_mps2'), ('masks', 'present'))}
    result = r.score(synthetic, rec, events, 360)
    for horizon in (60, 90):
        for row in range(5):
            end = {k: v[row].numpy() for k, v in frame.items()}
            ev = np.concatenate((events[row], [[420., 0.]]))
            expected_x = features(r.imagined_observation(end, ev, 360+horizon))
            np.testing.assert_array_equal(result['windows'][str(horizon)]['x'][row], expected_x)
            expected_j = horizon*(2*(1-15/31.29)+6.)
            assert abs(result['windows'][str(horizon)]['J'][row]-expected_j) < 1e-4

    # Shared identity columns + private per-arm plans must share the actual normal
    # and lane draws, independent of callback/batch row order.
    seed = r.frozen.FuturesConfig.from_trainer(rt['cfg'])
    plan1 = r.frozen.SampledFutures(seed, 190).plan(rt['models'], batches[0].state,
                                                batches[0].rollout_kwargs, 1080)[0]
    plan2 = r.frozen.SampledFutures(seed, 190).plan(rt['models'], batches[0].state,
                                                batches[0].rollout_kwargs, 1080)[0]
    for f1, f2 in zip(plan1, plan2):
        for key in f1:
            if isinstance(f1[key], dict):
                assert all(torch.equal(v, f2[key][k]) for k, v in f1[key].items())
            elif torch.is_tensor(f1[key]):
                assert torch.equal(f1[key], f2[key])
            else:
                assert f1[key] == f2[key]

    # Freeze full-long plans: short rollouts consume an identical stochastic prefix.
    # Above repeat includes full-history encoding, entry CRNs and entrant learned mix.
    assert one['steps'] == 4 and one['slots'] >= len(columns)
    assert sum(one['mix_assigned']) > 0, 'audit must exercise newborn driver assignment'
    # Suffix poisoning is also checked through production make_batch, not only features.
    altered = copy.deepcopy(ps['mappo'])
    altered['time_s'] = np.r_[altered['time_s'], 9999.]
    # Re-prefix before production adapter, mimicking the exporter/runner boundary.
    altered['vehicle_ids'] = np.array([altered['names'][i] for i in range(len(altered['names']))])
    again = prefix(altered, 360)
    bb, ee = r.make_batch(again, columns, rt['model'], meta['mappo']['queue300'],
                          meta['mappo']['previous_commands']['360'])
    for k in bb.rollout_kwargs:
        assert torch.equal(bb.rollout_kwargs[k], batches[2].rollout_kwargs[k])
    assert np.array_equal(ee, events[2])
    result = dict(schema='wmvalue_runtime_audit/1', verdict='PASS', device='cpu',
        fit_seed=969400, t0=360, reserved_outcomes_opened=False,
        restart_repeat_exact=True, entry_mix_exercised=one['mix_assigned'],
        sensing_parity=True, history_s=60, cost_union_end_speed=True,
        source_end_step=True, controller_prefix_replay=True,
        terminal_60_90_path=True, full_long_plan_crn=True,
        seconds=time.monotonic()-start, two_second_costs=one,
        prefix_sha256=digest(a.prefixes), wm_sha256=r.frozen.digest(r.frozen.WM_PATH),
        code_sha256={n: digest(safe(__file__).parent/n) for n in
                     ('runtime.py', 'restart.py', 'reconstruct.py', 'audit_runtime.py')})
    with safe(a.out).open('x') as out:
        json.dump(result, out, indent=1, allow_nan=False)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
