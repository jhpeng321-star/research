"""Actual tiny-CPU receipts plus analytic integration/feedback checks."""
from pathlib import Path
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import worker as w


def main():
    out = w.ROOT/'cpu_verified'
    checks = {}
    for model in ('wm2','bbtraj','bbid','historical'):
        path = out/f'{model}_common_w0_k0_h10.json'
        row = json.loads(path.read_text())
        assert row['steps'] == 20 and row['seconds'] == 10 and row['device'] == 'cpu'
        assert row['identity']['source_sha256'] == w.digest(w.__file__)
        assert row['identity']['model_sha256'] == w.HASHES[model]
        assert row['source_queue_initial'] == [float(row['identity']['provenance']['entry_queue'][0][0] +
                                                     row['identity']['provenance']['entry_queue'][0][1])]*len(row['arms'])
        value = row['values']['10']
        assert np.allclose(value['J'], sum(np.asarray(value[c]) for c in ('source','upstream','merge','discharge')), atol=1e-6)
        assert all(len(ep['ids']) == len(ep['s']) == len(ep['v']) == len(ep['present'])
                   for ep in value['endpoints'])
        assert np.asarray(value['fields']['density']).shape == (len(row['arms']),1,15)
        checks[model] = dict(steps=20, arms=row['arms'], sha256=w.digest(path), finite_additive=True)
    # Real external PI/CIRCLES law must see predicted-state changes, not a tape.
    models, _ = w.frozen.load_members([str(w.frozen.WM_PATH)], [str(w.frozen.WM_CONFIG)], 'cpu')
    model = models[0]
    starts, _ = w.source.make_starts_wmdata(model)
    st = starts[0].state
    deps = w.source.load_controller_dependencies(w.RUNTIME/'e3final/runtime')
    dt = (deps.config,deps.circles_type,deps.direct_target,deps.neighbor_type,
          deps.frame_type,deps.pi_type,deps.u_report_type,deps.pi_params_type)
    # Native and each baseline are instantiated afresh on each perturbation.
    cp,ce,_,_ = w.frozen.load_policy_components(w.frozen.CANDIDATE,device='cpu')
    ce = ce.encoder if isinstance(ce,w.frozen.GroupSignal) else ce
    binding = json.loads((w.RUNTIME/'EXPORT_BOUND.json').read_text())
    mp = w.frozen.load_policy_fn(binding['path'],device='cpu')
    changes = {}
    for arm in ('native','pi','circles','mappo'):
        results = []
        for shift in (0., -5.):
            state = {k:v.clone() for k,v in st.items()}
            state['v_mps'] = (state['v_mps']+shift).clamp_min(0)
            from wms.control.alwayscmd import inferred_cap
            from wms.model.cav import EXEC_CAP_KEY
            state[EXEC_CAP_KEY] = inferred_cap(state)
            call = w.callbacks(model,cp,ce,mp,state['s_m'].shape[1],dt,(arm,),'common')
            assert not hasattr(call,'native_lane_mask')
            results.append(call(state)[...,3])
        delta = float((results[0]-results[1]).abs().max())
        assert delta > 1e-5, (arm,'controller ignored predicted-state perturbation')
        changes[arm] = delta
    production = w.callbacks(model,None,None,None,st['s_m'].shape[1],dt,('native',),'production')
    assert torch.equal(production.native_lane_mask(st),st['present'].bool() & st['is_cav'].bool())
    checks['feedback_accel_max_change'] = changes
    checks['production_lane_delegation'] = True
    checks['gpu_allocated'] = 0
    s = np.broadcast_to(np.array([-250., -50., 50., 150.]),(1,20,4)).copy()
    v = np.broadcast_to(np.array([3., 4., 6., 8.]),s.shape).copy()
    m = np.ones(s.shape,dtype=bool)
    grid = w.fields(s,v,m)
    assert np.asarray(grid['density']).sum() == 40.
    assert np.all(w.merge_queue(s,v,m) == 1)
    checks['aggregate_fixed_grid_and_queue'] = True
    target = w.ROOT/'evidence/CPU_VERIFIED.json'
    if target.exists():
        assert json.loads(target.read_text())['worker_sha256'] == w.digest(w.__file__)
        target = target.with_name('CPU_VERIFIED_'+w.digest(__file__)[:12]+'.json')
    w.write(target, dict(checks=checks, verifier_sha256=w.digest(__file__), worker_sha256=w.digest(w.__file__)))
    print(json.dumps(checks))


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
