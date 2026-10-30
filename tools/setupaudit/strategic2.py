"""Prospective K1 repair check on all 24 starts, using the primary K8 plan's future 0."""
# ruff: noqa: E402 -- install isolated modules before importing the scoring harness.
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import torch

from strategic_test2 import install

install()
import panel2 as P
from wms.control.alwayscmd import inferred_cap
from wms.control.bridge import make_cav_controller
from wms.control.train_cli import GroupSignal
from wms.model.cav import EXEC_CAP_KEY, WM_MERGE_KEY


class Entries(P.Entries):
    """Normal endogenous admissions for BOTH arms; no frozen-admission row here."""
    def insert(self, state):
        before = self.process.pending.sum(-1).copy()
        self.due_area = np.zeros(state['present'].shape[0])
        mask = self.process.insert(state)
        self.states.append({k: state[k].clone() for k in
            ('present', 'is_cav', 'lane', 's_m', 'v_mps', 'entrant') if k in state})
        self.queues.append(self.process.backlog().copy())
        self.pending.append(self.process.pending.copy())
        self.new_masks.append(mask.clone())
        self.real_source.append(.5*before+self.due_area)
        return mask


def controller(candidate, mappo, native):
    cp, ce, _ = candidate

    def deterministic(policy):
        def call(obs, signal, **kw):
            return policy(obs, signal, deterministic=True, **kw)
        call.accel_max = policy.accel_max
        call.residual_on_native = bool(getattr(policy, 'residual_on_native', False))
        return call

    common = dict(native=native, native_accel_max=1.2, symmetric_exec_cap=True,
                  mask_forced_merge=True, native_ramp_merge='wm_law', native_lane_change=True)
    ctrl = make_cav_controller(deterministic(cp), GroupSignal(ce), P.safe_accel, **common)
    repaired = None

    def callback(state):
        nonlocal repaired
        states = [{k: v[i:i+1] if torch.is_tensor(v) and v.ndim and v.shape[0] == 2 else v
                   for k, v in state.items()} for i in range(2)]
        cs, ms = states
        if repaired is None:
            repaired = make_cav_controller(deterministic(mappo.policy),
                GroupSignal(mappo.group_encoder), P.safe_accel, lane_rules=mappo.lane_rules,
                act_mask=torch.ones_like(ms['present']), policy_ramp_merge=True,
                policy_strategic_merge=True, **common)
        cc = ctrl(cs)
        ms[EXEC_CAP_KEY] = inferred_cap(ms)
        mc = repaired(ms)
        applied = torch.maximum(-ms['v_mps']/.5, torch.minimum(mc[..., 3], ms[EXEC_CAP_KEY]))
        mc = P.parity_motion(ms, mc[..., 1], applied)
        out = torch.cat((cc, mc), 0)
        state[WM_MERGE_KEY] = torch.cat((cs[WM_MERGE_KEY], ms[WM_MERGE_KEY]), 0)
        mask = state['present'].bool() & state['is_cav'].bool()
        callback.requests.append(((out[..., 1] != state['lane']) & mask).sum(-1))
        return out

    def native_lanes(state):
        mask = state['present'].bool() & state['is_cav'].bool() & (state['lane'] >= 0)
        return torch.stack((mask[0], torch.zeros_like(mask[1])))

    callback.native_lane_mask = native_lanes
    callback.requests = []
    return callback


def main():
    if '--futures' not in sys.argv:
        sys.argv += ['--futures', '1']
    out = Path(sys.argv[sys.argv.index('--out')+1])
    assert P.allowed(out) and out.parent.is_dir() and not out.exists()
    manifest = out.with_suffix('.manifest.json')
    assert not manifest.exists()
    sources = {name: P.digest(path) for name, path in
        [('strategic2', __file__), ('panel2', inspect.getfile(P.main)),
         ('bridge', inspect.getfile(make_cav_controller)), ('bootstrap', inspect.getfile(install))]}
    sources['policy14'] = P.digest(Path(__file__).with_name('strategic_policy14.py'))
    manifest.write_text(json.dumps({'repair': 'policy_strategic_merge', 'default': False,
        'source_sha256': sources, 'sampling': 'all 24 starts; future 0 of primary K8 plan'}, indent=2)+'\n')
    P.ACTORS = ['CTRL0', 'MAPPO_strategic_merge']
    P.controller = controller
    P.Entries = Entries
    P.main()
    result = json.loads(out.read_text())
    result.update(repair='policy_strategic_merge', repair_source_sha256=sources,
                  metric_convention='same physical J and real-boundary recomputation as primary')
    out.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
