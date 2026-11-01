"""Regression gates for masked-prefix geometry and host insertion high-water."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from .speed2_runtime import bind_ledger, compact_geometry


class Speed2CPU(unittest.TestCase):
    def test_host_ledger_observer_preserves_return_and_tracks_all_rows(self):
        calls = []
        process = SimpleNamespace(_commit=lambda *a: calls.append(a) or a[-1])
        entries = SimpleNamespace(process=process)
        result = ({}, entries, {})
        proxy = SimpleNamespace(live_slots=3)
        restores=[]
        original=process._commit
        wrapped = bind_ledger(proxy, lambda *a: result, restores)
        self.assertIs(wrapped(None, None, None, None), result)
        mask = object()
        self.assertIs(process._commit({}, SimpleNamespace(inserted=[(0,4),(7,8)]), mask), mask)
        self.assertEqual(proxy.live_slots, 9)
        process._commit({}, SimpleNamespace(inserted=[(0,2)]), mask)
        process._commit({}, SimpleNamespace(inserted=[]), mask)
        self.assertEqual(proxy.live_slots, 9)
        self.assertEqual(len(calls), 3)
        self.assertEqual(restores,[(process,original)])

    @torch.no_grad()
    def test_real_frozen_bridge_keeps_memory_across_prefix_growth_and_exit(self):
        from tools.wmteach import runtime
        from wms.control.mappo.eval_bridge import load_policy_fn
        from wms.model.cav import WM_LANE_TARGET_KEY
        from .wm import executor_adapter
        from .plan_wm import plan_executor
        from tools.coopnative.test_cpu import state
        binding = json.loads(Path(os.environ['SPEED2_EXPORT_BOUND']).read_text())
        mp = load_policy_fn(binding['path'], device='cpu')
        rows = [dict(vid='a',s_m=-100.,v_mps=15.,lane=1,is_cav=True,length_m=5.),
                dict(vid='b',s_m=-60.,v_mps=10.,lane=1,is_cav=True,length_m=5.),
                dict(vid='h',s_m=-60.,v_mps=10.,lane=0,is_cav=False,length_m=5.)]
        full = state(rows)
        for key, value in full.items():
            full[key] = torch.nn.functional.pad(value, (0,13))
        full['a_mps2'] = torch.zeros_like(full['s_m'])
        acts = [dict(id='policy',window=300,residuals={'a':-.25,'b':.1}),
                dict(id='MAPPO',window=0)]
        full = {k:v.expand(2,*v.shape[1:]).clone() for k,v in full.items()}
        proxy = SimpleNamespace(live_slots=3)
        with executor_adapter(), plan_executor():
            old = runtime.option_controller(mp, acts, rows)
            with compact_geometry(proxy):new = runtime.option_controller(mp, acts, rows)
        for tick in range(10):
            if tick == 3:
                proxy.live_slots = 9
                for k in ('s_m','v_mps','lane','present','is_cav','length_m'):
                    full[k][:,8] = full[k][:,0]
                full['s_m'][:,8] -= 25.
            if tick == 6:
                full['present'][:,0] = False
            full[WM_LANE_TARGET_KEY] = full['lane'].clone()
            a = copy.deepcopy(full)
            b = copy.deepcopy(full)
            expected = old(a)
            with compact_geometry(proxy):actual = new(b)
            mask = full['present'].bool()
            self.assertTrue(torch.equal(expected[mask],actual[mask]), f'tick {tick}')
            # Advance observable kinematics; bridge memories remain independent.
            full['s_m'], full['v_mps'], full['a_mps2'] = expected[...,0],expected[...,2],expected[...,3]

    def test_trajectory_gate_includes_exit_step_but_ignores_virgin_placeholders(self):
        from .speed2_runtime import trajectory_errors
        a = {k:torch.zeros(1,3,2) for k in ('positions','speeds','lanes','accel')}
        a['masks'] = torch.tensor([[[True,False],[False,False],[False,False]]])
        b = copy.deepcopy(a)
        b['positions'][0,2,1] = 1000.
        self.assertEqual(trajectory_errors([a],[b])['positions'],0.)
        b['positions'][0,1,0] = .5
        self.assertEqual(trajectory_errors([a],[b])['positions'],.5)


if __name__ == '__main__':
    unittest.main()
