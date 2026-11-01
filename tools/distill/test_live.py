"""Portable inference, budget and fake-plant execution gates; no SUMO starts."""
from contextlib import nullcontext
from copy import deepcopy
import io
import os
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import numpy as np

from .data import read, KEYS
from .portable import predict, choose, Student
from .live import panel, budget, approval, admit_event, WORLD_CREDIT, episode, credit


def fixture():
    return (Path(os.environ['DISTILL_TEST_ROOT']) if 'DISTILL_TEST_ROOT' in os.environ
            else Path(__file__).resolve().parents[2]/'docs/paper_results/distill_live')


class ImmediatePool:
    def __init__(self,*args,**kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self,*args):
        pass

    def submit(self,fn,root,key,action):
        result = {'O0':100.,'O2':60.,'O2_solo0':90.,'O2_solo1':80.}[action['id']]
        return types.SimpleNamespace(result=lambda:result)


class Backend:
    def __init__(self):
        self.env = types.SimpleNamespace(now=300)
        self.totals = {'team_delay_vs':0.}
        self.actual = 300
        self.closed = False

    def fingerprint(self):
        return dict(time=self.env.now)

    def close(self):
        self.closed = True


class ScriptedStudent:
    def __init__(self,*args):
        pass

    def step(self,state):
        action = dict(id='O2',members=['a','b'],window=30) if state.time_s == 300 else dict(id='O0',members=[])
        return dict(time_s=state.time_s,executed_action=action,group=action['members'],
                    initial_rows=[],decision=dict(selected=action['id']))


class Tests(unittest.TestCase):
    def test_portable_original_1020_context_parity(self):
        f = fixture();policy = read(f/'POLICY.json');gate = read(f/'PARITY.json')
        pred = predict(policy,np.array(gate['x']))
        np.testing.assert_allclose(pred,np.array(gate['predictions']),atol=1e-8,rtol=1e-10)
        actual,_ = choose(policy,pred,np.array(gate['available']))
        np.testing.assert_array_equal(actual,gate['selected'])
        self.assertEqual(gate['states'],1020)
        self.assertTrue(np.isfinite(pred).all())

    def test_panel_exact_disjoint_all_cells(self):
        worlds = panel()
        self.assertEqual(len(worlds),24)
        self.assertEqual(len({w['seed'] for w in worlds}),24)
        self.assertEqual({w['seed'] for w in worlds},set(range(984000,984024)))
        self.assertEqual([sum(w['cell']==c for w in worlds) for c in range(12)],[2]*12)
        self.assertEqual(worlds[0]['penetration'],.4)
        self.assertTrue(all(w['seed'] not in set(read(fixture()/'POLICY.json')['train_worlds']) for w in worlds))

    def test_budget_complete_quartets(self):
        b = budget()
        self.assertEqual(b['total'],b['paired']+b['all_event_credit']+b['mechanics_extra'])
        self.assertEqual(b['total']+b['unallocated'],200000)
        event = dict(time_s=960,executed_action=dict(id='O2',members=['a','b']),group=['a','b'])
        self.assertEqual(admit_event(event,0),4320)
        event = dict(time_s=960,executed_action=dict(id='O2',members=['a','b']),group=['a','b'])
        self.assertEqual(admit_event(event,WORLD_CREDIT-4319),0)
        self.assertEqual(event['executed_action']['id'],'O0')
        self.assertEqual(event['proposed_action']['id'],'O2')
        self.assertTrue(event['budget_pruned'])

    def test_no_budget_outcome_selection(self):
        original = dict(time_s=300,executed_action=dict(id='O1',members=['a','b']),group=['a','b'])
        a,b = deepcopy(original),deepcopy(original)
        a['hypothetical_D'] = -1e12;b['hypothetical_D'] = 1e12
        self.assertEqual(admit_event(a,0),admit_event(b,0))
        self.assertEqual(a['executed_action'],b['executed_action'])

    def test_approval_exact_and_not_pending(self):
        for text in ('pending S-DISTILL BUDGET APPROVED 200000','S-WMTEACH BUDGET APPROVED 100000','S-DISTILL BUDGET APPROVED 100000'):
            with self.assertRaises(PermissionError):approval(text)
        self.assertIn('200000',approval('MAIN S-DISTILL BUDGET APPROVED 200000'))

    def test_live_inference_masks_and_cadence(self):
        from tools.coopplan.predictor import ObservableState
        policy = read(fixture()/'POLICY.json')
        student = Student(policy)
        event = student.step(ObservableState(300,tuple(),'fake'))
        self.assertEqual(event['executed_action']['id'],'O0')
        with self.assertRaises(ValueError):student.step(ObservableState(300,tuple(),'fake'))
        self.assertEqual(tuple(policy['keys']),KEYS)

    def test_fake_episode_all_group_twins_before_execution(self):
        from tools.coopplan import plant
        b = Backend();writes = {};order = []
        world = panel()[1]
        observations = types.ModuleType('tools.leverprobe.backend')
        observations.observations = lambda env:[]
        def fake_tick(backend,event,tape):
            if event and event['time_s'] == 300 and backend.env.now == 300:
                self.assertTrue(any(p.endswith('/reservations/s984001_t300.json') for p in writes))
            backend.env.now += 1;backend.actual += 1
            backend.totals['team_delay_vs'] += 1
            return dict(requests=1,overrides=0)
        def record(path,value):
            writes[str(path).replace('\\','/')] = deepcopy(value);order.append(str(path))
        with patch.dict('sys.modules',{'tools.leverprobe.backend':observations}), \
             patch('tools.distill.live.setup'),patch('tools.distill.live.verify'), \
             patch('tools.distill.live.episode_baseline',return_value=dict(warmup=b.fingerprint())), \
             patch('tools.distill.live.fresh',return_value=(b,Path('synthetic/student'))), \
             patch('tools.distill.live.read',return_value={}), \
             patch('tools.distill.live.write',side_effect=record), \
             patch('tools.distill.live.Student',ScriptedStudent), \
             patch('tools.distill.live.ProcessPoolExecutor',ImmediatePool), \
             patch('tools.distill.live.gzip.open',side_effect=lambda *a,**k:nullcontext(io.StringIO())), \
             patch.object(plant,'tick',side_effect=fake_tick):
            self.assertEqual(episode(Path('synthetic'),world),world['seed'])
        result = writes['synthetic/student/RESULT.json']
        self.assertEqual(len(result['history']),12)
        self.assertEqual(result['credits'][0]['D_G'],40.)
        self.assertEqual(result['credits'][0]['C_G'],10.)
        self.assertEqual(result['credits'][0]['member_alone_D'],[10.,20.])
        self.assertEqual(result['credit_sim_s'],1680)
        self.assertEqual(result['clocks']['600']['team_delay_vs'],600.)
        self.assertEqual(result['clocks']['900']['team_delay_vs'],900.)
        self.assertTrue(b.closed)

    def test_credit_rejects_mismatched_prefix_and_closes(self):
        from tools.coopplan import plant
        b = Backend()
        observations = types.ModuleType('tools.leverprobe.backend')
        observations.observations = lambda env:[]
        state = dict(time_s=300,world=panel()[0],history=[],rows=[],prefix={'wrong':1})
        with patch.dict('sys.modules',{'tools.leverprobe.backend':observations}), \
             patch('tools.distill.live.read',return_value=state), \
             patch('tools.distill.live.fresh',return_value=(b,Path('synthetic'))), \
             patch.object(plant,'replay'):
            with self.assertRaisesRegex(ValueError,'BITWISE'):
                credit(Path('synthetic'),'key',dict(id='O0',members=[]))
        self.assertTrue(b.closed)


if __name__ == '__main__':unittest.main()
