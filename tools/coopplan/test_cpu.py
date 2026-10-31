"""CPU-only contract/causal-accounting tests; no SUMO imports or process creation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .predictor import (ObservableState, Outcome, Prediction, SumoOraclePredictor,
                        WMPhysPredictor, BackendUnavailable)
from .planner import Planner, candidates, decide, cooperation_credit, TIMES
from .protocol import budget, CAP, approval, worlds


ROWS = (dict(vid='a',s_m=-10.,v_mps=20.,lane=0,is_cav=True),
        dict(vid='b',s_m=-12.,v_mps=22.,lane=1,is_cav=True),
        dict(vid='r',s_m=-5.,v_mps=15.,lane=2,is_cav=False))
STATE = ObservableState(300,ROWS,'toy-prefix')
ACTIONS = (dict(id='O0',members=[]),dict(id='O1',members=['a','b']))


def toy(action,draw=0.):
    delay = 10.+draw-(3. if action['id'] != 'O0' else 0.)
    return dict(team_delay=delay,breakdown=False,member_delay={'a':delay/2,'b':delay/2},
                safety={'collisions':0.,'teleports':0.})


def prediction(base,other,reverse=False):
    b = [Outcome(str(i),v,False) for i,v in enumerate(base)]
    a = [Outcome(str(i),v,False) for i,v in enumerate(other)]
    return Prediction({'O0':b,'O1':list(reversed(a)) if reverse else a},False,'toy')


class ContractTests(unittest.TestCase):
    def test_backend_parity_and_same_planner(self):
        oracle = SumoOraclePredictor(lambda s,a,h:toy(a))
        wm = WMPhysPredictor(lambda s,a,h,d:toy(a,d),lambda s,h,k:[0.]*k)
        for backend in (oracle,wm):
            event,pred = Planner(backend).step(STATE)
            self.assertEqual(event['decision']['selected'],'O1')
            self.assertEqual(event['decision']['member_savings']['O1'],{'a':1.5,'b':1.5})
            self.assertEqual(set(pred.actions),{a['id'] for a in candidates(STATE)})
        x = oracle.predict(STATE,ACTIONS,120,1)
        y = wm.predict(STATE,ACTIONS,120,1)
        for a in x.actions:
            self.assertEqual(x.actions[a][0].team_delay,y.actions[a][0].team_delay)
            self.assertEqual(x.actions[a][0].member_delay,y.actions[a][0].member_delay)

    def test_actual_shared_draws_not_only_matching_row_numbers(self):
        received = []
        calls = []
        def draws(s,h,k):
            calls.append(k)
            return [dict(driver={'tau':1+i},arrivals=[i,2*i]) for i in range(k)]
        def rollout(s,a,h,d):
            received.append((a['id'],deepcopy(d)))
            x = d['driver']['tau']+sum(d['arrivals'])
            d['arrivals'].append(999)  # action-local mutation must not leak
            d['driver']['tau'] = 999
            return toy(a,x)
        p = WMPhysPredictor(rollout,draws).predict(STATE,ACTIONS,120,4)
        self.assertEqual(calls,[4])
        self.assertEqual([d for a,d in received[:4]],[d for a,d in received[4:]])
        self.assertEqual([f.draw_id for f in p.actions['O0']],[f.draw_id for f in p.actions['O1']])
        self.assertEqual([x.team_delay-y.team_delay for x,y in zip(p.actions['O0'],p.actions['O1'])],[3.]*4)

    def test_reordered_action_futures_are_aligned_by_draw_id(self):
        p = prediction([10.,100.,1000.],[9.,99.,999.],reverse=True)
        self.assertEqual(decide(p)['probability_beats']['O1'],1.)

    def test_point_eight_inclusive_and_below_gate(self):
        self.assertEqual(decide(prediction([10.]*10,[9.]*8+[11.]*2))['selected'],'O1')
        self.assertEqual(decide(prediction([10.]*10,[9.]*7+[11.]*3))['selected'],'O0')

    def test_rare_large_loss_and_ties_do_not_switch(self):
        self.assertEqual(decide(prediction([10.]*10,[9.]*9+[100.]))['selected'],'O0')
        self.assertEqual(decide(prediction([10.]*10,[10.]*10))['selected'],'O0')

    def test_exact_strict_improvement_without_handset_tolerance(self):
        for other,expected in ((10.,'O0'),(10.-1e-12,'O1'),(11.,'O0')):
            p = Prediction({'O0':[Outcome('full-state',10.,False)],
                            'O1':[Outcome('full-state',other,False)]},True,'SUMO-ORACLE')
            self.assertEqual(decide(p)['selected'],expected)
        with self.assertRaises(ValueError):
            SumoOraclePredictor(lambda *a:toy(ACTIONS[0])).predict(STATE,ACTIONS,120,2)

    def test_missing_duplicate_nonfinite_futures_fail_closed(self):
        for futures in ([],[Outcome('0',9.,False),Outcome('0',8.,False)],
                        [Outcome('different-draw',9.,False)],[Outcome('0',float('nan'),False)]):
            p = Prediction({'O0':[Outcome('0',10.,False)],'O1':futures},False,'bad')
            with self.assertRaises(ValueError):
                decide(p)

    def test_unavailable_model_never_calls_sumo(self):
        with self.assertRaises(BackendUnavailable):
            WMPhysPredictor().predict(STATE,ACTIONS,120,8)

    def test_observable_whitelist_and_hdv_command_rejected(self):
        with self.assertRaises(ValueError):
            ObservableState(300,tuple(dict(r,driver_tau=2.) for r in ROWS),'bad')
        with self.assertRaises(ValueError):
            SumoOraclePredictor(lambda *a:None).predict(STATE,[dict(id='bad',members=['r'])],120,1)

    def test_option_pruning_no_new_behaviour_cutoffs(self):
        actions = candidates(STATE)
        self.assertEqual([a['id'] for a in actions],['O0','O1','O2','O4'])
        empty = ObservableState(300,(), 'empty')
        self.assertEqual(candidates(empty),[dict(id='O0',members=[])])

    def test_frozen_free_proposal_uses_identical_predictor_path(self):
        proposal = lambda s:dict(id='FREE',members=['a','b'],window=20,
                                targets={'a':dict(speed=15.,tau=2.),'b':dict(speed=16.,tau=2.)})
        seen = []
        def run(s,a,h):
            seen.append(deepcopy(a))
            result = toy(a)
            if a['id'] == 'FREE':
                result['team_delay'] -= 1.
            return result
        event,_ = Planner(SumoOraclePredictor(run),learned=proposal).step(STATE)
        self.assertEqual(event['executed_action'],proposal(STATE))
        self.assertEqual(seen[-1],proposal(STATE))

    def test_receding_cadence_and_persistent_group_action_log(self):
        logs = []
        planner = Planner(SumoOraclePredictor(lambda s,a,h:toy(a)),log=logs.append)
        e,_ = planner.step(STATE)
        self.assertEqual(e['group'],['a','b'])
        planner.step(ObservableState(360,ROWS,'next-actual-prefix'))
        self.assertEqual(len(logs),2)
        for t in (360,380):
            with self.assertRaises(ValueError):
                planner.step(ObservableState(t,ROWS,'duplicate-or-off-clock'))

    def test_credit_is_superadditivity_not_group_saving(self):
        p = Prediction({a:[Outcome('paired',v,False)] for a,v in
                        [('O0',10.),('O1',7.),('solo0',9.5),('solo1',9.)]},True,'toy')
        self.assertEqual(cooperation_credit(p,'O1',['solo0','solo1']),{'paired':1.5})

    def test_budget_counts_prefixes_and_both_j_clocks(self):
        b = budget()
        self.assertEqual(len(TIMES),12)
        self.assertEqual(sum(TIMES),7560)
        self.assertEqual(b,dict(episodes=28800,lookahead=648000,member_alone=25920,mechanics=2040))
        self.assertEqual(sum(b.values()),704760)
        self.assertLessEqual(sum(b.values()),CAP)
        self.assertEqual(len({w['seed'] for w in worlds()}),12)

    def test_approval_requires_exact_task_and_cap(self):
        for text in ('','S-COOPPLAN awaiting approval 800,000',
                     'S-COOPPLAN BUDGET APPROVED pending 800,000',
                     'S-COOPBANDIT BUDGET APPROVED 1,600,000'):
            with self.assertRaises(PermissionError):
                approval(text)
        self.assertIn('APPROVED',approval('MAIN S-COOPPLAN BUDGET APPROVED cap 800,000'))

    def test_full_rerun_replays_executed_history_and_gates_before_action(self):
        from . import plant
        history = [dict(time_s=300,executed_action=dict(id='O1',members=['a','b'],window=30),initial_rows=list(ROWS))]
        snapshot = dict(time_s=360,rows=list(ROWS),world=worlds()[0],history=history,prefix={'actual-history':True})
        order = []
        class Env:
            now = 300
        class Fake:
            env = Env()
            totals = {'team_delay_vs':0.,'road_delay_vs':0.,'source_wait_vs':0.,'collisions':0.}
            actual = 480
            def fingerprint(self):
                order.append('prefix-gate')
                return snapshot['prefix']
            def close(self):
                order.append('close')
        class Diag:
            delay = {'a':0.,'b':0.}
            def __init__(self,*a): pass
            def breakdown(self,*a): return {'event':False}
        def replay(b,h,until,tape=None):
            order.append(('replay',deepcopy(h),until))
            b.env.now = until
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            class Tape:
                def __enter__(self): return self
                def __exit__(self,*args): pass
            with patch.object(plant,'read',return_value=snapshot), \
                 patch.object(plant,'fresh',return_value=(Fake(),root)), \
                 patch.object(plant,'Diagnostics',Diag),patch.object(plant,'replay',side_effect=replay), \
                 patch.object(plant,'write'),patch.object(plant.gzip,'open',return_value=Tape()), \
                 patch.dict('sys.modules',{'tools.leverprobe.backend':type('Module',(),{'observations':lambda b:list(ROWS)})}):
                result = plant.branch(root,'key',ACTIONS[1],120)
        self.assertEqual(order[0],('replay',history,360))
        self.assertEqual(order[1],'prefix-gate')
        self.assertEqual(order[2][2],480)
        self.assertEqual(order[-1],'close')
        self.assertEqual(result['breakdown'],False)


if __name__ == '__main__':
    with patch('subprocess.Popen',side_effect=AssertionError('CPU gate: SUMO/process execution disabled')):
        unittest.main()
