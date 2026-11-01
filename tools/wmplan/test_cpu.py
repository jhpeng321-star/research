"""Scientific contract tests: paired choices, complete quality, hard budgets."""
from copy import deepcopy
import ast
from contextlib import nullcontext
import io
from pathlib import Path
import types
import unittest
from unittest.mock import patch

from tools.coopbandit.protocol import digest
from tools.coopplan.predictor import Prediction, Outcome
from tools.wmplan.design import (panel,budget,PANEL_SHA,CREDIT,model_choice,decision,admit)


class Contracts(unittest.TestCase):
    def test_remote_operations_compile_before_dispatch(self):
        from tools.wmplan import ops
        scripts = 0
        for name in ('ops.py','mirror.py','bridge.py','watch.py'):
            tree = ast.parse(Path(ops.__file__).with_name(name).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if (isinstance(node,ast.Call) and
                        ((isinstance(node.func,ast.Name) and node.func.id=='remote') or
                         (isinstance(node.func,ast.Attribute) and node.func.attr=='remote')) and
                        node.args and isinstance(node.args[0],ast.JoinedStr)):
                    # Replace interpolated data by a syntactically valid literal;
                    # this verifies generated Python, including remote indentation.
                    text = ''.join(v.value if isinstance(v,ast.Constant) else 'None' for v in node.args[0].values)
                    compile(text,'<remote-contract>','exec')
                    scripts += 1
        self.assertGreaterEqual(scripts,7)

    def test_shared_panel_and_budget(self):
        self.assertEqual(digest(panel()),PANEL_SHA)
        self.assertEqual(len(panel()),24)
        b = budget()
        self.assertEqual(b['episodes']+b['all_group_credit'],b['total'])
        self.assertLessEqual(b['total'],b['cap'])
        self.assertEqual(b['new_baselines'],0)

    def test_expected_and_strict_can_disagree_without_future_dropping(self):
        p = Prediction(dict(O0=tuple(Outcome(str(k),100,None) for k in range(8)),
                            O1=tuple(Outcome(str(k),v,None) for k,v in enumerate([1]*5+[101]*3))),
                       False,'WM-PHYS')
        self.assertEqual(decision(p,'expected')['selected'],'O1')
        self.assertEqual(decision(p,'strict')['selected'],'O0')
        q = Prediction(dict(p.actions,O1=p.actions['O1'][:-1]),False,'WM-PHYS')
        with self.assertRaises(ValueError):
            decision(q,'expected')

    def test_expected_never_acts_on_zero_or_negative_saving(self):
        p = Prediction({a:tuple(Outcome(str(k),v,None) for k in range(8))
                        for a,v in [('O0',100),('O1',100),('O2',101)]},False,'WM-PHYS')
        self.assertEqual(decision(p,'expected')['selected'],'O0')

    def test_complete_quartet_reserved_and_late_refusal_explicit(self):
        event = dict(time_s=300,executed_action=dict(id='O1',members=['a','b']),group=['a','b'])
        self.assertEqual(admit(event,0),1680)
        self.assertFalse(event['budget_pruned'])
        later = dict(time_s=360,executed_action=dict(id='O1',members=['a','b']),group=['a','b'])
        self.assertEqual(admit(later,1680),0)
        self.assertTrue(later['budget_pruned'])
        self.assertEqual(later['executed_action']['id'],'O0')
        self.assertEqual(later['proposed_action']['id'],'O1')
        latest = dict(time_s=600,executed_action=dict(id='O4',members=['a','b']),group=['a','b'])
        self.assertEqual(admit(latest,0),2880)
        self.assertLessEqual(2880,CREDIT)

    def test_v5_requires_both_noninferiority_conditions_on_same_complete_states(self):
        rows = [dict(state_id=str(i),world_id=i,horizon_s=300,exact_D=[0,10],
                     oracle_saving=10,exact_saving=5,agreement=i<6,gated_agreement=False,
                     regret=5,sign_count=1,sign_matches=1,within_state_spearman=1.) for i in range(12)]
        self.assertEqual(model_choice(rows,rows)['selected'],'FULL-v5')
        v5 = deepcopy(rows)
        v5[0]['exact_saving'] = 0
        self.assertEqual(model_choice(rows,v5)['selected'],'FULL-v4')
        v5 = deepcopy(rows)
        v5[0]['agreement'] = False
        self.assertEqual(model_choice(rows,v5)['selected'],'FULL-v4')
        with self.assertRaises(ValueError):
            model_choice(rows,v5[:-1])
        v5[0]['exact_D'] = [0,11]
        with self.assertRaises(ValueError):
            model_choice(rows,v5)

    def test_live_state_machine_reuses_baseline_and_finishes_twins_before_move(self):
        from tools.wmplan import live
        from tools.coopplan import plant
        from tools.wmplan import predict
        writes, credits_done = {}, []
        b = types.SimpleNamespace(env=types.SimpleNamespace(now=300),
                                  totals={'team_delay_vs':0.},actual=300)
        b.fingerprint = lambda:dict(time=b.env.now)
        b.close = lambda:None
        observations = types.ModuleType('tools.leverprobe.backend')
        observations.observations = lambda env:[]
        def step(state):
            a = dict(id='O1',members=['a','b']) if state.time_s==300 else dict(id='O0',members=[])
            return dict(time_s=state.time_s,executed_action=a,group=a['members'])
        class Pool:
            def __init__(self,*a,**kw):pass
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def submit(self,fn,root,key,action):
                credits_done.append(action['id'])
                value = dict(O0=100,O1=60,O1_solo0=90,O1_solo1=80)[action['id']]
                return types.SimpleNamespace(result=lambda:value)
        def tick(backend,event,tape):
            if backend.env.now==300:
                self.assertEqual(credits_done,['O0','O1','O1_solo0','O1_solo1'])
                self.assertIn('synthetic/reservations/s984000_expected_t300.json',writes)
            backend.env.now+=1;backend.actual+=1
            backend.totals['team_delay_vs']+=1
            return dict(requests=0,overrides=0)
        fake = types.SimpleNamespace(fresh=lambda *a:(b,Path('synthetic/job')))
        with patch.dict('sys.modules',{'tools.leverprobe.backend':observations}), \
             patch.object(live,'setup',return_value={'baseline_root':'baseline'}), \
             patch.object(live,'configure',return_value=fake), \
             patch.object(live,'read',return_value={'warmup':{'time':300}}), \
             patch.object(live,'write',side_effect=lambda p,v:writes.update({p.as_posix():deepcopy(v)})), \
             patch.object(live,'ProcessPoolExecutor',Pool), \
             patch.object(predict,'OnlinePlanner',return_value=types.SimpleNamespace(step=step)), \
             patch.object(live.gzip,'open',side_effect=lambda *a,**k:nullcontext(io.StringIO())), \
             patch.object(plant,'tick',side_effect=tick):
            live.episode(Path('synthetic'),panel()[0],'expected')
        result = writes['synthetic/job/RESULT.json']
        self.assertEqual(len(result['history']),12)
        self.assertEqual(result['credits'][0]['C_G'],10)
        self.assertEqual(result['clocks']['600']['team_delay_vs'],600)
        self.assertEqual(result['clocks']['900']['team_delay_vs'],900)
        self.assertEqual(result['credit_sim_s'],1680)


if __name__ == '__main__':
    unittest.main()
