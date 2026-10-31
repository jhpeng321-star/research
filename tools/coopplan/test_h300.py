"""CPU checks for the horizon amendment and fair complete-twin reservations."""
from collections import Counter
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from .protocol import h300_worlds,h300_budget,guard
from .predictor import ObservableState,SumoOraclePredictor
from .planner import Planner
from .quota import reserve_actions,outstanding
from .storage import write,charge
from .test_cpu import ROWS


class H300Tests(unittest.TestCase):
    def test_exact_panels_and_cap(self):
        ws=h300_worlds()
        self.assertEqual([w['seed'] for w in ws],list(range(983300,983324)))
        self.assertEqual(Counter(w['cohort'] for w in ws),dict(confirmation18=18,wall6=6))
        self.assertEqual(set(Counter(w['cell'] for w in ws[:18]).values()),{2})
        self.assertTrue(all(w['penetration'] in (.2,.4) and w['demand'] in (2400,2700) for w in ws[18:]))
        self.assertEqual(565140+sum(h300_budget().values()),799860)
        with self.assertRaises(ValueError):guard(dict(ws[0],seed=983324))

    def test_horizon_reaches_predictor_and_availability_survives_pruning(self):
        seen=[]
        def run(s,a,h):
            seen.append(h)
            return dict(team_delay=10. if a['id']=='O0' else 9.,breakdown=False)
        state=ObservableState(300,tuple(ROWS),'s1_t300')
        event,pred=Planner(SumoOraclePredictor(run),horizon=300).step(state)
        self.assertTrue(seen);self.assertEqual(set(seen),{300})
        self.assertEqual(event['horizon_s'],300)
        event,pred=Planner(SumoOraclePredictor(run),horizon=300,admit=lambda s,a:([],{})).step(state)
        self.assertIsNone(pred)
        self.assertGreater(len(event['available_candidates']),1)

    def test_world_quota_cannot_consume_peer_prediction_allowance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'jobs').mkdir()
            write(root/'ALLOCATION.json',dict(cap=19560,world_caps=dict(s1=9780,s2=9780),
                  reserved={f's{i}_{a}':1200 for i in (1,2) for a in ('mappo','planner')}))
            actions=[dict(id=a,members=[] if a=='O0' else ['a','b']) for a in ('O0','O1','O2','O4')]
            with patch('tools.coopplan.storage.verify',return_value=dict(horizon_s=300)):
                state=ObservableState(300,tuple(ROWS),'s1_t300')
                admitted,_=reserve_actions(root,state,actions)
                self.assertEqual(outstanding(root)['s1_t300_CG'],1200)
                for a in admitted:charge(root,state.replay_key+'_'+a['id'],600)
                for i in range(2):charge(root,state.replay_key+f'_O1_solo{i}',600)
                state=ObservableState(360,tuple(ROWS),'s1_t360')
                reserve_actions(root,state,actions)
                # Third batch cannot steal s2's untouched full prediction allowance.
                state=ObservableState(420,tuple(ROWS),'s1_t420')
                admitted,receipt=reserve_actions(root,state,actions)
                self.assertEqual(admitted,[])
                self.assertEqual(receipt['reason'],'budget_pruned')
                state=ObservableState(300,tuple(ROWS),'s2_t300')
                admitted,receipt=reserve_actions(root,state,actions)
                self.assertEqual(len(admitted),len(actions))
                self.assertEqual(receipt['available_sim_s'],7380)


if __name__=='__main__':unittest.main()
