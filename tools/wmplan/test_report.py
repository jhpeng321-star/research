"""Independent analysis rejects incomplete panels and altered decisions/prefixes."""
from copy import deepcopy
import unittest

from tools.coopbandit.protocol import digest
from tools.coopplan.predictor import Prediction, Outcome
from tools.coopplan.planner import TIMES
from tools.wmplan.design import ARMS, panel, decision
from tools.wmplan.report import build, ci


def fixture():
    r = dict(complete=True,stopped=False,incomplete=[],episodes=48,actual=57600,charged=57600,
             binding={'model':{'selected':'FULL-v4'}},charges=[],closed=[],forecasts={},
             observations={},results={},baselines={},reservations={})
    for w in panel():
        seed = w['seed']
        metrics = dict(team_delay_vs=100,road_delay_vs=60,source_wait_vs=40,
                       collisions=0,teleports=0,ttc_below_3_vs=2,hard_brake=1)
        base = dict(world=w,actual_sim_s=1200,clocks={k:dict(metrics) for k in ('600','900')},
                    final_prefix={'bits':'unchanged'})
        r['baselines'][str(seed)] = base
        for arm in ARMS:
            ident = f's{seed}_{arm}'
            history = []
            for t in TIMES:
                key = f'{seed}_{t}'
                fs = {'O0':[dict(draw_id=f'{key}_{k}',team_delay=10,breakdown=None) for k in range(8)]}
                p = Prediction({a:tuple(Outcome(**v) for v in values) for a,values in fs.items()},False,'WM-PHYS-FULL-v4')
                pid = digest({a:[vars(v) for v in values] for a,values in p.actions.items()})
                r['forecasts'][key] = dict(backend='FULL-v4',futures=fs)
                r['observations'][key] = dict(world_id=seed,time_s=t,rows=[])
                action = dict(id='O0',members=[])
                history.append(dict(time_s=t,prediction_id=pid,decision=decision(p,arm),
                                    candidates=[action],proposed_action=action,
                                    executed_action=action,budget_pruned=False))
            r['results'][ident] = dict(world=w,arm=arm,warmup_pass=True,actual_sim_s=1200,
                                     clocks=deepcopy(base['clocks']),final_prefix=base['final_prefix'],
                                     history=history,credits=[],credit_sim_s=0,requests=0,overrides=0,
                                     all_O0_bitwise_pass=True)
            r['charges'].append(dict(id=ident,sim_s=1200,kind='paired',seed=seed,arm=arm))
            r['closed'].append(dict(id=ident,actual_sim_s=1200,charged_sim_s=1200))
    return r


class Analysis(unittest.TestCase):
    def test_full_zero_effect_panel_and_budget(self):
        summary,worlds,credits = build(fixture())
        self.assertEqual(len(worlds),48)
        self.assertEqual(credits,[])
        for arm in ARMS:
            self.assertEqual(summary['arms'][arm]['savings']['600']['team_delay_vs']['ci95'],[0.,0.])
            self.assertEqual(summary['arms'][arm]['actions'],{'O0':288})

    def test_rejects_forecast_decision_and_bitwise_mutations(self):
        r = fixture()
        r['results']['s984000_expected']['history'][0]['decision']['selected'] = 'O1'
        with self.assertRaisesRegex(ValueError,'decision'):
            build(r)
        r = fixture()
        r['results']['s984000_expected']['final_prefix'] = {'bits':'changed'}
        with self.assertRaisesRegex(ValueError,'bitwise'):
            build(r)

    def test_rejects_unclosed_job_and_partial_bootstrap(self):
        r = fixture();r['closed'].pop()
        with self.assertRaisesRegex(ValueError,'closed'):
            build(r)
        with self.assertRaises(ValueError):
            ci([0]*23)


if __name__ == '__main__':
    unittest.main()
