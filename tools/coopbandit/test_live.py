"""Live budget and event-isolation semantics; no simulator or model fitting."""
import unittest
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace,ModuleType
from unittest.mock import patch
from .live import allocation,frozen_requests,completed_credit
from . import live,learn
from .storage import read,write
from .protocol import dev_worlds


class LiveTests(unittest.TestCase):
    def test_dev_budget_is_disjoint_and_bounded(self):
        jobs=allocation()
        self.assertEqual(sum(v['sim_s'] for v in jobs.values()),99360)
        self.assertEqual(sum('_MAPPO' in k for k in jobs),12)
        self.assertEqual(sum('_LEARNED' in k for k in jobs),12)
        self.assertTrue(all('9810' in k for k in jobs))

    def test_event_solo_keeps_joint_targets_and_later_factual_actions(self):
        event=dict(time_s=360,plan=dict(members=['a','b'],window=30))
        actions=[dict(vid='a',target_mps=8.),dict(vid='b',target_mps=8.),dict(vid='c',target_mps=12.)]
        rows=[dict(vid='a',v_mps=10.),dict(vid='b',v_mps=6.),dict(vid='c',v_mps=11.)]
        reference,_=frozen_requests(actions,rows,event,370,None)
        solo,_=frozen_requests(actions,rows,event,370,'a')
        self.assertEqual(set(reference),{'c'})
        self.assertEqual(set(solo),{'a','c'})
        self.assertEqual(solo['a']['target_mps'],8.)
        self.assertEqual(solo['a']['requested_raw'],-.4)
        later,_=frozen_requests(actions,rows,event,390,None)
        self.assertEqual(set(later),{'a','b','c'})
        self.assertEqual(later['b']['requested_raw'],.4)

    def test_credit_preserves_worst_case_reservation_and_blocks_bad_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'jobs'/'event';path.mkdir(parents=True)
            event=dict(time_s=300,plan=dict(members=['a','b']))
            write(path/'RESULT.json',dict(actual_sim_s=900,event=event))
            write(path/'CLOSE.json',dict(actual_sim_s=900,charged_sim_s=1140))
            self.assertEqual(completed_credit(root,'event',event)['actual_sim_s'],900)
            with self.assertRaisesRegex(ValueError,'mismatch'):
                completed_credit(root,'event',dict(event,time_s=360))

    def test_full_episode_clocks_decisions_and_paired_bitwise_gate_without_sumo(self):
        """Exercise actual episode orchestration with an in-memory fake plant."""
        class Plant:
            def __init__(self):
                self.env=SimpleNamespace(now=300);self.actual=300;self.closed=False
                self.totals=dict(team_delay_vs=0.,road_delay_vs=0.,source_wait_vs=0.,
                                 collisions=0.,teleports=0.,ttc_below_3_vs=0.,hard_brake=0.)
            def fingerprint(self): return dict(time=self.env.now)
            def close(self): self.closed=True
        def tick(b,**kwargs):
            b.env.now+=1;b.actual+=1
            b.totals['team_delay_vs']+=2;b.totals['road_delay_vs']+=2
            return dict(requests=0,overrides=0)
        probe=ModuleType('tools.leverprobe.backend');probe.observations=lambda env:[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);plants=[]
            def fresh(ident,*args):
                p=root/'jobs'/ident;p.mkdir(parents=True)
                b=Plant();plants.append(b);return b,p
            with patch.object(live,'ROOT',root),patch.object(live,'POLICY',dict(radius=1.)), \
                 patch.dict('sys.modules',{'tools.leverprobe.backend':probe}), \
                 patch.object(live.runner,'completed',return_value=None), \
                 patch.object(live.runner,'fresh',side_effect=fresh), \
                 patch('tools.coopbandit.backend.tick',side_effect=tick), \
                 patch.object(learn,'decide',return_value=('O0',[0.]*4)):
                world=dict(seed=981080)
                live.episode((world,'MAPPO'));live.episode((world,'LEARNED'))
                base=read(root/'jobs/dev_s981080_MAPPO/RESULT.json')
                learned=read(root/'jobs/dev_s981080_LEARNED/RESULT.json')
                self.assertEqual(base['clocks'],learned['clocks'])
                self.assertEqual([d['time_s'] for d in learned['decisions']],list(range(300,1200,60)))
                self.assertEqual(learned['metrics']['600']['team_delay_vs'],1200)
                self.assertEqual(learned['metrics']['900']['team_delay_vs'],1800)
                self.assertTrue(all(b.closed for b in plants))
                # Corrupt the reference before a distinct fresh learned attempt.
                base['clocks']['300']['prefix']['time']=301
                with patch.object(live,'read',return_value=base):
                    with self.assertRaisesRegex(ValueError,'bitwise trunk'):
                        # Fresh uses the already-existing path; provide another fake path.
                        p=root/'bad';p.mkdir()
                        with patch.object(live.runner,'fresh',return_value=(Plant(),p)):
                            live.episode((world,'LEARNED'))

    def test_live_report_reconciles_credit_bootstrap_and_unspent_reservations(self):
        from .live_report import report
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);charges=[]
            def result(ident,value,seconds,actual=None,trace=False):
                p=root/'jobs'/ident;p.mkdir(parents=True)
                actual=seconds if actual is None else actual
                write(p/'RESULT.json',dict(value,actual_sim_s=actual))
                write(p/'CLOSE.json',dict(actual_sim_s=actual,charged_sim_s=seconds))
                if trace: (p/'observable.jsonl.gz').write_bytes(b'fixture-not-read-by-detectors')
                charges.append(dict(id=ident,phase='live',sim_s=seconds))
            for i,w in enumerate(dev_worlds()):
                metrics=dict(team_delay_vs=2000.,collisions=0.,teleports=0.,ttc_below_3_vs=0.,hard_brake=0.)
                event=dict(time_s=300,plan=dict(id='O2',members=['a','b']))
                for arm in ('MAPPO','LEARNED'):
                    changed=i==0 and arm=='LEARNED'
                    m=dict(metrics,team_delay_vs=1900.) if changed else metrics
                    episode=dict(metrics={'600':m,'900':m},safety=dict(override_share=0.),
                        clocks={'300':dict(totals=dict(team_delay_vs=0.)),
                                '900':dict(totals=dict(team_delay_vs=m['team_delay_vs']))},
                        decisions=[event] if changed else [])
                    result(f'dev_s{w["seed"]}_{arm}',episode,1200,trace=True)
                for o in ('O1','O2','O3','O4'):
                    result(f'dev_s{w["seed"]}_oracle_{o}',dict(plan=dict(available=True),D=100.),900)
                if i==0:
                    for v,j in (('reference',2000.),('solo0',1980.),('solo1',1970.)):
                        result(f'dev_s{w["seed"]}_credit_{v}',dict(event=event,metrics=dict(team_delay_vs=j)),1140,actual=900)
            write(root/'LIVE_COMPLETE.json',dict(worlds=12))
            (root/'CHARGES.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in charges))
            summary=report(root)
            self.assertEqual(summary['charged_sim_s'],75420)
            self.assertEqual(summary['actual_sim_s'],74700)
            self.assertEqual(summary['measured_event_credits'],1)
            self.assertAlmostEqual(summary['paired']['600']['mean'],100/12)
            import csv
            with (root/'report/group_credit.csv').open() as stream:
                credit=list(csv.DictReader(stream))[0]
            self.assertEqual(float(credit['C_G']),50.)


if __name__=='__main__': unittest.main()
