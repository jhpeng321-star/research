"""Scientific boundaries, accounting, group context, uncertainty and leakage tests."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from .protocol import (worlds,dev_worlds,guard,install_seed_scope,group,option,observable,budget,DATA_CAP,FORBIDDEN)


def rows():
    return [dict(vid='a',s_m=-10.,v_mps=12.,lane=0,is_cav=True),
            dict(vid='b',s_m=-15.,v_mps=10.,lane=1,is_cav=True),
            dict(vid='h',s_m=-5.,v_mps=8.,lane=2,is_cav=False)]


class ProtocolTests(unittest.TestCase):
    def test_worlds_and_budget(self):
        self.assertEqual(len(worlds()),80)
        self.assertEqual(len(dev_worlds()),12)
        self.assertFalse({w['seed'] for w in worlds()} & {w['seed'] for w in dev_worlds()})
        self.assertEqual(sum(v for k,v in budget().items() if k!='live'),DATA_CAP)
        self.assertLessEqual(sum(budget().values()),1600000)

    def test_both_guards_exact_only(self):
        install_seed_scope()
        for w in worlds()+dev_worlds():
            guard(w)
        from traffic_shepherd.seeds import require_development_case
        from traffic_shepherd.c4.seeds import check_case
        for fn in (lambda s,d:require_development_case(s,d,'exposed'),check_case):
            for seed,d in ((981000,2700),(981092,1900),(964000,2400),(962000,2400),(977000,2400)):
                # Legacy guards may admit protected eval blocks; task guard must refuse them.
                if seed in (964000,962000,977000):
                    with self.assertRaises(ValueError):
                        guard(dict(seed=seed,demand=d))
                else:
                    with self.assertRaises(ValueError):
                        fn(seed,d)

    def test_group_geometry_and_anonymous_features(self):
        r=rows()
        self.assertEqual(group(r),['a','b'])
        f=observable(r,300,0)
        altered=[dict(x,driver_params=99,hidden_type=9,vid='other'+str(i)) for i,x in enumerate(r)]
        self.assertEqual(f,observable(altered,300,0))
        self.assertFalse(any(k in f for k in ('seed','penetration','demand','hidden_type')))

    def test_joint_context_preserved_for_solos(self):
        from tools.leverprobe.intervention import targets
        r=rows()
        joint=option('O2',r,0)
        solo=option('O2',r,0,members=['a'])
        self.assertEqual(targets(joint,r)['a'],targets(solo,r)['a'])
        self.assertEqual(targets(joint,r)['a'],7.)
        self.assertEqual(joint['pair'],solo['pair'])

    def test_unavailable_options_are_explicit(self):
        self.assertFalse(option('O3',rows(),0)['available'])
        self.assertEqual(option('O2',rows()[:1],0)['members'],[])
        r=[dict(x,lane=1) if x['is_cav'] else x for x in rows()]
        self.assertTrue(option('O3',r,3)['available'])
        self.assertFalse(option('O2',r,0)['available'])

    def test_cv_radius_world_max_and_abstention(self):
        from .learn import radius,choose
        err=np.array([[1,2,3,4],[5,6,7,8],[100,0,0,0]],float)
        self.assertEqual(radius(err,np.array([1,1,2])),100.)
        pred=np.array([[101,102,103,104],[1,2,3,4]],float)
        self.assertEqual(choose(pred,np.array([[1,0,0,0],[1,1,1,1]],bool),100.).tolist(),[1,0])

    def test_locked_precharge_duplicate_and_caps(self):
        from .storage import charge
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            charge(p,'one',900,'data')
            with self.assertRaises(ValueError): charge(p,'one',900,'data')
            with self.assertRaises(ValueError): charge(p,'bad',DATA_CAP,'data')
            with self.assertRaises(ValueError): charge(p,'badlive',100001,'live')
            self.assertEqual(len((p/'CHARGES.jsonl').read_text().splitlines()),1)

    def test_restricted_paths(self):
        from .protocol import checked
        for word in FORBIDDEN.pattern.split('|'):
            with self.assertRaises(ValueError): checked('/tmp/'+word+'/x')

    def test_host_allocation_rejects_unassigned_charge(self):
        from .storage import charge,write
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            write(root/'ALLOCATION.json',dict(attempts={'one':dict(sim_s=900,phase='data')}))
            with self.assertRaises(ValueError): charge(root,'other',900,'data')
            with self.assertRaises(ValueError): charge(root,'one',901,'data')
            charge(root,'one',900,'data')

    def test_resume_never_retries_charged_incomplete_attempt(self):
        from .storage import charge
        from .runner import completed
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.assertIsNone(completed(root,'one'))
            charge(root,'one',900,'data')
            with self.assertRaises(ValueError): completed(root,'one')

    def test_prefix_and_counter_bitpatterns(self):
        from tools.leverprobe.backend import exact
        self.assertNotEqual(exact(np.array([0.])),exact(np.array([-0.])))
        self.assertNotEqual(exact(np.array([1],dtype='f4')),exact(np.array([1],dtype='f8')))

    def test_trace_distinguishes_native_nan_sentinel_from_command(self):
        from .backend import command_record,action_record
        from .protocol import canonical
        native=command_record(float('nan'),False)
        self.assertEqual(json.loads(canonical(native))['issued_acceleration_mps2'],None)
        self.assertEqual(native['controller'],'native_sumo')
        self.assertFalse(native['command_present'])
        self.assertEqual(command_record(0.,True)['controller'],'option')
        with self.assertRaises(ValueError): command_record(float('inf'),False)
        log=action_record(dict(issued_safe=float('nan'),requested_raw=-1.))
        self.assertEqual(json.loads(canonical(log))['issued_safe'],None)
        self.assertEqual(log['native_no_command_fields'],['issued_safe'])
        with self.assertRaises(ValueError): action_record(dict(requested_raw=float('nan')))


if __name__=='__main__':
    unittest.main()
