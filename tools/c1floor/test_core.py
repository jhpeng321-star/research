import unittest
from contextlib import nullcontext
from unittest.mock import patch
from core import rng_only, seed_guard, admit, compare_tapes, audit


class Mechanics(unittest.TestCase):
    def test_rng_only_keeps_physical_and_route_bytes(self):
        before = b'<snapshot><rngState id="default" state="x"/><vehicle rngIndex="3" speed="1.2345678901234567"/><delay loaderTime="300"/></snapshot>'
        after,n = rng_only(before)
        self.assertEqual(n, 1)
        self.assertEqual(after, before.replace(b'<rngState id="default" state="x"/>', b''))

    def test_all_rng_streams_are_removed(self):
        after,n = rng_only(b'<x><rngState id="1"/><rngState id="2">abc</rngState></x>')
        self.assertEqual((after,n), (b'<x></x>',2))

    def test_missing_rng_refused(self):
        with self.assertRaises(ValueError):
            rng_only(b'<snapshot/>')

    def test_forbidden_seeds_and_types(self):
        for seed in (964000,964199,962123,977123,True,969050.,'969050'):
            with self.assertRaises(ValueError):
                seed_guard(seed)
        for seed in (969400,969423,969050,969097):
            seed_guard(seed)

    def test_budget_counts_all_prefixes_and_canary(self):
        old=[dict(id='canary',sim_s=1500)]
        for i in range(23):
            job=dict(id=str(i),charge=1200)
            admit(old,job)
            old.append(dict(id=job['id'],sim_s=job['charge']))
        self.assertEqual(sum(r['sim_s'] for r in old),29100)
        with self.assertRaises(ValueError):admit(old,dict(id='extra',charge=1200))

    def test_duplicate_reservation_and_corruption_refused(self):
        old=[dict(id='a',sim_s=1500)]
        with self.assertRaises(ValueError):admit(old,dict(id='a',charge=1500))
        with self.assertRaises(ValueError):admit(old*2,dict(id='b',charge=1200))

    def test_zero_negative_or_unbounded_reservation_refused(self):
        for charge in (0,-1,30001):
            with self.assertRaises(ValueError):admit([],dict(id='a',charge=charge))

    def test_manual_zero_step_repair_retains_charge(self):
        old=[dict(id='zero_step_failure',sim_s=1500)]
        for i in range(23):
            job=dict(id=str(i),charge=1500 if i==0 else 1200)
            admit(old,job)
            old.append(dict(id=job['id'],sim_s=job['charge']))
        self.assertEqual(sum(r['sim_s'] for r in old),29400)
        with self.assertRaises(ValueError):admit(old,dict(id='extra_world',charge=1200))

    def test_bitwise_canary_rejects_signed_zero_and_dtype_changes(self):
        import numpy as np
        a={k:np.array([300.],dtype=np.float32) for k in ('t','vid','s','v','a','lane')}
        a['s']=np.array([0.],dtype=np.float32)
        b={k:v.copy() for k,v in a.items()};b['s'][0]=-0.
        b['v']=b['v'].astype(np.float64)
        with patch('numpy.load',side_effect=[nullcontext(a),nullcontext(b)]):
            check=compare_tapes('one.npz','two.npz')
        self.assertFalse(check['s']['identical'])
        self.assertFalse(check['v']['identical'])
        self.assertTrue(check['t']['identical'])

    def test_generated_definition_write_only_not_read(self):
        for name in ('vtypes.add.xml','hetdata_truth.csv'):
            p='/root/c1floor20261031/jobs/test/scenarios/'+name
            audit('open',(p,'w',0))
            for mode in ('r','r+','a+'):
                with self.assertRaises(PermissionError):audit('open',(p,mode,0))
        with self.assertRaises(PermissionError):
            audit('open',('/other/vtypes.add.xml','w',0))


if __name__ == '__main__':
    unittest.main()
