"""Analytical invariants: departure stream, crossing identity, queue conservation."""
from contextlib import contextmanager
import uuid
import unittest
from pathlib import Path

import numpy as np
from labels import crossings, departures, recorded


@contextmanager
def kept_fixture():
    parent=Path('E:/Temp/opencode')
    assert parent.is_dir()
    root=parent/('wmrequal-label-test-'+uuid.uuid4().hex)
    root.mkdir()
    yield root  # Retain fixtures; owner prohibits deletions.


class LabelsTest(unittest.TestCase):
    def test_departure_rng_is_independent_and_repeatable(self):
        a=departures(973010,2400,'mainline')
        self.assertTrue(np.array_equal(a,departures(973010,2400,'mainline')))
        self.assertTrue((np.diff(a)>0).all())
        self.assertFalse(np.array_equal(a,departures(973010,2400,'ramp')))

    def test_crossings_bridge_junction_but_never_absence_or_identity(self):
        pos=np.array([[70.],[70.],[80.]])
        mask=np.ones_like(pos,bool)
        lane=np.array([[2],[-1],[1]])
        ids=np.ones_like(lane)
        got=crossings(pos,mask,lane,lane>=0,ids)
        np.testing.assert_allclose(got,[.544])
        mask[1]=False
        self.assertEqual(len(crossings(pos,mask,lane,lane>=0,ids)),0)
        mask[1]=True
        ids[-1]=2
        self.assertEqual(len(crossings(pos,mask,lane,lane>=0,ids)),0)

    def test_complete_queue_uses_registry_not_present_occupancy(self):
        with kept_fixture() as root:
            ts=np.arange(299.,362.,.5)
            n=len(ts)
            # Registry IDs 0..9 were inserted before t0, only ID9 remains.
            vid=np.full((n,1),9,np.int32)
            mask=np.ones((n,1),np.uint8)
            kin=np.zeros((n,1,7),np.float32)
            kin[:,:,0]=-100
            kin[:,:,1]=1
            kin[:,:,4]=31.29
            ctr=np.zeros((n,2,30),np.float32)
            ctr[:,0,0]=10
            for name,data in dict(time_s=ts,vid=vid,veh_mask=mask,veh_kin=kin,
                                  counter=ctr,is_cav=np.zeros_like(mask)).items():
                np.save(root/(name+'.npy'),data)
            e=dict(seed=973010,demand_vph=2400,warmup_s=300.,dir='.')
            r=recorded(root,e,60)
            due=sum(int((departures(973010,rate,stream)<=300).sum())
                    for rate,stream in ((2400,'mainline'),(400,'ramp')))
            self.assertEqual(r['queue0_total'],due-10)
            self.assertEqual(sum(r['queue0']),due-10)
            self.assertAlmostEqual(r['values']['60']['road'],0.,places=4)
            e['warmup_s']=301.
            unknown=recorded(root,e,60)
            known=recorded(root,e,60,identities=[f'mainline.{i}' for i in range(10)])
            self.assertIsNone(unknown['queue0'])
            self.assertEqual(sum(known['queue0']),known['queue0_total'])
            self.assertEqual(known['values'],unknown['values'])


if __name__=='__main__':
    unittest.main()
