"""Test the scientific one-physics-tick intervention and hard admission limit."""
import unittest
from types import SimpleNamespace
from fallback import PhysicsCounter, candidates


class Tests(unittest.TestCase):
    def test_one_tick_release_and_cap(self):
        calls=[]
        vehicle=SimpleNamespace(getSpeed=lambda vid:10.,setSpeed=lambda *a:calls.append(a))
        api=SimpleNamespace(vehicle=vehicle,simulationStep=lambda:calls.append('step'))
        c=PhysicsCounter(api,1.)
        c.perturb('h');c();c()
        self.assertEqual(calls,[('h',10.01),'step',('h',-1),'step'])
        self.assertEqual(c.effects[0]['release_tick']-c.effects[0]['tick_before'],1)
        with self.assertRaises(RuntimeError):c()
        self.assertEqual(c.ticks,2)

    def test_upstream_hdv_geometry_only(self):
        f=SimpleNamespace(s=[-500,-900,-700,-600,-200],present=[1]*5,
                          is_cav=[0,1,0,0,0],vid_index=list(range(5)))
        e=SimpleNamespace(frame=f,builder=SimpleNamespace(vehicle_ids=['a','cav','b','c','near']))
        self.assertEqual(candidates(e),['b','c'])


if __name__=='__main__':unittest.main()
