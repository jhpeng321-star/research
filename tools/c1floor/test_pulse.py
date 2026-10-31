import unittest
from types import SimpleNamespace
from pulse import pulse


class Tests(unittest.TestCase):
    def test_actual_initial_speed_delta_only(self):
        speeds={'h':20.,'other':30.};calls=[]
        def set_previous(vid,speed,accel):
            speeds[vid]=speed;calls.append((vid,speed,accel))
        vehicle=SimpleNamespace(getIDList=lambda:list(speeds),getSpeed=lambda v:speeds[v],
            getAcceleration=lambda v:-.2,getLanePosition=lambda v:10.,setPreviousSpeed=set_previous)
        r=pulse(SimpleNamespace(vehicle=vehicle),'h')
        self.assertAlmostEqual(r['delta_mps'],.01)
        self.assertEqual(speeds['other'],30.)
        self.assertEqual(calls,[('h',20.01,-.2)])
        self.assertFalse(r['persistent_speed_override'])

    def test_noop_api_is_rejected(self):
        vehicle=SimpleNamespace(getIDList=lambda:['h'],getSpeed=lambda v:20.,
            getAcceleration=lambda v:0.,getLanePosition=lambda v:10.,setPreviousSpeed=lambda *args:None)
        with self.assertRaises(AssertionError):pulse(SimpleNamespace(vehicle=vehicle),'h')


if __name__=='__main__':unittest.main()
