"""Independent post-safety parity against the actual CommandOwnership writer."""
import unittest
from types import SimpleNamespace
import torch
from .command_chain import ownership_target,SlowDown


class CommandChainTests(unittest.TestCase):
    def test_actual_ownership_writer(self):
        from traffic_shepherd.ownership import CommandOwnership
        calls=[]
        api=SimpleNamespace(vehicle=SimpleNamespace(
            getSpeedMode=lambda v:27,
            slowDown=lambda *a:calls.append(a)))
        cfg=SimpleNamespace(control_dt_s=1.,acceleration_min_mps2=-3.,
            acceleration_max_mps2=1.5,sensor_range_m=100.)
        for v,allowed,a in [(23.209720611572266,31.29,2.5999999046325684),
                            (21.486669540405273,31.29,2.5607497692108154),
                            (27.99481201171875,31.29,2.5999999046325684),
                            (26.094942092895508,31.29,2.565342664718628),
                            (31.1,31.29,2.6),(0.2,31.29,-3.),(24.,22.22,2.6),
                            (31.29,31.29,2.6),(2.,31.29,-1.5)]:
            with self.subTest(v=v,allowed=allowed,a=a):
                f=SimpleNamespace(vehicle_id='v',timestamp_s=0.,speed_mps=v,
                    allowed_speed_mps=allowed,front_coverage_m=100.,rear_coverage_m=100.,front=None,rear=None)
                owner=CommandOwnership(api,['v'],SimpleNamespace(config=cfg),cav_speed_mode=27)
                owner.wall_accel_max_mps2=3.
                actual=owner.step({'v':f},now_s=0.,live_ids=['v'],wall_commands={'v':a})[0]
                issued,target,ok=ownership_target(*[torch.tensor([x],dtype=torch.float64) for x in (v,a,allowed)])
                self.assertTrue(bool(ok[0]))
                self.assertEqual(actual.issued_acceleration_mps2,issued.item())
                self.assertEqual(actual.issued_target_speed_mps,target.item())
                self.assertEqual(calls[-1],('v',target.item(),1.))

    @staticmethod
    def state(speed,present=True):
        return dict(v_mps=torch.tensor([[speed]],dtype=torch.float64),
                    present=torch.tensor([[present]]),is_cav=torch.tensor([[True]]),lane=torch.tensor([[0]]))

    def test_timeline_and_speed_cap(self):
        chain=SlowDown();state=self.state(30.9)
        a=chain.apply(state,torch.tensor([[2.6]]),0,torch.tensor([[31.29]],dtype=torch.float64),torch.zeros(1,1))
        self.assertEqual(a.item(),(31.095-30.9)/.5)
        state=self.state(31.095)
        b=chain.apply(state,torch.tensor([[-3.]]),1,torch.tensor([[31.29]]),torch.zeros(1,1))
        self.assertEqual(b.item(),(31.29-31.095)/.5)

    def test_speedmode_ceiling_after_timeline(self):
        chain=SlowDown();state=self.state(20.)
        a=chain.apply(state,torch.tensor([[2.6]]),0,torch.tensor([[31.29]]),torch.zeros(1,1))
        # The physical ceiling binds at .5 on the first step; slowDown catches
        # up on the second, but the separate physical ceiling still wins.
        self.assertAlmostEqual(a.item(),2.6,places=6)
        second=chain.apply(self.state(20.25),None,1,torch.tensor([[31.29]]),torch.zeros(1,1))
        self.assertGreater(second.item(),a.item())
        self.assertEqual(min(second.item(),.5),.5)

    def test_odd_entry_native_and_next_decision(self):
        chain=SlowDown();chain.apply(self.state(10.,False),torch.zeros(1,1),0,torch.tensor([[31.29]]),torch.ones(1,1))
        self.assertEqual(chain.apply(self.state(10.),None,1,torch.tensor([[31.29]]),torch.ones(1,1)).item(),1.)
        self.assertEqual(chain.apply(self.state(10.),torch.zeros(1,1),2,torch.tensor([[31.29]]),torch.ones(1,1)).item(),0.)

    def test_continuation_requires_decision(self):
        with self.assertRaises(ValueError):
            SlowDown().apply(self.state(10.),None,1,torch.tensor([[31.29]]),torch.zeros(1,1))


if __name__=='__main__':unittest.main()
