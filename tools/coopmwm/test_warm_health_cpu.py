"""Registered health pauses and scoped release of the original transport STOP."""
import tempfile
import unittest
from pathlib import Path
from .storage import write,read
from .protocol import sha
from . import warm_health as h


class HealthCPU(unittest.TestCase):
    def test_matching_release_is_sha_and_authority_bound_and_new_stop_remains_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp)
            write(r/'WARM_GUIDED_STOP_V3.json',dict(error="AssertionError('R count/regime/group-size/duration gate')"))
            with self.assertRaises(ValueError):h.assert_running(r,'R')
            (r/'AUTHORITY.md').write_text('matching-amendment')
            write(r/'MATCH_GUIDED_RELEASE.json',dict(original_stop_sha256=sha(r/'WARM_GUIDED_STOP_V3.json'),amendment_sha256='matching-amendment'))
            h.assert_running(r,'R')
            (r/'AUTHORITY.md').write_text('unrelated')
            with self.assertRaises(ValueError):h.assert_running(r,'R')
            (r/'AUTHORITY.md').write_text('matching-amendment')
            write(r/'WARM_GUIDED_STOP_V4.json',dict(error='fresh fault'))
            with self.assertRaises(ValueError):h.assert_running(r,'R')

    def test_transport_release_only_exact_registered_original_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);write(r/'STOP.json',dict(error='Connection closed by 1.15.42.207 port 22\n',source='owned warm bridge; no automatic retry'))
            self.assertTrue(h.active_stop(r));(r/'AUTHORITY.md').write_text('registered-amendment')
            write(r/'WARM_TRANSPORT_RELEASE.json',dict(original_stop_sha256=sha(r/'STOP.json'),amendment_sha256='registered-amendment'))
            self.assertFalse(h.active_stop(r))
            (r/'AUTHORITY.md').write_text('other');self.assertTrue(h.active_stop(r))

    def test_safety_pause_stops_B_R_but_A_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);reference=dict(by_world={'1':dict(collisions=0.,teleports=0.)},unmatched_zero_event_benchmark=True,source='fixture')
            with self.assertRaises(h.HealthPause):h.live_safety(r,'B',dict(collisions=1.,teleports=0.),1,reference)
            self.assertEqual(read(r/'PAUSE_SAFETY.json')['evidence']['world'],1)
            with self.assertRaises(h.HealthPause):h.assert_running(r,'R')
            h.assert_running(r,'A')

    def test_harmful_plan_pause_after_30_and_negative_upper_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);records=[dict(executed=True)]*29
            h.before_update(r,'B',2,dict(executed=10),records,dict(beta=1.,ci95=[-2.,-1.]),10)
            with self.assertRaises(h.HealthPause):h.before_update(r,'B',3,dict(executed=10),records+[dict(executed=True)],dict(beta=1.,ci95=[-2.,-1.]),10)
            self.assertTrue((r/'PAUSE_HARMFUL_PLANS.json').exists())

    def test_two_update_starvation_and_early_zero_trust(self):
        for zero in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                r=Path(tmp)
                if not zero:write(r/'B_UPDATE_001.json',dict(proposal_routing=dict(executed=9)))
                with self.assertRaises(h.HealthPause):h.before_update(r,'B',2,dict(executed=9),[],dict(beta=0. if zero else 1.),10)
                self.assertTrue((r/'PAUSE_STARVATION.json').exists())

    def test_two_consecutive_train_checks_worse_than_A(self):
        report=dict(comparisons=dict(B_minus_A=dict(metrics={'900_team_delay_vs':dict(ci95=[1.,2.])})))
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);h.after_check(r,'B',4,report);write(r/'AB_CHECK_004.json',report)
            with self.assertRaises(h.HealthPause):h.after_check(r,'B',8,report)
            self.assertTrue((r/'PAUSE_LEARNING.json').exists())

    def test_significantly_higher_paired_braking_pauses(self):
        def eps(n):return [dict(world=dict(seed=i),metrics={'900':dict(hard_brake=n)}) for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(h.HealthPause):h.braking(Path(tmp),'B',1,eps(10),eps(0))


if __name__=='__main__':unittest.main()
