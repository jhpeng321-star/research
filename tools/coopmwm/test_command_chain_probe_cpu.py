"""Observable-only kinematic inversion, independently generated half-step paths."""
import unittest
from .command_chain_probe import observable_midpoint


class MidpointTests(unittest.TestCase):
    def test_nonconstant_acceleration_semi_implicit_path(self):
        # A genuinely non-linear pair of updates: linear interpolation would
        # give v=11.25, whereas the actual intermediate speed is11.5.
        s0,v0=80.,10.
        vm=v0+.5*3.
        sm=s0+.5*vm
        v1=vm+.5*-1.
        s1=sm+.5*v1
        actual,unresolved=observable_midpoint(
            {'v':dict(s_m=s0,v_mps=v0,lane=0)},
            {'v':dict(s_m=s1,v_mps=v1,lane=0)})
        self.assertEqual(actual['v']['s_m'],sm)
        self.assertEqual(actual['v']['v_mps'],vm)
        self.assertFalse(unresolved)

    def test_unobserved_entry_exit_and_lane_time_not_invented(self):
        a={v:dict(s_m=0.,v_mps=2.,lane=0) for v in ('gone','lane','stable')}
        b={v:dict(s_m=2.,v_mps=2.,lane=int(v=='lane')) for v in ('new','lane','stable')}
        actual,unresolved=observable_midpoint(a,b)
        self.assertEqual(set(actual),{'stable'})
        self.assertEqual(unresolved,dict(gone='roster_change',new='roster_change',lane='lane_change'))

    def test_inconsistent_endpoint_not_silently_clipped(self):
        actual,unresolved=observable_midpoint(
            {'v':dict(s_m=0.,v_mps=10.,lane=0)},
            {'v':dict(s_m=1.,v_mps=10.,lane=0)})
        self.assertFalse(actual)
        self.assertEqual(unresolved,{'v':'negative_reconstructed_speed'})


if __name__=='__main__':unittest.main()
