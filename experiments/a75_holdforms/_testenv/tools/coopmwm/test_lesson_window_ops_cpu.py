"""Check the actual continuation payload without remote execution or physics."""
import unittest
from unittest.mock import patch
from . import lesson_window_ops as ops


class ProbeComplete(Exception):pass


class ContinuationPayloadTests(unittest.TestCase):
    def test_launch_validates_idle_boundary_before_signals_and_adopts_graders(self):
        seen=[]
        def probe(host,code,timeout):
            self.assertEqual(host,'cpuQ')
            compile(code,'<lesson continuation launch>','exec');seen.append(code)
            raise ProbeComplete
        with patch.object(ops,'remote',side_effect=probe):
            with self.assertRaises(ProbeComplete):ops.launch()
        code=seen[0]
        self.assertLess(code.index("'not idle: live physics/optimizer child'"),code.index('os.kill'))
        self.assertLess(code.index("'PRIOR_OWNERS.json'"),code.index('os.kill'))
        self.assertIn("if d['component']=='grade':verified(d)",code)
        self.assertIn("for d in [old]+arms+[rg]:",code)  # only idle R grader is replaced
        self.assertIn("if (c/'DISPATCH.json').exists()",code)


if __name__=='__main__':unittest.main()
