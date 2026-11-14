import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parent))
import v2v_resource_watch as m


class AdditiveResourceGate(unittest.TestCase):
    def test_exact_six_gib_boundary(self):
        self.assertFalse(m.decision(m.REQUIRED-1,True,False)['ready'])
        self.assertTrue(m.decision(m.REQUIRED,True,False)['ready'])

    def test_storage_owner_must_be_live_and_not_stopped(self):
        self.assertFalse(m.decision(10*1024**3,False,False)['ready'])
        self.assertFalse(m.decision(10*1024**3,True,True)['ready'])


if __name__=='__main__':unittest.main()
