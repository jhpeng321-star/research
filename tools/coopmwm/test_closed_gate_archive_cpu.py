import json
from pathlib import Path
import tempfile
import time
import unittest

from .closed_gate_archive import GATES, kind, select, terminal


class ClosedGateArchiveTests(unittest.TestCase):
    def test_scope_excludes_live_gate_and_bank(self):
        home = Path('/home/ubuntu')
        for root in ('coopmwm20261104_gate_coop6', 'coopmwm20261102_warm1'):
            self.assertIsNone(kind(home, home/root/'branches/psi_u000_B_sv_fixture/SNAPSHOT.pt'))
        self.assertEqual(kind(home, home/GATES[0]/'branches/psi_u000_B_sv_fixture/SNAPSHOT.pt'), (0, 'B'))

    def test_scope_excludes_scenario_checkpoint_and_summaries(self):
        root = Path('/home/ubuntu')/GATES[0]/'branches/psi_u000_B_sv_fixture'
        for name in ('cav.sumocfg', 'input.xml', 'checkpoint.pt', 'SNAPSHOT.sha256',
                     'all.graded.json', 'all.spec.json', 'vehicle_params.json'):
            self.assertIsNone(kind('/home/ubuntu', root/name))

    def test_no_claim_only_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/GATES[0]
            root.mkdir()
            (root/'EXECUTE_STAGE_RESULT.json').write_text('{}')
            self.assertFalse(terminal(root, time.time()+4000))

    def fixture(self, home):
        root = home/GATES[0]
        branch = root/'branches/psi_u000_B_sv_fixture'
        branch.mkdir(parents=True)
        for name in ('COLLECT_STAGE_RESULT.json', 'COOPFIX_GATE.json'):
            (root/name).write_text(json.dumps(dict(complete=True)))
        p = branch/'all.result.json'
        p.write_text('{}')
        return root, p

    def test_age_and_each_reference_protection(self):
        with tempfile.TemporaryDirectory() as d:
            home = Path(d).resolve()
            root, p = self.fixture(home)
            now = time.time()
            self.assertEqual(select(home, set(), set(), set(), set(), now), {})
            now += 4000
            self.assertEqual(set(select(home, set(), set(), set(), set(), now)), {str(p)})
            for opened, files, trees, refs in (({str(p)}, set(), set(), set()),
                (set(), {str(p)}, set(), set()), (set(), set(), {str(p.parent)}, set()),
                (set(), set(), set(), {str(p.parent)}), (set(), set(), set(), {str(p)})):
                self.assertEqual(select(home, opened, files, trees, refs, now), {})
            self.assertEqual(select(home, set(), set(), set(), {str(root/'branches')}, now), {})
            self.assertTrue(select(home, set(), set(), set(), {str(root)}, now))

    def test_void_is_terminal_only_when_aged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/GATES[1]
            root.mkdir()
            (root/'VOID_ENVIRONMENT_FAILURE.json').write_text('{}')
            self.assertFalse(terminal(root, time.time()))
            self.assertTrue(terminal(root, time.time()+4000))


if __name__ == '__main__':
    unittest.main()
