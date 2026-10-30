"""Owner05:25 CONT0525 contracts: peer-aware S2 continuation, guard math, bindings."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.candB.pilot_cont0525_launch import launch_source
from tools.candB.pilot_cont0525_supervise import admit, peers_mib

PARENT = Path('/root/autodl-tmp/c4/candB_pilot_run_RETRY0300_v2_20261031/pilot_EXPLORATORY-LEAN_RETRY0300')
GUARD_EXIT = 'ValueError: lean aggregate memory bound including peers exceeds safe ceiling'


def workspace_tempdir():
    return tempfile.TemporaryDirectory(dir='.')


def synthetic_parent(root, **override):
    frozen = dict(update=191, scoring_phase_position=191, panel=[1, 2, 3, 4],
        independent=dict(raw_dJ_vs_CTRL0_dense=0., raw_dJ_vs_CTRL0_long=0.))
    row = dict(update=192, phase_update=1, label='EXPLORATORY-LEAN', optimizer_update_s=2252.4,
        lean_training_raw_J=dict(raw_dJ_vs_CTRL0_dense=8.6, raw_dJ_vs_CTRL0_long=60.2))
    prior = dict(label='EXPLORATORY-LEAN', status='failed', error=GUARD_EXIT,
        updates=[row], seed=965140, resumed_from=dict(update_count=191),
        operational_recovery=dict(phase_start=191, saved_update=191),
        independent_score_before=dict(raw_dJ_vs_CTRL0_dense=0., raw_dJ_vs_CTRL0_long=0., J=4694.8),
        frozen_scoring_baseline=frozen)
    prior.update(override)
    (root/'RUN_SUMMARY.json').write_text(json.dumps(prior))
    (root/'frozen_scoring_baseline.json').write_text(json.dumps(frozen))
    return prior


class LauncherContractTests(unittest.TestCase):
    def test_source_contract(self):
        source = launch_source(Path('tools/candB/pilot_lean_launch.sh').read_text())
        for expected in ('WMS_SPEED_ES_STACK_SIDES=2',
                'WMS_SPEED_FORWARD_WORKERS=1 WMS_SPEED_FORWARD_MEMORY_FRACTION=.28',
                'WMS_SPEED_FUTURE_WORKERS=2', 'WMS_SPEED_FUTURE_VJP_SLOTS=2',
                'WMS_SPEED_FUTURE_MEMORY_FRACTION=.16', 'PILOT_CONT0525_AUTHORITY.json',
                'tools.candB.pilot_cont0525_authority', 'tools.candB.pilot_cont0525 ',
                '--parent "$PILOT_PARENT"', '>= 21000',
                'PILOT_ADMISSION_PEER_MIB"]) <= 23000'):
            self.assertIn(expected, source)
        self.assertNotIn('--new-phase', source)
        self.assertNotIn('WMS_SPEED_FORWARD_WORKERS=2', source)
        self.assertNotIn('ES_STACK_SIDES=4', source)
        self.assertNotIn('.75', source)


class AdmissionContractTests(unittest.TestCase):
    def test_peer_aware_admission(self):
        verdict_peer = dict(epoch=1., total_MiB=49140, free_MiB=35999, used_MiB=12512,
            apps='82309, 12500')
        self.assertEqual(peers_mib(verdict_peer), 12500)
        self.assertTrue(admit(verdict_peer))
        self.assertFalse(admit(dict(verdict_peer, free_MiB=20000)))
        self.assertFalse(admit(dict(verdict_peer, apps='82309, 12500\n99, 12000')))
        self.assertTrue(admit(dict(verdict_peer, apps='')))


@unittest.skipUnless(PARENT.exists(), 'gpuA runtime only')
class GpuAContractTests(unittest.TestCase):
    def test_memory_bound_peer_fitting_recipe(self):
        from tools.candB.pilot_lean import memory_bound
        total = 49140*1024**2
        peer = 12500*1024**2
        reserved = 2532*1024**2
        # Owner05:25 S2 recipe admits next to the verdict peer at every pool.
        memory_bound(total, reserved, peer, workers=1, fraction=.28)
        memory_bound(total, reserved, peer, workers=2, fraction=.16)
        # The exact v2 S4 exit is reproduced by the unchanged guard.
        with self.assertRaisesRegex(ValueError, 'safe ceiling'):
            memory_bound(total, reserved, peer, workers=1, fraction=.75)
        with self.assertRaisesRegex(ValueError, 'safe ceiling'):
            memory_bound(total, reserved, 30000*1024**2, workers=2, fraction=.16)

    def test_authority_binds_peer_aware_s2_recipe(self):
        from tools.candB import pilot_authority, pilot_cont0525_authority
        path = Path('authority_gen/PILOT_CONT0525_AUTHORITY.json')
        doc = pilot_cont0525_authority.verify(path, exploratory=True)
        self.assertEqual(doc['lean_recipe']['wmfast_stack_sides'], 2)
        self.assertTrue(doc['lean_recipe']['continuation_only'])
        self.assertIn('05:25', doc['lean_recipe']['owner_amendment'])
        self.assertIn('peers<=23000MiB', doc['lean_recipe']['memory_selection'])
        environment = pilot_cont0525_authority.ENV
        self.assertEqual(environment['WMS_SPEED_ES_STACK_SIDES'], '2')
        self.assertEqual(environment['WMS_SPEED_FORWARD_WORKERS'], '1')
        self.assertEqual(environment['WMS_SPEED_FORWARD_MEMORY_FRACTION'], '.28')
        self.assertEqual(environment['WMS_SPEED_FUTURE_VJP_SLOTS'], '2')
        self.assertEqual(environment['WMS_SPEED_FUTURE_MEMORY_FRACTION'], '.16')
        self.assertEqual(doc['lean_recipe']['environment'], environment)

    def test_real_parent_validate_and_bindings(self):
        import torch
        from tools.candB import pilot_cont0525
        prior, origin = pilot_cont0525.validate_parent(PARENT)
        self.assertEqual(origin, 191)
        self.assertEqual(prior['updates'][-1]['update'], 192)
        payload = torch.load(PARENT/'policy_latest.pt', map_location='cpu', weights_only=True)
        self.assertEqual(payload['update_count'], 192)
        for name, digest in (('policy_latest.pt', pilot_cont0525.CHECKPOINT),
                ('frozen_scoring_baseline.json', pilot_cont0525.BASELINE)):
            import hashlib
            self.assertEqual(hashlib.sha256((PARENT/name).read_bytes()).hexdigest(), digest)

    def test_validate_parent_refusals(self):
        import hashlib
        from tools.candB import pilot_cont0525
        with workspace_tempdir() as directory:
            root = Path(directory)
            synthetic_parent(root)
            (root/'policy_latest.pt').write_bytes(b'synthetic checkpoint 192')
            bound = dict(PARENT=root,
                BASELINE=hashlib.sha256((root/'frozen_scoring_baseline.json').read_bytes()).hexdigest(),
                CHECKPOINT=hashlib.sha256((root/'policy_latest.pt').read_bytes()).hexdigest())
            with patch.multiple(pilot_cont0525, **bound):
                # Exact schema plus bound synthetic hashes: only deep checks remain.
                prior, origin = pilot_cont0525.validate_parent(root)
                self.assertEqual(origin, 191)
                for override, pattern in (
                        (dict(label='EXPLORATORY'), 'CONT0525 requires'),
                        (dict(error='RuntimeError: own-run STOP requested; checkpoint saved'),
                            'CONT0525 requires'),
                        (dict(error="TypeError: 'NoneType' object is not subscriptable"),
                            'CONT0525 requires'),
                        (dict(status='completed'), 'CONT0525 requires'),
                        (dict(updates=[dict(update=193, phase_update=1,
                            label='EXPLORATORY-LEAN')]), 'phase clock/label'),
                        (dict(updates=[dict(update=192, phase_update=1,
                            label='EXPLORATORY-LEAN', R4=dict(passed=False))]), 'R4'),
                        (dict(updates=[dict(update=192, phase_update=1, label='EXPLORATORY-LEAN',
                            lean_training_raw_J=dict(raw_dJ_vs_CTRL0_dense=float('inf'),
                                raw_dJ_vs_CTRL0_long=1.))]), 'nonfinite')):
                    synthetic_parent(root, **override)
                    with self.assertRaisesRegex(ValueError, pattern):
                        pilot_cont0525.validate_parent(root)
                synthetic_parent(root)
                (root/'policy_latest.pt').write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError, 'artifact drift'):
                    pilot_cont0525.validate_parent(root)
            synthetic_parent(root)
            with self.assertRaisesRegex(ValueError, 'restricted'):
                pilot_cont0525.validate_parent(root)


if __name__ == '__main__':
    unittest.main()
