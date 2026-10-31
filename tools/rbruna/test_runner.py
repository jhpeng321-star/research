"""Operational invariants only; no inference, SSH, CUDA or simulator execution."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

from tools.rbruna import remote, watch


class RunnerTests(unittest.TestCase):
    def test_candidate_preserves_int_variant_and_registered_future_builder(self):
        calls = []
        def builder():
            pass
        def harness(*, model_loader, identity_sources, future_builder):
            calls.append((os.environ['WMPHYS_VARIANT'], model_loader, future_builder, identity_sources))
        source = str(Path(__file__).resolve())
        loader = Mock()
        spec = dict(runtime_root='unused', loader_module='fake_recipe',
                    loader_function='load_candidate', recipe_variant='B',
                    future_builder_function='candidate_future', identity_sources=[source],
                    sources={source: 'bound'}, smoke_receipt='smoke.json')
        modules = {'fake_recipe': SimpleNamespace(load_candidate=loader, candidate_future=builder),
                   'run_wm': SimpleNamespace(main=harness)}
        with patch.dict(sys.modules, modules), patch.dict(os.environ), patch.object(sys, 'path', list(sys.path)):
            remote.candidate(spec)
        self.assertEqual(calls, [('B', loader, builder, [source, 'smoke.json'])])
        modules['run_wm'] = SimpleNamespace(main=lambda model_loader, identity_sources: self.fail('legacy harness executed'))
        with patch.dict(sys.modules, modules), patch.dict(os.environ), patch.object(sys, 'path', list(sys.path)):
            with self.assertRaisesRegex(AssertionError, 'before READY'):
                remote.candidate(spec)

    def test_atomic_claim_has_exactly_one_winner_and_preserves_owner(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(watch, 'SHARED', Path(tmp)):
            with ThreadPoolExecutor(max_workers=12) as pool:
                wins = list(pool.map(lambda _: watch.claim('WM-PHYS-A', {}), range(24)))
            self.assertEqual(sum(wins), 1)
            p = Path(tmp)/'WM-PHYS-A.claim'
            before = p.read_bytes()
            self.assertFalse(watch.claim('WM-PHYS-A', {'changed': True}))
            self.assertEqual(p.read_bytes(), before)
            self.assertEqual(watch.read_claim(p)['owner'], 'S-RB-RUNA')

    def test_only_ready_section_and_published_order(self):
        prefix = 'E:/agent-work/rb-shared/evalclaims/'
        text = (f'# Pending\n{prefix}C.runa.json\n## READY FOR EVAL\n'
                f'- {prefix}A.runa.json\n### smoke\n- {prefix}B.runa.json\n'
                f'- {prefix}A.runa.json\n## Known gaps\n{prefix}D.runa.json\n')
        self.assertEqual(watch.discover(text), [prefix+'A.runa.json', prefix+'B.runa.json'])

    def test_plain_peer_claim_is_respected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'B.claim'
            p.write_text('S-RB-RUNB\n')
            self.assertEqual(watch.read_claim(p)['owner'], 'S-RB-RUNB')

    def test_complete_panel_requires_all160_identity_bound_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
            with patch.object(remote, 'sha', digest):
                for i in range(3):
                    d = root/f'shard{i}'
                    d.mkdir()
                    (d/'IDENTITY.json').write_text(json.dumps({'shard': i}))
                    hashes = {}
                    for j, seed in enumerate(range(969420, 969440)):
                        if j % 3 != i:
                            continue
                        for k in range(8):
                            name = f'w{seed}_k{k}.json'
                            (d/name).write_text(json.dumps({'seed': seed, 'future': k}))
                            hashes[name] = digest(d/name)
                    remote.save(d/'COMPLETE.json', dict(tasks=len(hashes), hashes=hashes,
                                                       identity_sha256=digest(d/'IDENTITY.json')))
                self.assertTrue(remote.completed(root))
                (root/'shard0/w969420_k0.json').write_text('corrupt')
                with self.assertRaises(AssertionError):
                    remote.completed(root)

    def test_stop_never_signals_reused_pid(self):
        with tempfile.TemporaryDirectory() as tmp:
            child = Mock()
            child.poll.return_value = None
            with patch.object(remote, 'alive', return_value=False), \
                    patch.object(remote.time, 'monotonic', side_effect=[0, 21]):
                remote.stop_own(Path(tmp), [(child, {'pid': 123})])
            child.terminate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
