"""Completion must be identity-bound, not inferred from counts or process names."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalize
from core import SEEDS, digest, save_new


class CompletionSafety(unittest.TestCase):
    def test_markers_hashes_and_stopped_worker_block(self):
        parent = Path('E:/Temp/opencode') if os.name == 'nt' else Path('/tmp')
        self.assertTrue(parent.is_dir())
        root = Path(tempfile.mkdtemp(prefix='rbeval-completion-test-', dir=parent))
        for i in range(4):
            (root/f'wm2_shard{i}').mkdir()
        state = finalize.panel_state(root)
        self.assertFalse(state['complete'])
        self.assertTrue(state['blocked'])
        self.assertEqual(state['tasks'], 0)
        for i in range(4):
            shard = root/f'wm2_shard{i}'
            save_new(shard/'IDENTITY.json', dict(synthetic_test=True, shard=i))
            hashes = {}
            for j, seed in enumerate(SEEDS):
                if j % 4 != i:
                    continue
                for k in range(8):
                    task = shard/f'w{seed}_k{k}.json'
                    save_new(task, dict(synthetic_test=True, seed=seed, future=k))
                    hashes[task.name] = digest(task)
            save_new(shard/'COMPLETE.json', dict(tasks=40, identity_sha256=digest(shard/'IDENTITY.json'), hashes=hashes))
        state = finalize.panel_state(root)
        self.assertTrue(state['complete'])
        self.assertFalse(state['blocked'])
        self.assertEqual(state['tasks'], 160)
        actual_digest = finalize.digest
        with patch.object(finalize, 'digest', side_effect=lambda p: 'changed' if Path(p).name == 'w969420_k0.json' else actual_digest(p)):
            with self.assertRaises(AssertionError):
                finalize.panel_state(root)


if __name__ == '__main__':
    unittest.main()
