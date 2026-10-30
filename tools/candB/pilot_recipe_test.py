"""CPU mocked-update test: a mid-update admission cannot change that update."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


class RecipeTest(unittest.TestCase):
    def test_boundary_latch_budget_and_patch_restoration(self):
        import torch
        from tools.candB import pilot_cli as base, pilot_recipe
        from wms.control import parallel_futures
        from wms.control.dreamer import DreamerTrainer

        trainer = object.__new__(DreamerTrainer)
        trainer.update_count, trainer.device = 190, 'cuda'
        common = dict(eslong=True, kind='forward')
        observed = []

        class FakePool:
            def __init__(self, *args):
                pass

        class SerialPool(FakePool):
            pass

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def update(trainer):
                # This arrives after boundary latch: must not affect update190.
                (output / 'TWO_FORWARD_WORKERS.json').write_text(json.dumps(dict(
                    label='EXPLORATORY', forward_workers=2)))
                pool = parallel_futures.FuturePool(trainer, common)
                observed.append(isinstance(pool, SerialPool))
                trainer.update_count += 1
                return {}

            def fake_run(*args, **kwargs):
                with base.patch.object(parallel_futures, 'FuturePool', SerialPool), base.patch.object(
                        DreamerTrainer, 'update', update):
                    first = trainer.update()
                    second = trainer.update()
                    with patch.object(torch.cuda, 'memory_reserved', return_value=99 * 1024**3):
                        fallback = trainer.update()
                self.assertEqual(first['pilot_recipe']['actual'], 'serial')
                self.assertEqual(second['pilot_recipe']['actual'], 'two-forward-workers')
                self.assertEqual(fallback['pilot_recipe']['actual'], 'serial-budget-fallback')
                self.assertEqual([first['eslong_forward_workers'], second['eslong_forward_workers'],
                                  fallback['eslong_forward_workers']], [1, 2, 1])

            original_factory, original_update = base.patch, DreamerTrainer.update
            with patch.object(base, 'run', fake_run), patch.object(parallel_futures, 'FuturePool', FakePool), \
                    patch.object(torch.cuda, 'memory_reserved', return_value=5 * 1024**3), \
                    patch.object(torch.cuda, 'get_device_properties', return_value=SimpleNamespace(
                        total_memory=48 * 1024**3)):
                pilot_recipe.run('authority.json', output, exploratory=True)
                self.assertIs(parallel_futures.FuturePool, FakePool)
            self.assertEqual(observed, [True, False, True])
            self.assertIs(base.patch, original_factory)
            self.assertIs(DreamerTrainer.update, original_update)


if __name__ == '__main__':
    unittest.main()
