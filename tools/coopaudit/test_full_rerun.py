"""Regression gates for corrected branch mechanics; no SUMO construction."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import torch
from tools.coopaudit import full_rerun as F
from tools.coopaudit.design import sha


class CorrectedContracts(unittest.TestCase):
    def test_full_panel_cost_and_missing_not_zero(self):
        group=['a','b']
        self.assertEqual(len(F.branch_plan(group)),11)
        self.assertEqual(len(F.branch_plan(group,True)),16)
        self.assertEqual(F.branch_plan([]),[])
        for c in F.ARMS:
            branches=[(name,m) for arm,name,m in F.branch_plan(group) if arm==c]
            self.assertEqual(branches,[('GROUP',group),('ONLY:0',['a']),('ONLY:1',['b'])])

    def test_engine_never_loads_or_saves_state_and_charges_before_reset(self):
        source=Path(F.__file__).read_text()
        tree=ast.parse(source)
        attrs=[n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)]
        self.assertFalse({'save_state','load_state','snapshot_env','saveState','loadState'}&set(attrs))
        self.assertLess(source.index('Ledger(ROOT).charge('),source.index('env.reset('))

    def test_prefix_mismatch_blocks(self):
        with patch.object(F,'initial_state',return_value='different'),patch.object(F,'crn',return_value='same'):
            with self.assertRaisesRegex(ValueError,'prefix tensor/request'):
                F.verify_prefix(None,dict(initial_state_sha256='expected',crn_sha256='same'),[])

    def test_v2_hold_before_any_runtime_or_precharge(self):
        with patch.object(F,'selected',side_effect=ValueError('HOLD')),patch.object(F,'setup') as setup:
            with self.assertRaisesRegex(ValueError,'HOLD'):
                F.run_world({})
            setup.assert_not_called()

    def test_trained_original_source_and_fixed_checkpoint_cpu(self):
        from tools.coopaudit.trained import coordination,trained_bound
        selection=json.loads((F.ROOT/'CORRECTED_DESIGN.json').read_text())
        module=coordination(selection['coordination_source_sha256'])
        from wms.control.policy import load_policy_fn
        payload=torch.load(selection['checkpoint'],map_location='cpu',weights_only=False)
        a=trained_bound(selection)
        b=module.Coordinated(load_policy_fn(selection['checkpoint'],device='cpu'),payload)
        # Synthetic observable frame, same shapes/fields as the executor's CPU fixture.
        state=torch.load('/root/sumocf20261031/inputs/cpu_observed_start.pt',map_location='cpu',weights_only=False)['state']
        for step in range(3):
            state=copy.deepcopy(state)
            state['s_m']+=.125*step
            x=a.exe(copy.deepcopy(state),with_lanes=True)
            y=b(copy.deepcopy(state),with_lanes=True)
            self.assertIsNone(x[1])
            self.assertIsNone(y[1])
            np.testing.assert_array_equal(x[0].view(np.uint64),y[0].view(np.uint64))
            self.assertIsNone(a.exe.intervention_offsets)
        a.exe.close()
        b.close()


if __name__=='__main__':
    unittest.main(verbosity=2)
