"""Compare shared WM default-OFF paths to our pre-coordinate-fix git source.

Synthetic CPU inputs only. Execute repository WM source, never downloaded
SUMO code. Bind the comparison to an explicit preserved commit, not a mutable
reference file or another workstream's runtime.
"""
import hashlib
import json
import subprocess
import sys
import types

import torch

from .operate import RECEIPTS, ROOT
from wms.model.spec import ModelSpec
from wms.model.wms import WmSModel
from wms.physics.config import PhysicsConfig

BASELINE = 'b192e0c80a88f3d93075ff9620e1c42d0afbd741'


def compare(left, right, counters):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            compare(left[key], right[key], counters)
    elif torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
        counters['tensor_leaves'] += 1
        counters['tensor_elements'] += left.numel()
    else:
        assert left == right


def main():
    out = RECEIPTS/'DEFAULT_OFF_COORDINATES.json'
    assert out.parent.is_dir() and not out.exists()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = subprocess.check_output(['git', 'show', BASELINE + ':src/wms/model/wms.py'], cwd=ROOT)
    original = types.ModuleType('wms.model._rblane_original')
    original.__file__ = str(ROOT/'src/wms/model/wms.py')
    original.__package__ = 'wms.model'
    sys.modules[original.__name__] = original
    exec(compile(source, original.__file__, 'exec'), original.__dict__)
    state = dict(s_m=torch.tensor([[-100., -70., 20.]]),
        v_mps=torch.tensor([[15., 10., 5.]]), a_mps2=torch.zeros(1, 3),
        lane=torch.tensor([[1, 1, 2]]), length_m=torch.full((1, 3), 5.),
        present=torch.ones(1, 3).bool(), is_cav=torch.zeros(1, 3).bool())
    cases = []
    for name, spec in (
        ('legacy', ModelSpec()),
        ('legacy_conflict_enabled', ModelSpec(lane_conflict_resolve=True)),
        ('physical_history', ModelSpec(history_s=60., driver_param_mode='physical', rollout_mode='hybrid'))):
        physics = PhysicsConfig(sumo_fidelity=True)
        torch.manual_seed(419)
        previous = original.WmSModel(spec, physics=physics).eval()
        torch.manual_seed(419)
        current = WmSModel(spec, physics=physics).eval()
        count = dict(tensor_leaves=0, tensor_elements=0)
        compare(previous.state_dict(), current.state_dict(), count)
        kw = (dict(driver_history=torch.zeros(1, 3, 61, 9),
                   driver_history_mask=torch.zeros(1, 3, 61).bool())
              if spec.history_s is not None else {})
        with torch.no_grad():
            for operation in ('forward', 'step', 'rollout'):
                kwargs = dict(kw)
                if operation == 'rollout':
                    kwargs['n_steps'] = 4
                torch.manual_seed(427)
                left = getattr(previous, operation)(dict(state), **kwargs)
                torch.manual_seed(427)
                right = getattr(current, operation)(dict(state), **kwargs)
                compare(left, right, count)
        cases.append(dict(case=name, status='BIT_EXACT', **count))
    report = dict(status='PASS', baseline_commit=BASELINE,
        previous_shared_source_sha256=hashlib.sha256(source).hexdigest(),
        current_shared_source_sha256=hashlib.sha256((ROOT/'src/wms/model/wms.py').read_bytes()).hexdigest(),
        cases=cases, synthetic_only=True, torch_threads=1)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
