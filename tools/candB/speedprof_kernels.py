"""CPU batching/graphability screening, not an ES estimator replacement."""
import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time

import pytest
import torch

from tests.unit.test_speedprof import real_trainer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise ValueError('this screening probe is CPU-only')
    torch.set_num_threads(1)
    root = Path(tempfile.mkdtemp(prefix='speedprof_kernels_'))
    result = dict(device='cpu', torch=torch.__version__, scope='one synthetic physics step; NOT full ES',
                  retained_probe_directory=str(root))
    with pytest.MonkeyPatch.context() as patch, torch.no_grad():
        trainer = real_trainer(root, patch)
        model = trainer.world_models[0]
        state = {key: value.clone() for key, value in trainer.start_states.batch.state.items()}
        state['is_cav'][:, -2:] = False
        def step(scene):
            return model.step(scene, lane_uniform=torch.full_like(scene['s_m'], .5))
        stacked = {key: value.repeat((4,) + (1,)*(value.ndim-1)) for key, value in state.items()}
        serial = step(state)
        batch = step(stacked)
        keys = ('accel_mps2', 'next_s_m', 'next_v_mps', 'next_lane', 'next_present')
        comparison = {}
        for key in keys:
            expected = serial[key].repeat((4,) + (1,)*(serial[key].ndim-1))
            comparison[key] = dict(bitwise=torch.equal(expected, batch[key]),
                max_abs=float((expected.double()-batch[key].double()).abs().max()))
        result['batch4_comparison'] = comparison
        def elapsed(function):
            values = []
            for _ in range(5):
                start = time.monotonic()
                for _ in range(5):
                    function()
                values.append((time.monotonic()-start)/5)
            return dict(samples=values, median=statistics.median(values))
        result['four_serial_steps'] = elapsed(lambda: [step(state) for _ in range(4)])
        result['one_batch4_step'] = elapsed(lambda: step(stacked))
        try:
            # Eager backend screens full-graph eligibility without invoking a
            # compiler/toolchain or claiming fusion/reassociation equivalence.
            compiled = torch.compile(step, backend='eager', fullgraph=True)
            compiled(state)
            result['fullgraph_screen'] = 'accepted with eager backend; fusion exactness UNPROVEN'
        except Exception as error:
            result['fullgraph_screen'] = dict(error=type(error).__name__, message=str(error)[:1800])
        result['admitted_full_es_speedup'] = False
    args.output.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
