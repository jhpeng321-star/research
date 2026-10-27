"""Exact one-start microbatch (effective batch=1), with saved-tensor CPU storage.

The registered POLICY14 start batch is already one. Accumulation therefore has
one microbatch and one optimizer step, with weight 1. No row, future, timestep,
normalization, RNG draw, loss, clipping or optimizer order is changed. Autograd
saved tensors are copied losslessly to CPU and restored to their original device
for backward. This is necessary because starts cannot be split below one.
"""
import json
import os
from pathlib import Path
import sys
import time


def install():
    import torch
    from wms.control.dreamer import DreamerTrainer

    original = DreamerTrainer.update

    def update(self):
        if self.config.batch_size != 1:
            raise ValueError('E5: this registered microbatch adapter requires effective starts batch=1')
        start = time.time()
        if str(self.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        with torch.autograd.graph.save_on_cpu(pin_memory=False):
            metrics = original(self)
        if self.output_dir is not None:
            doc = {'update': self.update_count, 'effective_batch_starts': 1,
                   'microbatch_starts': 1, 'accumulation_steps': 1,
                   'loss_weight': 1.0, 'saved_tensor_storage': 'cpu_lossless',
                   'wall_s': time.time() - start}
            if str(self.device).startswith('cuda'):
                doc.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved())
            with (Path(self.output_dir) / 'AUTONOMY_MEMORY.jsonl').open('a') as stream:
                stream.write(json.dumps(doc) + '\n')
        return metrics

    DreamerTrainer.update = update


if __name__ == '__main__':
    root = Path(os.environ['AUTONOMY_RUNTIME'])
    sys.path[:0] = [str(root / 'src'), str(root)]
    install()
    from wms.control.policy14_cli import main
    raise SystemExit(main())
