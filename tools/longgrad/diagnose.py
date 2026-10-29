"""CPU-only synthetic horizon probe and explicit supplied-receipt summarizer.

Run with PYTHONPATH=src. No datasets, simulators, checkpoints or CUDA calls.
The random initialized learned WM is a mechanism probe, not a production replay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def receipts(paths):
    result = []
    for path in paths:
        # Only explicit files; never discover evidence or training data recursively.
        row = json.loads(Path(path).read_text(encoding='utf-8'))
        result.append({'source': str(path), **{k: v for k, v in row.items() if
            k.startswith(('gated300_', 'actor_module_')) or k in
            ('actor_grad_norm', 'return_scale', 'actor_loss', 'update', 'wall_time_s')}})
    if len(result) == 2:
        result.append({'grad_ratio': result[0]['actor_grad_norm'] / result[1]['actor_grad_norm']})
    return result


def probe(horizons):
    import os
    import torch
    from tests.unit.test_longmem import trainer
    from wms.model.spec import ModelSpec
    from wms.model.wms import WmSModel

    torch.set_num_threads(1)
    rows = []
    for horizon in horizons:
        for mode in ('0', '2'):
            os.environ['WMS_LM300_RECOMPUTE_REWARD'] = mode
            # Explicit hybrid mode: the fixture's default residual WM does NOT
            # use inferred drivers. Use a learned (untrained) encoder, never a
            # manually specified driver-parameter dictionary.
            t = trainer(1, 2, True, horizon=horizon, batch=1, slots=4, real_world=True,
                        imagination_grad_decay=1.)
            t.world_models = [WmSModel(ModelSpec(history_hidden=8, k_hidden=8,
                rollout_mode='hybrid')).eval().requires_grad_(False)]
            batch = t.start_states.sample(1, 'cpu')
            batch.state['is_cav'][:, -2:] = False
            t._training_imagination = True
            imagined = t.imagine(batch, deterministic=True)
            loss = -imagined['rewards'].sum()
            grads = torch.autograd.grad(loss, t.actor_parameters, allow_unused=True)
            norm = sum(float(g.double().square().sum()) for g in grads if g is not None) ** .5
            rows.append(dict(horizon_s=horizon, mode=mode, loss=float(loss.detach()),
                             grad_norm=norm))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipts', nargs='*', default=[])
    parser.add_argument('--horizons', nargs='*', type=int, default=[])
    args = parser.parse_args()
    print(json.dumps(dict(receipts=receipts(args.receipts), probe=probe(args.horizons)),
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
