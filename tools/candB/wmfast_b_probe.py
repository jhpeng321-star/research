"""Bounded ES-only parity/timing probe. Synthetic starts; no update/Adam/data/SUMO."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import time

import pytest
import torch

from tests.unit.test_wmfast_b import common_context, real_fixture, tensor_leaves
from wms.control import eslong, gated300
from wms.control.counterfactual import row_coefficients
from wms.control.dreamer import _rng_state, _restore_rng


def compare(left, right):
    a, b = tensor_leaves(left), tensor_leaves(right)
    assert len(a) == len(b)
    changed, maximum = 0, 0.
    for x, y in zip(a, b):
        assert x.shape == y.shape and x.dtype == y.dtype
        if not torch.equal(x, y):
            assert x.is_floating_point(), 'discrete WM outcome changed'
            changed += 1
            maximum = max(maximum, float((x.double()-y.double()).abs().max()))
            torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-6)
    return dict(tensors=len(a), nonbitwise=changed, max_abs=maximum,
                tolerance=dict(atol=2e-5, rtol=2e-6))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-small-shared-probe', action='store_true')
    args = parser.parse_args()
    if args.device == 'cuda':
        peers = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                         '--format=csv,noheader'], text=True).strip()
        if peers and not args.allow_small_shared_probe:
            raise RuntimeError('bounded probe requires no peers; retry at an idle boundary')
        torch.cuda.set_per_process_memory_fraction(.04)  # <1 GiB + context on gpuR
        free, total = torch.cuda.mem_get_info()
        if free < 8*1024**3 or total > 26*1024**3:
            raise RuntimeError('bounded probe requires idle 24GB gpuR')
    torch.set_num_threads(1)
    root = Path(tempfile.mkdtemp(prefix='wmfastB_'))
    result = dict(device=args.device, torch=torch.__version__, threads=1,
        retained_probe_directory=str(root), production_timing=False,
        physical_horizon_s=10, registered_gate_horizon_s=300, K=8, M=2,
        fixture='real hybrid WM; 4 synthetic initial slots + 2 birth slots; learned entrant mix',
        cases={})
    from wms.control import dreamer, parallel_futures
    result['source_sha256'] = {Path(module.__file__).name:
        hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for module in (eslong, dreamer, parallel_futures)}
    if args.device == 'cuda':
        result.update(peers_at_admission=peers, allocator_fraction=.04,
                      free_bytes_at_admission=free)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv('C4_ALLOWED_ROOTS', str(root))
        patch.setenv('WMS_WMSPEED_DEDUPE_MOBIL', '1')
        patch.setenv('WMS_WMSPEED_MOBIL_LAZY_SAFE', '1')
        trainer = real_fixture(root, patch, args.device)
        imagined, common = common_context(trainer)
        n = common['kwargs']['variants'].n_rows
        result['rows_per_side'] = n
        order = [(i, sign) for i in range(2) for sign in (1, -1)]
        rng = _rng_state(trainer.device)
        original_prefix, full = gated300.prefix, []
        def prefix(owner, out, dense):
            full.append([x.detach().cpu().clone() for x in tensor_leaves(out)])
            return original_prefix(owner, out, dense)
        def run(sides):
            if not sides:
                return [eslong.forward_task(trainer, common, (*side, k))
                        for side in order for k in range(8)]
            outputs = []
            for start in range(0, 4, sides):
                group = [eslong.forward_stacked_task(trainer, common, (order[start:start+sides], k))
                         for k in range(8)]
                outputs.extend(group[k][s] for s in range(sides) for k in range(8))
            return outputs
        coeff = row_coefficients(imagined['variants'], trainer.counterfactual.weights(0), n,
                                 imagined['n_base'])
        def estimator(outputs, factor=1., offset=0.):
            deltas, receipts = [], []
            for i in range(2):
                samples = [offset + factor * torch.cat([
                    out['samples'] for out in outputs[(2*i+s)*8:(2*i+s+1)*8]])
                           for s in range(2)]
                delta, receipt = eslong.paired_loss_difference(trainer, *samples, imagined, coeff, 1.)
                deltas.append(delta)
                receipts.append(receipt)
            return eslong.estimate(common['directions'], torch.stack(deltas), .08), receipts
        patch.setattr(gated300, 'prefix', prefix)
        expected = run(0)
        reference_full = full[:]
        full.clear()
        g, receipts = estimator(expected)
        scaled_g, scaled_receipts = estimator(expected, 10000., 1000000.)
        result['accepted_pairs'] = sum(r['accepted'] for r in receipts)
        result['g_ES_norm'] = float(g.norm())
        result['synthetic_gate_stress'] = dict(sample_multiplier=10000., sample_offset=1000000.,
            accepted_pairs=sum(r['accepted'] for r in scaled_receipts), g_ES_norm=float(scaled_g.norm()))
        assert result['synthetic_gate_stress']['accepted_pairs'] > 0 and bool(scaled_g.any())
        for sides in (2, 4):
            actual = run(sides)
            ordered_full = []
            for start in range(0, 4, sides):
                ordered_full.extend(full[start*8 + k*sides+s] for s in range(sides) for k in range(8))
            other_g, other_receipts = estimator(actual)
            other_scaled_g, other_scaled_receipts = estimator(actual, 10000., 1000000.)
            assert [r['on'] for r in receipts] == [r['on'] for r in other_receipts]
            assert [r['on'] for r in scaled_receipts] == [r['on'] for r in other_scaled_receipts]
            result['cases'][str(sides)] = dict(full_long_outputs=compare(reference_full, ordered_full),
                credit_and_actions=compare(expected, actual), g_ES=compare(g, other_g),
                accepted_masks_bitwise=True, receipts_bitwise=receipts == other_receipts,
                synthetic_gate_stress_g_ES=compare(scaled_g, other_scaled_g),
                stacked_rows=eslong.stacked_variants(common['kwargs']['variants'], sides,
                                                     share_reference=True)[0].n_rows)
            full.clear()
        patch.setattr(gated300, 'prefix', original_prefix)
        for sides in (0, 2, 4):
            gc.collect()
            if args.device == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            samples = []
            for _ in range(args.repeats):
                if args.device == 'cuda' and torch.cuda.mem_get_info()[0] < 4*1024**3:
                    raise RuntimeError('peer memory increased; stop own small probe')
                start = time.perf_counter()
                run(sides)
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                samples.append(time.perf_counter()-start)
            case = result['cases'].setdefault(str(sides), {})
            case.update(seconds=samples, median_seconds=statistics.median(samples))
            if args.device == 'cuda':
                case.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                            peak_reserved_bytes=torch.cuda.max_memory_reserved())
        for sides in (2, 4):
            result['cases'][str(sides)]['speedup'] = (
                result['cases']['0']['median_seconds'] / result['cases'][str(sides)]['median_seconds'])
        _restore_rng(rng, trainer.device)
        result['rng_restored_bitwise'] = all(torch.equal(x, y) for x, y in zip(
            tensor_leaves(rng), tensor_leaves(_rng_state(trainer.device))))
        if args.device == 'cuda':
            result['process_gpu_memory'] = subprocess.check_output([
                'nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                '--format=csv,noheader'], text=True).strip()
            own = [line for line in result['process_gpu_memory'].splitlines()
                   if int(line.split(',')[0]) == os.getpid()]
            assert len(own) == 1
            result['own_gpu_memory_MiB'] = int(own[0].split(',')[1].strip().split()[0])
            assert result['own_gpu_memory_MiB'] <= 3072
    args.output.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result, indent=1), flush=True)


if __name__ == '__main__':
    main()
