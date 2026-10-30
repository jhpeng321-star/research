"""Small bounded synthetic speed/exactness probe; never loads traffic data."""
from __future__ import annotations

import argparse
import copy
import cProfile
import json
from pathlib import Path
import pstats
import statistics
import subprocess
import tempfile
import time
from types import SimpleNamespace

import pytest
import torch

from tests.unit.test_speedprof import real_trainer
from tests.unit.test_eslong import make
from tests.unit.test_longmem import assert_tree_close
from wms.control import policy14_cli as cli
from wms.control.policy14 import independent_score
from wms.control.dreamer import _rng_state


def timed(function, repeats=3, *, device='cpu'):
    values = []
    for _ in range(repeats):
        start = time.monotonic()
        function()
        if device == 'cuda':
            torch.cuda.synchronize()
        values.append(time.monotonic()-start)
    return dict(samples=values, median=statistics.median(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.device == 'cpu' and torch.cuda.is_available():
        raise ValueError('hide CUDA for this CPU-only probe')
    if args.device == 'cuda':
        peers = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                         '--format=csv,noheader'], text=True).strip()
        if peers:
            raise ValueError('GPU probe requires an idle gpuR (no peers)')
        # <1.5 GiB allocator + CUDA context/libraries, well below 4 GB total.
        torch.cuda.set_per_process_memory_fraction(.06)
        free, total = torch.cuda.mem_get_info()
        if free < 8*1024**3 or total > 26*1024**3:
            raise ValueError('GPU probe requires free 24 GB gpuR')
    torch.set_num_threads(1)
    result = dict(torch=torch.__version__, device=args.device, threads=torch.get_num_threads(),
                  production_timing=False)
    tmp = tempfile.mkdtemp(prefix='speedprof_')
    result['retained_probe_directory'] = tmp
    with pytest.MonkeyPatch.context() as patch:
        root = Path(tmp)
        patch.setenv('C4_ALLOWED_ROOTS', tmp)
        trainer = real_trainer(root, patch)
        if args.device == 'cuda':
            from torch.utils._pytree import tree_map
            trainer.device = torch.device('cuda')
            for module in (*trainer.world_models, trainer.policy, trainer.group_encoder,
                           trainer.critic, trainer.target_critic):
                module.to('cuda')
            trainer.start_states.batch.state = tree_map(lambda value: value.to('cuda'),
                                                       trainer.start_states.batch.state)
        batch = trainer.start_states.batch
        # Current pilot exact fast paths, not an unaccelerated strawman.
        patch.setenv('WMS_WMSPEED_DEDUPE_MOBIL', '1')
        patch.setenv('WMS_WMSPEED_MOBIL_LAZY_SAFE', '1')
        patch.setenv('WMS_SPEED_PIVOT_DIAG_OFF', '1')
        independent_score(trainer, batch)  # Warm-up; freezes the scoring phase.
        before = _rng_state(trainer.device)
        val = SimpleNamespace(relabel=False, _load=lambda index: index,
                              _batch=lambda rows, device: copy.deepcopy(batch))
        panel = dict(starts=[dict(index=index) for index in range(16)])
        def panels():
            native = cli.score_native_panel(trainer, val, panel, args.device)
            gate = cli.score_gate_panel(trainer, val, panel, args.device)
            return native, gate, copy.deepcopy(trainer._gatemeas_panel_scores)
        patch.setenv('WMS_SPEED_REUSE_PANEL', '0')
        expected = panels()
        result['panel_uncached'] = timed(panels, device=args.device)
        patch.setenv('WMS_SPEED_REUSE_PANEL', '1')
        actual = panels()
        assert_tree_close(expected, actual, 0, 0)
        assert_tree_close(before, _rng_state(trainer.device), 0, 0)
        result['panel_reused'] = timed(panels, device=args.device)
        result['panel_exact'] = 'bitwise J/native gate/full matched score receipts/RNG'
        result['score_one_start'] = timed(lambda: independent_score(trainer, batch), device=args.device)
        profiler = cProfile.Profile()
        profiler.runcall(independent_score, trainer, batch)
        stats = pstats.Stats(profiler)
        top = sorted(stats.stats.items(), key=lambda pair: pair[1][2], reverse=True)[:18]
        result['score_profile_self'] = [dict(file=key[0], line=key[1], function=key[2],
            calls=value[1], self_s=value[2], cumulative_s=value[3]) for key, value in top]
        result['score_profile_calls'] = stats.total_calls
        result['score_profile_total_s'] = stats.total_tt
        patch.setenv('WMS_SPEED_REUSE_PANEL', '0')
        expected = independent_score(trainer, batch)
        patch.setenv('WMS_WMSPEED', '1')
        actual = independent_score(trainer, batch)
        assert_tree_close(expected, actual, 0, 0)
        result['wmspeed_master'] = dict(exact='bitwise score',
            remaining_flags_policy_call_sites=0,
            timing=timed(lambda: independent_score(trainer, batch), device=args.device))
        if args.device == 'cuda':
            result['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            result['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        # Logged-only extra VJPs on the complete M2/K8 antithetic toy update.
        base = make(root, patch)
        from dataclasses import replace
        base.config = replace(base.config, actor_gradient_audit=True)
        base.counterfactual = replace(base.counterfactual, term_grad_audit_every=1, n_individual=3)
        def update(skip):
            patch.setenv('WMS_SPEED_SKIP_GRAD_DIAGNOSTICS', str(int(skip)))
            t = copy.deepcopy(base)
            torch.manual_seed(131)
            start = time.monotonic()
            row = t.update()
            return time.monotonic()-start, t, row
        on, reference, ref_row = update(False)
        off, candidate, row = update(True)
        for left, right in zip(reference.actor_parameters, candidate.actor_parameters):
            assert torch.equal(left, right) and torch.equal(left.grad, right.grad)
        assert_tree_close(reference.actor_optimizer.state_dict(), candidate.actor_optimizer.state_dict(), 0, 0)
        assert row['eslong_pairs'] == ref_row['eslong_pairs']
        result['gradient_diagnostics'] = dict(on_seconds=on, off_seconds=off,
            exact='bitwise weights/gradients/Adam/ES pair gates', fixture='short toy M2 K8, not production')
    args.output.write_text(json.dumps(result, indent=1)+'\n')
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
