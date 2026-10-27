"""Three isolated production updates at the candidate horizon; never adopt outputs."""
import argparse
from dataclasses import replace
import json
import math
import time
from pathlib import Path


class SmokeComplete(BaseException):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizon', type=int, choices=[10, 30, 60], required=True)
    parser.add_argument('--receipt', required=True)
    args, cli_args = parser.parse_known_args()
    import torch
    from wms.control import policy14_cli as cli
    from wms.control.dreamer import DreamerTrainer

    # Policy14Trainer inherits this production update unchanged.
    original = DreamerTrainer.update
    imagination_config = DreamerTrainer._imagination_config
    observed, rows = set(), []
    started = time.monotonic()
    doc = {'status': 'FAIL', 'horizon_s': args.horizon, 'updates': rows,
           'isolated': True, 'adoptable': False,
           'stress_override': 'training-only base horizon and entrant cap; scoring unchanged'}

    def measured_config(self, batch):
        cfg = imagination_config(self, batch)
        if self._training_imagination:
            cfg = replace(cfg, horizon=args.horizon)
            observed.add(cfg.horizon)
        return cfg

    def measured_update(self):
        old = self.config
        # Stress even low-share starts at the requested horizon. No scoring or
        # full-phase behavior is changed, and no smoke checkpoint is adopted.
        pad = {10: 32, 30: 83, 60: 149}[args.horizon]
        self.config = replace(old, horizon=args.horizon, entry_pad_max=pad)
        torch.cuda.synchronize()
        tick = time.monotonic()
        try:
            metrics = original(self)
            torch.cuda.synchronize()
        finally:
            self.config = old
        if not all(math.isfinite(metrics[k]) for k in
                   ('actor_loss', 'critic_loss', 'actor_grad_norm', 'critic_grad_norm')):
            raise ValueError('nonfinite resource-smoke update')
        rows.append({'seconds': time.monotonic()-tick, 'metrics': metrics})
        if len(rows) == 3:
            doc['paircrit_monitor'] = self.paircrit.diagnostics(self)
            doc['status'] = 'PASS'
            raise SmokeComplete()
        return metrics

    DreamerTrainer.update = measured_update
    DreamerTrainer._imagination_config = measured_config
    try:
        code = cli.main(cli_args)
        output = Path(cli_args[cli_args.index('--output-dir')+1])/'RUN_SUMMARY.json'
        summary = json.loads(output.read_text()) if output.exists() else {}
        doc['error'] = summary.get('error', f'CLI returned before three updates: {code}')
        doc['resource_failure'] = 'CUDA out of memory' in doc['error']
    except SmokeComplete:
        pass
    except Exception as error:
        doc['error'] = repr(error)
        doc['resource_failure'] = isinstance(error, torch.cuda.OutOfMemoryError)
    finally:
        DreamerTrainer.update = original
        DreamerTrainer._imagination_config = imagination_config
        doc.update(wall_seconds=time.monotonic()-started,
                   observed_training_horizons=sorted(observed),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                   allocator_cap_bytes=int(torch.cuda.get_device_properties(0).total_memory * 0.65))
        Path(args.receipt).write_text(json.dumps(doc, indent=1, allow_nan=False))
    return 0 if doc['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
