"""Ordinary lean phase continuation: no re-zero, no baseline recomputation."""
import argparse
import inspect
import json
import math
from pathlib import Path
import types
from unittest.mock import patch

from tools.candB import pilot_continue as continuation
from tools.candB import pilot_lean_gate, pilot_wmfast_authority as authority


def validate_parent(parent):
    prior = json.loads((parent/'RUN_SUMMARY.json').read_text())
    frozen = json.loads((parent/'frozen_scoring_baseline.json').read_text())
    if (prior.get('label') != authority.LABEL or prior.get('status') != 'failed'
            or prior.get('error') != 'RuntimeError: own-run STOP requested; checkpoint saved'
            or frozen != prior.get('frozen_scoring_baseline')):
        raise ValueError('WMFAST requires clean owned STOP and durable original baseline')
    rows = prior['updates']
    origin = prior.get('operational_recovery', {}).get('phase_start', prior['resumed_from']['update_count'])
    if not rows or not 1 <= rows[-1]['update']-origin < 150:
        raise ValueError('WMFAST requires at least one completed phase update')
    for row in rows:
        if row.get('label') != authority.LABEL or row['phase_update'] != row['update']-origin:
            raise ValueError('WMFAST original phase clock/label mismatch')
    if rows[-1].get('R4', {}).get('passed') is False:
        raise ValueError('cannot continue a failed R4 gate')
    for key in ('raw_dJ_vs_CTRL0_dense', 'raw_dJ_vs_CTRL0_long'):
        if frozen['independent'][key] != 0.:
            raise ValueError('original zero phase baseline differs from CTRL0')
        if not math.isfinite(rows[-1]['independent_J'][key]):
            raise ValueError('nonfinite latest clean receipt')
    return prior, origin


def lean_module(base, prior):
    from tools.candB import pilot_lean
    source = inspect.getsource(pilot_lean)
    source = continuation.replace_once(source,
        'from tools.candB import pilot_cli as base, pilot_lean_authority as authority',
        'from tools.candB import pilot_wmfast_authority as authority')
    source = continuation.replace_once(source,
        'state = dict(phase=0, workers=[], capture=None, raw=None, own_worker_pids=set())',
        "state = dict(phase=prior['updates'][-1]['phase_update'], "
        "startup_s=prior['updates'][0]['startup_s'], workers=[], capture=None, raw=None, own_worker_pids=set())")
    module = types.ModuleType('tools.candB.pilot_wmfast_lean_adapter')
    module.__dict__.update(base=base, prior=prior)
    exec(compile(source, pilot_lean.__file__, 'exec'), module.__dict__)
    return module


def run(path, output, parent, *, exploratory=False):
    from tools.candB import pilot_authority
    from wms.control import nativeinit, policy14_cli as cli, policy14_recovery
    from wms.control.dreamer import DreamerTrainer
    import torch
    pilot_authority.require_opt_in(exploratory)
    authority.verify(path, exploratory=True)
    parent = Path(parent).resolve()
    with patch.object(continuation, 'validate_parent', validate_parent):
        base, verify_runtime, prior = continuation.prepare_base(parent)
    adapter = lean_module(base, prior)
    payload = torch.load(parent/'policy_latest.pt', map_location='cpu', weights_only=True)
    proof = dict(label=authority.LABEL, same_phase=True, baseline_recomputed=False,
        phase_origin=validate_parent(parent)[1], saved_update=payload['update_count'],
        phase_updates=150, original_R4_phase_update=10, parent=str(parent),
        seed=prior['seed'], authority_sha256=pilot_authority.sha256(path),
        parent_sha256={n:pilot_authority.sha256(parent/n) for n in
            ('policy_latest.pt', 'RUN_SUMMARY.json', 'frozen_scoring_baseline.json')},
        restored_state_checks=verify_runtime.restored_state_checks,
        before_seconds_per_update=[r['optimizer_update_s'] for r in prior['updates']],
        environment=authority.ENV)
    original_restore = policy14_recovery.restore_looplong
    original_update = DreamerTrainer.update

    def restore(trainer, *args, **kwargs):
        result = original_restore(trainer, *args, **kwargs)
        from wms.control.dreamer import _rng_state
        if not continuation.equal_state(_rng_state(trainer.device), payload['rng']):
            raise ValueError('WMFAST RNG was not restored exactly')
        if payload.get('sampler') is not None and not continuation.equal_state(
                trainer.start_states.state_dict(), payload['sampler']):
            raise ValueError('WMFAST TRAIN sampler was not restored exactly')
        proof.update(rng_restored_exact=True, sampler_restored_exact=True)
        return result

    def update(trainer, *args, **kwargs):
        marker = Path(output)/'pilot_WMFAST_CONTINUATION_PROOF.json'
        if not marker.exists():
            if (Path(output)/'frozen_scoring_baseline.json').read_bytes() != (
                    parent/'frozen_scoring_baseline.json').read_bytes():
                raise ValueError('WMFAST frozen baseline bytes changed')
            if not proof.get('rng_restored_exact') or not all(proof['restored_state_checks'].values()):
                raise ValueError('incomplete full-resume proof')
            with marker.open('x') as stream:
                json.dump(dict(proof, baseline_bytes_identical=True), stream, indent=1)
        return original_update(trainer, *args, **kwargs)

    def gate(p, n, *, update, panel_sha256=nativeinit.PANEL_SHA256):
        return pilot_lean_gate.panel_gate(p, n, update=update, panel_sha256=panel_sha256)

    with patch.object(pilot_authority, 'verify_runtime', verify_runtime), patch.object(
            cli, 'main', continuation.continued_main(cli)), patch.object(
            nativeinit, 'panel_gate', gate), patch.object(policy14_recovery,
            'restore_looplong', restore), patch.object(DreamerTrainer, 'update', update):
        return adapter.run(path, output, exploratory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(authority.base.FLAG, action='store_true', required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('output')
    args = parser.parse_args()
    raise SystemExit(run(args.authority, args.output, args.parent, exploratory=True))


if __name__ == '__main__':
    main()
