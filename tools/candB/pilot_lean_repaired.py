"""Correct four-start gate and persist each startup score, explicit fresh entry only.

Prepared for a separately acknowledged startup recovery; never automatically
relaunches a live or failed run or substitutes a partly computed baseline.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

from tools.candB import pilot_lean, pilot_lean_gate


def run(path, output, *, exploratory=False, failed_parent=None):
    from wms.control import nativeinit, policy14_cli as cli
    from tools.candB import pilot_authority, pilot_continue, pilot_lean_authority
    import torch
    doc = pilot_lean_authority.verify(path, exploratory=exploratory)
    prior = None
    proof = None
    if failed_parent is not None:
        parent = Path(failed_parent)
        prior = json.loads((parent/'RUN_SUMMARY.json').read_text())
        if (prior.get('label') != 'EXPLORATORY-LEAN' or prior.get('status') != 'failed'
                or prior.get('error') != 'ValueError: native gate requires 16 finite paired costs'
                or prior.get('updates') or (parent/'frozen_scoring_baseline.json').exists()
                or prior['seed'] != 965140):
            raise ValueError('not the bound incomplete lean startup')
        owner = json.loads((parent.parent/'OWNERSHIP.json').read_text())
        old_runtime = Path(owner['runtime'])
        old_path = old_runtime/'authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json'
        old_doc = json.loads(old_path.read_text())
        if (pilot_authority.sha256(old_path) != prior['exploratory_authority_sha256']
                or old_doc['input_sha256'] != doc['input_sha256']
                or old_doc['config_sha256'] != doc['config_sha256']
                or prior['resumed_from']['sha256'] != doc['input_sha256']['current_policy']):
            raise ValueError('saved initial score authority/input/config mismatch')
        old_manifest = json.loads((old_runtime/'SOURCE_MANIFEST.json').read_text())
        new_manifest = json.loads(Path('SOURCE_MANIFEST.json').read_text())
        drift = {n:dict(before=h, after=new_manifest.get(n)) for n,h in old_manifest.items()
            if n.startswith(('src/', 'configs/', 'traffic_shepherd/')) and new_manifest.get(n) != h}
        if set(drift) != {'src/wms/control/dreamer.py', 'src/wms/control/eslong.py',
                          'src/wms/control/parallel_futures.py'}:
            raise ValueError('unexpected startup scientific source drift')
        import os
        if os.environ.get('WMS_SPEED_ES_STACK_SIDES', '0') != '0':
            raise ValueError('stacked ES must wait for first completed update')
        proof = dict(label='EXPLORATORY-LEAN', startup_recovery=True,
            completed_baseline_recomputed=False, saved_independent_reused=True,
            incomplete_native_panel_scores_not_durable=True,
            parent_summary_sha256=pilot_authority.sha256(parent/'RUN_SUMMARY.json'),
            identical_input_sha256=doc['input_sha256'], seed=965140,
            scientific_diff=drift, wmfast_enabled=False)
        payload = torch.load(doc['inputs']['current_policy'], map_location='cpu', weights_only=True)
        raw = pilot_authority.config(doc['inputs'])
    original = cli.independent_score
    dumps = json.dumps
    count = 0

    def score(trainer, batch, *args, **kwargs):
        nonlocal count
        if prior is not None and count == 0 and not kwargs.get('native_reference'):
            from wms.control.dreamer import DreamerConfig
            from wms.control.contract import RewardWeights
            if (trainer.update_count != 191 or asdict(trainer.config) != asdict(DreamerConfig(**raw['trainer']))
                    or asdict(trainer.reward_weights) != asdict(RewardWeights(**raw['reward']))):
                raise ValueError('saved initial score config/phase mismatch')
            expected = dict(payload['policy'])
            for name in ('residual_head.weight', 'residual_head.bias'):
                expected[name] = torch.zeros_like(expected[name])
            if not pilot_continue.equal_state(trainer.policy.state_dict(), expected):
                raise ValueError('startup policy differs from original exact zero initialization')
            result = prior['independent_score_before']
            if any(result[k] != 0. for k in ('raw_dJ_vs_CTRL0_dense', 'raw_dJ_vs_CTRL0_long')):
                raise ValueError('saved initial score is not zero == CTRL0')
            with (Path(output)/'pilot_STARTUP_REUSE_PROOF.json').open('x') as stream:
                json.dump(dict(proof, policy_equal=True, config_equal=True, phase_origin=191), stream, indent=1)
        else:
            result = original(trainer, batch, *args, **kwargs)
        if trainer.update_count == 191:
            name = Path(output)/f'pilot_startup_score_{count}.json'
            with name.open('x') as stream:
                json.dump(dict(label='EXPLORATORY-LEAN', update=191,
                    native_reference=kwargs.get('native_reference', False), score=result),
                    stream, indent=1, allow_nan=False)
            count += 1
        return result

    def gate(p, n, *, update, panel_sha256=nativeinit.PANEL_SHA256):
        return pilot_lean_gate.panel_gate(p, n, update=update, panel_sha256=panel_sha256)

    def serialize(value, *args, **kwargs):
        text = dumps(value, *args, **kwargs)
        if isinstance(value, dict) and 'actor_loss' in value and value.get('phase_update') == 1:
            (Path(output)/'STOP').touch(exist_ok=True)
        return text

    with patch.object(cli, 'independent_score', score), patch.object(nativeinit, 'panel_gate', gate), patch.object(
            json, 'dumps', serialize):
        return pilot_lean.run(path, output, exploratory=exploratory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exploratory-owner-20261030', action='store_true', required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--failed-parent')
    parser.add_argument('output')
    args = parser.parse_args()
    raise SystemExit(run(args.authority, args.output, exploratory=True, failed_parent=args.failed_parent))


if __name__ == '__main__':
    main()
