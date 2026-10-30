"""Owner2240: SHA-bound ordinary continuation, without repeating phase baselines.

Only operational adapter branches are specialized. Scientific trainer, rollout,
config, objective and source manifest remain those independently verified by the
original exploratory authority. Exact replacements refuse unexpected source.
"""
import argparse
import copy
import hashlib
import inspect
import json
from pathlib import Path
import types
from unittest.mock import patch


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('continuation source contract changed: ' + old[:80])
    return source.replace(old, new, 1)


def continuation_args(original, parent):
    result = []
    index = 0
    while index < len(original):
        key = original[index]
        if key == '--new-phase':
            index += 1
        elif key in ('--resume-allow', '--prep-dry-run-updates'):
            index += 2
        elif key == '--resume':
            result.extend((key, str(parent / 'policy_latest.pt')))
            index += 2
        else:
            result.append(key)
            index += 1
    return result + ['--continue-run', str(parent)]


def validate_parent(parent):
    prior = json.loads((parent / 'RUN_SUMMARY.json').read_text())
    frozen = json.loads((parent / 'frozen_scoring_baseline.json').read_text())
    if (prior.get('label') != 'EXPLORATORY' or prior.get('status') != 'failed'
            or prior.get('error') != 'RuntimeError: own-run STOP requested; checkpoint saved'
            or frozen != prior.get('frozen_scoring_baseline')):
        raise ValueError('continuation requires owned clean STOP and original frozen baseline')
    rows = prior['updates']
    origin = prior.get('operational_recovery', {}).get('phase_start', prior['resumed_from']['update_count'])
    if not rows or rows[-1]['update'] - origin != rows[-1]['phase_update']:
        raise ValueError('original phase clock mismatch')
    for row in rows:
        if row.get('label') != 'EXPLORATORY':
            raise ValueError('unlabelled prior update')
    if any(frozen['independent'][k] != 0. for k in
           ('raw_dJ_vs_CTRL0_dense', 'raw_dJ_vs_CTRL0_long')):
        raise ValueError('original frozen zero != CTRL0')
    return prior, origin


def prepare_base(parent):
    import torch
    from tools.candB import pilot_cli as original
    prior, origin = validate_parent(parent)
    payload = torch.load(parent / 'policy_latest.pt', map_location='cpu', weights_only=True)
    if payload['update_count'] != prior['updates'][-1]['update']:
        raise ValueError('checkpoint is not the receipted clean update boundary')
    source = inspect.getsource(original)
    source = replace_once(source, "doc['input_sha256']['current_policy'],\n               boundary_sha256=",
                          "authority.sha256(parent / 'policy_latest.pt'),\n               boundary_sha256=")
    source = replace_once(source, 'state = dict(capture=None, phase_update=0, baseline=None, trainer=None, began=None)',
        "state = dict(capture=None, phase_update=prior['updates'][-1]['update']-origin, "
        "baseline=prior['frozen_scoring_baseline']['independent'], trainer=None, began=None)")
    # A same-runtime checkpoint binds the current LF driver, not the historical C1 driver.
    source = replace_once(source, "os.environ['C4_EXECUTION_DRIVER'] = doc['inputs']['current_policy_driver']",
                          "os.environ['C4_EXECUTION_DRIVER'] = current")
    module = types.ModuleType('tools.candB.pilot_continued_adapter')
    module.__dict__.update(parent=parent, prior=prior, origin=origin)
    exec(compile(source, original.__file__, 'exec'), module.__dict__)
    command = module.command
    module.command = lambda doc, out, mode: continuation_args(command(doc, out, mode), parent)
    verify = module.authority.verify_runtime

    def verify_runtime(doc, raw, **kwargs):
        trainer = kwargs.pop('trainer', None)
        verify(doc, raw, trainer=None, **kwargs)
        if trainer is not None:
            from dataclasses import asdict
            from wms.control.dreamer import DreamerConfig
            from wms.control.gated300 import LongCreditConfig
            # Full resume must restore the exact saved config (including entropy annealing).
            saved = payload['config']
            if (asdict(trainer.config) != asdict(DreamerConfig(**saved))
                    or trainer.policy.residual_reference != 'ctrl0'
                    or trainer.gated300['H_long'] != 300
                    or trainer.gated300['config'] != asdict(LongCreditConfig(**raw['long_credit']))):
                raise ValueError('same-phase saved runtime/config/mix identity mismatch')
            from tools.distill.runtime import check_metadata
            check_metadata(trainer.start_states.metadata.values())
        return doc

    return module, verify_runtime, prior


def continued_main(cli):
    """Reuse the already-computed phase baseline, not a new checkpoint baseline."""
    source = inspect.getsource(cli.main)
    for old, new in (
        ('if switches.get("zero_residual_output", False):',
         'if switches.get("zero_residual_output", False) and not recovery:'),
        ("ranked.get('current_policy_sha256') != sha256(args.resume)",
         "(recovery['checkpoint_sha256'] if recovery else ranked.get('current_policy_sha256')) != sha256(args.resume)"),
        ('before = independent_score(trainer, rescore_batch)',
         "before = (copy.deepcopy(recovery['updates'][-1]['independent_J']) if recovery else "
         'independent_score(trainer, rescore_batch))'),
        ('initial_native = score_native_panel(trainer, val, gate_panel, args.device)',
         "initial_native = (copy.deepcopy(recovery['curve'][-1]) if recovery else "
         'score_native_panel(trainer, val, gate_panel, args.device))'),
        ('panel_before = score_gate_panel(trainer, val, gate_panel, args.device)',
         "panel_before = (list(recovery['prior']['gate_panel']['J_before']) if recovery else "
         'score_gate_panel(trainer, val, gate_panel, args.device))'),
    ):
        # These bootstrap statements also occur in the separate WM-swap branch.
        if old.startswith(('before =', 'initial_native =', 'panel_before =')):
            if source.count(old) != 2:
                raise ValueError('unexpected bootstrap/swap source contract')
            source = source.replace(old, new, 1)
        else:
            source = replace_once(source, old, new)
    namespace = dict(cli.__dict__, copy=copy)
    exec(compile(source, cli.__file__, 'exec'), namespace)
    return namespace['main']


def run(authority, output, parent, *, exploratory=False):
    from tools.candB import pilot_authority, pilot_recipe
    from wms.control import policy14_cli as cli
    pilot_authority.require_opt_in(exploratory)
    parent = Path(parent).resolve()
    base, verify_runtime, prior = prepare_base(parent)
    proof = dict(label='EXPLORATORY', same_phase=True, baseline_recomputed=False,
        phase_origin=prior['resumed_from']['update_count'],
        saved_update=prior['updates'][-1]['update'], seed=prior['seed'],
        parent=str(parent), authority_sha256=pilot_authority.sha256(authority),
        parent_sha256={name: pilot_authority.sha256(parent / name) for name in
            ('policy_latest.pt', 'RUN_SUMMARY.json', 'frozen_scoring_baseline.json')},
        overlay_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        retained='full actor/Adam/RNG/sampler; frozen phase position; original update10 gate')
    with patch.object(pilot_authority, 'verify_runtime', verify_runtime), patch.object(
            cli, 'main', continued_main(cli)):
        result = pilot_recipe.run(authority, output, exploratory=True, base_module=base,
                                  parallel_initial=True, continuation_proof=proof)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exploratory-owner-20261030', action='store_true', required=True)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--parent', required=True)
    parser.add_argument('output')
    args = parser.parse_args()
    raise SystemExit(run(args.authority, args.output, args.parent, exploratory=True))


if __name__ == '__main__':
    main()
