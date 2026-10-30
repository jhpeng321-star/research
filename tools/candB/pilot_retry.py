"""Explicit original zero191 checkpoint recovery; no baseline forwards or new phase.

Path-scoped port of c4/candBL c73498cd and its authority-rebind repair2ac8ff65.
MAIN03:00 supersedes the earlier wait-for-update1 WMFAST restriction for B.
"""
import argparse
from contextlib import ExitStack
from functools import partial
import inspect
import json
from pathlib import Path
import types
from unittest.mock import patch

from tools.candB import pilot_authority, pilot_continue, pilot_lean_authority
from tools.candB import pilot_retry_authority as authority, pilot_wmfast

PARENT = Path('/root/autodl-tmp/c4/candB_pilot_run_WMFAST_EXPLORATORY-LEAN_20261031/pilot_EXPLORATORY-LEAN')
OLD_RUNTIME = Path('/root/autodl-tmp/c4/candB_pilot_WMFAST_EXPLORATORY-LEAN_20261031')
BASELINE = '744b30b809af60fb11acc54734cbc29bf92f6163793cd306b040ee26e430d680'
CHECKPOINT = 'abae6464e2ef0fbb08174b7d4b079b4bed634bbae7207ec21560f5d3f7873c02'


def validate_parent(parent):
    if parent != PARENT:
        raise ValueError('retry is restricted to the recorded failed original B phase')
    prior = json.loads((parent/'RUN_SUMMARY.json').read_text())
    frozen = json.loads((parent/'frozen_scoring_baseline.json').read_text())
    if (prior.get('label') != authority.LABEL or prior.get('status') != 'failed'
            or prior.get('error') != "TypeError: 'NoneType' object is not subscriptable"
            or prior.get('updates') != [] or prior.get('seed') != 965140
            or prior['resumed_from']['update_count'] != 191 or not prior.get('zero_start')
            or frozen != prior.get('frozen_scoring_baseline')
            or frozen['update'] != 191 or frozen['scoring_phase_position'] != 191
            or len(frozen['panel']) != 4
            or any(frozen['independent'][k] != 0. for k in
                   ('raw_dJ_vs_CTRL0_dense','raw_dJ_vs_CTRL0_long'))):
        raise ValueError('retry original zero-progress/phase/CTRL0 baseline mismatch')
    for name, digest in (('frozen_scoring_baseline.json',BASELINE),('policy_native_best.pt',CHECKPOINT)):
        if pilot_authority.sha256(parent/name) != digest:
            raise ValueError('retry original artifact drift: '+name)
    return prior, 191


def prepare_base(parent):
    # Reuse the independently checked full-resume implementation, specializing
    # only its completed-progress assumptions for this explicitly bound failure.
    source = inspect.getsource(pilot_continue.prepare_base)
    source = source.replace("'policy_latest.pt'", "'policy_native_best.pt'")
    source = pilot_continue.replace_once(source,
        "payload['update_count'] != prior['updates'][-1]['update']", "payload['update_count'] != origin")
    source = pilot_continue.replace_once(source,
        "prior['updates'][-1]['update']-origin", "0")
    namespace = dict(pilot_continue.__dict__, validate_parent=validate_parent)
    namespace['continuation_args'] = lambda args, p: [
        str(p/'policy_native_best.pt') if value == str(p/'policy_latest.pt') else value
        for value in pilot_continue.continuation_args(args,p)]
    exec(compile(source,__file__,'exec'),namespace)
    return namespace['prepare_base'](parent)


def validate_identity(doc, prior):
    old_path = OLD_RUNTIME/'authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json'
    if pilot_authority.sha256(old_path) != prior['exploratory_authority_sha256']:
        raise ValueError('original baseline authority drift')
    old = json.loads(old_path.read_text())
    if (old['input_sha256'] != doc['input_sha256'] or old['config_sha256'] != doc['config_sha256']
            or old['lean_recipe'] != pilot_lean_authority.recipe(doc)):
        raise ValueError('original baseline input/config/seed/panel/CRN mismatch')
    original = json.loads((OLD_RUNTIME/'SOURCE_MANIFEST.json').read_text())
    current = json.loads(Path('SOURCE_MANIFEST.json').read_text())
    drift = {n for n,h in original.items() if n.startswith(('src/','configs/','traffic_shepherd/'))
             and current.get(n) != h}
    if drift != {'src/wms/control/imagination_memory.py','src/wms/control/policy14_recovery.py'}:
        raise ValueError('unexpected baseline scientific source drift: '+str(sorted(drift)))
    # The only scientific-module changes are the EXACT B-L snapshot/retry repairs.
    for name in drift:
        previous = (OLD_RUNTIME/name).read_text()
        if name.endswith('imagination_memory.py'):
            marker = "        for cell in fn.__closure__ or ():\n"
            insert = """        # Explicitly observational wrappers keep run-wide telemetry in shared
        # closure cells. Snapshot/release ONLY the underlying physical controller;
        # clearing wrapper cells would corrupt its factory and the next replay.
        passthrough = getattr(fn, '_wms_checkpoint_passthrough', None)
        if passthrough is not None:
            visit(passthrough)
            return
"""
            expected = pilot_continue.replace_once(previous,marker,insert+marker)
        else:
            expected = pilot_continue.replace_once(previous,'def load_continuation(raw, args):',
                'def load_continuation(raw, args, *, initial_checkpoint=False):')
            expected = pilot_continue.replace_once(expected,
                "    if (prior['status'] != 'failed' or prior['seed'] != args.seed\n"
                "            or not prior['updates'] or prior['updates'][-1]['update'] != saved\n"
                "            or not start < saved < start + args.max_updates):",
                "    progress = (not prior['updates'] and saved == start if initial_checkpoint else\n"
                "                bool(prior['updates']) and prior['updates'][-1]['update'] == saved\n"
                "                and start < saved < start + args.max_updates)\n"
                "    if (prior['status'] != 'failed' or prior['seed'] != args.seed or not progress):")
        if Path(name).read_text() != expected:
            raise ValueError('source is not the exact reviewed B-L repair: '+name)


def run(path, output, *, exploratory=False):
    import torch
    from wms.control import policy14_cli as cli, policy14_recovery, nativeinit, eslong
    from wms.control.dreamer import DreamerTrainer, _rng_state
    from tools.candB import pilot_lean, pilot_lean_gate
    pilot_authority.require_opt_in(exploratory)
    doc = authority.verify(path,exploratory=True)
    if eslong.stack_sides() != 4:
        raise ValueError('owner03:00 retry must use bound S4 WMFAST-B')
    prior, origin = validate_parent(PARENT)
    validate_identity(doc,prior)
    base, verify_runtime, prior = prepare_base(PARENT)
    payload = torch.load(PARENT/'policy_native_best.pt',map_location='cpu',weights_only=True)
    if payload['update_count'] != origin or not payload.get('sampler') or not payload.get('rng'):
        raise ValueError('missing original phase-zero state')
    source = inspect.getsource(pilot_lean)
    source = pilot_continue.replace_once(source,
        'from tools.candB import pilot_cli as base, pilot_lean_authority as authority',
        'from tools.candB import pilot_retry_authority as authority')
    adapter = types.ModuleType('tools.candB.pilot_retry_lean_adapter')
    adapter.__dict__.update(base=base)
    exec(compile(source,pilot_lean.__file__,'exec'),adapter.__dict__)
    restore_original, update_original = policy14_recovery.restore_looplong, DreamerTrainer.update
    proof = dict(label=authority.LABEL,same_phase=True,baseline_recomputed=False,
        phase_origin=origin,saved_update=origin,seed=prior['seed'],original_R4_phase_update=10,
        parent=str(PARENT),baseline_sha256=BASELINE,checkpoint_sha256=CHECKPOINT,
        authority_sha256=pilot_authority.sha256(path),source_commit=doc['source_commit'],
        input_sha256=doc['input_sha256'],config_sha256=doc['config_sha256'],
        environment=authority.ENV,restored_state_checks=verify_runtime.restored_state_checks)

    def restore(trainer,*args,**kwargs):
        result = restore_original(trainer,*args,**kwargs)
        checks = dict(rng=pilot_continue.equal_state(_rng_state(trainer.device),payload['rng']),
            sampler=pilot_continue.equal_state(trainer.start_states.state_dict(),payload['sampler']),
            horizon=pilot_continue.equal_state(trainer.horizon_curriculum.as_dict(),payload['looplong']['horizon']),
            long_config=pilot_continue.equal_state(trainer.gated300['config'],payload['gated300']['config']))
        if not all(checks.values()):
            raise ValueError('initial recovery state drift: '+str(checks))
        proof.update(restored_extra_checks=checks)
        return result

    def update(trainer,*args,**kwargs):
        marker = Path(output)/'pilot_RETRY_CONTINUATION_PROOF.json'
        if not marker.exists():
            target = Path(output)/'frozen_scoring_baseline.json'
            if json.loads(target.read_text()) != prior['frozen_scoring_baseline']:
                raise ValueError('recovery baseline content changed')
            target.write_bytes((PARENT/'frozen_scoring_baseline.json').read_bytes())
            if (pilot_authority.sha256(target) != BASELINE or trainer.update_count != origin
                    or not proof.get('restored_extra_checks')
                    or not all(proof['restored_state_checks'].values())):
                raise ValueError('incomplete original-zero-state proof')
            trainer.save_checkpoint(Path(output)/'policy_phase_zero.pt')
            with marker.open('x') as stream:
                json.dump(dict(proof,baseline_bytes_identical=True),stream,indent=1)
        return update_original(trainer,*args,**kwargs)

    with ExitStack() as stack:
        for obj,name,value in ((pilot_authority,'verify_runtime',verify_runtime),
                (cli,'main',pilot_wmfast.continued_main(cli)),
                (policy14_recovery,'load_continuation',partial(policy14_recovery.load_continuation,initial_checkpoint=True)),
                (policy14_recovery,'restore_looplong',restore),(DreamerTrainer,'update',update),
                (nativeinit,'panel_gate',pilot_lean_gate.panel_gate)):
            stack.enter_context(patch.object(obj,name,value))
        return adapter.run(path,output,exploratory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(pilot_authority.FLAG,action='store_true',required=True)
    parser.add_argument('--authority',required=True)
    parser.add_argument('output')
    args = parser.parse_args()
    raise SystemExit(run(args.authority,args.output,exploratory=True))


if __name__ == '__main__':
    main()
