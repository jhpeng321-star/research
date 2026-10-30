"""Recover failed pre-optimizer phase zero using its IDENTICAL durable baseline."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

from tools.candB import pilot_authority, pilot_continue, pilot_lean
from tools.candB.pilot_replay_guard import vjp_guard


def validate(doc, parent):
    prior = json.loads((parent/'RUN_SUMMARY.json').read_text())
    baseline = json.loads((parent/'frozen_scoring_baseline.json').read_text())
    if (prior.get('label') != 'EXPLORATORY-LEAN' or prior.get('updates')
            or prior.get('status') != 'failed' or prior.get('error') !=
            "TypeError: 'NoneType' object is not subscriptable" or prior.get('seed') != 965140
            or prior.get('resumed_from', {}).get('update_count') != 191
            or prior.get('frozen_scoring_baseline') != baseline):
        raise ValueError('not the bound pre-optimizer failed phase zero')
    owner = json.loads((parent.parent/'OWNERSHIP.json').read_text())
    runtime = Path(owner['runtime'])
    old_path = runtime/'authority_gen/PILOT_EXPLORATORY_LEAN_AUTHORITY.json'
    old = json.loads(old_path.read_text())
    if (pilot_authority.sha256(old_path) != prior['exploratory_authority_sha256']
            or old['input_sha256'] != doc['input_sha256']
            or old['config_sha256'] != doc['config_sha256']
            or old['lean_recipe'] != doc['lean_recipe']):
        raise ValueError('baseline input/config/seed/recipe authority mismatch')
    original = json.loads((runtime/'SOURCE_MANIFEST.json').read_text())
    current = json.loads(Path('SOURCE_MANIFEST.json').read_text())
    drift = [n for n,h in original.items() if n.startswith(('src/','configs/','traffic_shepherd/'))
        and current.get(n) != h]
    if drift:
        raise ValueError('baseline scientific source drift: '+str(drift))
    if (baseline['update'] != 191 or len(baseline['panel']) != 4
            or any(baseline['independent'][k] != 0. for k in
                ('raw_dJ_vs_CTRL0_dense','raw_dJ_vs_CTRL0_long'))):
        raise ValueError('baseline phase-zero/CTRL0/panel mismatch')
    return prior, baseline


def run(path, output, parent, *, exploratory=False):
    import copy
    import torch
    from wms.control import policy14_cli as cli
    from wms.control.dreamer import DreamerTrainer, DreamerConfig, _rng_state
    from wms.control.contract import RewardWeights
    from tools.candB import pilot_lean_authority
    pilot_authority.require_opt_in(exploratory)
    doc = pilot_lean_authority.verify(path, exploratory=True)
    parent, output = Path(parent).resolve(), Path(output).resolve()
    prior, baseline = validate(doc, parent)
    payload = torch.load(parent/'policy_native_best.pt', map_location='cpu', weights_only=True)
    if payload['update_count'] != 191:
        raise ValueError('baseline checkpoint is not phase zero')
    raw = pilot_authority.config(doc['inputs'])
    score_original, native_original, panel_original = cli.independent_score, cli.score_native_panel, cli.score_gate_panel
    update_original, dumps = DreamerTrainer.update, json.dumps
    counts = dict(independent=0,native=0,panel=0)
    checked = False

    def score(trainer, *args, **kwargs):
        if trainer.update_count != 191:
            return score_original(trainer,*args,**kwargs)
        if (asdict(trainer.config) != asdict(DreamerConfig(**raw['trainer']))
                or asdict(trainer.reward_weights) != asdict(RewardWeights(**raw['reward']))
                or not pilot_continue.equal_state(trainer.policy.state_dict(),payload['policy'])):
            raise ValueError('baseline policy/effective-config equality failed')
        trainer._gatemeas_score_position = copy.deepcopy(baseline['scoring_phase_position'])
        counts['independent'] += 1
        return copy.deepcopy(baseline['independent'])

    def native(trainer, *args, **kwargs):
        if trainer.update_count != 191:
            return native_original(trainer,*args,**kwargs)
        counts['native'] += 1
        return copy.deepcopy(prior['native_curve'][0])

    def panel(trainer, *args, **kwargs):
        if trainer.update_count != 191:
            return panel_original(trainer,*args,**kwargs)
        counts['panel'] += 1
        trainer._gatemeas_panel_scores = copy.deepcopy(baseline['panel'])
        return list(prior['gate_panel']['J_before'])

    def update(trainer, *args, **kwargs):
        nonlocal checked
        if not checked:
            if counts != dict(independent=1,native=1,panel=1):
                raise ValueError('startup did not reuse exactly the complete original baseline')
            target = output/'frozen_scoring_baseline.json'
            if json.loads(target.read_text()) != baseline:
                raise ValueError('baseline reconstruction differs; refuse any optimizer step')
            target.write_bytes((parent/'frozen_scoring_baseline.json').read_bytes())
            restored = {}
            for key in ('policy','group_encoder','critic','target_critic','actor_optimizer','critic_optimizer'):
                restored[key] = pilot_continue.equal_state(getattr(trainer,key).state_dict(),payload[key])
            if not all(restored.values()) or not pilot_continue.equal_state(_rng_state(trainer.device),payload['rng']):
                raise ValueError('pre-optimizer phase-zero state/RNG changed')
            trainer.save_checkpoint(output/'policy_phase_zero.pt')
            proof = dict(label='EXPLORATORY-LEAN',baseline_recomputed=False,
                baseline_bytes_identical=True,phase_origin=191,seed=965140,counts=counts,
                state_equal=restored,rng_equal=True,scientific_source_equal=True,
                input_sha256=doc['input_sha256'],config_sha256=doc['config_sha256'],
                parent=str(parent),parent_baseline_sha256=pilot_authority.sha256(parent/'frozen_scoring_baseline.json'),
                parent_checkpoint_sha256=pilot_authority.sha256(parent/'policy_native_best.pt'),
                authority_sha256=pilot_authority.sha256(path),update1_ES_stack_sides=0)
            with (output/'pilot_BASELINE_REUSE_PROOF.json').open('x') as stream:
                json.dump(proof,stream,indent=1)
            checked = True
        return update_original(trainer,*args,**kwargs)

    def serialize(value,*args,**kwargs):
        text = dumps(value,*args,**kwargs)
        if isinstance(value,dict) and 'actor_loss' in value and value.get('phase_update') == 1:
            (output/'STOP').touch(exist_ok=True)
        return text

    from tools.candB import pilot_lean_gate
    from wms.control import nativeinit
    with ExitStack() as stack:
        stack.enter_context(vjp_guard())
        for obj,name,replacement in ((cli,'independent_score',score),(cli,'score_native_panel',native),
                (cli,'score_gate_panel',panel),(DreamerTrainer,'update',update),(json,'dumps',serialize),
                (nativeinit,'panel_gate',pilot_lean_gate.panel_gate)):
            stack.enter_context(patch.object(obj,name,replacement))
        return pilot_lean.run(path,str(output),exploratory=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(pilot_authority.FLAG,action='store_true',required=True)
    parser.add_argument('--authority',required=True)
    parser.add_argument('--parent',required=True)
    parser.add_argument('output')
    args = parser.parse_args()
    raise SystemExit(run(args.authority,args.output,args.parent,exploratory=True))


if __name__ == '__main__':
    main()
