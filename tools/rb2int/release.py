"""Bind smoke/source/exposure proofs into one immutable final execution release."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from tools.rb2int.audit import ROOT, load, sha


def make_release(root=ROOT):
    base = root/'docs/rb2int'
    reg_path = base/'EVALUATION_REGISTRATION.json'
    reg = load(reg_path)
    exposure_path = base/'EXPOSURE_B2DF7458_GPUA_V2.json'
    exposure = load(exposure_path)
    prefix_path = base/'CAUSAL_PREFIX_PROOF.json'
    prefix = load(prefix_path)
    if (not exposure['fit_separation_passed'] or not exposure['checkpoint_verified'] or
            not exposure['learned_corrections_off'] or not prefix['passed'] or
            prefix['max_absolute_prefix_error'] != 0 or exposure['registration_sha256'] != sha(reg_path)):
        raise ValueError('exact source/exposure/causal-prefix proofs required')
    baselines = {label:load(base/name) for label, name in (
        ('WM2','BASELINE_WM2_SOURCE_AUDIT.json'), ('FULL-v4','BASELINE_V4_SOURCE_AUDIT.json'))}
    bridges = {}
    for label, name in (('WM2','BRIDGE_WM2_378D7BE5_PASS.json'), ('FULL-v4','BRIDGE_V4_378D7BE5_PASS.json')):
        p = base/name
        proof = load(p)
        if not proof['passed'] or not proof['development_only'] or proof['world'] != 969420 or proof['future'] != 0 or proof['max_absolute_error'] > 1e-8:
            raise ValueError('successful fixed-development baseline bridge required')
        bridges[label] = dict(path=str(p.relative_to(root)), sha256=sha(p), proof=proof)
    canaries = {}
    for label, name in (('FULL-v5','CANARY_V5_B2DF7458_STATUS03.json'), ('FULL-v5-LP','CANARY_LP_B2DF7458_STATUS01.json')):
        p = base/name
        record = load(p)
        proof = record['receipts']['tasks/DEV_CANARY_PASS.json']
        recipe = 'wm_phys_v5.toml' if label == 'FULL-v5' else 'wm_phys_v5_lp.toml'
        if (record['alive'] or not proof['passed'] or not proof['learned_corrections_guard'] or proof['worlds'] != [969420,969421] or
                proof['K_completed'] != 1 or proof['horizon_s'] != 600 or
                proof['recipe_sha256'] != exposure['recipes'][recipe]):
            raise ValueError('successful immutable integrated canary required')
        canaries[label] = dict(path=str(p.relative_to(root)), sha256=sha(p), proof=proof)
    controller_prefix = '/root/autodl-tmp/c4/speedhead20261030/e3final/runtime/'
    controls = {p:v for p,v in baselines['WM2']['original_identity']['sources'].items() if p.startswith(controller_prefix)}
    if not controls or any(baselines['FULL-v4']['original_identity']['sources'].get(p) != v for p,v in controls.items()):
        raise ValueError('fixed external controller implementation differs across baselines')
    models = {}
    catalog = {}
    for label, audit in baselines.items():
        if not audit['passed']:
            raise ValueError('frozen original baseline audit missing')
        identity = audit['original_identity']
        binding = dict(runtime_root=audit['runtime_root'], config=audit['config'],
            config_sha256=audit['config_sha256'], wm_sha256=audit['wm_sha256'], variant=audit['variant'],
            candidate_sha256=identity['candidate_sha256'],export_sha256=identity['export_sha256'],
            sources=dict(audit['semantic_frozen_sources'], **controls))
        models[identity['model']] = binding
        catalog[label] = dict(identity_name=identity['model'], binding=binding,
            candidate_sha256=identity['candidate_sha256'], export_sha256=identity['export_sha256'],
            switches=identity['switches'], baseline_bridge=bridges[label])
    actor = baselines['WM2']['original_identity']
    runtime = exposure['root']
    for label, recipe, identity_name in (
            ('FULL-v5','wm_phys_v5.toml','WM-PHYS-FULL-v5'),
            ('FULL-v5-LP','wm_phys_v5_lp.toml','WM-PHYS-FULL-v5-LP')):
        binding = dict(runtime_root=runtime, config=runtime+'/configs/'+recipe,
            config_sha256=exposure['recipes'][recipe], wm_sha256=actor['wm_sha256'], variant='FULL',
            candidate_sha256=actor['candidate_sha256'],export_sha256=actor['export_sha256'],
            sources=dict(exposure['source_manifest'], **controls))
        models[identity_name] = binding
        catalog[label] = dict(identity_name=identity_name, binding=binding,
            candidate_sha256=actor['candidate_sha256'], export_sha256=actor['export_sha256'], switches={},
            baseline_bridge=canaries[label])
    return dict(schema='rb2int-final-release/1', status='READY', exposure_audit_passed=True,
        execution_purposes=['final_comparison'], registration_sha256=sha(reg_path),
        source_model_commit='b2df7458a41f5f4d3a048532bc40d8d4435ec381',
        original_model_config=dict(path=baselines['WM2']['config'],sha256=baselines['WM2']['config_sha256']),
        models=models, catalog=catalog, K=8, arms=reg['arms'], horizons_s=reg['horizons_s'],
        final_comparison_worlds=reg['final_comparison_worlds'], primary_worlds=reg['replicated_primary_worlds'],
        source_exposure_proof=dict(path=str(exposure_path.relative_to(root)),sha256=sha(exposure_path)),
        causal_prefix_proof=dict(path=str(prefix_path.relative_to(root)),sha256=sha(prefix_path)),
        baseline_bridges=bridges, canaries=canaries,
        historical_aggregate_exposure_disclosed=True, historically_untouched_claim=False,
        final_outcomes_used_for_candidate_selection=False,
        qualification='engineering execution release; distributional and switching verdicts pending')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if not out.parent.is_dir() or out.exists():
        raise ValueError('exclusive final release receipt required')
    release = make_release()
    release.update(released_at=datetime.now().astimezone().isoformat(), auditor_sha256=sha(__file__))
    with out.open('x', encoding='utf-8') as stream:
        json.dump(release, stream, indent=1)
    print(json.dumps(dict(receipt=str(out),sha256=sha(out),status='READY',models=list(release['catalog'])),indent=1))


if __name__ == '__main__':
    main()
