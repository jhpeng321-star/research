"""Named-artifact/source audit, without opening any final outcome or truth tape."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import tomllib


def safe(path):
    path = Path(path)
    if any(token in str(path).lower() for token in (
            'sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
            'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')):
        raise ValueError('prohibited audit path')
    return path


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(safe(path).read_text(encoding='utf-8'))


def audit(root, *, checkpoint=None):
    root = safe(root)
    registration_path = root/'docs/rb2int/EVALUATION_REGISTRATION.json'
    registration = read(registration_path)
    final = set(registration['final_comparison_worlds'])
    allowed = set(range(969400, 969432))
    paths = [
        'artifacts/rbfit/shrinkage_v1/idm_shrinkage.json',
        'artifacts/rbfit/lane_raw_c5602102891e/lc2013_population.json',
        'docs/paper_results/rb2cal/v1/fit/calibration.json',
        'artifacts/rb2lphys/learned_v1.json']
    artifacts = []
    for name in paths:
        data = read(root/name)
        fit, selected = data['fit_worlds'], data.get('selection_worlds', [])
        if (not fit or any(isinstance(s, bool) or not isinstance(s, int)
                           for s in fit+selected) or not set(fit+selected) <= allowed or
                set(fit+selected) & final):
            raise ValueError('declared fit/selection exposure overlaps final or unapproved worlds: '+name)
        artifacts.append(dict(path=name, sha256=sha(root/name), fit_worlds=fit,
                              selection_worlds=selected, final_overlap=[]))
    pop = root/'artifacts/rbfit/shrinkage_v1/idm_population.json'
    shrink = read(root/paths[0])
    cal = read(root/paths[2])
    lp = read(root/paths[3])
    if shrink['idm_population_sha256'] != sha(pop):
        raise ValueError('inherited IDM fit lineage hash differs')
    if (cal['idm_population_sha256'] != sha(pop) or
            cal['shrinkage_sha256'] != sha(root/paths[0]) or
            cal['arrivals_sha256'] != sha(root/'docs/paper_results/rbunc/observed_arrivals.json') or
            cal['learned_corrections'] is not False or lp['frozen_idm_population_sha256'] != sha(pop)):
        raise ValueError('CAL/LP inherited artifact lineage differs')
    package = read(root/'artifacts/rbfit/lane_raw_c5602102891e/PACKAGE.json')
    if (package['artifact_sha256']['lc2013_population.json'] != sha(root/paths[1]) or
            package['saved_driver_id_lookup'] is not False or package['population_refit'] is not False):
        raise ValueError('LC artifact/identity continuation lineage differs')
    base = tomllib.loads((root/'configs/wm_phys_v5.toml').read_text())
    lp_recipe = tomllib.loads((root/'configs/wm_phys_v5_lp.toml').read_text())
    normalized = json.loads(json.dumps(lp_recipe))
    normalized['physics']['physics_params'] = 'off'
    normalized['physics'].pop('physics_params_file')
    if normalized != base:
        raise ValueError('paired candidates differ outside declared LP parameters')
    for recipe in (base, lp_recipe):
        if not recipe['model']['residual_accel_off'] or not recipe['model']['temporal_accel_off'] or recipe['model']['lane_head'] != 'off':
            raise ValueError('learned model correction enabled')
        for key in ('training', 'learned_acceleration_noise', 'learned_admission',
                    'learned_lane_corrections', 'learned_merge_corrections', 'future_observation_inputs'):
            if recipe['inference'][key] is not False:
                raise ValueError('learned/future input switch enabled: '+key)
    checkpoint_verified = checkpoint is not None and sha(checkpoint) == 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
    if checkpoint is not None and not checkpoint_verified:
        raise ValueError('frozen WM2 encoder/checkpoint differs')
    source_paths = sorted((root/'src/wms').rglob('*.py'))
    for directory in ('tools/rbint', 'tools/rblane', 'tools/rbfit'):
        source_paths += sorted((root/directory).glob('*.py'))
    source_paths += [root/p for p in paths]
    source_paths += [pop, root/'artifacts/rbfit/shrinkage_v1/idm_posteriors.json.gz',
                     root/'docs/paper_results/rbunc/observed_arrivals.json',
                     root/'configs/execparity_cav_idm_v1.json', root/'configs/native_cav_head_v5.json']
    return dict(schema='rb2int-exposure-audit/1', audited_at=datetime.now().astimezone().isoformat(),
        root=str(root), stage=read(root/'INT_STAGE.json') if (root/'INT_STAGE.json').is_file() else None,
        registration_sha256=sha(registration_path), artifacts=artifacts,
        direct_declared_fit_selection_overlap=[], fit_separation_passed=True,
        checkpoint_verified=checkpoint_verified, learned_corrections_off=True,
        paired_recipe_only_lp_difference=True,
        recipes={name: sha(root/'configs'/name) for name in ('wm_phys_v5.toml', 'wm_phys_v5_lp.toml')},
        source_manifest={str(p): sha(p) for p in source_paths},
        worker_versions=dict(LANE='6b4bfbd5', CAL='7c1cf53c', BREAK='116b2147', LPHYS='f6cf52ab'),
        historical_exposure='Prior aggregates exposed; this is not a historically untouched panel.',
        lineage_basis='Delivered named artifact declarations and inherited population/source hashes; worker handoffs retained in intake receipts.',
        lp_support_note='Declared conditional upperbound900s; support audit found actual converted data only through600s.',
        final_outcomes_opened=False, new_fit=False, final_inference_authorized=False,
        remaining='successful integrated full-five-arm smoke, baseline identity/source bridge and final exclusive dispatch release')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--host', choices=('gpuA', 'gpuR'))
    parser.add_argument('--checkpoint')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if not out.parent.is_dir() or out.exists():
        raise ValueError('exclusive audit receipt in existing parent required')
    if args.host:
        from tools.rbint.remote import ssh
        # Execute this exact trusted local auditor in-memory; frozen stages are
        # never patched to add a new tool after a model smoke has started.
        code = Path(__file__).read_text(encoding='utf-8')
        program = f'''
import json,sys
sys.path[:0]=[{args.root!r}, {(args.root.rstrip('/')+'/src')!r}]
namespace={{'__name__':'rb2_exposure_inline','__file__':{(args.root.rstrip('/')+'/tools/rb2int/exposure.py')!r}}}
exec(compile({code!r},'<trusted INT auditor>','exec'),namespace)
print(json.dumps(namespace['audit']({args.root!r},checkpoint={args.checkpoint!r})))
'''
        result = json.loads(ssh(args.host, program, timeout=60))
    else:
        result = audit(args.root, checkpoint=args.checkpoint)
    result.update(auditor_sha256=sha(__file__), host=args.host)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(dict(receipt=str(out), sha256=sha(out),
        fit_separation_passed=result['fit_separation_passed'], checkpoint_verified=result['checkpoint_verified'],
        recipes=result['recipes'], sources=len(result['source_manifest']), final_inference_authorized=False), indent=1))


if __name__ == '__main__':
    main()
