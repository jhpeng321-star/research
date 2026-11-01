"""Verify original completed-run identities and semantic frozen baseline sources."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

WM_SHA = 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'


def safe(path):
    p = Path(path)
    if any(x in str(p).lower() for x in ('sealed', 'holdout', 'heldout', 'cal_split',
            'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')):
        raise ValueError('prohibited baseline audit path')
    return p


def sha(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def contract(identity, label, runtime):
    expected = 'wm2' if label == 'WM2' else 'WM-PHYS-FULL-v4-gpu'
    if (identity['model'] != expected or identity['wm_sha256'] != WM_SHA or identity['K'] != 8 or
            identity['horizon'] != 600 or identity['t0'] != 300 or identity['arms'] !=
            ['native', 'ctrl0', 'mappo', 'pi', 'circles'] or not identity['mix'] or
            not identity['native_lanes'] or identity['seed'] != 965140 or identity['update'] != 190):
        raise ValueError('original frozen baseline identity differs')
    root = str(runtime).rstrip('/')+'/'
    # New orchestration intentionally replaces only the runner and input receipt.
    # Every baseline model/helper/artifact actually in its old frozen runtime
    # remains source-bound; the saved original identity is independently verified.
    selected = {p: value for p, value in identity['sources'].items() if p.startswith(root)
                and not p.endswith('/tools/rbeval/run_wm.py')}
    if not any(p.startswith(root+'src/wms/') for p in selected):
        raise ValueError('baseline physical implementation is not bound')
    return selected


def audit(identity_path, runtime, config, label):
    identity_path, runtime, config = map(safe, (identity_path, runtime, config))
    identity = json.loads(identity_path.read_text())
    selected = contract(identity, label, runtime)
    if sha(config) != identity['config_sha256']:
        raise ValueError('original baseline config changed')
    for path, expected in identity['sources'].items():
        if sha(path) != expected:
            raise ValueError('original saved-run source changed: '+path)
    checkpoint = Path('/root/autodl-tmp/c4/wmcombined_wm2/wm_s.pt')
    if sha(checkpoint) != WM_SHA:
        raise ValueError('original baseline checkpoint changed')
    candidate = Path('/root/autodl-tmp/c4/setupaudit20261030/policy_000190.pt')
    binding_path = Path('/root/autodl-tmp/c4/speedhead20261030/EXPORT_BOUND.json')
    binding = json.loads(binding_path.read_text())
    if sha(candidate) != identity['candidate_sha256'] or sha(binding['path']) != identity['export_sha256']:
        raise ValueError('original external controller artifact changed')
    return dict(schema='rb2int-frozen-baseline-audit/1', audited_at=datetime.now().astimezone().isoformat(),
        label=label, passed=True, identity_path=str(identity_path), identity_sha256=sha(identity_path),
        original_identity=identity, runtime_root=str(runtime), config=str(config),
        config_sha256=identity['config_sha256'], wm_sha256=WM_SHA,
        variant='WM2' if label == 'WM2' else 'FULL', semantic_frozen_sources=selected,
        original_source_files_verified=len(identity['sources']),
        forecast_tasks_opened=False, final_outcomes_opened=False,
        remaining='development input/orchestration parity and explicit final release; reserved tasks reused')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', choices=('gpuA', 'gpuR'), required=True)
    parser.add_argument('--model', choices=('WM2', 'FULL-v4'), required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if not out.parent.is_dir() or out.exists():
        raise ValueError('exclusive local baseline audit receipt required')
    from tools.rbint.remote import ssh
    if args.model == 'WM2':
        identity = '/root/rbeval20261031/wm2_shard0/IDENTITY.json'
        runtime = '/root/autodl-tmp/c4/setupaudit20261030/fix'
        config = '/root/autodl-tmp/c4/restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml'
    else:
        identity = '/root/rbeval20261031/int/WM-PHYS-FULL-v4-gpu/shard0/IDENTITY.json'
        runtime = '/root/rbeval20261031/phys_code_22cc32f65a80'
        config = runtime+'/configs/wm_phys.toml'
    source = Path(__file__).read_text(encoding='utf-8')
    code = f'''
import json
namespace={{'__name__':'rb2_frozen_baseline_audit'}}
exec(compile({source!r},'<trusted baseline auditor>','exec'),namespace)
print(json.dumps(namespace['audit']({identity!r},{runtime!r},{config!r},{args.model!r})))
'''
    result = json.loads(ssh(args.host, code, timeout=60))
    result.update(host=args.host, auditor_sha256=sha(__file__))
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1)
    print(json.dumps(dict(receipt=str(out), sha256=sha(out), model=args.model,
        passed=result['passed'], original_source_files_verified=result['original_source_files_verified'],
        semantic_sources=len(result['semantic_frozen_sources']), runtime_root=result['runtime_root']), indent=1))


if __name__ == '__main__':
    main()
