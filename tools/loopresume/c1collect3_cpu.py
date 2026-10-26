"""S-C1COLLECT3 static checks of collector-resolved inputs; never starts SUMO."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

BASE = Path('/home/ubuntu/c4/main14/restart20261026')
OLD = BASE/'runtime'
NEW = BASE/'runtime_c1'
TOKENS = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
          'hetdata_truth', 'vtype', 'assignment')


def digest(path):
    path = Path(path)
    for name in (str(path).lower(), str(path.resolve()).lower()):
        assert not any(t in name for t in TOKENS), str(path)
        assert not name.endswith(('.rou.xml', '.rou.xml.gz')), str(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(cases):
    os.chdir(NEW)
    sys.path[:0] = [str(NEW/'src'), str(NEW)]
    os.environ['PATH'] = '/home/ubuntu/miniconda3/bin:/usr/local/bin:/usr/bin:/bin:/home/ubuntu/.local/bin'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    report = {'status': 'FAIL', 'zero_sumo': True, 'charged_sim_s': 0, 'assets': {},
              'worlds': cases, 'protected_assets_opened': False,
              'checker_change': 'Only collector-resolved direct-path assets are required; '
              'migration manifest installed separately from original git blob, not a direct-path dependency.'}
    try:
        import traffic_shepherd.c4.collect as collector
        from traffic_shepherd.config import CONFIG_PATH, load_config
        from traffic_shepherd.damping import CirclesController
        from traffic_shepherd.c4.seeds import check_case
        from wms.control.policy import load_policy_fn
        cfg = load_config()
        report['collector_import'] = {'path': collector.__file__, 'sha256': digest(collector.__file__)}
        # load_config resolves the ordinary network directly; its migration fallback is unused.
        for path in (CONFIG_PATH, cfg.model_path, cfg.network_path):
            report['assets'][str(path)] = digest(path)
        assert digest(cfg.model_path) == cfg.model_sha256
        assert os.path.samefile(cfg.scenario_template_path, OLD/'scenarios/idm/driver_routes.rou.xml')
        report['protected_template'] = {'same_original': True, 'opened': False,
                                        'path': str(cfg.scenario_template_path)}
        stage = BASE/'preflight_c1collect3'
        bindings = json.loads((stage/'BINDINGS.json').read_text())
        for rel in ('policy.pt', 'native_gate_result.json', 'wm/wm_s.pt', 'wm/config.json',
                    'REGISTRATION.json', 'wm/physics.json', 'wm/TRAIN_RECEIPT.json', 'wm/lane_calib.json'):
            path = stage/rel
            if rel.startswith('wm/') and rel not in ('wm/wm_s.pt', 'wm/config.json') and not path.exists():
                continue
            report['assets'][str(path)] = digest(path)
        assert digest(stage/'policy.pt') == bindings['policy_sha256']
        assert digest(stage/'wm/wm_s.pt') == bindings['wm_sha256']
        assert digest(stage/'wm/config.json') == bindings['wm_config_sha256']
        controller = CirclesController(cfg)
        report['onnx'] = {'loaded': True, 'providers': controller._session.get_providers(),
                          'model_sha256': cfg.model_sha256}
        exe = load_policy_fn(stage/'policy.pt', device='cpu')
        report['learned_policy_loaded'] = True
        report['residual_on_native'] = bool(exe.policy.residual_on_native)
        binary = shutil.which('sumo')
        assert binary, 'SUMO binary missing from configured PATH'
        report['assets'][binary] = digest(binary)  # Hash only, no SUMO execution.
        assert [case['seed'] for case in cases] == list(range(968157, 968170))
        for case in cases:
            check_case(case['seed'], int(case['demand']))
            assert 0 <= float(case['penetration']) <= 1
        report['prior_parity_sha256'] = digest(BASE/'C1COLLECT2_PARITY.json')
        report['manifest_install'] = json.loads((BASE/'C1COLLECT3_MANIFEST.json').read_text())
        assert report['manifest_install']['verified']
        report['status'] = 'PASS'
    except Exception as error:
        report['error'] = repr(error)
    (BASE/'C1COLLECT3_PREFLIGHT.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return report
