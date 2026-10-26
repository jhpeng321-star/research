"""Zero-SUMO runtime parity and static preflight, including protected-file nonaccess."""
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


def protected(p):
    text = str(p).lower()
    return any(t in text for t in TOKENS) or text.endswith(('.rou.xml', '.rou.xml.gz'))


def digest(p):
    assert not protected(p) and not protected(p.resolve()), str(p)
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for data in iter(lambda: stream.read(1 << 20), b''):
            h.update(data)
    return h.hexdigest()


def parity(install=False):
    report = {'missing': [], 'different': [], 'equal': 0, 'protected_metadata_only': [],
              'source_runtime': str(OLD), 'successor_runtime': str(NEW), 'copied': []}
    # Walk symlinked input trees too; prune protected names before descending/opening.
    def walk(src, rel=Path('.'), ancestors=()):
        if protected(src) or protected(src.resolve()):
            report['protected_metadata_only'].append(str(rel))
            return
        resolved = str(src.resolve())
        if src.is_dir():
            if resolved in ancestors:
                return
            for child in sorted(src.iterdir()):
                if child.name == '__pycache__' or child.suffix == '.pyc':
                    continue
                if rel == Path('.') and child.name == 'runs':
                    report['generated_runs_metadata_only'] = True
                    continue
                walk(child, rel/child.name, (*ancestors, resolved))
            return
        if not src.is_file():
            report.setdefault('unresolved', []).append(str(rel))
            return
        dst = NEW/rel
        if protected(dst) or protected(dst.resolve()):
            report['protected_metadata_only'].append(str(rel))
            return
        sha = digest(src)
        if not dst.is_file():
            row = {'path': str(rel), 'sha256': sha, 'bytes': src.stat().st_size}
            report['missing'].append(row)
            if install:
                dst.parent.mkdir(parents=True, exist_ok=True)
                assert not dst.exists(), str(dst)
                shutil.copy2(src, dst)
                assert digest(src) == sha == digest(dst)
                report['copied'].append(row)
        elif digest(dst) != sha:
            report['different'].append({'path': str(rel), 'original_sha256': sha,
                                        'successor_sha256': digest(dst)})
        else:
            report['equal'] += 1
    walk(OLD)
    # Protected scenario template is already an original-file link: metadata only.
    template = NEW/'scenarios/idm/driver_routes.rou.xml'
    original = OLD/'scenarios/idm/driver_routes.rou.xml'
    report['template_identity'] = {'path': str(template), 'same_original':
                                   template.exists() and os.path.samefile(template, original),
                                   'opened': False}
    out = BASE/('C1COLLECT2_PARITY.json' if install else 'C1COLLECT2_PARITY_PLAN.json')
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({**{k: v for k, v in report.items() if k not in ('missing', 'protected_metadata_only', 'copied')},
                      'missing_count': len(report['missing']),
                      'missing_bytes': sum(r['bytes'] for r in report['missing']),
                      'missing_preview': report['missing'][:30], 'copied_count': len(report['copied']),
                      'protected_count': len(report['protected_metadata_only']),
                      'receipt': str(out), 'receipt_sha256': digest(out)}))


def preflight(cases):
    os.chdir(NEW)
    sys.path[:0] = [str(NEW/'src'), str(NEW)]
    os.environ['PATH'] = '/home/ubuntu/miniconda3/bin:/usr/local/bin:/usr/bin:/bin:/home/ubuntu/.local/bin'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    report = {'status': 'FAIL', 'zero_sumo': True, 'charged_sim_s': 0, 'assets': {},
              'worlds': cases, 'protected_assets_opened': False}
    try:
        import traffic_shepherd.c4.collect as collector
        from traffic_shepherd.config import CONFIG_PATH, load_config
        from traffic_shepherd.damping import CirclesController
        from traffic_shepherd.c4.seeds import check_case
        from wms.control.policy import load_policy_fn
        cfg = load_config()
        for p in (CONFIG_PATH, cfg.model_path, cfg.network_path,
                  NEW/'assets/research/migration_manifest.json'):
            report['assets'][str(p)] = digest(p)
        assert report['assets'][str(cfg.model_path)] == cfg.model_sha256
        assert os.path.samefile(cfg.scenario_template_path, OLD/'scenarios/idm/driver_routes.rou.xml')
        report['protected_template'] = {'same_original': True, 'opened': False,
                                         'path': str(cfg.scenario_template_path)}
        stage = BASE/'preflight_c1collect2'
        for p in sorted(stage.rglob('*')):
            if p.is_file():
                report['assets'][str(p)] = digest(p)
        bindings = json.loads((stage/'BINDINGS.json').read_text())
        assert digest(stage/'policy.pt') == bindings['policy_sha256']
        assert digest(stage/'wm/wm_s.pt') == bindings['wm_sha256']
        assert digest(stage/'wm/config.json') == bindings['wm_config_sha256']
        # Instantiate the exact adopted ONNX session and deployment checkpoint loader.
        controller = CirclesController(cfg)
        report['onnx'] = {'loaded': True, 'providers': controller._session.get_providers(),
                           'model_sha256': cfg.model_sha256}
        exe = load_policy_fn(stage/'policy.pt', device='cpu')
        report['learned_policy_loaded'] = exe.policy.residual_on_native
        report['collector_import'] = {'path': collector.__file__, 'sha256': digest(Path(collector.__file__))}
        binary = shutil.which('sumo')
        assert binary, 'SUMO binary missing from configured PATH'
        report['assets'][binary] = digest(Path(binary))  # Never execute SUMO.
        for case in cases:
            assert 968157 <= case['seed'] <= 968169
            check_case(case['seed'], int(case['demand']))
            assert 0 <= float(case['penetration']) <= 1
        parity_doc = json.loads((BASE/'C1COLLECT2_PARITY.json').read_text())
        for row in parity_doc['copied']:
            assert digest(NEW/row['path']) == row['sha256']
        expected_differences = {'BOX_MANIFEST.json', 'scripts/c4/loop14_fast.py',
            'src/wms/control/paircrit.py', 'src/wms/control/real_share.py',
            'tests/unit/test_loopresume.py', 'tools/loopresume/AMENDMENT_C1.json',
            'tools/loopresume/box.py', 'tools/loopresume/operate.py'}
        assert {row['path'] for row in parity_doc['different']} <= expected_differences
        assert not parity_doc.get('unresolved')
        assert parity_doc['template_identity']['same_original']
        report['status'] = 'PASS'
    except Exception as error:
        report['error'] = repr(error)
    out = BASE/'C1COLLECT2_PREFLIGHT.json'
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    return report


if __name__ == '__main__':
    parity('--install' in sys.argv)
