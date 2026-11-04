"""Real bank-reader link and whole approved inventory regressions."""
import json
import os
import shutil
from copy import deepcopy

import pytest

from tools.coopmwm import rerun_cpus as rc
from tools.coopmwm.exact_grade import _seal
from tools.coopmwm.test_rerun_cpus_cpu import mirror, prep_dir, LEGACY_REMOTE, COLLECTION_REMOTE


def junction(source, destination):
    if os.name == 'nt':
        import _winapi
        _winapi.CreateJunction(str(source), str(destination))
    else:
        destination.symlink_to(source, target_is_directory=True)


def test_real_readers_discover_second_root_and_stage_complete_resolved_bank(mirror, tmp_path):
    home = mirror/'home'/'ubuntu'
    original = home/'coopmwm20261102_warm1'/'replay_bank'/'B'
    target = home/'synthetic_second_bank'/'replay_bank'/'B'
    target.parent.mkdir(parents=True)
    shutil.move(str(original), str(target))
    junction(target, original)
    m = rc.staging_manifest(remote_root=LEGACY_REMOTE, sources={'B': [COLLECTION_REMOTE]})
    assert '/home/ubuntu/synthetic_second_bank' in m['allowed_staged_roots']
    keys = m['entries']
    for p in target.rglob('*'):
        if (p.is_file() and p.suffix in ('.pt', '.json', '.gz')
                and not p.name.endswith('.spec.json') and 'claims' not in p.parts):
            resolved_key = 'home/ubuntu/'+p.relative_to(home).as_posix()
            assert resolved_key in keys
            alias = 'home/ubuntu/coopmwm20261102_warm1/replay_bank/B/'+p.relative_to(target).as_posix()
            assert alias in keys
            assert keys[resolved_key]['sha256'] == keys[alias]['sha256'] == rc.sha(p)
    stage = tmp_path/'materialized'
    stage.mkdir()
    for key in keys:
        dest = stage.joinpath(*key.split('/'))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(mirror.joinpath(*key.split('/')), dest)
    assert rc.verify_staged(m, stage)['files'] == len(keys)


def test_guards_run_before_lexical_or_resolved_read(tmp_path, monkeypatch):
    from pathlib import Path
    opened = []
    real = Path.open
    def spy(path, *a, **kw):
        opened.append(str(path))
        return real(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', spy)
    with rc._ReadRecorder([tmp_path], tmp_path):
        with pytest.raises(ValueError, match='forbidden'):
            (tmp_path/'holdout_invalid.json').read_text()
    assert not opened
    safe = tmp_path/'safe.json'
    original = Path.resolve
    def resolve(path, *a, **kw):
        if path == safe:
            return tmp_path/'sealed'/'invalid.json'
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, 'resolve', resolve)
    with rc._ReadRecorder([tmp_path], tmp_path):
        with pytest.raises(ValueError, match='forbidden'):
            safe.read_text()
    assert not opened


def test_already_resolved_reader_data_outside_staging_home_is_refused(tmp_path):
    home = tmp_path/'home'
    home.mkdir()
    outside = tmp_path/'outside.json'
    outside.write_text('{}')
    with rc._ReadRecorder([home], home):
        with pytest.raises(ValueError, match='outside staging home'):
            outside.read_text()


def test_preflight_rejects_substituted_inventory(tmp_path, mirror):
    prep, stage = prep_dir(tmp_path, mirror)
    path = prep/rc.MANIFEST_FILE
    original = json.loads(path.read_text())
    empty = _seal(dict(home=rc.HOME, entries={}), 'manifest_sha256')
    path.write_text(json.dumps(empty))
    with pytest.raises(SystemExit, match='empty staging'):
        rc.preflight_dry(prep, stage_dir=stage)
    altered = deepcopy(original)
    altered['notes'].append('substituted but correctly resealed')
    altered = _seal({k: v for k, v in altered.items() if k != 'manifest_sha256'}, 'manifest_sha256')
    path.write_text(json.dumps(altered))
    with pytest.raises(SystemExit, match='manifest_file_sha256'):
        rc.preflight_dry(prep, stage_dir=stage)
    apppath = prep/rc.APPROVAL
    app = json.loads(apppath.read_text())
    app['manifest_file_sha256'] = rc.sha(path)
    apppath.write_text(json.dumps(app))
    with pytest.raises(SystemExit, match='full budget request'):
        rc.preflight_dry(prep, stage_dir=stage)


@pytest.mark.parametrize('role', ['common_checkpoint', 'bank_B', 'bank_R', 'safety_rule'])
def test_inventory_missing_required_entries_are_refused(mirror, role):
    m = rc.staging_manifest(remote_root=LEGACY_REMOTE)
    del m['entries'][m['roles'][role][0]]
    m = _seal({k: v for k, v in m.items() if k != 'manifest_sha256'}, 'manifest_sha256')
    with pytest.raises(SystemExit, match='required staging role'):
        rc.verify_staged(m, mirror)


def test_checkpoint_digest_and_containment_are_checked_before_reads(mirror):
    m = rc.staging_manifest(remote_root=LEGACY_REMOTE)
    with pytest.raises(SystemExit, match='checkpoint/config u0 mismatch'):
        rc.verify_staged(m, mirror, checkpoint_sha256='f'*64)
    for key in ('home/ubuntu/../../escape.json', '/absolute.json', 'home/ubuntu/C:/escape.json'):
        bad = deepcopy(m)
        bad['entries'][key] = next(iter(m['entries'].values()))
        bad = _seal({k: v for k, v in bad.items() if k != 'manifest_sha256'}, 'manifest_sha256')
        with pytest.raises(SystemExit, match='unsafe staged destination'):
            rc.verify_staged(bad, mirror)
