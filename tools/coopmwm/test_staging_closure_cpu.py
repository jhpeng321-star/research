"""Production-reader parity on synthetic files only; no simulation or GPU."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path

import pytest
import torch

from tools.coopmwm import replay_bank as rb
from tools.coopmwm.accel_storage import read as read_result
from tools.coopmwm.staging_closure import closure


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(gzip.compress(data, mtime=0) if path.suffix == '.gz' else data)
    return path


def result(directory, value, compressed=True):
    path = directory/'RESULT.json'
    if not compressed:
        return put(path, value)
    data = json.dumps(value, sort_keys=True).encode()
    put(path, dict(encoding='coopmwm-lossless-gzip-v1', file='RESULT.json.gz',
                   sha256=hashlib.sha256(data).hexdigest(), uncompressed_bytes=len(data)))
    (directory/'RESULT.json.gz').write_bytes(gzip.compress(data, mtime=0))
    return path


def directory_link(source, target):
    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError:
        if os.name != 'nt':
            raise
        # Windows junctions exercise Path.resolve through directory reparse
        # points without requiring the symlink privilege on this CPU host.
        import _winapi
        _winapi.CreateJunction(str(source), str(target))


def ls_source(root, generation='', remote_episode=False):
    src = root/'home/ubuntu/collection'
    episode = root/'home/ubuntu/warm1/jobs/episode1'
    world = dict(seed=986001)
    ep = dict(id='episode1', world=world, prefixes={'300': 'prefix'},
              steps=[dict(time_s=299, sample='past')])
    result(episode, ep)
    label = put(src/'labels'/('search'+generation+'.json'), dict(search_id='search',
        verification=dict(verification_id='sv_ls', status='accepted',
                          branches=[dict(branch='policy'), dict(branch='plan')],
                          episode=dict(arm='B'))))
    pending = put(src/'pending/search.json.gz', dict(snapshot='snapshot1',
        episode_path='/home/ubuntu/warm1/jobs/episode1' if remote_episode else str(episode)))
    snap = dict(t0=300, episode=dict(id='episode1', world=world), current_sample='now')
    stream = io.BytesIO()
    torch.save(snap, stream)
    snapshot = src/'snapshots/snapshot1.pt.gz'
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(gzip.compress(stream.getvalue(), mtime=0))
    branches = src/'label_attempts'/generation[1:]/'branches' if generation else src/'branches'
    for branch in ('policy', 'plan'):
        result(branches/'sv_ls'/branch, dict(steps=[]))
        trace = branches/'sv_ls'/branch/'observable.jsonl.gz'
        trace.write_bytes(gzip.compress(b'{}\n', mtime=0))
    return src, label, pending, snapshot, episode


@pytest.mark.parametrize('generation', ['', '_SUMOENV1'])
def test_ls_transitive_files_match_actual_readers_and_generation_layout(tmp_path, generation):
    src, label, pending, snapshot, episode = ls_source(tmp_path, generation)
    receipts = list(rb._receipts(tmp_path/'home/ubuntu/warm1', 'B', [src]))
    receipt, branches, version = receipts[0]
    assert version == 'u3'
    restored = rb._decision_snapshot(tmp_path/'home/ubuntu/warm1', 'B',
        dict(receipt, _source=str(src), _branches=str(branches), _policy_version=version), {})
    assert restored['samples'] == {'299': 'past', '300': 'now'}
    files, manifest = closure([label])
    expected = {label, pending, snapshot, episode/'RESULT.json', episode/'RESULT.json.gz'}
    for branch in ('policy', 'plan'):
        expected.update(branches.joinpath('sv_ls', branch, name) for name in
                        ('RESULT.json', 'RESULT.json.gz', 'observable.jsonl.gz'))
    assert set(files) == expected
    assert manifest['file_count'] == manifest['physical_file_count'] == 11
    assert not any('SNAPSHOT.pt' in str(p) for p in files)


def test_filesystem_root_mirror_maps_remote_episode_without_accessing_host(tmp_path):
    src, label, pending, snapshot, episode = ls_source(tmp_path, remote_episode=True)
    files, manifest = closure([label])
    assert episode/'RESULT.json.gz' in files
    assert all(tmp_path in p.parents for p in files)
    assert closure([label], mirror=tmp_path) == (files, manifest)


def test_own_decision_link_matches_real_snapshot_reader_and_never_unpickles(tmp_path, monkeypatch):
    bank = rb.bank_root(tmp_path, 'B')
    link = dict(verification_id='sv_own', state_id='state1')
    decision = put(bank/'decisions/state1.json', link)
    snapshot = bank/'snapshots/state1.pt'
    snapshot.parent.mkdir(parents=True)
    torch.save(dict(observable='own'), snapshot)
    assert rb._decision_snapshot(tmp_path, 'B', link, {'sv_own': link}) == dict(observable='own')
    def forbidden(*args, **kwargs):
        raise AssertionError('closure must not deserialize snapshots')
    monkeypatch.setattr(torch, 'load', forbidden)
    files, _ = closure([decision])
    assert set(files) == {decision, snapshot}


def test_warm1_receipt_view_closes_external_symlink_target_and_gzip(tmp_path):
    view = tmp_path/'warm1/replay_bank/B/confirm_solo_replication'
    source = tmp_path/'solo_source/verification/branches/sv_source'
    for branch in ('policy', 'candidate1'):
        result(source/branch, dict(branch=branch))
    (view/'branches').mkdir(parents=True)
    directory_link(source, view/'branches/sv_view')
    receipt = put(view/'verifications/sv_view.json', dict(status='accepted',
        verification_id='sv_view', branches=[dict(branch='policy'), dict(branch='candidate1')]))
    files, manifest = closure([receipt])
    alias = view/'branches/sv_view/policy/RESULT.json'
    physical = source/'policy/RESULT.json'
    assert alias in files and physical in files
    assert alias.with_suffix('.json.gz') in files and physical.with_suffix('.json.gz') in files
    assert manifest['aliases'][alias.as_posix()] == physical.as_posix()
    assert physical.parent.as_posix() in manifest['allowed_roots']
    assert manifest['file_count'] == 9 and manifest['physical_file_count'] == 5
    assert read_result(alias) == dict(branch='policy')


def test_manifest_is_deterministic_and_hashes_physical_bytes(tmp_path):
    _, label, pending, _, _ = ls_source(tmp_path)
    files, manifest = closure([label, pending, label])
    assert closure([pending, label]) == (files, manifest)
    body = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    encoded = json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    assert hashlib.sha256(encoded).hexdigest() == manifest['manifest_sha256']
    for path in files:
        entry = manifest['entries'][path.as_posix()]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
        assert path.stat().st_size == entry['bytes']
    old_sha = manifest['manifest_sha256']
    snapshot = next(p for p in files if p.name.endswith('.pt.gz'))
    snapshot.write_bytes(b'opaque snapshot replacement')
    assert closure([label, pending])[1]['manifest_sha256'] != old_sha


@pytest.mark.parametrize('missing', ['pending/search.json.gz', 'snapshots/snapshot1.pt.gz',
                                     'episode_marker', 'episode_payload', 'branch_marker'])
def test_missing_transitive_input_refuses_instead_of_silently_skipping(tmp_path, missing):
    src, label, _, _, episode = ls_source(tmp_path)
    path = {'episode_marker': episode/'RESULT.json',
            'episode_payload': episode/'RESULT.json.gz',
            'branch_marker': src/'branches/sv_ls/plan/RESULT.json'}.get(missing, src/missing)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        closure([label])


def test_result_marker_digest_and_size_checked_against_real_reader(tmp_path):
    marker = result(tmp_path/'episode', dict(steps=[]))
    assert read_result(marker) == dict(steps=[])
    body = json.loads(marker.read_text())
    put(marker, dict(body, sha256='0'*64))
    with pytest.raises(ValueError, match='digest mismatch'):
        closure([marker])
    with pytest.raises(ValueError, match='digest mismatch'):
        read_result(marker)
    put(marker, dict(body, uncompressed_bytes=0))
    with pytest.raises(ValueError, match='size mismatch'):
        closure([marker])
    put(marker, dict(body, file='../escape.gz'))
    with pytest.raises(ValueError, match='payload path'):
        closure([marker])


@pytest.mark.parametrize('value', ['../outside', 'a/b', 'a\\b', '', '..', 'C:escape'])
def test_dependency_identifiers_cannot_escape_layout(tmp_path, value):
    pending = put(tmp_path/'pending/search.json.gz', dict(snapshot=value, episode_path=str(tmp_path)))
    with pytest.raises(ValueError, match='invalid snapshot'):
        closure([pending])


@pytest.mark.parametrize('token', ['sealed', 'holdout', 'heldout', 'cal_split', 'chunk5',
                                  'j4b', 'inc1', 'hetdata_truth', 'vType', 'driver', 'vehicle_params'])
def test_prohibited_paths_are_rejected_before_any_file_open(tmp_path, monkeypatch, token):
    def forbidden(*args, **kwargs):
        raise AssertionError('prohibited input must never be opened')
    monkeypatch.setattr(Path, 'open', forbidden)
    with pytest.raises(ValueError, match='prohibited staging path'):
        closure([tmp_path/token/'data.json'])


def test_forbidden_envelope_reference_refused_before_snapshot_or_episode_open(tmp_path):
    pending = put(tmp_path/'pending/search.json.gz', dict(snapshot='snapshot1',
                  episode_path='/home/ubuntu/sealed/episode'))
    with pytest.raises(ValueError, match='prohibited staging path'):
        closure([pending])


def test_plain_result_empty_inventory_directories_and_bad_label_generation(tmp_path):
    marker = result(tmp_path/'plain', dict(steps=[]), compressed=False)
    assert closure(marker)[0] == [marker]
    with pytest.raises(ValueError, match='empty'):
        closure([])
    with pytest.raises(ValueError, match='not a file'):
        closure([tmp_path])
    label = put(tmp_path/'labels/search_bad.json', dict(search_id='search', verification={}))
    with pytest.raises(ValueError, match='generation'):
        closure([label])


def test_skipped_label_needs_no_reconstruction_inputs(tmp_path):
    label = put(tmp_path/'labels/search.json', dict(search_id='search',
                verification=dict(status='skipped')))
    assert closure([label])[0] == [label]


def test_directory_link_target_is_guarded_before_any_file_open(tmp_path, monkeypatch):
    source = tmp_path/'sealed'
    source.mkdir()
    alias = tmp_path/'visible'
    directory_link(source, alias)
    def forbidden(*args, **kwargs):
        raise AssertionError('linked prohibited target must never be opened')
    monkeypatch.setattr(Path, 'open', forbidden)
    with pytest.raises(ValueError, match='prohibited staging path'):
        closure([alias/'data.json'])
