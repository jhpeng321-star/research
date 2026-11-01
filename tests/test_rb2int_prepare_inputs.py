import json
import tarfile

import numpy as np
import pytest
import torch

from tools.rb2int.prepare_inputs import build, package, prefix_bins, sha


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding='utf-8')


def fixture(root):
    source = root/'source'
    job = source/'s969550_ctrl0'
    job.mkdir(parents=True)
    start = dict(seed=969550, t0=300, initial_state_sha256='initial',
        state=dict(present=torch.tensor([[True]])), past_entries=[dict(time=1, lane=0)],
        observed_queue_history=np.array([[t, 0, 0] for t in np.arange(.5, 300.5, .5)]))
    torch.save(start, job/'start.pt')
    kin = np.zeros((241, 1, 7), dtype=np.float32)
    kin[..., 0], kin[..., 1], kin[..., 4] = -100, 1, 20
    np.savez(job/'start_history.npz', kin=kin, mask=np.ones((241, 1), bool))
    for arm in ('native', 'ctrl0', 'mappo', 'pi', 'circles'):
        put(source/f's969550_{arm}/result.json', dict(
            metrics=dict(initial_state_sha256='initial', FUTURE_DELAY_MUST_NOT_LEAK=999999),
            output_hashes={n: sha(job/n) for n in ('start.pt', 'start_history.npz')},
            clocks={'600': {'team_delay_vs': 999999}}))
    metadata = root/'metadata.json'
    put(metadata, dict(worlds=[dict(seed=969550, demand=2400, penetration=.2)]))
    registration = root/'registration.json'
    put(registration, dict(final_comparison_worlds=[969550], metadata_sha256=sha(metadata)))
    return registration, metadata, source


def test_export_is_causal_and_removes_ancillary_result_outcomes(tmp_path, monkeypatch):
    reg, metadata, source = fixture(tmp_path)
    # Avoid making the test depend on transient runner disk capacity.
    monkeypatch.setattr('tools.rb2int.prepare_inputs.shutil.disk_usage', lambda p: type('D', (), dict(free=2**40))())
    out = tmp_path/'inputs'
    result = build(reg, metadata, source, out, [969550])
    assert result['passed'] and result['final_outcomes_exported'] is False
    receipt = (out/'data/jobs/s969550_ctrl0/result.json').read_text()
    assert '999999' not in receipt and 'clocks' not in receipt
    prefix = json.loads((out/'prefix_observables.json').read_text())
    assert len(prefix['episodes']) == 5
    assert prefix['episodes'][0]['bins']['upstream']['speed'] == [20.]*6
    assert 'clocks' not in prefix['episodes'][0]
    archive = tmp_path/'prefix.tar'
    packed = package(out, archive)
    assert packed['sha256'] == sha(archive)
    with tarfile.open(archive) as stream:
        assert all(not m.issym() and not m.islnk() for m in stream.getmembers())
    with pytest.raises(ValueError, match='new input root'):
        build(reg, metadata, source, out, [969550])


def test_changed_metadata_or_arm_warmup_stops_before_forecasting(tmp_path, monkeypatch):
    reg, metadata, source = fixture(tmp_path)
    monkeypatch.setattr('tools.rb2int.prepare_inputs.shutil.disk_usage', lambda p: type('D', (), dict(free=2**40))())
    path = source/'s969550_pi/result.json'
    obj = json.loads(path.read_text())
    obj['metrics']['initial_state_sha256'] = 'different'
    put(path, obj)
    with pytest.raises(ValueError, match='observable warmup'):
        build(reg, metadata, source, tmp_path/'incomplete_preserved', [969550])
    metadata.write_text('{}')
    with pytest.raises(ValueError, match='metadata registration'):
        build(reg, metadata, source, tmp_path/'never_created', [969550])


def test_prefix_excludes_t0_and_never_claims_future_frames():
    kin = np.zeros((241, 1, 7), dtype=np.float32)
    kin[..., 0], kin[..., 1], kin[..., 4] = -100, 1, 20
    kin[240, :, 4] = 999999
    assert prefix_bins(dict(kin=kin, mask=np.ones((241, 1), bool)))['upstream']['speed'] == [20.]*6
    with pytest.raises(ValueError, match='causal history'):
        prefix_bins(dict(kin=kin[:-1], mask=np.ones((240, 1), bool)))
