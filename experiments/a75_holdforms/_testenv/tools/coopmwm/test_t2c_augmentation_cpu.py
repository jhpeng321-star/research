from copy import deepcopy
import pytest

from tools.coopmwm.test_t2c_runtime_cpu import prepare
from tools.coopmwm.rerun_tracks_runtime import Runtime
from tools.coopmwm.rerun_tracks_bank import validate_augmented, training_batch
from tools.coopmwm.exact_grade import _seal


def test_partial_admission_never_uses_unchecked_or_negative_proposals(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend)
    original = backend.verify_augmentation
    calls = []
    def verify(payload, **kw):
        calls.append(payload['proposal_id'])
        result = original(payload, **kw)
        return result if len(calls) == 1 else dict(examples=[], receipt=dict(status='rejected', advantage=-1))
    backend.verify_augmentation = verify
    bank = backend.bank_examples('B', 0, cfg=cfg)
    admitted = rt.prepare_augmentation('B', bank)
    assert len(calls) == 2 and len(admitted) == 1
    assert admitted[0]['id'].startswith(calls[0])
    assert admitted[0]['weight'] == 1.4
    assert admitted[0]['effective_sample']['action'] == [.4, .4]
    assert rt.prepare_augmentation('B', bank) is admitted and len(calls) == 2
    assert len(training_batch(bank, admitted, arm='B', encoder=rt.components['B']['merge_features'])) == 25


@pytest.mark.parametrize('fault', ['missing', 'copied', 'negative', 'state', 'seed', 'identity',
                                   'seed_with_claim', 'effective', 'weight'])
def test_augmented_evidence_is_bound_to_successful_execution(tmp_path, monkeypatch, fault):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend)
    bank = backend.bank_examples('B', 0, cfg=cfg)
    e = deepcopy(rt.prepare_augmentation('B', bank)[0])
    if fault == 'missing': e.pop('augmentation_admission')
    if fault == 'copied': e['plan'] = deepcopy(bank[0]['plan'])
    if fault == 'state': e['snapshot_file_sha256'] = '0'*64
    if fault == 'seed': e['seed'] += 1
    if fault == 'identity': e['snapshot_identity_sha256'] = '0'*64
    if fault == 'seed_with_claim':
        e['seed'] += 1
        e['augmentation_admission']['seed'] = e['seed']
    if fault == 'effective': e['effective_sample']['action'][0] += .1
    if fault == 'weight': e['weight'] *= 2
    if fault == 'negative':
        rec = e['augmentation_admission']['confirmation']
        rec['per_seed_v1']['examples'][0]['advantage_veh_s'] = -1
        e['augmentation_admission']['confirmation'] = _seal(
            {k: v for k, v in rec.items() if k != 'confirmation_sha256'}, 'confirmation_sha256')
    with pytest.raises(ValueError, match='all8 admission'):
        validate_augmented(e)


def test_exhausted_allowance_refuses_before_dispatch_without_shortening(tmp_path, monkeypatch):
    cfg, reg, backend = prepare(tmp_path, monkeypatch)
    rt = Runtime(cfg, reg, backend)
    def excessive(payload, *, execute, **kw):
        execute([(dict(sim_s=40000, window=payload['window']), {})], 'too_large')
    backend.verify_augmentation = excessive
    with pytest.raises(ValueError, match='allowance exhausted'):
        rt.prepare_augmentation('B', backend.bank_examples('B', 0, cfg=cfg))
    assert not backend.branch_calls and not rt.augmented
    assert rt.spend['augment_verify'] == 0


@pytest.mark.parametrize('gain,admitted', [(1., True), (-1., False)])
def test_production_boundary_uses_real_all8_both_seed_confirmation(tmp_path, monkeypatch, gain, admitted):
    from pathlib import Path
    import types
    import torch
    from tools.coopmwm import rerun_tracks_augment as aug, rerun_cpus as rc, sumo_verify as sv
    from tools.coopmwm.test_verify_perseed_cpu import SeedPlant
    from tools.coopmwm.test_replay_bank_confirm_cpu import snapshot, plan
    snap, p = snapshot(600), plan('modified', .2)
    path = tmp_path/'snapshot.pt'
    torch.save(snap, path)
    payload = dict(id='aug0', proposal_id='variant', arm='B', plan=p,
        snapshot_path=str(path), snapshot_file_sha256=rc.sha(path),
        seed=snap['episode']['world']['seed'], window=dict(start_s=600, end_s=1200))
    spent, windows = [], []
    def execute(pairs, suffix):
        out = []
        for spec, state in pairs:
            spent.append(spec['sim_s'])
            windows.append(spec['window'])
            work = Path(spec['t2c_artifact_root']).parent
            plant = SeedPlant(work, {('modified', 1): gain, ('modified', 2): gain})
            result = sv.run_branch(spec, state, plant)
            result['result_path'] = str(work/'branches'/spec['verification_id']/spec['branch']/'RESULT.json')
            out.append(result)
        return out
    records = []
    monkeypatch.setattr(aug, 'materialise', lambda backend, payload, cfg, record, *a:
                        records.append(record) or [])
    result = aug.verify(types.SimpleNamespace(), payload, dict(run_root=str(tmp_path)), execute)
    record = records[0]
    assert record['verify_mode'] == 'all8_both_seeds'
    assert {e['seed_index'] for e in record['per_seed_v1']['examples']} == {1, 2}
    assert all(e['admitted'] is admitted for e in record['per_seed_v1']['examples'])
    assert sum(spent) <= rc.GATE_SIM_S
    assert all(w['end_s'] == 1200 for w in windows)
