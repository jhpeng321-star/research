"""Meaningful CPU proofs for changed estimator, dense invariance and recipe identity."""
import json
from types import SimpleNamespace

import pytest
import torch

from tools.candB import pilot_lean as lean, pilot_lean_authority as authority


def test_memory_bound_accounts_for_peers_and_contexts():
    gib=1024**3
    assert lean.memory_bound(48*gib,2*gib,9.2*gib,2,.28)<45*gib
    with pytest.raises(ValueError,match='including peers'):
        lean.memory_bound(48*gib,7*gib,9.2*gib,2,.28)


def test_worker_executed_stats_use_final_acting_ticks_and_transport(tmp_path,monkeypatch):
    from tools.candB import pilot_lean_worker as worker
    from wms.control import execution
    import atexit
    root=tmp_path/'EXPLORATORY-LEAN'/'future_workers'/'pool'
    (root/'0').mkdir(parents=True)
    monkeypatch.setattr(worker,'_state',dict(totals=None,minima=None,registered=False))
    monkeypatch.setattr(atexit,'register',lambda *args: None)
    monkeypatch.setenv('WMS_SPEED_FUTURE_PARENT_PID','test')
    monkeypatch.setattr(worker.sys,'argv',['worker',str(root),'0'])
    monkeypatch.setattr(execution,'executed_residual',lambda frame,*args,**kw: torch.zeros_like(frame['s_m']))
    frame=dict(s_m=torch.zeros(1,2),v_mps=torch.zeros(1,2),present=torch.ones(1,2,dtype=torch.bool),
        is_cav=torch.ones(1,2,dtype=torch.bool),lane=torch.zeros(1,2,dtype=torch.long))
    result=torch.zeros(1,2,4); result[0,:,3]=torch.tensor([-1.,-100.])
    worker.observe(frame,result,torch.tensor([[True,False]]))
    worker.flush()
    row=json.loads((root/'0'/'executed.json').read_text())
    assert row['count']==row['nonzero']==1 and row['mean_braking']==1
    assert row['min_executed_acceleration_mps2']==-1
    total=worker.aggregate([dict(role='unperturbed_forward',workers=[dict(executed=row)]),
        dict(role='perturbed_ES',workers=[dict(executed=dict(row,count=9999))])])
    assert total['count']==1 and total['fraction_nonzero']==1


def test_default_off_and_bad_recipe(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='explicit'):
        lean.run('unused', 'unused')
    panel = tmp_path/'panel.json'
    panel.write_text(json.dumps(dict(starts=[dict(index=i, meta=dict(world=f'train{i}')) for i in range(16)])))
    doc = dict(inputs=dict(gate_panel=str(panel)), input_sha256=dict(gate_panel='sha'))
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    recipe = authority.recipe(doc)
    assert [s['index'] for s in recipe['panel_starts']] == [0,1,2,3]
    assert recipe['es_future_indices'] == [0,1,2,3]
    assert recipe['dense_futures'] == recipe['scoring_futures'] == 8
    doc['lean_recipe'] = recipe
    monkeypatch.setattr(authority.base, 'verify', lambda *args, **kwargs: doc)
    assert authority.verify('unused', exploratory=True) is doc
    doc['lean_recipe']['es_future_indices'] = [4,5,6,7]
    with pytest.raises(ValueError, match='identity'):
        authority.verify('unused', exploratory=True)


def test_es_truncation_preserves_original_tape_and_first_crn_boundaries():
    tape = [object() for _ in range(9)]
    items = [dict(tape_end=i+1, marker=object()) for i in range(8)]
    trainer = SimpleNamespace(_eslong_context=dict(tape=tape, cache=dict(tape_prefix=1, items=items)))
    def actual(t, imagined, weights, scale):
        assert t._eslong_context['tape'] is tape
        assert t._eslong_context['cache']['items'] == items[:4]
        return (), {}
    lean.es_prefix_gradient(actual, trainer, {}, {}, 1.)
    assert len(items) == 8


def test_real_k4_es_update_preserves_dense_k8_gradients_and_rng(tmp_path, monkeypatch):
    from tests.unit.test_eslong import make
    from tests.unit.test_longmem import assert_tree_close
    from wms.control import eslong
    from wms.control.dreamer import _rng_state
    from tools.candB.rankcheck import measurement_prefix
    baseline = make(tmp_path, monkeypatch)
    actual = make(tmp_path, monkeypatch)
    torch.manual_seed(49)
    full = baseline.update()
    full_rng = _rng_state(baseline.device)
    original_gradient, original_pair = eslong.gradient, eslong.paired_loss_difference
    seen = []
    def paired(t, plus, minus, *args, **kwargs):
        seen.append((plus.clone(), minus.clone()))
        assert plus.shape[0] == minus.shape[0] == 4
        with measurement_prefix(4):
            value, receipt = original_pair(t, plus, minus, *args, **kwargs)
            reverse, _ = original_pair(t, minus, plus, *args, **kwargs)
            assert value == -reverse
            return value, receipt
    monkeypatch.setattr(eslong, 'paired_loss_difference', paired)
    monkeypatch.setattr(eslong, 'gradient', lambda *args, **kwargs:
        lean.es_prefix_gradient(original_gradient, *args, **kwargs))
    torch.manual_seed(49)
    reduced = actual.update()
    assert len(seen) == 2
    assert reduced['eslong_perturbed_forwards'] == 16
    assert reduced['grad_norm_dense'] == full['grad_norm_dense']
    assert reduced['lean_dense_futures'] == 8
    assert actual.config.future_samples == 8
    assert_tree_close(full_rng, _rng_state(actual.device), 0, 0)
    assert torch.isfinite(torch.tensor(reduced['actor_grad_norm']))


@pytest.mark.parametrize('dense,stopped', [(-1., False), (0., True), (float('nan'), True)])
def test_lean_rows_initial_summary_and_original_phase10_gate(tmp_path, monkeypatch, dense, stopped):
    from wms.control import policy14_cli as cli
    from wms.control.dreamer import DreamerTrainer
    out = tmp_path/'pilot_EXPLORATORY-LEAN'
    path = tmp_path/'authority.json'
    path.write_text('{}')
    doc = dict(source_commit='commit', inputs=dict(current_policy_driver='test'), lean_recipe=dict(panel_starts=[]),
        input_sha256={k:'sha' for k in ('current_policy','boundary','pairing_receipt')})
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    monkeypatch.setenv('C4_EXECUTION_DRIVER', 'test')
    for key, value in authority.ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(authority, 'verify', lambda *args, **kwargs: doc)
    monkeypatch.setattr(authority.base, 'verify', lambda *args, **kwargs: doc)
    monkeypatch.setattr(lean.base, 'command', lambda *args: [])
    monkeypatch.setattr(DreamerTrainer, 'load_checkpoint', lambda *args, **kwargs: None)
    def update(t):
        t.update_count += 1
        return dict(actor_loss=1.)
    monkeypatch.setattr(DreamerTrainer, 'update', update)
    values = iter([0.] + [dense]*10)
    monkeypatch.setattr(cli, 'independent_score', lambda *args, **kwargs:
        dict(raw_dJ_vs_CTRL0_dense=next(values), raw_dJ_vs_CTRL0_long=0.))
    def fake_main(args):
        out.mkdir()
        trainer = SimpleNamespace(update_count=191, save_checkpoint=lambda path: None)
        DreamerTrainer.load_checkpoint(trainer, 'unused')
        assert json.loads((out/'RUN_SUMMARY.json').read_text())['label'] == authority.LABEL
        batch = SimpleNamespace(state=dict(s_m=torch.zeros(1,1)))
        cli.independent_score(trainer, batch)
        for phase in range(1,11):
            row = DreamerTrainer.update(trainer)
            row['independent_J'] = cli.independent_score(trainer, batch)
            row = json.loads(json.dumps(row))
            assert row['label'] == authority.LABEL and row['phase_update'] == phase
        assert row['R4']['passed'] is not stopped
        return 0
    monkeypatch.setattr(cli, 'main', fake_main)
    assert lean.run(str(path), str(out), exploratory=True) == 0
    assert (out/'STOP').exists() is stopped
