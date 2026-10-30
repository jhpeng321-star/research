"""CPU verification for the isolated owner-approved evidence waiver."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from tools.candB import pilot_authority as a, pilot_cli
from wms.control import gated300


def test_shared_staging_excludes_only_named_rankcheck_working_edits():
    from tools.candB.pilot_bundle import check_shared_status
    assert check_shared_status(b' M tools/candB/rankcheck.py\n') == ['tools/candB/rankcheck.py']
    with pytest.raises(ValueError, match='unexpected dirty'):
        check_shared_status(b' M tools/candB/pilot_cli.py\n')


def test_default_off():
    with pytest.raises(ValueError, match='explicit'):
        a.build({}, 'unused')
    with pytest.raises(ValueError, match='explicit'):
        a.verify('unused')
    with pytest.raises(ValueError, match='explicit'):
        pilot_cli.run('unused', 'unused', 'pilot')


def test_authority_identity_and_drift(tmp_path, monkeypatch):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    monkeypatch.setattr(a, 'inspect', lambda inputs: {})
    monkeypatch.setattr(a, 'source_commit', lambda root: 'committed')
    sources = {'pilot.py': 'original'}
    monkeypatch.setattr(a, 'manifest', lambda: dict(sources))
    artifact = tmp_path/'input'
    artifact.write_text('original')
    out = tmp_path/'authority.json'
    inputs = {key: str(artifact) for key in ('wm', 'current_policy', 'boundary', 'pairing_receipt')}
    doc = a.build(inputs, out, exploratory=True)
    assert doc['policy_authorized'] is False and doc['qualification_passed'] is False
    assert a.verify(out, exploratory=True)['label'] == 'EXPLORATORY'
    sources['pilot.py'] = 'drift'
    with pytest.raises(ValueError, match='runtime/config/source'):
        a.verify(out, exploratory=True)
    sources['pilot.py'] = 'original'
    artifact.write_text('drift')
    with pytest.raises(ValueError, match='input identity'):
        a.verify(out, exploratory=True)
    assert json.loads(out.read_text())['training_launched'] is False


def test_waiver_activates_exact_ctrl0_k8_mix_but_never_claims_qualification(tmp_path, monkeypatch):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    cct = tmp_path/'historical.json'
    cct.write_text('{}')
    cfg = gated300.LongCreditConfig(enabled=True, mode='gain_only', ctrl0_reference=True,
        score_training_horizon=True, require_objective_match=True, estimator='antithetic',
        cct2_file=str(cct))
    trainer = SimpleNamespace(world_model_ids=['wm'], config=SimpleNamespace(future_samples=8,
        pessimistic_credit=True, deterministic_crn=True, entrant_control=True, entrant_learned_mix=True),
        policy=SimpleNamespace(residual_reference='ctrl0'))
    with monkeypatch.context() as m:
        m.setattr(gated300, '_install_gain_only', pilot_cli.install_exploratory)
        receipt = gated300.install(trainer, asdict(cfg), [a.WM_SHA])
        assert receipt['H_long'] == 300 and receipt['qualification_passed'] is False
        assert receipt['policy_authorized'] is False and receipt['label'] == 'EXPLORATORY'
        assert trainer._long_credit_reference is gated300.ctrl0_reference_totals
        trainer.config.entrant_learned_mix = False
        with pytest.raises(ValueError, match='identity'):
            gated300.install(trainer, asdict(cfg), [a.WM_SHA])
    with pytest.raises(ValueError, match='pooled CCT-2'):
        gated300.install(trainer, asdict(cfg), [a.WM_SHA])


def test_config_identity_refuses_mixed_law_or_k(monkeypatch):
    from tools.candA.authority import effective_config
    from pathlib import Path
    raw = effective_config(Path.cwd(), a.ENTRY)
    a.check_identity(raw)
    raw['trainer']['future_samples'] = 4
    with pytest.raises(ValueError, match='identity'):
        a.check_identity(raw)


@pytest.mark.parametrize('dense,stopped', [(-1., False), (0., True)])
def test_cli_adapter_stamps_rows_and_summary_and_applies_R4(tmp_path, monkeypatch, dense, stopped):
    import torch
    from wms.control import policy14_cli as cli, policy14, execution
    from wms.control.dreamer import DreamerTrainer
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    path = tmp_path/'authority.json'
    path.write_text('{}')
    out = tmp_path/'pilot_EXPLORATORY'
    doc = dict(source_commit='commit', inputs={}, input_sha256={
        k: 'hash' for k in ('current_policy', 'boundary', 'pairing_receipt')})
    monkeypatch.setattr(a, 'verify', lambda *args, **kw: doc)
    monkeypatch.setattr(pilot_cli, 'command', lambda *args: [])
    monkeypatch.setattr(DreamerTrainer, 'update', lambda *args: dict(actor_loss=1.))
    scores = iter([0.] + [dense]*10)
    # A large non-acting output must never enter final executed-residual stats.
    def fake_controller(*args, **kw):
        def callback(frame):
            return torch.tensor([[[0., 0., 0., -1.], [0., 0., 0., -100.]]])
        callback.native_lane_mask = lambda frame: frame['present']
        return callback
    monkeypatch.setattr(policy14, 'make_cav_controller', fake_controller)
    monkeypatch.setattr(execution, 'executed_residual', lambda frame, *args, **kw: torch.zeros_like(frame['s_m']))
    def fake_score(*args, **kw):
        frame = dict(s_m=torch.zeros(1, 2), v_mps=torch.zeros(1, 2),
            present=torch.ones(1, 2, dtype=torch.bool), is_cav=torch.ones(1, 2, dtype=torch.bool),
            lane=torch.zeros(1, 2, dtype=torch.long))
        callback = policy14.make_cav_controller(act_mask=torch.tensor([[True, False]]))
        assert callback.native_lane_mask(frame).all()
        callback(frame)
        return dict(raw_dJ_vs_CTRL0_dense=next(scores), raw_dJ_vs_CTRL0_long=0.)
    monkeypatch.setattr(cli, 'independent_score', fake_score)
    batch = SimpleNamespace(state={'s_m': torch.zeros(1, 2)})
    def fake_cli(args):
        out.mkdir()
        cli.independent_score(None, batch)
        rows = []
        for _ in range(10):
            row = DreamerTrainer.update(None)
            row['independent_J'] = cli.independent_score(None, batch)
            rows.append(json.loads(json.dumps(row)))
        assert all(row['label'] == 'EXPLORATORY' for row in rows)
        assert all(row['independent_J']['executed_residual']['count'] == 1 for row in rows)
        assert all(row['independent_J']['executed_residual']['mean_braking'] == 1 for row in rows)
        assert rows[-1]['R4']['passed'] is not stopped
        summary = json.loads(json.dumps(dict(status='completed', updates=rows)))
        assert summary['label'] == 'EXPLORATORY' and not summary['deployment_authorized']
        return 0
    monkeypatch.setattr(cli, 'main', fake_cli)
    old_install = gated300._install_gain_only
    assert pilot_cli.run(path, str(out), 'pilot', exploratory=True) == 0
    assert (out/'STOP').exists() is stopped
    assert gated300._install_gain_only is old_install
