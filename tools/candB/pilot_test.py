"""CPU verification for the isolated owner-approved evidence waiver."""
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from tools.candB import pilot_authority as a, pilot_cli
from wms.control import gated300


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
