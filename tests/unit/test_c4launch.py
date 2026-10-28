"""MAIN authority cannot masquerade as WM3 measurement; exact focus mass."""
import json
import math

import pytest
import torch

from wms.control.start_states import StartStateSampler
from wms.control import c4launch_authority as authority


def test_focus_draws_exact_registered_mass_and_four_cells():
    class Dataset:
        receipt = {'split': 'TRAIN'}
        records = [{'meta': {'world': f'w{970000+i}m{m}p{p}t300'}}
                   for i, (m, p) in enumerate([(m, p) for m in (1900, 2400, 2700) for p in ('10', '20', '40')])]

        def __len__(self):
            return len(self.records)

    sampler = StartStateSampler(Dataset(), seed=21)
    receipt = sampler.configure_focus_cells(.6)
    assert len(receipt['cell_windows']) == 4
    picked = [sampler.metadata[sampler._draw_index()]['world'] for _ in range(20000)]
    focus = sum(any(f'm{m}p{p}' in name for m in (2400, 2700) for p in (20, 40)) for name in picked)
    # Distribution test: >5 standard errors would be a real sampler regression.
    assert abs(focus / len(picked) - .6) < 5 * math.sqrt(.6 * .4 / len(picked))


def test_inherited_authority_rejects_wrong_wm_and_wrong_horizon(monkeypatch):
    def verified(path, digest):
        if path == 'authority':
            return {'authority': 'MAIN2026-10-28T15:00', 'wm_sha256': authority.WM3, 'trusted_horizon_s': 90}
        raise AssertionError('invalid authority must refuse before reading further inputs')

    monkeypatch.setattr(authority, 'verified', verified)
    doc = {'schema': 'c4launch_inherited_horizon/1', 'authority': 'authority',
        'wm_sha256': authority.WM3, 'trusted_horizon_s': 120, 'entrant_control': True}
    with pytest.raises(ValueError, match='binding mismatch'):
        authority.inherited_gate(doc, [authority.WM3])
    with pytest.raises(ValueError, match='binding mismatch'):
        authority.inherited_gate({**doc, 'trusted_horizon_s': 90}, ['wrong-wm'])


def test_authority_hash_is_verified_before_parsing(tmp_path, monkeypatch):
    monkeypatch.setenv('C4_ALLOWED_ROOTS', str(tmp_path))
    path = tmp_path / 'authority.json'
    path.write_text(json.dumps({'authority': 'tampered'}))
    with pytest.raises(ValueError, match='hash mismatch'):
        authority.verified(path, authority.AUTHORITY)
