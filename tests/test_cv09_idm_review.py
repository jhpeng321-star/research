"""Independent CV09 source arithmetic and saved-evidence checks; no SUMO."""
import hashlib
import json
from pathlib import Path

import pytest

from output.controller_verification_v8.independent_review import verify
from output.controller_verification_v9.idm_reconstruction import idm_next, reconstruct

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'output/controller_verification_v9'


def result():
    return json.loads((ROOT / 'runs/active/controller_verification_v8/result.json').read_text())


def test_manual_idm_source_fixture():
    # Independently calculated from the published-source two .25 s updates.
    assert idm_next(15., 15., 45.) == pytest.approx(15.959487360856269, abs=1e-13)
    # An opening gap must not expand the virtual gap during either substep.
    first = 1. + .65 * (1. - (1./30.)**4 - (2.5/45.)**2)
    second = first + .65 * (1. - (first/30.)**4 - (2.5/45.)**2)
    assert idm_next(1., 15., 45.) == pytest.approx(second, abs=1e-12)


def test_saved_fourteen_transitions_have_expected_competing_bounds():
    data = result()
    assert verify(data)['passed']
    rows = reconstruct(data)
    assert [(r['from_s'], r['to_s']) for r in rows] == [(i/2, (i+1)/2) for i in range(2,16)]
    assert [r['active_bound'] for r in rows] == ['command_ramp']*2 + ['idm_candidate']*2 + ['native_idm']*2 + ['idm_candidate']*2 + ['native_idm']*2 + ['idm_candidate']*4
    assert max(abs(r['error_mps']) for r in rows) <= 1e-12
    assert rows[0]['command_advice_mps'] == 15.75
    assert rows[1]['command_advice_mps'] == 16.5
    assert rows[2]['idm_candidate_mps'] == pytest.approx(17.247508102927185, abs=1e-13)
    assert rows[-1]['predicted_mps'] == pytest.approx(17.195922338896096, abs=1e-13)


def test_pinned_primary_source_bytes_and_saved_input_identity():
    identities = json.loads((BASE/'sources/IDENTITIES.json').read_text())
    assert len(identities) == 4
    for item in identities:
        assert '/v1_27_1/' in item['url']
        raw = (BASE/'sources'/Path(item['path']).name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
    saved = json.loads((BASE/'IDM_RECONSTRUCTION.json').read_text())
    assert hashlib.sha256((ROOT/saved['input_path']).read_bytes()).hexdigest() == saved['input_sha256']
    assert saved['transitions'] == reconstruct(result())


def test_reconstruction_refuses_incomplete_saved_clock():
    data = result()
    data['physical'].pop(3)
    with pytest.raises(ValueError, match='strict source evidence'):
        reconstruct(data)
