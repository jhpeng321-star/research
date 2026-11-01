import copy

import pytest

from tools.rb2int import release


def test_final_release_binds_four_frozen_models_and_existing_qualification_limits():
    result = release.make_release()
    assert result['status'] == 'READY'
    assert len(result['models']) == 4
    assert result['historically_untouched_claim'] is False
    assert result['final_outcomes_used_for_candidate_selection'] is False
    assert result['catalog']['FULL-v5']['binding']['runtime_root'] == result['catalog']['FULL-v5-LP']['binding']['runtime_root']
    assert result['catalog']['FULL-v5']['binding']['config_sha256'] != result['catalog']['FULL-v5-LP']['binding']['config_sha256']


def test_an_incomplete_integrated_canary_cannot_release_final_inference(monkeypatch):
    original = release.load

    def tampered(path):
        data = copy.deepcopy(original(path))
        if str(path).endswith('CANARY_LP_B2DF7458_STATUS01.json'):
            data['alive'] = True
        return data

    monkeypatch.setattr(release, 'load', tampered)
    with pytest.raises(ValueError, match='integrated canary'):
        release.make_release()
