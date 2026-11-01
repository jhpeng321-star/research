"""Final delivery admission rejects substituted panels, sources and checkpoints."""
import copy
import json
from pathlib import Path

import pytest

from tools.c1switch2.selectorhandoff import per_cav_specs


def delivery():
    # This is public receipt metadata only; never opens model/training inputs.
    return json.loads((Path(__file__).resolve().parents[1]/
        'tools/c1switch2/evidence/SELECTOR_DELIVERY_cav_mappo.json').read_text())['delivery']


def test_final_delivery_binds_both_execution_packages_to_training():
    value = delivery()
    spec, peers = per_cav_specs(value, 'a'*64)
    assert spec['checkpoint_sources'] == value['frozen']['sources']
    assert peers['shared']['sources']['runtime.py'] == value['frozen']['sources']['runtime']
    assert peers['per_cav']['sources']['mixed.py'] == value['frozen']['sources']['selmappo2.mixed']


@pytest.mark.parametrize('field,replacement', [
    ('status','RUNNING'), ('iteration',168), ('variant','fleet'),
    ('local_evaluation_sim_s',1200), ('evaluation_owner','other'),
    ('panel_sha256','a'*64), ('checkpoint_sha256','b'*64),
])
def test_nonfinal_or_replayed_delivery_is_refused(field, replacement):
    value = copy.deepcopy(delivery())
    value[field] = replacement
    with pytest.raises(ValueError, match='final169'):
        per_cav_specs(value, 'a'*64)


@pytest.mark.parametrize('name', ['tools/selmappo/runtime.py', 'tools/selmappo/observations.py',
                                  'tools/selmappo2/mixed.py'])
def test_execution_and_feature_substitution_is_refused(name):
    value = copy.deepcopy(delivery())
    value['sources'][name] = 'c'*64
    with pytest.raises(ValueError, match='sources differ|executor differs'):
        per_cav_specs(value, 'a'*64)
