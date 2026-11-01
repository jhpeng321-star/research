import copy
import json

import pytest

from tools.rb2int.prove_prefix import compare, ARMS, REGIONS


def fixture():
    exported = dict(episodes=[dict(seed=969550, arm=a, demand=2400, penetration=.2,
        initial_sha256='initial', bins={r: dict(speed=[20.]*6) for r in REGIONS}) for a in ARMS])
    recorded = copy.deepcopy(exported['episodes'])
    for row in recorded:
        row['clocks'] = {'600': {'team_delay_vs': 999999}}
        for r in REGIONS:
            row['bins'][r]['speed'] += [999999.]*60
    return exported, recorded


def test_prefix_projection_never_exports_future_bins_or_targets():
    exported, recorded = fixture()
    canonical, proof = compare(exported, recorded, [969550])
    assert proof['passed'] and proof['max_absolute_prefix_error'] == 0
    assert '999999' not in json.dumps(canonical)
    assert all(len(row['bins'][r]['speed']) == 6 for row in canonical['episodes'] for r in REGIONS)


def test_wrong_causal_identity_or_speed_stops_bridge():
    exported, recorded = fixture()
    recorded[0]['initial_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='warmup identity'):
        compare(exported, recorded, [969550])
    exported, recorded = fixture()
    recorded[0]['bins']['upstream']['speed'][0] = 0
    with pytest.raises(ValueError, match='numeric tolerance'):
        compare(exported, recorded, [969550])
