import copy

from tools.rbeval.verify_fleet_prefix import compare


def rows():
    a = dict(seed=969400, arm='native', horizon_s=120, stream=190, cf_model='sumo_idm',
             native_model='sumo', settings='point', entry_phase='after',
             checkpoints={'120': {'ramp_total': 10.}}, merges={'events': [dict(time_s=310.)]},
             births=[dict(time_s=315.)], merge_front_trace=[dict(time_s=320.)])
    b = copy.deepcopy(a)
    b['horizon_s'] = 600
    b['merges']['events'].append(dict(time_s=450.))
    b['births'].append(dict(time_s=600.))
    b['merge_front_trace'].append(dict(time_s=600.))
    return [a], [b]


def test_extension_ignores_only_the_future_not_the_saved_prefix():
    a, b = rows()
    assert compare(a, b) == []
    b[0]['checkpoints']['120']['ramp_total'] += .001
    assert compare(a, b) == ['969400: changed saved checkpoint 120']


def test_different_physics_contract_cannot_reuse_prefix():
    a, b = rows()
    b[0]['native_model'] = 'head'
    assert '969400: changed input contract native_model' in compare(a, b)


def test_changed_birth_or_trace_is_not_hidden_by_matching_cost():
    a, b = rows()
    b[0]['births'][0]['time_s'] = 315.5
    b[0]['merge_front_trace'][0]['time_s'] = 320.5
    assert compare(a, b) == ['969400: changed birth events prefix', '969400: changed front trace prefix']
