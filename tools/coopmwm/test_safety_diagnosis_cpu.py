from .safety_diagnosis import comparison


def rows(hard=0.,ttc=0.,collision=0.):
    a=dict(collisions=0.,teleports=0.,hard_brake=10.,ttc_below_1p5_events=20.,critical_total=30.)
    b=dict(a,collisions=collision,hard_brake=10.+hard,ttc_below_1p5_events=20.+ttc,critical_total=30.+hard+ttc)
    return [dict(seed=i,A=dict(a),right=dict(b)) for i in range(24)]


def test_existing_combined_rule_can_offset_hard_braking_with_ttc():
    r=comparison(rows(hard=4.,ttc=-10.))
    assert r['release'] and r['metrics']['hard_brake']['ci95'][0]>0
    assert r['metrics']['critical_total']['ci95']==[-6.,-6.]


def test_collision_constraint_and_significant_critical_harm_never_release():
    assert not comparison(rows(collision=1.,ttc=-100.))['release']
    assert not comparison(rows(hard=1.,ttc=1.))['release']
    assert comparison(rows())['release']
    assert not comparison([])['release']
