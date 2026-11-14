from output.formation_harm_v1.interactions_phase_v3 import endpoint,select,changed


def sample(time=601,**extra):
    state=dict(speed_mps=20,lane_position_m=100,**extra)
    return dict(time_s=time,treatment_present=True,damping_present=True,
                treatment=state.copy(),damping=state.copy())


def test_absent_edge_identifiers_never_pass_endpoint_guard():
    r=endpoint(sample())
    assert r['position_difference'] is None
    assert r['speed_difference']==0


def test_different_lane_endpoint_is_unavailable():
    q=sample(edge_id='e',lane_id='e_0')
    q['damping']['lane_id']='e_1'
    assert endpoint(q)['position_difference'] is None


def test_actual_matching_edge_lane_allows_position_comparison():
    q=sample(edge_id='e',lane_id='e_0')
    q['treatment']['lane_position_m']=98
    assert endpoint(q)['position_difference']==-2


def test_phase_boundary_included_exactly_once():
    rows=[sample(t) for t in (600,600.5,601,601.5)]
    assert [q['time_s'] for q in select(rows,600,601)]==[600.5,601]
    assert [q['time_s'] for q in select(rows,601,602)]==[601.5]


def test_gap_change_is_not_motion_change():
    a=dict(speed_mps=20,acceleration_mps2=0,gap=5)
    b=dict(a,gap=4)
    assert not changed(a,b,('speed_mps','acceleration_mps2'))
    assert changed(a,b,('gap',))


def test_no_change_does_not_mean_missing_measurement():
    q=sample()
    assert q['treatment_present'] and q['damping_present']
    assert not changed(q['treatment'],q['damping'],('speed_mps',))
