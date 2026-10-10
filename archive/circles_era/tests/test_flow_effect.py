from output.flow_effect_v1.flow import movements, region


def test_lane_changes_inside_volume_do_not_create_flow():
    assert not movements({'a':'pre_merge'},{'a':'pre_merge'})


def test_departures_and_arrivals_conserve_inventory():
    result=movements({'a':'pre_merge','b':'downstream'},{'a':'connector','c':'ramp'})
    assert result=={'pre_merge>connector':1,'downstream>outside':1,'outside>ramp':1}


def test_ramp_merge_and_mainline_are_separate():
    result=movements({'a':'ramp','b':'pre_merge'},{'a':'connector','b':'merge'})
    assert result=={'ramp>connector':1,'pre_merge>merge':1}


def test_internal_connector_has_own_volume():
    assert region({'edge_id':':junction'}, {})=='connector'


def test_unknown_external_edge_rejected():
    import pytest
    with pytest.raises(AssertionError):
        region({'edge_id':'unmapped'}, {})


def test_window_counts_events_not_cumulative_totals():
    from output.flow_effect_v1.analyze import COUNTS, summarize
    from output.flow_effect_v1.flow import VOLUMES
    rows=[]
    for i in range(3):
        row=dict(time_s=i*.5,n={v:0 for v in VOLUMES},**{k:0 for k in COUNTS},
            interval_source_wait_vehicle_seconds=0,interval_system_vehicle_seconds=.5)
        rows.append(row)
    rows[0]['n']['pre_merge']=1
    rows[1]['n']['merge']=1
    rows[1]['mainline_toward_merge']=1
    rows[2]['n']['downstream']=1
    rows[2]['downstream_entries']=1
    result=summarize(rows,0,1)
    assert result['mainline_toward_merge']==result['downstream_entries']==1
    assert result['middle_inventory_change']==0
    # Interval(0.5,1] must not include the earlier entry event.
    later=summarize(rows,.5,1)
    assert later['mainline_toward_merge']==0 and later['downstream_entries']==1
    assert later['middle_inventory_change']==-1


def test_equal_final_counts_can_hide_different_timing():
    from output.flow_effect_v1.analyze import first_divergence
    a=[{'time_s':600,'x':0},{'time_s':600.5,'x':1},{'time_s':601,'x':0}]
    b=[{'time_s':600,'x':0},{'time_s':600.5,'x':0},{'time_s':601,'x':1}]
    assert sum(r['x'] for r in a)==sum(r['x'] for r in b)
    assert first_divergence(a,b,'x')==600.5
