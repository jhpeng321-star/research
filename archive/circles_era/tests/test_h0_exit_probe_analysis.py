import json

import pytest

from output.h0_exit_probe_v1 import analyze, prefix


def test_phase_windows_partition_and_common_windows():
    life = dict(handback_s=652., terminal_s=652.)
    windows = analyze.phase_windows(life, 642.)
    assert windows['shared_formation'] == (600., 642.)
    assert windows['owned_exit'] == (642., 652.)
    assert windows['recovery'] == (652., 712.)
    assert windows['remaining'] == (712., 2400.)
    # No gaps and no overlap.
    ordered = [windows[k] for k in ('pre_entry', 'shared_formation', 'owned_exit',
                                    'recovery', 'remaining')]
    for (_, end), (start, _) in zip(ordered, ordered[1:]):
        assert end == start


def test_interval_sums_only_counts_rows_inside_window(tmp_path):
    rows_ = [dict(start_s=float(s), end_s=float(s+1), braking_vehicle_seconds=1.,
                  acceleration_energy=0.5, ttc_below_3_vehicle_seconds=0.,
                  requested_not_arrived_vehicle_seconds=2., collision_involvements=0,
                  starting_teleports=0) for s in range(5)]
    path = tmp_path/'intervals.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in rows_), encoding='utf-8')
    sums = analyze.interval_sums(tmp_path, dict(a=(0., 3.), b=(3., 5.), empty=(100., 200.)))
    assert sums['a']['braking_vehicle_seconds'] == 3.
    assert sums['b']['braking_vehicle_seconds'] == 2.
    assert sums['empty']['braking_vehicle_seconds'] == 0.


def test_rear_neighbor_uses_the_members_own_sensed_rear_identity():
    member = dict(vehicle_id='mainline.2', edge_id='e', lane_id='L0', speed_mps=20.,
                  same_lane_rear_id='mainline.1', same_lane_rear_gap_m=18.5)
    rear = dict(member, vehicle_id='mainline.1', speed_mps=21.)
    states = {10.: {'mainline.2': member, 'mainline.1': rear}}
    got = analyze.rear_neighbor(states, 'mainline.2', 10.)
    assert got == dict(vehicle_id='mainline.1', gap_m=18.5, lane='L0')
    # A sensed rear id absent from the sample is missing, never silently equal.
    absent = {10.: {'mainline.2': member}}
    assert analyze.rear_neighbor(absent, 'mainline.2', 10.)['detected_but_absent'] is True
    assert analyze.rear_neighbor({10.: {}}, 'mainline.2', 10.) is None


def test_branch_tick_reads_receipt(tmp_path):
    rows_ = [dict(time_s=float(t), h0_exit_probe={'branch': {'x': 1}} if t == 7 else None)
             for t in range(10)]
    rows_ = [{k: v for k, v in r.items() if v is not None} for r in rows_]
    (tmp_path/'policy.jsonl').write_text('\n'.join(json.dumps(r) for r in rows_),
                                         encoding='utf-8')
    assert prefix.branch_tick(rows_)[0] == 7
    with pytest.raises(ValueError):
        prefix.branch_tick([dict(time_s=0.)])


def test_first_difference_reports_population_and_motion():
    def vehicle(vid, speed, lane='L0'):
        return dict(vehicle_id=vid, edge_id='e', lane_id=lane, speed_mps=speed)
    a = {10.: {'x': vehicle('x', 20.), 'y': vehicle('y', 20.)},
         10.5: {'x': vehicle('x', 20.), 'y': vehicle('y', 19.)}}
    b = {10.: {'x': vehicle('x', 20.), 'y': vehicle('y', 20.)},
         10.5: {'x': vehicle('x', 20.), 'y': vehicle('y', 21.)}}
    got = analyze.first_difference(a, b, after=10.)
    assert got == dict(time_s=10.5, kind='motion', vehicle_id='y',
                       speed_a=19., speed_b=21.)
    # Different lane identity is its own explicit difference kind.
    c = {10.: {'x': vehicle('x', 20.), 'y': vehicle('y', 20., lane='L1')}}
    assert analyze.first_difference(a, c, after=10.)['kind'] == 'lane_or_edge'
    # A missing tracked vehicle is missing measurement, not equality.
    d = {10.: {'x': vehicle('x', 20.)}}
    assert analyze.first_difference(a, d, after=10.)['kind'] == 'population_or_coverage'
    assert analyze.first_difference(a, a, after=10.) is None


def test_lifecycle_times_from_policy_stream(tmp_path):
    rows_ = []
    plan = [(600., 'assembly', {'m': -.1}), (620., 'hold', {'m': 0.}),
            (622., 'taper', {'m': 0.}), (623., 'release', {})]
    for time_s, phase, commands in plan:
        row = dict(time_s=time_s, phase=phase, reason='r', wall_commands=commands,
                   member_ids=['m'], proposal_sha256='p', evidence_sha256='e')
        if phase == 'taper':
            row['h0_exit_probe'] = dict(branch=dict(
                branch_request_time_s=620., exit_admitted=True, h0_delivered=True,
                fallback=None, delivered_hold_target_s=0., requested_profile_hold_s=20.))
        rows_.append(row)
    (tmp_path/'policy.jsonl').write_text('\n'.join(json.dumps(r) for r in rows_),
                                         encoding='utf-8')
    life = analyze.lifecycle_times(tmp_path)
    assert life['command_start_s'] == 600. and life['committed_hold_start_s'] == 620.
    assert life['exit_start_s'] == 622. and life['terminal_s'] == 623.
    assert life['delivered_hold_s'] == 2. and life['handback_s'] == 623.
    assert life['branch_receipt']['branch_request_time_s'] == 620.
