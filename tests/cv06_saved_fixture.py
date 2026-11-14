"""Shared CV-06 saved-run fixture machinery for the CV-12/13 PI review tests.

The synthetic manifest intentionally carries the historical follower-stopper
identity the frozen CV-06 review module expects; it is fixture data, not a
runtime action (FollowerStopper retired 2026-09-25).
"""
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location('cv06_review', Path(__file__).parents[1] /
    'output/controller_verification_v6/independent_review.py')
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def lines(path, value):
    path.write_text(''.join(json.dumps(r) + '\n' for r in value), encoding='utf-8')


def refresh(root):
    manifest = review.read(root / 'manifest.json')
    for name in ('summary', 'telemetry', 'intervals'):
        manifest[name + '_sha256'] = review.sha(root / (name + ('.json' if name == 'summary' else '.jsonl')))
    write(root / 'manifest.json', manifest)


@pytest.fixture
def saved(tmp_path):
    root = tmp_path / 'run'
    (root / 'scenario').mkdir(parents=True)
    requests = [dict(vehicle_id='cav', depart_s=.25, route='r', driver_type='normal_car', is_cav=True)]
    request_path = tmp_path / 'requests.json'
    write(request_path, requests)
    route = root / 'scenario/requests.rou.xml'
    route.write_text('<routes><vehicle id="cav" depart="0.25" route="r" type="normal_car"/></routes>')
    scene = root / 'scenario/scenario.sumocfg'
    scene.write_text('<configuration/>')
    expected = dict(seed=4107, loading=2000, penetration=.2, duration_s=2400, desired_speed_mps=31.29,
        config_sha256='config', model_sha256='model', sumo_binary_sha256='sumo',
        active_source_sha256={'pinned.py': 'source'}, scenario_identity='scene',
        requests_path=str(request_path), requests_sha256=review.sha(request_path),
        scene_sha256={'requests.rou.xml': review.sha(route), 'scenario.sumocfg': review.sha(scene)})
    manifest = dict(status='complete', seed=4107, mainline_vph=2000, penetration=.2, duration_s=2400,
        controller='follower-stopper', role='exposed', accepted_checkpoint=False, mechanics_only=False,
        controller_identity=dict(factory_version='controller_factory/1', controller='follower-stopper',
            desired_speed_mps=31.29, law_version='follower_stopper/1', adapter_version='follower_stopper_direct_bridge/1',
            u_provenance='fixed_registered_setpoint'),
        config=dict(control_dt_s=1., physics_dt_s=.5, sensor_range_m=100., speed_mode=31,
                    acceleration_min_mps2=-3., acceleration_max_mps2=1.5),
        sumo_version=[22, 'SUMO 1.27.1'],
        scenario=dict(requests_sha256=review.hashlib.sha256(json.dumps(requests, sort_keys=True).encode()).hexdigest(),
                      routes_sha256=review.sha(route)))
    for key in ('config_sha256', 'model_sha256', 'sumo_binary_sha256', 'active_source_sha256', 'scenario_identity'):
        manifest[key] = expected[key]
    write(root / 'manifest.json', manifest)
    lines(root / 'physics_steps.jsonl', [dict(time_s=i*.5, live_ids=['cav'] if 2 <= i < 6 else [],
        departed_ids=['cav'] if i == 2 else [], arrived_ids=['cav'] if i == 6 else [],
        speed_modes={'cav': 31} if 2 <= i < 6 else {}) for i in range(1, 4801)])
    telemetry = []
    for t in (1., 2.):
        frame = dict(vehicle_id='cav', timestamp_s=t, speed_mps=31.29, allowed_speed_mps=31.29,
                     front_coverage_m=100., rear_coverage_m=100., rear=None,
                     front=dict(gap_m=50., relative_speed_mps=0.))
        command = dict(kind='command', vehicle_id='cav', timestamp_s=t, owner='damping',
                       network_output_mps2=None, wall_request_mps2=None, issued_acceleration_mps2=0.,
                       issued_target_speed_mps=31.29, fallback_reason=None, released_to_native=False, actor_input_si=None)
        telemetry += [command, dict(kind='cav_report', frame=frame, sent_at_s=t, owner='damping',
            network_output_mps2=None, issued_acceleration_mps2=0., fallback_reason=None),
            dict(kind='analytical_audit', decision_vehicle_id='cav', decision_timestamp_s=t,
                audit_vehicle_id='cav', audit_timestamp_s=t, frame=frame,
                audit=dict(law_version='follower_stopper/1', adapter_version='follower_stopper_direct_bridge/1',
                    vehicle_id='cav', timestamp_s=t, desired_speed_mps=31.29, raw_speed_command_mps=31.29,
                    requested_acceleration_mps2=0., issued_acceleration_mps2=0., issued_target_speed_mps=31.29,
                    clipping_or_fallback=None))]
        for after in (t + .5, t + 1.):
            if after < 3:
                telemetry.append(dict(kind='response', vehicle_id='cav', timestamp_s=after, command_timestamp_s=t,
                    owner='damping', measured_speed_mps=31.29, measured_acceleration_mps2=0.))
    lines(root / 'telemetry.jsonl', telemetry)
    totals = dict(braking_vehicle_seconds=0., acceleration_energy=0., ttc_below_3_vehicle_seconds=0.,
                  requested_not_arrived_vehicle_seconds=2.75, collision_involvements=0, starting_teleports=0)
    lines(root / 'intervals.jsonl', [dict(start_s=i, end_s=i+1, **dict(totals,
        requested_not_arrived_vehicle_seconds=.75 if i == 0 else 1. if i in (1, 2) else 0.)) for i in range(2400)])
    summary = dict(seed=4107, controller='follower-stopper', duration_s=2400, complete_fixed_horizon=True,
        mechanics_only=False, accepted_checkpoint=False, scenario_identity='scene', population=1, inserted=1,
        arrived=1, requested_due_at_horizon=1, uninserted_due=0, unfinished_requested=0, designated_cavs=1,
        realized_designation_fraction=1., system_seconds_per_requested_vehicle=2.75, source_wait_vehicle_seconds=.75,
        command_counts={'damping': 2}, fallback_counts={}, totals=totals, minimum_closing_ttc_s=None)
    write(root / 'summary.json', summary)
    refresh(root)
    return root, expected


