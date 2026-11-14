from collections import Counter
from types import SimpleNamespace
from traffic_shepherd.sensor_batch import capture_batch
from traffic_shepherd.sensing import SumoLocalSensor


def fixture():
    calls = Counter()
    positions = {'a':100., 'b':150., 'hdv':200.}
    def track(name, value):
        def get(*args):
            calls[name] += 1
            return value(*args) if callable(value) else value
        return get
    vehicle = SimpleNamespace(**{name:track(name,value) for name,value in {
        'getLaneID':'road_0', 'getLanePosition':lambda v:positions[v], 'getLength':5.,
        'getSpeed':20., 'getAcceleration':0., 'getAllowedSpeed':30., 'getLaneIndex':0}.items()})
    lane = SimpleNamespace(getLength=track('laneLength',1000.),
                           getLastStepVehicleIDs=track('laneIDs',tuple(positions)))
    return SimpleNamespace(vehicle=vehicle,lane=lane), calls, positions


def test_batch_is_exact_with_fewer_identical_reads_and_no_cross_tick_cache():
    api, calls, positions = fixture()
    sensor = SumoLocalSensor(api)
    direct = {v:sensor.capture(v,now_s=0) for v in ['a','b']}
    direct_count = sum(calls.values())
    calls.clear()
    frames, errors, stats = capture_batch(api,['a','b'],now_s=0)
    assert frames == direct and not errors
    assert stats['sensor_transport_queries'] == sum(calls.values()) < direct_count
    positions['hdv'] += 10
    next_frames, _, _ = capture_batch(api,['a','b'],now_s=1)
    assert next_frames['b'].front.gap_m == frames['b'].front.gap_m+10


def test_invalid_sensor_measurement_stays_unavailable():
    api, _, positions = fixture()
    positions['hdv'] = 152.
    frames, errors, _ = capture_batch(api,['b'],now_s=0)
    assert not frames and 'b' in errors
