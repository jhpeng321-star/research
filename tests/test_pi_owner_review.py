"""Independent real-owner PI transaction and lifecycle checks; no simulator."""
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest

from output.controller_verification_v7.pi_owner_adapter import PIOwnerAdapter
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


def frame(v, t, speed=10., gap=20.):
    return SensorFrame(v, t, speed, 0., 31.29, 0,
        front=NeighborMeasurement(gap, 0.), front_coverage_m=100., rear_coverage_m=100.)


def setup(ids=('a', 'b', 'c')):
    writes = []
    fail = dict(vehicle=None, release=False)
    def slow(v, target, duration):
        if v == fail['vehicle']:
            raise RuntimeError('injected writer failure')
        writes.append(('slowDown', v, target, duration))
    def release(v, target):
        if fail['release']:
            raise RuntimeError('injected release failure')
        writes.append(('setSpeed', v, target))
    api = SimpleNamespace(vehicle=SimpleNamespace(getSpeedMode=lambda _: 31,
        slowDown=slow, setSpeed=release))
    adapter = PIOwnerAdapter(cav_ids=set(ids))
    owner = CommandOwnership(api, set(ids), adapter)
    return adapter, owner, writes, fail


def test_raw_memory_and_actual_per_vehicle_ack():
    adapter, owner, writes, _ = setup(('a', 'b'))
    initial = {'a': frame('a', 1.), 'b': frame('b', 1., speed=2.)}
    rows = adapter.step(owner, initial, now_s=1., live_ids={'a', 'b', 'hdv'})
    raw_a = .5 * (31.29 + 13 / 23) + .5 * 10.
    assert rows[0].issued_target_speed_mps == 11.5
    assert adapter.pi._state['a'][0] == pytest.approx(raw_a)
    second = {'a': frame('a', 2.), 'b': frame('b', 2., speed=2.)}
    adapter.step(owner, second, now_s=2., live_ids={'a', 'b', 'hdv'})
    assert adapter.pi._state['a'][0] == pytest.approx(.5 * (31.29 + 13 / 23) + .5 * raw_a)
    assert all(row[1] != 'hdv' for row in writes)


def test_partial_batch_keeps_first_commit_discards_unissued_and_stops():
    adapter, owner, writes, fail = setup()
    fail['vehicle'] = 'b'
    with pytest.raises(RuntimeError):
        adapter.step(owner, {v: frame(v, 1.) for v in ('a', 'b', 'c')},
                     now_s=1., live_ids={'a', 'b', 'c'})
    assert [w[1] for w in writes] == ['a']
    assert adapter.pi._state['a'][0] == pytest.approx(.5 * (31.29 + 13 / 23) + 5.)
    assert adapter.pi._state['b'][0] == adapter.pi._state['c'][0] == 10.
    assert not adapter._pending and not adapter.pi._pending
    assert owner._last['a'].issued_target_speed_mps == 11.5
    assert adapter.stopped
    with pytest.raises(RuntimeError):
        adapter.step(owner, {v: frame(v, 2.) for v in ('a', 'b', 'c')},
                     now_s=2., live_ids={'a', 'b', 'c'})
    assert len(writes) == 1


@pytest.mark.parametrize('unavailable', ['none', 'absent', 'no_leader', 'stale', 'invalid_coverage', 'infeasible_bridge'])
def test_unavailable_input_invalidates_and_reentry_resets(unavailable):
    adapter, owner, _, _ = setup(('a',))
    adapter.step(owner, {'a': frame('a', 1.)}, now_s=1., live_ids={'a'})
    bad = ({'a': None} if unavailable == 'none' else {} if unavailable == 'absent'
           else {'a': replace(frame('a', 2.), front=None)} if unavailable == 'no_leader'
           else {'a': replace(frame('a', 2.), front_coverage_m=101.)} if unavailable == 'invalid_coverage'
           else {'a': frame('a', 2., speed=40.)} if unavailable == 'infeasible_bridge'
           else {'a': frame('a', 1.)})
    rows = adapter.step(owner, bad, now_s=2., live_ids={'a'})
    assert rows[0].owner == 'native' and rows[0].released_to_native
    assert 'a' not in adapter.pi._state
    adapter.step(owner, {'a': frame('a', 3., speed=2.)}, now_s=3., live_ids={'a'})
    assert adapter.pi._state['a'][0] == pytest.approx(.5 * (31.29 + 13 / 23) + 1.)


def test_skip_after_native_tick_is_not_silently_accepted():
    adapter, owner, writes, _ = setup(('a',))
    adapter.step(owner, {'a': frame('a', 1.)}, now_s=1., live_ids={'a'}, native_ids={'a'})
    with pytest.raises((ValueError, RuntimeError)):
        adapter.step(owner, {'a': frame('a', 3.)}, now_s=3., live_ids={'a'}, native_ids={'a'})
    assert adapter.stopped
    assert writes == []


def test_outside_batch_decide_and_replayed_real_receipt_rejected():
    adapter, owner, _, _ = setup(('a',))
    with pytest.raises((ValueError, RuntimeError)):
        adapter.decide(frame('a', 1.), now_s=1.)
    adapter, owner, _, _ = setup(('a',))
    rows = adapter.step(owner, {'a': frame('a', 1.)}, now_s=1., live_ids={'a'})
    before = adapter.pi._state['a']
    with pytest.raises((ValueError, RuntimeError)):
        adapter.record_callback(rows[0])
    assert adapter.pi._state['a'] == before


def test_changed_frame_during_real_owner_call_stops_without_write():
    adapter, owner, writes, _ = setup(('a',))
    actual = owner.step
    def changed(frames, **kwargs):
        changed_frames = dict(frames, a=replace(frames['a'], speed_mps=11.))
        return actual(changed_frames, **kwargs)
    owner.step = changed
    with pytest.raises(RuntimeError):
        adapter.step(owner, {'a': frame('a', 1.)}, now_s=1., live_ids={'a'})
    assert writes == [] and adapter.stopped
    assert not adapter.pi._pending


def test_prepared_stale_refusal_cannot_be_replaced_by_fresh_frame():
    adapter, owner, writes, _ = setup(('a',))
    adapter.step(owner, {'a': frame('a', 1.)}, now_s=1., live_ids={'a'})
    actual = owner.step
    def changed(frames, **kwargs):
        return actual({'a': frame('a', 2.)}, **kwargs)
    owner.step = changed
    with pytest.raises(RuntimeError):
        adapter.step(owner, {'a': frame('a', 1.)}, now_s=2., live_ids={'a'})
    assert len(writes) == 1 and adapter.stopped
    assert not adapter.pi._pending


def test_invalid_masks_do_not_mutate_existing_recurrence():
    adapter, owner, writes, _ = setup(('a',))
    adapter.step(owner, {'a': frame('a', 1.)}, now_s=1., live_ids={'a'})
    before = adapter.pi._state['a']
    with pytest.raises((ValueError, RuntimeError)):
        adapter.step(owner, {'a': frame('a', 2.)}, now_s=2., live_ids={'a', 'hdv'},
                     native_ids={'a', 'hdv'})
    assert adapter.pi._state['a'] == before and len(writes) == 1
    assert adapter.stopped


def test_commit_exception_retains_written_owner_receipt_and_stops(monkeypatch):
    adapter, owner, writes, _ = setup(('a',))
    def fail(*args, **kwargs):
        raise RuntimeError('injected commit failure after API return')
    monkeypatch.setattr(adapter.pi, 'commit', fail)
    with pytest.raises(RuntimeError):
        adapter.step(owner, {'a': frame('a', 1.)}, now_s=1., live_ids={'a'})
    assert writes == [('slowDown', 'a', 11.5, 1.)]
    assert owner._last['a'].issued_target_speed_mps == 11.5
    assert any(r.status == 'written_owner_receipt' for r in adapter.receipts)
    assert adapter.stopped and not adapter.pi._pending


def test_native_release_write_failure_stops_and_keeps_prior_owner_record():
    adapter, owner, writes, fail = setup(('a',))
    adapter.step(owner, {'a': frame('a', 1.)}, now_s=1., live_ids={'a'})
    fail['release'] = True
    with pytest.raises(RuntimeError):
        adapter.step(owner, {'a': frame('a', 2.)}, now_s=2., live_ids={'a'}, native_ids={'a'})
    assert len(writes) == 1 and adapter.stopped
    assert owner._last['a'].timestamp_s == 1.


def test_independent_three_step_recurrence_alpha_zero_half_one():
    adapter, owner, _, _ = setup(('a',))
    previous = 2.
    for tick, gap, alpha in [(1., 3., 0.), (2., 5., .5), (3., 20., 1.)]:
        # Equal leader/ego2 gives literal safety gap4. The bridge saturates
        # while recurrence must retain the independently computed raw result.
        target = 31.29 + min(max((gap - 7) / 23, 0), 1)
        beta = 1 - alpha / 2
        raw = beta * (alpha * target + (1-alpha)*2.) + (1-beta)*previous
        records = adapter.step(owner, {'a': frame('a', tick, speed=2., gap=gap)},
                               now_s=tick, live_ids={'a'})
        assert adapter.pi._state['a'][0] == pytest.approx(raw)
        issued = min(max(raw-2., -2.), 1.5)
        assert records[0].issued_target_speed_mps == pytest.approx(2.+issued)
        previous = raw
    # Round-trip the actual sidecar: no object tokens, NaN startup sentinels,
    # or hidden live controller state is needed to reproduce the recurrence.
    evidence = json.loads(json.dumps(adapter.evidence, allow_nan=False))
    assert len(evidence) == 3
    for row in evidence:
        observed = row['frame']
        gap = observed['front']['gap_m']
        leader = observed['speed_mps'] + observed['front']['relative_speed_mps']
        u_report = row['u_report']
        assert u_report['provenance'] == 'fixed_registered_setpoint'
        assert u_report['value_mps'] == 31.29 and u_report['evidence_id']
        assert observed['timestamp_s'] == row['timestamp_s']
        assert observed['vehicle_id'] == row['vehicle_id']
        alpha = min(max((gap-max(2*(leader-observed['speed_mps']), 4))/2, 0), 1)
        beta = 1-alpha/2
        target = u_report['value_mps']+min(max((gap-7)/23, 0), 1)
        expected_raw = beta*(alpha*target+(1-alpha)*leader)+(1-beta)*row['previous_raw_command_mps']
        assert row['raw_proposal_mps'] == pytest.approx(expected_raw)
        assert row['owner_record']['issued_target_speed_mps'] == row['issued_target_speed_mps']
    json.dumps([asdict(row) for row in adapter.receipts], allow_nan=False)


def test_native_wall_and_id_reuse_reset_from_current_measured_speed():
    adapter, owner, _, _ = setup(('a', 'b'))
    adapter.step(owner, {v: frame(v, 1.) for v in ('a', 'b')}, now_s=1., live_ids={'a', 'b'})
    rows = adapter.step(owner, {v: frame(v, 2.) for v in ('a', 'b')}, now_s=2., live_ids={'a', 'b'},
                        native_ids={'a'}, wall_commands={'b': -.5})
    assert [row.owner for row in rows] == ['native', 'wall']
    assert not adapter.pi._state
    adapter.step(owner, {v: frame(v, 3., speed=2.) for v in ('a', 'b')}, now_s=3., live_ids={'a', 'b'})
    expected = .5 * (31.29 + 13/23) + 1.
    assert adapter.pi._state['a'][0] == adapter.pi._state['b'][0] == pytest.approx(expected)
    adapter.step(owner, {'a': frame('a', 4., speed=4.)}, now_s=4., live_ids={'a'},
                 departed_ids={'b'}, id_reuse_ids={'a'})
    assert 'b' not in adapter.pi._state
    assert adapter.pi._state['a'][0] == pytest.approx(.5 * (31.29 + 13/23) + 2.)


@pytest.mark.parametrize('period', [.5, 2.])
def test_nonregistered_sample_period_rejected(period):
    with pytest.raises(ValueError):
        PIOwnerAdapter(cav_ids={'a'}, sample_period_s=period)


@pytest.mark.parametrize('timestamp', [float('nan'), 2.5])
def test_invalid_tick_preserves_committed_state_and_stops(timestamp):
    adapter, owner, writes, _ = setup(('a',))
    adapter.step(owner, {'a': frame('a', 1.)}, now_s=1., live_ids={'a'})
    before = adapter.pi._state['a']
    with pytest.raises((ValueError, RuntimeError)):
        adapter.step(owner, {'a': frame('a', 2.)}, now_s=timestamp, live_ids={'a'})
    assert adapter.pi._state['a'] == before and len(writes) == 1
    assert adapter.stopped


def test_native_mask_cannot_be_replaced_with_unprepared_pi_decision():
    adapter, owner, writes, _ = setup(('a',))
    actual = owner.step
    def changed(frames, **kwargs):
        return actual(frames, **dict(kwargs, native_ids=set()))
    owner.step = changed
    with pytest.raises(RuntimeError):
        adapter.step(owner, {'a': frame('a', 1.)}, now_s=1., live_ids={'a'}, native_ids={'a'})
    assert writes == [] and adapter.stopped
    assert not adapter.pi._pending
