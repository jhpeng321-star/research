"""CPU-only lane adapter contracts. No model, CUDA, worker or simulator access.

python -B -m unittest tools.mining.wm_lane_tests -v
"""
import ast
import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from .wm_lane_schedule import make_schedule, validate_schedule, action_set, targets_at
from .wm_lane_adapter import LaneRuntime, compile_bridge, compile_option, raw_decoder_factory


def native_bridge(policy, group_encoder, safety, record=None, *, native_lane_change=True):
    """Fixture retains exact frozen insertion structure and old/target safety."""
    def native_lane_mask(state):
        mask = state['present'] & state['is_cav']
        if True:
            return mask
        return mask & ~mask
    def controller(state):
        mask = state['present'] & state['is_cav']
        native_lanes = mask
        lane_now = state['lane']
        log_entry = {'lane_changed': torch.zeros_like(mask)}
        intent = torch.zeros_like(state['s_m'])
        if native_lane_change:
            targets = state['native_targets']
            lane_now = torch.where(native_lanes & (state['lane'] != 2), targets, lane_now)
        state['cav_wm_merge_law'] = mask & (lane_now == 2)
        motion_lane = torch.where(native_lanes, state['lane'], lane_now)
        executed = safety(dict(state, lane=motion_lane), policy)
        if bool(log_entry['lane_changed'].any()):
            old_safe = safety(dict(state, lane=state['lane']), policy)
            executed = torch.where(log_entry['lane_changed'], torch.minimum(executed, old_safe), executed)
        return torch.stack((state['s_m'], lane_now.to(executed), state['v_mps'], executed), -1)
    controller.native_lane_mask = native_lane_mask
    return controller


def option_fixture(mp, candidate_actions, initial_rows):
    ids = [r['vid'] for r in initial_rows]
    fixed = [None for a in candidate_actions]
    tick = 0
    def safe_accel(state, raw, **kw):
        return raw.clamp(-3, 3)
    def injected(state, accel, **kw):
        return accel
    bridge = make_cav_controller(mp, None, injected)
    def callback(state):
        nonlocal tick
        out = bridge(state)
        tick += 1
        return out
    callback.native_lane_mask = bridge.native_lane_mask
    return callback


def native_sample(self, context, generator, deterministic=False, memory=None):
    previous = torch.tensor([[memory.get(v, 0.)] for v in context['members']]).reshape(-1, 1)
    innovation = math.sqrt(.5) * torch.randn((1, 1), generator=generator)
    innovation = innovation + math.sqrt(.5) * torch.randn((len(previous), 1), generator=generator)
    noise = previous * math.exp(-.2) + self.log_std.exp() * math.sqrt(1 - math.exp(-.4)) * innovation
    return {'own_action': noise.tolist()}, dict(zip(context['members'], noise.flatten().tolist()))


def load_actual_lane_helpers():
    """Use existing ORIGINAL lane geometry helpers, not a fake gap-benefit model.

    Local G5 lane helpers share the default observable rules with frozen FULL-v4.
    Runtime separately requires the exact GPU frozen source SHA before use.
    """
    root = Path('E:/agent-work/c4-grp-lane')
    path = root / 'src/wms/control/lanes.py'
    if not path.is_file():
        raise unittest.SkipTest('native local lane helper source absent')
    import sys
    old = list(sys.path)
    try:
        sys.path.insert(0, str(root / 'src'))
        spec = importlib.util.spec_from_file_location('wms.control._mining_test_actual_lanes', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old


class LaneContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lanes = load_actual_lane_helpers()

    def state(self, batch=1):
        # In-zone, away from boundaries; safe empty left mainline and a safe
        # front/rear merge gap when the yielder has left. Additional CAV and HDV
        # are native nonactors whose targets must be preserved.
        return dict(s_m=torch.tensor([[20., 30., 120., -30.]] * batch),
                    v_mps=torch.tensor([[2., 2., 2., 2.]] * batch),
                    lane=torch.tensor([[1, 2, 0, 0]] * batch),
                    present=torch.ones(batch, 4, dtype=torch.bool),
                    is_cav=torch.tensor([[True, True, True, False]] * batch),
                    length_m=torch.full((batch, 4), 5.),
                    native_targets=torch.tensor([[1, 2, 1, 0]] * batch))

    def runtime(self, schedule=None, active=None, coverage=None, telemetry=None):
        s = schedule or make_schedule('yield_then_merge', 'main', 'ramp')
        a = action_set(s)[0]
        if active is not None: a['wm_active_members'] = active
        return LaneRuntime([a], ['main', 'ramp', 'other', 'hdv'],
                           {'main': [0, 1, 2], 'ramp': [0, 1, 2]} if coverage is None else coverage,
                           self.lanes, telemetry)

    def test_two_actors_are_not_mislabeled_as_canonical_yield(self):
        s = make_schedule('yield_then_merge', 'main', 'ramp', onset=10)
        self.assertEqual(targets_at(s, 10, s['members']), {'main': (1, 0)})
        self.assertEqual(targets_at(s, 11, s['members']), {'ramp': (2, 1)})
        self.assertEqual(targets_at(make_schedule('g5_yield_lane', 'main', 'ramp'), 1, s['members']), {})
        bad = copy.deepcopy(s); bad['lane_requests'][1]['second_s'] = 10
        with self.assertRaises(ValueError): validate_schedule(bad)

    def test_native_mask_pure_and_missing_coverage_no_override(self):
        state = self.state(); runtime = self.runtime(coverage={})
        original = state['present'] & state['is_cav']
        self.assertTrue(torch.equal(runtime.native_mask(state, original), original))
        self.assertTrue(torch.equal(runtime.native_mask(state, original), original))
        self.assertEqual(runtime.tick, 0)
        self.assertIsNone(runtime.dwell.age)
        target, owned, _ = runtime.apply(state, state['native_targets'])
        self.assertFalse(bool(owned.any()))
        self.assertTrue(torch.equal(target, state['native_targets']))

    def test_original_lane_safety_blocks_overlap_hdv_and_no_retry(self):
        state = self.state(); state['s_m'][0, 3] = 20.
        runtime = self.runtime()
        mask, *_ = runtime.planned(state)
        self.assertFalse(bool(mask.any()))  # HDV alongside target lane blocks command
        runtime.apply(state, state['native_targets'])
        state['s_m'][0, 3] = -30.
        mask, *_ = runtime.planned(state)
        self.assertFalse(bool(mask.any()))  # odd halfstep: blocked command is not retried
        runtime.apply(state, state['native_targets'])
        mask, *_ = runtime.planned(state)
        self.assertFalse(bool(mask[0, 0]))  # next second: mainline attempt is over

    def test_compiled_pipeline_two_actors_native_nonactors_and_dual_lane_safety(self):
        state = self.state(); runtime = self.runtime(); safety_calls = []
        def safety(s, raw):
            safety_calls.append(s['lane'].clone())
            # A restrictive old-lane leader must still bound the command.
            return torch.where(s['lane'] == 1, torch.full_like(raw, -.5), raw)
        cb = compile_bridge(native_bridge)(torch.ones(1, 4), None, safety, mining_lane=runtime)
        original = state['present'] & state['is_cav']
        mask = cb.native_lane_mask(state)
        self.assertFalse(bool(mask[0, 0])); self.assertTrue(bool(mask[0, 1]))
        out = cb(state)
        self.assertEqual(int(out[0, 0, 1]), 0)
        self.assertEqual(float(out[0, 0, 3]), -.5)
        self.assertEqual(int(out[0, 2, 1]), 1)  # nonactor retains ORIGINAL native target
        self.assertEqual(int(out[0, 3, 1]), 0)  # HDV uncommanded
        self.assertGreaterEqual(len(safety_calls), 2)
        state['lane'] = out[..., 1].long()
        self.assertFalse(bool(cb.native_lane_mask(state)[0, 0]))  # hold one control interval
        cb(state)
        self.assertTrue(runtime.events[0]['next_halfstep_lane_confirmed'])
        # t=1: ramp now receives its OWN request, not native first-passage.
        mask = cb.native_lane_mask(state)
        self.assertTrue(bool(mask[0, 0])); self.assertFalse(bool(mask[0, 1]))
        out = cb(state)
        self.assertEqual(int(out[0, 1, 1]), 1)
        self.assertFalse(bool(state['cav_wm_merge_law'][0, 1]))

    def test_member_release_drops_only_that_actors_lane_and_residual(self):
        s = make_schedule('open_gap_merge', 'main', 'ramp', ramp_offset=.3)
        joint, solo_main, solo_ramp = action_set(s)
        self.assertEqual(solo_main['wm_active_members'], ['main'])
        self.assertEqual(solo_ramp['wm_active_members'], ['ramp'])
        r = self.runtime(schedule=s, active=['main'])
        state = self.state()
        r.apply(state, state['native_targets']); r.apply(state, state['native_targets'])
        mask, *_ = r.planned(state)
        self.assertFalse(bool(mask[0, 1]))  # main-only solo cannot issue ramp lane request

    def test_stale_source_absent_and_hdv_actor_denied(self):
        for field, value in (('lane', 0), ('present', False), ('is_cav', False)):
            state = self.state(); state[field][0, 0] = value
            mask, *_ = self.runtime().planned(state)
            self.assertFalse(bool(mask[0, 0]))

    def test_no_schedule_is_bit_identical_to_native_bridge(self):
        state = self.state()
        runtime = LaneRuntime([dict(id='policy')], ['main', 'ramp', 'other', 'hdv'], {}, self.lanes)
        safety = lambda s, raw: raw.clamp(-3., 3.)
        policy = torch.tensor([[.1, -.2, .3, .4]])
        old = native_bridge(policy, None, safety)
        new = compile_bridge(native_bridge)(policy, None, safety, mining_lane=runtime)
        for _ in range(4):
            a, b = copy.deepcopy(state), copy.deepcopy(state)
            self.assertTrue(torch.equal(old.native_lane_mask(a), new.native_lane_mask(b)))
            self.assertTrue(torch.equal(old(a), new(b)))
            self.assertTrue(torch.equal(a['cav_wm_merge_law'], b['cav_wm_merge_law']))
        self.assertEqual(runtime.events, [])

    def test_raw_decoder_old_and_target_safety_calls_share_single_rng_advance(self):
        schedule = make_schedule('open_gap_merge', 'main', 'ramp', depth=.3, ramp_offset=.3)
        actions = action_set(schedule)
        query = dict(base_continuation=dict(deterministic=False,
            generator_state_after_t0=torch.Generator().manual_seed(8).get_state().tolist(),
            memory_after_t0={'main': 0., 'ramp': 0., 'other': 0.}),
            record=dict(policy_residuals={'main': 0., 'ramp': 0., 'other': 0.}))
        weights = {'actor.4.weight': torch.zeros(1, 64), 'actor.4.bias': torch.zeros(1),
                   'log_std': torch.tensor([math.log(.5)])}
        def roster(state, ids):
            return [dict(vid=v, is_cav=bool(state['is_cav'][0, i]), s_m=float(state['s_m'][0, i]))
                    for i, v in enumerate(ids) if bool(state['present'][0, i])]
        raw = raw_decoder_factory(query, weights, native_sample, None, roster, [])(actions,
                                                        ['main', 'ramp', 'other', 'hdv'])
        state = self.state(3); base = torch.full((3, 4), .1)
        first = raw(state, base, 0)
        self.assertTrue(torch.equal(first, raw(state, base, 0)))
        raw(state, base, 1)
        second = raw(state, base, 2)
        self.assertTrue(torch.equal(second, raw(state, base, 2)))
        # Both solos use the same base innovation; only their own role offset differs.
        self.assertEqual(second[0, 0], second[1, 0])
        self.assertEqual(second[0, 1], second[2, 1])
        self.assertEqual(float(first[1, 1]), float(base[1, 1]))
        self.assertEqual(float(first[2, 0]), float(base[2, 0]))

    def test_compile_option_requires_exact_insertion_abi(self):
        compiled = compile_option(option_fixture, lambda *args: None, lambda *args: None)
        self.assertEqual(compiled.__name__, 'option_fixture')
        with self.assertRaises(ValueError): compile_bridge(option_fixture)


if __name__ == '__main__':
    unittest.main()
