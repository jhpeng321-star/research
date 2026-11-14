"""CPU tests for the cooperative move library (tools/coopmwm/moves.py, T1A).

No SUMO, no WM checkpoint, no GPU: candidate generation runs on synthetic
observable rows; from_plan runs on plan dicts produced by the REAL bank-format
code paths (mppi_search.build_plan, the variety generators, plans.validate) --
the record formats are read from the replay_bank/bank code, never from data
files. The feasibility clamp mirrors feasible.issued_target / damping.
direct_target with the shared spec's explicit CAV cap.
"""
import copy
import json
import unittest
from itertools import combinations

import numpy as np

from . import mppi_search as ms
from . import moves as mv
from . import plans as pl
from . import variety as va
from .test_mppi_search_cpu import BOUNDS, T

REG = va.VARIETY_REGISTERED
ONSET_LO, ONSET_HI = REG['variety_onset_window_s']
HOLD_LO, HOLD_HI = REG['variety_hold_window_s']
DEPTH_LO, DEPTH_HI = REG['variety_depth_window']


def row(vid, s, lane, cav=True, v=20.):
    return dict(vid=vid, s_m=float(s), v_mps=float(v), lane=lane, is_cav=cav, length_m=5.)


def obs(rows, self_vid):
    return dict(self=self_vid, rows=list(rows))


MAIN_ROWS = [row('m0', -30., 1), row('m1', -45., 1), row('r0', -50., 2, v=18.),
             row('c0', -20., 0), row('h0', -25., 0, cav=False)]


def block_group(rows):
    return ms.group_components(list(rows))[0]


def block_plan(members_rows, blocks, duration=60, P_value=0.):
    """A real mppi_search profile plan whose perturbation is a known per-second
    block per member: blocks = {member_index: (start, end, value)}."""
    group = block_group(members_rows)
    m = len(group['members'])
    P = np.full((m, T), float(P_value))
    absolute = P.copy()
    for i, (start, end, value) in blocks.items():
        absolute[i, start:end] = value
    return ms.build_plan(group, P, None, duration, BOUNDS, 'sampled', 'free', 'T',
                         absolute=absolute, feasibility=None)


def offset_error(plan, move):
    """Independent full-record metric, including omitted members and release padding."""
    recorded = np.asarray(plan['perturbation_profile'], float)
    horizon = max(plan['duration_s'], int(ONSET_HI+HOLD_HI))
    target = np.zeros((len(plan['members']), horizon))
    target[:, :plan['duration_s']] = recorded
    predicted = np.array([[mv.offsets(move, t).get(vid, 0.) for t in range(horizon)]
                          for vid in plan['members']])
    return float(np.square(target-predicted).sum())


class Families(unittest.TestCase):
    def test_registered_families(self):
        self.assertEqual(mv.FAMILIES, ('none', 'ease_off', 'wall', 'gap_open', 'ramp_assist'))
        self.assertEqual(mv.STRUCTURED, mv.FAMILIES[1:])
        # the variety families survive by name; variety 'free' projects onto ease_off
        self.assertTrue({'wall', 'gap_open', 'ramp_assist'} <= set(mv.FAMILIES))
        self.assertEqual(set(va.FAMILIES)-{'free'}, set(mv.FAMILIES)-{'none', 'ease_off'})

    def test_member_arity_is_the_v2v_partner_arity(self):
        from .v2v_obs import V2V_K
        self.assertEqual(mv.MOVE_MAX_MEMBERS, int(V2V_K))
        self.assertEqual(mv.CONTROL_DT_S, 1.0)


class MoveValidation(unittest.TestCase):
    def test_valid_structured_move(self):
        move = mv.Move('wall', ('a', 'b'), 3, 40, 0.5)
        self.assertEqual(move.actors, ('a', 'b'))
        self.assertEqual(mv.Move('ease_off', ('a',), 0, 20, 0.3).actors, ('a',))
        self.assertEqual(mv.Move('gap_open', ('a', 'b'), 30, 90, 1.0).actors, ('a',))

    def test_none_move_is_the_zero_move(self):
        move = mv.none_move('m0')
        self.assertEqual((move.family, move.members, move.onset, move.hold, move.depth),
                         ('none', ('m0',), 0, 0, 0.0))
        self.assertEqual(move.actors, ())

    def test_rejects_bad_family_and_members(self):
        with self.assertRaises(ValueError):
            mv.Move('fly', ('a', 'b'), 1, 20, 0.5)
        with self.assertRaises(ValueError):
            mv.Move('wall', (), 1, 20, 0.5)
        with self.assertRaises(ValueError):
            mv.Move('wall', ('a', 'b', 'c', 'd', 'e'), 1, 20, 0.5)
        with self.assertRaises(ValueError):
            mv.Move('wall', ('a', 'a'), 1, 20, 0.5)
        with self.assertRaises(ValueError):
            mv.Move('ease_off', ('a', 'b'), 1, 20, 0.5)
        with self.assertRaises(ValueError):
            mv.Move('wall', ('a',), 1, 20, 0.5)

    def test_rejects_parameters_outside_the_registered_windows(self):
        for kw in (dict(onset=-1, hold=20), dict(onset=int(ONSET_HI)+1, hold=20),
                   dict(onset=1.5, hold=20), dict(onset=1, hold=19), dict(onset=1, hold=91),
                   dict(onset=1, hold=20.5)):
            with self.assertRaises(ValueError):
                mv.Move('wall', ('a', 'b'), depth=0.5, **kw)
        for depth in (DEPTH_LO-0.01, DEPTH_HI+0.01, float('nan')):
            with self.assertRaises(ValueError):
                mv.Move('wall', ('a', 'b'), 1, 20, depth)

    def test_rejects_nonzero_none_parameters(self):
        with self.assertRaises(ValueError):
            mv.Move('none', ('a',), 1, 20, 0.5)
        with self.assertRaises(ValueError):
            mv.Move('none', ('a', 'b'), 0, 0, 0.0)

    def test_dict_round_trip_and_json(self):
        move = mv.Move('gap_open', ('a', 'b'), 7, 40, 0.8)
        restored = mv.Move.from_dict(json.loads(json.dumps(move.to_dict())))
        self.assertEqual(restored, move)

    def test_members_are_frozen_and_deserialisation_does_not_truncate(self):
        members = ['a', 'b']
        move = mv.Move('wall', members, 2, 20, 0.5)
        members.append('c')
        self.assertEqual(move.members, ('a', 'b'))
        with self.assertRaises(ValueError):
            mv.Move.from_dict(dict(family='wall', members=['a', 'b'],
                                   onset=2.5, hold=20, depth=0.5))


class Offsets(unittest.TestCase):
    def test_none_offsets_are_zero(self):
        for t in (0, 10, 500):
            self.assertEqual(mv.offsets(mv.none_move('a'), t), {'a': 0.0})

    def test_ease_off_shape(self):
        move = mv.Move('ease_off', ('a',), 5, 20, 0.5)
        self.assertEqual(mv.offsets(move, 4), {'a': 0.0})
        for t in range(5, 25):
            self.assertEqual(mv.offsets(move, t), {'a': -0.5})
        self.assertEqual(mv.offsets(move, 25), {'a': 0.0})
        self.assertEqual(mv.offsets(move, 10**6), {'a': 0.0})

    def test_wall_moves_every_member_together(self):
        move = mv.Move('wall', ('a', 'b', 'c'), 2, 30, 0.7)
        self.assertEqual(mv.offsets(move, 1), {'a': 0., 'b': 0., 'c': 0.})
        self.assertEqual(mv.offsets(move, 2), {'a': -0.7, 'b': -0.7, 'c': -0.7})
        self.assertEqual(mv.offsets(move, 31), {'a': -0.7, 'b': -0.7, 'c': -0.7})
        self.assertEqual(mv.offsets(move, 32), {'a': 0., 'b': 0., 'c': 0.})

    def test_gap_open_acts_through_members0_only(self):
        move = mv.Move('gap_open', ('eased', 'merger'), 10, 25, 0.4)
        self.assertEqual(mv.offsets(move, 10), {'eased': -0.4, 'merger': 0.0})
        self.assertEqual(mv.offsets(move, 34), {'eased': -0.4, 'merger': 0.0})

    def test_ramp_assist_moves_both_actors(self):
        move = mv.Move('ramp_assist', ('ahead', 'merger'), 3, 45, 0.9)
        self.assertEqual(mv.offsets(move, 3), {'ahead': -0.9, 'merger': -0.9})
        self.assertEqual(mv.offsets(move, 48), {'ahead': 0., 'merger': 0.})

    def test_second_argument_contract(self):
        move = mv.Move('ease_off', ('a',), 1, 20, 0.3)
        with self.assertRaises(ValueError):
            mv.offsets(move, -1)
        with self.assertRaises(ValueError):
            mv.offsets(move, 2.5)
        self.assertEqual(mv.offsets(move, 3.0), {'a': -0.3})   # integral float second
        with self.assertRaises(ValueError):
            mv.offsets('wall', 0)


class OffsetFeasibility(unittest.TestCase):
    MOVE = mv.Move('ease_off', ('a',), 0, 40, 0.5)

    def state(self, **kw):
        base = dict(base={'a': 0.0}, v_mps={'a': 20.}, allowed={'a': 25.},
                    accel_bounds=(-3., 2.6))
        base.update(kw)
        return base

    def test_unresolvable_member_keeps_the_raw_offset(self):
        raw = mv.offsets(self.MOVE, 5)
        self.assertEqual(mv.offsets(self.MOVE, 5, self.state()), raw)
        self.assertEqual(mv.offsets(self.MOVE, 5, self.state(v_mps={})), raw)
        self.assertEqual(mv.offsets(self.MOVE, 5, self.state(accel_bounds=None)), raw)

    def test_offset_respects_the_direct_target_band(self):
        # base at the deceleration floor (lo = max(bounds0, -v/dt) = -2): no further
        # deceleration is issuable, so the offset collapses to 0
        state = self.state(base={'a': -2.0}, v_mps={'a': 2.0})
        self.assertEqual(mv.offsets(self.MOVE, 5, state), {'a': 0.0})
        # free headroom: the raw offset survives unchanged
        state = self.state(base={'a': -0.5}, v_mps={'a': 20.})
        self.assertEqual(mv.offsets(self.MOVE, 5, state), {'a': -0.5})

    def test_infeasible_second_gets_no_offset(self):
        # allowed far below the current speed: hi = (allowed-v)/dt = -4.5 < lo = -3,
        # damping.direct_target returns None -> no offset
        state = self.state(v_mps={'a': 31.5}, allowed={'a': 27.0})
        self.assertEqual(mv.offsets(self.MOVE, 5, state), {'a': 0.0})

    def test_clamp_is_per_member(self):
        move = mv.Move('wall', ('a', 'b'), 0, 40, 0.5)
        state = dict(base={'a': -2.0, 'b': 0.0}, v_mps={'a': 2.0, 'b': 20.},
                     allowed={'a': 25., 'b': 25.}, accel_bounds=(-3., 2.6))
        out = mv.offsets(move, 5, state)
        self.assertEqual(out, {'a': 0.0, 'b': -0.5})

    def test_clamp_final_command_against_original_base(self):
        # MAPPO's above-cap command must not swallow the correction. Applying
        # the returned offset to the actual base reaches the issuable cap.
        state = self.state(base={'a': 2.6})
        off = mv.offsets(self.MOVE, 5, state)['a']
        self.assertAlmostEqual(state['base']['a']+off, 1.2)
        state = self.state(base={'a': 0.9}, allowed={'a': 20.1})
        off = mv.offsets(self.MOVE, 5, state)['a']
        self.assertAlmostEqual(state['base']['a']+off, 0.1)
        state = self.state(base={'a': 0.9}, accel_bounds=(-3., 0.2))
        off = mv.offsets(self.MOVE, 5, state)['a']
        self.assertAlmostEqual(state['base']['a']+off, 0.2)

    def test_none_and_release_preserve_base_identity_with_clamp_context(self):
        state = self.state(base={'a': 2.6})
        self.assertEqual(mv.offsets(mv.none_move('a'), 0, state), {'a': 0.})
        self.assertEqual(mv.offsets(self.MOVE, 40, state), {'a': 0.})

    def test_wall_releases_at_the_observed_merge_boundary(self):
        move = mv.Move('wall', ('a', 'b'), 0, 40, 0.5)
        zone_start = REG['variety_merge_zone_m'][0]
        self.assertEqual(mv.offsets(move, 5, dict(s_m={'a': zone_start-1, 'b': zone_start-2})),
                         {'a': -0.5, 'b': -0.5})
        self.assertEqual(mv.offsets(move, 5, dict(s_m={'a': zone_start, 'b': zone_start-1})),
                         {'a': 0., 'b': 0.})


class Candidates(unittest.TestCase):
    def test_count_members_and_windows(self):
        out = mv.candidates(obs(MAIN_ROWS, 'm0'), np.random.default_rng(7), 12)
        self.assertEqual(len(out), 12)
        vids = {r['vid'] for r in MAIN_ROWS}
        for move in out:
            self.assertIsInstance(move, mv.Move)
            self.assertIn('m0', move.members)
            self.assertTrue(set(move.members) <= vids)
            self.assertNotIn('h0', move.members)          # HDVs are never members
            self.assertLessEqual(len(move.members), mv.MOVE_MAX_MEMBERS)
            if move.family != 'none':
                self.assertTrue(ONSET_LO <= move.onset <= ONSET_HI)
                self.assertTrue(HOLD_LO <= move.hold <= HOLD_HI)
                self.assertTrue(DEPTH_LO <= move.depth <= DEPTH_HI)

    def test_deterministic_given_the_rng_seed(self):
        a = mv.candidates(obs(MAIN_ROWS, 'm0'), np.random.default_rng(11), 9)
        b = mv.candidates(obs(MAIN_ROWS, 'm0'), np.random.default_rng(11), 9)
        self.assertEqual(a, b)
        self.assertTrue(any(x.depth != y.depth for x, y in zip(a, mv.candidates(
            obs(MAIN_ROWS, 'm0'), np.random.default_rng(12), 9))))

    def test_merge_geometry_has_easing_and_merge_families(self):
        families = {m.family for m in mv.candidates(obs(MAIN_ROWS, 'm0'),
                                                    np.random.default_rng(3), 40)}
        self.assertTrue({'ease_off', 'gap_open', 'ramp_assist'} <= families)
        self.assertNotIn('wall', families)    # these CAVs are already in the merge zone
        self.assertNotIn('none', families)     # the caller rolls 'none' out separately

    def test_wall_is_upstream_and_releases_by_observable_merge_arrival(self):
        rows = [row('a', -180., 0, v=2.), row('b', -190., 1, v=2.),
                row('outside_span', -270., 1, v=2.)]
        walls = [m for m in mv.candidates(obs(rows, 'a'), np.random.default_rng(7), 40)
                 if m.family == 'wall']
        self.assertTrue(walls)
        by_id = {r['vid']: r for r in rows}
        for move in walls:
            self.assertNotIn('outside_span', move.members)
            for vid in move.members:
                r = by_id[vid]
                self.assertLess(r['s_m'], REG['variety_merge_zone_m'][0])
                arrival = (REG['variety_merge_zone_m'][0]-r['s_m'])/r['v_mps']
                self.assertLessEqual(move.onset+move.hold, arrival)

    def test_wall_unavailable_when_registered_hold_cannot_finish_upstream(self):
        rows = [row('a', -110., 0), row('b', -120., 1)]
        self.assertNotIn('wall', {m.family for m in mv.candidates(
            obs(rows, 'a'), np.random.default_rng(7), 20)})

    def test_gap_open_actor_order(self):
        moves = [m for m in mv.candidates(obs(MAIN_ROWS, 'm0'), np.random.default_rng(5), 40)
                 if m.family == 'gap_open']
        self.assertTrue(moves)
        for move in moves:
            self.assertEqual(move.members[0], 'm0')        # the easing mainline CAV acts
            self.assertEqual(move.members[1], 'r0')

    def test_ramp_assist_from_the_merger_side(self):
        rows = [row('r0', -50., 2, v=18.), row('m0', -30., 1), row('m1', -45., 1),
                row('c0', -20., 0)]
        moves = [m for m in mv.candidates(obs(rows, 'r0'), np.random.default_rng(5), 40)
                 if m.family == 'ramp_assist']
        self.assertTrue(moves)
        for move in moves:
            self.assertIn('r0', move.members)
            self.assertIn('m0', move.members)              # the mainline CAV ahead

    def test_wall_needs_two_lanes(self):
        rows = [row('m0', -30., 1), row('m1', -45., 1), row('m2', -60., 1)]
        families = {m.family for m in mv.candidates(obs(rows, 'm0'),
                                                    np.random.default_rng(3), 40)}
        self.assertNotIn('wall', families)
        self.assertTrue(families <= {'ease_off'})

    def test_merge_families_need_the_acting_cav_inside_the_zone(self):
        rows = [row('m0', -500., 1), row('m1', -45., 1), row('r0', -50., 2, v=18.),
                row('c0', -20., 0)]
        families = {m.family for m in mv.candidates(obs(rows, 'm0'),
                                                    np.random.default_rng(3), 40)}
        self.assertTrue(families <= {'ease_off', 'wall'})
        self.assertEqual(families, {'ease_off'})   # wall also needs in-range partners

    def test_no_ramp_row_no_merge_families(self):
        rows = [row('m0', -30., 1), row('m1', -45., 1), row('c0', -20., 0)]
        families = {m.family for m in mv.candidates(obs(rows, 'm0'),
                                                    np.random.default_rng(3), 40)}
        self.assertNotIn('gap_open', families)
        self.assertNotIn('ramp_assist', families)

    def test_members_stay_within_the_v2v_range(self):
        rows = [row('m0', -30., 1), row('far', -1000., 0), row('r0', -50., 2, v=18.)]
        for move in mv.candidates(obs(rows, 'm0'), np.random.default_rng(3), 40):
            self.assertNotIn('far', move.members)

    def test_empty_and_invalid_calls(self):
        self.assertEqual(mv.candidates(obs(MAIN_ROWS, 'm0'), np.random.default_rng(1), 0), [])
        with self.assertRaises(ValueError):
            mv.candidates(obs(MAIN_ROWS, 'm0'), np.random.default_rng(1), -1)
        with self.assertRaises(ValueError):
            mv.candidates(obs(MAIN_ROWS, 'm0'), None, 3)
        with self.assertRaises(ValueError):
            mv.candidates(obs(MAIN_ROWS, 'ghost'), np.random.default_rng(1), 3)
        with self.assertRaises(ValueError):
            mv.candidates(obs([r for r in MAIN_ROWS if r['vid'] != 'm0'], 'm0'),
                          np.random.default_rng(1), 3)
        with self.assertRaises(ValueError):
            mv.candidates(obs([row('h0', -25., 0, cav=False)], 'h0'),
                          np.random.default_rng(1), 3)
        bad = row('m0', -30., 1)
        del bad['v_mps']
        with self.assertRaises(ValueError):
            mv.candidates(obs([bad], 'm0'), np.random.default_rng(1), 3)
        with self.assertRaises(ValueError):
            mv.candidates(obs([MAIN_ROWS[0]]*2, 'm0'), np.random.default_rng(1), 3)
        with self.assertRaises(ValueError):
            mv.candidates(obs([row('m0', float('nan'), 1)], 'm0'), np.random.default_rng(1), 3)


class FromPlan(unittest.TestCase):
    def test_profile_plan_known_block(self):
        plan = block_plan([row('a', -50., 1), row('b', -60., 1)], {0: (10, 40, -0.5)})
        np.testing.assert_allclose(np.asarray(plan['perturbation_profile'])[0, 10:40], -0.5)
        move = mv.from_plan(plan)
        self.assertEqual(move.family, 'ease_off')          # one active member, untagged
        self.assertEqual(move.members, ('a',))             # the move acts through that member
        self.assertEqual((move.onset, move.hold), (10, 30))
        self.assertAlmostEqual(move.depth, 0.5)

    def test_variety_family_tag_only_breaks_equal_distance_ties(self):
        plan = block_plan([row('a', -50., 1), row('b', -60., 1)], {0: (10, 40, -0.5)})
        self.assertEqual(mv.from_plan(dict(plan, variety_family='gap_open')).family, 'gap_open')
        self.assertEqual(mv.from_plan(dict(plan, variety_family='wall')).family, 'ease_off')
        aligned = block_plan([row('a', -50., 1), row('b', -60., 1)],
                             {0: (10, 40, -0.5), 1: (10, 40, -0.5)})
        self.assertEqual(mv.from_plan(dict(aligned, variety_family='ramp_assist')).family,
                         'ramp_assist')

    def test_untagged_equal_multi_actor_shapes_use_canonical_wall_family(self):
        two = [row('a', -50., 1), row('b', -60., 1)]
        aligned = block_plan(two, {0: (10, 40, -0.5), 1: (10, 40, -0.5)})
        self.assertEqual(mv.from_plan(aligned).family, 'wall')
        staggered = block_plan(two, {0: (10, 40, -0.5), 1: (20, 50, -0.5)})
        self.assertEqual(mv.from_plan(staggered).family, 'wall')

    def test_variety_generated_wall_plan(self):
        vcfg = va.split_config(dict(variety_families=True))
        rows = [row('w0', -30., 0, v=21.), row('w1', -35., 1, v=20.), row('w2', -40., 0, v=19.)]
        group = block_group(rows)
        self.assertGreaterEqual(len(group['lanes']), 2)
        P = np.full((len(group['members']), T), 0.2)
        plan = va.wall_candidate(group, P, BOUNDS, 'free', np.random.default_rng(3),
                                 vcfg, None, [60])
        self.assertIsNotNone(plan)
        move = mv.from_plan(plan)
        self.assertEqual(move.family, 'wall')
        self.assertEqual(move.members, tuple(group['members']))
        self.assertTrue(ONSET_LO <= move.onset <= ONSET_HI)
        self.assertTrue(HOLD_LO <= move.hold <= HOLD_HI)
        self.assertTrue(DEPTH_LO <= move.depth <= DEPTH_HI)

    def _variety_merge_plan(self, generator, ramp_speed=18.):
        vcfg = va.split_config(dict(variety_families=True))
        rows = [row('m1', -50., 1, v=20.), row('m0', -60., 1, v=19.),
                 row('r0', -70., 2, v=ramp_speed)]
        group = block_group(rows)
        P = np.full((len(group['members']), T), 0.2)
        return group, generator(group, P, BOUNDS, 'free', np.random.default_rng(4),
                                vcfg, None, [60])

    def test_variety_generated_gap_open_plan(self):
        group, plan = self._variety_merge_plan(va.gap_open_candidate)
        self.assertIsNotNone(plan)
        move = mv.from_plan(plan)
        self.assertEqual(move.family, 'gap_open')
        self.assertEqual(len(move.members), 2)  # zero-offset partners do not change distance
        self.assertTrue(set(move.members) <= set(group['members']))
        # Seed 4 eases m0, which is NOT the first member in the bank plan.
        self.assertEqual(move.actors, ('m0',))
        self.assertLess(np.asarray(plan['perturbation_profile'])[group['members'].index('m0')].sum(), 0.)
        self.assertTrue(ONSET_LO <= move.onset <= ONSET_HI
                        and HOLD_LO <= move.hold <= HOLD_HI
                        and DEPTH_LO <= move.depth <= DEPTH_HI)

    def test_variety_generated_ramp_assist_plan(self):
        group, plan = self._variety_merge_plan(va.ramp_assist_candidate, ramp_speed=20.)
        self.assertIsNotNone(plan)
        move = mv.from_plan(plan)
        self.assertEqual(move.family, 'ramp_assist')
        self.assertTrue(set(move.members) <= set(group['members']))
        self.assertEqual(set(move.members), {'m1', 'r0'})

    def test_legacy_knots_plan(self):
        plan = pl.validate(dict(id='X', members=['a', 'b'], duration_s=30,
                                knots=[[0, 0, -1, 0, 0, 0, 0], [0.5]*7],
                                family='O1', origin='inherited'), BOUNDS)
        plan['policy_profile'] = [[0.]*30, [0.5]*30]
        move = mv.from_plan(plan)
        self.assertEqual(move.family, 'ease_off')       # member a deviates, b is constant
        self.assertEqual(move.members, ('a',))          # the single active member
        self.assertTrue(ONSET_LO <= move.onset <= ONSET_HI)
        self.assertTrue(HOLD_LO <= move.hold <= HOLD_HI)
        self.assertTrue(DEPTH_LO <= move.depth <= DEPTH_HI)
        self.assertNotIn(move.family, ('O1',))          # legacy family labels are not move families

    def test_zero_perturbation_is_none_and_metadata_cannot_override_offsets(self):
        group = block_group([row('a', -50., 1), row('b', -60., 1)])
        P = np.zeros((2, T))
        plan = ms.build_plan(group, P, None, 60, BOUNDS, 'centre', 'free', 'T',
                             absolute=P, feasibility=None)
        self.assertEqual(mv.from_plan(plan), mv.none_move('a'))
        tagged = block_plan([row('a', -50., 1), row('b', -60., 1)], {0: (10, 40, -0.5)})
        self.assertEqual(mv.from_plan(dict(tagged, policy_duplicate=True)), mv.from_plan(tagged))

    def test_six_member_plan_truncates_to_the_move_arity(self):
        rows = [row(f'v{i}', -10.-50*i, 1) for i in range(6)]
        plan = block_plan(rows, {i: (5, 25, -depth) for i, depth in enumerate(
            (0.8, 0.4, 0.5, 0.3, 0.35, 0.6))})
        self.assertEqual(len(plan['members']), 6)
        move = mv.from_plan(plan)
        self.assertEqual(len(move.members), mv.MOVE_MAX_MEMBERS)
        # the largest-offset members survive; kept members stay in plan order
        self.assertEqual(move.members, ('v0', 'v1', 'v2', 'v5'))
        self.assertEqual((move.onset, move.hold), (5, 20))
        self.assertAlmostEqual(move.depth, (0.8+0.4+0.5+0.6)/4)

    def test_r_arm_random_candidate_format(self):
        # the exact R-arm centre candidate format (mppi_search.sample_unscored)
        group = block_group([row('a', -50., 1), row('b', -60., 1)])
        P = np.full((2, T), 0.2)
        plan = ms.sample_unscored(group, P, BOUNDS, 'free', 12345)
        move = mv.from_plan(plan)
        self.assertIn(move.family, mv.FAMILIES)
        self.assertTrue(set(move.members) <= set(group['members']))

    def test_chosen_plan_receipt_shape(self):
        # sumo_verify receipts carry chosen_plan = deepcopy(candidate plan dict)
        plan = block_plan([row('a', -50., 1), row('b', -60., 1)], {0: (10, 40, -0.5)})
        plan['variety_family'] = 'wall'
        receipt = dict(chosen_plan=copy.deepcopy(plan))
        self.assertEqual(mv.from_plan(receipt), mv.from_plan(plan))
        self.assertEqual(mv.from_plan(receipt['chosen_plan']).family, 'ease_off')
        self.assertIsInstance(json.dumps(mv.from_plan(receipt['chosen_plan']).to_dict()), str)

    def test_all8_per_seed_example_uses_its_own_candidate(self):
        # verify_perseed.build_example persists these keys; replay_bank.bank_pairs
        # selects ex['plan'] over receipt['chosen_plan'], even across seed winners.
        first = block_plan([row('a', -50., 1), row('b', -60., 1)], {0: (0, 20, -0.4)})
        second = block_plan([row('a', -50., 1), row('b', -60., 1)], {1: (10, 40, -0.8)})
        record = dict(chosen_plan=first, bank_rule='per_seed_v1', verify_mode='all8_both_seeds',
                      per_seed_v1=dict(examples=[dict(seed_index=2, candidate_index=1,
                          example_id='synthetic#c2#s2', plan=copy.deepcopy(second), admitted=True)]))
        label = mv.from_plan(record['per_seed_v1']['examples'][0])
        self.assertEqual((label.family, label.members, label.onset, label.hold),
                         ('ease_off', ('b',), 10, 30))
        self.assertAlmostEqual(label.depth, 0.8)
        self.assertNotEqual(label, mv.from_plan(record))

    def test_positive_offsets_and_short_impulses_prefer_none(self):
        rows = [row('a', -50., 1), row('b', -60., 1)]
        positive = block_plan(rows, {0: (10, 40, 0.5)})
        self.assertEqual(mv.from_plan(positive).family, 'none')
        # A 1-second tiny dip is closer to none than any registered 20s box.
        impulse = block_plan(rows, {0: (10, 11, -0.01)})
        self.assertEqual(mv.from_plan(impulse).family, 'none')

    def test_depth_projection_is_least_squares_not_absolute_magnitude(self):
        rows = [row('a', -50., 1), row('b', -60., 1)]
        plan = block_plan(rows, {0: (5, 35, -0.4)})
        plan['perturbation_profile'][0][5:15] = [-1.]*10
        move = mv.from_plan(plan)
        self.assertEqual((move.onset, move.hold), (5, 30))
        self.assertAlmostEqual(move.depth, (10.+20*0.4)/30)

    def test_absolute_profile_with_recorded_policy_and_invalid_knots(self):
        plan = dict(members=['a'], duration_s=60, profile=[[0.]*10+[-0.5]*20+[0.]*30],
                    policy_profile=[[0.]*60])
        self.assertEqual(mv.from_plan(plan), mv.Move('ease_off', ('a',), 10, 20, 0.5))
        for knots, stride in (([0., 1.], 10), ([[0., 1.]], 0), ([[0., float('nan')]], 10)):
            with self.assertRaises(ValueError):
                mv.from_plan(dict(members=['a'], duration_s=30, knots=knots, knot_s=stride))

    def test_tagged_wall_subset_has_same_family_exact_match(self):
        rows = [row(v, -50.-i*10, 1) for i, v in enumerate('abcd')]
        plan = dict(block_plan(rows, {0: (0, 20, -0.5), 1: (0, 20, -0.5)}, duration=120),
                    variety_family='wall')
        oracle = mv.Move('wall', ('a', 'b'), 0, 20, 0.5)
        self.assertEqual(offset_error(plan, oracle), 0.)
        self.assertEqual(mv.from_plan(plan), oracle)
        self.assertEqual(offset_error(plan, mv.from_plan(plan)), 0.)

    def test_staggered_weak_partner_does_not_exclude_closer_single_actor(self):
        plan = block_plan([row('a', -50., 1), row('b', -60., 1)],
                          {0: (0, 20, -0.8), 1: (100, 120, -0.3)}, duration=120)
        oracle = mv.Move('ease_off', ('a',), 0, 20, 0.8)
        move = mv.from_plan(plan)
        self.assertEqual((move.family, move.members, move.onset, move.hold),
                         ('ease_off', ('a',), 0, 20))
        self.assertAlmostEqual(move.depth, oracle.depth)
        self.assertAlmostEqual(offset_error(plan, move), offset_error(plan, oracle))
        self.assertAlmostEqual(offset_error(plan, move), 1.8)

    def test_tagged_wall_with_one_active_member_is_exact_ease_off(self):
        rows = [row(v, -50.-i*10, 1) for i, v in enumerate('abcd')]
        plan = dict(block_plan(rows, {0: (0, 20, -0.5)}, duration=120), variety_family='wall')
        self.assertEqual(mv.from_plan(plan), mv.Move('ease_off', ('a',), 0, 20, 0.5))
        self.assertEqual(offset_error(plan, mv.from_plan(plan)), 0.)

    def test_member_selection_depends_on_window_fit_not_total_braking_energy(self):
        rows = [row(v, -50.-i*10, 1) for i, v in enumerate('abcdef')]
        # f has the highest braking energy but poor overlap with the best box.
        plan = block_plan(rows, {**{i: (0, 20, -0.8) for i in range(5)},
                                 5: (100, 120, -1.)}, duration=120)
        move = mv.from_plan(plan)
        self.assertEqual((move.family, move.members, move.onset, move.hold),
                         ('wall', tuple('abcd'), 0, 20))
        self.assertAlmostEqual(move.depth, 0.8)

    def test_projection_matches_exhaustive_distance_oracle(self):
        # Independent direct-distance oracle enumerates all actor shapes,
        # integer windows, and each window's constrained optimal depth. Gap
        # and ramp families have identical offset shapes to ease_off and wall.
        rng = np.random.default_rng(743)
        recorded = rng.uniform(-0.8, 0.3, (3, 135))
        recorded[1:, 7:48] -= 0.4
        plan = dict(members=list('abc'), duration_s=135,
                    perturbation_profile=recorded.tolist(), variety_family='gap_open')
        best_error = float(np.square(recorded).sum())  # none
        for count in range(1, 4):
            for actors in combinations(range(3), count):
                for onset in range(int(ONSET_LO), int(ONSET_HI)+1):
                    for hold in range(int(HOLD_LO), int(HOLD_HI)+1):
                        mask = np.zeros_like(recorded)
                        mask[list(actors), onset:onset+hold] = -1.
                        depth = float(np.clip((recorded*mask).sum()/np.square(mask).sum(),
                                              DEPTH_LO, DEPTH_HI))
                        error = float(np.square(recorded-depth*mask).sum())
                        best_error = min(best_error, error)
        self.assertAlmostEqual(offset_error(plan, mv.from_plan(plan)), best_error, places=10)

    def test_projection_charges_for_release_beyond_short_record(self):
        plan = dict(members=['a'], duration_s=10, perturbation_profile=[[-0.5]*10])
        # A 20-second hold is not truncated to the 10-second record. The
        # unconstrained depth 0.25 clips to the registered lower bound 0.3.
        move = mv.from_plan(plan)
        self.assertEqual(move, mv.Move('ease_off', ('a',), 0, 20, 0.3))
        self.assertAlmostEqual(offset_error(plan, move), 1.3)

    def test_identical_absolute_profiles_require_different_labels_under_recorded_policies(self):
        absolute = dict(members=['a', 'b'], duration_s=60, profile=[[-0.5]*60, [0.]*60])
        zero_policy = dict(absolute, policy_profile=[[0.]*60, [0.]*60])
        same_policy = dict(absolute, policy_profile=copy.deepcopy(absolute['profile']))
        self.assertEqual(mv.from_plan(zero_policy), mv.Move('ease_off', ('a',), 0, 60, 0.5))
        self.assertEqual(mv.from_plan(same_policy), mv.none_move('a'))
        self.assertEqual(mv.from_plan(dict(absolute, perturbation_profile=absolute['profile'])),
                         mv.from_plan(zero_policy))
        with self.assertRaisesRegex(ValueError, 'require a recorded policy_profile'):
            mv.from_plan(absolute)

    def test_validated_constant_legacy_knots_require_recorded_policy(self):
        plan = pl.validate(dict(id='X', members=['a', 'b'], duration_s=60,
                                knots=[[-0.5]*7, [0.]*7], family='O1', origin='inherited'), BOUNDS)
        with self.assertRaisesRegex(ValueError, 'require a recorded policy_profile'):
            mv.from_plan(plan)
        zero_policy = dict(plan, policy_profile=[[0.]*60, [0.]*60])
        same_policy = dict(plan, policy_profile=[[-0.5]*60, [0.]*60])
        self.assertEqual(mv.from_plan(zero_policy), mv.Move('ease_off', ('a',), 0, 60, 0.5))
        self.assertEqual(mv.from_plan(same_policy), mv.none_move('a'))

    def test_recorded_policy_is_time_aligned_not_a_constant_summary(self):
        base = np.stack([np.linspace(-0.2, 0.7, 60), np.linspace(0.5, -0.6, 60)])
        absolute = base.copy()
        absolute[1, 8:43] -= 0.6
        plan = dict(members=['a', 'b'], duration_s=60,
                    profile=absolute.tolist(), policy_profile=base.tolist())
        move = mv.from_plan(plan)
        self.assertEqual((move.family, move.members, move.onset, move.hold),
                         ('ease_off', ('b',), 8, 35))
        self.assertAlmostEqual(move.depth, 0.6)

    def test_legacy_reference_matches_actual_knot_interpolation(self):
        plan = pl.validate(dict(id='X', members=['a', 'b'], duration_s=60, knot_s=5,
                                knots=[[0., 0.1, -0.2, 0.3, -0.4, 0.5, -0.5], [0.]*7],
                                family='O1', origin='inherited'), BOUNDS)
        # Match the recorded baseline even after the final knot (the legacy
        # executor extrapolates its final segment at this explicit stride).
        plan['policy_profile'] = [[pl.acceleration_at(plan, t)[vid] for t in range(60)]
                                  for vid in plan['members']]
        self.assertEqual(mv.from_plan(plan), mv.none_move('a'))

    def test_recorded_policy_shape_finiteness_and_missing_reference_wrappers(self):
        plan = dict(members=['a', 'b'], duration_s=60, profile=[[-0.5]*60, [0.]*60])
        for policy in ([[0.]*60], [[0.]*59, [0.]*59], [0., 0.],
                       [[float('nan')]*60, [0.]*60], [[float('inf')]*60, [0.]*60]):
            with self.subTest(policy_shape=np.shape(policy)):
                with self.assertRaisesRegex(ValueError, 'policy_profile must be finite and aligned'):
                    mv.from_plan(dict(plan, policy_profile=policy))
        for wrapper in ({'plan': plan}, {'chosen_plan': plan}, dict(plan, policy_duplicate=True)):
            with self.assertRaisesRegex(ValueError, 'require a recorded policy_profile'):
                mv.from_plan(wrapper)
        plan['policy_profile'] = [[0.]*60, [0.]*60]
        for wrapper in ({'plan': plan}, {'chosen_plan': plan}):
            self.assertEqual(mv.from_plan(wrapper), mv.Move('ease_off', ('a',), 0, 60, 0.5))

    def test_invalid_plans(self):
        with self.assertRaises(ValueError):
            mv.from_plan('wall')
        with self.assertRaises(ValueError):
            mv.from_plan(dict(members=['a'], duration_s=0, profile=[[0.]]))
        with self.assertRaises(ValueError):
            mv.from_plan(dict(members=['a', 'b'], duration_s=10,
                              perturbation_profile=[[0.]*10]))
        with self.assertRaises(ValueError):
            mv.from_plan(dict(members=['a'], duration_s=10))
