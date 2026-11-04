"""CPU tests for T2E group_credit: counterfactual GROUP credit for the cooperation
layer. No SUMO, no WM checkpoint, no GPU - injected stub scorers only."""
import unittest
import copy
from dataclasses import dataclass
import numpy as np
from . import credit as cr
from . import group_credit as gc

MEMBERS = ('cav_a', 'cav_b')
BELIEF = dict(ego=dict(v=20., lane=1), partners=[dict(id='cav_b', gap=12.)],
              lead_gap=25., merge_zone=dict(start=-100., end=75.))
MOVE = dict(family='gap_open', members=list(MEMBERS), onset_s=0, hold_s=10, depth=.2)


def stub(values, calls=None):
    """values[(kind, member)] -> per-sample outcomes; internal per-key counter."""
    count = {}

    def scorer(belief, itv, rng):
        assert belief == BELIEF
        key = (itv['kind'], itv.get('member'))
        i = count.get(key, 0)
        count[key] = i + 1
        if calls is not None:
            calls.append(key)
        return values[key][i % len(values[key])]
    return scorer


def jitter_stub(base_values, spread=10):
    """Outcome = base + rng jitter: exercises the per-sample child streams."""
    def scorer(belief, itv, rng):
        return base_values[(itv['kind'], itv.get('member'))] \
            + float(rng.integers(0, spread))
    return scorer


def good_move_values():
    """A useful move: joint delay 40 vs base 50; solos 45/46."""
    return {('base', None): [50.] * 4, ('joint', None): [40.] * 4,
            ('alone', 'cav_a'): [45.] * 4, ('alone', 'cav_b'): [46.] * 4,
            ('revert', 'cav_a'): [46.] * 4, ('revert', 'cav_b'): [45.] * 4}


class Config(unittest.TestCase):
    def test_defaults_are_registration_subset(self):
        cfg = gc.credit_config(None)
        self.assertEqual(set(cfg), gc.CONFIG_KEYS)
        for k, v in cfg.items():
            self.assertEqual(v, gc.REGISTRATION[k])

    def test_unknown_keys_rejected(self):
        with self.assertRaises(ValueError):
            gc.credit_config(dict(group_credit_bonus=1.))
        with self.assertRaises(ValueError):
            gc.credit_config(dict(metric='progress'))

    def test_type_validation(self):
        for bad in (dict(sumo_verify_fraction=1.5), dict(sumo_verify_fraction=True),
                    dict(sumo_verify_fraction=-.1), dict(c_g_share=2.),
                    dict(sumo_verify_fraction=float('nan')),
                    dict(sumo_verify_fraction=float('inf')),
                    dict(wm_samples=0), dict(wm_samples=2.0),
                    dict(verify_mode='per_seed'), dict(isolated=False),
                    dict(direction='higher')):
            with self.assertRaises(ValueError):
                gc.credit_config(bad)

    def test_valid_override(self):
        cfg = gc.credit_config(dict(wm_samples=2, sumo_verify_fraction=0.))
        self.assertEqual(cfg['wm_samples'], 2)
        self.assertEqual(cfg['sumo_verify_fraction'], 0.)


class Observables(unittest.TestCase):
    def test_truth_shaped_keys_refused(self):
        for key in ('wm_truth', 'global_state', 'oracle_speed', 'vtype_profile',
                    'driver_model'):
            with self.assertRaises(ValueError):
                gc.assert_observable({key: 1.})
        with self.assertRaises(ValueError):
            gc.assert_observable(dict(partners=[dict(hidden_v=1.)]))
        with self.assertRaises(ValueError):
            gc.assert_observable({'GLOBAL_STATE': {}})

    def test_clean_belief_passes_and_estimate_guards(self):
        gc.assert_observable(BELIEF)
        gc.wm_estimate(stub(good_move_values()), BELIEF, MEMBERS, MOVE,
                       np.random.default_rng(0))
        with self.assertRaises(ValueError):
            gc.wm_estimate(stub(good_move_values()), dict(truth=1.), MEMBERS,
                            MOVE, np.random.default_rng(0))

    def test_move_dataclass_truth_guard_before_scorer(self):
        @dataclass
        class BadMove:
            global_state: dict
        calls = []
        for move in (dict(hidden_profile={}), BadMove({})):
            with self.assertRaises(ValueError):
                gc.wm_estimate(stub(good_move_values(), calls), BELIEF, MEMBERS,
                               move, np.random.default_rng(0))
        self.assertEqual(calls, [])


class Estimate(unittest.TestCase):
    def test_decomposition_hand_check(self):
        e = gc.wm_estimate(stub(good_move_values()), BELIEF, MEMBERS, MOVE,
                           np.random.default_rng(1))
        self.assertEqual(e['outcome'], dict(base=50., joint=40., alone=[45., 46.],
                                          reverted=[46., 45.]))
        self.assertAlmostEqual(e['credit']['D_i'][0], 6.)
        self.assertAlmostEqual(e['credit']['D_i'][1], 5.)
        self.assertAlmostEqual(e['credit']['D_G'], 10.)
        self.assertAlmostEqual(e['credit']['C_G'], -1.)
        self.assertAlmostEqual(e['credit']['group_credit'], 10.)   # utility with - without
        self.assertAlmostEqual(e['credit']['delay_delta'], -10.)
        self.assertAlmostEqual(e['credit']['improvement'], 10.)
        self.assertEqual(e['members'], list(MEMBERS))
        self.assertIs(e['move'], MOVE)

    def test_same_decomposition_as_isolated_synergy_algebra(self):
        e = gc.wm_estimate(stub(good_move_values()), BELIEF, MEMBERS, MOVE,
                           np.random.default_rng(2))
        ref = cr.algebra(joint=e['outcome']['joint'], reverted=e['outcome']['reverted'],
                         all_reverted=e['outcome']['base'])
        self.assertEqual(ref['D_i'], e['credit']['D_i'])
        self.assertEqual(ref['D_G'], e['credit']['D_G'])
        self.assertEqual(ref['C_G'], e['credit']['C_G'])

    def test_multiple_wm_samples_mean_and_call_shape(self):
        values = {('base', None): [50., 52., 48., 50.],
                  ('joint', None): [40., 44., 36., 40.],
                  ('alone', 'cav_a'): [45., 47., 43., 45.],
                  ('alone', 'cav_b'): [46., 48., 44., 46.],
                  ('revert', 'cav_a'): [46., 48., 44., 46.],
                  ('revert', 'cav_b'): [45., 47., 43., 45.]}
        calls = []
        e = gc.wm_estimate(stub(values, calls), BELIEF, MEMBERS, MOVE,
                           np.random.default_rng(3))
        self.assertAlmostEqual(e['outcome']['base'], 50.)
        self.assertAlmostEqual(e['outcome']['joint'], 40.)
        self.assertEqual(e['outcome']['alone'][0], 45.)
        self.assertEqual(len(calls), 4 * (2 + 2 * len(MEMBERS)))

    def test_child_streams_deterministic_per_seed(self):
        base = {('base', None): 50., ('joint', None): 40.,
                 ('alone', 'cav_a'): 45., ('alone', 'cav_b'): 46.,
                 ('revert', 'cav_a'): 46., ('revert', 'cav_b'): 45.}
        a = gc.wm_estimate(jitter_stub(base), BELIEF, MEMBERS, MOVE,
                           np.random.default_rng(7))
        b = gc.wm_estimate(jitter_stub(base), BELIEF, MEMBERS, MOVE,
                           np.random.default_rng(7))
        c = gc.wm_estimate(jitter_stub(base), BELIEF, MEMBERS, MOVE,
                           np.random.default_rng(8))
        self.assertEqual(a, b)
        self.assertNotEqual(c, a)
        self.assertNotEqual(a['outcome']['base'], 50.)  # jitter actually applied
        self.assertAlmostEqual(b['outcome']['base'], a['outcome']['base'])
        self.assertEqual(a['credit']['D_G'], 10.)  # matched jitter cancels exactly
        self.assertEqual(a['credit']['D_i'], [6., 5.])

    def test_member_list_validation_and_nonfinite_refusal(self):
        for members in ([], ('x', 'x'), ('a', 'b', 'c', 'd', 'e'), ('a', 1), 'ab'):
            with self.assertRaises(ValueError):
                gc.wm_estimate(stub(good_move_values()), BELIEF, members, MOVE,
                               np.random.default_rng(4))
        bad = {k: ([float('nan')] * 4 if k == ('joint', None) else v)
               for k, v in good_move_values().items()}
        with self.assertRaises(ValueError):
            gc.wm_estimate(stub(bad), BELIEF, MEMBERS, MOVE,
                           np.random.default_rng(4))

    def test_sign_convention_good_move(self):
        e = gc.wm_estimate(stub(good_move_values()), BELIEF, MEMBERS, MOVE,
                           np.random.default_rng(5))
        self.assertGreater(e['credit']['D_G'], 0.)        # delay reduction positive
        self.assertGreater(e['credit']['group_credit'], 0.)  # utility (with - without)
        self.assertGreater(e['credit']['improvement'], 0.)

    def test_three_member_solos_are_not_member_reversions(self):
        members = ['a', 'b', 'c']
        active = []

        def scorer(belief, itv, rng):
            active.append((itv['kind'], itv.get('member'), itv['active_members']))
            self.assertEqual(itv['background'], 'base_mappo')
            self.assertEqual(itv['group_members'], members)
            return {0: 100., 1: 90., 2: 70., 3: 50.}[len(itv['active_members'])]

        e = gc.wm_estimate(scorer, BELIEF, members, dict(MOVE, members=members),
                           np.random.default_rng(0))
        self.assertEqual(e['credit']['D_i'], [20.] * 3)
        self.assertEqual(e['credit']['C_G'], -10.)
        self.assertEqual(e['isolated']['delta_only_member_new_veh_s'],
                         dict.fromkeys(members, 10.))
        self.assertEqual(e['isolated']['policy_cooperation_gain_veh_s'], 40.)
        self.assertIn(('alone', 'a', ['a']), active)
        self.assertIn(('revert', 'a', ['b', 'c']), active)
        from .policy_synergy import metrics
        ref = metrics(e['outcome']['base'],
                      [dict(status='graded', J_veh_s=x)
                       for x in [e['outcome']['joint'], *e['outcome']['alone']]], members)
        for k, v in e['isolated'].items():
            self.assertEqual(ref[k], v)

    def test_full_horizon_forwarded_and_inputs_isolated(self):
        belief, move = copy.deepcopy(BELIEF), copy.deepcopy(MOVE)
        before = copy.deepcopy((belief, move))

        def scorer(b, itv, rng):
            self.assertEqual(b, BELIEF)
            self.assertEqual(itv['horizon_s'], 600)
            self.assertEqual(itv['group_members'], list(MEMBERS))
            b['ego']['v'] = -100.
            itv['group_members'].clear()
            if 'move' in itv:
                self.assertEqual(itv['move'], MOVE)
                itv['move'].clear()
            return 50.

        e = gc.wm_estimate(scorer, belief, MEMBERS, move, np.random.default_rng(0),
                           horizon_s=600)
        self.assertEqual((belief, move), before)
        self.assertEqual(e['horizon_s'], 600)
        self.assertEqual(e['credit']['group_credit'], 0.)
        for h in (0, -1, True, 60.):
            with self.assertRaises(ValueError):
                gc.wm_estimate(scorer, belief, MEMBERS, move, np.random.default_rng(0),
                               horizon_s=h)

    def test_noop_single_member_and_harmful_moves(self):
        for delta in (0., -10., 10.):
            def scorer(b, itv, rng):
                return 50. + (delta if itv['active_members'] else 0.)
            e = gc.wm_estimate(scorer, BELIEF, ['cav_a'], dict(MOVE, members=['cav_a']),
                               np.random.default_rng(0))
            self.assertEqual(e['credit']['group_credit'], -delta)
            self.assertEqual(e['credit']['C_G'], 0.)
            self.assertEqual(e['isolated']['policy_cooperation_gain_veh_s'], 0.)
            self.assertEqual(gc.layer_advantage(e)['per_member'], [-delta])

    def test_move_members_must_match_group_before_scoring(self):
        calls = []
        for members in (['cav_b', 'cav_a'], ['cav_a']):
            with self.assertRaises(ValueError):
                gc.wm_estimate(stub(good_move_values(), calls), BELIEF, members,
                               MOVE, np.random.default_rng(0))
        self.assertEqual(calls, [])


class Advantage(unittest.TestCase):
    def estimate(self):
        return gc.wm_estimate(stub(good_move_values()), BELIEF, MEMBERS, MOVE,
                              np.random.default_rng(6))

    def test_per_member_hand_check(self):
        e = self.estimate()
        adv = gc.layer_advantage(e)
        self.assertEqual(adv['members'], list(MEMBERS))
        for i, m in enumerate(MEMBERS):
            self.assertAlmostEqual(adv['per_member'][i], e['credit']['D_G'])
        self.assertEqual(adv['team'], 10.)

    def test_diagnostic_local_credit_does_not_change_group_advantage(self):
        e = self.estimate()
        e['credit'].update(D_i=[-1000., 1000.], C_G=-9999.)
        self.assertEqual(gc.layer_advantage(e)['per_member'], [10., 10.])

    def test_pure_function_replaces_per_cav_reward_only(self):
        e = self.estimate()
        a1, a2 = gc.layer_advantage(e), gc.layer_advantage(e)
        self.assertEqual(a1, a2)
        self.assertEqual(len(a1['per_member']), len(e['members']))


class VerifyRequests(unittest.TestCase):
    def estimates(self):
        rng = np.random.default_rng(8)
        return [gc.wm_estimate(stub(good_move_values()), BELIEF, MEMBERS, MOVE,
                                rng, draw_id=i) for i in range(6)]

    def test_fraction_endpoints_and_fields(self):
        es = self.estimates()
        self.assertEqual(gc.verify_requests(es, np.random.default_rng(0),
                                            cfg=dict(sumo_verify_fraction=0.)), [])
        reqs = gc.verify_requests(es, np.random.default_rng(0),
                                  cfg=dict(sumo_verify_fraction=1.))
        self.assertEqual(len(reqs), 6)
        for r, e in zip(reqs, es):
            self.assertEqual(r['kind'], 'group_credit_verification')
            self.assertEqual(r['verify_mode'], 'all8_both_seeds')
            self.assertIs(r['isolated'], True)
            self.assertEqual(r['draw_id'], e['draw_id'])
            self.assertEqual(r['members'], e['members'])
            self.assertEqual(r['move'], e['move'])
            self.assertIsNot(r['move'], e['move'])
            self.assertEqual(r['wm'], e['credit'])
            self.assertEqual(r['sample_seeds'], e['sample_seeds'])
            self.assertEqual(r['background'], 'base_mappo')

    def test_registered_fraction_selection_deterministic_subset(self):
        es = self.estimates()
        a = gc.verify_requests(es, np.random.default_rng(11))
        b = gc.verify_requests(es, np.random.default_rng(11))
        self.assertEqual(a, b)
        self.assertLess(len(a), 6)
        ids = {r['draw_id'] for r in a}
        self.assertTrue(ids.issubset({e['draw_id'] for e in es}))

    def test_cpu_helpers_never_execute_verification(self):
        from unittest.mock import patch
        from . import sumo_verify, verify_perseed
        from .test_sumo_verify_cpu import plan, snapshot
        with patch.object(sumo_verify, 'verify', side_effect=AssertionError('execution')), \
                patch.object(verify_perseed, 'confirm_all8', side_effect=AssertionError('execution')):
            e = self.estimates()[0]
            gc.layer_advantage(e)
            req = gc.verify_requests([e], np.random.default_rng(0),
                                     cfg=dict(sumo_verify_fraction=1.))[0]
            gc.verification_candidates(req, plan('p', MEMBERS), snapshot())


class VerifiedStats(unittest.TestCase):
    def records(self):
        rng = np.random.default_rng(9)
        es = [gc.wm_estimate(stub(good_move_values()), BELIEF, MEMBERS, MOVE, rng,
                             draw_id=i) for i in range(3)]
        exact_dg = [9., -8., 0.]
        return [dict(wm=e['credit'], exact=dict(D_G=d))
                for e, d in zip(es, exact_dg)]

    def test_hand_check(self):
        s = gc.verified_stats(self.records())
        self.assertEqual(s['n'], 3)
        self.assertEqual(s['paired'], 2)          # exact 0 is a tie
        self.assertEqual(s['ties'], 1)
        self.assertEqual(s['sign_agreement'], .5)  # wm +/exact - disagree on pair 2
        self.assertAlmostEqual(s['mae_D_G'], (1. + 18. + 10.) / 3.)

    def test_empty_refused(self):
        with self.assertRaises(ValueError):
            gc.verified_stats([])

    def test_nonfinite_exact_records_refused(self):
        for value in (float('nan'), float('inf')):
            records = self.records()
            records[0]['exact']['D_G'] = value
            with self.assertRaises(ValueError):
                gc.verified_stats(records)


class VerificationAdapter(unittest.TestCase):
    def request(self, members):
        estimate = dict(draw_id='train#0', members=members, move=dict(MOVE, members=members),
                        credit=dict(D_G=10.), horizon_s=720, sample_seeds=[1, 2])
        return gc.verify_requests([estimate], np.random.default_rng(0),
                                  cfg=dict(sumo_verify_fraction=1.))[0]

    def test_real_verifier_specs_match_full_window_and_masks(self):
        from . import sumo_verify as sv
        from .test_sumo_verify_cpu import plan, snapshot, FakePlant, charge_log
        members = ['a', 'b', 'c']
        p = plan('move0', members, duration=600, gain=1.)
        p['profile'] = [[float(i)] * 600 for i in range(1, 4)]
        before = copy.deepcopy(p)
        bundle = gc.verification_candidates(self.request(members), p, snapshot())
        cands, idx = bundle['candidates'], bundle['branch_indices']
        self.assertEqual(p, before)
        self.assertEqual(len(cands), 7)  # joint + 3 solos + 3 reversions
        specs = sv.build_branches(snapshot(), cands)
        self.assertEqual(specs, bundle['specs'])
        self.assertEqual(len(specs), 8)
        self.assertTrue(all(s['window'] == specs[0]['window'] for s in specs))
        self.assertEqual(specs[0]['window']['end_s'], 1120)  # unchanged 600s + tail
        self.assertEqual(specs[idx['joint']]['plan']['members'], members)
        for i, m in enumerate(members):
            solo = specs[idx['alone'][i]]['plan']
            reverted = specs[idx['reverted'][i]]['plan']
            self.assertEqual(solo['members'], [m])
            self.assertEqual(solo['profile'], [p['profile'][i]])
            self.assertEqual(reverted['members'], [x for x in members if x != m])
            self.assertEqual(reverted['profile'], [r for j, r in enumerate(p['profile']) if j != i])
        log, charge = charge_log()
        receipt = sv.verify(snapshot(), cands, FakePlant(), charge)
        self.assertEqual(len(log), 1)
        self.assertEqual(len(log[0][1]), 8)
        self.assertEqual(receipt['J_policy_veh_s'], 500.)
        self.assertEqual(receipt['J_candidates_veh_s'][idx['joint'] - 1], 480.)

    def test_single_and_two_member_duplicate_masks_reuse_branches(self):
        from .test_sumo_verify_cpu import plan, snapshot
        for members, count in ((['a'], 1), (['a', 'b'], 3)):
            b = gc.verification_candidates(self.request(members),
                                           plan('p', members, duration=600), snapshot())
            self.assertEqual(len(b['candidates']), count)
            idx = b['branch_indices']
            if count == 1:
                self.assertEqual(idx['alone'], [idx['joint']])
                self.assertEqual(idx['reverted'], [0])
            else:
                self.assertEqual(idx['reverted'], idx['alone'][::-1])

    def test_invalid_request_or_plan_refused(self):
        from .test_sumo_verify_cpu import plan, snapshot
        req = self.request(['a', 'b'])
        for field, value in (('kind', 'other'), ('verify_mode', 'legacy'),
                             ('isolated', False), ('background', 'new_policy')):
            with self.assertRaises(ValueError):
                gc.verification_candidates(dict(req, **{field: value}),
                                           plan('p', ['a', 'b']), snapshot())
        with self.assertRaises(ValueError):
            gc.verification_candidates(req, plan('p', ['b', 'a']), snapshot())

    def test_short_horizon_and_background_cooperation_refused(self):
        from .test_sumo_verify_cpu import plan, snapshot
        req = self.request(['a', 'b'])
        for snap, p in ((snapshot(), plan('p', ['a', 'b'], duration=120)),
                        (snapshot(), plan('p', ['a', 'b'], duration=650)),
                        (snapshot(t0=1000), plan('p', ['a', 'b'], duration=600)),
                        (snapshot(committed=[dict(plan=dict(members=['c']))]),
                         plan('p', ['a', 'b'], duration=600))):
            with self.assertRaises(ValueError):
                gc.verification_candidates(req, p, snap)


class Chain(unittest.TestCase):
    def test_estimate_advantage_request_stats_chain(self):
        rng = np.random.default_rng(12)
        e = gc.wm_estimate(stub(good_move_values()), BELIEF, MEMBERS, MOVE, rng,
                           draw_id='u12#0')
        adv = gc.layer_advantage(e)
        reqs = gc.verify_requests([e], np.random.default_rng(13),
                                  cfg=dict(sumo_verify_fraction=1.))
        rec = dict(wm=reqs[0]['wm'], exact=dict(D_G=e['credit']['D_G'] + 1.))
        s = gc.verified_stats([rec])
        self.assertEqual(len(adv['per_member']), 2)
        self.assertEqual(s['paired'], 1)
        self.assertEqual(reqs[0]['members'], adv['members'])


if __name__ == '__main__':
    unittest.main()
