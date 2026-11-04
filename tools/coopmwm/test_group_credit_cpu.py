"""CPU tests for T2E group_credit: counterfactual GROUP credit for the cooperation
layer. No SUMO, no WM checkpoint, no GPU - injected stub scorers only."""
import unittest
import numpy as np
from . import credit as cr
from . import group_credit as gc

MEMBERS = ('cav_a', 'cav_b')
BELIEF = dict(ego=dict(v=20., lane=1), partners=[dict(id='cav_b', gap=12.)],
              lead_gap=25., merge_zone=dict(start=-100., end=75.))
MOVE = object()


def stub(values, calls=None):
    """values[(kind, member)] -> per-sample outcomes; internal per-key counter."""
    count = {}

    def scorer(belief, itv, rng):
        assert belief is BELIEF
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
            ('alone', 'cav_a'): [45.] * 4, ('alone', 'cav_b'): [46.] * 4}


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
                    dict(wm_samples=0), dict(wm_samples=2.0),
                    dict(verify_mode='per_seed'), dict(isolated=False),
                    dict(direction='higher')):
            with self.assertRaises(ValueError):
                gc.credit_config(bad)

    def test_valid_override(self):
        cfg = gc.credit_config(dict(wm_samples=2, c_g_share=0.))
        self.assertEqual(cfg['wm_samples'], 2)
        self.assertEqual(cfg['c_g_share'], 0.)


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


class Estimate(unittest.TestCase):
    def test_decomposition_hand_check(self):
        e = gc.wm_estimate(stub(good_move_values()), BELIEF, MEMBERS, MOVE,
                           np.random.default_rng(1))
        self.assertEqual(e['outcome'], dict(base=50., joint=40., alone=[45., 46.]))
        self.assertAlmostEqual(e['credit']['D_i'][0], 5.)
        self.assertAlmostEqual(e['credit']['D_i'][1], 6.)
        self.assertAlmostEqual(e['credit']['D_G'], 10.)
        self.assertAlmostEqual(e['credit']['C_G'], -1.)
        self.assertAlmostEqual(e['credit']['group_credit'], -10.)   # with - without
        self.assertAlmostEqual(e['credit']['improvement'], 10.)
        self.assertEqual(e['members'], list(MEMBERS))
        self.assertIs(e['move'], MOVE)

    def test_same_decomposition_as_isolated_synergy_algebra(self):
        e = gc.wm_estimate(stub(good_move_values()), BELIEF, MEMBERS, MOVE,
                           np.random.default_rng(2))
        ref = cr.algebra(joint=e['outcome']['joint'], reverted=e['outcome']['alone'],
                         all_reverted=e['outcome']['base'])
        self.assertEqual(ref['D_i'], e['credit']['D_i'])
        self.assertEqual(ref['D_G'], e['credit']['D_G'])
        self.assertEqual(ref['C_G'], e['credit']['C_G'])

    def test_multiple_wm_samples_mean_and_call_shape(self):
        values = {('base', None): [50., 52., 48., 50.],
                  ('joint', None): [40., 44., 36., 40.],
                  ('alone', 'cav_a'): [45., 47., 43., 45.],
                  ('alone', 'cav_b'): [46., 48., 44., 46.]}
        calls = []
        e = gc.wm_estimate(stub(values, calls), BELIEF, MEMBERS, MOVE,
                           np.random.default_rng(3))
        self.assertAlmostEqual(e['outcome']['base'], 50.)
        self.assertAlmostEqual(e['outcome']['joint'], 40.)
        self.assertEqual(e['outcome']['alone'][0], 45.)
        self.assertEqual(len(calls), 4 * (2 + len(MEMBERS)))

    def test_child_streams_deterministic_per_seed(self):
        base = {('base', None): 50., ('joint', None): 40.,
                ('alone', 'cav_a'): 45., ('alone', 'cav_b'): 46.}
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

    def test_member_list_validation_and_nonfinite_refusal(self):
        with self.assertRaises(ValueError):
            gc.wm_estimate(stub(good_move_values()), BELIEF, [], MOVE,
                           np.random.default_rng(4))
        with self.assertRaises(ValueError):
            gc.wm_estimate(stub(good_move_values()), BELIEF, ('x', 'x'), MOVE,
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
        self.assertLess(e['credit']['group_credit'], 0.)  # literal (with - without)
        self.assertGreater(e['credit']['improvement'], 0.)


class Advantage(unittest.TestCase):
    def estimate(self):
        return gc.wm_estimate(stub(good_move_values()), BELIEF, MEMBERS, MOVE,
                              np.random.default_rng(6))

    def test_per_member_hand_check(self):
        e = self.estimate()
        adv = gc.layer_advantage(e)
        share = .5 * e['credit']['C_G'] / 2
        self.assertEqual(adv['members'], list(MEMBERS))
        for i, m in enumerate(MEMBERS):
            self.assertAlmostEqual(adv['per_member'][i], e['credit']['D_i'][i] + share)
        self.assertAlmostEqual(adv['team'],
                               np.mean(e['credit']['D_i']) + .5 * e['credit']['C_G'])

    def test_c_g_share_zero_gives_pure_alone_credit(self):
        adv = gc.layer_advantage(self.estimate(), cfg=dict(c_g_share=0.))
        self.assertEqual(adv['per_member'], [5., 6.])

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
            self.assertIs(r['move'], e['move'])
            self.assertEqual(r['wm'], e['credit'])

    def test_registered_fraction_selection_deterministic_subset(self):
        es = self.estimates()
        a = gc.verify_requests(es, np.random.default_rng(11))
        b = gc.verify_requests(es, np.random.default_rng(11))
        self.assertEqual(a, b)
        self.assertLess(len(a), 6)
        ids = {r['draw_id'] for r in a}
        self.assertTrue(ids.issubset({e['draw_id'] for e in es}))

    def test_no_sumo_in_module(self):
        import inspect
        src = inspect.getsource(gc)
        self.assertNotIn('subprocess', src)
        self.assertNotIn('.remote(', src)
        self.assertNotIn('import sumo', src)


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
