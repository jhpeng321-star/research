"""CPU tests for the c4/lane-plans feature: (acceleration, lane keep/left/right) plans.

No SUMO, no GPU. Tests exercise the REAL code paths: wms.control.lanes.lane_feasibility
(the SUMO-calibrated lane-change safety model), the lane_v5 lane model, the search, the
verification Controller/audit path, and the bank imitation lane targets.
"""
import copy
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from tools.coopmwm import lane_plans as lp
from tools.coopmwm import mppi_search as ms
from tools.coopmwm.test_mppi_search_cpu import BOUNDS, T, Fake, group_of, run


def rows():
    """Two mainline CAVs (lane 1 and lane 0, 40 m apart) + HDV traffic, exact sensor whitelist.

    s_m in the WM contract frame (s_down, merge zone [0, 75.44]; negative = approaching)."""
    out = [dict(vid='c0', s_m=-40., v_mps=20., lane=1, is_cav=True, length_m=5.),
           dict(vid='c1', s_m=-80., v_mps=19., lane=0, is_cav=True, length_m=5.)]
    out += [dict(vid=f'h{i}', s_m=-120.-25*i, v_mps=18., lane=i % 3, is_cav=False, length_m=5.)
            for i in range(2)]
    return out


def record():
    return dict(rows=rows(), state_id='s', policy_residuals={'c0': 0., 'c1': 0.})


def ramp_record(s=30.):
    """A ramp-lane (lane 2) CAV inside the merge zone; a lane-1 CAV far enough behind."""
    return dict(rows=[dict(vid='m0', s_m=s, v_mps=15., lane=2, is_cav=True, length_m=5.),
                      dict(vid='m1', s_m=s-35., v_mps=16., lane=1, is_cav=True, length_m=5.)],
                state_id='s', policy_residuals={'m0': 0., 'm1': 0.})


class Envelope(unittest.TestCase):
    def test_only_the_physical_transition_is_allowed(self):
        env = lp.decision_envelope(record(), ['c0', 'c1'])
        self.assertIsNotNone(env)
        self.assertEqual(env['c0'], dict(lane=1, left=True, right=False))   # overtake 1->0 only
        self.assertEqual(env['c1'], dict(lane=0, left=False, right=True))   # return 0->1 only

    def test_ramp_merge_inside_zone_only(self):
        env = lp.decision_envelope(ramp_record(), ['m0'])
        self.assertIsNotNone(env)
        self.assertTrue(env['m0']['left'])           # ramp 2->1 merge inside [0, 75.44]
        self.assertFalse(env['m0']['right'])         # 1->2 is never allowed
        env_far = lp.decision_envelope(ramp_record(s=-400.), ['m0'])   # outside the merge zone
        self.assertFalse(env_far['m0']['left'])

    def test_gap_rule_blocks_when_the_target_lane_is_blocked(self):
        r = record()
        r['rows'].append(dict(vid='h9', s_m=-42., v_mps=20., lane=0, is_cav=False,
                              length_m=5., a_mps2=0.))              # alongside c0's target lane
        env = lp.decision_envelope(r, ['c0'])
        self.assertFalse(env['c0']['left'])          # the SUMO-calibrated gap rule (no invented rule)

    def test_unresolvable_state_degrades_to_none(self):
        r = dict(rows=[dict(vid='c0', s_m=-40., v_mps=20., lane=1)])   # no is_cav: unusable
        self.assertIsNone(lp.decision_envelope(r, ['c0']))
        self.assertIsNone(lp.decision_envelope(dict(rows=[dict(vid='c9', s_m=0., v_mps=1., lane=0,
                                                              is_cav=True)]), ['c0']))

    def test_lane_model_predicts_outcomes(self):
        model = lp.lane_model()
        self.assertIsNotNone(model)
        preds = lp.predict(record(), ['c0', 'c1'], [[lp.LANE_LEFT], [0]])
        self.assertEqual(len(preds), 1)
        p = preds[0]
        self.assertEqual(p['member'], 'c0')
        self.assertEqual(p['head'], 'overtake')      # lane 1 -> 0 is the overtake head
        self.assertTrue(0. <= p['model_prob'] <= 1.)
        self.assertAlmostEqual(p['model_prob'], float(1./(1.+np.exp(-p['model_logit']))), places=12)


class Sampling(unittest.TestCase):
    def test_rate_one_always_draws_inside_the_envelope(self):
        env = lp.decision_envelope(record(), ['c0', 'c1'])
        for seed in range(20):
            acts = lp.sample_lane_actions(env, ['c0', 'c1'], 60, np.random.default_rng(seed), rate=1.)
            for i, v in enumerate(['c0', 'c1']):
                self.assertEqual(sum(a != 0 for a in acts[i]), 1)      # one attempt per member
                t = acts[i].index(next(a for a in acts[i] if a))
                self.assertTrue(0 <= t < 60)
                self.assertEqual(acts[i][t], lp.LANE_LEFT if v == 'c0' else lp.LANE_RIGHT)

    def test_rate_zero_and_no_envelope_are_inert(self):
        env = lp.decision_envelope(record(), ['c0', 'c1'])
        self.assertEqual(lp.sample_lane_actions(env, ['c0', 'c1'], 60, np.random.default_rng(0), rate=0.),
                         [[0]*60, [0]*60])
        self.assertEqual(lp.sample_lane_actions(None, ['c0', 'c1'], 60, np.random.default_rng(0), rate=1.),
                         [[0]*60, [0]*60])
        none = {v: dict(e, left=False, right=False) for v, e in env.items()}
        self.assertEqual(lp.sample_lane_actions(none, ['c0', 'c1'], 60, np.random.default_rng(0), rate=1.),
                         [[0]*60, [0]*60])

    def test_attach_digests_and_validate(self):
        g = group_of(2)
        plan = ms.build_plan(g, np.zeros((2, T)), np.zeros((2, 10)), 60, BOUNDS, 'sampled', 'ordinary')
        env = lp.decision_envelope(record(), g['members'])
        acts = lp.sample_lane_actions(env, g['members'], 60, np.random.default_rng(3), rate=1.)
        old_id = plan['id']
        lp.attach(plan, acts, env, lp.predict(record(), g['members'], acts))
        self.assertNotEqual(plan['id'], old_id)                 # the id covers the lane fields
        self.assertEqual(len(plan['id']), len(old_id))
        self.assertTrue(plan['uses_lane_actions'])
        lp.validate(plan)
        json.dumps(plan)
        with self.assertRaises(ValueError):
            lp.validate(dict(plan, lane_actions=[[3]*60, [0]*60]))
        with self.assertRaises(ValueError):
            lp.validate(dict(plan, lane_actions=[[1]*30+[0]*30, [0]*60]))   # two attempts
        with self.assertRaises(ValueError):
            bad = copy.deepcopy(plan)
            bad['lane_envelope']['c0']['left'] = False
            lp.validate(bad)                                    # outside the recorded envelope
        with self.assertRaises(ValueError):
            lp.validate(dict(plan, uses_lane_actions=False))
        ok = dict(plan, lane_actions=None, uses_lane_actions=False)
        self.assertIs(lp.validate(ok), ok)

    def test_at_accessor(self):
        plan = dict(members=['a', 'b'], duration_s=4, lane_actions=[[0, 1, 0, 0], [0, 0, 2, 0]])
        self.assertEqual(lp.at(plan, 0), {})
        self.assertEqual(lp.at(plan, 1), {'a': 1})
        self.assertEqual(lp.at(plan, 2), {'b': 2})
        self.assertEqual(lp.at(plan, 4), {})
        self.assertEqual(lp.at(dict(plan, lane_actions=[[0]*4, [0]*4]), 1), {})


class OptionSplit(unittest.TestCase):
    def test_off_removes_keys_and_rejects_unknown(self):
        cfg = dict(ms.REGISTERED)
        self.assertIsNone(lp.split_config(cfg))
        self.assertNotIn('lane_actions', cfg)
        with self.assertRaises(ValueError):
            lp.split_config(dict(lane_nonsense=1))
        with self.assertRaises(ValueError):
            lp.split_config(dict(lane_actions=True, lane_rate=2.))
        with self.assertRaises(ValueError):
            lp.split_config(dict(lane_actions=True, lane_model=3))

    def test_search_off_has_no_lane_trace(self):
        res, _ = run(seed=9)
        self.assertNotIn('lane_actions', res)
        out = json.dumps(res)
        for key in ('lane_actions', 'lane_envelope', 'lane_predictions', 'uses_lane_actions'):
            self.assertNotIn(key, out)
        p = ms.sample_unscored(group_of(2), [[0.]*T]*2, BOUNDS, 'ordinary', 3)
        self.assertNotIn('lane_actions', p)

    def test_config_validated_through_niche_collect(self):
        from tools.coopmwm.niche_collect import validate, search_config
        self.assertEqual(validate(dict(search_config(0), lane_actions=True))['lane_actions'], True)
        with self.assertRaises(ValueError):
            validate(dict(search_config(0), lane_nope=1))


class SearchOn(unittest.TestCase):
    def test_lane_plans_are_scored_tagged_and_within_the_envelope(self):
        fake = Fake(2)
        g = group_of(2)
        res = ms.search(record(), g, fake.policy, BOUNDS, fake, 'ordinary', seed=11,
                        config=dict(lane_actions=True))
        self.assertEqual(res['lane_actions'], True)
        self.assertEqual(res['config']['lane_actions'], True)
        self.assertTrue(res['accepted'])
        self.assertIn(res['plan']['uses_lane_actions'], (True, False))
        lane_plans = [p for p in fake.seen if p.get('uses_lane_actions')]
        self.assertTrue(lane_plans)                      # the search really sampled lane plans
        env = lp.decision_envelope(record(), g['members'])
        for p in lane_plans:
            ms.validate_plan(p, BOUNDS, ms.allowed_durations())
            for i, v in enumerate(p['members']):
                for t, a in enumerate(p['lane_actions'][i]):
                    if a == lp.LANE_LEFT:
                        self.assertTrue(env[v]['left'], (v, t))
                    if a == lp.LANE_RIGHT:
                        self.assertTrue(env[v]['right'], (v, t))
            self.assertEqual(len(p['lane_predictions']),
                             sum(a != 0 for row in p['lane_actions'] for a in row))
        for f in res['finalists']:
            self.assertIn('uses_lane_actions', f)
        json.dumps(res)

    def test_unresolvable_envelope_keeps_the_search_inert(self):
        fake = Fake(2)
        g = group_of(2)
        res = ms.search(dict(state_id='s', policy_residuals={}), g, fake.policy, BOUNDS, fake,
                        'ordinary', seed=11, config=dict(lane_actions=True))
        self.assertTrue(res['accepted'])
        self.assertFalse(any(p.get('uses_lane_actions') for p in fake.seen))
        self.assertNotIn('lane_actions', res['plan'])    # degraded, never guessed

    def test_determinism_with_lane_actions(self):
        def one():
            fake = Fake(2)
            return ms.search(record(), group_of(2), fake.policy, BOUNDS, fake, 'ordinary', seed=21,
                             config=dict(lane_actions=True))
        self.assertEqual(json.dumps(one(), sort_keys=True), json.dumps(one(), sort_keys=True))

    def test_niche_search_samples_lane_actions(self):
        fake = Fake(2)
        g = group_of(2)
        res = ms.search(record(), g, fake.policy, BOUNDS, fake, 'ordinary', seed=11,
                        config=dict(niches=2, lane_actions=True))
        self.assertEqual(res['lane_actions'], True)
        lane_plans = [p for p in fake.seen if p.get('uses_lane_actions')]
        self.assertTrue(lane_plans)
        env = lp.decision_envelope(record(), g['members'])
        for p in lane_plans:
            for i, v in enumerate(p['members']):
                for a in p['lane_actions'][i]:
                    if a == lp.LANE_LEFT:
                        self.assertTrue(env[v]['left'])
                    if a == lp.LANE_RIGHT:
                        self.assertTrue(env[v]['right'])
        json.dumps(res)

    def test_library_seed_keeps_its_lane_actions(self):
        fake = Fake(2);g = group_of(2);desc = ms.describe(g, 'ordinary')
        env = lp.decision_envelope(record(), g['members'])
        base = ms.build_plan(g, fake.policy, np.zeros((2, 10)), 60, BOUNDS, 'centre', 'ordinary')
        acts = [[1]+[0]*59, [0]*60]                  # one envelope-feasible attempt for c0
        lp.attach(base, acts, env, [])
        lib = ms.PlanLibrary()
        lib.add(desc, base, 40.)
        res = ms.search(record(), g, fake.policy, BOUNDS, fake, 'ordinary', seed=1, library=lib,
                        config=dict(library_seeds=1, lane_actions=True))
        seeded = [p for p in fake.seen if p['origin'] == 'library']
        self.assertEqual(len(seeded), 1)
        self.assertEqual(seeded[0]['lane_actions'], acts)     # the SUMO-positive lane plan reseeds
        json.dumps(res)

    def test_solo_plans_keep_the_members_lane_row(self):
        from tools.coopmwm.coop_search import solo_plan
        from tools.coopmwm.replay_bank import solo_candidate
        plan = dict(id='Mp', members=['a', 'b'], duration_s=3, profile=[[0.]*3, [0.]*3],
                    knot_times_s=[0., 3.], origin='sampled', regime='ordinary', family='MPPI',
                    lane_actions=[[0, 1, 0], [0, 0, 2]],
                    lane_envelope={'a': dict(lane=1, left=True, right=False),
                                   'b': dict(lane=0, left=False, right=True)},
                    lane_predictions=[], uses_lane_actions=True)
        for solo_of in (solo_plan, solo_candidate):
            s = solo_of(copy.deepcopy(plan), 'b')
            self.assertEqual(s['lane_actions'], [[0, 0, 2]])
            self.assertEqual(list(s['lane_envelope']), ['b'])
            self.assertTrue(s['uses_lane_actions'])
        s = solo_plan(dict(plan, lane_actions=None, uses_lane_actions=False), 'a')
        self.assertNotIn('lane_actions', s)

    def test_plan_distance_ignores_the_lane_channel(self):
        g = group_of(2)
        a = ms.build_plan(g, np.zeros((2, T)), np.zeros((2, 10)), 60, BOUNDS, 'sampled', 'ordinary')
        b = copy.deepcopy(a)
        lp.attach(b, [[1]+[0]*59, [0]*60], dict(a={'lane': 1, 'left': True, 'right': False}), [])
        self.assertEqual(ms.plan_distance(a, b), 0.)     # distance is an acceleration concept


class VerifyExecution(unittest.TestCase):
    def _plan(self, members):
        plan = ms.build_plan(dict(id='G', members=list(members)), np.zeros((len(members), T)),
                             np.zeros((len(members), 10)), 60, BOUNDS, 'sampled', 'ordinary')
        env = lp.decision_envelope(record(), list(members))
        acts = [[lp.LANE_LEFT]+[0]*59, [lp.LANE_RIGHT]+[0]*59]   # each member's allowed direction
        return lp.attach(plan, acts[:len(members)], env, [])

    def test_controller_emits_requests_and_records_outcomes(self):
        from tools.coopmwm.acceleration import Controller, Policy
        data = rows()
        members = ['c0', 'c1']
        plan = self._plan(members)
        seen = {}

        def executor(prows, requests, now):
            seen.update(requests=dict(requests), now=now)
            return {v: dict(issued=True, blocked=None, executed=(v == members[0]))
                    for v in requests}

        c = Controller(Policy(), torch.Generator().manual_seed(1), guide=None, beta=0.,
                       lane_executor=executor)
        c.active_plans = [dict(plan=copy.deepcopy(plan), start_s=300)]
        with patch('tools.coopmwm.exploration.regime', lambda rows: 'ordinary'):
            c.apply(data, {v: 0. for v in members}, 300)
        sample = c.events[-1]['sample']
        self.assertEqual(sample['lane_requests'], {members[0]: lp.LANE_LEFT, members[1]: lp.LANE_RIGHT})
        self.assertEqual(seen['now'], 300)
        events = sample['lane_events']
        self.assertEqual(events[members[0]], dict(requested=1, issued=True, blocked=None, executed=True))
        self.assertEqual(events[members[1]]['executed'], False)
        with patch('tools.coopmwm.exploration.regime', lambda rows: 'ordinary'):
            c.apply(data, {v: 0. for v in members}, 305)
        self.assertNotIn('lane_requests', c.events[-1]['sample'])    # attempt second passed

    def test_unbound_executor_records_unbound_never_executed(self):
        from tools.coopmwm.acceleration import Controller, Policy
        data = rows()
        members = ['c0', 'c1']
        c = Controller(Policy(), torch.Generator().manual_seed(1), guide=None, beta=0.)
        c.active_plans = [dict(plan=copy.deepcopy(self._plan(members)), start_s=300)]
        with patch('tools.coopmwm.exploration.regime', lambda rows: 'ordinary'):
            c.apply(data, {v: 0. for v in members}, 300)
        for e in c.events[-1]['sample']['lane_events'].values():
            self.assertEqual(e['blocked'], 'unbound')
            self.assertFalse(e['executed'])
            self.assertFalse(e['issued'])

    def test_record_events_and_counts(self):
        rec = lp.record_events({'a': 1, 'b': 2},
                               {'a': dict(issued=True, executed=True), 'b': dict(issued=True, blocked='gap')})
        self.assertEqual(rec['a'], dict(requested=1, issued=True, blocked=None, executed=True))
        self.assertEqual(rec['b']['blocked'], 'gap')
        self.assertFalse(rec['b']['executed'])
        self.assertEqual(lp.outcome_counts(rec), dict(requests=2, issued=2, blocked=1, executed=1))
        unbound = lp.record_events({'a': 1}, None)
        self.assertEqual(unbound['a']['blocked'], 'unbound')
        self.assertEqual(lp.outcome_counts(unbound)['executed'], 0)

    def test_branch_audit_counts_lane_outcomes(self):
        """The real VerifyController records issued/blocked/executed in the branch audit."""
        import hashlib
        from tools.coopmwm.acceleration import Policy
        from tools.coopmwm.sumo_verify import verify_controller_class
        from tools.coopmwm.test_sumo_verify_cpu import snapshot
        members = ['c0', 'c1']
        plan = self._plan(members)
        snap = snapshot(t0=400)
        gen = torch.Generator().manual_seed(9)
        snap['generator_state'] = gen.get_state().clone()
        snap['rng']['generator_state_sha256'] = hashlib.sha256(
            snap['generator_state'].numpy().tobytes()).hexdigest()
        snap['rng']['memory_after_t0'] = {}
        snap['samples'] = {str(t): dict(action=[[0.], [0.]], context=dict(members=members),
                                        behaviour={}, memory={}) for t in range(300, 401)}
        data = rows()
        controller = verify_controller_class()(Policy(), torch.Generator().manual_seed(9), snap, plan)
        controller.lane_executor = lambda prows, requests, now: {v: dict(issued=True, blocked=None,
                                                                         executed=True) for v in requests}
        with patch('tools.coopmwm.exploration.regime', lambda rows: 'ordinary'):
            for t in (400, 401, 402):
                controller.apply(data, {v: 0. for v in members}, t)
        audit = controller.audit
        self.assertIn('lane_actions', audit)
        self.assertEqual(audit['lane_actions'], dict(requests=2, issued=2, blocked=0, executed=2))
        self.assertEqual(audit['live_steps'], 3)

    def test_build_branches_validates_lane_fields(self):
        from tools.coopmwm.sumo_verify import build_branches
        from tools.coopmwm.test_sumo_verify_cpu import snapshot
        plan = self._plan(['c0', 'c1'])
        with self.assertRaises(ValueError):
            build_branches(snapshot(), [dict(plan, lane_actions=[[9]*60, [0]*60])])
        build_branches(snapshot(), [plan])                      # well-formed lane plans pass


class RSymmetry(unittest.TestCase):
    def test_random_candidates_same_lane_rule(self):
        from tools.coopmwm.sumo_verify import random_candidates
        g = dict(members=['c0', 'c1'])
        rec = record()
        lane = dict(lp.REGISTERED, lane_actions=True)
        cands = random_candidates(g, [[0.]*300]*2, BOUNDS, 'ordinary', 123, k=4, lane_config=lane, record=rec)
        self.assertEqual(len(cands), 4)
        env = lp.decision_envelope(rec, g['members'])
        tagged = 0
        for p in cands:
            self.assertIn('lane_actions', p)
            for i, v in enumerate(p['members']):
                for a in p['lane_actions'][i]:
                    if a == lp.LANE_LEFT:
                        self.assertTrue(env[v]['left'])
                    if a == lp.LANE_RIGHT:
                        self.assertTrue(env[v]['right'])
            tagged += bool(p['uses_lane_actions'])
        self.assertGreater(tagged, 0)                    # rate > 0 draws really carry lane actions
        again = random_candidates(g, [[0.]*300]*2, BOUNDS, 'ordinary', 123, k=4, lane_config=lane, record=rec)
        self.assertEqual([p['id'] for p in cands], [p['id'] for p in again])   # seeds from base only
        for p in random_candidates(g, [[0.]*300]*2, BOUNDS, 'ordinary', 123, k=2):
            self.assertNotIn('lane_actions', p)          # option off: unchanged draws

    def test_sample_unscored_lane_without_record_is_inert(self):
        p = ms.sample_unscored(dict(members=['c0', 'c1']), [[0.]*300]*2, BOUNDS, 'ordinary', 3,
                               config=dict(lane_actions=True))
        self.assertNotIn('lane_actions', p)              # envelope unresolvable: never guessed


class LaneHeadPolicy(torch.nn.Module):
    """The coopmwm policy plus the c4/lane-policy lane-head contract (lane_logits)."""

    def __init__(self):
        super().__init__()
        from tools.coopmwm.acceleration import Policy
        self.policy = Policy()
        self.lane_lin = torch.nn.Linear(4, 3)

    def forward(self, context):
        return self.policy(context)

    def __getattr__(self, name):     # delegate the policy attributes (log_std, score, ...)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.policy, name)

    def lane_logits(self, context):
        return self.lane_lin(torch.ones(len(context['members']), 4))


class Imitation(unittest.TestCase):
    def _steps(self, with_events=True):
        steps = []
        for t, events in ((300, None), (301, {'c0': dict(requested=1, issued=True, blocked=None,
                                                          executed=True)}), (302, None)):
            if not with_events:
                events = None
            steps.append(dict(time_s=t, sample=dict(
                action=[[0.], [0.]], context=dict(members=['c0', 'c1']),
                behaviour=dict(executed=True, proposal_members=['c0', 'c1']),
                plan_executions=[dict(plan_id='lane1', proposal_members=['c0', 'c1'])],
                lane_events=events)))
        return steps

    def _pairs(self):
        from tools.coopmwm.exact_grade import imitation_pairs
        entry = dict(start_s=300, duration_s=3, plan_id='lane1', weight=2.5,
                     lane_actions=[[0, 1, 0], [0, 0, 2]])
        return imitation_pairs(dict(steps=self._steps()), entry)

    def test_pairs_carry_lane_targets_and_outcomes(self):
        pairs = self._pairs()
        self.assertEqual([s['lane_target'] for s, _ in pairs],
                         [{'c0': 0, 'c1': 0}, {'c0': 1, 'c1': 0}, {'c0': 0, 'c1': 2}])
        self.assertNotIn('lane_target_outcome', pairs[0][0])
        self.assertNotIn('lane_target_outcome', pairs[2][0])
        outcome = pairs[1][0]['lane_target_outcome']
        self.assertEqual(outcome['c0']['executed'], True)   # executed lane actions recorded
        entry = dict(start_s=300, duration_s=3, plan_id='lane1', weight=2.5)   # no lane fields
        from tools.coopmwm.exact_grade import imitation_pairs
        for s, _ in imitation_pairs(dict(steps=self._steps()), entry):
            self.assertNotIn('lane_target', s)

    def test_intent_lane_targets_include_lane(self):
        from tools.coopmwm.v2v_intent import attach_bank_targets
        entry = dict(start_s=300, duration_s=3, plan_id='lane1', weight=1.,
                     lane_actions=[[0, 1, 0], [0, 0, 2]])
        steps = self._steps(with_events=False)
        pairs = attach_bank_targets([(copy.deepcopy(steps[0]['sample']), 1.)], dict(steps=steps),
                                    entry, 4, (-3., 2.))
        sample = pairs[0][0]
        self.assertEqual(sample['intent_lane_target'], [[1, 0, 0, 0], [0, 2, 0, 0]])
        self.assertEqual(sample['intent_lane_mask'], [[1, 1, 0, 0], [1, 1, 0, 0]])
        self.assertEqual(sample['intent_target'][0][:3], [0., 0., 0.])   # accel targets unchanged

    def test_lane_ce_weighted_like_awr_and_inert_without_head_or_targets(self):
        def pair(targets, weight=2.5):
            return (dict(action=[[0.], [0.]], context=dict(members=['c0', 'c1']),
                         behaviour=dict(executed=True, proposal_members=['c0', 'c1']),
                         lane_target=targets), weight)
        head = LaneHeadPolicy()
        with torch.no_grad():
            logits = head.lane_logits(pair({}, 0)[0]['context'])
        ce = lambda j, a: float(torch.nn.functional.cross_entropy(logits[j:j+1], torch.tensor([a])))
        attempt = pair({'c0': 1, 'c1': 2})               # the attempt second
        keep = pair({'c0': 0, 'c1': 0})                  # keep seconds are valid lane targets too
        l1 = 2.5*(ce(0, 1)+ce(1, 2))/2                   # weight x mean member CE
        l0 = 2.5*(ce(0, 0)+ce(1, 0))/2
        self.assertAlmostEqual(float(lp.policy_lane_ce(head, [attempt]).detach()), l1, places=5)
        self.assertAlmostEqual(float(lp.policy_lane_ce(head, [attempt, keep]).detach()),
                               (l1+l0)/2, places=5)
        self.assertIsNone(lp.policy_lane_ce(object(), [attempt]))                     # no lane head
        self.assertIsNone(lp.policy_lane_ce(head, [(dict(attempt[0], lane_target=None), 2.5)]))
        loss = lp.policy_lane_ce(head, [attempt])
        loss.backward()                                  # gradients reach the lane head
        self.assertIsNotNone(head.lane_lin.weight.grad)

    def test_bank_imitation_lane_term_and_report(self):
        from tools.coopmwm import bank_imitation as bi
        from tools.coopmwm.acceleration import Policy
        from tools.coopnative.policy import encode
        context = encode(rows())
        def bank_pairs():
            targets = ({'c0': 0, 'c1': 0}, {'c0': 1, 'c1': 0}, {'c0': 0, 'c1': 2})
            return [(dict(action=[[0.], [0.]], context=dict(context),
                          behaviour=dict(executed=True, proposal_members=list(context['members'])),
                          lane_target=t), 1.) for t in targets]
        head = LaneHeadPolicy()
        report = bi.run_separate(head, torch.optim.Adam(head.parameters(), lr=1e-4), bank_pairs(),
                                 torch.Generator().manual_seed(0), copy.deepcopy(head), passes=5)
        self.assertGreaterEqual(report['epochs_completed'], 1)
        self.assertEqual(report['lane_targets'], dict(pairs=3, head=True))
        plain = Policy()
        report = bi.run_separate(plain, torch.optim.Adam(plain.parameters(), lr=1e-4), bank_pairs(),
                                 torch.Generator().manual_seed(0), copy.deepcopy(plain), passes=5)
        self.assertGreaterEqual(report['epochs_completed'], 1)
        self.assertEqual(report['lane_targets'], dict(pairs=3, head=False))   # recorded, term inert
        no_lane = [(dict(s, lane_target=None), w) for s, w in bank_pairs()]
        report = bi.run_separate(plain, torch.optim.Adam(plain.parameters(), lr=1e-4), no_lane,
                                 torch.Generator().manual_seed(0), copy.deepcopy(plain), passes=5)
        self.assertNotIn('lane_targets', report)         # legacy bank (no lane targets): byte-identical
        self.assertEqual(report['rule'], 'full-bank transactional passes; separate imitation KL; '
                                         'AWR weights unchanged')


if __name__ == '__main__':
    unittest.main()
