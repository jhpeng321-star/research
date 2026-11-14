"""CPU tests for the NULL_DIAGNOSIS sections 8.1/8.2/8.3 fix (no SUMO, no GPU).

8.1 collector-side actionable gate (sumo_verify.verified_guide.call): provably
ceiling-infeasible candidates are dropped and recorded; a group with no
actionable candidate skips verification entirely (accepted=False, no snapshot,
no charge, the CRN pad draw stays consumed exactly once). 8.2 search-side
feasibility clamp (mppi_search): candidate profiles are clipped to the
damping.direct_target ceiling/floor from the centre rollout, and candidates the
clip turns into the policy centre are excluded from the top-k. 8.3 per-member
t0 label fields on verification receipts. Feasible groups are byte-identical to
the pre-fix behaviour, proven against the committed fixtures.
"""
import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from . import feasible as fe
from . import mppi_search as ms
from . import sumo_verify as sv
from .exact_grade import verify_sealed
from .test_mppi_search_cpu import BOUNDS, Fake, T, group_of, row
from .test_sumo_verify_cpu import FakeController, FakePlant, base_delay, fp, snapshot

GOLDEN = Path(__file__).resolve().parent/'fixtures'/'niche1_golden.json'


def row_allowed(vid, s, lane, v, allowed):
    """Observable row carrying the CAV's subscribed allowed speed."""
    return dict(vid=vid, s_m=float(s), v_mps=float(v), lane=int(lane), is_cav=True,
                length_m=5., allowed_speed_mps=float(allowed))


def run_rows(seed=1, rows=None, m=2, **cfg):
    """search() on the committed Fake fixture, optionally with t0 state rows."""
    fake = Fake(m); g = group_of(m)
    record = dict(state_id='s', policy_residuals={v: 0. for v in g['members']})
    if rows is not None:
        record['rows'] = rows
    return ms.search(record, g, fake.policy, BOUNDS, fake, 'ordinary',
                     mappo_traj=fake.mappo, seed=seed, config=cfg), fake


def bytes_of(res):
    return json.dumps(res, sort_keys=True)


def gate_plan(pid, members, delta, bounds=(-3., 2.6), accel=None):
    """A real build_plan candidate (so the centre profile is reconstructable)."""
    g = dict(id='Gtest', members=list(members), rows=[dict(vid=v) for v in members])
    z = np.full((len(members), 10), float(delta))
    traj = np.full((len(members), 300), bounds[1] if accel is None else float(accel))
    return ms.build_plan(g, traj, z, 60, bounds, 'sampled', 'ordinary', pid+'-')


def sat_rows():
    """Two members one step below the mainline limit (0.29 m/s^2 of headroom)."""
    return [row_allowed('c1', -20., 0, 31.0, 31.29), row_allowed('c2', -60., 1, 31.0, 31.29)]


def commands(vids=('c1', 'c2')):
    return {v: 1.5 for v in vids}


def net_xml():
    return ('<network>'
            '<edge id="1191554636"><lane id="1191554636_0" speed="31.29"/>'
            '<lane id="1191554636_1" speed="31.29"/></edge>'
            '<edge id="1487606179"><lane id="1487606179_0" speed="31.29"/>'
            '<lane id="1487606179_1" speed="31.29"/><lane id="1487606179_2" speed="31.29"/></edge>'
            '<edge id="1487606178"><lane id="1487606178_0" speed="31.29"/>'
            '<lane id="1487606178_1" speed="31.29"/></edge>'
            '<edge id="11553048#1"><lane id="11553048#1_0" speed="22.22"/></edge>'
            '</network>')


# ---------------------------------------------------------------- 8.2 search clamp

class ClampIdentity(unittest.TestCase):
    """Feasible groups stay byte-identical to the pre-fix search."""

    def test_existing_fixture_matches_the_committed_golden(self):
        # Match the unchanged Git reference on the qualification host's own BLAS;
        # retain the committed Windows golden when no reference is supplied.
        from .test_niche_search_cpu import golden_reference
        golden = golden_reference()
        self.assertEqual(bytes_of(run_rows(seed=1)[0]), golden['plain1'])

    def test_all_feasible_rows_give_byte_identical_output(self):
        # armed clamp that never binds: a huge allowed speed leaves full headroom
        rows = [row_allowed(f'c{i}', -200-40*i, i % 2, 20., 1000.) for i in range(2)]
        with_rows, _ = run_rows(seed=1, rows=rows)
        without_rows, _ = run_rows(seed=1)
        self.assertEqual(bytes_of(with_rows), bytes_of(without_rows))
        self.assertNotIn('policy_duplicate', with_rows['topk'][0])

    def test_unresolvable_rows_keep_the_legacy_search(self):
        rows = [dict(row(f'c{i}', -200-40*i, 9), allowed_speed_mps=None) for i in range(2)]
        r, _ = run_rows(seed=1, rows=rows)     # lane 9 is outside the lane table
        self.assertEqual(bytes_of(r), bytes_of(run_rows(seed=1)[0]))


class ClampBehaviour(unittest.TestCase):
    """Ceiling-bound candidates collapse onto the policy centre; room keeps plans."""

    def test_saturated_duplicates_excluded_while_braking_room_remains(self):
        # every member AT the speed limit: hi = (allowed - v)/dt = 0. Upward
        # candidates collapse onto the centre (flagged, excluded from the top-k);
        # braking profiles pass through the ceiling and stay choosable.
        rows = [row_allowed(f'c{i}', -200-40*i, i % 2, 20., 20.) for i in range(2)]
        res, fake = run_rows(seed=1, rows=rows)
        for p in fake.seen:
            self.assertTrue((np.asarray(p['profile']) <= 1e-12).all())   # ceiling-clipped
        self.assertTrue(any(p.get('policy_duplicate') for p in fake.seen))
        self.assertTrue(res['accepted'])
        for p in res['topk']:
            self.assertNotIn('policy_duplicate', p)                      # excluded from topk
        json.dumps(res)

    def test_degenerate_saturated_group_declines(self):
        # the policy itself is the optimum: every ceiling-legal deviation is worse,
        # so the saturated group yields no choosable candidate at all (F6 family).
        rows = [row_allowed(f'c{i}', -200-40*i, i % 2, 20., 20.) for i in range(2)]
        fake = Fake(2);fake.target = fake.policy.copy()
        g = group_of(2)
        res = ms.search(dict(state_id='s', rows=rows, policy_residuals={v: 0. for v in g['members']}),
                        g, fake.policy, BOUNDS, fake, 'ordinary',
                        mappo_traj=fake.mappo, seed=1)
        self.assertFalse(res['accepted'])
        json.dumps(res)

    def test_headroom_group_keeps_searching_and_stays_within_limits(self):
        # 1 m/s of headroom: the ceiling binds (the centre saturates after ~5 s) but
        # braking/upward room remains and the top-k only holds non-duplicate plans.
        rows = [row_allowed(f'c{i}', -200-40*i, i % 2, 20., 21.) for i in range(2)]
        res, fake = run_rows(seed=1, rows=rows)
        members = group_of(2)['members']
        feas = ms.centre_feasibility(dict(state_id='s', rows=rows), members,
                                     np.full((2, T), .2), BOUNDS, res['durations_s'])
        self.assertTrue(res['accepted'])
        for p in fake.seen:
            a = np.asarray(p['profile'])
            self.assertTrue((a >= feas['lo'][:, :p['duration_s']]-1e-12).all())
            self.assertTrue((a <= feas['hi'][:, :p['duration_s']]+1e-12).all())
            ms.validate_plan(p, BOUNDS, res['durations_s'])
        for p in res['topk']:
            self.assertNotIn('policy_duplicate', p)
        json.dumps(res)

    def test_clamp_mirrors_damping_direct_target(self):
        from traffic_shepherd.damping import direct_target
        rows = [row_allowed('c0', -200, 0, 30.5, 31.29), row_allowed('c1', -240, 1, 0., 22.22)]
        members = ['c0', 'c1']; P = np.full((2, T), .2)
        feas = ms.centre_feasibility(dict(state_id='s', rows=rows), members, P, BOUNDS)
        cfg = SimpleNamespace(control_dt_s=1., acceleration_min_mps2=BOUNDS[0],
                              acceleration_max_mps2=BOUNDS[1])
        for i, vid in enumerate(members):
            for t in (0, 1, 5, 40):
                v = float(feas['v'][i, t])
                frame = SimpleNamespace(speed_mps=v, allowed_speed_mps=float(feas['allowed'][i]))
                for raw in (-2.9, -1., 0., .5, 1.499):
                    command, target = direct_target(frame, raw, cfg)
                    issued = fe.issued_target(v, raw, feas['allowed'][i],
                                              accel_min=BOUNDS[0], accel_max=BOUNDS[1])
                    self.assertAlmostEqual(command, float(np.clip(raw, feas['lo'][i, t], feas['hi'][i, t])))
                    self.assertAlmostEqual(issued, target)

    def test_niched_search_clamps_too(self):
        rows = [row_allowed(f'c{i}', -200-40*i, i % 2, 20., 20.) for i in range(2)]
        fake = Fake(2);fake.target = fake.policy.copy()      # only the centre is good
        g = group_of(2)
        res = ms.search(dict(state_id='s', rows=rows, policy_residuals={v: 0. for v in g['members']}),
                        g, fake.policy, BOUNDS, fake, 'ordinary',
                        mappo_traj=fake.mappo, seed=1, config=dict(niches=2))
        for p in fake.seen:
            self.assertTrue((np.asarray(p['profile']) <= 1e-12).all())   # ceiling-clipped
        self.assertTrue(any(p.get('policy_duplicate') for p in fake.seen))
        for p in res.get('topk', []):                                # duplicates excluded
            self.assertNotIn('policy_duplicate', p)
        json.dumps(res)


# ---------------------------------------------------------------- 8.1 collector gate

class Gate(unittest.TestCase):
    BOUNDS = (-3., 2.6)

    def test_saturated_upward_candidates_are_inactionable(self):
        cand = gate_plan('s', ('c1', 'c2'), +.5, self.BOUNDS)   # raw 3.1 clips to the 2.6 centre
        centre = gate_plan('c', ('c1', 'c2'), 0., self.BOUNDS)
        self.assertEqual(cand['profile'], centre['profile'])     # identical post-clip profiles
        gate = sv.actionable_gate([cand], sat_rows(), commands(), {},
                                  dict(accel_bounds=list(self.BOUNDS)))
        self.assertEqual(gate['inactionable'], [cand['id']])
        self.assertEqual(gate['actionable'], [])
        self.assertIsNotNone(gate['members_t0'])

    def test_braking_candidate_keeps_headroom(self):
        cand = gate_plan('b', ('c1', 'c2'), -2.5, self.BOUNDS)   # dips below the ceiling
        gate = sv.actionable_gate([cand], sat_rows(), commands(), {},
                                  dict(accel_bounds=list(self.BOUNDS)))
        self.assertEqual(gate['actionable'], [cand])
        self.assertEqual(gate['inactionable'], [])

    def test_unknown_context_never_blocks(self):
        cand = gate_plan('s', ('c1', 'c2'), +.5, self.BOUNDS)
        cases = (([], commands(), dict(accel_bounds=list(self.BOUNDS))),   # no rows
                 (sat_rows(), commands(), {}))                             # no bounds
        for rows, cmds, job in cases:
            gate = sv.actionable_gate([cand], rows, cmds, {}, job)
            self.assertEqual(gate['actionable'], [cand])
            self.assertEqual(gate['inactionable'], [])
            self.assertIsNone(gate['members_t0'])

    def test_uncommanded_member_is_inactionable(self):
        cand = gate_plan('b', ('c1', 'c2'), -2.5, self.BOUNDS)
        gate = sv.actionable_gate([cand], sat_rows(), {'c1': 1.5}, {},
                                  dict(accel_bounds=list(self.BOUNDS)))
        self.assertEqual(gate['inactionable'], [cand['id']])

    def test_lane_table_fallback_resolves_allowed_speeds(self):
        from traffic_shepherd.c4.geometry import lane_allowed_speeds_mps
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'assets'/'research'/'highway_merge_shepherd_long').mkdir(parents=True)
            path = root/'assets'/'research'/'highway_merge_shepherd_long'/'osm.net.xml.gz'
            with gzip.open(path, 'wt') as f:
                f.write(net_xml())
            self.assertEqual(lane_allowed_speeds_mps(path), {0: 31.29, 1: 31.29, 2: 22.22})
            old = os.environ.get('RESEARCH_DENSE_ROOT')
            os.environ['RESEARCH_DENSE_ROOT'] = str(root)
            try:
                rows = [dict(vid='c1', s_m=-20., v_mps=31., lane=0, is_cav=True),
                        dict(vid='c2', s_m=-60., v_mps=0., lane=2, is_cav=True)]
                self.assertEqual(fe.resolve_allowed_mps(rows, ('c1', 'c2')),
                                 dict(c1=31.29, c2=22.22))
                cand = gate_plan('b', ('c1', 'c2'), -2.5, self.BOUNDS)
                gate = sv.actionable_gate([cand], rows, commands(), {},
                                          dict(accel_bounds=list(self.BOUNDS)))
                self.assertEqual(gate['actionable'], [cand])   # stopped ramp member can brake
                self.assertIsNotNone(gate['members_t0'])
                self.assertEqual(gate['members_t0']['c2']['headroom_mps2'], 22.22)
            finally:
                if old is None:
                    os.environ.pop('RESEARCH_DENSE_ROOT', None)
                else:
                    os.environ['RESEARCH_DENSE_ROOT'] = old

    def test_missing_net_leaves_speeds_unresolvable(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get('RESEARCH_DENSE_ROOT')
            os.environ['RESEARCH_DENSE_ROOT'] = tmp          # no net file here
            try:
                rows = [dict(vid='c1', s_m=-20., v_mps=31., lane=0, is_cav=True)]
                self.assertIsNone(fe.resolve_allowed_mps(rows, ('c1',)))
                cand = gate_plan('b', ('c1', 'c2'), -2.5, self.BOUNDS)
                gate = sv.actionable_gate([cand], rows, {'c1': 1.5}, {},
                                          dict(accel_bounds=list(self.BOUNDS)))
                self.assertEqual(gate['inactionable'], [cand['id']])   # uncommanded c2
            finally:
                if old is None:
                    os.environ.pop('RESEARCH_DENSE_ROOT', None)
                else:
                    os.environ['RESEARCH_DENSE_ROOT'] = old


# ------------------------------------------------- verified_guide integration

def skip_verifier(*a, **kw):
    raise AssertionError('verification must be skipped for inactionable groups')


class GuideSkip(unittest.TestCase):
    BOUNDS = (-3., 2.6)

    def job(self):
        return dict(id='e', world=dict(seed=7), arm='B', update=5, runtime='rt',
                    checkpoint='ck', stop=1200, accel_bounds=list(self.BOUNDS))

    def prefixes(self, t0=400):
        return {str(t): fp(t, base_delay(7, t)) for t in range(300, t0+1)}

    def guide(self, proposal):
        return lambda *a: json.loads(json.dumps(proposal))

    def test_saturated_group_skips_verification_without_charge_or_snapshot(self):
        cands = [gate_plan('up1', ('c1', 'c2'), +.5, self.BOUNDS),
                 gate_plan('up2', ('c1', 'c2'), +1., self.BOUNDS)]
        proposal = dict(accepted=True, plan=cands[0], topk=cands, behaviour=dict(route='wm'))
        gen = torch.Generator().manual_seed(5)
        call = sv.verified_guide(self.guide(proposal), skip_verifier, FakeController(400),
                                 self.prefixes(), self.job(), gen)
        out = call(sat_rows(), commands(), 400, dict(action=[[0.]]))
        self.assertFalse(out['accepted'])
        self.assertFalse(out.get('sumo_verified', False))
        self.assertIn('no actionable candidate', out['reason'])
        self.assertEqual(out['behaviour']['verify_k']['inactionable'],
                         [c['id'] for c in cands])
        self.assertEqual(out['behaviour']['verify_k']['available'], 2)
        # exactly one pad draw consumed: the live CRN stream stays aligned
        ref = torch.Generator().manual_seed(5);torch.rand((), generator=ref)
        self.assertTrue(torch.equal(gen.get_state(), ref.get_state()))

    def test_headroom_group_verifies_only_actionable_candidates(self):
        good = gate_plan('brake', ('c1', 'c2'), -2.5, self.BOUNDS)
        dead = gate_plan('up', ('c1', 'c2'), +.5, self.BOUNDS)
        proposal = dict(accepted=True, plan=good, topk=[dead, good], behaviour=dict(route='wm'))
        seen = {}
        receipt = sv.verify(snapshot(), [gate_plan('m', ('c1', 'c2'), 0., self.BOUNDS)],
                            FakePlant(), lambda *a: None)
        def verifier(snap, c, proposer, **kw):
            seen['cands'] = c;seen['kw'] = kw
            return receipt
        gen = torch.Generator().manual_seed(5)
        call = sv.verified_guide(self.guide(proposal), verifier, FakeController(400),
                                 self.prefixes(), self.job(), gen)
        out = call(sat_rows(), commands(), 400, dict(action=[[0.]]))
        self.assertEqual([c['id'] for c in seen['cands']], [good['id']])   # dead dropped
        self.assertEqual(out['behaviour']['verify_k']['inactionable'], [dead['id']])
        self.assertEqual(out['behaviour']['verify_k']['available'], 2)     # pre-gate count
        self.assertIn('members_t0', seen['kw'])                            # 8.3 fields flow
        ref = torch.Generator().manual_seed(5);torch.rand((), generator=ref)
        self.assertTrue(torch.equal(gen.get_state(), ref.get_state()))

    def test_members_t0_reaches_the_receipt(self):
        rows = sat_rows()
        rows[0]['leader'] = ('hdv1', 12.5)                      # sumo_io subscription tuple
        rows[1]['leader_v_mps'] = 28.                           # explicit field variant
        cands = [gate_plan('brake', ('c1', 'c2'), -2.5, self.BOUNDS)]
        proposal = dict(accepted=True, plan=cands[0], topk=cands, behaviour=dict(route='wm'))
        seen = {}
        def verifier(snap, c, proposer, members_t0=None):
            seen['members_t0'] = members_t0
            return sv.verify(snapshot(), c, FakePlant(), lambda *a: None, members_t0=members_t0)
        gen = torch.Generator().manual_seed(5)
        sample = dict(action=[[0.]], context=dict(members=['c1', 'c2']), own_action=[[.4], [-.1]])
        call = sv.verified_guide(self.guide(proposal), verifier, FakeController(400),
                                 self.prefixes(), self.job(), gen)
        out = call(rows, dict(c1=2.5, c2=-.5), 400, sample)
        mt0 = seen['members_t0']
        self.assertEqual(set(mt0), {'c1', 'c2'})
        self.assertEqual(mt0['c1']['allowed_mps'], 31.29)
        self.assertAlmostEqual(mt0['c1']['headroom_mps2'], 0.29)
        self.assertTrue(mt0['c1']['policy_raw_at_clip'])        # 2.5 + .4 clips at 2.6
        self.assertFalse(mt0['c2']['policy_raw_at_clip'])       # -.5 - .1
        self.assertAlmostEqual(mt0['c1']['policy_raw_mps2'], 2.6)
        self.assertEqual(mt0['c1']['leader_gap_m'], 12.5)
        self.assertEqual(mt0['c2']['leader_v_mps'], 28.)
        self.assertEqual(out['behaviour']['sumo_verification']['status'], 'accepted')

    def test_legacy_receipt_bytes_without_members_t0(self):
        cand = gate_plan('brake', ('c1', 'c2'), -2.5, self.BOUNDS)
        r = sv.verify(snapshot(), [cand], FakePlant(), lambda *a: None)
        self.assertNotIn('members_t0', r)
        r2 = sv.verify(snapshot(), [cand], FakePlant(), lambda *a: None,
                        members_t0=dict(c1=dict(allowed_mps=31.29)))
        self.assertEqual(r2['members_t0'], dict(c1=dict(allowed_mps=31.29)))
        verify_sealed(r2, 'receipt_sha256')


if __name__ == '__main__':
    unittest.main()
