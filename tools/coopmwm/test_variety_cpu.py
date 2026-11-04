"""CPU tests for option ``search_families`` (plan variety: additive decision groups,
structured candidate generators, R family-mix fairness, family tags in receipts).

Real torch on CPU with the synthetic WM stubs; no SUMO, no WM checkpoint, no GPU job.
"""
import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import coop_search as cs
from . import ls_collection as ls
from . import mppi_search as ms
from . import replay_bank as rb
from . import sumo_verify as sv
from . import variety as va
from .test_coop_search_cpu import OFF_CASES, JointStub, coop_search_run, golden_reference, group as coop_group
from .test_mppi_search_cpu import BOUNDS, Fake, T, row

LIVE = dict(iterations=1, screen_K=2, top_k=8, finalists_per_k=1)   # live label-list8 shape
VC = dict(variety_families=True)
# Patch targets must resolve through THIS module's own package instance (the runner may
# import it top-level or under tools.*: one namespace, never both).
PKG = ls.__package__


def tgt(name):
    return PKG+'.'+name


def merge_group():
    """Ramp + mainline CAVs inside the observable merge zone (gap_open / ramp_assist)."""
    return ms.group_components([row('m1', -50., 1, v=20.), row('m0', -60., 1, v=19.),
                                row('r0', -70., 2, v=18.)])[0]


def lane_group():
    """Multi-lane abreast CAVs (wall candidates), lanes [0, 1, 0]."""
    return coop_group(3)


def split(**kw):
    cfg = dict(VC, **kw)
    vcfg = va.split_config(cfg)
    return vcfg


def sha_text(text):
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


class Config(unittest.TestCase):
    def test_registered_keys_and_validation(self):
        cfg = dict(VC)
        v = va.split_config(cfg)
        self.assertEqual(sorted(cfg), sorted(va.VARIETY_REGISTERED))   # resolved echo
        self.assertEqual(v['variety_structured_share'], .25)
        self.assertEqual(v['variety_wall_span_m'], 75.)     # wall v2.1 executor MAX_WALL_SPAN_M
        self.assertEqual(v['variety_merge_zone_m'], (-100., 75.44))   # coopfree.selection zone
        self.assertEqual(v['variety_gap_link_m'], 100.)     # coopbandit mainline association window
        self.assertEqual(v['variety_ramp_lane'], 2)         # merge_lane
        self.assertEqual(v['variety_mainline_lane'], 1)     # merge_target_lane
        self.assertEqual(va.FAMILIES, ('free', 'wall', 'gap_open', 'ramp_assist'))

    def test_off_and_unknown_keys(self):
        cfg = dict(variety_families=False, variety_structured_share=.9)
        self.assertIsNone(va.split_config(cfg))
        self.assertEqual(cfg, {})                            # keys popped even when off
        with self.assertRaises(ValueError):
            va.split_config(dict(variety_bogus=1))
        with self.assertRaises(ValueError):
            va.split_config(dict(VC, variety_structured_share=1.5))
        with self.assertRaises(ValueError):
            va.split_config(dict(VC, variety_structured_share=-.1))
        with self.assertRaises(ValueError):
            va.split_config(dict(VC, variety_merge_zone_m=(10., -10.)))
        with self.assertRaises(ValueError):
            va.split_config(dict(VC, variety_ramp_lane=1, variety_mainline_lane=1))
        with self.assertRaises(ValueError):
            va.split_config(dict(VC, variety_wall_min_lanes=1))
        with self.assertRaises(ValueError):
            va.split_config(dict(VC, variety_hold_window_s=(5., 1.)))

    def test_plan_config_leaves_caller_dict_untouched(self):
        cfg = dict(VC, variety_structured_share=.5)
        v = va.plan_config(cfg)
        self.assertEqual(v['variety_structured_share'], .5)
        self.assertEqual(cfg, dict(VC, variety_structured_share=.5))
        self.assertIsNone(va.plan_config({}))
        self.assertIsNone(va.plan_config(dict(variety_families=False)))


class GroupSelection(unittest.TestCase):
    rows = [row('a', -30., 0), row('b', -35., 1), row('c', -40., 2), row('d', -130., 0)]

    def test_off_is_exactly_group_components(self):
        from .mppi_search import group_components
        for rows in (self.rows, [row('x', -10., 1)], []):
            self.assertEqual(va.decision_groups(rows), group_components(rows))
            self.assertEqual(va.decision_groups(rows, None), group_components(rows))

    def test_additional_groups_are_additive_and_deterministic(self):
        base = ms.group_components(self.rows)
        groups = va.decision_groups(self.rows, split())
        self.assertEqual([g['members'] for g in groups[:len(base)]], [g['members'] for g in base])
        self.assertEqual([g['members'] for g in groups],
                         [['a', 'b', 'c', 'd'], ['a', 'b', 'c'], ['b', 'c']])
        self.assertEqual([g.get('variety_kind') for g in groups], [None, 'wall', 'gap_open'])
        # Registered distances only: the wall rule carries its span, the gap rule its link.
        self.assertEqual(groups[1]['rule']['span_m'], va.VARIETY_REGISTERED['variety_wall_span_m'])
        self.assertEqual(groups[1]['rule']['min_lanes'], va.VARIETY_REGISTERED['variety_wall_min_lanes'])
        self.assertEqual(groups[2]['rule']['chain_m'], va.VARIETY_REGISTERED['variety_gap_link_m'])
        self.assertEqual(groups[2]['rule']['max_size'], ms.REGISTERED['max_size'])
        again = va.decision_groups(self.rows, split())
        self.assertEqual([g['id'] for g in again], [g['id'] for g in groups])
        json.dumps(groups)

    def test_duplicates_and_size_cap_and_exclude(self):
        # A single tight multi-lane component: base == wall == gap candidate set (deduped).
        rows = [row('a', -30., 1), row('c', -40., 2)]
        self.assertEqual([g['members'] for g in va.decision_groups(rows, split())], [['a', 'c']])
        # More mainline CAVs than max_size around one ramp: capped to the registered max_size,
        # and the gap group differs from the registered chain window (the ramp stays in the
        # chain window, the downstream mains fill the gap group).
        rows = [row('r', -100., 2)]+[row(f'm{i}', -50-9*i, 1) for i in range(6)]
        gap = [g for g in va.decision_groups(rows, split()) if g.get('variety_kind') == 'gap_open']
        self.assertEqual(len(gap), 1)
        self.assertEqual(len(gap[0]['members']), ms.REGISTERED['max_size'])
        self.assertNotIn('r', gap[0]['members'])           # size cap keeps the downstream mains
        self.assertTrue(any('r' in g['members'] for g in va.decision_groups(rows, split())
                            if g.get('variety_kind') is None))
        # Committed members are excluded from the additional groups too, and the base
        # groups stay first and unchanged.
        base = ms.group_components(self.rows, exclude={'a'})
        got = va.decision_groups(self.rows, split(), exclude={'a'})
        self.assertEqual([g['members'] for g in got[:len(base)]], [g['members'] for g in base])

    def test_no_ramp_in_zone_gives_no_gap_groups(self):
        rows = [row('m', -50., 1), row('r', 300., 2)]   # ramp outside the merge zone
        base = ms.group_components(rows)
        self.assertEqual([g['members'] for g in va.decision_groups(rows, split())],
                         [g['members'] for g in base])


class Generators(unittest.TestCase):
    def _gen(self, group, family, seed=7):
        rng = np.random.default_rng(seed)
        P = np.zeros((len(group['members']), T))           # zero policy: pace rows == profile
        return va._GENERATORS[family](group, P, BOUNDS, 'ordinary', rng, split(), None,
                                      ms.allowed_durations())

    def test_wall_sync_decelerate_release_shape(self):
        g = ms.group_components([row('a', -30., 0, v=20.), row('b', -35., 1, v=22.)])[0]
        p = self._gen(g, 'wall')
        self.assertEqual(p['variety_family'], 'wall')
        self.assertIn(p['duration_s'], ms.allowed_durations())
        ms.validate_plan(p, BOUNDS, ms.allowed_durations())
        prof = np.asarray(p['profile'], float)
        hold = prof[:, 20:60].mean(1).min()                # decelerate together
        self.assertLess(hold, prof[:, :5].mean(1).max())   # below the sync phase
        self.assertGreater(prof[:, -5:].mean(1).max(), hold)   # released afterwards

    def test_gap_open_eases_before_predicted_arrival_then_recovers(self):
        g = merge_group()
        p = self._gen(g, 'gap_open', seed=3)
        self.assertEqual(p['variety_family'], 'gap_open')
        ms.validate_plan(p, BOUNDS, ms.allowed_durations())
        prof = np.asarray(p['profile'], float)
        byid = {r['vid']: r for r in g['rows']}
        mainline = [g['members'].index(v) for v in ('m1', 'm0')]
        i = min(mainline, key=lambda j: prof[j].min())     # the eased mainline member
        merger = byid['r0']
        arrival = (byid[g['members'][i]]['s_m']-merger['s_m'])/merger['v_mps']
        active = np.abs(prof[i]) > 1e-9
        onset = int(np.argmax(active)) if active.any() else 0
        self.assertLessEqual(onset, max(1, int(np.floor(arrival))-1))   # eases BEFORE arrival
                                                                          # (clamped to 1 when due)
        hold_dev = prof[i, onset:min(onset+15, p['duration_s'])].min()
        self.assertLess(hold_dev, 0.)
        others = [j for j in range(len(g['members'])) if j != i]
        np.testing.assert_allclose(prof[others], 0., atol=1e-9)   # policy continues
        self.assertGreater(prof[i, min(p['duration_s']-1, onset+40)], hold_dev)   # recovery

    def test_ramp_assist_coordinates_ramp_and_mainline_ahead(self):
        g = merge_group()
        p = self._gen(g, 'ramp_assist', seed=11)
        self.assertEqual(p['variety_family'], 'ramp_assist')
        ms.validate_plan(p, BOUNDS, ms.allowed_durations())
        prof = np.asarray(p['profile'], float)
        ahead = g['members'].index('m1')                   # mainline CAV furthest downstream
        ramp = g['members'].index('r0')
        self.assertLess(prof[ahead].min(), 0.)             # the CAV ahead eases
        self.assertGreater(prof[ramp].max(), 0.)           # the ramp member synchronises to it
        third = next(j for j in range(len(g['members'])) if j not in (ahead, ramp))
        np.testing.assert_allclose(prof[third], 0., atol=1e-9)

    def test_unavailable_family_returns_none_and_pool_rotation_falls_through(self):
        g = lane_group()                                   # no ramp-lane member: no gap families
        rng = np.random.default_rng(1)
        P = np.zeros((3, T))
        self.assertIsNone(va.gap_open_candidate(g, P, BOUNDS, 'ordinary', rng, split(), None,
                                                ms.allowed_durations()))
        pool = va.pool_candidates(g, P, BOUNDS, 'ordinary', rng, split(), None, 6,
                                  ms.allowed_durations())
        self.assertEqual(len(pool), 6)                     # wall slots cover the rotation
        self.assertTrue(all(p['variety_family'] == 'wall' for p in pool))

    def test_pool_candidates_balanced_and_in_bounds(self):
        g = merge_group()
        rng = np.random.default_rng(5)
        pool = va.pool_candidates(g, np.zeros((3, T)), BOUNDS, 'ordinary', rng, split(), None,
                                  9, ms.allowed_durations())
        self.assertEqual(len(pool), 9)
        self.assertEqual({p['variety_family'] for p in pool}, set(va.STRUCTURED))
        for p in pool:
            ms.validate_plan(p, BOUNDS, ms.allowed_durations())
            prof = np.asarray(p['profile'], float)
            self.assertTrue((prof >= BOUNDS[0]-1e-9).all())
            self.assertTrue((prof <= BOUNDS[1]+1e-9).all())

    def test_feasibility_clamp_applies_to_structured_profiles(self):
        g = merge_group()
        members = g['members']
        feas = dict(members=members, allowed=np.full(3, 10.), v0=np.full(3, 20.),
                    lo=np.full((3, T), -3.), hi=np.full((3, T), .1))
        rng = np.random.default_rng(2)
        p = va.wall_candidate(g, np.zeros((3, T)), BOUNDS, 'ordinary', rng, split(), feas,
                              ms.allowed_durations())
        prof = np.asarray(p['profile'], float)
        self.assertTrue((prof <= .1+1e-9).all())           # actuator ceiling honoured
        self.assertLess(prof.min(), -.05)


class FamilyStub:
    """Deterministic WM that scores by the candidate's family tag: a legitimate black box
    for the search - structured candidates compete through the SAME score path and reach
    the finalists only because this WM prefers them (never by rule)."""

    def __init__(self, m):
        self.policy = np.full((m, T), .2); self.mappo = np.zeros((m, T)); self.calls = []
        self.seen = []

    def __call__(self, record, actions, K, H):
        self.calls.append(([a['id'] for a in actions], K, H))
        common = np.random.default_rng(7).normal(0., .01, K)
        out = {}
        for a in actions:
            if a['id'] == 'policy':
                cost = 0.
            elif a['id'] == 'MAPPO':
                cost = 30.
            else:
                self.seen.append(a)
                cost = 10. if a.get('variety_family') in va.STRUCTURED else 40.
            own = np.random.default_rng(int(ms._digest(a['id'])[:8], 16)).normal(0, .01, K)
            out[a['id']] = [dict(team_delay=cost+float(common[k])+float(own[k])) for k in range(K)]
        return out


class SearchIntegration(unittest.TestCase):
    def test_off_golden_and_variety_off_variants_identical(self):
        ref = golden_reference()
        variants = (None, dict(variety_families=False), dict(variety_structured_share=.3),
                    dict(variety_families=False, variety_structured_share=.3))
        for variant in variants:
            got = {}
            for name, niches, m, cfg in OFF_CASES:
                fake = Fake(m);g = coop_group(m)
                config = dict(cfg, **(variant or {}))
                if niches > 1:
                    config['niches'] = niches
                res = ms.search(dict(state_id='s', policy_residuals={v: 0. for v in g['members']}), g,
                                fake.policy, BOUNDS, fake, 'ordinary', mappo_traj=fake.mappo,
                                seed=5, config=config)
                got[name] = dict(result=sha_text(json.dumps(res, sort_keys=True)),
                                 calls=sha_text(json.dumps(fake.calls)),
                                 coop_key=any('variety' in k for k in res) or
                                          any(k.startswith('variety') for k in res['config']))
            self.assertEqual(got, ref, variant)
            self.assertFalse(any(v['coop_key'] for v in got.values()))

    def test_unregistered_variety_key_rejected(self):
        with self.assertRaises(ValueError):
            coop_search_run(Fake(3), coop_group(3), **dict(VC, variety_bogus=1))

    def test_structured_share_capped_and_free_dominates(self):
        g = coop_group(3)
        stub = Fake(3)
        res = coop_search_run(stub, g, **dict(VC, **LIVE))
        seen = {p['id']: p for p in stub.seen}             # finalists are evaluated twice
        families = {}
        for p in seen.values():
            families[p.get('variety_family')] = families.get(p.get('variety_family'), 0)+1
        n = res['scaling']['population']
        budget = int(np.floor(.25*n))
        structured = sum(v for k, v in families.items() if k in va.STRUCTURED)
        self.assertEqual(structured, budget)               # structured take exactly their capped slots
        self.assertGreater(families.get('free', 0), budget)   # free search dominates
        self.assertEqual(len(seen), n)                     # total population unchanged
        self.assertEqual(res['imagined_futures'], sum(len(ids)*K for ids, K, H in stub.calls))
        self.assertEqual(res['config']['variety_structured_share'], .25)
        for p in seen.values():
            self.assertIn(p.get('variety_family'), va.FAMILIES)

    def test_structured_candidates_compete_and_can_reach_finalists(self):
        g = coop_group(3)
        stub = FamilyStub(3)
        res = coop_search_run(stub, g, **dict(VC, **LIVE))
        seen = {p['id']: p for p in stub.seen}
        for p in seen.values():
            self.assertIn(p.get('variety_family'), va.FAMILIES)
        # WM scoring is identical for every candidate: one screen call per iteration carries
        # the whole population (structured + free) on the same paired futures.
        screens = [(ids, K, H) for ids, K, H in stub.calls if K == 2]
        self.assertEqual(len(screens), 1)
        self.assertEqual(len(screens[0][0]), res['scaling']['population']+2)
        # A structured plan that the WM scores better than the free pool reaches the
        # finalists and the top-k on that score alone (never by rule).
        structured_ids = {p['id'] for p in seen.values() if p.get('variety_family') != 'free'}
        self.assertTrue(structured_ids & {f['id'] for f in res['finalists']})
        self.assertTrue(structured_ids & {p['id'] for p in res['topk']})
        chosen = res['plan']
        self.assertIn(chosen.get('variety_family'), va.STRUCTURED)

    def test_no_guarantee_without_reward_and_zero_share(self):
        g = coop_group(3)
        # Trap: only (member0 brakes AND member1 accelerates) helps; structured plans
        # (synchronise/decelerate together) never win, and the topk carries none of them.
        stub = JointStub(g, solo_w=0., joint_w=60.)
        res = coop_search_run(stub, g, **dict(VC, **LIVE))
        self.assertTrue(res['topk'])
        for p in res['topk']:
            self.assertEqual(p.get('variety_family'), 'free')
        stub0 = Fake(3)
        res0 = coop_search_run(stub0, g, **dict(VC, variety_structured_share=0., **LIVE))
        self.assertTrue(all(p.get('variety_family') == 'free' for p in stub0.seen))
        self.assertEqual(res0['imagined_futures'], sum(len(ids)*K for ids, K, H in stub0.calls))

    def test_niche_mode_per_niche_cap_and_totals_unchanged(self):
        g = coop_group(3)
        stub = Fake(3)
        res = coop_search_run(stub, g, **dict(VC, niches=4, **LIVE))
        self.assertEqual(res['niche_sizes'], [16, 16, 16, 12])
        seen = {p['id']: p for p in stub.seen}
        families = {}
        for p in seen.values():
            families[p.get('variety_family')] = families.get(p.get('variety_family'), 0)+1
        expected = sum(int(np.floor(.25*s)) for s in res['niche_sizes'])
        self.assertEqual(sum(v for k, v in families.items() if k in va.STRUCTURED), expected)
        self.assertEqual(len(seen), res['scaling']['population'])
        for p in seen.values():
            self.assertIn(p.get('variety_family'), va.FAMILIES)


class RFairness(unittest.TestCase):
    def _draws(self, group, config, k=8, base_seed=123):
        centre = np.full((len(group['members']), 300), .2)
        return sv.random_candidates(group, centre, BOUNDS, 'ordinary', base_seed, k=k, config=config)

    def test_no_config_and_families_off_are_identical_legacy_draws(self):
        g = coop_group(2)
        base = [p['id'] for p in self._draws(g, None)]
        self.assertEqual(base, [p['id'] for p in self._draws(g, None)])   # deterministic
        self.assertEqual(base, [p['id'] for p in self._draws(g, dict(variety_families=False))])
        self.assertEqual(base, [p['id'] for p in self._draws(g, dict(VC, variety_structured_share=0.))])
        self.assertTrue(all('variety_family' not in p for p in self._draws(g, None)))

    def test_same_family_mix_with_share_one_and_mixed(self):
        g = merge_group()
        full = self._draws(g, dict(VC, variety_structured_share=1.), k=6)
        self.assertTrue(all(p.get('variety_family') in va.STRUCTURED for p in full))
        for p in full:
            ms.validate_plan(p, BOUNDS, ms.allowed_durations())
        mixed = self._draws(g, dict(VC), k=12)
        kinds = {p.get('variety_family', 'free') for p in mixed}
        self.assertIn('free', kinds)
        self.assertTrue(kinds & set(va.STRUCTURED))
        for p in mixed:
            self.assertIn(p.get('variety_family', 'free'), va.FAMILIES)

    def test_sample_unscored_matches_share_and_falls_back_identically(self):
        g = merge_group()
        centre = [[0.]*300]*3
        legacy = ms.sample_unscored(g, centre, BOUNDS, 'ordinary', 42)
        off = ms.sample_unscored(g, centre, BOUNDS, 'ordinary', 42,
                                 config=dict(variety_families=False))
        self.assertEqual(legacy['id'], off['id'])          # byte-identical fallback
        self.assertNotIn('variety_family', off)
        zero = ms.sample_unscored(g, centre, BOUNDS, 'ordinary', 42,
                                  config=dict(VC, variety_structured_share=0.))
        self.assertEqual(zero['id'], legacy['id'])
        self.assertEqual(zero.get('variety_family'), 'free')
        ones = {ms.sample_unscored(g, centre, BOUNDS, 'ordinary', s,
                                   config=dict(VC, variety_structured_share=1.)).get('variety_family')
                for s in range(20)}
        self.assertTrue(ones and ones <= set(va.STRUCTURED))

    def test_matched_duration_respected_by_structured_draws(self):
        g = merge_group()
        centre = [[0.]*300]*3
        for s in range(8):
            p = ms.sample_unscored(g, centre, BOUNDS, 'ordinary', s,
                                   config=dict(VC, variety_structured_share=1.), duration_s=60)
            self.assertEqual(p['duration_s'], 60)


class ReceiptTags(unittest.TestCase):
    def test_family_counts_reads_receipts(self):
        def receipt(vid, chosen_index, graded):
            branches = [dict(branch='policy', index=0, status='graded', plan=None, J_veh_s=100.)]
            for i, (fam, status) in enumerate(graded, 1):
                plan = {} if fam is None else dict(variety_family=fam)
                branches.append(dict(branch=f'cand{i}', index=i, status=status,
                                     plan=plan, J_veh_s=90.))
            return dict(verification_id=vid, chosen=(chosen_index-1 if chosen_index else None),
                        grade_veh_s=(10. if chosen_index else None), branches=branches)
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'verifications';root.mkdir(parents=True)
            (root/'a.json').write_text(json.dumps(receipt('a', 1, [('wall', 'graded')])))
            (root/'b.json').write_text(json.dumps(receipt('b', 1,
                                                          [('free', 'graded'), ('wall', 'graded')])))
            (root/'c.json').write_text(json.dumps(receipt('c', 1, [(None, 'graded')])))   # legacy
            (root/'d.json').write_text(json.dumps(receipt('d', None, [('gap_open', 'failed')])))
            counts = va.family_counts(root)
            self.assertEqual(counts['wall'], dict(verified=2, admitted=1))
            self.assertEqual(counts['free'], dict(verified=1, admitted=1))
            self.assertEqual(counts['gap_open'], dict(verified=0, admitted=0))
            self.assertEqual(counts['ramp_assist'], dict(verified=0, admitted=0))
            self.assertEqual(va.family_counts(Path(d)/'missing'),
                             {k: dict(verified=0, admitted=0) for k in va.FAMILIES})


class Wiring(unittest.TestCase):
    def test_ls_collection_decision_config(self):
        self.assertIsNone(ls.decision_config(dict()))
        self.assertEqual(ls.decision_config(dict(ls_collect=dict(variety=dict(VC))))['variety_structured_share'],
                         .25)
        with self.assertRaises(ValueError):
            ls.decision_config(dict(ls_collect=dict(variety=dict(variety_bogus=1))))

    def test_replay_bank_collect_config_and_meta_config(self):
        self.assertIsNone(rb.collect_config(dict()))
        self.assertEqual(rb.collect_config(dict(collect=dict(variety=dict(VC))))['variety_families'], True)
        self.assertIsNone(rb.variety_config({}))
        self.assertEqual(rb.variety_config(dict(variety_config=dict(VC)))['variety_families'], True)
        with self.assertRaises(ValueError):
            rb.variety_config(dict(variety_config=dict(variety_bogus=1)))

    def test_install_writes_envelopes_for_variety_groups(self):
        from types import SimpleNamespace
        rows = [dict(vid=r['vid'], s_m=r['s_m'], v_mps=r['v_mps'], lane=r['lane'], is_cav=r['is_cav'])
                for r in (row('a', -30., 0), row('b', -35., 1), row('c', -40., 2), row('d', -130., 0))]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d);sample = dict(context=dict(members=['a']), own_action=[[0.]])
            controller = SimpleNamespace(active_plans=[], guide=None)
            job = dict(id='lscollect1_B3_986000', world=dict(seed=986000), policy_state={},
                       path=str(root/'episode'),
                       ls_collect=dict(root=str(root), variety=dict(VC)))
            snap = dict(samples={}, prefixes={}, t0=300)
            with patch(tgt('sumo_verify.capture'), return_value=snap), \
                 patch(tgt('exploration.regime'), return_value='ordinary'), \
                 patch('time.sleep', side_effect=AssertionError('collection blocked')):
                ls.install(controller, job, {}, torch.Generator())
                self.assertIsNone(controller.guide(rows, {}, 300, sample))
            records = [json.loads(gzip.decompress(p.read_bytes())) for p in (root/'pending').glob('*.gz')]
            members = [tuple(r['record']['group']) for r in records]
            self.assertIn(('a', 'b', 'c', 'd'), members)       # the registered chain group first
            self.assertIn(('a', 'b', 'c'), members)            # additional wall group
            self.assertIn(('b', 'c'), members)                 # additional gap_open group

    def test_collector_guide_carries_variety_meta(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d);sample = dict(context=dict(members=['a', 'b']), own_action=[[0.], [0.]])
            job = dict(id='R_meta', arm='R', queue_root=str(r), world=dict(seed=986001),
                       collector_policy_update=1, policy_state={},
                       collect=dict(slots=4, demand_phase_end_s=1200., variety=dict(VC)))
            rows = [dict(vid=r_['vid'], s_m=r_['s_m'], v_mps=r_['v_mps'], lane=r_['lane'],
                         is_cav=r_['is_cav']) for r_ in (row('a', -50., 1), row('b', -70., 2))]
            with patch(tgt('sumo_verify.capture'), return_value={}), \
                 patch(tgt('exploration.regime'), return_value='ordinary'), \
                 patch('torch.save'):
                guide = rb.collector_guide(job, None, None, {})
                self.assertIsNone(guide(rows, dict(a=0., b=0.), 300, sample))
            metas = [json.loads(p.read_text()) for p in (r/'replay_bank/R/snapshots').glob('*.json')]
            self.assertEqual(len(metas), 1)
            self.assertEqual(metas[0]['variety_config'], dict(VC))
            self.assertEqual(metas[0]['group']['members'], ['a', 'b'])

    def test_binding_preflight_gates_novelty_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            bank = [[0.]*15, [1.]*15]
            sha = cs.bank_digest(bank)
            (Path(tmp)/cs.bank_file_name(sha)).write_text(json.dumps(dict(descriptors=bank)))
            self.assertIsNone(rb.preflight_binding({}))
            self.assertIsNone(rb.preflight_binding(dict(search_config=dict(coop_objective=True))))
            cfg = dict(search_config=dict(coop_objective=True, coop_novelty=True,
                                          coop_novelty_bank_sha256=sha))
            with patch.dict(os.environ, {'COOPMWM_COOP_BANK_DIR': tmp}):
                self.assertEqual(len(rb.preflight_binding(cfg)), 2)
                bad = dict(cfg, search_config=dict(cfg['search_config'], coop_novelty_bank_sha256='0'*64))
                with self.assertRaises(ValueError):
                    rb.preflight_binding(bad)

    def test_niche_search_config_and_validate_carry_variety(self):
        from .niche_collect import search_config, validate
        plain = search_config(0)
        self.assertNotIn('variety_families', search_config(0, variety=None))
        with_variety = search_config(0, variety=dict(VC))
        self.assertEqual(with_variety, dict(plain, **VC))
        # variety rides on EVERY ordinal, independent of the coop alternation parity
        both = search_config(3, dict(coop_objective=True), True, dict(VC))
        self.assertEqual(both, dict(search_config(3, dict(coop_objective=True), True), **VC))
        self.assertEqual(validate(with_variety), with_variety)
        self.assertEqual(validate(both), both)
        with self.assertRaises(ValueError):
            validate(dict(plain, variety_bogus=1))
        with self.assertRaises(ValueError):
            validate(dict(plain, variety_families=False))
        with self.assertRaises(ValueError):
            search_config(0, variety=dict(variety_structured_share=7.))

    def test_plan_uses_variety_group_lookup(self):
        from . import mppi_runtime
        import inspect
        source = inspect.getsource(mppi_runtime.plan)
        self.assertIn('decision_groups', source)
        self.assertIn('plan_config', source)


if __name__ == '__main__':
    unittest.main()
