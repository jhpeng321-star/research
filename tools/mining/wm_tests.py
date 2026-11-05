"""Meaningful CPU contracts; no bootstrap, CUDA, simulator, or filesystem writes.

Run explicitly: python -B -m unittest tools.mining.wm_tests -v
"""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from .wm_adapter import (UnsupportedLaneABI, U0Sampler, compile_controller,
                         decoder_factory, require_longitudinal)
from .wm_contract import U0_SHA, grid, validate_query
from .wm_prescreen import rankings


def query():
    rows = [dict(vid='main', s_m=-30., v_mps=5., lane=1, is_cav=True, length_m=5.),
            dict(vid='ramp', s_m=-20., v_mps=5., lane=2, is_cav=True, length_m=5.)]
    return dict(ego='main', source_sha256='a' * 64,
        record=dict(state_id='captured', world_id=986000, time_s=360, rows=rows,
                    group=['main', 'ramp'], commands={'main': .1, 'ramp': .2},
                    policy_residuals={'main': 0., 'ramp': 0.}, history=[]),
        base_continuation=dict(checkpoint='frozen_u0.pt', checkpoint_sha256=U0_SHA,
            deterministic=False, generator_state_after_t0=torch.Generator().manual_seed(7).get_state().tolist(),
            memory_after_t0={'main': 0., 'ramp': 0.}))


def native_sample(self, context, generator, deterministic=False, memory=None):
    """Independent sampler fixture with same documented AR recursion/draw order."""
    memory = memory or {}
    n = len(context['members']); mean, _ = self(context)
    previous = torch.tensor([[memory.get(v, 0.)] for v in context['members']]).reshape(-1, 1)
    noise = previous * math.exp(-.2)
    innovation = math.sqrt(.5) * torch.randn((1, 1), generator=generator)
    innovation = innovation + math.sqrt(.5) * torch.randn((n, 1), generator=generator)
    noise = noise + self.log_std.exp() * math.sqrt(1 - math.exp(-.4)) * innovation
    action = mean if deterministic else mean + noise
    return dict(own_action=action.tolist()), dict(zip(context['members'], noise.flatten().tolist()))


class Move:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


def offsets(move, t, state=None):
    active = move.onset <= t < move.onset + move.hold
    if move.family == 'wall' and any((state or {}).get('s_m', {}).get(v, -999) >= -100 for v in move.members):
        active = False
    return {v: -move.depth if active else 0. for v in move.members}


def roster(state, ids):
    return [dict(vid=v, s_m=float(state['s_m'][0, i]), is_cav=bool(state['is_cav'][0, i]))
            for i, v in enumerate(ids) if bool(state['present'][0, i])]


def fixture_controller(mp, candidate_actions, initial_rows):
    ids = [r['vid'] for r in initial_rows]
    fixed = [None for a in candidate_actions]
    tick = 0
    def injected(state, accel, **kw):
        return accel
    def safe_accel(state, raw, **kw):
        mp.calls.append(raw.clone())
        return raw.clamp(-.7, .7)
    def callback(state):
        nonlocal tick
        out = injected(state, mp.base)
        tick += 1
        return out
    callback.native_lane_mask = mp.native_lane_mask
    return callback


class Contracts(unittest.TestCase):
    def weights(self):
        return {'actor.4.weight': torch.zeros(1, 64), 'actor.4.bias': torch.zeros(1),
                'log_std': torch.tensor([math.log(.5)])}

    def test_query_exposure_and_locality_fail_closed(self):
        q = query(); validate_query(q)
        for change in ('truth', 'world', 'radius', 'future'):
            bad = copy.deepcopy(q)
            if change == 'truth': bad['record']['rows'][0]['driver'] = 1
            if change == 'world': bad['record']['world_id'] = 969420
            if change == 'radius': bad['record']['rows'][1]['s_m'] = 500.
            if change == 'future': bad['record']['time_s'] = 601
            with self.assertRaises(ValueError): validate_query(bad)

    def test_lane_grid_retains_roles_timing_and_refuses(self):
        actions, _ = grid(query(), ('yield_lane', 'open_gap_merge'))
        joints = [a for a in actions if 'wm_move' in a and 'solo_of' not in a]
        self.assertEqual(len(joints), 15)
        for a in joints:
            self.assertEqual(a['wm_move']['members'], ['main', 'ramp'])
            m = a['wm_move']; actor = 0 if m['family'] == 'yield_lane' else 1
            at = m['onset'] + int(actor == 1)
            self.assertEqual(a['lane_actions'][actor][at], 1)
            self.assertEqual(sum(map(sum, a['lane_actions'])), 1)
        with self.assertRaises(UnsupportedLaneABI): require_longitudinal(actions)

    def test_wall_geometry_and_full_hold(self):
        q = query()
        for i, r in enumerate(q['record']['rows']):
            r.update(s_m=-500. + i * 5, v_mps=2., lane=i)
        actions, _ = grid(q, ('merge_timed_wall',))
        self.assertEqual(len(actions), 38)  # 12 joint + 24 exact member solos + two bases
        require_longitudinal(actions)
        q['record']['rows'][1]['s_m'] = -99.
        with self.assertRaises(ValueError): grid(q, ('merge_timed_wall',))  # no longer ego local

    def test_original_sampler_state_and_lexical_order(self):
        q = query(); rows = list(reversed(q['record']['rows']))
        a = U0Sampler(q['base_continuation'], q['record']['policy_residuals'], self.weights(), native_sample)
        b = U0Sampler(q['base_continuation'], q['record']['policy_residuals'], self.weights(), native_sample)
        self.assertEqual(a.at(rows, 0), {'main': 0., 'ramp': 0.})
        self.assertEqual(a.at(rows, 1), b.at(list(reversed(rows)), 1))
        self.assertTrue(torch.equal(a.generator.get_state(), b.generator.get_state()))
        bad = self.weights(); bad['actor.4.bias'][0] = .01
        with self.assertRaises(ValueError): U0Sampler(q['base_continuation'], {}, bad, native_sample)

    def test_original_policy_sample_source_matches_independent_recursion(self):
        # Execute the unchanged native sampler method, with only its unused
        # log-probability reducer mocked. No policy/model is constructed.
        source = Path('E:/agent-work/c4-grp-lane/tools/coopmwm/acceleration.py')
        if not source.is_file():
            self.skipTest('local native source not available on this host')
        tree = ast.parse(source.read_text(encoding='utf-8'))
        policy = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Policy')
        sample = next(n for n in policy.body if isinstance(n, ast.FunctionDef) and n.name == 'sample')
        ns = dict(torch=torch, math=math, PHI=math.exp(-1 / 5), RHO=.5,
                  gaussian_log_prob=lambda *args: torch.tensor(0.))
        exec(compile(ast.Module(body=[sample], type_ignores=[]), '<native-policy-sample>', 'exec'), ns)
        q = query()
        native = U0Sampler(q['base_continuation'], q['record']['policy_residuals'], self.weights(), ns['sample'])
        reference = U0Sampler(q['base_continuation'], q['record']['policy_residuals'], self.weights(), native_sample)
        for t in range(1, 8):
            self.assertEqual(native.at(q['record']['rows'], t), reference.at(q['record']['rows'], t))
            self.assertTrue(torch.equal(native.generator.get_state(), reference.generator.get_state()))

    def test_dynamic_offsets_halfstep_safety_solo_release_and_hdvs(self):
        q = query(); q['base_continuation']['deterministic'] = True
        current = {'main': .123456789, 'ramp': .2}
        move = dict(family='wall', members=['main', 'ramp'], onset=0, hold=20, depth=1.)
        actions = [dict(id='joint', wm_move=move, wm_active_members=['main', 'ramp']),
                   dict(id='solo', wm_move=move, wm_active_members=['main'])]
        stats = {'active_member_seconds': 0}
        make = decoder_factory(q['base_continuation'], current, self.weights(), native_sample,
                               SimpleNamespace(Move=Move, offsets=offsets), roster, stats)
        compiled = compile_controller(fixture_controller, make)
        mp = SimpleNamespace(base=torch.tensor([[.1, .2, .3], [.1, .2, .3]]), calls=[],
                             native_lane_mask=object())
        cb = compiled(mp, actions, [{'vid': v} for v in ('main', 'ramp', 'hdv')])
        state = dict(s_m=torch.tensor([[-200., -190., -195.]] * 2),
                     present=torch.ones(2, 3, dtype=torch.bool),
                     is_cav=torch.tensor([[True, True, False]] * 2), lane=torch.zeros(2, 3))
        first = cb(state); second = cb(state)
        expected_delta = float(torch.tensor(current['main'], dtype=torch.float32) + torch.tensor(-1.))
        expected = (mp.base[0, 0].double() + expected_delta).float()
        self.assertEqual(mp.calls[0][0, 0], expected)
        self.assertTrue(torch.equal(first, second))  # hold ONE decision, no extra RNG halfstep
        self.assertEqual(float(first[0, 0]), float(torch.tensor(-.7)))  # original safety still acts
        self.assertEqual(float(first[1, 1]), float((mp.base[1, 1].double() + float(torch.tensor(.2))).float()))
        self.assertTrue(torch.equal(first[:, 2], mp.base[:, 2]))  # HDV never commanded
        self.assertIs(cb.native_lane_mask, mp.native_lane_mask)
        state['s_m'][:, 0] = -199.
        state['s_m'][:, 1] = -99.
        released = cb(state)
        self.assertTrue(torch.equal(released, mp.base))  # live wall release, zero deterministic u0
        self.assertEqual(stats['active_member_seconds'], 3)

    def test_paired_crn_rankings_and_mismatch(self):
        actions = [dict(id='policy'), dict(id='MAPPO'),
                   dict(id='j', wm_move={}), dict(id='s0', solo_of='j'), dict(id='s1', solo_of='j')]
        costs = {'policy': 10, 'MAPPO': 11, 'j': 6, 's0': 8, 's1': 9}
        scores = {str(h): {a: [dict(draw_id=str(k), team_delay=v) for k in range(8)]
                          for a, v in costs.items()} for h in (60, 120, 300, 600)}
        self.assertEqual(rankings(scores, actions)[0]['horizons']['300']['mean_synergy'], 2)
        scores['600']['s0'][0]['draw_id'] = 'different'
        with self.assertRaises(ValueError): rankings(scores, actions)


if __name__ == '__main__':
    unittest.main()
