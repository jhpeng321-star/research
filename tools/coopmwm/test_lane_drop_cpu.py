"""N2 lane-drop observations, closing_pair / abreast_wall moves, V2V initiate/join and RSU
suggestions: CPU contracts with fakes (no simulator, GPU or network).

Real EpisodePolicy / TriggerPolicy / MoveExecutor / controller / bank code paths. Fakes: a
local N1 map-descriptor stub, a test-only trigger scoring head, fixed registered draws and
synthetic observation rows in a frame whose lane-drop point is s = 0.
"""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from . import bank_breadth as breadth, lane_drop as ld, lane_drop_rsu as rsu_mod
from . import launch_production as launch, moves, track2_backend as tb
from .acceleration import Controller
from .merge_features import ObservedVehicle, SensorReport
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)
from .trigger_policy import ppo_update
from .v2v_intent import IntentPolicy
from .v2v_runtime import controller_class
from tools.coopnative.policy import encode

PROV = 'N2 CPU fixture (MAIN 2026-11-06 newmap_spec N2)'
REPO = Path(__file__).resolve().parents[2]
PRE_N2_COMMIT = '7fb5cb17'   # c4/nm-moves after merging W1 + W2, before any N2 code


class StubMap:
    """Local stand-in for the N1 (c4/nm-abstract) map descriptor API."""
    name = 'wz3to2-stub'
    lane_ids = (0, 1, 2)
    dropped_lane = 2

    @staticmethod
    def distance_to_drop(s_m):
        return 0.-float(s_m)


MAP = ld.MapView.from_descriptor(StubMap()).to_dict()


def ld_cfg(cfg, *, features=True, moves_on=True, v2v=True, range_m=None, cap=None, rsu=None, slots=4):
    out = dict(cfg, trigger_partner_slots=slots)
    out.update(ld.config_keys(provenance=PROV, descriptor=StubMap(), features=features, moves_on=moves_on,
                              v2v=v2v, v2v_range_m=range_m, member_cap=cap, rsu=rsu))
    return out


def make(cfg, arm='B'):
    torch.manual_seed(7)
    return tb.EpisodePolicy(IntentPolicy(), cfg, arm)


def row(vid, s, lane, cav=True, v=20., **extra):
    return dict(vid=vid, s_m=float(s), v_mps=v, lane=lane, is_cav=cav, length_m=5., **extra)


def advance(rows, dt, speed=20.):
    return [dict(r, s_m=r['s_m']+speed*dt) for r in rows]


def tick(p, vehicles, now, deterministic=True, seed=1, native=.3):
    p.observe(vehicles, now, native_commands={r['vid']: native for r in vehicles if r['is_cav']})
    context = encode(vehicles)
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    sample, _ = p.sample(context, torch.Generator().manual_seed(seed), deterministic)
    return sample


def canon(value):
    return json.dumps(value, sort_keys=True, default=repr)


def families(sample, ego):
    trig = sample['track2_trigger']
    i = trig['context']['members'].index(ego)
    return [m and m['family'] for m in trig['tables'][i] if m]


class Prefer(nn.Module):
    """Test-only trigger head: score = sum_k w_k * descriptor column k."""

    def __init__(self, trigger, weights):
        super().__init__()
        hidden = trigger.base_policy.actor[-1].in_features
        base = len(trigger.families)+trigger.partner_slots+3
        ld_start = hidden+base+(trigger.extension.base.descriptor_width if trigger.extension.base else 0)
        self.cols = []
        for name, w in weights.items():
            if name in trigger.families:
                self.cols.append((hidden+trigger.families.index(name), w))
            else:
                self.cols.append((ld_start+ld.LD_DESCRIPTOR.index(name), w))

    def forward(self, x):
        return sum(w*x[..., c] for c, w in self.cols)[..., None]


FIXED = dict(onset=2, hold=40, depth=.6, target=18., gap_dv=2., zip=False)


@pytest.fixture
def fixed(monkeypatch):
    values = dict(FIXED)

    def params(rng, kind):
        moves._sampled(rng)   # keep the stream shape
        if kind == 'abreast_wall':
            return values['onset'], values['hold'], values['depth'], values['target'], 0., False
        return (values['onset'], values['hold'], values['depth'], values['target'], values['gap_dv'],
                values['zip'])
    monkeypatch.setattr(ld, '_params', params)
    monkeypatch.setattr(moves, '_sampled', lambda rng, onset=None: (2, 40, .6))
    return values


def pair_scene():
    """a: closing lane 2; b: adjacent lane 1 abreast; c: far lane 0; h1: HDV in lane 1."""
    return [row('a', -1200, 2), row('b', -1195, 1), row('c', -1210, 0), row('h1', -1150, 1, cav=False)]


# ------------------------------------------------------------------ settings / map API

def test_settings_default_off_registered_values_and_map_api(cfg):
    assert ld.settings(cfg) is None and ld.settings(dict(cfg, lane_drop=False)) is None
    assert ld.extension(cfg) is None
    on = ld_cfg(cfg, range_m=300., cap=3, rsu=dict(range_m=800.))
    s = ld.settings(on)
    assert s['view'] == ld.MapView('wz3to2-stub', 0., (0, 1, 2), 2)
    assert s['view'].adjacent_lane == 1 and s['view'].through_lanes == (0, 1)
    assert s['v2v'] == dict(range_m=300., member_cap=3) and s['rsu']['range_m'] == 800.
    assert ld.settings(ld_cfg(cfg))['v2v'] == dict(range_m=ld.DEFAULT_V2V_RANGE_M, member_cap=3)
    raw = on['lane_drop']
    bad = [dict(raw, rule='x'), dict(raw, provenance=' '), dict(raw, enabled=False),
           dict(raw, v2v=dict(range_m=99.)), dict(raw, v2v=dict(range_m=301.)),
           dict(raw, v2v=dict(member_cap=1)), dict(raw, v2v=dict(member_cap=5)),
           dict(raw, features=False, moves=False), dict(raw, moves=False),
           dict(raw, map=dict(MAP, dropped_lane=3)), dict(raw, map=dict(MAP, extra=1)),
           dict(raw, rsu=dict(range_m=50.)), dict(raw, v2v=None), dict(raw, extra=1)]
    for value in bad:
        with pytest.raises(ValueError):
            ld.settings(dict(cfg, lane_drop=value))
    with pytest.raises(ValueError):
        tb._check(dict(cfg, lane_drop=bad[0]), 'B')
    # A 0-dropped frame variant: adjacent is the nearest through lane.
    view = ld.MapView('variant', 2700., (0, 1, 2), 0)
    assert (view.adjacent_lane, view.through_lanes, view.distance_to_drop(2000.)) == (1, (1, 2), 700.)


WZ3TO2 = REPO/'assets'/'research'/'newmap_candidates'/'wz3to2'


def test_map_view_from_the_n1_wz3to2_descriptor_is_the_single_wm_frame(cfg, fixed):
    """F6: N1's MapDescriptor (map_descriptor.load) -> drop at s=75.44, closing lane = WM lane 2."""
    from . import map_descriptor as md
    desc = md.load(WZ3TO2)
    view = ld.MapView.from_descriptor(desc)
    assert view == ld.MapView('wz3to2', 75.44, (0, 1, 2), 2)
    assert (view.adjacent_lane, view.through_lanes) == (desc.bottleneck['adjacent_lane'], (0, 1))
    assert view.lane_drop_s_m == desc.frame.merge_zone_s_m[1]
    assert ld.from_map_spec(WZ3TO2) == view.to_dict() == ld.from_map_spec(WZ3TO2/'MAP_SPEC.json')
    # The SUMO x=2696 m end of up_0 projects onto the same drop point.
    lane, s = desc.lane_and_s('up_0', desc.bottleneck['drop_x_m']-desc.lane_table[('up', 0)][1])
    assert (lane, s) == (2, pytest.approx(75.44)) and view.distance_to_drop(s) == pytest.approx(0.)
    with pytest.raises(ValueError):
        ld.MapView.from_descriptor(md.ORIGINAL)                 # merge map: no lane-drop bottleneck
    bad = md.MapDescriptor(**{**{k: getattr(desc, k) for k in desc.__dataclass_fields__},
                              'bottleneck': dict(desc.bottleneck, adjacent_lane=0)})
    with pytest.raises(ValueError):
        ld.MapView.from_descriptor(bad)
    # Registered config from the N1 descriptor; candidates in that frame are CAV-only, role-correct.
    on = dict(cfg, trigger_partner_slots=4, **ld.config_keys(provenance=PROV, descriptor=desc, v2v=True))
    assert ld.settings(on)['view'] == view
    rows = [row('a', 75.44-1200, 2), row('b', 75.44-1195, 1), row('c', 75.44-1210, 0),
            row('h1', 75.44-1150, 1, cav=False)]
    groups = ld.joint_groups('closing_pair', view, rows[0], rows[1:3])
    assert [tuple(r['vid'] for r in g) for g in groups] == [('a', 'b')]
    p = make(on)
    p.trigger.trigger_head = Prefer(p.trigger, {'ld_kind_closing_pair': 50., 'ld_is_initiate': -60.})
    s = tick(p, rows, 300)
    assert s['track2_trigger']['accepted'] == dict(a=True, b=False, c=False)
    assert [e['move'].family for e in p.executor.active] == ['closing_pair']
    tick(p, advance(rows, 1), 301)
    s = tick(p, advance(rows, 2), 302)                          # registered onset 2 s
    assert set(s['track2']['lane_drop']['law_commands']) == {'a', 'b'}
    assert 'h1' not in s['context']['move_offsets']


# ------------------------------------------------------------------ OFF = byte-identical

def test_off_keys_absent_or_false_are_identical(cfg):
    outs = []
    for value in (cfg, dict(cfg, lane_drop=False)):
        p = make(value)
        samples = [tick(p, advance(pair_scene(), t-300), t, deterministic=False, seed=t) for t in range(300, 304)]
        outs.append(canon(samples)+canon({k: v.tolist() for k, v in p.worker_state()['layer'].items()})
                    + canon({k: v.tolist() for k, v in p.worker_state()['trigger'].items()}))
        assert p.lane_drop is None and p.ld_protocol is None and p.rsu is None
        assert 'lane_drop' not in p.local_states['a'] and 'lane_drop' not in samples[0]['track2']
        assert 'rsu' not in p.worker_state()
    assert outs[0] == outs[1]


SCENARIO = r'''
import hashlib, json, sys, torch
sys.path.insert(0, sys.argv[1])
import tools.coopmwm as pkg
if sys.argv[2] != '-':
    pkg.__path__.insert(0, sys.argv[2])
from tools.coopmwm import exploration, track2_backend as tb
from tools.coopmwm.acceleration import Controller
from tools.coopmwm.v2v_intent import IntentPolicy
from tools.coopmwm.v2v_runtime import controller_class
exploration.regime = lambda rows: 'ordinary'
torch.set_num_threads(1)
cfg = json.loads(sys.argv[3])
torch.manual_seed(7)
p = tb.EpisodePolicy(IntentPolicy(), cfg, 'B')
controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), False)
def rows(dt):
    out = [('a', -400., 0, True), ('b', -330., 1, True), ('c', -250., 0, True), ('d', -1200., 2, True),
           ('h1', -300., 0, False)]
    return [dict(vid=v, s_m=s+2.*dt, v_mps=2., lane=l, is_cav=c, length_m=5.) for v, s, l, c in out]
for now in range(300, 306):
    r = rows(now-300)
    p.observe(r, now)
    controller.apply(r, {x['vid']: 0. for x in r if x['is_cav']}, now)
digest = hashlib.sha256()
digest.update(json.dumps([e['sample'] for e in controller.events], sort_keys=True, default=repr).encode())
digest.update(json.dumps(controller.g5_lane_targets, sort_keys=True).encode())
for part in ('layer', 'trigger'):
    for k, v in sorted(p.worker_state()[part].items()):
        digest.update(k.encode()); digest.update(v.detach().numpy().tobytes())
print(digest.hexdigest())
'''


def test_off_is_byte_identical_to_the_pre_n2_commit(cfg, tmp_path):
    """Default-off samples, lane targets and initial weights equal the pre-N2 sources."""
    touched = ('track2_backend.py', 'feasible_proposals.py', 'group_lane.py', 'moves.py', 'bank_breadth.py',
               'v2v_wall.py', 'sparse_synergy_credit.py', 'launch_production.py', 'trigger_policy.py',
               'launch_evaluation.py', 'acceleration.py')
    overlay = tmp_path/'overlay'
    overlay.mkdir()
    for name in touched:
        try:
            text = subprocess.run(['git', '-C', str(REPO), 'show', f'{PRE_N2_COMMIT}:tools/coopmwm/{name}'],
                                  capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip('pre-N2 source unavailable in this checkout')
        (overlay/name).write_bytes(text)
    value = json.dumps(dict(cfg, run_root=str(tmp_path)))
    digests = []
    for path in ('-', str(overlay)):
        run = subprocess.run([sys.executable, '-c', SCENARIO, str(REPO), path, value], cwd=tmp_path,
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-2000:]
        digests.append(run.stdout.strip())
    assert len(digests[0]) == 64 and digests[0] == digests[1]


# ------------------------------------------------------------------ observation block

def report(vid, s, lane, now, v=20., seen=(), lanes=(0, 1, 2)):
    return SensorReport(float(now), ObservedVehicle(vid, float(s), lane, v, 5.), tuple(lanes),
                        tuple(ObservedVehicle(i, float(x), ln, sp, 5.) for i, x, ln, sp in seen))


def test_lane_drop_block_values_coverage_ages_and_missing_bits():
    view = ld.MapView.from_descriptor(StubMap())
    f = {n: i for i, n in enumerate(ld.LANE_DROP_FEATURES)}
    own = report('a', -450, 2, 301, seen=[('q1', -400, 2, 1.), ('h2', -430, 1, 25.), ('h3', -470, 1, 22.)])
    partner = report('p', -360, 1, 300, seen=[('q2', -300, 2, 0.5), ('q1', -401, 2, 9.)])
    out = ld.lane_drop_features(own, 'a', {'p': partner}, now_s=301., partner_ids=['p'], view=view)
    assert out[f['own_missing']] == 0 and out[f['dist_to_drop_m']] == 450. and out[f['in_dropped_lane']] == 1
    assert out[f['lane_offset_from_dropped']] == 0. and out[f['in_adjacent_lane']] == 0
    # Closing lane in ROI [-500, 0]: a, q1 (own row wins: 1 m/s), q2 (partner, 0.5 m/s), p is lane 1.
    assert out[f['closing_count_veh']] == 3 and out[f['closing_queue_veh']] == 2
    assert out[f['closing_queue_tail_m']] == 400. and out[f['closing_report_age_s']] == 1.
    assert out[f['closing_coverage_frac']] == pytest.approx(240./500.)   # [-500,-350] U [-460,-260]
    # Adjacent lane 1 gaps at the ego position (own packet only): h2 ahead, h3 behind.
    assert out[f['adjacent_front_gap_m']] == 15. and out[f['adjacent_front_dv_mps']] == 5.
    assert out[f['adjacent_rear_gap_m']] == 15. and out[f['adjacent_rear_dv_mps']] == 2.
    assert out[f['adjacent_count_veh']] == 2 and out[f['adjacent_missing']] == 0
    assert out[f['partner_report_count']] == 1 and out[f['partner_reports_missing']] == 0
    missing = ld.lane_drop_features(None, 'a', {}, now_s=301., view=view)
    assert missing.tolist() == ld.missing_block().tolist()
    lone = ld.lane_drop_features(report('a', -1200, 1, 301), 'a', {}, now_s=301., view=view)
    assert lone[f['closing_missing']] == 1 and lone[f['adjacent_front_none']] == 1
    assert lone[f['adjacent_rear_gap_m']] == 100. and lone[f['partner_reports_missing']] == 1


def test_lane_drop_block_is_observable_only():
    view = ld.MapView.from_descriptor(StubMap())
    own = report('a', -450, 2, 301, seen=[('q1', -400, 2, 1.)])
    partner = report('p', -360, 1, 300, seen=[('q2', -300, 2, .5)])
    base = ld.lane_drop_features(own, 'a', {'p': partner}, now_s=301., partner_ids=['p'], view=view)
    # Same-step, stale, unselected or out-of-range partner packets are unavailable.
    for reports, ids in (({'p': report('p', -360, 1, 301, seen=[('q9', -50, 2, 0.)])}, ['p']),
                         ({'p': report('p', -360, 1, 299, seen=[('q9', -50, 2, 0.)])}, ['p']),
                         ({'p': partner, 'u': report('u', -420, 1, 300, seen=[('q9', -50, 2, 0.)])}, ['p']),
                         ({'p': partner, 'r': report('r', -200, 1, 300, seen=[('q9', -50, 2, 0.)])},
                          ['p', 'r'])):
        got = ld.lane_drop_features(own, 'a', reports, now_s=301., partner_ids=ids, view=view)
        if 'u' in reports or 'r' in reports:
            assert got.tolist() == base.tolist()
        else:
            assert got.tolist() != base.tolist() and got[1] == base[1]   # partner part missing only


def test_episode_blocks_ignore_unsensed_vehicles_and_hidden_fields(cfg):
    on = ld_cfg(cfg, v2v=False)
    p, q = make(on), make(on)
    scene = pair_scene()
    noisy = [dict(r, tau=9., driver_type='aggressive', vtype='truth') for r in scene]
    noisy += [row('zz', 900., 2, cav=False, v=0.), row('zc', 2000., 1)]   # far from every CAV
    for t in (300, 301, 302):
        p.observe(advance(scene, t-300), t)
        q.observe(advance(noisy, t-300), t)     # observe() whitelists rows; encode() refuses extras
        for v in ('a', 'b', 'c'):
            assert p.lane_drop_blocks[v] == q.lane_drop_blocks[v]
            assert len(p.lane_drop_blocks[v]) == ld.LANE_DROP_DIM
        assert p.local_states['a']['lane_drop']['view'] == q.local_states['a']['lane_drop']['view']


def test_features_only_are_zero_init_additive_and_mappo_untouched(cfg):
    off, on = make(dict(cfg, trigger_partner_slots=4)), make(ld_cfg(cfg, moves_on=False, v2v=False))
    w_off, w_on = off.layer.residual[0].weight, on.layer.residual[0].weight
    assert w_on.shape[1] == w_off.shape[1]+ld.LANE_DROP_DIM
    assert torch.equal(w_on[:, :w_off.shape[1]-1], w_off[:, :-1]) and torch.equal(w_on[:, -1], w_off[:, -1])
    assert not bool(w_on[:, w_off.shape[1]-1:-1].any())
    t_off, t_on = off.trigger.trigger_head[0].weight, on.trigger.trigger_head[0].weight
    assert torch.equal(t_on[:, :t_off.shape[1]], t_off) and not bool(t_on[:, t_off.shape[1]:].any())
    assert off.trigger.families == on.trigger.families
    base_off = {k: v.clone() for k, v in off.layer.base.state_dict().items()}
    for t in range(300, 304):
        a = tick(off, advance(pair_scene(), t-300), t, deterministic=False, seed=t)
        b = tick(on, advance(pair_scene(), t-300), t, deterministic=False, seed=t)
        for key in ('members', 'x', 'v2v', 'rsu', 'anchors'):
            assert a['context'][key] == b['context'][key]           # frozen MAPPO observation
        assert a['action'] == b['action'] and a['logp'] == b['logp']
        for key in ('old_residual', 'old_logits', 'choices', 'offsets'):
            assert a['coop_layer'][key] == b['coop_layer'][key]
        if 'track2_trigger' in a:
            for key in ('tables', 'actions', 'logp', 'values', 'mask'):
                assert a['track2_trigger'][key] == b['track2_trigger'][key]
    assert all(torch.equal(v, on.layer.base.state_dict()[k]) for k, v in base_off.items())
    assert all(not p.requires_grad for p in on.layer.base.parameters())


def test_moves_on_keep_every_pre_existing_trigger_weight(cfg):
    """Joint/protocol lane-drop options are extension families: zero-init descriptor columns,
    unchanged one-hot, so pre-existing candidates keep their logits at initialisation."""
    off = make(dict(cfg, trigger_partner_slots=4))
    on = make(ld_cfg(cfg, rsu=dict(range_m=1000.)))
    assert on.trigger.families == off.trigger.families
    assert on.trigger.option_families == off.trigger.families+ld.JOINT_FAMILIES+ld.PROTOCOL_FAMILIES
    t_off, t_on = off.trigger.trigger_head[0].weight, on.trigger.trigger_head[0].weight
    assert torch.equal(t_on[:, :t_off.shape[1]], t_off) and not bool(t_on[:, t_off.shape[1]:].any())
    for a, b in zip(off.trigger.parameters(), on.trigger.parameters()):
        if a.shape == b.shape:
            assert torch.equal(a, b)
    assert all(torch.equal(v, on.layer.base.state_dict()[k]) for k, v in off.layer.base.state_dict().items())


# ------------------------------------------------------------------ moves / execution

def test_joint_candidates_roles_windows_and_hdvs(cfg, fixed):
    view = ld.MapView.from_descriptor(StubMap())
    rows = {r['vid']: r for r in pair_scene()}
    cavs = [rows[v] for v in ('b', 'c')]
    pairs = ld.joint_groups('closing_pair', view, rows['a'], cavs)
    assert [tuple(r['vid'] for r in g) for g in pairs] == [('a', 'b')]
    assert [tuple(r['vid'] for r in g) for g in ld.joint_groups('closing_pair', view, rows['b'],
                                                               [rows['a'], rows['c']])] == [('a', 'b')]
    assert ld.joint_groups('closing_pair', view, rows['c'], [rows['a'], rows['b']]) == []
    walls = ld.joint_groups('abreast_wall', view, rows['a'], cavs)
    assert [tuple(r['vid'] for r in g) for g in walls] == [('c', 'b', 'a')]   # lane order 0, 1, 2
    move = ld.pool('abreast_wall', dict(view=view), rows['a'], cavs, np.random.default_rng(0))[0]
    assert (move.members, move.lanes, move.target_mps, move.zip, move.gap_dv_mps) == (('c', 'b', 'a'),
                                                                                    (0, 1, 2), 18., False, 0.)
    near_drop = dict(rows['a'], s_m=-200.)                      # outside the abreast window
    assert ld.joint_groups('abreast_wall', view, near_drop, cavs) == []
    # HDVs are never members (pools see only acquirable CAV rows); the span is registered.
    far = dict(rows['b'], s_m=-1100.)
    assert ld.joint_groups('closing_pair', view, rows['a'], [far]) == []
    with pytest.raises(ValueError):
        ld.LaneDropMove('abreast_wall', ('a', 'b'), 0, 30, .5, 18., 1., False, (1, 2))
    with pytest.raises(ValueError):
        ld.LaneDropMove('closing_pair', ('a', 'b'), 0, 30, .5, 40., 0., False, (2, 1))


def plan_inputs(move, seconds):
    rng = np.random.default_rng(3)
    speeds = 20.+rng.normal(0., 1.5, (len(move.members), seconds))
    positions = -1200.+np.cumsum(speeds, 1)
    base = rng.normal(0., .2, (len(move.members), seconds))
    return base, speeds, positions


def test_executable_plan_binding_and_refusals():
    view = ld.MapView.from_descriptor(StubMap())
    move = ld.LaneDropMove('closing_pair', ('a', 'b'), 3, 25, .8, 17., 2., True, (2, 1))
    base, speeds, positions = plan_inputs(move, 28)
    plan = ld.plan(move, view, policy_profile=base.tolist(), member_speeds=speeds.tolist(),
                   member_positions=positions.tolist())
    assert moves.from_plan(plan) == move == ld.from_plan(plan)
    profile = np.array(plan['profile'])
    assert np.allclose(profile[:, :3], base[:, :3])                       # before onset: the policy
    for t in range(3, 28):
        assert profile[0, t] == pytest.approx(ld.law(17., speeds[0, t], .8))
        assert profile[1, t] == pytest.approx(ld.law(15., speeds[1, t], .8))   # partner: gap_dv lower
    assert plan['lane_drop_lane_actions'][0][27] == 1 and sum(map(sum, plan['lane_drop_lane_actions'])) == 1
    for key, value in (('profile', (profile+1e-3).tolist()), ('perturbation_profile', base.tolist()),
                       ('member_speeds_mps', (speeds+1.).tolist()), ('duration_s', 27),
                       ('lane_drop_lane_actions', [[0]*28, [0]*28])):
        with pytest.raises(ValueError):
            ld.from_plan(dict(plan, **{key: value}))
    for extra in ('knots', 'g5_move', 'wall_macro', 'lane_actions'):
        with pytest.raises(ValueError):
            ld.from_plan(dict(plan, **{extra: 1}))
    bad_roles = dict(plan, lane_drop_macro=dict(plan['lane_drop_macro'], lanes=[1, 2]))
    with pytest.raises(ValueError):
        ld.from_plan(bad_roles)
    # Release: a member within the registered distance of the drop stops the law for all.
    close = positions.copy()
    close[1, 10:] = -20.
    released = ld.plan(move, view, policy_profile=base.tolist(), member_speeds=speeds.tolist(),
                       member_positions=close.tolist())
    assert np.allclose(np.array(released['profile'])[:, 10:], base[:, 10:])


def test_runtime_execution_matches_the_verified_profile(cfg, fixed):
    """Single-writer path: native + MAPPO action == the law == the plan's profile."""
    p = make(ld_cfg(cfg, v2v=False))
    with torch.no_grad():
        p.layer.base.actor[-1].bias.fill_(.25)                   # nonzero frozen MAPPO mean
    p.trigger.trigger_head = Prefer(p.trigger, {'ld_kind_closing_pair': 50.})
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), True)
    natives = {'a': .4, 'b': -.2, 'c': .1}
    recorded, requests = {'a': [], 'b': []}, {'a': [], 'b': []}
    scene = pair_scene()
    for now in range(300, 344):
        speeds = {'a': 18.4-.02*(now-300), 'b': 16.5-.05*(now-300), 'c': 20.}
        rows = [dict(r, v_mps=speeds.get(r['vid'], 20.)) for r in advance(scene, now-300)]
        p.observe(rows, now, native_commands=natives)
        controller.apply(rows, dict(natives), now)
        if now == 300:
            active = [e for e in p.executor.active]
            assert [e['move'].family for e in active] == ['closing_pair']
            move = active[0]['move'].move
            assert move.members == ('a', 'b') and (move.onset, move.hold) == (2, 40)
        for v in ('a', 'b'):
            recorded[v].append((speeds[v], next(r['s_m'] for r in rows if r['vid'] == v)))
            raw = controller.events[-1]['requests'].get(v)
            # No request entry = a zero MAPPO delta: the CAV executes its native command.
            requests[v].append(natives[v] if raw is None else raw['requested_raw'])
        assert controller.events[-1]['requests'].get('c', {}).get('requested_raw', .1+.25) == \
            pytest.approx(.1+.25)                               # non-member: native + MAPPO mean
    base = [[natives[v]+.25]*42 for v in ('a', 'b')]
    plan = ld.plan(move, StubMap(), policy_profile=base,
                   member_speeds=[[s for s, _ in recorded[v][:42]] for v in ('a', 'b')],
                   member_positions=[[x for _, x in recorded[v][:42]] for v in ('a', 'b')])
    ld.from_plan(plan)
    for i, v in enumerate(('a', 'b')):
        assert requests[v][:42] == pytest.approx(plan['profile'][i], abs=1e-5)
        assert requests[v][42] == pytest.approx(natives[v]+.25, abs=1e-6)   # released after the hold


def zip_scene():
    """b holds 40 m behind a in the adjacent lane: the zip gap ahead of b is open."""
    return [row('a', -1200, 2), row('b', -1240, 1), row('c', -1210, 0), row('h1', -1150, 1, cav=False)]


def test_zip_lane_change_uses_the_g5_env_path_for_cavs_only(cfg, fixed):
    fixed.update(onset=0, hold=20, zip=True)
    p = make(ld_cfg(cfg, v2v=False))
    p.trigger.trigger_head = Prefer(p.trigger, {'ld_kind_closing_pair': 50.})
    controller = controller_class(Controller)(p, torch.Generator().manual_seed(9), True)
    natives = {'a': 0., 'b': 0., 'c': 0.}

    class Env:
        def slot_vehicle_ids(self):
            return ['a', 'b', 'c', 'h1']
    for now in range(300, 321):
        rows = advance(zip_scene(), now-300)
        if now == 319:
            blocked = rows+[row('h2', rows[0]['s_m']+2., 1, cav=False)]   # occupies the zip gap
        p.observe(rows, now, native_commands=natives)
        controller.apply(rows, dict(natives), now)
        sample = controller.events[-1]['sample']
        if now < 319:
            assert 'lane_drop_lane_requests' not in sample and controller.g5_lane_targets == {}
    assert 'lane_drop_lane_requests' not in sample   # 320: expired; the single attempt has passed
    # Replay the zip second: safe gap -> the closing-lane CAV only; occupied gap -> blocked.
    for scene, expected in ((advance(zip_scene(), 19), {'a': 1}), (blocked, {})):
        q = make(ld_cfg(cfg, v2v=False))
        q.trigger.trigger_head = Prefer(q.trigger, {'ld_kind_closing_pair': 50.})
        c2 = controller_class(Controller)(q, torch.Generator().manual_seed(9), True)
        for now in range(300, 320):
            rows = advance(zip_scene(), now-300) if now < 319 else scene
            q.observe(rows, now, native_commands=natives)
            c2.apply(rows, dict(natives), now)
        sample = c2.events[-1]['sample']
        assert sample['lane_drop_lane_requests'] == {'a': 1}
        assert c2.g5_lane_targets == expected
        assert c2.lane_actors() == frozenset(expected)
        if expected:
            tensor = c2.lane_request(Env())
            assert tensor.tolist() == [[1., -1., -1., -1.]]          # HDV slot stays native
        else:
            assert sample['lane_drop_lane_gate']['blocked'] == {'a': 'unsafe_gap_or_edge'}
    with pytest.raises(ValueError):    # an abreast wall never changes lane
        ld.LaneDropMove('abreast_wall', ('c', 'b', 'a'), 0, 20, .5, 18., 0., True, (0, 1, 2))


# ------------------------------------------------------------------ V2V initiate/join

def proto_scene():
    """a initiates (closing lane); b (lane 1) 150 m ahead: outside sensing, inside 200 m radio;
    d (lane 1) 260 m away: outside radio; c (lane 0) is not a closing_pair role; h1 an HDV."""
    return [row('a', -1300, 2), row('b', -1150, 1), row('c', -1290, 0), row('d', -1040, 1),
            row('h1', -1160, 1, cav=False)]


# Closing-lane CAVs initiate closing pairs; everyone joins; nobody else initiates.
INITIATE_CP = {'ld_role_dropped_lane': 40., 'ld_kind_closing_pair': 5., 'ld_is_initiate': -5., 'ld_is_join': 50.}


def test_message_visible_only_at_t_plus_one_within_range_and_join_executes(cfg, fixed):
    p = make(ld_cfg(cfg, range_m=200.))
    p.trigger.trigger_head = Prefer(p.trigger, INITIATE_CP)
    s = tick(p, proto_scene(), 300)
    assert s['track2_trigger']['accepted']['a'] is True
    assert 'lane_drop_join' not in families(s, 'b')
    published = s['track2']['lane_drop']['protocol']['published']
    assert [(m['wall_id'], m['kind'], m['covered'], m['member_count']) for m in published] == [
        ('a@300', 'closing_pair', [2], 1)]
    assert p.ld_protocol.offers == {}
    s = tick(p, advance(proto_scene(), 1), 301)
    stats = s['track2_trigger']['proposals']['lane_drop']
    assert stats['event'] and stats['event_cavs'] == ['b']
    assert set(p.ld_protocol.offers) == {'b'}                  # c: no role lane; d: out of radio; h1: HDV
    assert set(families(s, 'b')) == {'lane_drop_join'}
    assert s['track2_trigger']['accepted'] == dict(a=False, b=True, c=False, d=False)
    offer = p.ld_protocol.offers['b'][0]
    assert (offer.onset, offer.hold, offer.role_lane, offer.covered) == (1, 40, 1, (2,))
    assert offer.own_target(p.lane_drop['view']) == 16.         # partner role: target - gap_dv
    s = tick(p, advance(proto_scene(), 2), 302)
    law = s['track2']['lane_drop']['law_commands']
    assert law == {'a': pytest.approx(ld.law(18., 20., .6)), 'b': pytest.approx(ld.law(16., 20., .6))}
    assert s['context']['move_offsets']['a'] == pytest.approx(law['a']-.3)
    assert s['context']['move_offsets']['c'] == 0. and 'h1' not in s['context']['move_offsets']
    assert s['track2']['lane_drop']['protocol']['published'][0]['covered'] == [1, 2]
    assert {tuple(m['move']['members']): m['until'] for m in s['track2']['active_moves']} == {
        ('a',): 342, ('b',): 342}


def test_range_is_configurable_and_default_100_would_not_reach(cfg, fixed):
    for range_m, offered in ((100., set()), (300., {'b', 'd'})):
        p = make(ld_cfg(cfg, range_m=range_m))
        p.trigger.trigger_head = Prefer(p.trigger, INITIATE_CP)
        tick(p, proto_scene(), 300)
        p.observe(advance(proto_scene(), 1), 301)
        assert set(p.ld_protocol.offers) == offered


def test_ownership_member_cap_and_one_cav_per_lane(cfg, fixed):
    p = make(ld_cfg(cfg, range_m=300., cap=2))
    p.trigger.trigger_head = Prefer(p.trigger, INITIATE_CP)
    extra = [row('e', -1170, 1), row('f', -1180, 1)]
    busy = tb.TriggerMove(moves.Move('ease_off', ('f',), 0, 30, .5))
    p.observe(proto_scene()+extra, 300)
    assert p.executor.start(busy, ego='f', observable_cav_ids=p.visible['f'], now=300, until=330)
    p.last_decision = None
    s = tick(p, proto_scene()+extra, 300)
    assert s['track2_trigger']['accepted']['a'] is True and families(s, 'f') == []
    s = tick(p, advance(proto_scene()+extra, 1), 301)
    stats = s['track2_trigger']['proposals']['lane_drop']
    assert stats['event_cavs'] == ['b', 'd', 'e']              # f is owned: no event
    assert stats['selected_join'] == 3 and stats['accepted_join'] == 1   # cap 2 -> one joiner only
    assert sum(s['track2_trigger']['accepted'].values()) == 1
    session = p.ld_protocol.sessions['a@300']
    assert len(session['members']) == 2 and sorted(session['lanes'].values()) == [1, 2]
    joined = [v for v in session['members'] if v != 'a'][0]
    # A member is exclusively owned: no new offer, no second initiation; covered lanes get none.
    s = tick(p, advance(proto_scene()+extra, 2), 302)
    assert joined not in p.ld_protocol.offers and 'a' not in p.ld_protocol.offers
    late = ld.LaneDropOption('lane_drop_join', ('e',), 'closing_pair', 0, 39, .6, 18., 2., False, 1,
                             'a@300', 'a', member_count=1, cap=2, range_m=300.)
    assert joined == 'b' and not p.ld_protocol.start(tb.TriggerMove(late), ego='e', executor=p.executor,
                                                     observable_cav_ids=p.visible['e'], now=302)
    again = ld.LaneDropOption('lane_drop_initiate', (joined,), 'closing_pair', 0, 30, .6, 18., 2., False, 1,
                              f'{joined}@302', joined, cap=2, range_m=300.)
    assert not p.ld_protocol.start(tb.TriggerMove(again), ego=joined, executor=p.executor,
                                   observable_cav_ids=p.visible[joined], now=302)
    assert {v for e in p.executor.active for v in e['move'].members} == {'a', joined, 'f'}


def test_release_when_the_initiator_nears_the_drop(cfg, fixed):
    p = make(ld_cfg(cfg, range_m=300.))
    p.trigger.trigger_head = Prefer(p.trigger, INITIATE_CP)

    def two(a_s, b_s):
        return [row('a', a_s, 2), row('b', b_s, 1), row('h1', b_s-30, 1, cav=False)]
    tick(p, two(-1300, -1150), 300)
    s = tick(p, two(-1280, -1130), 301)
    assert s['track2_trigger']['accepted']['b'] is True
    s = tick(p, two(-1260, -1110), 302)
    assert set(s['track2']['lane_drop']['law_commands']) == {'a', 'b'}
    s = tick(p, two(-25, -260), 303)                           # initiator within 30 m of the drop
    assert 'a' not in p.execution_positions['b']               # 235 m: not sensed by b
    assert set(s['track2']['lane_drop']['law_commands']) == {'b'}
    assert s['track2']['lane_drop']['protocol']['published'][0]['released'] is True
    s = tick(p, two(-5, -240), 304)                            # b receives the t-1 release message
    assert s['track2']['lane_drop']['law_commands'] == {}
    assert ('b', 'a@300') in p.ld_protocol.release_latch
    assert {e['until'] for e in p.executor.active} == {342}     # ownership held to expiry


# ------------------------------------------------------------------ B = R

def test_b_and_r_run_identically(cfg, fixed):
    out = {}
    for arm in ('B', 'R'):
        p = make(ld_cfg(cfg, range_m=300., rsu=dict(range_m=1000.)), arm)
        p.trigger.trigger_head = Prefer(p.trigger, INITIATE_CP)
        samples = [tick(p, advance(proto_scene(), t-300), t, deterministic=False, seed=t) for t in range(300, 305)]
        for s in samples:
            assert s['track2'].pop('arm') == arm
        out[arm] = canon(samples)+canon(p.ld_protocol.membership)+canon(
            {k: v.tolist() for k, v in p.worker_state()['rsu'].items()})
    assert out['B'] == out['R']
    assert '"lane_drop_join"' in out['B'] and '"lane_drop_initiate"' in out['B']


def test_recorded_decisions_replay_for_trigger_ppo_and_g4_skips(cfg, fixed):
    p = make(ld_cfg(cfg, range_m=200.))
    p.trigger.trigger_head = Prefer(p.trigger, INITIATE_CP)
    tick(p, proto_scene(), 300)
    s = tick(p, advance(proto_scene(), 1), 301, deterministic=False, seed=5)
    record = json.loads(json.dumps(s['track2_trigger']))
    batch = tb._batch(record)
    assert any(isinstance(m, ld.LaneDropOption) for table in batch.tables for m in table)
    logp, _, _ = p.trigger.evaluate(batch)
    assert logp.tolist() == pytest.approx(record['logp'])
    from . import sparse_synergy_credit as g4
    assert g4.opportunities([dict(id='ep', steps=[dict(time_s=301, sample=s)])]) == []


# ------------------------------------------------------------------ bank ingestion

def pair_frames(t0=300, seconds=4):
    frames = {}
    for t in range(t0-1, t0+seconds):
        frames[t] = dict(vehicles=advance([row('m1', -1300, 2), row('m2', -1290, 1),
                                           row('h1', -1250, 1, cav=False)], t-t0))
    return frames


def pair_bank(frames, t0=300):
    move = ld.LaneDropMove('closing_pair', ('m1', 'm2'), 0, 30, .6, 18., 2., False, (2, 1))
    speeds = np.full((2, 30), 20.)
    positions = np.array([[-1300.+20*t for t in range(30)], [-1290.+20*t for t in range(30)]])
    plan = ld.plan(move, StubMap(), policy_profile=np.zeros((2, 30)).tolist(), member_speeds=speeds.tolist(),
                   member_positions=positions.tolist())
    pairs = []
    for t in sorted(frames):
        if t < t0:
            continue
        context = encode(frames[t]['vehicles'])
        context['v2v'] = {v: [0.]*48 for v in context['members']}
        pairs.append((dict(context=context, intent_link=dict(t=t)), 1.))
    return move, plan, pairs


def test_bank_contexts_generic_and_through_the_protocol(cfg):
    frames = pair_frames()
    move, plan, pairs = pair_bank(frames)
    generic = make(ld_cfg(cfg, v2v=False))
    got = list(breadth.observable_contexts(generic, plan, pairs, frames, t0=300))
    assert [(o.family, c['members']) for c, o, *_ in got] == [('closing_pair', ['m1', 'm2'])]
    assert set(got[0][0]['lane_drop']) == {'m1', 'm2'}
    p = make(ld_cfg(cfg, range_m=200.))
    got = list(breadth.observable_contexts(p, plan, pairs, frames, t0=300))
    labels = [(o.family, o.members[0], o.role_lane, t) for _, o, _, _, t in got]
    assert labels == [('lane_drop_initiate', 'm1', 2, 300), ('lane_drop_join', 'm2', 1, 301)]
    for context, option, _, _, t in got:
        ego = option.members[0]
        start = {'m1': -1300., 'm2': -1290.}[ego]+20.*(t-300)
        assert context['members'] == [ego] and set(context['lane_drop']) == {ego}
        assert context['lane_drop'][ego][1] == -start           # own causal block at its own time


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_bank_supervision_identical_for_b_and_r(cfg, monkeypatch, arm):
    from . import bank_v2v
    frames = pair_frames()
    _, plan, pairs = pair_bank(frames)
    on = ld_cfg(cfg, range_m=200.)

    def visit(config, requested_arm, consume):
        consume(cfg=config, receipt=dict(verification_id='ld_pair_s989200', t0=300),
                example=dict(example_id='ld_pair_s989200#s1'), plan=plan, pairs=deepcopy(pairs), trace='CPU')
    monkeypatch.setattr(breadth, 'visit', visit)
    monkeypatch.setattr(bank_v2v, 'trace_frames', lambda path: frames)
    results = {}
    for value in ('B', 'R'):
        policy = make(on, value)
        results[value] = launch.bank_move_supervision(SimpleNamespace(cfg=on, policy=policy, arm=value))
    assert results[arm]['family_counts'] == {'lane_drop_initiate': 1, 'lane_drop_join': 1}
    assert results[arm]['final_ce'] < results[arm]['initial_ce']

    def strip(r):
        return canon({k: v for k, v in r.items() if k != 'passes'})
    assert strip(results['B']) == strip(results['R'])


# ------------------------------------------------------------------ RSU-initiated moves

def rsu_scene():
    """Reporting CAVs a (lane 2) and b (lane 1) abreast 900 m before the drop; c upstream
    outside the 1000 m RSU range; h1 an HDV."""
    return [row('a', -900, 2), row('b', -895, 1), row('c', -1600, 1), row('h1', -860, 1, cav=False)]


def test_rsu_field_is_built_from_t_minus_1_cav_uplink_only():
    view = ld.MapView.from_descriptor(StubMap())
    s = rsu_mod.rsu_settings(dict(range_m=500.), view)
    f = {n: i for i, n in enumerate(rsu_mod.RSU_FIELD)}
    reports = {'a': report('a', -450, 2, 300, seen=[('q1', -400, 2, 1.)]),
               'b': report('b', -440, 1, 300),
               'c': report('c', -1600, 1, 300, seen=[('q9', -1550, 2, 0.)]),     # outside RSU range
               'n': report('n', -300, 1, 301, seen=[('q8', -250, 2, 0.)])}       # same-step: unavailable
    field = rsu_mod.rsu_field(reports, 301., view, s)
    assert field[f['report_count']] == 2 and field[f['report_age_s']] == 1. and field[f['reports_missing']] == 0
    assert field[f['lane_2_count_veh']] == 2 and field[f['lane_2_queue_veh']] == 1
    assert field[f['closing_queue_veh']] == 1 and field[f['closing_queue_tail_m']] == 400.
    assert field[f['lane_1_cav_reports']] == 1 and field[f['lane_2_cav_reports']] == 1
    assert field[f['lane_0_lane_present']] == 1 and field[f['lane_0_speed_missing']] == 1
    assert rsu_mod.rsu_field({}, 301., view, s)[f['reports_missing']] == 1
    moved = dict(reports, c=report('c', -1600, 1, 300, seen=[('q9', -1500, 2, 30.)]))
    assert rsu_mod.rsu_field(moved, 301., view, s).tolist() == field.tolist()


class PickFirst(nn.Module):
    def forward(self, x):
        return (100.*x[..., rsu_mod.RSU_FIELD_DIM+0])[..., None]   # prefer closing_pair entries


def test_rsu_suggestion_broadcast_join_and_single_writer_execution(cfg, fixed):
    p = make(ld_cfg(cfg, range_m=200., rsu=dict(range_m=1000.)))
    p.trigger.trigger_head = Prefer(p.trigger, {'ld_is_join': 50.})
    p.rsu.head.score = PickFirst()
    tick(p, rsu_scene(), 300)                                   # uplink needs t-1 packets
    s = tick(p, advance(rsu_scene(), 1), 301)
    decision = s['track2']['lane_drop']['rsu']['decision']
    assert decision is None                                    # not a periodic tick
    p.last_decision = None
    s = tick(p, advance(rsu_scene(), 2), 302)
    decision = s['track2']['lane_drop']['rsu']['decision']
    assert decision['uplink'] == ['a', 'b'] and decision['opened']
    chosen = decision['slate'][decision['action']]
    assert chosen['kind'] == 'closing_pair' and chosen['anchor_s_m'] == pytest.approx((-880.-875.)/2)
    assert [m['initiator'] for m in s['track2']['lane_drop']['protocol']['published']] == ['RSU']
    assert p.ld_protocol.offers == {}                           # visible only at t+1
    s = tick(p, advance(rsu_scene(), 3), 303)
    assert set(p.ld_protocol.offers) == {'a', 'b'}             # c out of RSU range, h1 an HDV
    assert s['track2_trigger']['accepted'] == dict(a=True, b=True, c=False)
    s = tick(p, advance(rsu_scene(), 4), 304)
    law = s['track2']['lane_drop']['law_commands']
    assert law == {'a': pytest.approx(ld.law(18., 20., .6)), 'b': pytest.approx(ld.law(16., 20., .6))}
    session = p.ld_protocol.sessions['RSU@302']
    assert sorted(session['lanes'].values()) == [1, 2] and session['initiator'] == 'RSU'


def test_rsu_head_zero_init_learnable_and_teachers_identical_for_b_and_r(cfg, fixed):
    p = make(ld_cfg(cfg, range_m=200., rsu=dict(range_m=1000.)))
    tick(p, rsu_scene(), 300)
    p.observe(advance(rsu_scene(), 1), 301)
    decisions = [p.rsu.decide(301)]
    p.observe(advance(rsu_scene(), 2), 302)
    decisions.append(p.rsu.decide(302, deterministic=True))
    first = decisions[0]
    available = sum(first['mask'])
    assert available >= 2 and first['logp'] == pytest.approx(-np.log(available))   # uniform
    batch = rsu_mod.batch(decisions)
    logp, values, _ = p.rsu.head.evaluate(batch)
    assert logp.tolist() == pytest.approx([d['logp'] for d in decisions], abs=1e-6)
    assert values.tolist() == [0., 0.]
    target = rsu_mod.teacher_target(first, 'closing_pair', first['slate'][1]['anchor_s_m']+10.)
    assert first['slate'][target]['kind'] == 'closing_pair'
    random_target = rsu_mod.random_teacher_target(first, np.random.default_rng(0))
    assert first['mask'][random_target] and random_target != 0
    head = p.rsu.head
    optimizer = head.optimizer(lr=.01)
    targets = [target, rsu_mod.teacher_target(decisions[1], 'closing_pair', decisions[1]['slate'][1]['anchor_s_m'])]
    assert targets[1] != 0
    initial = float(head.imitation_loss(batch, targets).detach())
    for _ in range(5):
        loss = head.imitation_loss(batch, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert float(head.imitation_loss(batch, targets).detach()) < initial
    stats = ppo_update(head, head.optimizer(lr=.001), batch, [1., -1.], [0., 0.], clip_ratio=.2,
                       value_weight=.5, entropy_weight=.01, max_grad_norm=.5)
    assert stats['cav_count'] == 2
    # Worker state carries the head only when the option is ON; B and R init identically.
    state = p.worker_state()
    r = make(ld_cfg(cfg, range_m=200., rsu=dict(range_m=1000.)), 'R')
    assert canon({k: v.tolist() for k, v in r.worker_state()['rsu'].items()}) != canon(
        {k: v.tolist() for k, v in state['rsu'].items()})          # p's head has been trained
    r.restore_worker(state)
    assert canon({k: v.tolist() for k, v in r.worker_state()['rsu'].items()}) == canon(
        {k: v.tolist() for k, v in state['rsu'].items()})
    with pytest.raises(ValueError):
        make(ld_cfg(cfg, range_m=200.)).restore_worker(state)   # same layer, no RSU option


def test_rsu_rollout_ppo_uses_uplink_returns_and_the_kl_guard(cfg, fixed):
    p = make(ld_cfg(cfg, range_m=200., rsu=dict(range_m=1000.)))
    samples, returns = {}, {}
    for now in range(300, 303):
        p.last_decision = None                                   # periodic RSU decision each tick
        s = tick(p, advance(rsu_scene(), now-300), now, deterministic=False, seed=now)
        samples[('ep', now)] = s
        returns[('ep', now)] = [{'a': 1., 'b': 3., 'c': 100.}[v] for v in s['context']['members']]
    records = rsu_mod.rollout_records(samples, returns)
    assert [r for _, r in records] == [2., 2.]                   # mean over own uplink (a, b); 300 has none
    assert all(d['uplink'] == ['a', 'b'] for d, _ in records)
    head = p.rsu.head
    before = deepcopy(head.state_dict())
    reverted = rsu_mod.train_rsu(head, p.rsu_optimizer, records, kl_budget=0.,
                                 ppo=dict(clip_ratio=.2, value_weight=.5, entropy_weight=.01, max_grad_norm=.5))
    assert reverted['accepted'] is False and reverted['decisions'] == 2
    assert all(torch.equal(v, head.state_dict()[k]) for k, v in before.items())
    kept = rsu_mod.train_rsu(head, p.rsu_optimizer, records, kl_budget=.045,
                             ppo=dict(clip_ratio=.2, value_weight=.5, entropy_weight=.01, max_grad_norm=.5))
    assert kept['accepted'] is True and not all(torch.equal(v, head.state_dict()[k]) for k, v in before.items())
    assert rsu_mod.train_rsu(head, p.rsu_optimizer, [], kl_budget=.045, ppo={}) == dict(decisions=0, accepted=None)
