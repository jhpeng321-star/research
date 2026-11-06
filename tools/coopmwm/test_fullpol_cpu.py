"""FULLPOL CPU contracts: free_program candidates, descriptor head, learning signals (no SUMO).

Real EpisodePolicy / TriggerPolicy / feasible_proposals / MoveExecutor / PPO / BC code paths.
Fakes: the N1 map stub of test_lane_drop_cpu (drop at s = 0, WM lanes 0..2, closing lane 2),
synthetic observation rows, and - until LCEXEC lands lc_program on this branch - a local stub
of LCEXEC's per-member executor (lcexec-free-program-v1 semantics for the primitives used).
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

from . import feasible_proposals, fullpol_head as fh, fullpol_learn as fl, fullpol_programs as fp
from . import launch_production as launch, track2_backend as tb
from .test_lane_drop_cpu import StubMap, advance, canon, ld_cfg, make, row
from .test_track2_backend_cpu import cfg, cpu_threads  # noqa: F401 (fixture + autouse)
from .trigger_policy import ppo_update
from tools.coopnative.policy import encode

PROV = 'FULLPOL CPU fixture (owner 2026-11-06 17:55; MAIN 18:4x approval)'
REPO = Path(__file__).resolve().parents[2]
BASE_COMMIT = '12c80d49'   # c4/g4ld tip (contains c4/nm-train 62955c86): the merged base without FULLPOL hunks
VIEW = dict(StubMap and {'name': 'wz3to2-stub', 'lane_drop_s_m': 0., 'lane_ids': [0, 1, 2], 'dropped_lane': 2})


def fp_cfg(cfg, *, templates=None, learn=None, guidance=None, parity=None, classes=fp.CLASSES, lane_drop=True):
    out = ld_cfg(cfg, v2v=False) if lane_drop else dict(cfg, trigger_partner_slots=4, trigger_candidates=8)
    out['fullpol'] = fp.config_keys(provenance=PROV, view=VIEW, templates=templates, classes=classes,
                                    learn=learn, regime_guidance=guidance, parity_families=parity)
    return out


def scene():
    """Closing-lane CAV a (lane 2), b in lane 1, c in lane 0, d behind a; HDVs h1/h2."""
    return [row('a', -1200, 2), row('b', -1190, 1), row('c', -1185, 0), row('d', -1240, 2),
            row('h1', -1150, 1, cav=False), row('h2', -1210, 2, cav=False)]


def make_fp(value, arm='B'):
    p = make(value, arm)
    if p.fullpol is not None:
        p.free_program_runtime = StubRuntime()
    return p


def state_for(policy, ego):
    return policy.local_states[ego]


def tick(p, vehicles, now, deterministic=True, seed=1, native=.3):
    p.observe(vehicles, now, native_commands={r['vid']: native for r in vehicles if r['is_cav']})
    context = encode(vehicles)
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    sample, _ = p.sample(context, torch.Generator().manual_seed(seed), deterministic)
    return sample


# ------------------------------------------------------------------ LCEXEC executor stub

class StubMemberExecutor:
    """Local stand-in for lc_program.MemberExecutor (speed / track / match / lane until_reached)."""

    def __init__(self, vid, prog):
        self.vid, self.p, self.active, self.released, self.lane_done = vid, prog, False, False, None

    def step(self, el, own, sensed):
        p, out = self.p, []
        if self.released or el < p['start_el']:
            return out
        if p['hold_s'] is not None and el >= p['start_el']+p['hold_s']:
            self.released = True
            return [('release', 'hold_end')]
        if not self.active:
            out.append(('lcMode', p['lc_mode']))
            if p['tau'] is not None:
                out.append(('tau', p['tau']))
            self.active = True
        q = p['lane']
        if q is not None and self.lane_done is None and el >= q['from_el']:
            if own['lane'] == q['target']:
                self.lane_done = el
            elif q['giveup_s'] is None or own['s_m'] < q['giveup_s']:
                out.append(('changeLane', (q['target'], q['duration_s'])))
        lon = p['lon']
        if lon['kind'] == 'speed' and el >= lon['from_el']:
            a = lon['v_w']-own['v_mps']
            out.append(('accel', max(-lon['dmax'], min(lon['amax'], a))))
        return out


class StubRuntime:
    """Stand-in for LCEXEC's runtime: steps each active member from its OWN observation and
    returns committed-offset accelerations (accel - native - MAPPO mean is LCEXEC's job; the
    stub returns accel - native) plus the per-member primitive commands."""

    def __init__(self):
        self.executors, self.log = {}, []

    def step(self, policy, context, entries):
        offsets, receipt = {}, {}
        for entry in entries:
            prog = entry['move'].move.program
            el = int(policy.now-entry['start'])
            for vid, p in zip(prog['members'], prog['programs']):
                key = (entry['start'], vid)
                ex = self.executors.setdefault(key, StubMemberExecutor(vid, p))
                own = policy.fp_table.get(vid)
                if own is None:
                    continue
                if not own['is_cav']:
                    raise AssertionError('HDV commanded')
                commands = ex.step(el, own, [])
                receipt[vid] = commands
                self.log.append((policy.now, vid, commands))
                for kind, value in commands:
                    if kind == 'accel':
                        offsets[vid] = value-policy.native_commands.get(vid, 0.)
        return offsets, receipt


# ------------------------------------------------------------------ settings / OFF identity

def test_settings_default_off_and_validation(cfg):
    assert fp.settings(cfg) is None and fp.settings(dict(cfg, fullpol=False)) is None
    on = fp_cfg(cfg)
    assert fp.settings(on)['rule'] == fp.RULE
    with pytest.raises(ValueError):
        fp.settings(dict(on, fullpol=dict(on['fullpol'], rule='other')))
    with pytest.raises(ValueError):
        fp.settings(dict(on, fullpol=dict(on['fullpol'], classes=['bogus'])))
    with pytest.raises(ValueError):
        fp.settings(dict(on, fullpol=dict(on['fullpol'], extra=1)))
    for key, value in fp.REGISTERED.items():
        assert value['source'] and value['values'], key


def test_off_absent_or_false_identical(cfg):
    outs = []
    for value in (ld_cfg(cfg, v2v=False), dict(ld_cfg(cfg, v2v=False), fullpol=False)):
        p = make(value)
        samples = [tick(p, advance(scene(), t-300), t, deterministic=False, seed=t) for t in range(300, 303)]
        outs.append(canon(samples)+canon({k: v.tolist() for k, v in p.worker_state()['trigger'].items()}))
        assert p.fullpol is None and 'free_program' not in p.local_states['a']
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
    out = [('a', -1200., 2, True), ('b', -1190., 1, True), ('c', -1185., 0, True), ('d', -1240., 2, True),
           ('h1', -1150., 1, False)]
    return [dict(vid=v, s_m=s+2.*dt, v_mps=2., lane=l, is_cav=c, length_m=5.) for v, s, l, c in out]
for now in range(300, 306):
    r = rows(now-300)
    if p.lane_drop is not None:
        p.observe(r, now, native_commands={x['vid']: 0. for x in r if x['is_cav']})
    else:
        p.observe(r, now)
    controller.apply(r, {x['vid']: 0. for x in r if x['is_cav']}, now)
digest = hashlib.sha256()
digest.update(json.dumps([e['sample'] for e in controller.events], sort_keys=True, default=repr).encode())
for part in ('layer', 'trigger'):
    for k, v in sorted(p.worker_state()[part].items()):
        digest.update(k.encode()); digest.update(v.detach().numpy().tobytes())
print(digest.hexdigest())
'''


@pytest.mark.parametrize('lane_drop', [False, True])
def test_off_is_byte_identical_to_the_pre_fullpol_sources(cfg, tmp_path, lane_drop):
    """FULLPOL OFF: samples and initial weights equal the c4/nm-train sources FULLPOL branched from."""
    touched = ('track2_backend.py', 'feasible_proposals.py', 'trigger_policy.py', 'launch_production.py',
               'launch_training.py')
    overlay = tmp_path/'overlay'
    overlay.mkdir()
    for name in touched:
        try:
            text = subprocess.run(['git', '-C', str(REPO), 'show', f'{BASE_COMMIT}:tools/coopmwm/{name}'],
                                  capture_output=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            pytest.skip('pre-FULLPOL source unavailable in this checkout')
        (overlay/name).write_bytes(text)
    base = ld_cfg(cfg, v2v=False) if lane_drop else dict(cfg)
    value = json.dumps(dict(base, run_root=str(tmp_path)), default=list)
    digests = []
    for path in ('-', str(overlay)):
        run = subprocess.run([sys.executable, '-c', SCENARIO, str(REPO), path, value], cwd=tmp_path,
                             capture_output=True, text=True)
        assert run.returncode == 0, run.stderr[-2000:]
        digests.append(run.stdout.strip())
    assert len(digests[0]) == 64 and digests[0] == digests[1]


# ------------------------------------------------------------------ candidates: coverage / guards

def _state(arm='B', templates=(), offered=()):
    return dict(view=fp.view_of(dict(view=VIEW)), classes=fp.CLASSES, structured=fp.STRUCTURED,
                templates=tuple(templates), offered=tuple(offered))


def _draws(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    rows = {r['vid']: r for r in scene()}
    st = _state()
    out = []
    for i in range(n):
        ego = ('a', 'b', 'c', 'd')[i % 4]
        partners = [r for v, r in rows.items() if r['is_cav'] and v != ego]
        cls = ('random_r', 'random_prim', 'structured')[i % 3]
        key = f'free_program:{cls}'
        option = fp._draw(key, st, rows[ego], partners, rows, rng)
        if option is not None:
            out.append(option)
    return out


def test_random_and_structured_programs_cover_every_registered_value_and_lane_type():
    draws = _draws()
    members = [(o, i, p) for o in draws for i, p in enumerate(o.program['programs'])]
    seen = lambda f: {f(o, i, p) for o, i, p in members}
    speed = [p for _, _, p in members if p['lon']['kind'] == 'speed']
    assert {p['lon']['v_w'] for p in speed} >= set(fp.registered('v_w'))
    assert {p['lon']['dmax'] for p in speed} == set(fp.registered('dmax'))
    assert {p['hold_s'] for _, _, p in members} >= set(fp.registered('hold_s'))
    assert {p['tau'] for _, _, p in members} == set(fp.registered('tau'))
    assert {p['start_el'] for _, _, p in members} == set(fp.registered('offsets_s'))
    assert {p['lane']['from_el'] for _, _, p in members if p['lane']} == set(fp.registered('offsets_s'))
    assert seen(lambda o, i, p: p['lon']['kind']) == set(fp.LON_KINDS)
    assert {len(o.members) for o in draws} == {1, 2, 3, 4}

    def lane_type(o, i, p):
        if p['lane'] is None:
            return 'none'
        own = o.observed[o.members[i]][0]
        t = p['lane']['target']
        return 'into_closing' if t == 2 else ('out_of_closing' if own == 2 else ('left' if t < own else 'right'))
    assert seen(lane_type) == {'none', 'left', 'right', 'into_closing', 'out_of_closing'}
    assert {o.source['family'] for o in draws if o.cls == 'structured'} >= {'closing_pair', 'abreast_wall'}
    # every value inside the registered grids (no new numbers)
    for _, _, p in members:
        assert p['lc_mode'] == 513 and p['release_s'] == 0.
        if p['lon']['kind'] == 'accel':
            assert (p['lon']['a'], p['hold_s']) == (-1.0, 2)


def test_structured_lane_change_families_build_from_registered_windows():
    rng = np.random.default_rng(1)
    st = _state()
    # wall_fill: a (lane 2) + b (lane 1) + filler f (lane 1, doubled) -> f into empty lane 0
    rows = {r['vid']: r for r in [row('a', -1200, 2), row('b', -1190, 1), row('f', -1230, 1)]}
    option = fp.structured('wall_fill', st['view'], rows['f'], [rows['a'], rows['b']], rows, rng)
    filler = option.program['programs'][0]
    assert option.members[0] == 'f' and filler['lane']['target'] == 0 and filler['lon']['kind'] == 'track'
    assert filler['lon']['refs'] == ['a', 'b'] and filler['lon']['settle_gap'] == 25.
    # gap_open_merge: merger m in the closing lane 200..900 m before the drop, yielder g -120..+30
    rows = {r['vid']: r for r in [row('m', -500, 2), row('g', -560, 1)]}
    option = fp.structured('gap_open_merge', st['view'], rows['g'], [rows['m']], rows, rng)
    assert option.members == ('g', 'm')
    g, m = option.program['programs']
    assert m['lane']['target'] == 1 and m['lane']['rule'] == 'from_lane' and m['lon']['v_w'] == 22.
    assert g['lon']['kind'] == 'track' and g['lon']['off'] == -25. and g['partner'] == 'm'
    assert fp.structured('gap_open_merge', st['view'], row('g', -1500, 1), [row('m', -1450, 2)], {}, rng) is None
    # lane_redistribute: 2 CAVs of the most loaded lane move to the least loaded adjacent lane
    rows = {r['vid']: r for r in [row('x', -1000, 1), row('y', -1050, 1), row('h', -1020, 1, cav=False),
                                  row('z', -990, 0, cav=False)]}
    option = fp.structured('lane_redistribute', st['view'], rows['x'], [rows['y']], rows, rng)
    assert {p['lane']['target'] for p in option.program['programs']} == {2}   # closing lane 2 is empty
    assert all(p['lon']['kind'] == 'match' for p in option.program['programs'])


def test_closing_lane_change_needs_room_before_the_registered_giveup():
    view = fp.view_of(dict(view=VIEW))
    assert fp.adjacent_targets(view, row('x', -1000, 1)) == [0, 2]
    assert fp.adjacent_targets(view, row('x', -200, 1)) == [0]          # past drop-300: no closing change
    assert fp.adjacent_targets(view, row('x', -200, 2)) == []            # stuck in the closing lane
    assert fp.adjacent_targets(view, row('x', 50, 1)) == [0]              # 2-lane section


def test_hdvs_never_members_and_hidden_fields_change_nothing():
    rng = lambda: np.random.default_rng(3)
    st = _state()
    rows = {r['vid']: r for r in scene()}
    hidden = {v: dict(r, driver_type='aggressive', tau_truth=.9, demand=3200, penetration=.6)
              for v, r in rows.items()}
    for key in fp.SLOTS[1:]:
        for ego in ('a', 'b'):
            a = fp._draw(key, st, rows[ego], [r for v, r in rows.items() if r['is_cav'] and v != ego], rows, rng())
            b = fp._draw(key, st, hidden[ego], [r for v, r in hidden.items() if r['is_cav'] and v != ego],
                         hidden, rng())
            assert canon(None if a is None else a.to_dict()) == canon(None if b is None else b.to_dict())
            if a is not None:
                assert all(rows[v]['is_cav'] for v in a.members)


def test_none_guard_keeps_the_feasible_template(cfg, monkeypatch):
    p = make(fp_cfg(cfg))
    p.observe(scene(), 300, native_commands={})
    calls = []

    def flaky(template, key, state, own, partners, by_vid, rng):
        calls.append(key)
        return None
    monkeypatch.setattr(fp, 'redraw', flaky)
    out = feasible_proposals.candidates(state_for(p, 'a'), np.random.default_rng(0), 8)
    assert calls and all(o is not None for o in out) and len(out) == 8
    assert any(fp.is_free_program(o) for o in out)


def test_verified_template_rebinds_by_lane_role_and_order():
    view = fp.view_of(dict(view=VIEW))
    prog = fp.program(['m1', 'm2'], [
        fp.member(fp.lon('speed', v_w=22., dmax=1., amax=1.), lane=fp.lane_req(1, from_el=20, giveup_s=-300.),
                  hold_s=120, release_s=0.),
        fp.member(fp.lon('track', refs=('m1',), off=-25., dmax=1.5, amax=1.), hold_s=120, release_s=0.,
                  partner='m1', partner_lane=2, releases=('hold_end', 'release_s', 'partner_left'))],
        source=dict(family='gap_open_merge'))
    t = fp.template(prog, member_lanes=[2, 1], member_s=[-1500., -1560.],
                    source=dict(verification_id='v1', family='gap_open_merge', arm='B'))
    rows = [row('a', -1200, 2), row('b', -1255, 1), row('c', -1100, 1)]
    option = fp.adapt(t, view, rows[1], [rows[0], rows[2]], np.random.default_rng(0))
    assert option.members == ('b', 'a') and option.cls == 'verified'
    assert option.program['programs'][0]['lon']['refs'] == ['a'] and option.program['programs'][0]['partner'] == 'a'
    assert fp.adapt(t, view, rows[2], [row('q', -1150, 0)], np.random.default_rng(0)) is None


# ------------------------------------------------------------------ head: zero-init, relative, no labels

def test_descriptor_block_is_zero_initialised_and_existing_logits_unchanged(cfg):
    off, on = make(ld_cfg(cfg, v2v=False)), make_fp(fp_cfg(cfg))
    assert on.trigger.descriptor_width == off.trigger.descriptor_width+fh.width(4)
    w_on, w_off = on.trigger.trigger_head[0].weight, off.trigger.trigger_head[0].weight
    hidden = on.trigger.base_policy.actor[-1].in_features
    start = hidden+off.trigger.descriptor_width
    assert torch.equal(w_on[:, start:start+fh.width(4)], torch.zeros_like(w_on[:, start:start+fh.width(4)]))
    assert torch.equal(torch.cat((w_on[:, :start], w_on[:, start+fh.width(4):]), 1), w_off)
    # every legacy candidate keeps its logit: program columns are zero for non-program entries
    vehicles = scene()
    s_off, s_on = tick(off, vehicles, 300), tick(on, vehicles, 300)
    t_off, t_on = s_off['track2_trigger'], s_on['track2_trigger']
    assert t_on['tables'][0][0] is None
    fams = {m['family'] for table in t_on['tables'] for m in table if m}
    assert fp.FAMILY in fams


def test_no_regime_label_demand_or_penetration_reaches_the_policy(cfg):
    """MAIN condition 2: guidance on/off and episode labels change no slate, input or logit."""
    guidance = dict(schema=fl.GUIDANCE_SCHEMA, regime_rule='cell', provenance=PROV, weight=1., clip=1.,
                    source=dict(sweep_id='S-REGIME', sha256='0'*64),
                    rows=[dict(arm='B', regime='D3200_p60', family='wall_g', mean_veh_s=900., lb95_veh_s=100., n=8)])
    outs = []
    for value in (fp_cfg(cfg), fp_cfg(cfg, guidance=guidance, parity=['wall_g'],
                                      learn=dict(ev_bc=True, ev_credit=True))):
        p = make_fp(value)
        samples = []
        for t in range(300, 303):
            clean = advance(scene(), t-300)
            labelled = [dict(r, demand=3200, penetration=.6, regime='D3200_p60') for r in clean]
            p.observe(labelled, t, native_commands={r['vid']: .3 for r in clean if r['is_cav']})
            context = encode(clean)   # the encoder itself refuses unregistered row keys
            context['v2v'] = {v: [0.]*48 for v in context['members']}
            samples.append(p.sample(context, torch.Generator().manual_seed(t), True)[0])
        with pytest.raises(ValueError, match='whitelist'):
            encode(labelled)
        outs.append(canon([s.get('track2_trigger') for s in samples])+canon(p.local_states))
        flat = canon(samples)+canon(p.local_states)
        assert 'D3200_p60' not in flat and 'penetration' not in flat and '"demand"' not in flat
    assert outs[0] == outs[1]


# ------------------------------------------------------------------ learning: EV terms / symmetry

def _guidance(rows):
    return dict(schema=fl.GUIDANCE_SCHEMA, regime_rule='cell', provenance=PROV, weight=1., clip=1.,
                source=dict(sweep_id='S-REGIME', sha256='0'*64), rows=rows)


def _episode(sample, *, seed=989220, practice=False, eid='e1'):
    return dict(id=eid, arm='B', practice=practice, world=dict(seed=seed, demand=3200, penetration=.6),
                steps=[dict(time_s=300, sample=sample)])


def _decision(cfg, *, arm='B'):
    p = make_fp(fp_cfg(cfg), arm)
    sample = tick(p, scene(), 300, deterministic=False, seed=5)
    return p, sample


def test_ev_terms_use_parity_passed_families_only_and_r_without_sweep_gets_none(cfg):
    rows = [dict(arm='B', regime='D3200_p60', family='abreast_wall', mean_veh_s=2000., lb95_veh_s=500., n=10),
            dict(arm='B', regime='D3200_p60', family='closing_pair', mean_veh_s=3000., lb95_veh_s=900., n=10),
            dict(arm='B', regime='D3200_p60', family='random', mean_veh_s=-500., lb95_veh_s=-900., n=10)]
    learn = dict(ev_bc=True, ev_credit=True)
    on = fp_cfg(cfg, guidance=_guidance(rows), parity=['abreast_wall', 'random'], learn=learn)
    table, receipt = fl.ev_table(on, 'B')
    assert set(table['D3200_p60']) == {'abreast_wall', 'random'}      # closing_pair: no parity -> dropped
    assert receipt['dropped_without_parity'] == {'closing_pair': 1}
    assert fl.best_family(table['D3200_p60']) == 'abreast_wall'
    r_table, r_receipt = fl.ev_table(on, 'R')
    assert r_table == {} and r_receipt['r_without_matched_sweep']
    p, sample = _decision(cfg)
    trig = sample['track2_trigger']
    # force every CAV's recorded choice onto a free_program slot and mark it accepted
    forced = 0
    for i, table in enumerate(trig['tables']):
        j = next((k for k, m in enumerate(table) if m and m['family'] == fp.FAMILY), None)
        if j is None:
            continue
        trig['actions'][i] = j
        trig['accepted'][trig['context']['members'][i]] = True
        forced += 1
    assert forced >= 1
    eps = [_episode(sample)]
    terms, rec = fl.ev_credit_terms(on, 'B', eps, eligible={('e1', 300)})
    labels = {vid: fl.option_labels(trig['tables'][i][trig['actions'][i]])
              for i, vid in enumerate(trig['context']['members'])}
    for vid, value in terms.items():
        fam = [f for f in labels[vid[2]] if f in ('abreast_wall', 'random')][0]
        assert value == pytest.approx(max(-1., min(1., {'abreast_wall': 2000., 'random': -500.}[fam]/10000.)))
    r_terms, r_rec = fl.ev_credit_terms(on, 'R', eps, eligible={('e1', 300)})
    assert r_terms == {} and r_rec['r_without_matched_sweep'] and r_rec['credited_rows'] == 0
    off_terms = fl.ev_credit_terms(fp_cfg(cfg), 'B', eps, eligible={('e1', 300)})
    assert off_terms == ({}, None) and fl.ev_credit_terms(cfg, 'B', eps, eligible={('e1', 300)}) == ({}, None)


def test_ev_bc_targets_recorded_slots_or_none(cfg):
    rows = [dict(arm='B', regime='D3200_p60', family='random', mean_veh_s=800., lb95_veh_s=10., n=10)]
    on = fp_cfg(cfg, guidance=_guidance(rows), parity=['random'], learn=dict(ev_bc=True))
    _, sample = _decision(cfg)
    records, receipt = fl.ev_bc_records(on, 'B', [_episode(sample)])
    assert records and receipt['rows'].get('act', 0) > 0
    for r in records:
        b = r['batch']
        for i in range(len(b.actions)):
            t = int(b.actions[i])
            if float(r['row_weight'][i]) and t:
                assert b.tables[i][t].source['family'] == 'random'
    neg = fp_cfg(cfg, guidance=_guidance([dict(rows[0], lb95_veh_s=-5.)]), parity=['random'], learn=dict(ev_bc=True))
    records, receipt = fl.ev_bc_records(neg, 'B', [_episode(sample)])
    assert records and all(int(a) == 0 for r in records for a in r['batch'].actions)   # no lb95>0 -> none


# ------------------------------------------------------------------ bank BC (exact programs)

def prog_frames(t0=300, seconds=4):
    frames = {}
    for t in range(t0-1, t0+seconds):
        frames[t] = dict(vehicles=advance([row('m1', -1300, 1), row('m2', -1290, 0),
                                           row('h1', -1250, 1, cav=False), row('h2', -1320, 2, cav=False)], t-t0))
    return frames


def closing_program():
    """m1 (lane 1) changes INTO the closing lane 2 and holds 15 m/s; m2 (lane 0) holds 22 m/s."""
    return fp.program(['m1', 'm2'], [
        fp.member(fp.lon('speed', v_w=15., dmax=1.5, amax=1.), lane=fp.lane_req(2, from_el=0, giveup_s=-300.),
                  start_el=0, hold_s=90, release_s=0.),
        fp.member(fp.lon('speed', v_w=22., dmax=1., amax=1.), start_el=20, hold_s=90, release_s=0.)],
        source=dict(family='lc_into_closing', verification_id='fp_v1'))


def prog_bank(frames, t0=300, negative=False):
    plan = dict(members=['m1', 'm2'], duration_s=110, free_program=closing_program(),
                fullpol_source=dict(verification_id='fp_v1', family='lc_into_closing'),
                free_program_labels=[dict(vid='m1', el=0, event='activate'), dict(vid='m2', el=20, event='activate')])
    if negative:
        plan['fullpol_negative'] = True
    pairs = []
    for t in sorted(frames):
        if t < t0:
            continue
        context = encode(frames[t]['vehicles'])
        context['v2v'] = {v: [0.]*48 for v in context['members']}
        pairs.append((dict(context=context, intent_link=dict(t=t)), 1.))
    return plan, pairs


def test_program_contexts_exact_window_and_hdv_rows(cfg):
    frames = prog_frames()
    plan, pairs = prog_bank(frames)
    p = make(fp_cfg(cfg))
    got = list(fl.program_contexts(p, plan, pairs, frames, t0=300, labels=plan['free_program_labels']))
    assert [(c['members'], o.members, t) for c, o, _, _, t in got] == [(['m1'], ('m1', 'm2'), 300),
                                                                     (['m2'], ('m2', 'm1'), 300)]
    _, option, reader, _, _ = got[0]
    assert option.program['programs'][0]['lane']['target'] == 2       # exact: into the closing lane
    assert any(not r['is_cav'] for r in reader.local_states['m1']['rows'])   # HDVs in the context rows
    with pytest.raises(ValueError):
        fl.shift(plan['free_program'], 1)          # m1 acts at el 0: no later exact context


def _visit(plan, pairs, frames, monkeypatch):
    from . import bank_breadth, bank_v2v

    def visit(config, requested_arm, consume):
        consume(cfg=config, receipt=dict(verification_id='fp_v1', t0=300, world=dict(seed=989220)),
                example=dict(example_id='fp_v1#s1'), plan=deepcopy(plan), pairs=deepcopy(pairs), trace='CPU')
    monkeypatch.setattr(bank_breadth, 'visit', visit)
    monkeypatch.setattr(bank_v2v, 'trace_frames', lambda path: frames)


@pytest.mark.parametrize('arm', ['B', 'R'])
def test_bc_teaches_the_exact_program_identically_for_b_and_r(cfg, monkeypatch, arm):
    frames = prog_frames()
    plan, pairs = prog_bank(frames)
    _visit(plan, pairs, frames, monkeypatch)
    on = fp_cfg(cfg, learn=dict(bc=True))
    out = {}
    for value in ('B', 'R'):
        p = make(on, value)
        out[value] = fl.supervise(SimpleNamespace(cfg=on, arm=value, policy=p), episodes=())
    res = out[arm]
    assert res['bank']['records'] == 2 and res['bank']['contexts_with_hdv_rows'] == 2
    assert res['fit']['final_ce'] < res['fit']['initial_ce']
    assert all(ps['kl'] <= .045 for ps in res['fit']['passes'] if ps['accepted'])
    strip = lambda r: canon({k: v for k, v in r.items() if k != 'arm'})
    assert strip(out['B']) == strip(out['R'])


def test_bank_move_supervision_never_projects_free_program_plans(cfg, monkeypatch):
    frames = prog_frames()
    plan, pairs = prog_bank(frames)
    _visit(plan, pairs, frames, monkeypatch)
    from . import moves
    monkeypatch.setattr(moves, 'from_plan', lambda plan: (_ for _ in ()).throw(AssertionError('projected')))
    p = make(fp_cfg(cfg))
    with pytest.raises(ValueError, match='nonempty observable verified-plan'):
        launch.bank_move_supervision(SimpleNamespace(cfg=fp_cfg(cfg), policy=p, arm='B'))


def test_report_only_receipts_kl_unlabelled_and_valsplit_auc(cfg, monkeypatch):
    frames = prog_frames()
    plans = []
    for k, seed in enumerate((989220, 989221, 989222, 989223)):
        plan, pairs = prog_bank(frames, negative=bool(k % 2))
        plans.append((plan, pairs, seed))
    from . import bank_breadth, bank_v2v

    def visit(config, requested_arm, consume):
        for plan, pairs, seed in plans:
            consume(cfg=config, receipt=dict(verification_id=f'fp_{seed}', t0=300, world=dict(seed=seed)),
                    example=None, plan=deepcopy(plan), pairs=deepcopy(pairs), trace='CPU')
    monkeypatch.setattr(bank_breadth, 'visit', visit)
    monkeypatch.setattr(bank_v2v, 'trace_frames', lambda path: frames)
    on = fp_cfg(cfg, learn=dict(bc=True, negatives=True, report_kl_unlabelled=True, report_valsplit_auc=True))
    p = make_fp(on)
    sample = tick(p, scene(), 300, deterministic=False, seed=5)
    before = deepcopy(p.trigger.state_dict())
    out = fl.supervise(SimpleNamespace(cfg=on, arm='B', policy=p), episodes=[_episode(sample)])
    assert out['bank']['kinds'] == {'program': 4, 'negative': 4}
    assert out['report_kl_unlabelled']['contexts'] == 1 and out['report_kl_unlabelled']['kl'] >= 0.
    v = out['report_valsplit_auc']
    assert v['seeds'] == 4 and len(v['folds']) == 2 and 'shadow' in v['rule']
    assert fl.auc([.9, .8], [.1, .2]) == 1. and fl.auc([.5], [.5]) == .5 and fl.auc([], [1.]) is None
    assert any(not torch.equal(before[k], t) for k, t in p.trigger.state_dict().items())


# ------------------------------------------------------------------ end-to-end (fake env)

def test_e2e_group_chooses_lane_change_into_closing_lane_executes_and_learns(cfg, monkeypatch):
    """A CAV group chooses a free_program (lane change INTO the closing lane + speed changes),
    it executes through the (stub) LCEXEC runtime for CAV members only, and the decision row
    receives BC, PPO (with EV credit) and G4 signals."""
    frames = prog_frames()
    plan, pairs = prog_bank(frames)
    _visit(plan, pairs, frames, monkeypatch)
    rows_ev = [dict(arm='B', regime='D3200_p60', family='lc_into_closing', mean_veh_s=3000., lb95_veh_s=800., n=9)]
    on = fp_cfg(cfg, learn=dict(bc=True, ev_credit=True), guidance=_guidance(rows_ev), parity=['lc_into_closing'])
    on['imitation_epochs'] = 40
    on['trigger_lr'] = .05
    p = make(on)
    template = fp.template(closing_program(), member_lanes=[1, 0], member_s=[-1300., -1290.],
                           source=dict(verification_id='fp_v1', family='lc_into_closing', arm='B'))
    p.fp_state = dict(p.fp_state, templates=(template,))
    # (1) BC on the verified exact program (KL budget and lr raised ONLY in this fixture so the
    # CPU test converges in one call; production keeps the registered .045 cap and lr)
    bc = fl.bc_steps(p, fl.bank_records(p, on, 'B')[0], epochs=40, kl_budget=10., max_grad_norm=.5)
    assert bc['final_ce'] < .1
    # (2) the group decides in a fresh situation: the verified program is re-bound and chosen
    p.free_program_runtime = runtime = StubRuntime()
    vehicles = [row('m1', -1300, 1), row('m2', -1290, 0), row('h1', -1250, 1, cav=False),
                row('h2', -1320, 2, cav=False)]
    p.observe(vehicles, 300, native_commands={'m1': .3, 'm2': .3})
    context = encode(vehicles)
    context['v2v'] = {v: [0.]*48 for v in context['members']}
    sample, _ = p.sample(context, torch.Generator().manual_seed(1), True)
    trig = sample['track2_trigger']
    i = trig['context']['members'].index('m1')
    chosen = trig['tables'][i][trig['actions'][i]]
    assert chosen and chosen['family'] == fp.FAMILY and chosen['members'] == ['m1', 'm2']
    assert chosen['program']['programs'][0]['lane']['target'] == 2 and trig['accepted']['m1']
    # (3) execution through the LCEXEC interface: lane change into the closing lane + speed laws
    cmds = sample['free_program_commands']
    assert ('changeLane', (2, 2.0)) in cmds['m1'] and any(k == 'accel' for k, _ in cmds['m1'])
    assert set(cmds) <= {'m1', 'm2'} and all(v.startswith('m') for _, v, _ in runtime.log)
    for now in range(301, 325):
        moved = advance(vehicles, now-300)
        p.observe(moved, now, native_commands={'m1': .3, 'm2': .3})
        ctx = encode(moved)
        ctx['v2v'] = {v: [0.]*48 for v in ctx['members']}
        p.sample(ctx, torch.Generator().manual_seed(now), True)
    m2 = [c for t, v, c in runtime.log if v == 'm2']
    assert any(k == 'accel' for c in m2 for k, _ in c)          # m2 starts its own speed law at el 20
    assert not any(v.startswith('h') for _, v, _ in runtime.log)  # HDVs never commanded
    # (4) PPO on the policy's own choice, with the additive EV credit for its family
    episode = _episode(sample)
    terms, receipt = fl.ev_credit_terms(on, 'B', [episode], eligible={('e1', 300)})
    assert terms[('e1', 300, 'm1')] == pytest.approx(.3) and receipt['credited_rows'] >= 1
    batch = tb._batch(trig)
    before = deepcopy(p.trigger.state_dict())
    adv = [terms.get(('e1', 300, v), 0.)+.1 for v in trig['context']['members']]
    metrics = ppo_update(p.trigger, p.trigger_optimizer, batch, adv, [0.]*len(adv), **on['trigger_ppo'])
    assert np.isfinite(metrics['loss']) and any(not torch.equal(before[k], v) for k, v in p.trigger.state_dict().items())
    # (5) G4: the free_program start becomes a G4LD opportunity row with a solo per member
    from . import sparse_synergy_credit as g4
    monkeypatch.setattr(g4, 'ADAPTERS', {}, raising=False)
    assert fl.register_g4_adapter()
    adapter = g4.ADAPTERS[fp.FAMILY]
    row_ = adapter['row'](episode, episode['steps'], 0, 'm1', chosen, trig, trig['context']['members'], False)
    adapter['check'](row_)
    assert row_['g4ld']['members'] == ['m1', 'm2'] and row_['g4ld']['credit'] == [dict(vid='m1', time_s=300)]
    solo = adapter['solo'](chosen['program'], 'm2')
    assert solo['members'] == ['m2'] and solo['programs'][0]['lon']['v_w'] == 22.


def test_selected_program_without_lcexec_runtime_fails_closed(cfg):
    on = fp_cfg(cfg)
    p = make(on)
    template = fp.template(closing_program(), member_lanes=[1, 0], member_s=[-1300., -1290.],
                           source=dict(verification_id='fp_v1', family='lc_into_closing', arm='B'))
    p.fp_state = dict(p.fp_state, templates=(template,), classes=('verified',))
    with torch.no_grad():
        # test-only preference: every free_program candidate outscores none
        hidden = p.trigger.base_policy.actor[-1].in_features
        col = hidden+p.trigger.descriptor_width-fh.width(4)
        p.trigger.trigger_head[0].weight.zero_()
        p.trigger.trigger_head[0].bias.zero_()
        p.trigger.trigger_head[0].weight[0, col] = 50.
        p.trigger.trigger_head[2].weight.fill_(1.)
    vehicles = [row('m1', -1300, 1), row('m2', -1290, 0)]
    with pytest.raises(ValueError, match='LCEXEC runtime'):
        tick(p, vehicles, 300)


def test_ppo_records_free_program_rows_and_backend_batch_roundtrip(cfg):
    p, sample = _decision(cfg)
    trig = sample['track2_trigger']
    batch = tb._batch(trig)
    logp, values, entropy = p.trigger.evaluate(batch)
    assert torch.allclose(logp, torch.tensor(trig['logp']), atol=1e-6)
    for table in batch.tables:
        for m in table:
            if m is not None and fp.is_free_program(m):
                assert fp.FreeProgramOption.from_dict(m.to_dict()) == m
