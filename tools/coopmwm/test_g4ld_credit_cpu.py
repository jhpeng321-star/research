"""G4LD CPU contracts: G4 measured synergy credit for N2 lane-drop moves and W1/N2 V2V
sessions (default OFF). All 'sumo' receipts are explicit test doubles; no SUMO/GPU."""
from copy import deepcopy

import pytest
import torch

from . import g4_runtime, lane_drop, moves, v2v_wall
from .merge_features import ObservedVehicle
from . import sparse_synergy_credit as g4
from .test_sparse_synergy_credit_cpu import config, measured

VIEW = lane_drop.MapView('wz3to2', 75.44, (0, 1, 2), 2)
EXT = g4.extension_keys(provenance='G4LD CPU test (MAIN 2026-11-06 LEARNAUDIT finding 3)')
# Golden values from the ORIGINAL module (c4/nm-train 5eb22ab4 sparse_synergy_credit.py) on
# the fixtures below; the extension OFF must reproduce them byte for byte.
GOLDEN_PLAN_SHA = {2: '51edf24f45bf43bccbbafe4f657c857d7640daf600c033490b1a7faee5180991', 3: '18e0b60c5cf054b1d06411c9c9ae2986bffb6c392df61dbb5a811a64da82f245'}
GOLDEN_MIXED_PLAN_SHA = 'a2b65d9ed503fa4252d687ef6ef65358b256c8810f94dbcd3b27fed6710d64a0'
GOLDEN_TERMS_SHA = '651451d75964f7fe9aa019fbd227f0e03c9b67960cd0edeef008ca35b0b35e8a'


def ext_config():
    return dict(config(), g4_extension=deepcopy(EXT))


def times(proposer, n, cfg=None):
    cfg = cfg or config()
    return [t for t in range(1, 1000) if g4._sampled(dict(time_s=t, proposer=proposer),
                                                       dict(world=dict(seed=7)), cfg)][:n]


def _trigger(members, picks, accepted):
    """picks: vid -> selected option dict (others choose 'none' = slot 0)."""
    return dict(context=dict(members=members),
                tables=[[None, picks[v]] if v in picks else [None] for v in members],
                actions=[1 if v in picks else 0 for v in members],
                mask=[[True, True] if v in picks else [True] for v in members],
                accepted={v: accepted for v in picks}, values=[0.]*len(members), logp=[-.5]*len(members))


def _step(t, members, picks, accepted=True):
    return dict(time_s=t, sample=dict(context=dict(members=members),
                                      track2_trigger=_trigger(members, picks, accepted)))


def _episode(arm, steps):
    return dict(id=arm+'_ep', arm=arm, world=dict(seed=7), g4_source='ordinary_training',
                actual_sim_s=1200, steps=sorted(steps, key=lambda s: s['time_s']))


def joint_move(family):
    if family == 'closing_pair':
        return lane_drop.LaneDropMove('closing_pair', ('c0', 'c1'), 0, 120, 1.0, 22., lanes=(2, 1))
    return lane_drop.LaneDropMove('abreast_wall', ('c0', 'c1', 'c2'), 0, 120, 1.0, 22., lanes=(0, 1, 2))


def joint_steps(family, count=1, proposer='c1'):
    move = joint_move(family)
    members = list(move.members)
    return [_step(t, members, {proposer: move.to_dict()}) for t in times(proposer, count)]


def session_options(kind, t, joiners):
    """(initiate option, [(join time, join option)]) of one recorded session started at t."""
    wall_id = f'c0@{t}'
    if kind == 'w1':
        start = v2v_wall.WallOption('wall_initiate', ('c0',), 0, 60, .5, wall_id=wall_id,
                                    initiator='c0', cap=4, range_m=200.)
        joins = [(t+1+i, v2v_wall.WallOption('wall_join', (v,), 0, 59-i, .5, wall_id=wall_id,
                  initiator='c0', gap_m=10.*(i+1), member_count=1+i, cap=4, range_m=200.))
                 for i, v in enumerate(joiners)]
        return start, joins
    lanes = dict(c0=2, c1=1, c2=0)
    session = 'closing_pair' if len(joiners) == 1 else 'abreast_wall'
    cap = lane_drop.kind_cap(session, VIEW, 4)
    start = lane_drop.LaneDropOption('lane_drop_initiate', ('c0',), session, 0, 120, 1.0, 22., 0., False,
                                     lanes['c0'], wall_id=wall_id, initiator='c0', cap=cap, range_m=200.)
    joins = [(t+1+i, lane_drop.LaneDropOption('lane_drop_join', (v,), session, 0, 119-i, 1.0, 22., 0.,
              False, lanes[v], wall_id=wall_id, initiator='c0', gap_m=5.*(i+1), member_count=1+i, cap=cap,
              range_m=200., covered=tuple(sorted(lanes[w] for w in ['c0', *joiners[:i]]))))
             for i, v in enumerate(joiners)]
    return start, joins


def session_steps(kind, joiners=('c1',), count=1):
    members = ['c0', *joiners]
    steps = []
    for t in times('c0', count):
        start, joins = session_options(kind, t, list(joiners))
        steps.append(_step(t, members, {'c0': start.to_dict()}))
        for when, option in joins:
            steps.append(_step(when, members, {option.members[0]: option.to_dict()}))
    return steps


def captures(episodes, ext):
    return {a: {r['opportunity_id']: dict(stage='pre_action', time_s=r['time_s'],
        **{k: g4.digest([a, r['time_s'], k]) for k in g4.CAPTURE_KEYS})
        for r in (g4.opportunities(es) if ext is None else g4.opportunities(es, ext))}
        for a, es in episodes.items()}


def plan_for(steps_fn, cfg=None):
    cfg = cfg or ext_config()
    episodes = {a: [_episode(a, steps_fn())] for a in ('B', 'R')}
    ext = g4.extension(cfg)
    plan, state = g4.plan_update(cfg, 1, episodes, captures(episodes, ext), g4.initial_state(cfg))
    return cfg, episodes, plan, state


def envelope(plan, costs):
    return g4.measured_envelope(plan, [measured(j, costs) for j in plan['jobs']])


GOOD = dict(base=100., group=60., solo0=90., solo1=95., solo2=97., solo3=98.)   # synergy +30
BAD = dict(base=100., group=95., solo0=80., solo1=90., solo2=97., solo3=98.)    # synergy -15


# ---------------------------------------------------------------- (a) N2 lane-drop moves

@pytest.mark.parametrize('family,arity', [('closing_pair', 2), ('abreast_wall', 3)])
@pytest.mark.parametrize('costs,synergy', [(GOOD, 30.), (BAD, -15.)])
def test_joint_lane_drop_credit_signed_at_the_initiating_decision(family, arity, costs, synergy):
    cfg, episodes, plan, state = plan_for(lambda: joint_steps(family))
    assert [s['g4ld']['kind'] for s in plan['starts']] == ['n2_joint']*2
    assert len(plan['jobs']) == 2*2*(arity+2) and plan['quote_sim_s'] == 2*2*(arity+2)*1200
    assert {j['branch'] for j in plan['jobs']} == {'base', 'group', *[f'solo{i}' for i in range(arity)]}
    for arm in ('B', 'R'):
        terms, receipt = g4.trigger_terms(cfg, arm, 1, episodes[arm], envelope(plan, costs))
        start = next(s for s in plan['starts'] if s['arm'] == arm)
        # A joint move has ONE initiating decision (the proposer's); passive members get none.
        assert terms == {(start['episode_id'], start['time_s'], 'c1'): synergy/g4.UNIT}
        assert receipt['rows'][0]['synergy_veh_s'] == synergy


@pytest.mark.parametrize('joiners', [('c1',), ('c1', 'c2')])
@pytest.mark.parametrize('costs,synergy', [(GOOD, 30.), (BAD, -15.)])
def test_n2_session_credit_reaches_every_member_own_decision(joiners, costs, synergy):
    cfg, episodes, plan, _ = plan_for(lambda: session_steps('n2', joiners))
    start = next(s for s in plan['starts'] if s['arm'] == 'B')
    assert start['g4ld']['members'] == ['c0', *joiners]
    assert len(plan['jobs']) == 2*2*(len(joiners)+3)
    terms, _ = g4.trigger_terms(cfg, 'B', 1, episodes['B'], envelope(plan, costs))
    t0 = start['time_s']
    expected = {('B_ep', t0, 'c0'): synergy/g4.UNIT}
    expected.update({('B_ep', t0+1+i, v): synergy/g4.UNIT for i, v in enumerate(joiners)})
    assert terms == expected and all(v != 0 and (v > 0) == (synergy > 0) for v in terms.values())


# ---------------------------------------------------------------- (b) W1 V2V walls

@pytest.mark.parametrize('costs,synergy', [(GOOD, 30.), (BAD, -15.)])
def test_w1_wall_session_credit_reaches_initiator_and_every_joiner(costs, synergy):
    cfg, episodes, plan, _ = plan_for(lambda: session_steps('w1', ('c1', 'c2')))
    start = next(s for s in plan['starts'] if s['arm'] == 'R')
    assert start['g4ld']['kind'] == 'w1_session' and start['g4ld']['members'] == ['c0', 'c1', 'c2']
    terms, receipt = g4.trigger_terms(cfg, 'R', 1, episodes['R'], envelope(plan, costs))
    t0 = start['time_s']
    assert terms == {('R_ep', t0, 'c0'): synergy/g4.UNIT, ('R_ep', t0+1, 'c1'): synergy/g4.UNIT,
                     ('R_ep', t0+2, 'c2'): synergy/g4.UNIT}
    assert receipt['measured_starts'] == 1


def test_session_without_joins_is_not_a_joint_start_but_is_captured_pending():
    cfg = ext_config()
    ext = g4.extension(cfg)
    t = times('c0', 1)[0]
    start, _ = session_options('w1', t, [])
    episode = _episode('B', [_step(t, ['c0', 'c1'], {'c0': start.to_dict()})])
    assert g4.opportunities([episode], ext) == []
    pending = g4.opportunities([episode], ext, pending=True)
    assert len(pending) == 1
    # The capture key at the initiation second equals the final (joined) row's identity.
    full = _episode('B', session_steps('w1', ('c1',)))
    assert g4.opportunities([full], ext)[0]['opportunity_id'] == g4.opportunities(
        [dict(full, steps=full['steps'][:1])], ext, pending=True)[0]['opportunity_id']


def test_credit_skips_ineligible_join_rows_and_validates_session_binding():
    def steps():
        out = session_steps('w1', ('c1', 'c2'))
        out[2]['sample']['deterministic'] = True   # c2's join decision has no PPO row
        return out
    cfg, episodes, plan, _ = plan_for(steps)
    start = next(s for s in plan['starts'] if s['arm'] == 'B')
    assert start['g4ld']['members'] == ['c0', 'c1', 'c2']          # physics: c2 still joined
    assert [c['vid'] for c in start['g4ld']['credit']] == ['c0', 'c1']
    bad = deepcopy(plan)
    bad['starts'][0]['g4ld']['credit'].append(dict(vid='c9', time_s=0))
    bad['plan_sha256'] = g4.digest({k: v for k, v in bad.items() if k != 'plan_sha256'})
    with pytest.raises(ValueError, match='own initiating decision'):
        g4._validate_plan(bad)


def test_single_member_hold_is_not_a_joint_start():
    """A 1-member group_hold row (e.g. a solo hold subclass) is skipped like legacy 1-actor moves."""
    cfg = ext_config()
    t = times('c0', 1)[0]
    solo = dict(joint_move('closing_pair').to_dict(), family='group_hold', members=['c0'], lanes=[2])
    episode = _episode('B', [_step(t, ['c0', 'c1'], {'c0': solo})])
    assert g4.opportunities([episode], g4.extension(cfg)) == []
    assert g4.opportunities([episode], g4.extension(cfg), pending=True) == []


# ---------------------------------------------------------------- (c) caps / budget accounting

def test_extension_keeps_the_registered_start_and_sim_s_caps():
    cfg, episodes, plan, state = plan_for(lambda: session_steps('n2', ('c1', 'c2'), count=12))
    per_start = 2*(3+2)*1200
    assert state['starts'] == {'B': 8, 'R': 8} and len(plan['starts']) == 16
    assert plan['quote_sim_s'] == state['reserved_sim_s'] == 16*per_start == 192000
    assert state['reserved_sim_s'] <= g4.REGISTRATION['budget_cap_sim_s']
    cfg['worlds_by_update']['2'] = cfg['worlds_by_update']['1']
    later, final = g4.plan_update(cfg, 2, episodes, captures(episodes, g4.extension(cfg)), state)
    assert later['jobs'] == [] and final['reserved_sim_s'] == 192000


def test_mixed_families_match_by_arity_and_cost_formula():
    def steps():
        return joint_steps('closing_pair', 2)+session_steps('w1', ('c1', 'c2'), 1)
    cfg, _, plan, state = plan_for(steps)
    arities = sorted(len(g4.members_of(s)) for s in plan['starts'])
    assert arities == [2, 2, 2, 2, 3, 3]
    assert plan['quote_sim_s'] == sum(2*(a+2)*1200 for a in arities) == 2*(2*9600+12000)


def test_sessions_beyond_the_registered_actor_arity_are_never_sampled():
    cfg = ext_config()
    t = times('c0', 1)[0]
    start = v2v_wall.WallOption('wall_initiate', ('c0',), 0, 60, .5, wall_id=f'c0@{t}', initiator='c0',
                                cap=6, range_m=200.)
    members = [f'c{i}' for i in range(6)]
    steps = [_step(t, members, {'c0': start.to_dict()})]
    for i, v in enumerate(members[1:]):
        join = v2v_wall.WallOption('wall_join', (v,), 0, 59-i, .5, wall_id=f'c0@{t}', initiator='c0',
                                   member_count=1+i, cap=6, range_m=200.)
        steps.append(_step(t+1+i, members, {v: join.to_dict()}))
    episodes = {a: [_episode(a, deepcopy(steps))] for a in ('B', 'R')}
    ext = g4.extension(cfg)
    assert len(g4.members_of(g4.opportunities(episodes['B'], ext)[0])) == 6 > moves.MOVE_MAX_MEMBERS
    plan, state = g4.plan_update(cfg, 1, episodes, captures(episodes, ext), g4.initial_state(cfg))
    assert plan['jobs'] == [] and state['reserved_sim_s'] == 0


def test_extension_binding_is_journalled_and_cannot_switch_mid_phase():
    cfg, episodes, plan, state = plan_for(lambda: joint_steps('closing_pair'))
    assert plan['extension'] == state['extension'] == EXT
    off = config()
    with pytest.raises(ValueError):
        g4.plan_update(off, 2, episodes, captures(episodes, None), state)
    with pytest.raises(ValueError):
        g4.trigger_terms(off, 'B', 1, episodes['B'], envelope(plan, GOOD))


@pytest.mark.parametrize('slot', [
    dict(EXT, families=['wall_join']), dict(EXT, families=[]), dict(EXT, rule='other'),
    dict(EXT, families=['wall_initiate', 'closing_pair']), dict(EXT, provenance=' '),
    dict(EXT, extra=1), dict(EXT, families=['nm_early_merge'])])
def test_extension_registration_refuses_unregistered_shapes(slot):
    with pytest.raises(ValueError):
        g4.extension(dict(config(), g4_extension=slot))
    with pytest.raises(ValueError):
        g4.extension(dict(g4_extension=deepcopy(EXT)))   # needs the enabled G4 registration


# ---------------------------------------------------------------- (d) OFF identity (golden)

def _legacy_episode(arm, arity, count):
    members = [f'c{i}' for i in range(arity)]
    move = moves.Move('wall', tuple(members), 2, 20, .3).to_dict()
    return [_step(t, members, {'c0': move}) for t in times('c0', count)]


def _off_plan(steps_fn):
    cfg = config()
    episodes = {a: [_episode(a, steps_fn(a))] for a in ('B', 'R')}
    plan, _ = g4.plan_update(cfg, 1, episodes, captures(episodes, None), g4.initial_state(cfg))
    return cfg, episodes, plan


@pytest.mark.parametrize('arity', [2, 3])
def test_off_is_byte_identical_to_the_original_registration(arity):
    _, _, plan = _off_plan(lambda a: _legacy_episode(a, arity, 10))
    assert plan['plan_sha256'] == GOLDEN_PLAN_SHA[arity] and 'extension' not in plan
    assert 'extension' not in g4.initial_state(config())


def test_off_ignores_lane_drop_and_wall_rows_exactly_like_the_original():
    def steps(arm):
        return _legacy_episode(arm, 2, 3)+joint_steps('closing_pair', 2, 'c0')+session_steps('w1', ('c1',))
    cfg, episodes, plan = _off_plan(steps)
    assert plan['plan_sha256'] == GOLDEN_MIXED_PLAN_SHA
    assert all('g4ld' not in s for s in plan['starts'])
    terms, receipt = g4.trigger_terms(cfg, 'B', 1, episodes['B'], envelope(plan, GOOD))
    assert g4.digest([sorted(map(list, terms)), sorted(terms.values()), receipt]) == GOLDEN_TERMS_SHA


# ---------------------------------------------------------------- branch executor (CPU stub)

class _Report:
    def __init__(self, vid, s, lane):
        self.source = ObservedVehicle(vid, s, lane, 20., 5.)


class _Observer:
    def __init__(self, positions, lanes, *, w1=False, n2=False):
        self.lane_drop = dict(view=VIEW)
        self.execution_positions = {v: dict(positions) for v in positions}
        self.reports = {v: _Report(v, s, lanes[v]) for v, s in positions.items()}
        self.wall = v2v_wall.Protocol(dict(member_cap=4, range_m=200.)) if w1 else None
        self.ld_protocol = lane_drop.Protocol(dict(view=VIEW, v2v=dict(member_cap=4, range_m=200.),
                                                   provenance='test')) if n2 else None


def _job(start, retained):
    return dict(start=start, retained_members=list(retained), job_sha256='0'*64)


def _rows(positions, lanes, speed=20.):
    return [dict(vid=v, s_m=s, v_mps=speed, lane=lanes[v], is_cav=True, length_m=5.) for v, s in positions.items()]


@pytest.mark.parametrize('retained', [('c0', 'c1'), ('c0',), ('c1',), ()])
def test_branch_macro_n2_joint_masks_actors_and_runs_the_law(retained):
    cfg, _, plan, _ = plan_for(lambda: joint_steps('closing_pair', proposer='c0'))
    start = plan['starts'][0]
    positions, lanes = dict(c0=-900., c1=-880.), dict(c0=2, c1=1)
    macro = g4_runtime.ExtensionMacro(_job(start, retained), _Observer(positions, lanes))
    macro.before(_rows(positions, lanes), dict(c0=.2, c1=-.1), start['time_s'])
    context = dict(members=['c0', 'c1'])
    offsets, lanes_out = macro.offsets(context, torch.tensor([.05, -.05]))
    law = lane_drop.law(22., 20., 1.0)
    expected = {v: law-n-m for v, n, m in (('c0', .2, .05), ('c1', -.1, -.05)) if v in retained}
    assert offsets == pytest.approx(expected) and lanes_out == {}


def test_branch_macro_w1_session_replays_joins_and_masks():
    cfg, _, plan, _ = plan_for(lambda: session_steps('w1', ('c1', 'c2')))
    start = plan['starts'][0]
    t0 = start['time_s']
    positions, lanes = dict(c0=-400., c1=-390., c2=-380.), dict(c0=1, c1=0, c2=2)
    observer = _Observer(positions, lanes, w1=True)
    macro = g4_runtime.ExtensionMacro(_job(start, ('c0', 'c2')), observer)
    context = dict(members=['c0', 'c1', 'c2'])
    seen = []
    for t in (t0, t0+1, t0+2):
        observer.wall.receive(t, {r['vid']: r for r in _rows(positions, lanes)})
        macro.before(_rows(positions, lanes), {}, t)
        seen.append(macro.offsets(context, torch.zeros(3))[0])
        macro.after()
    assert seen == [dict(c0=-.5), dict(c0=-.5), dict(c0=-.5, c2=-.5)]
    assert set(observer.wall.sessions[f'c0@{t0}']['members']) == {'c0', 'c1', 'c2'}


def test_branch_macro_n2_session_uses_protocol_ledger():
    cfg, _, plan, _ = plan_for(lambda: session_steps('n2', ('c1',)))
    start = plan['starts'][0]
    t0 = start['time_s']
    positions, lanes = dict(c0=-600., c1=-590.), dict(c0=2, c1=1)
    observer = _Observer(positions, lanes, n2=True)
    macro = g4_runtime.ExtensionMacro(_job(start, ('c0', 'c1')), observer)
    law = lane_drop.law(22., 20., 1.0)
    for t, expected in ((t0, {'c0'}), (t0+1, {'c0', 'c1'})):
        observer.ld_protocol.receive(t, {r['vid']: r for r in _rows(positions, lanes)})
        macro.before(_rows(positions, lanes), dict(c0=0., c1=0.), t)
        offsets, _ = macro.offsets(dict(members=['c0', 'c1']), torch.zeros(2))
        assert offsets == pytest.approx({v: law for v in expected})
        macro.after()


def test_legacy_command_offsets_refuse_extension_jobs():
    _, _, plan, _ = plan_for(lambda: joint_steps('closing_pair'))
    with pytest.raises(ValueError):
        g4.command_offsets(plan['jobs'][0], 0, observable_state={'s_m': {}})


# ---------------------------------------------------------------- budget-derived start cap (opt-in)

CAP = g4.start_cap_keys(provenance='MAIN 2026-11-06 clean5a via N4: start cap derived from the G4 sim-s cap')


def cap_plan(steps_fn, *, ext=False, state=None, update=1):
    cfg = dict(config(), g4_start_cap=deepcopy(CAP), **({'g4_extension': deepcopy(EXT)} if ext else {}))
    cfg['worlds_by_update'][str(update)] = cfg['worlds_by_update']['1']
    episodes = {a: [_episode(a, steps_fn(a))] for a in ('B', 'R')}
    plan, after = g4.plan_update(cfg, update, episodes, captures(episodes, g4.extension(cfg)),
                                 state or g4.initial_state(cfg))
    return cfg, episodes, plan, after


def test_derived_caps_are_the_registered_budget_over_the_pair_effort():
    assert g4.derived_caps() == {2: 12, 3: 9, 4: 8}
    assert all(g4.pair_sim_s(n)*c <= g4.REGISTRATION['budget_cap_sim_s'] < g4.pair_sim_s(n)*(c+1)
               for n, c in g4.derived_caps().items())


@pytest.mark.parametrize('arity,cap,quote', [(2, 12, 230400), (3, 9, 216000), (4, 8, 230400)])
def test_budget_derived_cap_per_arity(arity, cap, quote):
    _, _, plan, state = cap_plan(lambda a: _legacy_episode(a, arity, 15))
    assert state['starts'] == {'B': cap, 'R': cap} and len(plan['starts']) == 2*cap
    assert plan['quote_sim_s'] == state['reserved_sim_s'] == quote <= g4.REGISTRATION['budget_cap_sim_s']
    assert state['arity_starts'][str(arity)] == cap and plan['start_cap'] == CAP
    g4._validate_plan(plan)


def test_cumulative_sim_s_cap_binds_across_arities_and_updates():
    def steps(arm):
        return _legacy_episode(arm, 2, 10)+_legacy_episode(arm, 3, 5)
    _, _, plan, state = cap_plan(steps)
    # 10 two-member pairs (192000) leave 38400: one three-member pair (24000) fits, a second does not.
    assert state['arity_starts'] == {'2': 10, '3': 1, '4': 0}
    assert plan['quote_sim_s'] == state['reserved_sim_s'] == 10*19200+24000 == 216000
    _, _, later, final = cap_plan(lambda a: _legacy_episode(a, 2, 10), state=state, update=2)
    assert later['jobs'] == [] and final['reserved_sim_s'] == 216000      # 14400 left < 19200
    tampered = deepcopy(plan)
    tampered['reservation_after']['arity_starts']['3'] = 2
    tampered['plan_sha256'] = g4.digest({k: v for k, v in tampered.items() if k != 'plan_sha256'})
    with pytest.raises(ValueError):
        g4._validate_plan(tampered)


def test_budget_derived_cap_with_the_extension_and_journal_binding():
    _, _, plan, state = cap_plan(lambda a: session_steps('n2', ('c1', 'c2'), count=12), ext=True)
    assert state['starts'] == {'B': 9, 'R': 9} and plan['quote_sim_s'] == 216000
    with pytest.raises(ValueError):          # the cap cannot be switched off mid-phase
        g4.plan_update(ext_config(), 2, {a: [] for a in ('B', 'R')}, {'B': {}, 'R': {}}, state)


@pytest.mark.parametrize('slot', [dict(CAP, rule='x'), dict(CAP, provenance=''), dict(CAP, extra=1),
                                  dict(CAP, enabled=False)])
def test_start_cap_registration_shape(slot):
    with pytest.raises(ValueError):
        g4.start_cap(dict(config(), g4_start_cap=slot))
    with pytest.raises(ValueError):
        g4.start_cap(dict(g4_start_cap=deepcopy(CAP)))       # needs the enabled G4 registration
    assert g4.start_cap(config()) is None and 'start_cap' not in g4.initial_state(config())
