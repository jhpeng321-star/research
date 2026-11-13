"""Amendment 75 hold-form executor variant (CPU only, zero SUMO): A37/A44 digests unchanged, plain-plan command
identity vs the A44 executor (synthetic + recorded rows, hand + real searched hold plans + an lc plan), schema /
key / refusals, zones, short holds, the gap start (own-sensing window, timer / clear / arm end / repeats),
relative target, tapers, pulses, and the hook (CAV-only requests)."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.coopmwm import a37_cellbank as C
from tools.coopmwm import a44_lcvariant as V
from tools.coopmwm import a75_holdforms as F
from tools.coopmwm import oracle_plan as O
from tools.coopmwm import test_a44_lcvariant_cpu as T44
from tools.coopmwm.test_a44_lcvariant_cpu import CELL, Ctl, drive, plans, recorded, role, spec, traffic

EX = Path('E:/research-dense/experiments/a75_holdforms/EXAMPLE_PLANS.json')


def receipts_core(r):
    return {k: v for k, v in r.items() if k not in ('schema', 'executor', 'a37_arm', 'a44_arm', 'a75_arm', 'lc_members',
                                                     'lc_summary', 'refused', 'refused_counts', 'extended', 'holdforms',
                                                     'holdform_sessions', 'holdform_summary')}


# ------------------------------------------------------------------------------------------- old paths unchanged

def test_a37_a44_digests_unchanged_and_a75_is_its_own():
    assert C.a37_digest() == T44.A37_DIGEST
    assert V.a44_digest().startswith('53b64cbe')                 # ops/a73_lcside20.A44_SHA8
    assert all(O.vendored_check().values())
    d = F.a75_digest()
    assert d == F.a75_digest() and len(d) == 64 and d not in (V.a44_digest(), C.a37_digest())


# ------------------------------------------------------------------------------------------- plain-plan identity

PLAIN = T44.HOLDS + (spec('h_short', [role(2, 18.0, hold_s=5), role(1, 20.0, hold_s=8, onset_s=3)],
                          trigger='lanes', zone='far', spread_m=300.0, cooldown_s=10),)


@pytest.mark.parametrize('source', ['synthetic_a', 'recorded'])
@pytest.mark.parametrize('which', ['hand', 'a37_search'])
def test_plain_plans_issue_identical_commands_to_a44(source, which):
    specs = PLAIN + (T44.LC,) if which == 'hand' else T44.a37_search_holds()
    bank = V.bank_doc(plans(*specs), CELL)
    mine = F.bank_doc(plans(*specs), CELL)
    assert mine['rows'] == bank['rows'] and mine['clean5c'] == bank['clean5c']      # same bank form and keys
    old, new = V.make_hook('A44_BANK', CELL, bank=bank), F.make_hook('A75_BANK', CELL, bank=bank)
    stream = (lambda: traffic(11)) if source == 'synthetic_a' else (lambda: recorded(250))
    assert drive(old, stream()) == drive(new, stream())
    ra, rb = old.receipts(), new.receipts()
    assert ra['activations'] > 0 and ra['issued'] > 0
    assert receipts_core(ra) == receipts_core(rb)
    assert ra['lc_members'] == rb['lc_members'] and rb['extended'] == [] and rb['holdforms'] == []
    for fam in [s['name'] for s in specs][:2]:
        o, n = V.make_hook('A44_PLAN', CELL, bank=bank, family=fam), F.make_hook('A75_PLAN', CELL, bank=bank, family=fam)
        assert drive(o, traffic(5, t1=600)) == drive(n, traffic(5, t1=600))


def test_plain_cores_are_the_a44_code_objects():
    core = F.HoldFormCellBankCore(F.bank_doc(plans(*PLAIN), CELL), CELL)
    assert all(type(c) is V.LcStrategyCore for c in core.cores)
    assert F.HoldFormCellBankCore._gate is V.LcCellBankCore._gate


def test_key_and_canonical_form():
    for s in PLAIN + (T44.LC,):
        assert F.strategy_key(s) == O.strategy_key(s) and F.canonical(s) == s and not F.is_extended(s)
    s = spec('x', [dict(role(2), start='onset', taper_in_s=0, pulse_on_s=None, dv_rel=None)])
    assert F.canonical(s) == spec('x', [role(2)]) and F.strategy_key(F.canonical(s)) == O.strategy_key(spec('x', [role(2)]))
    e = spec('x', [dict(role(2), taper_in_s=5)])
    assert F.is_extended(e) and F.strategy_key(e) != O.strategy_key(spec('x', [role(2)]))
    z = spec('z', [role(2)], zone='custom', zone_d=[1000, 400])
    assert F.canonical(z)['zone_d'] == [1000.0, 400.0] and F.strategy_key(F.canonical(z)) != F.strategy_key(
        F.canonical(spec('z', [role(2)], zone='custom', zone_d=[1000, 300])))


# ------------------------------------------------------------------------------------------- refusals

@pytest.mark.parametrize('s, why', [
    (spec('a', [role(2)], zone='nowhere'), 'bad_zone'),
    (spec('a', [role(2)], zone='custom'), 'bad_zone'),
    (spec('a', [role(2)], zone='far', zone_d=[1000, 400]), 'bad_zone'),
    (spec('a', [role(2)], zone='custom', zone_d=[400, 1000]), 'bad_zone'),
    (spec('a', [role(2, hold_s=0)]), 'bad_holdform'),
    (spec('a', [dict(role(2), bogus=1)]), 'bad_holdform'),
    (spec('a', [dict(role(2), start='gap', gap_dx_lo=-10, gap_dx_hi=40, gap_arm_s=60)]), 'gap_role_lane'),
    (spec('a', [dict(role(1), start='gap', gap_dx_lo=-10, gap_dx_hi=40)]), 'bad_holdform'),
    (spec('a', [dict(role(1), start='gap', gap_dx_lo=40, gap_dx_hi=-10, gap_arm_s=60)]), 'bad_holdform'),
    (spec('a', [dict(role(1), start='gap', gap_dx_lo=-10, gap_dx_hi=140, gap_arm_s=60)]), 'bad_holdform'),
    (spec('a', [dict(role(2), gap_arm_s=60)]), 'bad_holdform'),
    (spec('a', [dict(role(2), pulse_on_s=5)]), 'bad_holdform'),
    (spec('a', [dict(role(2, hold_s=10), taper_in_s=6, taper_out_s=6)]), 'bad_holdform'),
    (spec('a', [dict(role(2), dv_rel=-1.0)]), 'bad_holdform'),
    (spec('a', [role(2), dict(role(1, lc_target=2), taper_in_s=5)]), 'lc_with_holdform'),
    (spec('a', [dict(role(2), taper_in_s=5)], phases=[[0]]), 'gate_or_phases'),
])
def test_refusals(s, why):
    assert F.executor_refusal(s) == why


def test_plain_refusals_are_the_a44_ones():
    for s in PLAIN + (T44.LC, spec('nc', [role(0, lc_target=1)]), spec('na', [role(2, lc_target=0)])):
        assert F.executor_refusal(s) == V.executor_refusal(s)


def test_bank_admission_refuses_with_counts():
    bad = spec('b', [dict(role(2), taper_in_s=5)])
    doc = V.bank_doc(plans(spec('ok', [role(2)])), CELL)
    t = deepcopy(doc['clean5c']['templates'][0])
    t.update(family='b')
    t['offer_condition']['roles'][0]['taper_in_s'] = 500           # > hold_s
    t['template']['source']['strategy_key'] = F.strategy_key(F.canonical(dict(bad, roles=[dict(role(2), taper_in_s=500)])))
    doc['clean5c']['templates'].append(t)
    doc['rows'].append(dict(doc['rows'][0], family='b'))
    plans_, refused = F.a75_bank_plans(doc, CELL)
    assert [p['family'] for p in plans_] == ['ok'] and refused == {'bad_holdform': ['b']}


# ------------------------------------------------------------------------------------------- zones

def test_zones():
    assert F.zone_bounds(dict(zone='far')) == O.ZONES['far']
    assert F.zone_bounds(dict(zone='merge')) == (O.G.X_B - 600.0, O.G.X_B - 100.0)
    assert F.zone_bounds(dict(zone='mid')) == (O.G.X_B - 1400.0, O.G.X_B - 600.0)
    assert F.zone_bounds(dict(zone='custom', zone_d=[1000.0, 400.0])) == (O.G.X_B - 1000.0, O.G.X_B - 400.0)


def test_select_members_equals_oracle_on_old_zones():
    n = 0
    for s in PLAIN:
        for t, rows, cmds in traffic(7, t1=500):
            free = [O.to_sumo_row(r) for r in rows if r['is_cav']]
            obs = O.observed_rows(rows)
            a, b = O.select_members(s, free, obs), F.select_members(s, free, obs)
            assert a == b
            n += a is not None
    assert n > 0


def cav(vid, x_sumo, lane_sumo, v=25.0):
    return dict(vid=vid, s_m=x_sumo - O.S_OFF, v_mps=v, lane=O.wm_lane_of(lane_sumo), is_cav=True, length_m=5.0)


def hdv(vid, x_sumo, lane_sumo, v=25.0):
    return dict(cav(vid, x_sumo, lane_sumo, v), is_cav=False)


def test_merge_zone_selects_only_near_the_drop():
    s = spec('m', [dict(role(2, hold_s=10))], trigger='lanes', zone='merge', spread_m=300.0)
    far = [O.to_sumo_row(cav('c1', O.G.X_B - 900, 2))]
    near = [O.to_sumo_row(cav('c2', O.G.X_B - 300, 2))]
    assert F.select_members(s, far) is None and [c['id'] for c in F.select_members(s, near)] == ['c2']
    c = dict(s, zone='custom', zone_d=[1000.0, 800.0])
    assert [m['id'] for m in F.select_members(c, far)] == ['c1'] and F.select_members(c, near) is None


# ------------------------------------------------------------------------------------------- hold timeline

def run_core(s, frames, warmup=0):
    """frames: list of (t, rows); commands: 0.3 for every CAV row. Returns [(t, accel, lanes)] and the core."""
    bank = F.bank_doc(plans(s), CELL)
    core = F.make_core('A75_PLAN', CELL, bank=bank, family=s['name'], warmup_s=warmup)
    out = []
    for t, rows in frames:
        a, ln = core.step(rows, {r['vid']: 0.3 for r in rows if r['is_cav']}, t)
        out.append((t, a, ln))
    return out, core


def moving(rows0, t0, t1, speeds=None):
    rows = deepcopy(rows0)
    for t in range(t0, t1):
        yield t, deepcopy(rows)
        for r in rows:
            r['s_m'] += (speeds or {}).get(r['vid'], r['v_mps'])


def test_short_hold_is_exactly_hold_s_seconds_and_released():
    s = spec('sh', [dict(role(2, v_w=18.0, hold_s=5), taper_in_s=0)], trigger='lanes', zone='far', spread_m=300.0,
             cooldown_s=1000)
    s = F.canonical(s)                                                  # plain (default dropped) - also via A44
    out, core = run_core(s, list(moving([cav('c1', O.G.X_B - 1500, 2, v=25.0)], 0, 12)))
    held = [t for t, a, _ in out if 'c1' in a]
    assert held == [0, 1, 2, 3, 4] and all(a['c1'] == -1.0 for t, a, _ in out if 'c1' in a)
    assert core.acts and core.acts[0]['release'] == {'c1': [5, 'hold_end']}


def test_onset_hold_events_and_release():
    s = spec('on', [dict(role(2, v_w=18.0, hold_s=4, onset_s=2), taper_in_s=2)], trigger='lanes', zone='far',
             spread_m=300.0, cooldown_s=1000)
    out, core = run_core(s, list(moving([cav('c1', O.G.X_B - 1500, 2, v=25.0)], 0, 10)))
    held = [t for t, a, _ in out if 'c1' in a]
    assert held == [2, 3, 4, 5]
    assert all('c1' in ln for t, a, ln in out if t < 6)                  # lane pin while in session (as A37)
    ev = core.cores[0].events
    assert len(ev) == 1 and ev[0]['start_kind'] == 'onset' and ev[0]['issued_s'] == 4 and ev[0]['end_why'] == 'hold_end'


def test_hold_target_taper_dv_rel_and_pulse():
    r = dict(role(2, v_w=16.0, hold_s=10), taper_in_s=4, taper_out_s=2)
    assert [round(F.hold_target(r, h, 24.0), 3) for h in range(10)] == [22.0, 20.0, 18.0, 16.0, 16.0, 16.0, 16.0,
                                                                        16.0, 20.0, 24.0]
    r = dict(role(2, v_w=10.0, hold_s=10), dv_rel=4.0)
    assert F.hold_target(r, 0, 24.0) == 20.0 and F.hold_target(r, 0, 12.0) == 10.0       # v_w is the floor
    assert F.hold_target(role(2, v_w=19.0), 3, 30.0) == 19.0                              # plain: v_w exactly
    p = dict(role(2, hold_s=20), pulse_on_s=3, pulse_off_s=2)
    assert [F.pulse_on(p, h) for h in range(7)] == [True, True, True, False, False, True, True]
    assert F.law(role(2, dmax=1.5), 10.0, 20.0) == -1.5 and F.law(role(2), 30.0, 20.0) == 1.0


def test_pulsed_hold_issues_only_in_on_seconds():
    s = spec('pu', [dict(role(2, v_w=18.0, hold_s=10), pulse_on_s=2, pulse_off_s=3)], trigger='lanes', zone='far',
             spread_m=300.0, cooldown_s=1000)
    out, core = run_core(s, list(moving([cav('c1', O.G.X_B - 1500, 2)], 0, 14)))
    assert [t for t, a, _ in out if 'c1' in a] == [0, 1, 5, 6]
    assert core.acts[0]['release'] == {'c1': [10, 'hold_end']}


# ------------------------------------------------------------------------------------------- gap start

GAP = dict(start='gap', gap_dx_lo=-10.0, gap_dx_hi=40.0, gap_arm_s=30, gap_drop_m=None)


def gap_spec(**kw):
    r = dict(role(1, v_w=0.0, hold_s=6), **dict(GAP, **kw))
    return F.canonical(spec('gp', [r], trigger='lanes', zone='merge', spread_m=300.0, cooldown_s=1000))


def test_gap_candidate_window_sensing_and_type_blind():
    r = dict(role(1), **GAP)
    me = cav('c1', O.G.X_B - 400, 1)
    assert F.gap_candidate(r, me, [me]) is None
    for kind in (hdv, cav):                                             # vehicle type is never read
        assert F.gap_candidate(r, me, [me, kind('v', O.G.X_B - 380, 0)]) == ('v', pytest.approx(20.0))
    assert F.gap_candidate(r, me, [me, hdv('v', O.G.X_B - 350, 0)]) is None       # dx 50 > hi
    assert F.gap_candidate(r, me, [me, hdv('v', O.G.X_B - 415, 0)]) is None       # dx -15 < lo
    assert F.gap_candidate(r, me, [me, hdv('v', O.G.X_B - 380, 1)]) is None       # not the closing lane
    me2 = cav('c1', O.G.X_B - 400, 2)                                             # SUMO lane 2 cannot sense lane 0
    assert F.gap_candidate(r, me2, [me2, hdv('v', O.G.X_B - 380, 0)]) is None
    w = dict(r, gap_dx_hi=100.0)
    assert F.gap_candidate(dict(w, gap_dx_lo=50.0), me, [me, hdv('v', O.G.X_B - 300, 0)]) == ('v', pytest.approx(100.0))
    assert F.gap_candidate(dict(r, gap_drop_m=300.0), me, [me, hdv('v', O.G.X_B - 380, 0)]) is None   # 380 m > 300
    assert F.gap_candidate(dict(r, gap_drop_m=400.0), me, [me, hdv('v', O.G.X_B - 380, 0)]) is not None
    assert F.gap_candidate(r, me, [me, hdv('a', O.G.X_B - 370, 0), hdv('b', O.G.X_B - 395, 0)])[0] == 'b'  # nearest


def test_gap_hold_fires_on_the_merging_vehicle_and_releases_on_timer():
    me = cav('c1', O.G.X_B - 500, 1, v=20.0)
    h = hdv('h1', O.G.X_B - 400, 0, v=10.0)        # closing-lane vehicle ahead, the CAV closes in at 10 m/s
    frames = list(moving([me, h], 0, 30))
    out, core = run_core(gap_spec(), frames)
    held = [t for t, a, _ in out if 'c1' in a]
    # dx = 100 - 10 t; enters dx <= 40 at t = 6 -> 6 s hold (6..11), one event, then 'hold_end' (events used)
    assert held == list(range(6, 12))
    assert all(a['c1'] == -1.0 for t, a, _ in out if 'c1' in a)          # v_w 0 -> -dmax
    ev = core.cores[0].events
    assert len(ev) == 1 and ev[0]['trigger_vid'] == 'h1' and ev[0]['fire_t'] == 6 and ev[0]['end_why'] == 'timer'
    assert core.acts[0]['release'] == {'c1': [12, 'hold_end']}
    assert all('h1' not in a and 'h1' not in ln for _, a, ln in out)    # HDVs never commanded


def test_gap_clear_release_repeat_and_arm_end():
    me = cav('c1', O.G.X_B - 500, 1, v=20.0)
    h = hdv('h1', O.G.X_B - 400, 0, v=10.0)
    s = gap_spec(gap_release='clear', gap_events=2, gap_dx_lo=0.0, gap_dx_hi=40.0, gap_arm_s=20)
    s['roles'][0]['hold_s'] = 30
    out, core = run_core(s, list(moving([me, h], 0, 40)))
    held = [t for t, a, _ in out if 'c1' in a]
    # in window while 0 <= 100 - 10 t <= 40: t = 6..10; clear at t = 11 -> event 6..10, no second vehicle;
    # armed period ends at el 20
    assert held == [6, 7, 8, 9, 10]
    ev = core.cores[0].events
    assert len(ev) == 1 and ev[0]['end_why'] == 'clear' and ev[0]['end_t'] == 11
    assert core.acts[0]['release'] == {'c1': [20, 'gap_arm_end']}
    r = core.receipts()
    assert r['holdform_summary']['gap_fired'] == 1 and r['holdform_summary']['clear_ends'] == 1


def test_gap_never_fires_without_a_vehicle_and_arm_ends():
    out, core = run_core(gap_spec(), list(moving([cav('c1', O.G.X_B - 500, 1, v=10.0)], 0, 40)))
    assert all('c1' not in a for _, a, _ in out)
    assert core.acts[0]['release'] == {'c1': [30, 'gap_arm_end']}
    assert core.receipts()['holdform_summary']['gap_armed_no_event'] == 1


def test_gap_hold_releases_at_the_drop():
    me = cav('c1', O.G.X_B - 150, 1, v=25.0)
    h = hdv('h1', O.G.X_B - 140, 0, v=25.0)
    s = gap_spec()
    s['roles'][0]['hold_s'] = 30
    out, core = run_core(s, list(moving([me, h], 0, 12)))
    held = [t for t, a, _ in out if 'c1' in a]
    assert held and held[0] == 0 and core.acts[0]['release']['c1'][1] == 'at_drop'


def test_dv_rel_gap_target_uses_speed_at_fire():
    me = cav('c1', O.G.X_B - 450, 1, v=20.0)
    h = hdv('h1', O.G.X_B - 430, 0, v=20.0)
    s = gap_spec(dv_rel=3.0)
    out, _ = run_core(s, list(moving([me, h], 0, 3)))
    assert out[0][1]['c1'] == pytest.approx(-1.0)       # target 17 - v 20 = -3 -> clipped at dmax 1.0
    assert F.hold_target(s['roles'][0], 0, 20.0) == 17.0


# ------------------------------------------------------------------------------------------- hook / bank

def example_specs():
    return [
        F.canonical(spec('ex_short', [role(2, 18.0, dmax=1.5, hold_s=10)], trigger='lanes', zone='far', spread_m=300.0,
                         cooldown_s=30)),
        F.canonical(spec('ex_gap', [dict(role(1, v_w=0.0, dmax=1.5, hold_s=10), start='gap', gap_dx_lo=-10.0,
                                         gap_dx_hi=40.0, gap_drop_m=300.0, gap_arm_s=60, gap_release='clear',
                                         gap_events=3, dv_rel=4.0)],
                         trigger='lanes', zone='merge', spread_m=300.0, cooldown_s=60)),
        F.canonical(spec('ex_zone', [role(2, 19.0, hold_s=60), role(1, 20.0, hold_s=60)], trigger='lanes',
                         zone='custom', zone_d=[1000.0, 400.0], spread_m=300.0, cooldown_s=60)),
        F.canonical(spec('ex_pulse', [dict(role(2, 16.0, hold_s=60), pulse_on_s=10, pulse_off_s=10)], trigger='lanes',
                         zone='mid', spread_m=300.0, cooldown_s=60)),
        F.canonical(spec('ex_taper', [dict(role(2, 16.0, hold_s=60), taper_in_s=10, taper_out_s=10)], trigger='lanes',
                         zone='far', spread_m=300.0, cooldown_s=60)),
        F.canonical(spec('ex_rolling', [role(2, 18.0, hold_s=60), role(2, 18.0, hold_s=60, onset_s=30, rank=1),
                                        role(1, 18.0, hold_s=60, onset_s=60)], trigger='lanes', zone='full',
                         spread_m=300.0, cooldown_s=120)),
    ]


def test_extended_bank_hook_on_synthetic_traffic_cav_only_and_deterministic():
    bank = F.bank_doc(plans(*example_specs()), CELL)
    a, b = F.make_hook('A75_BANK', CELL, bank=bank), F.make_hook('A75_BANK', CELL, bank=bank)
    ra = drive(a, traffic(11))
    assert ra == drive(b, traffic(11))
    hdvs = {r['vid'] for _, rows, _ in traffic(11) for r in rows if not r['is_cav']}
    for _, desired, lanes in ra:
        d = json.loads(desired)
        assert not any(v in hdvs for v, x in d.items() if 'oracle_plan' in x)
        assert not set(json.loads(lanes)) & hdvs
    r = a.receipts()
    assert r['a75_arm'] == 'A75_BANK' and r['activations'] > 0 and r['issued'] > 0
    assert sorted(r['extended']) == sorted(s['name'] for s in example_specs() if F.is_extended(s))
    assert r['holdform_summary']['events'] > 0 and r['holdform_summary']['gap_fired'] > 0


def test_example_plans_file_matches_the_bank_form():
    bank = F.bank_doc(plans(*example_specs()), CELL)
    for t in bank['clean5c']['templates']:
        assert F.strategy_key(F.spec_of(t['family'], t['offer_condition'])) == t['template']['source']['strategy_key']
    if EX.is_file():
        assert json.loads(EX.read_text(encoding='utf-8'))['bank'] == bank


# ------------------------------------------------------------------------------------------- runner wiring

def test_runner_hook_checks_bank_sha_digest_arm_and_cell(tmp_path):
    import hashlib
    from tools.coopmwm import nm_eval_runner as R
    specs = example_specs()
    bank = tmp_path/'bank.json'
    bank.write_text(json.dumps(F.bank_doc(plans(*specs), CELL)), encoding='utf-8')
    ck = dict(a75_arm='A75_PLAN', cell_key=CELL, a75_sha256=F.a75_digest(), a75_family='ex_gap',
              a75_bank=dict(path=str(bank), file_sha256=hashlib.sha256(bank.read_bytes()).hexdigest()))
    world = dict(cell_key=CELL)
    hook, loaded = R.a75_hook(ck, world, 'A75_PLAN')
    assert loaded == F.a75_digest() and [p['family'] for p in hook.core.plans] == ['ex_gap']
    for bad in (dict(a75_arm='A75_BANK'), dict(a75_sha256=V.a44_digest()), dict(bank_cell_key='D2800_p20'),
                dict(a75_bank=dict(path=str(bank), file_sha256='0'*64)), dict(cell_key='D2400_p20')):
        with pytest.raises(ValueError):
            R.a75_hook(dict(ck, **bad), world, 'A75_PLAN')
    hook, _ = R.a75_hook(dict(ck, a75_arm='A75_BANK'), world, 'A75_BANK')
    assert {p['family'] for p in hook.core.plans} == {s['name'] for s in specs}
    assert R.A75_ARMS == F.A75_ARMS and R.A44_ARMS == V.A44_ARMS and R.A37_ARMS == C.A37_ARMS


def test_runner_patch_is_additive_and_idempotent():
    import importlib.util
    here = Path(__file__).resolve()
    p = here.parents[2]/'ops'/'a75'/'host'/'patch_runner_a75.py'
    if not p.is_file():
        pytest.skip('ops/a75 not in this tree')
    spec_ = importlib.util.spec_from_file_location('patch_runner_a75', p)
    P = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(P)
    s = (here.parent/'nm_eval_runner.py').read_text(encoding='utf-8')
    assert 'A75_ARMS' in s and P.patch(s) == s
    for old, new in P.EDITS:                                  # every edit only adds to its anchor
        assert s.count(new) == 1
    with pytest.raises(SystemExit):
        P.patch('A44_ARMS = ()\n')


def test_ops_spec_helper():
    import importlib.util
    p = Path(__file__).resolve().parents[2]/'ops'/'a75'/'a75_specs.py'
    if not p.is_file():
        pytest.skip('ops/a75 not in this tree')
    spec_ = importlib.util.spec_from_file_location('a75_specs', p)
    S = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(S)
    base = [dict(arm='LEARNED_TRIGGER', world_id=w, job_id='x', checkpoint=dict(kind='learned_trigger', cell_key=CELL,
                                                                              learned_trigger_sha256='s',
                                                                              learned_trigger_model={}))
            for w in (1, 2)]
    out = S.a75_specs(base, 'A75_PLAN', dict(path='/b.json', file_sha256='f'), family='ex_gap', tag='t', digest='d')
    assert [o['job_id'] for o in out] == ['t_0000', 't_0001'] and base[0]['checkpoint']['kind'] == 'learned_trigger'
    ck = out[0]['checkpoint']
    assert ck['kind'] == 'a75' and ck['a75_family'] == 'ex_gap' and 'learned_trigger_model' not in ck
    with pytest.raises(ValueError):
        S.a75_specs(base, 'A75_PLAN', {}, family=None, tag='t', digest='d')
    rec = dict(a75_sha256_loaded='d', a75=dict(a75_arm='A75_PLAN', holdform_summary={}))
    assert S.check_a75(rec, out[0]) == rec['a75']
    with pytest.raises(ValueError):
        S.check_a75(dict(rec, a75_sha256_loaded='e'), out[0])
