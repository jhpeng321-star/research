"""Amendment 44 lane-change executor variant (CPU only, zero SUMO): the A37 executor and digest unchanged, hold-only
command identity vs the A37 executor (synthetic moving traffic + recorded plant rows + the real A37 WM-searched
hold plans), lc admission (adjacent, G5-executable), lane request at the onset, give-up point, completion
bookkeeping, refusals, and the search-semantics constants."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import re

import pytest

from tools.coopmwm import a37_cellbank as C
from tools.coopmwm import a44_lcvariant as V
from tools.coopmwm import oracle_plan as O
from tools.coopmwm.test_learned_trigger_cpu import wm_rows

HERE = Path(__file__).resolve().parent
OPS37 = HERE.parents[1]/'ops'/'a37'
CELL = 'D2800_p60'
A37_DIGEST = '0a9024371e0ea0686df8ac8ccdfa189fcceb1b413fb2604b31c0cf3a169707cf'
SEARCH_TREE = Path('E:/agent-work/c4-wmc-p20/tools/wzprop')


def spec(name, roles, **kw):
    s = dict(name=name, trigger='wall', zone='full', spread_m=200.0, cooldown_s=120, slots=1, gate=None,
             start_gate=None, phases=None, roles=roles)
    s.update(kw)
    return s


def role(lane, v_w=20.0, dmax=1.0, hold_s=120, onset_s=0, lc_target=None, rank=0):
    return dict(lane=lane, rank=rank, v_w=v_w, dmax=dmax, hold_s=hold_s, onset_s=onset_s, lc_target=lc_target)


def plans(*specs):
    return [dict(family=s['name'], strategy=s, lb95_veh_s=float(10 - i)) for i, s in enumerate(specs)]


HOLDS = (spec('h_duo', [role(1, 20.0), role(2, 21.0, onset_s=20)]),
         spec('h_solo', [role(2, 22.0, hold_s=90)], cooldown_s=90, start_gate='empty100'),
         spec('h_lanes', [role(2, 18.0, dmax=1.5), role(1, 22.0, onset_s=40), role(2, 22.0, rank=1)],
              trigger='lanes', zone='near', spread_m=300.0, cooldown_s=60, slots=2))
LC = spec('lc_fix', [role(2, 22.0), role(1, 22.0, onset_s=20, lc_target=2)])          # SUMO 1 -> 2 (WM 1 -> 0)


# ------------------------------------------------------------------------------------------- A37 path unchanged

def test_a37_digest_and_vendored_executor_unchanged():
    assert C.a37_digest() == A37_DIGEST
    assert all(O.vendored_check().values())
    assert C.executor_refusal(LC) == 'non_hold_plan'                     # the A37 rule itself is untouched
    core = C.CellBankCore(V.bank_doc(plans(LC), CELL), CELL)
    assert core.plans == [] and core.refused == {'non_hold_plan': ['lc_fix']}


def test_a44_digest_is_its_own_and_covers_the_a37_digest():
    d = V.a44_digest()
    assert d == V.a44_digest() and len(d) == 64 and d != A37_DIGEST


# ------------------------------------------------------------------------------------------- hold-only identity

class Ctl:
    def __init__(self):
        self.g5_lane_targets = {}

    def apply(self, rows, commands, now):
        self.g5_lane_targets = {}                         # the v2v controller recomputes its own targets every second
        return {vid: dict(requested_raw=a, base=True) for vid, a in commands.items()}


def traffic(seed, t0=280, t1=1000):
    """Synthetic moving WM-frame plant rows (CAVs + HDVs, lane changes, arrivals, exits) and U0 commands."""
    rng = random.Random(seed)
    veh, n = {}, 0
    for t in range(t0, t1):
        for _ in range(rng.choice((0, 1, 1, 2))):
            n += 1
            veh[f'{"c" if rng.random() < .6 else "h"}{n}'] = dict(s_m=-2700.0+rng.uniform(0, 400), v=rng.uniform(14, 30),
                                                                  lane=rng.choice((0, 1, 2)))
        for vid, x in list(veh.items()):
            x['v'] = min(31.0, max(2.0, x['v']+rng.uniform(-1.2, 1.0)))
            x['s_m'] += x['v']
            if rng.random() < .01:
                x['lane'] = min(2, max(0, x['lane']+rng.choice((-1, 1))))
            if x['s_m'] > 600:
                veh.pop(vid)
        rows = [dict(vid=vid, s_m=x['s_m'], v_mps=x['v'], lane=x['lane'], is_cav=vid.startswith('c'), length_m=5.0)
                for vid, x in sorted(veh.items())]
        yield t, rows, {r['vid']: round(rng.uniform(-1.5, 1.0), 3) for r in rows if r['is_cav']}


def recorded(t1=400):
    """Recorded plant rows (learned_trigger parity fixture, real episode tapes at decision seconds) advanced
    kinematically second by second."""
    fx = json.loads((HERE/'fixtures'/'learned_trigger_parity.json').read_text(encoding='utf-8'))
    for it in fx['items']:
        rows = deepcopy(it['rows'])
        for t in range(300, 300+t1):
            yield t, deepcopy(rows), {r['vid']: 0.25 for r in rows if r['is_cav']}
            for r in rows:
                r['s_m'] = float(r['s_m'])+float(r['v_mps'])


def drive(hook, stream):
    c = Ctl()
    hook(c, None, None)
    out = []
    for t, rows, cmds in stream:
        desired = c.apply(rows, cmds, t)
        out.append((t, json.dumps(desired, sort_keys=True), json.dumps(c.g5_lane_targets, sort_keys=True)))
    return out


def receipts_wo_identity(r):
    return {k: v for k, v in r.items() if k not in ('schema', 'executor', 'a37_arm', 'a44_arm', 'lc_members',
                                                     'lc_summary', 'refused', 'refused_counts')}


def a37_search_holds():
    out = []
    for cell in ('D3200_p20', 'D2400_p20'):
        doc = json.loads((OPS37/f'CANDIDATES_{cell}.json').read_text(encoding='utf-8'))
        out += [c['strategy'] for c in doc['candidates']]
    return out


@pytest.mark.parametrize('source', ['synthetic_a', 'synthetic_b', 'recorded'])
@pytest.mark.parametrize('which', ['hand', 'a37_search'])
def test_hold_only_plans_issue_identical_commands(source, which):
    specs = HOLDS if which == 'hand' else a37_search_holds()
    bank = C.bank_doc(plans(*specs), CELL)                                     # the A37 bank form, read by both
    assert V.bank_doc(plans(*specs), CELL)['rows'] == bank['rows']
    old, new = C.make_hook('CELL_BANK', CELL, bank=bank), V.make_hook('A44_BANK', CELL, bank=bank)
    stream = (lambda: traffic(11)) if source == 'synthetic_a' else (lambda: traffic(29)) if source == 'synthetic_b' \
        else recorded
    a, b = drive(old, stream()), drive(new, stream())
    assert a == b
    ra, rb = old.receipts(), new.receipts()
    assert ra['activations'] > 0 and ra['issued'] > 0                        # non-vacuous
    assert receipts_wo_identity(ra) == receipts_wo_identity(rb)
    assert rb['lc_members'] == [] and rb['executor'] == 'cell_bank_always_lc'
    for fam in [s['name'] for s in specs][:3]:                                 # single-plan arms too
        o, n = C.make_hook('A37_PLAN', CELL, bank=bank, family=fam), V.make_hook('A44_PLAN', CELL, bank=bank, family=fam)
        assert drive(o, traffic(5, t1=700)) == drive(n, traffic(5, t1=700))


def test_hold_only_admission_equals_oracle_bank_plans():
    for specs in (HOLDS, a37_search_holds()):
        doc = C.bank_doc(plans(*specs), CELL)
        assert V.lc_bank_plans(doc, CELL) == O.bank_plans(doc, CELL)
    bad = C.bank_doc(plans(HOLDS[0]), CELL)
    bad['clean5c']['templates'][0]['template']['source']['strategy_key'] = '0'*16
    assert V.lc_bank_plans(bad, CELL) == O.bank_plans(bad, CELL)


# ------------------------------------------------------------------------------------------- lane-change plans

def lc_rows(s_m=-1000.0, speed=25.0, lane_c1=1):
    rows = wm_rows(speed=speed)
    for r in rows:
        if r['is_cav']:
            r['s_m'] = s_m + {'c0': 0.0, 'c1': 10.0, 'c2': 20.0}[r['vid']]
    next(r for r in rows if r['vid'] == 'c1')['lane'] = lane_c1
    return rows


def test_lc_plan_is_admitted_and_requests_the_target_lane_from_its_onset():
    core = V.make_core('A44_BANK', CELL, bank=V.bank_doc(plans(LC), CELL))
    assert [p['family'] for p in core.plans] == ['lc_fix'] and core.refused == {}
    cmds = {'c0': 0.0, 'c1': 0.0, 'c2': 0.0}
    seen = []
    for t in range(300, 330):
        accel, lanes = core.step(lc_rows(), cmds, t)
        seen.append((t, accel, lanes))
    t0, accel, lanes = seen[0]
    # wall members: SUMO lane 2 = c0 (WM 0, onset 0), SUMO lane 1 = c1 (WM 1, onset 20, lc_target SUMO 2 = WM 0)
    assert accel == {'c0': -1.0} and lanes == {'c0': 0, 'c1': 1}               # before onset: G5 pin to own lane
    for t, accel, lanes in seen[1:20]:
        assert 'c1' not in accel and lanes['c1'] == 1
    for t, accel, lanes in seen[20:]:
        assert accel['c1'] == -1.0 and lanes['c1'] == O.wm_lane_of(2) == 0     # onset: law + lane request
    lc = core.receipts()['lc_members']
    assert len(lc) == 1 and lc[0]['vid'] == 'c1' and lc[0]['first_request'] == 320 and lc[0]['requests'] == 10
    assert lc[0]['completed'] is None and lc[0]['gave_up'] is None


def test_lane_request_reaches_the_g5_targets_through_the_hook_only_for_cavs():
    hook = V.make_hook('A44_PLAN', CELL, bank=V.bank_doc(plans(LC), CELL), family='lc_fix')
    c = Ctl()
    hook(c, None, None)
    for t in range(300, 321):
        rows = lc_rows()
        out = c.apply(rows, {r['vid']: 0.3 for r in rows if r['is_cav']}, t)
    assert c.g5_lane_targets == {'c0': 0, 'c1': 0}
    assert out['c1']['oracle_plan'] == 'lc_fix' and out['c1']['requested_raw'] == -1.0
    assert not any(v.startswith('h') for v in list(out) + list(c.g5_lane_targets))
    assert hook.receipts()['a44_arm'] == 'A44_PLAN'


def test_completion_is_recorded_and_the_request_stays_the_target():
    core = V.make_core('A44_BANK', CELL, bank=V.bank_doc(plans(LC), CELL))
    cmds = {'c0': 0.0, 'c1': 0.0, 'c2': 0.0}
    for t in range(300, 325):
        accel, lanes = core.step(lc_rows(lane_c1=0 if t >= 323 else 1), cmds, t)
    assert lanes['c1'] == 0
    lc = core.receipts()['lc_members'][0]
    assert lc['first_request'] == 320 and lc['completed'] == 323 and lc['gave_up'] is None
    assert core.receipts()['lc_summary'] == dict(members=1, requested=1, completed=1, gave_up=0, never_requested=0)


def test_give_up_point_is_the_search_lc_giveup_x():
    assert V.LC_GIVEUP_X == 2396.0
    core = V.make_core('A44_BANK', CELL, bank=V.bank_doc(plans(LC), CELL))
    cmds = {'c0': 0.0, 'c1': 0.0, 'c2': 0.0}
    out = {}
    base = 2276.0 - O.S_OFF          # all three wall members inside zone 'full' (x <= X_B - 400 = 2296) at t = 300
    for k, t in enumerate(range(300, 340)):
        # c1 (base + 10 m) moves 3.7 m/s: x = 2393.3 at t = 329, 2397.0 at t = 330 (lc_giveup_x = 2396)
        out[t] = core.step(lc_rows(s_m=base + 3.7*k), cmds, t)
    assert all(out[t][1]['c1'] == 0 for t in range(320, 330))                   # x < lc_giveup_x: request
    assert all(out[t][1]['c1'] == 1 for t in range(330, 340))                   # x >= lc_giveup_x: own lane (pin)
    assert all(out[t][0]['c1'] == -1.0 for t in range(320, 340))                # the speed program continues
    lc = core.receipts()['lc_members'][0]
    assert lc['gave_up'] == 330 and lc['completed'] is None and lc['requests'] == 10


def test_onset_after_the_give_up_point_never_requests():
    s = spec('lc_late', [role(2, 22.0), role(1, 22.0, onset_s=20, lc_target=2)])
    core = V.make_core('A44_BANK', CELL, bank=V.bank_doc(plans(s), CELL))
    cmds = {'c0': 0.0, 'c1': 0.0, 'c2': 0.0}
    # zone 'full' ends at X_B - 400 = 2296 < 2396: members start inside the zone, c1 passes 2396 before its onset
    base = 2276.0 - O.S_OFF
    for k, t in enumerate(range(300, 330)):
        accel, lanes = core.step(lc_rows(s_m=base + 6.0*k), cmds, t)
        if t >= 320:
            assert lanes['c1'] == 1
    lc = core.receipts()['lc_members'][0]
    assert lc['first_request'] is None and lc['gave_up'] == 320
    assert core.receipts()['lc_summary']['never_requested'] == 1


def test_refusals_non_adjacent_and_closing_lane():
    far = spec('lc_far', [role(2, 22.0, lc_target=0)])                        # SUMO 2 -> 0: not adjacent
    closing = spec('lc_close', [role(2, 22.0), role(1, 22.0, lc_target=0)])  # SUMO 1 -> 0 (WM 2): G5 refuses
    out_of = spec('lc_out', [role(2, 22.0), role(0, 22.0, lc_target=1)])     # SUMO 0 -> 1: executable
    assert V.executor_refusal(far) == 'lc_not_adjacent'
    assert V.executor_refusal(closing) == 'lc_into_closing_lane'
    assert V.executor_refusal(out_of) is None and V.executor_refusal(LC) is None
    assert V.executor_refusal(spec('x', [role(1, lc_target=2)], phases=[[0]])) == 'gate_or_phases'
    with pytest.raises(ValueError):
        V.bank_doc(plans(far), CELL)
    # a bank doc carrying them (A37 form, built by hand) is refused at admission, never executed
    doc = V.bank_doc(plans(LC, out_of), CELL)
    for s, why in ((far, 'lc_not_adjacent'), (closing, 'lc_into_closing_lane')):
        extra = V.bank_doc(plans(LC), CELL)
        cond = extra['clean5c']['templates'][0]['offer_condition']
        cond['roles'] = [dict(sumo_lane=r['lane'], rank=r['rank'], v_w=r['v_w'], dmax=r['dmax'], hold_s=r['hold_s'],
                              onset_s=r['onset_s'], lc_target_sumo=r['lc_target']) for r in s['roles']]
        extra['clean5c']['templates'][0]['template']['source']['strategy_key'] = O.strategy_key(dict(s, name='lc_fix'))
        extra['rows'][0]['family'] = extra['clean5c']['templates'][0]['family'] = s['name']
        merged = dict(doc, rows=doc['rows']+extra['rows'],
                      clean5c=dict(templates=doc['clean5c']['templates']+extra['clean5c']['templates']))
        plans_, refused = V.lc_bank_plans(merged, CELL)
        assert {p['family'] for p in plans_} == {'lc_fix', 'lc_out'} and refused == {why: [s['name']]}
        with pytest.raises(ValueError):
            V.make_core('A44_BANK', CELL, bank=merged)                         # bank arm: every row must run
    with pytest.raises(ValueError):
        V.make_core('A44_PLAN', CELL, bank=doc, family='nope')


def test_mixed_bank_hold_rows_still_match_a37_when_no_lc_member_is_active():
    """A hold row next to an lc row: the hold row's commands equal A37's one-row bank while the lc row never fires."""
    lc_never = spec('lc_never', [role(2, 22.0, rank=1), role(1, 22.0, lc_target=2, rank=1)], trigger='lanes',
                    zone='far', spread_m=0.0)   # two CAVs at exactly the same x: never
    bank = V.bank_doc(plans(HOLDS[0], lc_never), CELL)
    new = V.make_hook('A44_BANK', CELL, bank=bank)
    old = C.make_hook('CELL_BANK', CELL, bank=C.bank_doc(plans(HOLDS[0]), CELL))
    a, b = drive(old, traffic(3)), drive(new, traffic(3))
    fired = new.receipts()['activations_by_family']
    assert not fired.get('lc_never') and fired.get('h_duo')
    assert a == b


# ------------------------------------------------------------------------------------------- search semantics

def test_constants_are_the_search_lane_change_semantics():
    """The WM search (c4/wmc-p20 strategy_search.member_specs) runs an lc role as lcm_plan.prog(lc_from=onset,
    lc_giveup_x=X_B - GIVEUP_OFF, speed hold v_w/dmax from onset, hold_s = onset + hold_s, release_x = X_B, amax 1.0).
    Read as text only (no import of the other worktree)."""
    ss, lp = SEARCH_TREE/'strategy_search.py', SEARCH_TREE/'lcm_pipeline'/'lcm_plan.py'
    if not ss.is_file() or not lp.is_file():
        pytest.skip('c4/wmc-p20 worktree not on this host')
    src, plan = ss.read_text(encoding='utf-8'), lp.read_text(encoding='utf-8')
    assert ("P.prog(c, lc_target=r['lc_target'], lc_from=int(r['onset_s']), lc_giveup_x=P.X_B - P.GIVEUP_OFF,"
            in src)
    assert "sp_from=int(r['onset_s'])," in src and "hold_s=int(r['onset_s']) + int(r['hold_s'])))" in src
    assert re.search(r'^GIVEUP_OFF = 300\.0\b', plan, re.M) and re.search(r'^X_B = 2696\.0\b', plan, re.M)
    assert re.search(r'^AMAX = 1\.0\b', plan, re.M)
    assert float(re.search(r'^GIVEUP_OFF = ([0-9.]+)', plan, re.M).group(1)) == V.GIVEUP_OFF
    assert O.G.X_B == 2696.0 and O.S_OFF + O.RELEASE_S == O.G.X_B          # release at the drop = release_x X_B
    assert "if r['lc_target'] is not None and abs(int(r['lc_target']) - int(r['lane'])) != 1:" in src


def test_g5_writer_refuses_the_closing_lane_target():
    """The existing lane path (G5 env overlay) never issues a request into WM lane 2: the A44 refusal mirrors it."""
    text = (HERE/'g5_runtime_env.pysrc').read_text(encoding='utf-8')
    assert 'if target not in (0, 1, 2) or abs(target - current) != 1 or target == 2:' in text
    assert V.G5_REFUSED_WM_TARGET == 2 and O.wm_lane_of(0) == 2


# ------------------------------------------------------------------------------------------- runner wiring

def test_runner_hook_checks_bank_sha_digest_arm_and_cell(tmp_path):
    import hashlib
    from tools.coopmwm import nm_eval_runner as R
    bank = tmp_path/'bank.json'
    bank.write_text(json.dumps(V.bank_doc(plans(LC, HOLDS[0]), CELL)), encoding='utf-8')
    ck = dict(a44_arm='A44_PLAN', cell_key=CELL, a44_sha256=V.a44_digest(), a44_family='lc_fix',
              a44_bank=dict(path=str(bank), file_sha256=hashlib.sha256(bank.read_bytes()).hexdigest()))
    world = dict(cell_key=CELL)
    hook, loaded = R.a44_hook(ck, world, 'A44_PLAN')
    assert loaded == V.a44_digest() and [p['family'] for p in hook.core.plans] == ['lc_fix']
    for bad in (dict(a44_arm='A44_BANK'), dict(a44_sha256=A37_DIGEST), dict(bank_cell_key='D2800_p20'),
                dict(a44_bank=dict(path=str(bank), file_sha256='0'*64)), dict(cell_key='D2400_p20')):
        with pytest.raises(ValueError):
            R.a44_hook(dict(ck, **bad), world, 'A44_PLAN')
    hook, _ = R.a44_hook(dict(ck, a44_arm='A44_BANK'), world, 'A44_BANK')
    assert {p['family'] for p in hook.core.plans} == {'lc_fix', 'h_duo'}
    assert R.A44_ARMS == V.A44_ARMS and R.A37_ARMS == C.A37_ARMS


def test_runner_seed_ranges_a44_block_added_others_unchanged():
    from tools.coopmwm import map_scenarios as ms
    from tools.coopmwm import nm_eval_runner as R
    R.install_eval_ranges(ms)
    assert R.A44_MECH_RANGE == (992400, 992499) and R._eval_base(992403) == 992400
    assert ms.world(992403, 2800, .6)['repeat'] == 3 and ms.seed_scope(992403) == 'heldout'
    assert R._eval_base(991699) == 991600 and R._eval_base(991900) is None and R._eval_base(991805) == 991800
    assert R._eval_base(992399) is None and R._eval_base(992500) is None
    assert R.EVAL_HELDOUT_RANGES[:-1] == ((993000, 993299, 993000), (999500, 999699, 999500), (991600, 991699, 991600),
                                          (998000, 998099, 998000), (998100, 998199, 998100), (998200, 998299, 998200),
                                          (998300, 998399, 998300), (991700, 991799, 991700), (991800, 991899, 991800))
