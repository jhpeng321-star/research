"""A75 WM hold-form controller tests (CPU torch, zero SUMO, no WM weights except the optional smoke).

The controller's per-second commands are compared with a reference timeline built from the EXECUTOR's own pure
functions (tools/coopmwm/a75_holdforms: gap_candidate, hold_target, pulse_on, law; HoldFormCore._member order) on
scripted scenes, falling back to the a75ss mirrors when the executor test environment is absent.
"""
import json
import math
import os
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TREE = ROOT / 'ref' / 'code_v12r7b'
SHIP = ROOT / 'ship'
TESTENV = ROOT.parent / '_testenv'
for p in (str(SHIP), str(TREE / 'src'), str(TREE)):
    if p not in sys.path:
        sys.path.insert(0, p)
torch = pytest.importorskip('torch')

import wms.planapi as P  # noqa: E402
import a75_wmforms as WF  # noqa: E402

X_B, DROP_S = 2696.0, 75.44
ANCHOR = X_B - DROP_S
NATIVE = -0.123


def executor_funcs():
    """(gap_candidate, hold_target, pulse_on, law) from the executor when importable, else None."""
    if not (TESTENV / 'tools' / 'coopmwm' / 'a75_holdforms.py').exists():
        return None
    saved = {k: v for k, v in sys.modules.items() if k == 'tools' or k.startswith('tools.')}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(TESTENV))
    try:
        import tools.coopmwm.a75_holdforms as H
        return H.gap_candidate, H.hold_target, H.pulse_on, H.law
    finally:
        sys.path.remove(str(TESTENV))
        for k in [k for k in sys.modules if k == 'tools' or k.startswith('tools.')]:
            sys.modules['_exec2_' + k] = sys.modules.pop(k)
        sys.modules.update(saved)


H_FUNCS = executor_funcs()


def _gap_candidate(role, own, rows):
    lo, hi, dm = float(role['gap_dx_lo']), float(role['gap_dx_hi']), role.get('gap_drop_m')
    best = None
    for r in rows:
        if r['vid'] == own['vid'] or int(r['lane']) != 2 or float(r['s_m']) >= DROP_S:
            continue
        dx = float(r['s_m']) - float(own['s_m'])
        if abs(dx) > 100 or abs(int(r['lane']) - int(own['lane'])) > 1 or not lo <= dx <= hi:
            continue
        if dm is not None and DROP_S - float(r['s_m']) > float(dm):
            continue
        k = (abs(dx), str(r['vid']))
        if best is None or k < best[0]:
            best = (k, (r['vid'], dx))
    return None if best is None else best[1]


def _hold_target(role, h, v0):
    tgt = float(role['v_w']) if role.get('dv_rel') is None else max(float(role['v_w']), v0 - float(role['dv_rel']))
    f = 1.0
    if role.get('taper_in_s'):
        f = min(f, (h + 1) / float(role['taper_in_s']))
    if role.get('taper_out_s'):
        f = min(f, max(0, int(role['hold_s']) - 1 - h) / float(role['taper_out_s']))
    return tgt if f >= 1.0 else v0 + (tgt - v0) * f


def _pulse_on(role, h):
    on, off = role.get('pulse_on_s'), role.get('pulse_off_s')
    return on is None or h % (on + off) < on


def _law(role, target, v):
    return max(-float(role['dmax']), min(1.0, float(target) - float(v)))


GAP_CAND, HOLD_TARGET, PULSE_ON, LAW = H_FUNCS or (_gap_candidate, _hold_target, _pulse_on, _law)


def reference(role, scene, T):
    """Executor HoldFormCore._member order for one member activated at t = 0 -> {t: accel or None}."""
    hold, onset = int(role['hold_s']), int(role['onset_s'])
    gap = role.get('start') == 'gap'
    ev, n_ev, released, out = None, 0, False, {}
    for t in range(T):
        own, rows = scene(t)
        out[t] = None
        if released:
            continue
        if not gap:
            if t >= onset + hold:
                released = True
                continue
        else:
            if ev is not None:
                h = t - ev['fire']
                if h >= hold:
                    ev = None
                elif role.get('gap_release') == 'clear' and h >= 1 and GAP_CAND(role, own, rows) is None:
                    ev = None
            if ev is None:
                if n_ev >= role.get('gap_events', 1) or t >= onset + int(role['gap_arm_s']):
                    released = True
                    continue
        if float(own['s_m']) >= DROP_S:
            released = True
            continue
        if t < onset:
            continue
        if gap and ev is None:
            if GAP_CAND(role, own, rows) is None:
                continue
            n_ev += 1
            ev = dict(fire=t, v0=float(own['v_mps']))
        if not gap and ev is None and t == onset:
            ev = dict(fire=t, v0=float(own['v_mps']))
        if ev is None:
            continue
        h = t - ev['fire']
        if not PULSE_ON(role, h):
            continue
        out[t] = LAW(role, HOLD_TARGET(role, h, ev['v0']), own['v_mps'])
    return out


def dx_script(t):
    """Closing-lane vehicle position relative to the member (m) over time: in / out of every window."""
    seq = [(5, 80.0), (12, 30.0), (16, 120.0), (30, 10.0), (34, -50.0), (45, 5.0), (52, 70.0), (70, 15.0)]
    for end, dx in seq:
        if t < end:
            return dx
    return -15.0


def make_scene(s_member0, v_member0=22.0, dx=dx_script):
    def scene(t):
        s_m = s_member0 + 20.0 * t
        v = v_member0 - 0.08 * t
        own = dict(vid='m', s_m=s_m, lane=1, v_mps=v)
        rows = [own, dict(vid='z', s_m=s_m + dx(t), lane=2, v_mps=15.0),
                dict(vid='o', s_m=s_m + 3.0, lane=0, v_mps=25.0)]
        return own, rows
    return scene


ROLES = [
    dict(lane=2, v_w=12.0, dmax=1.5, hold_s=20, onset_s=4, dv_rel=3.0, taper_in_s=5, taper_out_s=5,
         pulse_on_s=3, pulse_off_s=2),
    dict(lane=1, v_w=18.0, dmax=1.0, hold_s=10, onset_s=0, start='gap', gap_dx_lo=-10.0, gap_dx_hi=40.0,
         gap_arm_s=30),
    dict(lane=1, v_w=0.0, dmax=1.5, hold_s=15, onset_s=2, start='gap', gap_dx_lo=-20.0, gap_dx_hi=20.0,
         gap_arm_s=60, gap_release='clear', gap_events=3, dv_rel=4.0),
    dict(lane=1, v_w=14.0, dmax=1.5, hold_s=20, onset_s=0, start='gap', gap_dx_lo=0.0, gap_dx_hi=60.0,
         gap_arm_s=120, gap_drop_m=900.0, gap_events=3, pulse_on_s=5, pulse_off_s=5, taper_in_s=4),
    dict(lane=2, v_w=16.0, dmax=1.0, hold_s=30, onset_s=10),                       # plain (no form)
]
S0 = [-1500.0, -1500.0, -1500.0, -1500.0, -1500.0]
T = 90


def _member(role, slot):
    form = {k: role[k] for k in WF.FORM_KEYS if k in role}
    m = dict(slot=slot, v_w=float(role['v_w']), start_s=float(role['onset_s']), hold_s=float(role['hold_s']),
             dmax=float(role['dmax']), amax=1.0, release_x=X_B, anchor=None, anchor_release_x=math.inf)
    if form:
        m['form'] = form
    return m


def _state(k, scenes, n=4):
    t = k * 0.5
    ti = int(math.floor(t))
    B = len(scenes)
    s = torch.zeros(B, n)
    v = torch.zeros(B, n)
    lane = torch.zeros(B, n, dtype=torch.long)
    present = torch.ones(B, n, dtype=torch.bool)
    for b, sc in enumerate(scenes):
        own, rows = sc(ti)
        for j, r in enumerate(rows):
            s[b, j], v[b, j], lane[b, j] = r['s_m'], r['v_mps'], r['lane']
        s[b, 3], lane[b, 3], present[b, 3] = own['s_m'] + 5.0, 2, False      # absent slot inside every window
    return {"s_m": s, "v_mps": v, "lane": lane, "present": present, "physics_step_index": k,
            "merge_pivot_native_accel": torch.full((B, n), NATIVE)}


def run_controller(roles, scenes):
    ctrl = WF.HoldFormController([[_member(r, 0)] for r in roles], 4, ANCHOR, 0.5, 'cpu')
    acc = {b: {} for b in range(len(roles))}
    for k in range(2 * T):
        out = ctrl(_state(k, scenes))
        if k % 2 == 0:
            for b in range(len(roles)):
                acc[b][k // 2] = float(out[b, 0, 3])
    return ctrl, acc


def test_controller_matches_executor_timeline():
    scenes = [make_scene(s0) for s0 in S0]
    ctrl, acc = run_controller(ROLES, scenes)
    for b, role in enumerate(ROLES[:4]):
        ref = reference(role, scenes[b], T)
        n_cmd = 0
        for t in range(T):
            want = NATIVE if ref[t] is None else ref[t]
            n_cmd += ref[t] is not None
            assert abs(acc[b][t] - want) < 1e-4, (b, t, acc[b][t], want)
        assert n_cmd > 5, b
    assert int(ctrl.nev[2, 0]) == 3 and int(ctrl.nev[1, 0]) == 1
    assert H_FUNCS is not None, 'executor functions were not available; mirror reference used'


def test_plain_rows_identical_to_original_controller():
    scenes = [make_scene(s0) for s0 in S0]
    mixed = WF.HoldFormController([[_member(r, 0)] for r in ROLES], 4, ANCHOR, 0.5, 'cpu')
    orig = P.HoldSpeedController([[_member(ROLES[4], 0)]], 4, ANCHOR, 0.5, 'cpu')
    plain_only = WF.HoldFormController([[_member(ROLES[4], 0)]], 4, ANCHOR, 0.5, 'cpu')
    assert not plain_only.any_form
    for k in range(2 * T):
        st = _state(k, scenes)
        a = mixed(dict(st))
        one = {kk: (vv[4:5] if torch.is_tensor(vv) and vv.dim() == 2 else vv) for kk, vv in st.items()}
        b = orig(dict(one))
        c = plain_only(dict(one))
        assert torch.equal(a[4:5], b) and torch.equal(b, c)


def test_drop_release_and_window_edges():
    # member starts 300 m before the drop: released at the drop like the original controller
    roles = [dict(lane=2, v_w=12.0, dmax=1.5, hold_s=60, onset_s=0, taper_in_s=10)]
    sc = [make_scene(DROP_S - 300.0)]
    _ctrl, acc = run_controller(roles, sc)
    ref = reference(roles[0], sc[0], T)
    assert all(abs(acc[0][t] - (NATIVE if ref[t] is None else ref[t])) < 1e-4 for t in range(T))
    assert ref[20] is None and ref[10] is not None


def test_member_specs_pass_form_and_reject_unknown():
    state_ids = {'a': 0, 'b': 1}
    acts = [dict(id='x', members=[dict(id='a', role='hold', params=dict(v_w=0.0, hold_s=10, dmax=1.5, amax=1.0,
                                                                         form=dict(start='gap', gap_dx_lo=0.0,
                                                                                   gap_dx_hi=40.0, gap_arm_s=30))),
                                  dict(id='b', role='hold', params=dict(v_w=18.0, hold_s=10, dmax=1.5, amax=1.0))])]
    out = WF.member_specs(acts, state_ids)
    assert out[0][0]['form']['start'] == 'gap' and 'form' not in out[0][1]
    acts[0]['members'][0]['params']['form'] = dict(bogus=1)
    with pytest.raises(P.WMContractError):
        WF.member_specs(acts, state_ids)


@pytest.mark.skipif(not os.environ.get('A75_CPU_SMOKE'), reason='set A75_CPU_SMOKE=1 for the real-WM CPU smoke')
def test_cpu_smoke_real_wm():
    """Frozen wm_s.pt on CPU, one D2800_p60 start state, K=1, H=20 s: plain plan identical through both APIs,
    forms run, base/plan draws common, patches restored."""
    import wm_fix_crn as CRN
    sys.path.insert(0, str(TREE))
    os.chdir(TREE)
    from tools.wzprop import strategy_search as SS
    from tools.wzprop import wm_backend as W
    import a75ss as A
    cfg = json.loads((ROOT / 'cfg' / 'A75B_SEARCH_CFG.json').read_text())
    A.install(cfg)
    try:
        ck = ROOT / 'ref' / 'ckpt' / 'ckpt' / 'wm_s.pt'
        h = P.load(str(ck), map_spec='maps/wz3to2/MAP_SPEC.json', device='cpu', max_rows=3)
        st = [m for m in json.loads(Path('data/regime_STATES.json').read_text())
              if m.get('valid') and m['cell'] == 'D2800_p60'][0]
        mi = W.moment_input(st)
        state = P.observe(h, mi)
        cavs = st['all_cavs_t0']
        found = None
        for zone in ('merge', 'near', 'full', 'far', 'mid'):
            gap = dict(name='g', trigger='lanes', zone=zone, spread_m=300.0, cooldown_s=60, slots=1, gate=None,
                       start_gate=None, phases=None,
                       roles=[dict(lane=1, rank=0, v_w=0.0, dmax=1.5, hold_s=10, onset_s=0, lc_target=None,
                                   start='gap', gap_dx_lo=-20.0, gap_dx_hi=60.0, gap_arm_s=20, dv_rel=6.0,
                                   gap_events=3)])
            if SS.activation(gap, cavs):
                found = gap
                break
        assert found is not None
        plain = dict(found, roles=[dict(lane=1, rank=0, v_w=14.0, dmax=1.5, hold_s=10, onset_s=0, lc_target=None)])
        acts = [W.action('base', 'base', None, 'base', [], 20),
                W.action('p', 'group', 's', 'p', SS.activation(plain, cavs), 20),
                W.action('g', 'group', 's', 'g', SS.activation(found, cavs), 20)]
        seed = mi['crn_seed']
        a = WF.evaluate(h, state, acts, K=1, H=20, seed=seed)
        b = CRN.evaluate(h, state, acts[:2], K=1, H=20, seed=seed)
        c = WF.evaluate(h, state, [acts[2], acts[0]], K=1, H=20, seed=seed)
        assert P.HoldSpeedController is WF._REAL['controller'] and P._member_specs is WF._REAL['member_specs']
        assert a['base'] == b['base'] and a['p'] == b['p']
        assert a['g'] == c['g'] and a['base'] == c['base']
        print('SMOKE', a)
    finally:
        A.uninstall()
