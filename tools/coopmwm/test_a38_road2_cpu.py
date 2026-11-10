"""Amendment 38 (CPU only, no SUMO): the wz2to1 binding (descriptor, map_scenarios profile, runner map / ranges,
nm_finaleval a38 settings), the road2 hold executor on fake rows, the runner hook / digest, the ops/a38_road2 driver
(spec building with a fake U0 binding, caps, ledger runtime, fail-closed WM stage) and the unchanged wz3to2 defaults.
wz2to1-process checks run in a subprocess (map_scenarios binds its registry at import from COOPMWM_MAP_SPEC)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.coopmwm import map_descriptor as md
from tools.coopmwm import map_scenarios as ms
from tools.coopmwm import nm_eval_runner as R
from tools.coopmwm import oracle_plan as O
from tools.coopmwm import road2

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO/'ops'))
import nm_finaleval as N  # noqa: E402
import a38_road2 as D  # noqa: E402

WZ2 = REPO/'assets/research/newmap_candidates/wz2to1/MAP_SPEC.json'
WZ3 = REPO/'assets/research/newmap_candidates/wz3to2/MAP_SPEC.json'
WZ3_DIGEST = '6fc6be652176948efdf594fdc48b564c2041d8ab3e0f3b15f6c7792cf02da20f'   # c4/a35-transfer b030aeef
CELL = 'D1200_p20'


def run_wz2(code, tmp_path):
    """Run ``code`` in a fresh wz2to1 process; it must print one JSON line last."""
    env = dict(os.environ, COOPMWM_MAP_SPEC=str(WZ2), PYTHONPATH=str(REPO))
    script = tmp_path/'probe.py'
    script.write_text('import json, sys\nsys.path.insert(0, %r)\n' % str(REPO/'ops') + code, encoding='utf-8')
    out = subprocess.run([sys.executable, str(script)], cwd=str(REPO), env=env, capture_output=True, text=True,
                         timeout=600)
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


# ------------------------------------------------------------------------------------------- descriptor
def test_wz2to1_descriptor_projection():
    d = md.load(WZ2)
    assert d.name == 'wz2to1' and d.network_file_sha256 == N.A38_NET_SHA256
    assert d.lane_table == {('up', 0): (2, 0.0), ('up', 1): (1, 0.0), ('wz', 0): (1, 2704.0)}
    assert d.bottleneck['through_lanes'] == [1] and d.bottleneck['closing_lane'] == 2
    assert d.bottleneck['drop_s_m'] == pytest.approx(75.44)
    assert d.lane_and_s('up_1', 100.0)[0] == 1 and d.lane_and_s('up_0', 100.0)[0] == 2
    ok, why = road2.wm_feasibility(d)
    assert not ok and 'WM lanes [0]' in why


def test_wz3to2_descriptor_unchanged():
    d = md.load(WZ3)
    assert d.digest == WZ3_DIGEST and road2.wm_feasibility(d)[0]


# ------------------------------------------------------------------------------------ wz3to2 defaults
def test_wz3to2_process_defaults_unchanged():
    assert ms.MAP == 'wz3to2' and ms.BOUND_PROFILE is None and ms.DEMANDS == (2400, 2800, 3200)
    assert R.MAP == 'wz3to2' and ms._N6_EVAL_RANGES == R.EVAL_HELDOUT_RANGES
    assert ms.seed_scope(996000) is None and ms.seed_scope(997000) is None
    assert N.MAP == 'wz3to2' and N.NET_SHA256 == 'b62778986347e8c21bca90d7ddcc8d9faeb288290a4dbdc3f53ee64ae8c21dfa'
    assert N.TRAIN_RANGE == (989200, 989599) and N.EXPOSURE_BASE is None and N.PANEL_TRAIN_TYPE is False
    assert N.RUNNER_CONTRACT['MAP'] == 'wz3to2'
    for name in ('p20', 'p20x', 'p20c', 'a33', 'a33t') + tuple(c[0] for c in N.A35_CELLS):
        s = N.SETTINGS[name]
        assert not {'MAP', 'NET_SHA256', 'TRAIN_RANGE', 'RUNNER_CONTRACT', 'EXPOSURE_BASE', 'PANEL_TRAIN_TYPE'} & set(s)
    assert N.SETTINGS['a35_D2400_p20']['EVAL_RANGE'] == (998000, 998099)


def test_a38_settings():
    for d in (1000, 1200, 1400):
        c, e = N.SETTINGS[f'a38_cal_D{d}'], N.SETTINGS[f'a38_D{d}_p20']
        for s in (c, e):
            assert (s['MAP'], s['NET_SHA256'], s['TRAIN_RANGE']) == ('wz2to1', N.A38_NET_SHA256, (996000, 996199))
            assert s['MAP_SPEC_REL'] == 'assets/research/newmap_candidates/wz2to1/MAP_SPEC.json'
            assert s['CELL'] == dict(demand=d, penetration=0.2) == s['PARITY_CELL']
            assert s['RUNNER_CONTRACT']['MAP'] == 'wz2to1' and 'wz3to2' not in s['RUNNER_CONTRACT']['episode']
            assert s['PANEL_SHA256'] is None
        assert (c['N_WORLDS'], c['EVAL_RANGE'], c['SEEDS_RESERVED'], c['PANEL_TRAIN_TYPE']) == (
            8, (996000, 996007), (996008, 996199), True)
        assert (e['N_WORLDS'], e['EVAL_RANGE'], e['SEEDS_RESERVED'], e['PANEL_TRAIN_TYPE']) == (
            96, (997000, 997099), (997096, 997099), False)
        assert f'a38_cal_D{d}' not in N.SETTINGS_PINS and f'a38_D{d}_p20' not in N.SETTINGS_PINS
    assert len({N.SETTINGS[n]['PANEL_FILE'] for n in D.SETTINGS}) == 6


def test_settings_rebinding(monkeypatch):
    for k, v in dict(N.SETTINGS['a38_D1200_p20'], SETTING='a38_D1200_p20').items():
        monkeypatch.setattr(N, k, v)
    assert N.MAP == 'wz2to1' and N.EVAL_RANGE == (997000, 997099) and N.NET_SHA256 == N.A38_NET_SHA256
    p20c = N.read(REPO/'ops'/'nm_eval_panel_wz3to2_p20c_D2800.json')     # a sealed wz3to2 panel
    with pytest.raises(N.Blocked, match='96 distinct wz2to1 worlds'):
        N.validate_panel(p20c)
    with pytest.raises(N.Blocked, match='not the staged probe network 7c7e9aa3'):   # wz3to2 MAP_SPEC, a38 setting
        N.build_panel(WZ3, 'x')


def test_caps():
    assert D.CAL_PLANNED_SIM_S == 28_800 and D.CAL_CAP_SIM_S == round(D.CAL_PLANNED_SIM_S*1.1)
    assert D.EVAL_PLANNED_SIM_S == 235_200 and D.EVAL_CAP_SIM_S == round(D.EVAL_PLANNED_SIM_S*1.1)
    assert (D.A38_CAL_PHASE, D.A38_EVAL_PHASE) == ('road2_a38_cal', 'road2_a38_eval')


# ------------------------------------------------------------------------------------------ road2 core
def crow(vid, wm_lane, s, v=25.0):
    return dict(vid=vid, s_m=float(s), v_mps=float(v), lane=wm_lane, is_cav=True, length_m=5.0)


def hrow(vid, wm_lane, s, v=25.0):
    return dict(vid=vid, s_m=float(s), v_mps=float(v), lane=wm_lane, is_cav=False, length_m=5.0)


def cmds(rows):
    return {r['vid']: 0.0 for r in rows if r['is_cav']}


@pytest.fixture
def wz2_oracle(monkeypatch):
    monkeypatch.setattr(O, '_pick_wall', O._pick_wall)          # restored after the test
    assert road2.install_oracle(O, map_name='wz2to1') is True
    return O


def test_pick_wall_lanes_rule():
    a, b = dict(id='a', lane=1, x=1000.0), dict(id='b', lane=0, x=1150.0)
    assert road2.pick_wall_lanes([a, b]) == {1: a, 0: b}
    assert road2.pick_wall_lanes([a, dict(b, x=1250.0)]) is None              # spread > 200 m
    assert road2.pick_wall_lanes([a, dict(id='c', lane=1, x=900.0)]) is None   # no CAV in lane 0
    c = dict(id='c', lane=0, x=1010.0)
    assert road2.pick_wall_lanes([a, b, c]) == {1: a, 0: c}                   # smallest spread


def test_install_oracle_noop_for_wz3to2(monkeypatch):
    monkeypatch.setattr(O, '_pick_wall', O._pick_wall)
    before = O._pick_wall
    assert road2.install_oracle(O, map_name='wz3to2') is False and O._pick_wall is before
    assert road2.install_oracle(O) is False and O._pick_wall is before        # test process map is not wz2to1
    assert O.oracle_digest()[:8] == '7409a7b4'


def test_hold_r2_lane1_member_only(wz2_oracle):
    core = road2.make_core('HOLD_R2', CELL)
    rows = [crow('c1', 1, -1000.0), crow('c2', 2, -990.0), hrow('h1', 1, -900.0)]
    accel, lanes = core.step(rows, cmds(rows), 300)
    assert accel == {'c1': -1.0} and lanes == {'c1': 1}                      # clip(20 - 25, -1, 1) on SUMO lane 1
    assert core.acts[0]['members'] == ['c1'] and core.acts[0]['family'] == 'duo12_v20_r2port'


def test_hold_duo01_both_members(wz2_oracle):
    core = road2.make_core('HOLD_R2_DUO01', CELL)
    rows = [crow('c1', 1, -1000.0, 19.0), crow('c2', 2, -990.0, 20.5)]
    accel, _ = core.step(rows, cmds(rows), 300)
    assert accel == {'c1': 1.0, 'c2': -0.5}                                  # v_w 20 law, dmax 1


def test_hold_needs_one_cav_per_lane_within_200m(wz2_oracle):
    core = road2.make_core('HOLD_R2', CELL)
    for rows in ([crow('c1', 1, -1000.0), crow('c2', 2, -750.0)],                 # 250 m apart
                 [crow('c1', 1, -1000.0), crow('c3', 1, -990.0)],                 # no CAV in the closing lane
                 [crow('c1', 1, -100.0), crow('c2', 2, -90.0)]):                  # outside the zone (x > 2296)
        assert core.step(rows, cmds(rows), 300) == ({}, {})
    assert core.acts == []


def test_unpatched_oracle_never_fires_on_two_lanes(monkeypatch):
    monkeypatch.setattr(O, '_pick_wall', O._pick_wall)
    core = road2.make_core('HOLD_R2', CELL)                                  # wz3to2/test process: no rebinding
    rows = [crow('c1', 1, -1000.0), crow('c2', 2, -990.0)]
    assert core.step(rows, cmds(rows), 300) == ({}, {})


def test_release_at_drop_and_cooldown(wz2_oracle):
    core = road2.make_core('HOLD_R2', CELL)
    rows = [crow('c1', 1, -1000.0), crow('c2', 2, -990.0)]
    assert core.step(rows, cmds(rows), 300)[0] == {'c1': -1.0}
    rows = [crow('c1', 1, 75.44), crow('c2', 2, -980.0)]
    assert core.step(rows, cmds(rows), 301) == ({}, {})                      # released at s >= 75.44
    assert core.acts[0]['release'] == {'c1': [1, 'at_drop']}
    fresh = [crow('d1', 1, -1000.0), crow('d2', 2, -995.0)]
    for t in (302, 360, 419):
        assert core.step(fresh, cmds(fresh), t) == ({}, {})                  # cooldown 120 s
    assert core.step(fresh, cmds(fresh), 420)[0] == {'d1': -1.0}
    assert len(core.acts) == 2


def test_hold_end_after_120s(wz2_oracle):
    core = road2.make_core('HOLD_R2', CELL)
    rows = [crow('c1', 1, -1500.0), crow('c2', 2, -1490.0)]
    core.step(rows, cmds(rows), 300)
    assert core.step(rows, cmds(rows), 419)[0] == {'c1': -1.0}
    core.step(rows, cmds(rows), 420)
    assert core.acts[0]['release']['c1'] == [120, 'hold_end']


def test_hook_wraps_and_never_commands_hdvs(wz2_oracle):
    class C:
        def apply(self, rows, commands, now):
            return {}
    hook = road2.make_hook('HOLD_R2', CELL)
    c = C()
    hook(c, None, None)
    rows = [crow('c1', 1, -1000.0), crow('c2', 2, -990.0), hrow('h1', 2, -995.0)]
    out = c.apply(rows, {'c1': 0.3, 'c2': 0.1}, 300)
    assert set(out) == {'c1'} and out['c1']['requested_raw'] == -1.0
    assert out['c1']['acceleration_residual'] == pytest.approx(-1.3) and out['c1']['oracle_plan'] == 'duo12_v20_r2port'
    r = hook.receipts()
    assert r['a38_arm'] == 'HOLD_R2' and r['activations'] == 1 and r['wall_lanes'] == [0, 1] and r['pick_wall_a38']


def test_digest_stable_across_install(monkeypatch):
    monkeypatch.setattr(O, '_pick_wall', O._pick_wall)
    d = road2.a38_digest()
    assert len(d) == 64 and d == road2.a38_digest()
    road2.install_oracle(O, map_name='wz2to1')
    assert road2.a38_digest() == d


# ---------------------------------------------------------------------------------------------- runner
def test_runner_a38_wiring():
    assert R.A38_ARMS == ('HOLD_R2', 'HOLD_R2_DUO01') and R.A38_EVAL_RANGES == ((996000, 996007, 996000),)
    ck = dict(kind='a38', a38_arm='HOLD_R2', a38_sha256=road2.a38_digest(), cell_key=CELL)
    with pytest.raises(ValueError, match='process map'):                     # wz3to2/test process refuses
        R.a38_hook(ck, dict(cell_key=CELL), 'HOLD_R2')
    with pytest.raises(ValueError, match='a38_arm'):
        R.a38_hook(dict(ck, a38_arm='HOLD_R2_DUO01'), dict(cell_key=CELL), 'HOLD_R2')


# ---------------------------------------------------------------------------------------------- driver
FAKE_U0 = dict(kind='u0_transferred_d9042cc5', label='u0', path='/x/exe.pt', sha256='e'*64, binding_sha256='b'*64,
               binding_path='/x/BOUND_U0.json', residual_path='/x/prerun_plant/A_checkpoint_000.pt',
               residual_sha256='r'*64)


def test_wm_stage_blocked(tmp_path):
    with pytest.raises(N.Blocked, match='WM-S cannot load wz2to1'):
        D.wm_search()
    with pytest.raises(N.Blocked, match='road2.wm_feasibility'):
        D.main(['wm-search', '--final-root', str(tmp_path)])


def test_wz2_commands_refuse_wz3to2_process(tmp_path):
    with pytest.raises(N.Blocked, match='COOPMWM_MAP_SPEC=.*wz2to1'):
        D.main(['dry', '--final-root', str(tmp_path), '--setting', 'a38_D1200_p20'])


def test_runtime_phase_cap(tmp_path):
    rt = D.A38Runtime(N.FakeRuntime(), tmp_path, D.A38_CAL_PHASE, 2400)
    rt.reserve(None, [dict(id='a', sim_s=1200)])
    (tmp_path/'CHARGES.jsonl').open('a').write(json.dumps(dict(id='z', phase='other', sim_s=99999, unix=0))+'\n')
    rt.check_capacity(1200)                                                  # other phases are not debited
    rt.reserve(None, [dict(id='b', sim_s=1200)])
    with pytest.raises(N.Blocked, match='cap'):
        rt.check_capacity(1)
    with pytest.raises(N.Blocked):
        rt.reserve(None, [dict(id='c', sim_s=1200)])
    assert [r['phase'] for r in N.fe.ledger_rows(tmp_path)] == ['road2_a38_cal', 'other', 'road2_a38_cal']


def test_check_a38_pins():
    spec = dict(job_id='a38_d1200_p20_nmfinal_hold_r2_0000', arm='HOLD_R2', world_id='wz2to1_997000',
                world=dict(seed=997000), checkpoint=dict(kind='a38', executor_sha256='e'*64, a38_sha256='a'*64),
                window=dict(start_s=0, end_s=1200), panel_sha256='p'*64, capture=None)
    rec = dict({k: spec[k] for k in N.IDENTITY}, status='graded', J_veh_s=1.0,
               safety=dict(collisions=0, teleports=0, hard_brakes=0, ttc_events=0),
               vehicle_log=dict(file=N.VEHICLE_LOG, sha256='f'*64, ticks=900, vehicles=10),
               executor_sha256_loaded='e'*64, a38_sha256_loaded='a'*64, a38=dict(a38_arm='HOLD_R2'))
    out = D.check_a38(rec, spec)
    assert out['a38']['a38_arm'] == 'HOLD_R2'
    for bad in (dict(a38_sha256_loaded='0'*64), dict(executor_sha256_loaded='0'*64), dict(a38=None)):
        with pytest.raises(N.Blocked):
            D.check_a38(dict(rec, **bad), spec)


WZ2_PROBE = r'''
from pathlib import Path
from tools.coopmwm import map_scenarios as ms, map_descriptor as md, nm_eval_runner as R, road2
import nm_finaleval as N
import a38_road2 as D
tmp = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(%(tmp)r)
out = dict(ms_map=ms.MAP, demands=list(ms.DEMANDS), train=list(ms.TRAIN_RANGE), heldout=list(ms.HELDOUT_RANGE),
           runner_map=R.MAP, md=md.current().name, scope={s: ms.seed_scope(s) for s in (996000, 996007, 996008,
           997000, 997099, 993100, 998000)}, repeat=ms.world(996003, 1200, 0.2)['repeat'],
           cells=[c['cell_key'] for c in ms.cells()])
for name in ('a38_cal_D1000', 'a38_cal_D1200', 'a38_cal_D1400', 'a38_D1000_p20', 'a38_D1200_p20', 'a38_D1400_p20'):
    N.SETTINGS[name]['PANEL_FILE'] = tmp/f'{name}.json'
D.u0_binding = lambda root: dict(%(u0)r)
N.apply_setting('a38_cal_D1000')
out['contract_map'] = N.RUNNER_CONTRACT['MAP']
out['runner_ok'] = N.Runtime(N.RUNNER).runner_status()['ok']
for name in D.CAL_SETTINGS:
    c = N.SETTINGS[name]['CELL']
    v = N.build_panel(N.default_spec_path(), '2026-01-01T00:00:00+00:00', demand=c['demand'], penetration=c['penetration'])
    N.publish(N.SETTINGS[name]['PANEL_FILE'], v)
waves = D.cal_waves(tmp/'root')
out['cal'] = [(w[0], len(w[2]), w[2][0]['job_id'], w[2][0]['world']['cell_key'], w[2][0]['world']['seed'],
               w[2][-1]['world']['seed']) for w in waves]
out['cal_specs_ok'] = all(R._check_spec(s)['seed'] == s['world']['seed'] for w in waves for s in w[2])
out['map_api_ranges'] = N.N1MapAPI().seed_ranges('wz2to1')
print(json.dumps(out))
'''

WZ2_EVAL = r'''
from pathlib import Path
from tools.coopmwm import map_scenarios as ms, nm_eval_runner as R, road2
import nm_finaleval as N
import a38_road2 as D
tmp = Path(%(tmp)r)
name = 'a38_D1200_p20'
N.SETTINGS[name]['PANEL_FILE'] = tmp/f'{name}.json'
D.u0_binding = lambda root: dict(%(u0)r)
N.apply_setting(name)
v = N.build_panel(N.default_spec_path(), '2026-01-01T00:00:00+00:00')
N.publish(N.PANEL_FILE, v)
res = D.main(['dry', '--final-root', str(tmp/'root'), '--setting', name, '--arms', 'U0,HOLD_R2,HOLD_R2_DUO01'])
panel = D.load_setting_panel(name)
h = D.arm_specs('HOLD_R2', panel, tmp/'root')
u = D.arm_specs('U0', panel, tmp/'root')
out = dict(dry={k: res[k] for k in ('U0', 'HOLD_R2', 'HOLD_R2_DUO01', 'map', 'cell', 'pinned')},
           n_worlds=len(panel['worlds']), seeds=panel['seeds_used'], reserved=panel['seeds_reserved_unused'],
           map=panel['map'], net=panel['map_binding']['net_sha256'], ck=h[0]['checkpoint'],
           u0_ck_kind=u[0]['checkpoint']['kind'], same_world=[a['world'] == b['world'] for a, b in zip(u, h)].count(True),
           check=R._check_spec(h[5])['seed'], digest=road2.a38_digest(), status=N.Runtime(N.RUNNER).runner_status(),
           exposure=panel['exposure']['disclosed'][:40], role_ok=ms.seed_scope(997050))
try:
    D.wm_search()
except N.Blocked as e:
    out['wm'] = str(e)
print(json.dumps(out))
'''


def test_wz2to1_process_calibration_blocks(tmp_path):
    out = run_wz2(WZ2_PROBE % dict(tmp=str(tmp_path), u0=FAKE_U0), tmp_path)
    assert (out['ms_map'], out['runner_map'], out['md'], out['contract_map']) == ('wz2to1',)*4
    assert out['demands'] == [1000, 1200, 1400] and out['cells'] == ['D1000_p20', 'D1200_p20', 'D1400_p20']
    assert out['train'] == [996000, 996199] and out['heldout'] == [997000, 997099] and out['repeat'] == 3
    assert out['scope'] == {'996000': 'heldout', '996007': 'heldout', '996008': 'train', '997000': 'heldout',
                            '997099': 'heldout', '993100': None, '998000': None}
    assert out['runner_ok'] is True and out['cal_specs_ok'] is True
    assert [c[:2] for c in out['cal']] == [['A38CAL_D1000_p20_U0', 8], ['A38CAL_D1200_p20_U0', 8],
                                           ['A38CAL_D1400_p20_U0', 8]]
    assert all(c[4:] == [996000, 996007] for c in out['cal'])
    assert out['cal'][1][2] == 'a38cal_d1200_p20_nmfinal_u0_0000' and out['cal'][1][3] == 'D1200_p20'
    assert out['map_api_ranges'] == dict(train=[996000, 996199], eval=[997000, 997099])


def test_wz2to1_process_eval_panel_and_dry(tmp_path):
    out = run_wz2(WZ2_EVAL % dict(tmp=str(tmp_path), u0=FAKE_U0), tmp_path)
    assert out['map'] == 'wz2to1' and out['net'] == N.A38_NET_SHA256 and out['n_worlds'] == 96
    assert out['seeds'] == [997000, 997095] and out['reserved'] == [997096, 997099]
    dry = out['dry']
    assert (dry['map'], dry['cell'], dry['pinned']) == ('wz2to1', 'D1200_p20', None)
    for arm in ('U0', 'HOLD_R2', 'HOLD_R2_DUO01'):
        assert dry[arm]['n'] == 96 and dry[arm]['seeds'] == [997000, 997095] and dry[arm]['sim_s'] == 115_200
    assert dry['HOLD_R2']['first'] == 'a38_d1200_p20_nmfinal_hold_r2_0000'
    ck = out['ck']
    assert (ck['kind'], ck['a38_arm'], ck['cell_key'], ck['arm'], ck['update']) == ('a38', 'HOLD_R2', CELL, 'A', 0)
    assert ck['a38_sha256'] == out['digest'] == road2.a38_digest()           # same digest in both processes
    assert out['u0_ck_kind'] == 'u0' and out['same_world'] == 96 and out['check'] == 997005
    assert out['status']['ok'] is True and out['exposure'].startswith('wz2to1 worlds are map-namespaced')
    assert out['role_ok'] == 'heldout' and 'WM-S cannot load wz2to1' in out['wm']
