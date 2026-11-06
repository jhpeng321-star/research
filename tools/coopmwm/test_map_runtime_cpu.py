"""N1 map runtime binding: original map untouched; wz3to2 bound onto the (G5) env, RSU from CAV reports.

Runtime installs run in fresh subprocesses (they rebind process-wide modules, exactly like the
spawned training workers). The checkout's traffic_shepherd/src.wms stand in for the frozen e3final
modules (same frame constants; the env is the committed G5 overlay pysrc = frozen env + G5 hunks).
The frozen cpuV runtime test runs only where /home/ubuntu/e3final exists. No SUMO is started.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tools.coopmwm import map_descriptor as md, map_scenarios as ms
from tools.coopmwm.map_observations import detected_mask, rsu_block

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO/'assets/research/newmap_candidates/wz3to2/MAP_SPEC.json'
FROZEN = Path('/home/ubuntu/e3final')


@pytest.fixture
def unbound():
    md._reset_for_tests()
    yield
    md._reset_for_tests()


def run_py(code, *, map_spec=True, extra_env=None):
    env = dict(os.environ)
    env.pop(md.ENV, None)
    env.pop('COOPMWM_G5_RUNTIME', None)
    if map_spec:
        env[md.ENV] = str(SPEC)
    env['PYTHONPATH'] = os.pathsep.join([str(REPO/'src'), str(REPO)])
    env.update(extra_env or {})
    out = subprocess.run([sys.executable, '-c', code], cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout+out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


PRELUDE = r'''
import importlib.machinery, importlib.util, json, sys, types
from pathlib import Path
# The committed G5 overlay (= frozen e3final env + accepted G5 hunks) as traffic_shepherd.c4.env,
# exactly how g5_runtime.install registers it (hash checks need the frozen cpuV base).
import traffic_shepherd.c4 as pkg
src = Path('tools/coopmwm/g5_runtime_env.pysrc')
loader = importlib.machinery.SourceFileLoader('traffic_shepherd.c4.env', str(src))
spec = importlib.util.spec_from_file_location('traffic_shepherd.c4.env', str(src), loader=loader)
env = importlib.util.module_from_spec(spec); sys.modules['traffic_shepherd.c4.env'] = env
loader.exec_module(env); pkg.env = env
lp = types.ModuleType('tools.leverprobe.backend')
def _detector_rows(e): return 'rsu_detector_rows'
lp.observations = _detector_rows
pkg_lp = types.ModuleType('tools.leverprobe'); pkg_lp.__path__ = []; pkg_lp.backend = lp
sys.modules['tools.leverprobe'] = pkg_lp; sys.modules['tools.leverprobe.backend'] = lp
from tools.coopmwm import map_runtime, map_scenarios as ms, map_descriptor as md
'''


def test_original_install_is_a_no_op_and_imports_nothing():
    out = run_py(r'''
import json, sys
from tools.coopmwm import map_runtime, accel_protocol
before = set(sys.modules)
world = dict(seed=986000, cell=0, repeat=0, penetration=.05, demand=1900)
r = map_runtime.install(dict(world=world, role='train'))
new = sorted(m for m in set(sys.modules)-before)
print(json.dumps(dict(result=r, new=new, rec=map_runtime.installed_record())))
''', map_spec=False)
    assert out['result'] is None and out['rec'] == {}
    assert not [m for m in out['new'] if m.startswith(('traffic_shepherd', 'wms', 'tools.leverprobe'))]


def test_map_world_and_process_map_must_agree(unbound, monkeypatch):
    w = ms.world(989200, 2800, 0.4)
    monkeypatch.delenv(md.ENV, raising=False)
    with pytest.raises(ValueError):
        md.for_world(w)                                    # map world in a merge-map process
    md._reset_for_tests()
    monkeypatch.setenv(md.ENV, str(SPEC))
    assert md.for_world(w).name == 'wz3to2'
    with pytest.raises(ValueError):
        md.for_world(dict(seed=986000, cell=0, repeat=0, penetration=.05, demand=1900))
    from tools.coopmwm import accel_protocol
    accel_protocol.guard(w, 'train')                       # registered map world passes the guard
    with pytest.raises(ValueError):
        accel_protocol.guard(w, 'final_test')
    with pytest.raises(ValueError):
        accel_protocol.guard(dict(w, seed=964001), 'train')


def test_wz3to2_install_binds_projection_scenario_lanes_slots_and_rsu(tmp_path):
    out = run_py(PRELUDE+r'''
import numpy as np
from traffic_shepherd.c4 import geometry, state
world = ms.world(989201, 3200, 0.6)
desc = map_runtime.install(dict(world=world, role='train'))
assert map_runtime.install(dict(world=world, role='train')) is desc       # idempotent
res = {}
res['lane_and_s'] = [geometry.lane_and_s(l, p) for l, p in (('up_0', 2650.), ('up_1', 0.), ('up_2', 2696.), ('wz_0', 10.), ('wz_1', 1296.))]
res['env_table'] = env.LANE_TABLE is geometry.LANE_TABLE and env.LANE_TABLE == desc.lane_table
fb = state.FrameBuilder(8, {'c1'})
f = fb.build(10., {'c1': dict(lane_id='up_0', lane_position_m=2650., speed_mps=5., acceleration_mps2=0.),
                   'h1': dict(lane_id=':n1_0_0', lane_position_m=1., speed_mps=9., acceleration_mps2=0.)})
res['frame'] = [[int(f.lane[i]), float(f.s[i]), bool(f.junction[i])] for i in range(2)]
scene, trips, manifest = env.build_scenario(Path(sys.argv[1] if len(sys.argv) > 1 else 'unused'), 989201, 3200,
    penetration=0.6, role='exposed', config=None, cav_vtype='cav_typical', population='textbook') if False else (None, None, None)
res['observation_map'] = env.observation_map(None, None)
res['edges'] = env.boundary_edges(None)
ok = True
try: env.check_case(964001, 3200); ok = False
except ValueError: pass
env.check_case(989201, 3200); env.check_case(989650, 2400)
res['check_case'] = ok
e = env.CavSumoEnv(k_slots=192, controller='direct', ttc=True, physics_log=False, population='textbook', sumo_binary='sumo-not-started')
res['k'] = e.k
try: env.CavSumoEnv(k_slots=100, sumo_binary='x'); res['k_refused'] = False
except ValueError: res['k_refused'] = True
# G5 lane commands use the map table: closing lane (WM 2) CAV -> WM 1 on edge 'up' = SUMO index 1.
class Lane:
    def getLength(self, lane_id): return 2696.0
class Veh:
    calls = []
    def changeLane(self, vid, idx, dur): Veh.calls.append([vid, idx, dur])
class Api: lane, vehicle = Lane(), Veh()
e.api = Api(); e._lane_lengths = {}
e._table = {'c1': dict(lane_id='up_0', lane_position_m=1000., speed_mps=20., road_id='up'),
            'c2': dict(lane_id='up_1', lane_position_m=1000., speed_mps=20., road_id='up')}
issued, counts = e._issue_lane_commands({'c1': 1, 'c2': 2}, {'c1', 'c2'})
res['lane_cmd'] = [Veh.calls, dict(counts)]
import tools.leverprobe.backend as lp, tools.coopnative.plant as cnp
from wms.control import rsu_field
res['rows_patched'] = lp.observations.__name__ == '<lambda>' and cnp.observations.__name__ == '<lambda>'
res['field_patched'] = getattr(rsu_field.build_rsu_field, 'map_name', None)
res['record'] = map_runtime.installed_record()
try:
    md.bind(md.ORIGINAL); res['rebind'] = False
except ValueError:
    res['rebind'] = True
print(json.dumps(res))
''')
    assert out['lane_and_s'][0][0] == 2 and abs(out['lane_and_s'][0][1]-29.44) < 1e-9
    assert out['lane_and_s'][1] == [1, -2620.56] and abs(out['lane_and_s'][2][1]-75.44) < 1e-9
    assert out['lane_and_s'][3][0] == 1 and abs(out['lane_and_s'][3][1]-93.44) < 1e-9 and out['lane_and_s'][4][0] == 0
    assert out['env_table'] and out['frame'][0][:1] == [2] and out['frame'][1][2] is True
    assert out['observation_map']['up_0'] == 0.0 and out['observation_map']['wz_1'] == -1304.0
    assert out['edges'] == {'mainline_entry': 'up', 'mainline_exit': 'wz', 'ramp_entry': '__no_ramp_entry__'}
    assert out['check_case'] and out['k'] == 512 and out['k_refused']
    assert out['lane_cmd'][0] == [['c1', 1, 1.0]] and out['lane_cmd'][1] == {'issued': 1, 'refused_not_allowed': 1}
    assert out['rows_patched'] and out['field_patched'] == 'wz3to2' and out['rebind']
    rec = out['record']
    assert 'traffic_shepherd.c4.env' in rec['lane_table_rebound'] and 'traffic_shepherd.c4.geometry' in rec['lane_table_rebound']
    assert rec['g5_overlay'] == '5aac1769679964e6df6950cfaf115902c2787ac34d5a20622d8bf67bc7a8bcde' or rec['g5_overlay'] is None


def test_wz3to2_env_builder_after_install_writes_probe_identical_scenario(tmp_path):
    out = run_py(PRELUDE+r'''
world = ms.world(989202, 2400, 0.2)
map_runtime.install(dict(world=world, role='train'))
root = Path(sys.argv[-1])
''' + f'''
root = Path({str(tmp_path/'env')!r})
scene, trips, manifest = env.build_scenario(root, 989202, 2400, penetration=0.2, role='exposed', config=None,
                                            cav_vtype='cav_typical', population='textbook')
print(json.dumps(dict(n=len(trips), sha=manifest['routes_sha256'], ep=manifest['episode_seconds'])))
''')
    from tools.coopmwm.test_map_scenarios_cpu import _load, _spec, MAP_DIR
    probe = _load('probe/probe_core.py', 'n1_probe_rt')
    probe.write_scenario(tmp_path/'probe', MAP_DIR, _spec(), 'wz3to2', 989202, {'main': 2400}, 0.2)
    for name in ('requests.rou.xml', 'scenario.sumocfg'):
        assert (tmp_path/'env'/name).read_bytes() == (tmp_path/'probe'/name).read_bytes()
    assert out['ep'] == 1200 and out['n'] > 500


def test_install_refuses_a_frame_mismatch():
    run_py(PRELUDE+r'''
from wms.control import contract
contract.MERGE_S1_M = 80.0
try:
    map_runtime.install(dict(world=ms.world(989200, 2800, 0.4), role='train')); print(json.dumps(dict(ok=False)))
except ValueError as e:
    assert 'frame' in str(e); print(json.dumps(dict(ok=True)))
''')


# ------------------------------------------------------------------ RSU from CAV reports only
class _Frame:
    def __init__(self, rows):
        k = len(rows)
        self.time_s = 500.
        self.s = np.array([r[1] for r in rows], np.float32)
        self.lane = np.array([r[2] for r in rows], np.int8)
        self.lane_carry = self.lane.copy()
        self.v = np.array([r[3] for r in rows], np.float32)
        self.present = np.ones(k, bool)
        self.is_cav = np.array([r[4] for r in rows], bool)
        self.junction = np.zeros(k, bool)
        self.vid_index = np.arange(k, dtype=np.int32)


class _Env:
    def __init__(self, rows):
        self.frame = _Frame(rows)
        self.builder = type('B', (), {'vehicle_ids': [r[0] for r in rows]})()


ROWS = [('cav.a', 0., 0, 20., True), ('hdv.near_same', 60., 0, 18., False), ('hdv.near_adj', -80., 1, 17., False),
        ('hdv.two_lanes', 30., 2, 0., False),       # two lanes from every CAV: not detected
        ('hdv.far_in_rsu', -500., 1, 3., False),    # inside the frozen RSU field zone, no CAV near
        ('hdv.down', 150., 0, 25., False),          # downstream of the drop, near cav.b
        ('cav.b', 120., 0, 24., True), ('hdv.unseen_block', -90., 2, 1., False)]
UNSEEN = ('hdv.two_lanes', 'hdv.far_in_rsu', 'hdv.unseen_block')


def test_wz3to2_rows_and_rsu_block_use_only_cav_reports(unbound, monkeypatch):
    monkeypatch.setenv(md.ENV, str(SPEC))
    desc = md.current()
    from tools.coopmwm.map_observations import cav_report_rows, report_summary
    rows = cav_report_rows(_Env(ROWS), desc)
    vids = [r['vid'] for r in rows]
    assert not set(UNSEEN) & set(vids)
    assert {'cav.a', 'cav.b', 'hdv.near_same', 'hdv.near_adj', 'hdv.down'} == set(vids)
    assert all(set(r) == {'vid', 's_m', 'v_mps', 'lane', 'is_cav'} for r in rows)
    from tools.coopnative.policy import encode
    enc = encode([dict(r, length_m=5.) for r in rows])
    assert enc['rsu'] == rsu_block([dict(r, length_m=5.) for r in rows], desc.rsu.block_zone_s_m)
    # lane 0 block covers the approach AND 100 m downstream of the drop (s <= 175.44): hdv.down counts.
    assert enc['rsu'][0] == 4/10 and enc['rsu'][3] == 2/10 and enc['rsu'][8:] == [0., 0., 0., 0.]
    # changing an undetected HDV never changes the observation (CAV reports only)
    moved = [r if r[0] != 'hdv.far_in_rsu' else (r[0], -450., 1, 30., False) for r in ROWS]
    assert cav_report_rows(_Env(moved), desc) == rows
    summ = report_summary(_Env(ROWS).frame, desc)
    assert summ['source'] == 'cav_reports' and summ['receipt_age_s'] == 0.0 and summ['reporting_cavs'] == 2
    cov = {lane['lane']: lane for lane in summ['lanes']}
    assert 0. < cov[0]['coverage'] <= 1. and 0. < cov[1]['coverage'] <= 1. and not cov[0]['missing']
    assert cov[2]['coverage'] == 0. and cov[2]['missing'] and cov[2]['reporters'] == 0   # no CAV reaches lane 2


def test_wz3to2_frozen_mappo_field_counts_only_cav_detected_vehicles(unbound, monkeypatch):
    import torch
    from wms.control.rsu_field import build_rsu_field
    from tools.coopmwm.map_observations import masked_field_builder
    monkeypatch.setenv(md.ENV, str(SPEC))
    desc = md.current()
    f = _Frame(ROWS)
    state = {'s_m': torch.tensor(f.s)[None], 'v_mps': torch.tensor(f.v)[None], 'lane': torch.tensor(f.lane.astype(np.int64))[None],
             'present': torch.tensor(f.present)[None], 'is_cav': torch.tensor(f.is_cav)[None]}
    masked = masked_field_builder(build_rsu_field, desc)(state)
    det = detected_mask(f.s, f.lane, f.present, f.is_cav, 100., 1)
    reference = build_rsu_field({**state, 'present': torch.tensor(det)[None]})
    assert torch.equal(masked, reference) and not torch.equal(masked, build_rsu_field(state))
    assert masked.shape == build_rsu_field(state).shape          # frozen cell layout kept
    # all vehicles detected -> identical to the frozen field
    near = [r for r in ROWS if r[0] not in UNSEEN]
    g = _Frame(near)
    st = {'s_m': torch.tensor(g.s)[None], 'v_mps': torch.tensor(g.v)[None], 'lane': torch.tensor(g.lane.astype(np.int64))[None],
          'present': torch.tensor(g.present)[None], 'is_cav': torch.tensor(g.is_cav)[None]}
    assert torch.equal(masked_field_builder(build_rsu_field, desc)(st), build_rsu_field(st))


def test_original_rsu_block_is_the_frozen_detector_layout(unbound, monkeypatch):
    monkeypatch.delenv(md.ENV, raising=False)
    d = md.current()
    assert d is md.ORIGINAL and d.rsu.source == 'rsu_detector'
    rows = [dict(vid=r[0], s_m=r[1], v_mps=r[3], lane=r[2], is_cav=r[4], length_m=5.) for r in ROWS]
    from tools.coopnative.policy import encode
    assert encode(rows)['rsu'] == rsu_block(rows, ((-100., 75.44),)*3)


@pytest.mark.skipif(not (FROZEN/'runtime/traffic_shepherd/c4/env.py').is_file(), reason='frozen cpuV runtime only')
def test_real_frozen_runtime_with_g5_binds_wz3to2():
    out = run_py(r'''
import json, sys
sys.path[:0] = ['/home/ubuntu/e3final/runtime/src', '/home/ubuntu/e3final/runtime']
from tools.coopmwm import g5_runtime, map_runtime, map_scenarios as ms
env = g5_runtime.install('/home/ubuntu/e3final')
import types
lp = types.ModuleType('tools.leverprobe.backend'); lp.observations = None
sys.modules.setdefault('tools.leverprobe', types.ModuleType('tools.leverprobe')); sys.modules['tools.leverprobe.backend'] = lp
d = map_runtime.install(dict(world=ms.world(989200, 2800, 0.4), role='train'))
import traffic_shepherd.c4.geometry as g
print(json.dumps(dict(file=g.__file__, s=g.lane_and_s('up_0', 2650.)[1], k=env.CavSumoEnv(k_slots=192, sumo_binary='x').k)))
''', extra_env={'COOPMWM_G5_RUNTIME': '1'})
    assert out['file'].startswith('/home/ubuntu/e3final/') and abs(out['s']-29.44) < 1e-9 and out['k'] == 512
