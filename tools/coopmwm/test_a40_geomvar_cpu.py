"""Amendment 40 (CPU only, no SUMO): the wz3to2 geometry variant wz3to2v80 (work zone posted 22.22 m/s) loads through
the UNCHANGED projection (same lane table / anchor / bottleneck as wz3to2), the wz3to2 bytes and descriptor digest are
unchanged, the builder reproduces the committed sources, and a wz3to2v80 process binds the A40 registry / runner
ranges while a default process keeps the wz3to2 values. wz3to2v80-process checks run in a subprocess (map_scenarios
binds its registry at import from COOPMWM_MAP_SPEC)."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.coopmwm import build_geomvar_a40 as B
from tools.coopmwm import map_descriptor as md
from tools.coopmwm import map_scenarios as ms
from tools.coopmwm import nm_eval_runner as R
from tools.coopmwm import oracle_plan as O

ROOT = Path(__file__).resolve().parents[2]          # repo (laptop) or <src>/pytools (host overlay)
CAND = ROOT/'assets/research/newmap_candidates'
WZ3, WZV = CAND/'wz3to2/MAP_SPEC.json', CAND/'wz3to2v80/MAP_SPEC.json'
WZ3_DIGEST = '6fc6be652176948efdf594fdc48b564c2041d8ab3e0f3b15f6c7792cf02da20f'   # c4/a35-transfer b030aeef
WZ3_NET_FILE = 'b62778986347e8c21bca90d7ddcc8d9faeb288290a4dbdc3f53ee64ae8c21dfa'
WZV_NET_FILE = '1fed6f59cfcfa63d2e30f1344eac5e59a1597eca77c7d9edbae3e0bc08161f03'


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_wz3to2_bytes_and_descriptor_unchanged():
    assert _sha(CAND/'wz3to2/wz3to2.net.xml') == WZ3_NET_FILE
    d = md.load(WZ3)
    assert d.digest == WZ3_DIGEST and md.REGISTERED['wz3to2']['network_file_sha256'] == WZ3_NET_FILE


def test_variant_projection_identical_to_wz3to2():
    a, b = md.load(WZ3), md.load(WZV)
    assert b.name == 'wz3to2v80' and b.network_file_sha256 == WZV_NET_FILE == _sha(CAND/'wz3to2v80/wz3to2v80.net.xml')
    assert a.lane_table == b.lane_table and a.here_offsets == b.here_offsets
    assert a.bottleneck == b.bottleneck and a.bottleneck['drop_x_m'] == O.G.X_B == 2696.0
    assert a.lane_end_distance == b.lane_end_distance and a.rsu == b.rsu and a.frame == b.frame
    assert a.edges == b.edges and a.k_slots == b.k_slots and a.routes == b.routes


def test_variant_network_differs_only_in_workzone_speed():
    na = md.parse_network(CAND/'wz3to2/wz3to2.net.xml')
    nb = md.parse_network(CAND/'wz3to2v80/wz3to2v80.net.xml')
    assert na['connections'] == nb['connections'] and set(na['lanes']) == set(nb['lanes'])
    diff = {k for k in na['lanes'] if na['lanes'][k] != nb['lanes'][k]}
    assert diff == {'wz_0', 'wz_1'}
    for k in diff:
        x, y = dict(na['lanes'][k]), dict(nb['lanes'][k])
        assert (x.pop('speed'), y.pop('speed')) == (31.29, 22.22) and x == y
    for k in na['internal']:                                 # junction lanes: netconvert speed only
        x, y = dict(na['internal'][k]), dict(nb['internal'][k])
        x.pop('speed'), y.pop('speed')
        assert x == y
    assert all(v['speed'] == 31.29 for k, v in nb['lanes'].items() if v['edge'] == 'up')


def test_builder_reproduces_committed_sources():
    for name, text in B.sources().items():
        assert (CAND/'wz3to2v80'/name).read_text(encoding='utf-8') == text
    assert json.loads(WZV.read_text(encoding='utf-8')) == json.loads(json.dumps(B.meta()))
    base = {f'wz3to2.{k}': (CAND/f'wz3to2/wz3to2.{k}').read_text(encoding='utf-8') for k in ('nod.xml', 'con.xml', 'routes.xml')}
    for k, text in base.items():                             # identical layout inputs (only the name differs)
        assert B.sources()[k.replace('wz3to2', 'wz3to2v80')] == text
    edg3 = (CAND/'wz3to2/wz3to2.edg.xml').read_text(encoding='utf-8')
    assert B.sources()['wz3to2v80.edg.xml'] == edg3.replace('id="wz" from="n1" to="n2" numLanes="2" speed="31.29"',
                                                            'id="wz" from="n1" to="n2" numLanes="2" speed="22.22"')


def test_default_process_is_wz3to2():
    if os.environ.get(md.ENV):
        return                                               # a bound process: the subprocess test covers it
    assert ms.MAP == 'wz3to2' and R.MAP == 'wz3to2' and ms.DEMANDS == (2400, 2800, 3200)
    assert R.RANGES_BY_MAP['wz3to2v80'] == ((994200, 994207, 994200),) and 'wz3to2' not in R.RANGES_BY_MAP
    assert ms.MAP_PROFILES['wz3to2v80']['TRAIN_RANGE'] == (994200, 994399)


PROBE = r'''
import json
from tools.coopmwm import map_scenarios as ms, map_descriptor as md, nm_eval_runner as R
before = {s: ms.seed_scope(s) for s in (994200, 994207, 994208, 994399, 994000, 994099, 993100, 996000, 989600)}
R.install_eval_ranges(ms)
after = {s: ms.seed_scope(s) for s in (994200, 994207, 994208, 994399, 994000, 994099, 993100)}
print(json.dumps(dict(ms_map=ms.MAP, runner_map=R.MAP, md=md.current().name, demands=list(ms.DEMANDS),
                      shares=list(ms.SHARES), cells=[c['cell_key'] for c in ms.cells()], before=before, after=after,
                      repeat=ms.world(994203, 2800, 0.2)['repeat'], heldout_repeat=ms.world(994005, 2800, 0.2)['repeat'],
                      world_map=ms.world(994000, 2800, 0.2)['map'])))
'''


def test_variant_process_registry_and_ranges(tmp_path):
    env = dict(os.environ, COOPMWM_MAP_SPEC=str(WZV),
               PYTHONPATH=os.pathsep.join([str(ROOT)]+[p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]))
    script = tmp_path/'probe.py'
    script.write_text(PROBE, encoding='utf-8')
    out = subprocess.run([sys.executable, str(script)], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    v = json.loads(out.stdout.strip().splitlines()[-1])
    assert (v['ms_map'], v['runner_map'], v['md'], v['world_map']) == ('wz3to2v80',)*4
    assert v['demands'] == [2400, 2800, 3200] and v['shares'] == [0.2]
    assert v['cells'] == ['D2400_p20', 'D2800_p20', 'D3200_p20']
    # importing the runner already installs its process-map ranges (as on wz2to1): calibration block = scope label
    assert v['before'] == {'994200': 'heldout', '994207': 'heldout', '994208': 'train', '994399': 'train',
                           '994000': 'heldout', '994099': 'heldout', '993100': None, '996000': None, '989600': None}
    assert v['after'] == {'994200': 'heldout', '994207': 'heldout', '994208': 'train', '994399': 'train',
                          '994000': 'heldout', '994099': 'heldout', '993100': None}
    assert v['repeat'] == 3 and v['heldout_repeat'] == 5
