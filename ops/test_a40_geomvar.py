"""Amendment 40 driver (CPU only, no SUMO): nm_finaleval a40 settings / pins / caps, the registered demand pick rule,
the check delegation to the a35 path, and (when a binding mirror root is available: A40_MIRROR or the laptop
E:/agent-work/a40/local_root) a wz3to2v80-process dry run building U0 / LEARNED_TRIGGER / DUO_HOLD specs with the
amendment-35 bank_cell_key and the unchanged executor digests."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nm_finaleval as N  # noqa: E402
import a40_geomvar as D  # noqa: E402
from tools.coopmwm import map_descriptor as md  # noqa: E402

TOOLS_ROOT = Path(md.__file__).resolve().parents[2]        # repo (laptop) or <src>/pytools (host)
WZV = TOOLS_ROOT/'assets/research/newmap_candidates/wz3to2v80/MAP_SPEC.json'
MIRROR = Path(os.environ.get('A40_MIRROR', 'E:/agent-work/a40/local_root'))


def test_settings_and_pins():
    assert N.A40_CAL_SETTINGS == ('a40_cal_D2400', 'a40_cal_D2800', 'a40_cal_D3200')
    assert N.A40_EVAL_SETTINGS == ('a40_D2400_p20', 'a40_D2800_p20', 'a40_D3200_p20')
    assert set(N.A40_EVAL_SETTINGS) <= set(N.BANK_SETTINGS) and not set(N.A40_CAL_SETTINGS) & set(N.BANK_SETTINGS)
    for name in N.A40_CAL_SETTINGS+N.A40_EVAL_SETTINGS:
        s = N.SETTINGS[name]
        assert s['MAP'] == 'wz3to2v80' and s['NET_SHA256'] == N.A40_NET_SHA256 and s['RUNNER_CONTRACT']['MAP'] == 'wz3to2v80'
        assert 'wz3to2v80 world' in s['RUNNER_CONTRACT']['episode'] and s['CELL']['penetration'] == 0.2
        assert len(N.SETTINGS_PINS[name]['PANEL_SHA256']) == 64 and Path(s['PANEL_FILE']).is_file()
    assert N.SETTINGS['a40_D2800_p20']['N_WORLDS'] == 96 and N.SETTINGS['a40_D2800_p20']['SEEDS_RESERVED'] == (994096, 994099)
    assert N.SETTINGS['a40_cal_D2800']['EVAL_RANGE'] == (994200, 994207) and N.SETTINGS['a40_cal_D2800']['PANEL_TRAIN_TYPE']
    assert N.SETTINGS['p20c'].get('MAP', N.MAP) == 'wz3to2' and 'MAP' not in N.SETTINGS['a35_D2400_p20']   # untouched


def test_panels_load_under_pins():
    for name in N.A40_CAL_SETTINGS+N.A40_EVAL_SETTINGS:
        p = N.read(N.SETTINGS[name]['PANEL_FILE'])
        N.verify(p, 'panel_sha256', name)
        assert p['panel_sha256'] == N.SETTINGS_PINS[name]['PANEL_SHA256'] and p['map'] == 'wz3to2v80'
        seeds = [w['world']['seed'] for w in p['worlds']]
        lo, hi = N.SETTINGS[name]['EVAL_RANGE']
        assert all(lo <= s <= hi for s in seeds) and len(set(seeds)) == N.SETTINGS[name]['N_WORLDS']
        assert p['map_binding']['net_sha256'] == N.A40_NET_SHA256
        assert p['contract']['bootstrap'] == dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')


def test_caps():
    assert D.CAL_PLANNED_SIM_S == 28_800 and D.CAL_CAP_SIM_S == 31_680
    assert D.EVAL_PLANNED_SIM_S == 3*96*1200 + 4*1200 == 350_400 and D.EVAL_CAP_SIM_S == 385_440
    assert D.ARMS == ('U0', 'LEARNED_TRIGGER', 'DUO_HOLD') and D.BANK_CELL == 'D2800_p20'


def _doc(cells):
    return dict(cells={f'a40_cal_D{d}': dict(cell=f'D{d}_p20', worlds=8, worlds_published=8,
                                             mean_approach_speed_mps=v, safety_totals=dict(collisions=c, teleports=0))
                       for d, (v, c) in cells.items()})


def test_pick_rule():
    assert D.pick_demand(_doc({2400: (28, 0), 2800: (17, 0), 3200: (9, 0)})) == 'a40_D2800_p20'
    assert D.pick_demand(_doc({2400: (20, 0), 2800: (4, 0), 3200: (2, 0)})) == 'a40_D2400_p20'      # D2800 gridlocked
    assert D.pick_demand(_doc({2400: (30, 0), 2800: (26, 0), 3200: (20, 0)})) == 'a40_D3200_p20'    # 2800 uncongested
    assert D.pick_demand(_doc({2400: (20, 1), 2800: (17, 1), 3200: (9, 1)})) is None                 # collisions
    assert D.pick_demand(_doc({2400: (30, 0), 2800: (30, 0), 3200: (30, 0)})) is None
    bad = _doc({2400: (20, 0), 2800: (17, 0), 3200: (9, 0)})
    bad['cells']['a40_cal_D2800']['worlds_published'] = 7
    with pytest.raises(N.Blocked):
        D.pick_demand(bad)


def test_check_delegates_to_a35_path():
    # CHECKS is process-global (other drivers imported later in the same pytest run may rebind it): check a fresh
    # process that imports only this driver, as on the host.
    code = ('import sys; sys.path[:0] = [%r, %r]; import nm_finaleval as N, a40_geomvar as D; '
            'print(N.CHECKS["episode"] is D.check_a40)') % (str(HERE), str(HERE.parent))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(TOOLS_ROOT)]+[p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]))
    out = subprocess.run([sys.executable, '-c', code], cwd=str(HERE.parent), env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().splitlines()[-1] == 'True', out.stderr[-2000:]


DRY = r'''
import json, sys
sys.path.insert(0, %(ops)r)
import a40_geomvar as D
out = D.main(['dry', '--final-root', %(root)r, '--setting', sys.argv[1]])
print(json.dumps(out))
'''


@pytest.mark.skipif(not (MIRROR/'BOUND_U0.json').is_file() or not (MIRROR/'a32_banks').is_dir(),
                    reason='no binding mirror root (A40_MIRROR)')
def test_variant_dry_specs(tmp_path):
    env = dict(os.environ, COOPMWM_MAP_SPEC=str(WZV),
               PYTHONPATH=os.pathsep.join([str(TOOLS_ROOT)]+[p for p in os.environ.get('PYTHONPATH', '').split(os.pathsep) if p]))
    script = tmp_path/'dry.py'
    script.write_text(DRY % dict(ops=str(HERE), root=str(MIRROR)), encoding='utf-8')
    v = {}
    for setting in ('a40_D2800_p20', 'a40_cal_D2800'):          # one process = one setting
        out = subprocess.run([sys.executable, str(script), setting], cwd=str(HERE.parent), env=env, capture_output=True,
                             text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-3000:]
        v[setting] = json.loads(out.stdout.strip().splitlines()[-1])
    ev = v['a40_D2800_p20']
    assert ev['map'] == 'wz3to2v80' and ev['pinned'] == ev['panel_sha256'] == N.SETTINGS_PINS['a40_D2800_p20']['PANEL_SHA256']
    assert ev['learned_trigger_sha256'].startswith('41fc8d1a') and ev['a32_sha256'].startswith('1c51f22c')
    for arm in D.ARMS:
        assert ev[arm]['n'] == 96 and ev[arm]['seeds'] == [994000, 994095] and ev[arm]['sim_s'] == 115_200
        assert ev[arm]['bank_cell_key'] == (None if arm == 'U0' else 'D2800_p20')
    cal = v['a40_cal_D2800']
    assert cal['U0']['n'] == 8 and cal['U0']['seeds'] == [994200, 994207] and 'LEARNED_TRIGGER' not in cal
