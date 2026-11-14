"""G5 env binding into the frozen runtime that native episodes actually import."""
import difflib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools.coopmwm import g5_runtime

HERE = Path(__file__).resolve().parent
FROZEN = Path('/home/ubuntu/e3final')
FIXTURE = json.loads((HERE/'g5_env_change_lines.json').read_text())['lines']


def change(a, b):
    return sorted(l for l in difflib.unified_diff(a.splitlines(), b.splitlines(), lineterm='', n=0)
                  if not l.startswith(('+++', '---', '@@')))


def test_overlay_hash_is_bound():
    assert g5_runtime.normalized_sha(g5_runtime.OVERLAY) == g5_runtime.OVERLAY_SHA256


def test_accepted_g5_change_fixture_matches_repository_commits():
    try:
        pre = subprocess.run(['git', 'show', '6d8046e7:traffic_shepherd/c4/env.py'], cwd=HERE,
                             capture_output=True, text=True, check=True).stdout
        g5 = subprocess.run(['git', 'show', 'b903c6c2:traffic_shepherd/c4/env.py'], cwd=HERE,
                            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip('accepted G5 commits not available (deployed source has no git objects)')
    assert change(pre, g5) == FIXTURE


def test_install_is_inert_without_the_flag(monkeypatch):
    monkeypatch.delenv(g5_runtime.FLAG, raising=False)
    assert g5_runtime.install('/nonexistent') is None


@pytest.mark.skipif(not (FROZEN/g5_runtime.BASE_REL).is_file(), reason='frozen cpuV runtime only')
def test_overlay_is_exactly_the_accepted_g5_change_on_the_frozen_base():
    base = (FROZEN/g5_runtime.BASE_REL).read_text().replace('\r\n', '\n')
    overlay = g5_runtime.OVERLAY.read_text().replace('\r\n', '\n')
    assert g5_runtime.normalized_sha(FROZEN/g5_runtime.BASE_REL) == g5_runtime.BASE_SHA256
    assert change(base, overlay) == FIXTURE


SCRIPT = r'''
import os, sys, json
import numpy as np
from types import SimpleNamespace
sys.path[:0] = ['/home/ubuntu/e3final/runtime/src', '/home/ubuntu/e3final/runtime']  # leverprobe.setup order
os.environ['COOPMWM_G5_RUNTIME'] = '1'
from tools.coopmwm import g5_runtime
env = g5_runtime.install('/home/ubuntu/e3final')
import traffic_shepherd, traffic_shepherd.c4.env as imported
assert imported is env and traffic_shepherd.__file__.startswith('/home/ubuntu/e3final/runtime/')
import importlib.util, importlib.machinery
loader = importlib.machinery.SourceFileLoader('traffic_shepherd.c4._frozen_env', '/home/ubuntu/e3final/runtime/traffic_shepherd/c4/env.py')
spec = importlib.util.spec_from_file_location('traffic_shepherd.c4._frozen_env', loader.path, loader=loader)
frozen = importlib.util.module_from_spec(spec); loader.exec_module(frozen)
from tools.coopmwm import test_group_lane_cpu as t
class MP:
    @staticmethod
    def setattr(obj, name, value):
        setattr(obj, name, value)
accel = np.array([[0., 0., 0., np.nan]]); lanes = np.array([[0., -1., -1., -1.]])
g5 = t._real_env(MP, env_module=env)
_, info = g5.step(accel, lanes=lanes, lane_actors={'main'})
assert g5.api.changes == [('main', 1, 1.)] and g5.api.mode_writes == ['main']
assert info['lane_vehicles'] == {'main': dict(target=0, status='issued', completed=True)}
def run(module, **kw):
    e = t._real_env(MP, env_module=module)
    outs = [e.step(accel, lanes=lanes, **kw)[1] for _ in range(2)]
    return json.dumps(dict(info=outs, changes=e.api.changes, modes=e.api.modes, writes=e.api.mode_writes),
                      sort_keys=True, default=str)
assert run(env) == run(frozen), 'G5 overlay must be byte-identical to the frozen env without lane_actors'
print('G5_RUNTIME_OK')
'''


@pytest.mark.skipif(not (FROZEN/g5_runtime.BASE_REL).is_file(), reason='frozen cpuV runtime only')
def test_real_frozen_runtime_path_executes_g5_lane_actors_and_is_identical_when_absent():
    env = dict(os.environ)
    env.pop(g5_runtime.FLAG, None)
    result = subprocess.run([sys.executable, '-B', '-c', SCRIPT], capture_output=True, text=True, env=env,
                            cwd='/home/ubuntu', timeout=300)
    assert result.returncode == 0 and 'G5_RUNTIME_OK' in result.stdout, result.stderr[-3000:]
