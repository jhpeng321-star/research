"""Read-only process/budget/source checks; zero-charge prospective config parse."""
import json
import os
from pathlib import Path
import sys
import time

R = Path('/root/autodl-tmp/c4/restart20261026')
ROOT = R / 'runtime_c1'
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'scripts/c4')]
os.chdir(ROOT)
import loop14 as loop
from wms.control.train_cli import _read_config

cfg = loop.load_config(R / 'arm/CONFIG.json')
os.environ['C4_ALLOWED_ROOTS'] = os.pathsep.join(cfg['policy']['allowed_roots'])
state = loop.load_state(cfg)
assert loop.check_registration(cfg)['budget']['sim_s_total'] == 280000
for name, digest in loop.read_json(R / 'AUTONOMY_SOURCE.json').items():
    assert loop.sha256(R / 'autonomy' / name) == digest, name
candidate = R / 'autonomy/POLICY_CONFIG_PARSE.toml'
candidate.write_text('extends = ' + json.dumps(cfg['policy']['config']) +
                     '\n[run]\noperational_recovery = ' + json.dumps('') + '\n')
raw = _read_config(candidate)
assert raw['trainer']['batch_size'] == 1
assert max(raw['horizon']['horizons_s']) == 60
processes = {}
for name in ('AUTONOMY_PID.json', 'AUTONOMY_MIRROR_PID.json'):
    pid = loop.read_json(R / name)
    fields = Path('/proc/' + str(pid['pid']) + '/stat').read_text().rsplit(')', 1)[1].split()
    assert fields[0] != 'Z' and fields[19] == pid['start_ticks'] and fields[1] == '1'
    assert os.getsid(pid['pid']) == pid['pid']
    processes[name] = {**pid, 'ppid': 1, 'live': True}
mirror = loop.read_json(R / 'AUTONOMY_MIRROR_STATUS.json')
assert time.time() - mirror['unix'] < 120
for box in ('cpuP', 'cpuQ'):
    b = mirror['boxes'][box]
    assert b['reachable'] and time.time() - b['watcher']['unix'] < 120
    assert b['watcher']['sid'] == b['watcher']['pid']
proof = {'t': loop.now(), 'processes': processes, 'mirror': mirror,
         'status': loop.read_json(R / 'AUTONOMY_STATUS.json'),
         'preflight': loop.read_json(R / 'AUTONOMY_PREFLIGHT.json'),
         'tests': loop.read_json(R / 'AUTONOMY_TESTS.json'),
         'charged': state['sim_s_charged'], 'cap': 280000,
         'horizon_config': raw['horizon'], 'source_sha256': loop.sha256(R / 'AUTONOMY_SOURCE.json')}
loop.write_atomic(R / 'AUTONOMY_VERIFIED.json', proof)
print(json.dumps(proof, indent=2))
