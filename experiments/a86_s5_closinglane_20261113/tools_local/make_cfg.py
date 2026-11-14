"""Build cfg/A86S5_SEARCH_CFG.json from the S4 cfg (same key order, same json format: indent=1).
usage: python -I -B make_cfg.py ROOT n_init generations n_children n_immigrants"""
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
n_init, gens, kids, imm = (int(x) for x in sys.argv[2:6])
src = ROOT / 'cfg' / 'S4_REF' / 'A85S4_SEARCH_CFG.json'
c = json.loads(src.read_text(encoding='utf-8'))
assert c['seed'] == 20261185 and c['exclude_lane_sets'] == [[1, 2]]
c['seed'] = 20261187
c['n_init'], c['generations'], c['n_children'], c['n_immigrants'] = n_init, gens, kids, imm
c['exclude_lane_sets'] = [[1], [2], [1, 2]]
c['niching'] = {'definition': 'a80_form_family x role_lane_set', 'parents': True, 'finalists': True, 'top_cap': 2}
out = ROOT / 'cfg' / 'A86S5_SEARCH_CFG.json'
# the S4 file was written by json.dumps(indent=1) without a trailing newline: check the format round-trips
assert json.dumps(json.loads(src.read_text(encoding='utf-8')), indent=1) == src.read_text(encoding='utf-8')
out.write_bytes(json.dumps(c, indent=1).encode('utf-8'))
print(out)
