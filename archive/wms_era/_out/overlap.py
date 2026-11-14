import json, sys
sys.path.insert(0, 'E:/rd-wt/ws-r3-evalstates3/output/wd01/wm_n1/lpwo')
from pathlib import Path
from verify_state_futures import _states
out = {}
for name in ('wall_same', 'wall_pilot', 'seen_controls'):
    v2 = {r['key'] for r in _states(Path('E:/rd-wt/ws-wallc1/_out/wallc1_states_v2/%s.json' % name))}
    new = {}
    for r in _states(Path('E:/rd-wt/ws-r3-evalstates3/_out/wallc1_states_h60_real/%s.json' % name)):
        new[r['key']] = r['pack']
    out[name] = {'v2_states': len(v2), 'real_states': len(new), 'common': len(v2 & set(new)),
                 'real_only': len(set(new) - v2), 'v2_only': len(v2 - set(new)),
                 'real_by_pack': {p: sum(1 for x in new.values() if x == p) for p in set(new.values())}}
json.dump(out, open('E:/rd-wt/ws-r3-evalstates3/_out/logs/overlap.json', 'w'), indent=1)
print(json.dumps(out))
