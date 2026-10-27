"""Summarize the owned foreground replay receipt without transferring model/data."""
import subprocess

SCRIPT = r'''
import json
import numpy as np
from pathlib import Path
p = Path.home()/'probe/DRY_RUN.json'
d = json.loads(p.read_text())
out = {k: v for k,v in d.items() if k != 'results'}
out['slots'] = []
for r in d['results']:
    candidates = r['candidates']
    negative = [c for c in candidates if np.mean(c['paired_benefit']['300']) < 0 and np.std(c['paired_benefit']['300']) > 0]
    top = max(negative, key=lambda c: np.quantile(c['paired_benefit']['300'], .9)) if negative else None
    out['slots'].append({'slot': r['slot'], 'candidates': len(candidates), 'uncertain_negative_mean': len(negative),
                         'selected': r['selected'], 'fallback': r['fallback'],
                         'uncalibrated_top': None if top is None else {'id': top['id'], 'action_type': top['action_type'],
                             'mean300': float(np.mean(top['paired_benefit']['300'])),
                             'empirical_upper300': float(np.quantile(top['paired_benefit']['300'], .9))}})
print(json.dumps(out, indent=2))
'''
result = subprocess.run(["ssh", "cpuP", "python3 -"], input=SCRIPT, text=True, capture_output=True, check=True)
print(result.stdout)
