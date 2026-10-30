"""Read exact frozen scorer inputs over SSH; CPU only, no remote file writes."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/wmrequal'))
from remote import remote

SCRIPT = r'''
import ast, hashlib, json, os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
root = Path('/root/autodl-tmp/c4/wmrequal20261030')
bad = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth')
def safe(p):
    assert not any(s in str(p.resolve()).lower() for s in bad)
    return p
source = json.loads(safe(root/'SOURCE.json').read_text())
names = ('src/wms/control/bridge.py', 'src/wms/control/entrant.py',
         'src/wms/train/cav_drive.py', 'src/wms/train/inputs.py',
         'src/wms/model/wms.py', 'tools/wmrequal/score.py')
files = {}
for name in names:
    p = safe(root/name)
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == source['sha256'][name], name
    tree = ast.parse(raw)
    functions = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in (
            'make_cav_controller', 'policy_feedback', '__call__',
            'pivot_decision_geometry', '_apply_merge_pivot'):
            defaults = dict(zip([a.arg for a in n.args.args][-len(n.args.defaults):],
                                [ast.unparse(d) for d in n.args.defaults])) if n.args.defaults else {}
            defaults.update({a.arg: ast.unparse(d) for a,d in zip(n.args.kwonlyargs,n.args.kw_defaults) if d is not None})
            functions.setdefault(n.name, []).append(dict(line=n.lineno, defaults=defaults))
    lines = raw.decode().splitlines()
    needles = ('policy_ramp_merge', 'policy_strategic_merge', 'make_cav_controller(',
               'mask_forced_merge=True', 'WM_MERGE_KEY', 'old_merge', 'new_merge',
               'return torch.stack((position, lane.float()',
               'decider = ~cav', 'own = own |', 'EntrantControllers(',
               'cav_controller = TimelineControllers')
    snippets = {str(i+1): line.strip() for i,line in enumerate(lines) if any(s in line for s in needles)}
    files[name] = dict(sha256=sha, source_receipt_match=True, functions=functions, snippets=snippets)
p = safe(Path('/root/autodl-tmp/c4/overnight20261028/W5_COMBINED/DESIGN.json'))
raw = p.read_bytes()
design = json.loads(raw)
print(json.dumps(dict(schema='wmrequal2_runtime_audit/1', cuda_visible='',
    remote_writes=False, source_base=source['base'],
    source_sha256=hashlib.sha256((root/'SOURCE.json').read_bytes()).hexdigest(),
    files=files, controller_design=dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(),
                                      controllers=design['controllers'])), indent=1))
'''


if __name__ == '__main__':
    dest = ROOT / 'tools/wmrequal2/evidence/RUNTIME.json'
    assert dest.parent.is_dir()
    result = json.loads(remote(SCRIPT))
    with dest.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1, allow_nan=False)
        stream.write('\n')
