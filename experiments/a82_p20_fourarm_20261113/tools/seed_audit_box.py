"""Zero-SUMO seed audit on a run host (A81 pattern, evaluation dirs pruned): panel files, runner range tables and
6-digit tokens in code/docs under the given tops, for the block LO-HI. usage: seed_audit_box.py LO HI TOP [TOP ...]"""
import json
import os
import re
import sys

LO, HI = int(sys.argv[1]), int(sys.argv[2])
TOPS = sys.argv[3:]
BAN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'truth', 'driver_params')
PRUNE = {'evaluation', 'venv', 'site-packages', '__pycache__', '.git', 'node_modules', 'wheelhouse'}
out = dict(block=[LO, HI], tops=TOPS, n_files_scanned=0, panels=[], panel_overlap=[], runner_ranges=set(),
           runner_overlap=[], code_hits=[])
pat = re.compile(r'(?<![\d.])(\d{6})(?![\d])')
rng = re.compile(r'\((\d{6}), ?(\d{6})\)')


def ok(p):
    return not any(b in p.lower() for b in BAN)


for top in TOPS:
    for d, dirs, files in os.walk(top):
        dirs[:] = [x for x in dirs if x not in PRUNE and ok(os.path.join(d, x)) and not os.path.islink(os.path.join(d, x))]
        for f in files:
            p = os.path.join(d, f)
            if not ok(p) or os.path.islink(p):
                continue
            ext = os.path.splitext(f)[1]
            try:
                size = os.path.getsize(p)
            except OSError:
                continue
            if ext == '.json' and re.search(r'(panel|PANEL|trainblock)', f) and size < 20_000_000:
                out['n_files_scanned'] += 1
                try:
                    v = json.load(open(p))
                except Exception:
                    continue
                if isinstance(v, dict) and isinstance(v.get('worlds'), list):
                    seeds = [w['world']['seed'] for w in v['worlds'] if isinstance(w, dict) and 'world' in w]
                    if seeds:
                        out['panels'].append([p, min(seeds), max(seeds), len(seeds)])
                        if any(LO <= s <= HI for s in seeds):
                            out['panel_overlap'].append(p)
            elif ext in ('.py', '.sh', '.md', '.txt', '.toml', '.yaml', '.cfg') and size < 5_000_000:
                out['n_files_scanned'] += 1
                s = open(p, encoding='utf-8', errors='replace').read()
                if f == 'nm_eval_runner.py' or 'nm_eval_runner' in f:
                    for a, b in rng.findall(s):
                        a, b = int(a), int(b)
                        out['runner_ranges'].add((a, b))
                        if a <= HI and b >= LO:
                            out['runner_overlap'].append([p, a, b])
                for m in pat.finditer(s):
                    v = int(m.group(1))
                    if LO <= v <= HI:
                        out['code_hits'].append([p, v, s[max(0, m.start()-60):m.end()+30].replace('\n', ' ')])
out['runner_ranges'] = sorted(out['runner_ranges'])
out['n_panels'] = len(out['panels'])
out['panels_summary'] = sorted({(a, b, n) for _, a, b, n in out['panels']})
del out['panels']
print(json.dumps(out, default=str))
