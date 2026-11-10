"""A40 seed audit (zero SUMO, no outcome read; adapted from A38): TRAIN 994200-994399 + panel 994000-994099 (wz3to2v80) as standalone integers in the registration log, the
ledger, specific run dirs, every c4/* branch tip (git grep) and gpuB/a32-a35 dir names. Paths whose name contains a
forbidden token are skipped unopened."""
import json, re, subprocess, sys, time
from pathlib import Path

RX = re.compile(r'(?<![\da-fA-F.])(994[0-3]\d\d)(?![\da-fA-F.])')
FORBID = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1')
RD = Path('E:/research-dense')
RUN = RD/'runs/active/C4-V7FT-20261017'
hits, checked = [], []


def scan_file(p):
    if any(t in str(p).lower() for t in FORBID):
        return
    try:
        txt = p.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return
    for m in RX.finditer(txt):
        line = txt.count('\n', 0, m.start())+1
        hits.append(dict(path=str(p), line=line, token=m.group(1)))


scan_file(RUN/'REGISTRATION.md'); checked.append('REGISTRATION.md')
scan_file(RD/'protocols/experiment_history.json'); checked.append('protocols/experiment_history.json')
n = 0
dirs = [RUN/'paper_data', RUN/'a32_ablation', RUN/'a33_combined', RUN/'a35_transfer', RUN/'p20c_seed_audit', RUN/'p20x_seedaudit',
        RUN/'newmap_probe', RUN/'newmap2', Path('E:/agent-work/a33'), Path('E:/agent-work/a35'), Path('E:/agent-work/a32-ablation/_out'), Path('E:/agent-work/a37'), Path('E:/agent-work/a38'), RUN/'a34_horizon', RUN/'a36_analysis',
        RD/'protocols', RD/'docs']
for d in dirs:
    if not d.exists() or any(t in str(d).lower() for t in FORBID):
        continue
    for p in d.rglob('*'):
        if p.is_file() and p.suffix in ('.json', '.md', '.jsonl', '.txt', '.sha256', '.py', '.tsv') \
                and not any(t in str(p).lower() for t in FORBID):
            scan_file(p); n += 1
checked.append(f'{n} json/md/jsonl/txt/sha256/py/tsv files under ' + ', '.join(str(d) for d in dirs if d.exists()))
refs = subprocess.run(['git', '-C', str(RD), 'for-each-ref', '--format=%(objectname)', 'refs/heads/c4/', 'refs/heads/main'],
                      capture_output=True, text=True).stdout.split()
refs = sorted(set(refs))
g = subprocess.run(['git', '-C', str(RD), 'grep', '-I', '-P', '-o', r'(?<![\da-fA-F.])(?:994[0-3]\d\d)(?![\da-fA-F.])'] + refs,
                   capture_output=True, text=True)
ghits = sorted(set(l for l in g.stdout.splitlines() if l and not any(t in l.lower() for t in FORBID)))
checked.append(f'git grep -P over {len(refs)} unique c4/* + main branch tips: {len(ghits)} hit lines')
out = dict(schema='a40.seed_audit.v1', map='wz3to2v80', train_seeds=[994200, 994399], panel_seeds=[[994000, 994095]],
           reserved_unused=[[994096, 994099]],
           regex=RX.pattern + ' (standalone integers; hex/decimal fragments excluded)', checked=checked,
           file_hits=hits, git_hits=ghits[:200], forbidden_skipped=list(FORBID),
           audited=time.strftime('%F %T') + ' +08 by A40 (zero SUMO, no outcome read)')
Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + '\n', encoding='utf-8')
print(json.dumps(dict(file_hits=len(hits), git_hits=len(ghits), files=n, refs=len(refs)), indent=1))
for h in hits[:20]: print(h)
for h in ghits[:20]: print(h)
