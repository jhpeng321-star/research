"""AMENDMENT 42 seed audit (zero SUMO, no outcome read): are the proposed D2800_p60 verification block 992000-992099
(24 used: 992000-992023) and confirmation block 998600-998699 (96 used: 998600-998695) unused?

Method (as the amendment-37 audit, ops/a37/SEED_AUDIT.json): standalone integers of each block (hex / decimal fragments
excluded) in REGISTRATION.md, protocols/experiment_history.json, text files of the C4 run dirs and agent work dirs, and
``git grep`` over every unique c4/* + main branch tip. Paths containing a forbidden token are never opened; per-episode
evaluation dirs and outcome files (WAVE_* / RESULT* / *_RESULTS* / vehicles* / observable*) are skipped (never read).
Writes ops/a42/SEED_AUDIT.json with the classification left to the build agent (``hits`` are listed verbatim).

Usage: python ops/a42/seed_audit.py --out ops/a42/SEED_AUDIT_RAW.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

RD = Path('E:/research-dense')
RUN = RD/'runs/active/C4-V7FT-20261017'
AW = Path('E:/agent-work')
BLOCKS = dict(verification=(992000, 992099), confirmation=(998600, 998699))
FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth', 'driver_params')
SKIP_DIRS = {'evaluation', '.git', '__pycache__', 'node_modules', 'venv', '.venv'}
SKIP_FILE = re.compile(r'^(WAVE_|RESULT|A\d+B?_RESULTS|vehicles|observable)', re.I)
TEXT_EXT = {'.json', '.jsonl', '.md', '.txt', '.py', '.sh', '.csv', '.log', '.out', '.yaml', '.yml', '.toml'}
MAX_BYTES = 40 << 20
RUN_DIRS = ('a32_ablation', 'a33_combined', 'a34_horizon', 'a35_transfer', 'a36_analysis', 'a37_celltailor', 'a40_geomvar', 'wmc_strategy_v3',
            'paper_data', 'paper_analysis_a32', 'paper_fairness', 'p20c_seed_audit', 'p20x_seedaudit', 'p20x_bank',
            'nm_finaleval_p20c_confirm', 'nm_finaleval_p20x_r2_bank', 'regimeu0', 'wmc_p20', 'lowcav', 'multicrit') + tuple(
    d.name for d in sorted(RUN.glob('wmverify_*')))
AGENT_DIRS = ('a32', 'a33', 'a34', 'a35', 'a37', 'a38', 'a39', 'a40', 'distill', 'distill-eval', 'distill-rl',
              'distill-train', 'a38-road2', 'a40-geomvar', 'a41-d3200bound', 'a42')


def forbidden(p):
    s = str(p).lower()
    return any(t in s for t in FORBIDDEN)


def pattern(lo, hi):
    """Standalone integers lo..hi (same width): the A37 regex form."""
    return re.compile(r'(?<![\da-fA-F.])(\d{%d})(?![\da-fA-F.])' % len(str(lo)))


def scan_text(text, lo, hi, rx):
    return [(m.start(), m.group(1)) for m in rx.finditer(text) if lo <= int(m.group(1)) <= hi]


def files():
    yield RUN/'REGISTRATION.md'
    yield RD/'protocols/experiment_history.json'
    roots = [RUN/d for d in RUN_DIRS] + [AW/d for d in AGENT_DIRS]
    for root in roots:
        if not root.is_dir() or forbidden(root):
            continue
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not forbidden(Path(dp)/d)]
            for f in fns:
                p = Path(dp)/f
                if forbidden(p) or SKIP_FILE.match(f) or p.suffix.lower() not in TEXT_EXT:
                    continue
                try:
                    if p.stat().st_size > MAX_BYTES:
                        continue
                except OSError:
                    continue
                yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--repo', default='E:/agent-work/a42-bigp60')
    a = ap.parse_args()
    out = dict(schema='a42.seed_audit_raw.v1', blocks={k: list(v) for k, v in BLOCKS.items()},
               forbidden_skipped=list(FORBIDDEN), skipped_files=SKIP_FILE.pattern, file_hits={k: [] for k in BLOCKS},
               git_hits={k: [] for k in BLOCKS})
    n = 0
    for p in files():
        try:
            text = p.read_text(encoding='utf-8', errors='replace')
        except OSError:
            continue
        n += 1
        for name, (lo, hi) in BLOCKS.items():
            for pos, tok in scan_text(text, lo, hi, pattern(lo, hi)):
                line = text.count('\n', 0, pos)+1
                out['file_hits'][name].append(dict(path=str(p), line=line, token=tok,
                                                   context=text[max(0, pos-60):pos+40].replace('\n', ' ')))
    out['n_files'] = n
    tips = sorted(set(subprocess.run(['git', '-C', a.repo, 'for-each-ref', '--format=%(objectname)', 'refs/heads/c4',
                                      'refs/heads/main'], capture_output=True, text=True).stdout.split()))
    out['n_tips'] = len(tips)
    for name, (lo, hi) in BLOCKS.items():
        # 3-digit-prefix regex then exact range filter
        pre = str(lo)[:3]
        rx = r'(^|[^0-9a-fA-F.])%s[0-9]{3}([^0-9a-fA-F.]|$)' % pre
        hits = set()
        for i in range(0, len(tips), 40):
            r = subprocess.run(['git', '-C', a.repo, 'grep', '-I', '-o', '-E', rx]+tips[i:i+40],
                               capture_output=True, text=True, errors='replace')
            for line in r.stdout.splitlines():
                parts = line.split(':', 2)
                if len(parts) < 3:
                    continue
                for m in re.finditer(r'\d{6}', parts[2]):
                    if lo <= int(m.group()) <= hi:
                        hits.add(f'{parts[0][:12]}:{parts[1]}:{m.group()}')
        # collapse identical path:token across tips
        coll = {}
        for h in hits:
            c, path, tok = h.split(':', 2)
            coll.setdefault(f'{path}:{tok}', []).append(c)
        out['git_hits'][name] = [dict(path_token=k, n_tips=len(v), example_tip=sorted(v)[0]) for k, v in sorted(coll.items())]
    out['audited'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
    Path(a.out).write_text(json.dumps(out, indent=1)+'\n', encoding='utf-8')
    print(json.dumps({k: {b: len(v) for b, v in out[k].items()} for k in ('file_hits', 'git_hits')}),
          'files', n, 'tips', len(tips))


if __name__ == '__main__':
    main()
