"""A38 laptop analysis (zero physics): verify every offloaded file against OFFLOAD.sha256 (the sha the box reported),
then run the registered analysis of ops/a38_road2.py on the offloaded mirror root: per-arm J, contrasts (paired
percentile bootstrap 10,000, seed 20261104, diff = treatment - reference, negative = less delay): HOLD_R2 - U0
[, HOLD_R2_DUO01 - U0, HOLD_R2_DUO01 - HOLD_R2], total + CVaR90 + groups (10 s/veh rule; origin:L2 = WM lane 0 is
empty on wz2to1) + hard brakes / TTC / collisions / teleports.
usage: C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe laptop_analyse.py <setting a38_D<d>_p20> <arms> [offload dir] [out dir]"""
import hashlib, json, os, sys
from pathlib import Path

SRC = Path('E:/agent-work/a38-road2')
setting, arms = sys.argv[1], sys.argv[2].split(',')
off = Path(sys.argv[3] if len(sys.argv) > 3 else 'E:/agent-work/a38/offload')
out = Path(sys.argv[4] if len(sys.argv) > 4 else 'E:/agent-work/a38/analysis_local')
latest = {}
for line in (off/'OFFLOAD.sha256').read_text().splitlines():
    h, f = line.split(None, 1)
    latest[f.strip()] = h                       # last verified sha per file (live ledgers are re-pulled)
bad = [f for f, h in latest.items() if hashlib.sha256((off/f).read_bytes()).hexdigest() != h]
if bad:
    raise SystemExit(f'{len(bad)} offloaded files differ from their verified sha, e.g. {bad[:3]}')
os.environ['COOPMWM_MAP_SPEC'] = str(SRC/'assets/research/newmap_candidates/wz2to1/MAP_SPEC.json')
sys.path[:0] = [str(SRC/'ops'), str(SRC)]
import a38_road2 as D  # noqa: E402
panel = D.setup(setting)
value = D.analyse(panel, off/'root', arms)
out.mkdir(parents=True, exist_ok=True)
p = out/f'A38_RESULTS_{D.cell_tag(panel)}.json'
p.write_text(json.dumps(value, indent=1) + '\n', encoding='utf-8')
print(json.dumps(dict(out=str(p), analysis_sha256=value['analysis_sha256'], files_verified=len(latest),
                      totals={f"{c['treatment']}-{c['reference']}": c.get('total_minus_ref') for c in value['contrasts']}),
                 default=str)[:3000])
