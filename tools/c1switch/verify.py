"""Verify publication exports and stopped-job accounting without rerunning anything."""
import ast
from collections import defaultdict
import csv
import json
from pathlib import Path

from .design import sha
from .storage import write


def main():
    root = Path('docs/paper_results')
    design = root/'C1SWITCH_DESIGN.md'
    design_lines = len(design.read_text(encoding='utf-8').splitlines())
    assert design_lines <= 80
    assert sha(design) == 'c4154bbe1a9281d61eb465de0e64123f232151b60dcf5a10a2fbdde30377967a'
    with (root/'c1switch/recorded_world_winners.csv').open(newline='') as f:
        worlds = list(csv.DictReader(f))
    with (root/'c1switch/recorded_cell_winshares.csv').open(newline='') as f:
        cells = list(csv.DictReader(f))
    assert len(worlds) == 270 and len(cells) == 360
    assert len({(r['dataset'],r['seed'],r['clock_s']) for r in worlds}) == 270
    shares = defaultdict(float)
    for r in cells:
        shares[tuple(r[k] for k in ('dataset','clock_s','penetration','demand_vph'))] += float(r['fractional_win_share'])
    assert len(shares) == 72 and all(abs(v-1) < 1e-12 for v in shares.values())
    reconciliation = json.loads(Path('tools/c1switch/evidence/RECONCILIATION.json').read_text())
    assert reconciliation['charged_sim_s'] == reconciliation['actual_sim_s'] == 930
    assert reconciliation['complete_decisions'] == 0 and not any(reconciliation['owned_pids_alive'].values())
    assert reconciliation['GPU_hours'] == 0
    repair = json.loads(Path('tools/c1switch/evidence/CPU_REPAIR.json').read_text())
    assert repair['returncode'] == 0 and repair['new_sim_s'] == 0 and repair['STOP_retained']
    assert all('PASS' in value for value in repair['checks'].values())
    for p in Path('tools/c1switch').glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    files = [design, root/'C1SWITCH_WINNERS.md']+sorted((root/'c1switch').glob('*'))
    report = dict(pass_=True, design_lines=design_lines,
                  recorded_world_clock_rows=270, cell_controller_rows=360,
                  cell_share_sums_checked=72, charged_sim_s=930, actual_sim_s=930,
                  complete_fresh_decisions=0, GPU_hours=0,
                  artifacts={p.as_posix():sha(p) for p in files if p.is_file()})
    write('tools/c1switch/evidence/VERIFICATION.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
