"""Harm-calibration input from WMVERIFY D3200_p60 (zero SUMO): per verified family its strategy spec, the SUMO
per-group mean per-vehicle delay increase vs u0 (s/veh) and the SUMO J diff vs the incumbent (veh-s, negative =
better).   python ops/wmc_cal_input.py ANALYSIS_DIR OUT.json"""
import json
import sys
from pathlib import Path

GROUPS = {'origin:L0': 'lane0', 'origin:L1': 'lane1', 'origin:L2': 'lane2', 'class:hdv': 'hdv', 'class:cav': 'cav'}


def main(adir, out, cell='D3200_p60'):
    adir = Path(adir)
    v = json.loads((adir / 'VERIFY.json').read_text())
    specs = {}
    for line in (adir / f'BANK_{cell}.jsonl').open():
        r = json.loads(line)
        if r.get('spec') and r['family'] not in specs:
            specs[r['family']] = r['spec']
    rows = []
    for fam, f in v['fairness'][cell].items():
        g = f['fairness']['u0']['groups']
        vi = v['vs_incumbent'][cell].get(fam) or {}
        rows.append(dict(family=fam, spec=specs.get(fam), allowed=f['allowed'],
                         sumo_inc={GROUPS[k]: g[k]['increase_mean_s'] for k in GROUPS},
                         sumo_inc_ci95={GROUPS[k]: g[k]['ci95'] for k in GROUPS},
                         sumo_diff_vs_inc=vi.get('mean_diff'), sumo_diff_vs_inc_ci95=vi.get('ci95'),
                         n_worlds=f['fairness']['u0']['n_worlds']))
    Path(out).write_text(json.dumps(dict(cell=cell, source=str(adir / 'VERIFY.json'), rows=rows), indent=1))
    print(len(rows), sum(r['spec'] is not None for r in rows))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
