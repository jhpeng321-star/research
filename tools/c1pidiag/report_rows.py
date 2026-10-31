"""Compact, sourced console views of the full attribution export."""
from pathlib import Path
import argparse
import json

ROOT=Path(__file__).resolve().parents[2]


def fmt(s):
    if s['point'] is None:return 'NA'
    return f"{s['point']:.2f} [{s['lo']:.2f},{s['hi']:.2f}] n={s['n']}/{s['worlds']}"


def main():
    ap=argparse.ArgumentParser();ap.add_argument('section',choices=('headline','history','switches','runtime','fullfeedback'));args=ap.parse_args()
    rows=json.loads((ROOT/'docs/paper_results/c1pidiag/SUMMARY.json').read_text())['summary']
    if args.section=='fullfeedback':rows=json.loads((ROOT/'docs/paper_results/c1pidiag/aux_feedback/SUMMARY.json').read_text())['summary']
    for r in rows:
        match=False
        if args.section=='headline':
            match=r['panel']=='original' and r['mode'] in ('tf','clh') and not r['population'] and r['residual'] and r['lanes']=='original' and r['component']=='baseline' and r['history']==('h5' if r['cohort']=='wall_same' else 'snapshot')
        elif args.section=='history':match=r['panel']=='aux' and not r['population']
        elif args.section=='switches':match=r['panel']=='original' and r['mode']=='tf' and not r['population'] and r['tag'] in ('wm13e','wm2') and (r['component']!='baseline' or not r['residual'] or r['lanes']!='original')
        elif args.section=='runtime':match=r['panel']=='legacy'
        elif args.section=='fullfeedback':match=True
        if match:
            print(r['panel'],r['mode'],r['cohort'],r['config'])
            for h in ('60','120'):
                s=r['horizons'][h]
                print(h,'own',fmt(s['own']),'vsBB',fmt(s['paired_wm_minus_bb']),
                      'vsBase',fmt(s['paired_vs_model_baseline']),'vsSnap',fmt(s['paired_vs_snapshot']),
                      'vsC1',fmt(s['paired_vs_c1']))


if __name__=='__main__':main()
