"""Quantify archived-to-current runtime differences on fixed original baselines."""
from pathlib import Path
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[2]


def main():
    rows=json.loads((ROOT/'_out/c1pidiag/primary/VERIFICATION.json').read_text())['refresh_reproduction']['details']
    for tag in ('c1struct','wm13e','bbtraj'):
        for h in (60,120):
            rs=[r for r in rows if r['tag']==tag and r['horizon']==h and r['abs_difference'] is not None]
            print(tag,h,'n',len(rs),'old/new medians',np.median([r['archived'] for r in rs]),np.median([r['new'] for r in rs]),
                  'abs difference quantiles',np.quantile([r['abs_difference'] for r in rs],[0,.5,.9,1]))
            print('largest',sorted(rs,key=lambda r:r['abs_difference'],reverse=True)[:2])


if __name__=='__main__':main()
