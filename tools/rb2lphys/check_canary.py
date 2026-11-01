"""Verify the completed own paired canaries before spending the panel budget."""
import argparse
import json
from pathlib import Path

from tools.rb2lphys.compare import compare
from tools.rbeval.core import save_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--targets',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    records = {params:[dict(world=969420,directory=str(args.evidence/name/'task'))]
               for params,name in (('off','canary_off'),('learned_v1','canary_lp'))}
    report = compare(dict(worlds=[969420],K=2,records=records),args.targets,expected_worlds=(969420,))
    report['purpose'] = 'source/CRN/finite/conservation canary verification; no candidate choice'
    save_new(args.out,report)
    print(json.dumps(dict(canary_pass=True,tasks=len(report['audit']),summary=report['summary'])),flush=True)


if __name__ == '__main__':
    main()
