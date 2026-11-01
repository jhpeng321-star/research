"""Protocol-only validation of currently cached complete worlds; no score selection."""
import argparse
import json

from tools.rb2int.predictions import read_model
from tools.rbeval.core import load,save_new,digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',required=True)
    parser.add_argument('--real',required=True)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    manifest=load(args.runs);reg=load(manifest['registration_path']);real=load(args.real)['episodes']
    result={}
    for model in ('WM2','FULL-v4','FULL-v5','FULL-v5-LP'):
        _,audit=read_model(args.runs,model,real,reg['final_comparison_worlds'],partial=True)
        result[model]=dict(tasks=audit['tasks'],complete_worlds=audit['complete_worlds'],
            missing_tasks=len(audit['missing_tasks']))
    save_new(args.out,dict(passed=True,protocol_only=True,scores_computed=False,
        registry_sha256=digest(args.runs),real_sha256=digest(args.real),models=result))
    print(json.dumps(result,indent=1))


if __name__=='__main__':
    main()
