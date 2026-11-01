"""Finite read-only collection/scoring/delivery continuation; never dispatches inference."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MODULE_ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(MODULE_ROOT),str(MODULE_ROOT/'src')]

from tools.rb2int.audit import ROOT
from tools.rb2int.collect import collect_host,registry
from tools.rbint.local_resources import child_incarnation
from tools.rbeval.core import load,digest,save_new

PYTHON=Path('C:/Users/jpeng/miniconda3/python.exe')
OLD_ENDPOINTS=ROOT.parent/'c4-rb-int/runs/rbint_evidence/noon_wm2_hdv60_20261101T054124+0800'
FINAL_DEADLINE=datetime.fromisoformat('2026-11-02T09:40:00+08:00').timestamp()


def command(args,cwd=ROOT):
    result=subprocess.run(args,cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=1800)
    if result.returncode: raise RuntimeError(result.stdout+'\n'+result.stderr)
    return result.stdout


def git_commit(paths,message):
    # Inspect the requested status/diff/log before staging; never stage another
    # worktree's source or broad untracked directories.
    status=command(['git','status','--short'])
    diff=command(['git','diff','--stat'])
    recent=command(['git','log','--oneline','-10'])
    if command(['git','branch','--show-current']).strip()!='c4/rb2-int':
        raise ValueError('delivery branch changed')
    if command(['git','diff','--cached','--name-only']).strip():
        raise ValueError('foreign staging present; preserve it')
    intended=[str(Path(p).relative_to(ROOT)).replace('\\','/') for p in paths]
    for p in intended:
        if not (p.startswith('docs/rb2int/') or p in ('docs/paper_results/WMPHYS_V5.md','docs/paper_results/WMPHYS_V5.delivery.json','HANDOFF.md')):
            raise ValueError('delivery may commit only its own named evidence/report')
    command(['git','add','--',*intended])
    command(['git','diff','--cached','--check'])
    if not command(['git','diff','--cached','--name-only']).strip():
        return None
    output=command(['git','commit','-m',message])
    return dict(status_before=status,diff_before=diff,recent_before=recent,commit_output=output)


def score_once(delivery):
    source=ROOT/'docs/rb2int/scoring_data_v1'
    real=source/'REAL_OBSERVABLES.json';factual=source/'FACTUAL_ENDPOINTS.json'
    provenance=load(source/'SCORING_SOURCE.json')
    if digest(real)!=provenance['real_sha256'] or digest(factual)!=provenance['endpoint_sha256']:
        raise ValueError('immutable scoring data changed')
    cache=ROOT/'runs/rb2int_final_cache'
    runs=delivery/'RUNS.json'
    if not runs.exists(): registry(cache,runs)
    scores=delivery/'scores';scores.mkdir(exist_ok=True)
    steps=[]
    for model in ('WM2','FULL-v4','FULL-v5','FULL-v5-LP'):
        own=scores/model;own.mkdir(exist_ok=True)
        for module,destination,scope,marker in (
                ('tools.rb2int.score_gate',own/'gate','all58','gate.json'),
                ('tools.rb2int.credibility',own/'credibility','primary54','CREDIBILITY.json')):
            if (destination/marker).is_file():
                continue
            if destination.exists(): raise ValueError('partial scoring output preserved; explicit recovery required')
            argv=[str(PYTHON),'-B','-m',module,'--runs',str(runs),'--real',str(real),
                  '--model',model,'--scope',scope,'--out',str(destination)]
            output=command(argv)
            record=delivery/(module.rsplit('.',1)[1]+'_'+model+'_PROCESS.json')
            save_new(record,dict(argv=argv,completed=True,output=output,inference_repeated=False))
            steps.append(record)
    for scope in ('primary54','all58','reserved8'):
        out=delivery/('COMPARISON_'+scope+'.json')
        if out.is_file(): continue
        argv=[str(PYTHON),'-B','-m','tools.rb2int.compare','--runs',str(runs),'--real',str(real),
            '--factual-endpoints',str(factual),'--wm2-reserved-endpoints',str(OLD_ENDPOINTS),
            '--scope',scope,'--out',str(out)]
        command(argv)
    report=ROOT/'docs/paper_results/WMPHYS_V5.md'
    if not report.with_suffix('.delivery.json').exists():
        command([str(PYTHON),'-B','-m','tools.rb2int.report','--scores',str(scores),
            '--comparison',str(delivery/'COMPARISON_primary54.json'),'--out',str(report)])
    return report,scores


def publish(delivery,report,scores):
    out=ROOT/'docs/rb2int/final_delivery'
    if out.exists(): raise ValueError('preserve existing final publication')
    out.mkdir()
    created=[];manifest={}
    for path in sorted(delivery.rglob('*')):
        if not path.is_file(): continue
        relative=path.relative_to(delivery)
        manifest[str(relative)]=digest(path)
        # Full bin/observation data remain preserved in runs with SHA receipts;
        # publish all score summaries, per-world components and audits.
        if path.name in ('distribution_world_bins.csv','OBSERVATIONS.json'):
            continue
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);created.append(target)
    manifest_path=out/'FULL_EVIDENCE_MANIFEST.json'
    save_new(manifest_path,dict(root=str(delivery),hashes=manifest,raw_data_preserved=True,
        all_four_models_complete=True,final_time=datetime.now().astimezone().isoformat()))
    created.append(manifest_path)
    created += [report,report.with_suffix('.delivery.json')]
    handoff=ROOT/'HANDOFF.md'
    text=handoff.read_text(encoding='utf-8')
    block='\n<!-- RB2-FINAL:BEGIN -->\n## Round2 final delivery complete\n'+\
        'All58/allfivearms/K8/fourmodels collected; primary54 paired comparison, full58 gate and reserved8 sensitivity complete.\n'+\
        'Final report docs/paper_results/WMPHYS_V5.md; audit/score/CI/credibility receipts docs/rb2int/final_delivery/.\n'+\
        'No inference repeated; raw bin/forecast data preserved with full SHA manifest. Read report for qualification and switching verdicts.\n'+\
        '<!-- RB2-FINAL:END -->\n'
    if '<!-- RB2-FINAL:BEGIN -->' in text: raise ValueError('final handoff already published')
    handoff.write_text(text+block,encoding='utf-8');created.append(handoff)
    receipt=git_commit(created,'S-RB2-INT: deliver complete registered four-model v5 evaluation')
    return dict(publication=str(out),commit=receipt)


def run(out,once=False):
    cache=ROOT/'runs/rb2int_final_cache'
    cache.mkdir(exist_ok=True)
    for host in ('gpuA','gpuR'): (cache/host).mkdir(exist_ok=True)
    watched=[Path(__file__)]+[ROOT/'tools/rb2int'/name for name in (
        'collect.py','prepare_scoring.py','score_gate.py','compare.py','credibility.py','report.py',
        'predictions.py','uncertainty.py','gate_bootstrap.py')]
    source_hashes={str(p):digest(p) for p in watched}
    progress_report=ROOT/'docs/paper_results/WMPHYS_V5.md'
    progress_hash=digest(progress_report)
    save_new(out/'START_ONCE.json',dict(pid=os.getpid(),source_hashes=source_hashes,
        final_deadline=FINAL_DEADLINE,inference_dispatch=False,scientific_retry=False,
        progress_report_sha256=progress_hash))
    while True:
        if any(digest(p)!=value for p,value in source_hashes.items()):
            raise ValueError('collection/scoring source changed; preserve completed work and resume explicitly')
        stamp=datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')
        results=[];paths=[]
        for host in ('gpuA','gpuR'):
            receipt=ROOT/'docs/rb2int'/('COLLECT_'+stamp+'_'+host+'.json')
            results.append(collect_host(host,cache/host,receipt));paths.append(receipt)
        git_commit(paths,'S-RB2-INT: collect complete-K8 registered final worlds '+stamp)
        total=sum(r['complete_world_jobs'] for r in results)
        print('COLLECT',stamp,total,'/216 complete world jobs',flush=True)
        if total==216:
            if digest(progress_report)!=progress_hash:
                raise ValueError('progress report changed during collection; preserve edits and assemble explicitly')
            delivery=ROOT/'runs/rb2int_delivery_v1'
            delivery.mkdir(exist_ok=True)
            report,scores=score_once(delivery)
            result=publish(delivery,report,scores)
            save_new(out/'COMPLETE.json',result);print('FINAL DELIVERY COMPLETE',flush=True);return
        if once: return
        if time.time()>=FINAL_DEADLINE:
            save_new(out/'INCOMPLETE_AT_DEADLINE.json',dict(world_jobs=total,expected=216,
                inference_repeated=False,qualification='NOT_ESTABLISHED; full evaluation incomplete'))
            return
        time.sleep(min(1200,max(1,FINAL_DEADLINE-time.time())))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('start','watch','once'))
    parser.add_argument('--out',default=str(ROOT/'runs/rb2int_finish_v1'))
    args=parser.parse_args();out=Path(args.out)
    if not out.parent.is_dir(): raise ValueError('existing continuation parent required')
    out.mkdir(exist_ok=True)
    if args.action=='start':
        save_new(out/'DISPATCH_ONCE.json',dict(source_sha256=digest(__file__),inference_dispatch=False))
        argv=[str(PYTHON),'-B','-u',str(Path(__file__).resolve()),'watch','--out',str(out)]
        flags=(subprocess.DETACHED_PROCESS|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0
        with (out/'watch.log').open('x',encoding='utf-8') as log:
            child=subprocess.Popen(argv,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=flags,start_new_session=os.name!='nt',env=dict(os.environ,
                PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        receipt=dict(pid=child.pid,argv=argv,**child_incarnation(child),inference_dispatch=False)
        save_new(out/'PROCESS.json',receipt);print(json.dumps(receipt));return
    try:
        run(out,once=args.action=='once')
    except Exception as error:
        save_new(out/'STOP.json',dict(error=repr(error),inference_repeated=False,automatic_retry=False))
        raise


if __name__=='__main__':
    main()
