"""Exclusive finite continuation: quality -> freeze -> one dispatch -> full report.

No scientific retry, process signal, training, peer mutation or budget expansion.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.coopbandit.protocol import sha
from tools.wmplan import ops

REPORT = ops.LOCAL/'docs/paper_results/WMPLAN.md'
BEGIN, END = '<!-- WMPLAN-PROGRESS:BEGIN -->', '<!-- WMPLAN-PROGRESS:END -->'


def git(*args):
    return subprocess.run(['git',*args],cwd=ops.LOCAL,check=True,capture_output=True,
                          text=True,encoding='utf-8').stdout


def commit(message):
    if git('branch','--show-current').strip() != 'c4/wmplan':
        raise RuntimeError('unexpected branch')
    if git('diff','--cached','--name-only').strip():
        raise RuntimeError('foreign staging; preserve and stop')
    allowed = ('docs/paper_results/WMPLAN.md','docs/paper_results/wmplan/')
    status = git('status','--porcelain','--untracked-files=all')
    for row in status.splitlines():
        path = row[3:].strip('"')
        if path != allowed[0] and not path.startswith(allowed[1]):
            raise RuntimeError('foreign work; do not stage: '+path)
    print(status,flush=True)
    print(git('diff','--',allowed[0],allowed[1]),flush=True)
    print(git('log','--oneline','-10'),flush=True)
    git('add','--',*allowed)
    git('diff','--cached','--check')
    if git('diff','--cached','--name-only').strip():
        print(git('commit','-m',message),flush=True)


def progress(text):
    current = REPORT.read_text(encoding='utf-8')
    section = BEGIN+'\n## Current continuation status\n'+text+'\n'+END
    if BEGIN in current:
        before,rest = current.split(BEGIN,1)
        _,after = rest.split(END,1)
    else:
        before,after = current.rstrip()+'\n\n','\n'
    with REPORT.open('w',encoding='utf-8') as f:
        f.write(before+section+after)


def main(hours=96,interval=60):
    # A permanent exclusive claim protects against accidental second collectors.
    ops.save('AUTO_CLAIM_1448_V2.json',dict(pid=os.getpid(),unix=time.time(),max_hours=hours,
                                  command='tools.wmplan.watch',interval_s=interval,
                                  role='complete_registered_plan_once',peer_process_control=False))
    frozen = {p:sha(p) for p in (ops.LOCAL/'tools/wmplan').glob('*.py')}
    frozen[ops.LOCAL/'docs/paper_results/WMPLAN_DESIGN.md'] = sha(ops.LOCAL/'docs/paper_results/WMPLAN_DESIGN.md')
    deadline = time.monotonic()+3600*hours
    def guard():
        if any(sha(p) != value for p,value in frozen.items()):
            raise RuntimeError('own continuation source/design changed; stop without mutation')
        if time.monotonic() >= deadline:
            raise TimeoutError('finite continuation deadline; preserve all remote work')
    try:
        progress('- Exclusive amended continuation is waiting for the already-running FULL-v5 quality check.\n'
                 '- Expected arm first, report before strict, both GPUs; exact199200upper retained.\n'
                 '- Original owned continuation stopped and retained; no peer process action or duplicated quality.')
        commit('wmplan: bind expected-first dual-GPU continuation')
        while not (ops.OUT/'MODEL_COMPARISON.json').is_file():
            guard();ops.compare()
            if not (ops.OUT/'MODEL_COMPARISON.json').is_file():
                time.sleep(interval)
        model = json.loads((ops.OUT/'MODEL_COMPARISON.json').read_text())
        progress(f'- Model comparison COMPLETE before physics: selected {model["selected"]}.\n'
                 f'- FULL-v4 helpful agreement {model["v4_helpful"]["helpful_choice_agreement"]:.3f}, '
                 f'all12 chosen saving {model["v4_all"]["exact_chosen_saving"]:+.6f}; '
                 f'FULL-v5 {model["v5_helpful"]["helpful_choice_agreement"]:.3f}, '
                 f'{model["v5_all"]["exact_chosen_saving"]:+.6f} veh-s/state.\n'
                 '- Reused existing predictions; no new quality forecasts or SUMO.')
        commit('wmplan: freeze preregistered model comparison before live outcomes')
        guard();ops.stage();commit('wmplan: bind immutable CPU GPU and reused baseline sources')
        from tools.wmplan.mirror import stage as mirror_stage, check as mirror_check
        guard();mirror_stage();commit('wmplan: freeze byte-identical gpuR inference mirror')
        guard();mirror_check();commit('wmplan: verify gpuR frozen runtime bootstrap without forecasting')
        guard();ops.cpu();commit('wmplan: verify frozen host contracts and all24 dual guards')
        guard();ops.freeze();commit('wmplan: register final model source and exact budget freeze')
        guard();ops.launch();commit('wmplan: bind one-use online planner dispatch and owned PIDs')
        while True:
            guard();s=ops.status()
            progress(f'- Model fixed as {model["selected"]}; {s["episodes"]}/48 episodes complete.\n'
                     f'- Charged {s["charged"]:g}; closed actual {s["actual"]:g} sim-s; '
                     f'{len(s["incomplete"])} open jobs. Registered upper199200/cap200000.\n'
                     f'- Complete K8 forecast states {s["forecasts"]}; future tasks {s["forecast_futures"]}.\n'
                     '- Full paired efficacy and safety report waits for all24 pairs per arm. '
                     'No claim of beating MAPPO is supported by progress counts.')
            commit('wmplan: collect owned live budget and complete-forecast progress')
            if s['stopped']:
                raise RuntimeError('remote durable STOP: '+str(s['stop']))
            if s['primary_complete'] and not (ops.OUT/'PRIMARY_RESULT_REGISTRATION.json').is_file():
                guard();ops.status(collect=True,primary=True)
                from tools.wmplan import report
                report.run(primary=True);report.run(verify=True,primary=True);report.publish(primary=True)
                commit('wmplan: publish verified primary expected-arm results before strict')
                print(ops.remote(f'''import pathlib,json,time
root=pathlib.Path({ops.ROOT!r})
with (root/'PRIMARY_REPORTED.json').open('x') as f:json.dump(dict(unix=time.time(),
 report_sha256={sha(REPORT)!r}),f)
print(json.dumps(dict(primary_published=True,strict_released=True)))
'''),flush=True)
            live = next(v for v in s['owners'] if v['name']=='LIVE')
            if not s['complete'] and (not live['identity_match'] or live['state']=='Z'):
                raise RuntimeError('owned supervisor ended incomplete; no automatic replay')
            if s['complete']:
                break
            time.sleep(interval)
        guard();ops.status(collect=True)
        commit('wmplan: preserve complete paired episodes forecasts and all group twins')
        from tools.wmplan import report
        guard();report.run();report.run(verify=True);report.publish()
        ops.save('AUTO_COMPLETE.json',dict(unix=time.time(),new_dispatches=1,
                                         report_sha256=sha(REPORT)))
        commit('wmplan: publish independently verified paired results and safety')
    except BaseException:
        error = traceback.format_exc()
        ops.save('AUTO_FAILURE.json',dict(unix=time.time(),error=error,automatic_retry=False))
        progress('- Continuation stopped; preserve all completed evidence and existing remote work.\n'
                 '- No automatic retry or new dispatch is armed. See `wmplan/AUTO_FAILURE.json`.')
        try:
            commit('wmplan: preserve continuation fault without retry or peer mutation')
        except BaseException:
            print(traceback.format_exc(),flush=True)
        raise


def launch():
    assert (ops.LOCAL/'_out').is_dir()
    assert (ops.OUT/'RECOVERY_1448.json').exists()
    assert (ops.OUT/'PRELIVE_PATH_ISOLATION.json').exists()
    assert not (ops.OUT/'AUTO_CLAIM_1448_V2.json').exists()
    logfile = ops.LOCAL/'_out'/f'wmplan_watch_{int(time.time())}.log'
    with logfile.open('x',encoding='utf-8') as log:
        p = subprocess.Popen([sys.executable,'-B','-u','-m','tools.wmplan.watch'],cwd=ops.LOCAL,
                             stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                             creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    return dict(pid=p.pid,log=str(logfile),command='tools.wmplan.watch',new_sumo_sim_s=0)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--launch',action='store_true')
    a = p.parse_args()
    if a.launch:
        print(json.dumps(launch()))
    else:
        main()
