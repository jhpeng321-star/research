"""File-IPC WM-PHYS service using the EXACT frozen teacher-quality path."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from tools.coopbandit.protocol import digest
from tools.coopplan.predictor import Outcome, Prediction, validate_request
from tools.coopplan.storage import read, write
from tools.wmplan.design import decision


def publish(path, value):
    """Expose complete JSON atomically; retain temporary evidence (no deletion)."""
    tmp = path.with_name(path.name+f'.{os.getpid()}.{time.time_ns()}.tmp')
    write(tmp,value)
    os.link(tmp,path)


class Client:
    def __init__(self, root, seed, arm):
        self.root, self.seed, self.arm = root, seed, arm

    def predict(self, state, actions, horizon, K):
        validate_request(state, actions, horizon, K)
        if (horizon,K) != (300,8):
            raise ValueError('frozen H300 K8')
        setting = ((state.time_s-300)//60)%4
        # Arm identity is excluded: identical physical observations reuse a
        # completed forecast once for each consumer, rather than recompute it.
        record = dict(state_id=digest(dict(seed=self.seed,time_s=state.time_s,
                                          rows=state.rows,setting=setting)),
                      world_id=self.seed, time_s=state.time_s, setting=setting,
                      role='QUALITY_ONLY', rows=list(state.rows))
        key = record['state_id']
        request = self.root/'requests'/f'{key}.json'
        if not request.exists():
            try:
                publish(request,record)
            except FileExistsError:
                pass
        response = self.root/'responses'/f'{key}.json'
        while not response.exists():
            if (self.root/'STOP.json').exists():
                raise RuntimeError('durable STOP during forecast; no replay')
            time.sleep(1)
        result = read(response)
        prediction = Prediction({a:tuple(Outcome(**f) for f in fs)
                                 for a,fs in result['futures'].items()},
                                False, 'WM-PHYS-'+result['backend'])
        return prediction.validate(actions,K)


class OnlinePlanner:
    def __init__(self, root, seed, arm):
        self.client, self.arm, self.last_time = Client(root,seed,arm),arm,None

    def step(self, state):
        from tools.coopplan.planner import candidates, TIMES
        if state.time_s not in TIMES or (self.last_time is not None and state.time_s <= self.last_time):
            raise ValueError('frozen cadence')
        actions = candidates(state)
        prediction = self.client.predict(state,actions,300,8)
        chosen = decision(prediction,self.arm)
        action = next(a for a in actions if a['id'] == chosen['selected'])
        self.last_time = state.time_s
        return dict(time_s=state.time_s,replay_key=state.replay_key,candidates=actions,
                    available_candidates=actions,horizon_s=300,group=list(action['members']),
                    executed_action=action,decision=chosen,
                    prediction_id=digest({a:[vars(f) for f in fs]
                                          for a,fs in prediction.actions.items()}))


def serve(root, worker):
    import shutil
    import subprocess
    from tools.wmplan.live import verify
    binding = verify(root)
    os.environ['WMTEACH_BACKEND'] = binding['model']['selected']
    from tools.wmteach.runtime import bootstrap, fit_futures
    # Admission happens before CUDA allocation and never changes a peer process.
    while True:
        if (root/'STOP.json').exists():
            return
        free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                   '--format=csv,noheader,nounits'],text=True).strip())
        if free > 12*1024 and shutil.disk_usage(root).free >= 2*2**30:
            break
        time.sleep(15)
    rt = bootstrap()
    write(root/f'GPU_READY_{worker}.json',dict(pid=os.getpid(),backend=binding['model']['selected']))
    try:
        while not (root/'COMPLETE.json').exists() and not (root/'STOP.json').exists():
            found = False
            for request in sorted((root/'requests').glob('*.json')):
                key = request.stem
                claim = root/'forecast_claims'/f'{key}.json'
                try:
                    write(claim,dict(worker=worker,pid=os.getpid(),unix=time.time()))
                except FileExistsError:
                    continue
                found = True
                r = read(request)
                folder = root/'forecasts'/key
                folder.mkdir()
                values = {}
                for k in range(8):
                    # This is the same function, not a newly qualified input or
                    # sampling adapter. Its source hashes are frozen at staging.
                    result = fit_futures(rt,None,r,k)
                    audit = result['population_audit']
                    if not audit['finite'] or not audit['conserved']:
                        raise ValueError('WM population integrity')
                    for h, rows in result['results'].items():
                        for a,f in rows.items():
                            road,source = result['road_source'][h][a]
                            if abs(f['team_delay']-road-source) > 1e-6:
                                raise ValueError('source-inclusive delay integrity')
                    write(folder/f'k{k}.json',result)
                    for a,f in result['results']['300'].items():
                        values.setdefault(a,[]).append(f)
                    import gc
                    import torch
                    gc.collect()
                    torch.cuda.empty_cache()
                from tools.wmteach.labels import actions
                Prediction({a:tuple(Outcome(**f) for f in fs) for a,fs in values.items()},
                           False,'WM-PHYS').validate(actions(r['rows'],r['setting']),8)
                publish(root/'responses'/f'{key}.json',dict(futures=values,
                      backend=binding['model']['selected'],complete_futures=8))
            if not found:
                time.sleep(1)
    except BaseException:
        from tools.coopplan.storage import stop
        stop(root,traceback.format_exc())
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('--worker',type=int,required=True)
    a = p.parse_args()
    try:
        serve(a.root,a.worker)
    except BaseException:
        from tools.coopplan.storage import stop
        stop(a.root,traceback.format_exc())
        raise
