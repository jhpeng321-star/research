"""Frozen WM scoring: physical exits, full fleet/source, K8, mix ON/OFF CRNs.

Refuses unknown per-source initial queues. Does not create policy authority.
Each file is a completed group-batch and is reused only with identical sources.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

BASE = Path('/root/autodl-tmp/c4')
OUT = BASE/'wmrequal20261030'
RT = BASE/'restart20261026/runtime_longint'
sys.path[:0] = [str(OUT), str(OUT/'tools/wmrequal'), str(OUT/'src'), str(RT/'src'), str(RT)]
from labels import HORIZONS, safe, crossings


def digest(path):
    return hashlib.sha256(safe(path).read_bytes()).hexdigest()


def bootstrap_runtime():
    # Override exactly the model file containing the ported newborn hook. All
    # remaining imports use the already frozen verdict runtime, never a live tree.
    import wms.model.wms
    assert Path(wms.model.wms.__file__).resolve() == (OUT/'src/wms/model/wms.py').resolve()
    spec = importlib.util.spec_from_file_location('requal_e3',OUT/'tools/wmcombined/e3_score.py')
    e3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e3)
    return e3


class Entries:
    def __init__(self, process):
        self.process = process
        self.present, self.pos, self.lane, self.queues = [], [], [], []

    def insert(self, state):
        result = self.process.insert(state)
        for dest, key in ((self.present,'present'),(self.pos,'s_m'),(self.lane,'lane')):
            dest.append(state[key].detach().clone())
        self.queues.append(self.process.pending.copy())
        return result


def source_for(panel, real_rows):
    from wms.data.c4_pairs import PairedBranchSource
    from real_panel import entries_for
    root, entries = entries_for(panel)
    role_of = {e['world']:'train' for e in entries}
    source = PairedBranchSource(root, horizon_steps=240 if panel == 'panel120' else 600,
        history_s=60., history_stride_s=1., entries_v2=True,junction_fix='ffill',
        command_known=True,arm_kinds=('B','POLICY') if panel != 'pairs300' else ('B','SEED'),
        role='train',role_of={e['world']: ('train' if e['world'] in role_of else 'val') for e in
            json.loads((root/'C4_EPISODES_MANIFEST.json').read_text())['episodes']})
    keys={r['pair_key'] for r in real_rows}
    indexes=[i for i,g in enumerate(source.groups) if g['pair_key'] in keys]
    assert len(indexes) == len(keys)
    return source,indexes


def score_batch(model, settings, e3, source, indexes, real_rows, device, steps, smoke,
                *, native_lanes=False, mix_modes=(False, True)):
    import numpy as np
    import torch
    from tools.errsrc.inference import LearnedMix
    from wms.control.futures import SampledFutures, FuturesConfig
    from wms.data.collate import collate
    from wms.train.inputs import prepare_inputs
    from wms.train.scoring_random import scoring_rng, bind_merge_crn
    records, layout, base_refs, base_map, row_keys, mode_rows = [], [], [], [], [], []
    lookup={(r['pair_key'],r['arm']):r for r in real_rows}
    for gi in indexes:
        key, ref, arms = e3.group_rows(source,gi)
        arm_order=[('native',ref),*arms]
        for draw in range(1 if smoke else 8):
            shared=len(base_refs)
            base_refs.append(len(records))
            for mix in mix_modes:
                block=len(records)
                recs=source.group_records(gi,[ref]+[r for _,r in arms],base_row=block)
                for (arm,_),rec in zip(arm_order,recs):
                    lab=lookup.get((key,arm))
                    if lab is None and arm=='native':
                        lab=next(r for r in real_rows if r['pair_key']==key and r['native'])
                    if lab['queue0'] is None:
                        raise ValueError('exact per-source Q0 unavailable: '+key)
                    rec['entry_queue']=np.asarray([*lab['queue0'],*lab['demand_rate'],
                                                  lab['penetration'],lab['penetration']],np.float32)
                    rec['pair_group']=shared
                    rec['pair_key']=key+f':future{draw}'
                    records.append(rec)
                    base_map.append(shared)
                    row_keys.append(f'{key}:{draw}')
                    mode_rows.append(mix)
                    layout.append((key,draw,mix,arm))
    batch=collate(records)
    with scoring_rng(140014,device):
        inp=prepare_inputs(batch,model,settings,device,steps)
    if native_lanes:
        from tools.wmrequal.native_lanes import repair
        inp.cav_controller = repair(inp.cav_controller)
    bind_merge_crn(inp)
    # K8 training driver/embedding uncertainty, drawn at start level and shared
    # across every arm and ON/OFF row; never independent per intervention.
    refs=torch.tensor(base_refs,device=device)
    mapping=torch.tensor(base_map,device=device)
    select=lambda t:t[refs] if torch.is_tensor(t) and t.ndim and t.shape[0]==len(records) else t
    state={k:select(v) for k,v in inp.initial.items()}
    hist=dict(driver_history=select(inp.observed_hist),driver_history_mask=select(inp.observed_hist_mask),
              history=select(inp.hist),history_mask=select(inp.hist_mask))
    plan=SampledFutures(FuturesConfig(n_samples=1,driver_noise_scale=1.,embedding_noise_std=.05),
                        140014).plan([model],state,hist,steps)[0][0]
    expand=lambda value:{k:expand(v) for k,v in value.items()} if isinstance(value,dict) else value[mapping]
    kw={k:expand(v) for k,v in plan.items()}
    mix=LearnedMix(kw['driver_params'],inp.observed_hist_mask,140014,row_keys)
    active=torch.tensor(mode_rows,device=device,dtype=torch.bool)
    def sampler(st,newborn,params):
        return mix(st,newborn & active[:,None],params)
    entries=Entries(inp.entries)
    with torch.no_grad(),scoring_rng(140014,device):
        roll=model.rollout(inp.initial,n_steps=steps,entries=entries,
            cav_teacher=None if inp.teacher is None else inp.teacher[:,:steps+1],
            # Free physical exits; recorded outcomes never censor model HDVs.
            present_seq=None,is_cav_seq=None,history=inp.hist,history_mask=inp.hist_mask,
            controller_id=inp.controller_id,action_timeline=inp.action_timeline,
            cav_controller=inp.cav_controller,driver_history=inp.observed_hist,
            driver_history_mask=inp.observed_hist_mask,entrant_driver_sampler=sampler,
            lane_uniform=inp.paired_random.uniform('lane',range(steps)),**kw)
    present=torch.stack(entries.present,1).bool()
    pos=torch.stack(entries.pos,1)
    road=.5*torch.where(present,(1-roll['speeds']/31.29).clamp_min(0),0.)
    queue=torch.as_tensor(np.stack(entries.queues,1),device=device)
    source=.5*queue.sum(-1)
    upstream=(road*(pos<=75.44)).sum(-1)
    downstream=(road*(pos>75.44)).sum(-1)
    oldsel=inp.initial['present'].bool() & ~inp.initial['is_cav'].bool()
    legacy=.5*torch.where(roll['masks'].bool() & oldsel[:,None],
                         (1-roll['speeds']/31.29).clamp_min(0),0.).sum(-1)
    columns=dict(road=road.sum(-1),source=source,J=road.sum(-1)+source,
                 upstream=upstream,downstream=downstream,t0_HDV=legacy)
    result=[]
    arr=lambda x:x.detach().cpu().numpy()
    wmpos=arr(torch.cat((inp.initial['s_m'][:,None],roll['positions']),1))
    wmmask=arr(torch.cat((inp.initial['present'][:,None],roll['masks']),1)).astype(bool)
    wmlane=arr(torch.cat((inp.initial['lane'][:,None],roll['lanes']),1))
    for b,(key,draw,on,arm) in enumerate(layout):
        ids=np.broadcast_to(np.arange(wmpos.shape[-1]),wmpos[b].shape)
        events=crossings(wmpos[b],wmmask[b],wmlane[b],wmlane[b]>=0,ids)
        values={}
        for h in HORIZONS:
            if h*2>steps:
                continue
            n=h*2
            values[str(h)]={k:float(v[b,:n].double().sum().cpu()) for k,v in columns.items()}
            values[str(h)].update(queue=float(queue[b,n-1].sum().cpu()),
                                  discharge=int((events<=h).sum()))
        result.append(dict(pair_key=key,draw=draw,mix=on,arm=arm,values=values,
                           mix_assigned=mix.assigned[b]))
    del inp,roll,entries,batch,road,columns,kw,plan,mix
    gc.collect()
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--panel',choices=('validation','calibration','pairs300','panel120'),required=True)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--groups-per-batch',type=int,default=1)
    a=ap.parse_args()
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.85)
    e3=bootstrap_runtime()
    ckpt=BASE/'restart20261026/arm/loop/wm/c02/wm_s.pt'
    cfg=BASE/'restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml'
    assert digest(ckpt)=='b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
    _,settings,model=e3.load_for_scoring(cfg,ckpt)
    controllers=json.loads((BASE/'overnight20261028/W5_COMBINED/DESIGN.json').read_text())['controllers']
    settings=replace(settings,seed=140014,entrant_control=True,entrant_controllers=controllers,
                     entry_queue_state=True,entry_pad_auto=True,entry_pad_max=1024,
                     entry_insert_rule='sumo_like_v2')
    model=model.to('cuda').eval().requires_grad_(False)
    model.cav_exec_cap='idm'
    rows=json.loads((OUT/'REAL.json').read_text())['panels'][a.panel]
    source,indexes=source_for(a.panel,rows)
    steps=20 if a.smoke else (240 if a.panel=='panel120' else 600)
    if a.smoke:
        indexes=indexes[:1]
    target=OUT/('smoke' if a.smoke else a.panel)
    target.mkdir(exist_ok=True)
    receipt=dict(pid=os.getpid(),gpu_fraction=.85,wm_sha256=digest(ckpt),config_sha256=digest(cfg),
                 source_sha256=digest(OUT/'SOURCE.json'),real_sha256=digest(OUT/'REAL.json'),
                 K=1 if a.smoke else 8,physical_exits=True,queue_state=True,
                 futures=dict(driver_scale=1.,embedding_std=.05,accel_noise=False),
                 source_receipt=source.receipt,steps=steps)
    (target/f'RUN_{os.getpid()}.json').write_text(json.dumps(receipt,indent=1))
    print('OWN_PID',os.getpid(),'groups',len(indexes),'steps',steps,flush=True)
    for start in range(0,len(indexes),a.groups_per_batch):
        dest=target/f'batch_{start:03d}.json'
        if dest.is_file():
            old=json.loads(dest.read_text())
            assert old['receipt']['source_sha256']==receipt['source_sha256']
            continue
        begin=time.monotonic()
        result=score_batch(model,settings,e3,source,indexes[start:start+a.groups_per_batch],
                           rows,torch.device('cuda'),steps,a.smoke)
        with dest.open('x') as stream:
            json.dump(dict(receipt=receipt,rows=result,seconds=time.monotonic()-begin),stream,allow_nan=False)
        print('DONE_BATCH',a.panel,start,round(time.monotonic()-begin,2),flush=True)
    (target/'DONE.json').write_text(json.dumps(dict(pid=os.getpid(),groups=len(indexes),steps=steps)))


if __name__=='__main__':
    main()
