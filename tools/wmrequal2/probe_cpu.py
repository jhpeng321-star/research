"""Bounded CPU-only ownership checks and recorded original-ramp exposure census."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/wmrequal'))
from remote import remote

SCRIPT = r'''
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
import hashlib, json, sys, time
from pathlib import Path
from types import SimpleNamespace
begin = time.monotonic()
root = Path('/root/autodl-tmp/c4/wmrequal20261030')
rt = Path('/root/autodl-tmp/c4/restart20261026/runtime_longint')
bad = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth')
def guard(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        name = os.fsdecode(args[0]).lower()
        if any(x in name for x in bad) or name.endswith('.xml'):
            raise RuntimeError('prohibited input path')
sys.addaudithook(guard)
sys.path[:0] = [str(root), str(root/'tools/wmrequal'), str(root/'src'), str(rt/'src'), str(rt)]
import numpy as np
import torch
torch.set_num_threads(2)
assert not torch.cuda.is_available()
import score
e3 = score.bootstrap_runtime()
real = json.loads((root/'REAL.json').read_text())['panels']['validation']
source, indexes = score.source_for('validation', real)
exposure = []
for gi in indexes:
    key, ref, arms = e3.group_rows(source, gi)
    names = ['native'] + [a for a,_ in arms]
    records = source.group_records(gi, [ref]+[r for _,r in arms], base_row=0)
    for name, rec in zip(names, records):
        states, present, cav = (np.asarray(rec[k]) for k in ('states','present','is_cav'))
        lanes = states[...,1:4].argmax(-1)
        ramp = present[0].astype(bool) & cav[0].astype(bool) & (lanes[0] == 2)
        original_mainline = present[0].astype(bool) & cav[0].astype(bool) & (lanes[0] != 2)
        merged = ramp & (present.astype(bool) & (lanes == 1)).any(0)
        original_lane_change = original_mainline & (present.astype(bool) & (lanes != lanes[0])).any(0)
        timeline = rec['action_timeline']
        exposure.append(dict(pair_key=key, arm=name, original_ramp_cavs=int(ramp.sum()),
            recorded_original_ramp_merged=int(merged.sum()),
            original_mainline_cavs_with_recorded_lane_change=int(original_lane_change.sum()),
            timeline_ids=sorted(map(int,np.unique(timeline['controller'])))))

from wms.control.entrant import EntrantControllers, feedback_from_specs
from wms.train.cav_drive import TimelineControllers
from wms.physics.config import PhysicsConfig
from wms.model.cav import WM_MERGE_KEY, WM_LANE_TARGET_KEY
design = json.loads(Path('/root/autodl-tmp/c4/overnight20261028/W5_COMBINED/DESIGN.json').read_text())
names = ['native','mappo','ours_c1','coopoff_c1']
specs = {n:design['controllers'][n] for n in names[1:]}
for spec in specs.values():
    assert hashlib.sha256(Path(spec['checkpoint']).read_bytes()).hexdigest() == spec['sha256']
physics = PhysicsConfig()
plant = SimpleNamespace(physics=physics, physics_accel=lambda state: torch.zeros_like(state['s_m']))
timeline = dict(controller=torch.zeros(4,2,2,dtype=torch.long),
    acceleration_mps2=torch.zeros(4,2,2), target_speed_mps=torch.zeros(4,2,2))
checks = []
for entrant in (False, True):
    # The ramp slot is before the merge zone, forcing every actor to stay.
    # No longitudinal outcome/qualification is inferred from this synthetic scene.
    state = dict(s_m=torch.tensor([[-20.,150.]]).expand(4,-1).clone(),
        v_mps=torch.full((4,2),10.), a_mps2=torch.zeros(4,2),
        lane=torch.tensor([[2,1]]).expand(4,-1).clone(),
        length_m=torch.full((4,2),5.), present=torch.ones(4,2,dtype=torch.bool),
        is_cav=torch.tensor([[True,False]]).expand(4,-1).clone(),
        entrant=torch.full((4,2),entrant,dtype=torch.bool),
        inferred_driver_parameters=torch.tensor([31.29,1.2,2.,1.2,2.,.3,.1,6.,1.]).expand(4,2,-1).clone(),
        inferred_idm_delta=4., secure_gap_T_s=torch.full((4,2),1.2),
        secure_gap_b_mps2=torch.full((4,2),2.))
    state[WM_LANE_TARGET_KEY] = state['lane'].clone()
    feedback = feedback_from_specs(specs,names,plant,timeline,2,'cpu')
    callback = EntrantControllers(TimelineControllers(plant,timeline),feedback)
    with torch.no_grad():
        motion = callback(state)
    delegated = state[WM_MERGE_KEY][:,0].tolist()
    assert motion[:,0,1].tolist() == [2.]*4
    assert delegated == ([False,True,True,True] if entrant else [False]*4)
    checks.append(dict(cohort='entrant' if entrant else 'original', arms=names,
        ramp_stay_delegated_to_first_passage=delegated,
        returned_ramp_lane=motion[:,0,1].tolist()))

totals = {}
for arm in names:
    rows = [r for r in exposure if r['arm']==arm]
    totals[arm] = dict(worlds=len(rows), worlds_with_original_ramp_cavs=sum(r['original_ramp_cavs']>0 for r in rows),
        original_ramp_cavs=sum(r['original_ramp_cavs'] for r in rows),
        recorded_original_ramp_merged=sum(r['recorded_original_ramp_merged'] for r in rows),
        original_mainline_cavs_with_recorded_lane_change=sum(r['original_mainline_cavs_with_recorded_lane_change'] for r in rows))
print(json.dumps(dict(schema='wmrequal2_cpu_probe/1', cuda_visible='', threads=2,
    no_remote_writes=True, no_optimizer=True, cpu_seconds=time.monotonic()-begin,
    ownership_checks=checks, totals=totals, exposure=exposure,
    conclusion='All original ramp CAVs lack merge delegation; native entrant CAVs also lack delegation. Policy entrants delegate stays with policy_ramp_merge=False.'),indent=1))
'''


if __name__ == '__main__':
    dest = ROOT / 'tools/wmrequal2/evidence/CPU_PROBE.json'
    assert dest.parent.is_dir()
    result = json.loads(remote(SCRIPT, timeout=600))
    with dest.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=1, allow_nan=False)
        stream.write('\n')
