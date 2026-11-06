"""S-LCEXEC-SMOKE attempt a2 (registered amendment 2): the 4 low-share worlds 989068-989071 on the
registered cell D3200_p40 (p30 is not a registered wz3to2 cell; a1 jobs failed in scenario build, 0 sim-s)."""
import json, sys, time, hashlib
from pathlib import Path
from tools.coopmwm import lc_smoke as S, map_scenarios, map_descriptor as md
root = Path(sys.argv[1]); assert not root.exists()
worlds = [dict(seed=s, demand=3200, penetration=.4) for s in range(989068, 989072)]
jobs = []
for w in worlds:
    world = dict(w, map=S.MAP, cell_key=S.cell_key(w))
    map_scenarios.check_case(w['seed'], w['demand'])
    map_scenarios.world(w['seed'], w['demand'], w['penetration'])   # the a1 failure point, must pass now
    jid = f"{world['cell_key']}_s{w['seed']}_a2"
    jobs.append(dict(id=jid, wave='main', world=world, warmup_s=S.WARMUP_S, end_s=S.END_S, sim_s=S.END_S,
                     moments=list(S.MOMENTS), runtime=S.DEFAULT_RUNTIME, checkpoint=S.DEFAULT_CHECKPOINT,
                     path=str(root/'jobs'/jid)))
desc = md.current()
body = dict(schema=S.SCHEMA, study=S.STUDY, phase=S.PHASE, created_unix=time.time(), host='gpuA', map=desc.name,
            map_descriptor_sha256=desc.digest, jobs=jobs, moments=list(S.MOMENTS), families=list(S.FAMILIES),
            budget=dict(jobs=len(jobs), total_sim_s=sum(j['sim_s'] for j in jobs), cap_sim_s=S.PHASE_CAP_SIM_S),
            source_commit='dcb8a792', amendment='2: p30 -> registered p40 for the low-share worlds')
body['plan_sha256'] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
root.mkdir(parents=True)
(root/'PLAN.json').write_text(json.dumps(body, sort_keys=True)+'\n')
print(json.dumps(dict(plan_sha256=body['plan_sha256'], budget=body['budget'])))
