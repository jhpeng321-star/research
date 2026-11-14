"""Register MAIN14:06's exact extra phases; no physics or process dispatch."""
import json
from . import ops
from .protocol import REGISTRY, checked, sha
from .storage import read, write
from .ls_collect_ops import ROOT
from .ls_validity_ops import remote

QUOTE='V2V_LAUNCH_QUOTE_1406_LIVE.json'
REG='V2V_CAP_REGISTRATION_1406.json'
BINDING='V2V_CAP_1406.json'


def register():
    q=read(ops.EVIDENCE/QUOTE);selected=q['selected']
    assert q['launch_cost_gate'] and selected['within_ceiling']
    assert selected['arms']==['A-v2v','B-v2v','R-v2v']
    assert selected['additional_quote_sim_s']<=3000000
    p=ops.EVIDENCE/REG
    if not p.exists():
        base=json.loads(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});p=r/'B_REPLAY_EXPOSURE_V2.json'
print(json.dumps(dict(budget=json.loads(p.read_text())['budget'],base_binding=str(p))))''',timeout=40))
        budget=base['budget'];assert budget['total']==14500000
        assert sum(budget['phases'].values())==budget['allocated']==14500000
        budget['phases']=dict(budget['phases'],**selected['phase_costs'])
        budget['total']=budget['allocated']=14500000+selected['additional_quote_sim_s']
        assert sum(budget['phases'].values())==budget['total']<=17500000
        line=('\nS-COOPMWM MAIN2026-11-03 14:06 OWNER CONDITIONAL V2V LAUNCH CAP: '
            f"14500000->{budget['total']} sim-s; exact +{selected['additional_quote_sim_s']} "
            'for ALL A/B/R-v2v from SAME MAPPOu0,updates1..24,24episodes/update. '
            'B lessons+isolated6/12/18/24; R isolated12/24; A no extra diagnostics; '
            'episode-only u8/u24 all3, no V2V final paired branches. '
            'Fresh baseline references conservatively quoted; SHA-qualified reuse only '
            'when identical. No arm/test/diagnostic dropped in this within-ceiling quote. '
            'Current bank slate5B/11R cooperative states and8B lessons; future growth '
            'requires phase room BEFORE any branch, never silent overspend. New phases '
            'are restricted to V2V workloads; old phase allocations unchanged. Shared '
            'authoritative ledger/lock, all prior charges retained. Automatic durable '
            'launch only after integrationREADY+LinuxPASS+rehearsal or real all-arm '
            '1-update smokes; resource guards active. NonV2Vtraining/collection/builders '
            'continue. No physics/inference/reservation in this registration.\n')
        write(p,dict(authority='MAIN2026-11-03 13:45/14:06',line=line,budget=budget,
            quote_sha256=sha(ops.EVIDENCE/QUOTE),quote=selected,metadata=q['metadata'],
            base_binding=base['base_binding'],new_sim_s=0,new_inference=0,reserved_sim_s=0))
    v=read(p);h=sha(p);registry=checked(REGISTRY)
    if h not in registry.read_text(encoding='utf-8'):
        with registry.open('a',encoding='utf-8') as f:f.write(v['line']+'amendmentSHA'+h+'\n')
    value=dict(v,amendment_sha256=h)
    print(remote('cpuQ',f'''from pathlib import Path
import json
r=Path({ROOT!r});v=json.loads({json.dumps(value)!r});p=r/{BINDING!r}
if p.exists():assert json.loads(p.read_text())==v
else:
 with p.open('x') as f:json.dump(v,f)
a=r/'AUTHORITY.md'
if {h!r} not in a.read_text():
 with a.open('a') as f:f.write(v['line']+'amendmentSHA'+{h!r}+chr(10))
print(json.dumps(dict(registered=True,sha256={h!r},total_cap=v['budget']['total'],new_phases=v['quote']['phase_costs'],new_sim_s=0,new_inference=0,physics_dispatched=False)))''',timeout=40))


if __name__=='__main__':register()
