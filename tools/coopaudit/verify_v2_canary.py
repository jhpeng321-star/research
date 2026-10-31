"""Verify already-completed full-prefix canary locally; no extra simulator work."""
import hashlib
import json
from pathlib import Path
from tools.coopaudit.accounting import write
from tools.coopaudit.design import interaction, sha


def main():
    evidence=Path(__file__).resolve().parent/'evidence'
    root=evidence/'v2_canary'
    directory=root/'jobs/v2_canary_s969699'
    r=json.loads((directory/'result.json').read_text())
    assert r['ok'] and r['actual_sim_s']==14400 and r['branch_mode']=='full_prefix_rerun'
    metrics,hashes=r['metrics'],r['trajectory_sha256']
    for name in ('native_B_repeat','mappo_DISABLED','sumocf_DISABLED'):
        assert metrics[name]==metrics['native_B'] and hashes[name]==hashes['native_B']
    prefix_hashes=set()
    for p in directory.glob('branch_*.json'):
        doc=json.loads(p.read_text())
        assert doc['prefix_verified'] and doc['actual_sim_s']==900
        prefix_hashes.add(doc['prefix_trajectory_sha256'])
    assert len(prefix_hashes)==1
    assert len(list(directory.glob('branch_*.json')))==16
    wall=r['effects']['wall_GROUP']
    assert wall['wall_issued_commands']>0 and wall['wall_events']
    contrasts={}
    for c in ('mappo','wall','sumocf'):
        assert hashes[c+'_GROUP']==hashes[c+'_GROUP_repeat']
        assert metrics[c+'_GROUP']==metrics[c+'_GROUP_repeat']
        contrasts[c]={outcome:interaction(metrics['native_B'][outcome]['team_delay_vs'],
                              metrics[c+'_GROUP'][outcome]['team_delay_vs'],
                              [metrics[c+f'_ONLY:{i}'][outcome]['team_delay_vs'] for i in range(2)])
                      for outcome in ('300','600')}
    for clocks in metrics.values():
        for m in clocks.values():
            assert not any(m[k] for k in ('collisions','teleports','dropped_vehicle_steps'))
    scientific_files={name:sha(Path(__file__).with_name(name)) for name in
                      ('full_rerun.py','controllers.py','trained.py','runner.py','design.py','release.py','accounting.py')}
    assert scientific_files==json.loads((root/'COLLECTION.json').read_text())['scientific_files']
    receipt=dict(ok=True,actual_sim_s=14400,charged_total_sim_s=sum(json.loads(s)['sim_s'] for s in
                 (root/'accounting.jsonl').read_text().splitlines()),
                 prefix_sha256=next(iter(prefix_hashes)),branch_count=16,zero_effect_bitwise=True,
                 repeated_native_and_three_controllers_bitwise=True,
                 wall_issued_commands=wall['wall_issued_commands'],wall_events=wall['wall_events'],
                 contrasts=contrasts,scientific_files=scientific_files,
                 result_sha256=sha(directory/'result.json'),collection_sha256=sha(root/'COLLECTION.json'))
    write(evidence/'V2_CANARY_VERIFIED.json',receipt)
    print(json.dumps(receipt,indent=1))


if __name__=='__main__':
    main()
