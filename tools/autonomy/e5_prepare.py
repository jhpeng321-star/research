"""Exact zero-SUMO E5 handoff command; prepares bindings, never launches a job."""
import json
import subprocess
from pathlib import Path


def main():
    source = '''import pathlib,sys,json
r=pathlib.Path('/root/autodl-tmp/c4/coopoff220261027');root=r/'runtime_c2'
sys.path[:0]=[str(root),str(root/'src'),str(root/'scripts/c4')]
import loop14
from wms.control.nativeinit import verify_collection_gate
s=loop14.read_json(r/'arm/loop/LOOP14_STATE.json');p=s['policy']
assert p['sha256']=='e1fa4881bbc5c89d05298e979d98b23284308331ad1bbb12f9608fa67e105675'
g=verify_collection_gate(p)
assert g['passed'] and len(g['policy_J'])==16
print(json.dumps({'policy':p,'native_gate':g,'gate_sha256':loop14.sha256(pathlib.Path(p['checkpoint']).with_name('native_gate_result.json')),'pause':s.get('owner_priority_pause')}))
'''
    p = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'gpuR',
                        '/root/miniconda3/bin/python -B -'], input=source, text=True,
                       capture_output=True, timeout=90, check=True)
    binding = json.loads(p.stdout)
    request = {'task': 'S-E3-LONG-1_CYCLE1_E5_PREPARATION', 'status': 'PREPARED_NOT_LAUNCHED',
        'ours_coopoff': binding, 'ours_coopon_sha256': '64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c',
        'worlds': [973000, 973227], 'count': 228, 'warmup_s': 300, 'intervention_s': 600,
        'clocks_s': [120, 300, 600], 'new_charged_sim_s': 205200,
        'funding': 'unused410400 registered E3 policy allocation; separate from loop280000 caps',
        'contrast': 'ours-coopON(cycle1) minus ours-coopOFF(cycle1), world-cluster bootstrap5000, equal-cell weighting',
        'execution': 'same frozen E3 runtime;6nice19one-thread workers/box; live collections first; canary-first',
        'launch_blockers': ['new ours-coopoff adapter and GO input hashes',
            'both-box zero-charge preflight', 'bound append-only E3 amendment',
            'detached mirror + automatic matched analysis before canary dispatch'],
        'no_launch_authority_exercised': True}
    out = Path(__file__).resolve().parents[2] / 'E5_CYCLE1_PREPARATION.json'
    out.write_text(json.dumps(request, indent=2), encoding='utf-8')
    print(json.dumps(request, indent=2))


if __name__ == '__main__':
    main()
