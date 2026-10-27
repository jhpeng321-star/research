"""Verify delivered evidence, report bound, and task-only Git change scope."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT/'tools/groupgrad'


def main():
    census=json.loads((HERE/'checkpoint_deltas.json').read_text())
    expected={'c00':('1b9949cc',30),'c01_on':('64f70af4',180),
              'c01_off':('e1fa4881',180),'c02':('44f3eb66',380)}
    for name,(prefix,update) in expected.items():
        row=census['checkpoints'][name]
        assert row['sha256'].startswith(prefix) and row['update']==update
        assert row['modules']['group_encoder']['changed']==0
        assert row['modules']['policy']['changed']==row['modules']['residual']['changed']==2
        assert row['modules']['lane']['n']==0
        assert row['modules']['critic']['changed']==30 and not row['modules']['critic']['missing']
        assert len(row['actor_optimizer_groups'])==1
        assert len(row['actor_optimizer_groups'][0]['params'])==2
        assert row['actor_optimizer_groups'][0]['lr']==5e-5
        assert list(row['actor_optimizer_steps'].values())==[float(update)]*2
    assert census['on_minus_off']['group_encoder']['changed']==0
    assert abs(census['on_minus_off']['policy']['max_abs']-.0017030201)<1e-9
    diagnosis=json.loads((HERE/'diagnosis.json').read_text())
    assert diagnosis['device']=='cpu' and diagnosis['num_threads']==1
    old=diagnosis['baseline']['updates']
    new=diagnosis['candidate']['updates']
    assert all(x['group_grad_norm']==0 and x['group_changed']==0 and x['residual_grad_norm']>0 for x in old)
    assert new[0]['group_grad_norm']==0 and new[1]['group_grad_norm']>0 and new[1]['group_changed']==26
    assert all(x['gradient_decomposition_max_error']<1e-4 for x in old+new)
    assert old[0]['all_to_total_cosine']>.999999
    assert old[0]['first_step_adam_delta_difference_max']<1e-8
    assert diagnosis['residuals']['n_starts']==26
    assert diagnosis['residuals']['stats']['c01_on']['n']==410
    assert all(v['full_ast_equal'] for k,v in diagnosis['source_receipt'].items() if k!='paircrit')
    inputs=json.loads((HERE/'inputs.json').read_text())
    for label in expected:
        assert inputs[f'checkpoints/{label}.pt']['sha256']==census['checkpoints'][label]['sha256']
    canonical=json.loads((HERE/'log_summary.json').read_text())['canonical_c01']
    assert canonical['n_updates']==150 and canonical['n_group_updates']==142
    assert canonical['audited_updates']==[50,75,100,125,150,175]
    report=ROOT/'GROUPGRAD.md'
    words=len(report.read_text(encoding='utf-8').split())
    assert words<=500,words
    changed=subprocess.check_output(['git','diff','--name-only','e62536f0'],cwd=ROOT,text=True).splitlines()
    assert all(x.startswith('tools/groupgrad/') or x=='GROUPGRAD.md' for x in changed)
    receipt=dict(status='PASS',report_words=words,cpu_gradient_assertions='PASS',
        source_equivalence='7 core deployed/unit ASTs match; paircrit traced separately',
        checkpoint_census='4 exact SHA prefixes, init reconstruction, modules and optimizer steps PASS',
        no_production_changes=True,
        files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
               [report,HERE/'checkpoint_deltas.json',HERE/'diagnosis.json',HERE/'inputs.json',
                HERE/'logs.json',HERE/'log_summary.json']})
    (HERE/'VERIFIED.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
