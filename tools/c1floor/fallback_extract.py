"""Invoke frozen C1CLOSED scorers on the fixed chaos panel; no new metrics."""
from pathlib import Path
import json
import sys
import os

ROOT=Path('/root/c1floor20261031')
sys.path.insert(0,str(ROOT))
from core import digest,write,audit
sys.addaudithook(audit)


def main():
    import labels
    import numpy as np
    import torch
    torch.set_num_threads(1)
    binding=json.loads((ROOT/'PREFLIGHT_BINDING.json').read_text())
    for name in ('labels.py','aggregates.py'):
        assert digest(ROOT/name)==binding['source_hashes'][name]
    worlds=json.loads(Path('/root/wmcalib/data/design.json').read_text())['worlds'][:6]
    rows=[];effects=[];results=[]
    for w in worlds:
        source=ROOT/'jobs'/f"pulse_s{w['seed']}"
        r=json.loads((source/'result.json').read_text());assert r['ok'] and r['actual_sim_s']==1200
        assert all(x['identical'] for x in r['reproduction'].values())
        results.append(r)
        reference=Path(r['original_reference'])
        assert digest(reference)==r['original_reference_sha256']
        assert all(abs(p['delta_mps']-.01)<1e-10 for p in r['pulses'])
        with np.load(reference,allow_pickle=False) as ref:
            meta=json.loads(str(ref['meta']));excluded=[meta['vids'].index(t) for t in r['targets']]
            for k,target in enumerate(excluded):
                with np.load(source/f'alt{k}.npz',allow_pickle=False) as alt:
                    ia=np.flatnonzero((ref['t']==300.5)&(ref['vid']==target))
                    ib=np.flatnonzero((alt['t']==300.5)&(alt['vid']==target))
                    assert len(ia)==len(ib)==1
                    effects.append(dict(seed=w['seed'],draw=k,target=r['targets'][k],target_index=target,
                        first_tick_speed_delta=float(alt['v'][ib[0]])-float(ref['v'][ia[0]]),
                        first_tick_position_delta=float(alt['s'][ib[0]])-float(ref['s'][ia[0]]),
                        initial_delta_mps=r['pulses'][k]['delta_mps'],
                        tape_identical=r['outputs'][f'alt{k}.npz']==r['original_reference_sha256']))
        for variant in ('reference','alt0','alt1'):
            view=ROOT/'pulse'/f'label_view_{variant}'
            (view/'jobs').mkdir(parents=True,exist_ok=True)
            target=view/'jobs'/f"s{w['seed']}_native";target.mkdir()
            original=ROOT/'reference'/f"s{w['seed']}_native"
            tape=reference if variant=='reference' else source/f'{variant}.npz'
            assert digest(tape)==(r['original_reference_sha256'] if variant=='reference' else r['outputs'][tape.name])
            for name,path in (('start.pt',original/'start.pt'),('start_history.npz',original/'start_history.npz'),('traj_base.npz',tape)):
                (target/name).symlink_to(path)
            write(target/'result.json',dict(ok=True,output_hashes={'traj_base.npz':digest(tape)},
                  schema='c1floor_fallback_label_adapter/1',source_receipt_sha256=digest(source/'result.json')))
            labels.ROOT=view;row=labels.one(w,'native')
            row.update(variant=variant,excluded_target_indices=excluded,targets=r['targets'])
            rows.append(row)
    write(ROOT/'pulse/LABELS.json',dict(schema='c1floor_chaos_labels/1',rows=rows,
          source_sha256=digest(ROOT/'labels.py'),aggregates_sha256=digest(ROOT/'aggregates.py'),adapter_sha256=digest(__file__)))
    write(ROOT/'pulse/EFFECT_AUDIT.json',dict(effects=effects,identical_draws=sum(x['tape_identical'] for x in effects)))
    ledgers=[json.loads(s) for p in (ROOT/'accounting.jsonl',ROOT/'fallback/accounting.jsonl',ROOT/'pulse/accounting.jsonl') for s in p.read_text().splitlines()]
    charged=sum(x['sim_s'] for x in ledgers)
    old=[json.loads(p.read_text()) for p in (ROOT/'jobs').glob('fallback_*/*.json') if p.name in ('result.json','failure.json')]
    actual=2400+sum(r['actual_sim_s'] for r in old)+sum(r['actual_sim_s'] for r in results)
    assert charged==29700 and actual==21900
    owners=[]
    for p in list((ROOT/'fallback').glob('*_PID.json'))+list((ROOT/'pulse').glob('*_PID.json')):
        owner=json.loads(p.read_text());q=Path('/proc')/str(owner['pid'])
        state=q.joinpath('stat').read_text().split()[2] if q.exists() else 'exited'
        assert state in ('exited','Z');owners.append(dict(**owner,state=state))
    write(ROOT/'pulse/COMPLETE.json',dict(ok=True,method='instantaneous_speed_state_pulse_chaos',worlds=6,
          draws=12,actual_sim_s=actual,charged_sim_s=charged,original_continuations_bitwise=6,
          workers=1,nice=19,owners=owners,labels_sha256=digest(ROOT/'pulse/LABELS.json'),
          result_sha256={str(w['seed']):digest(ROOT/'jobs'/f"pulse_s{w['seed']}"/'result.json') for w in worlds}))
    print(json.dumps(dict(rows=len(rows),worlds=6,actual_sim_s=actual,charged_sim_s=charged)))


if __name__=='__main__':main()
