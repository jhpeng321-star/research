"""Schema adapters only: invoke the byte-identical C1CLOSED label/field scorer."""
from pathlib import Path
import json
import sys
from core import digest, write, audit

ROOT=Path('/root/c1floor20261031')
sys.path.insert(0,str(ROOT))
sys.addaudithook(audit)


def extract():
    import labels
    from run import jobs
    assert json.loads((ROOT/'COMPLETE.json').read_text())['passed']
    binding=json.loads((ROOT/'PREFLIGHT_BINDING.json').read_text())
    assert digest(ROOT/'labels.py')==binding['source_hashes']['labels.py']
    assert digest(ROOT/'aggregates.py')==binding['source_hashes']['aggregates.py']
    all_rows=[]
    for variant in ('reference','alt0','alt1'):
        view=ROOT/f'label_view_{variant}'
        view.mkdir(exist_ok=False);(view/'jobs').mkdir()
        for j in jobs():
            source=ROOT/'jobs'/j['id']
            target=view/'jobs'/f"s{j['seed']}_native"
            target.mkdir()
            tape=ROOT/'reference'/f"s{j['seed']}_native"/'traj_base.npz' if variant=='reference' else source/f'{variant}.npz'
            for name,path in (('start.pt',source/'start.pt'),('start_history.npz',source/'start_history.npz'),('traj_base.npz',tape)):
                (target/name).symlink_to(path)
            # Explicit adapter record, not a fabricated original-run receipt.
            write(target/'result.json',dict(ok=True,output_hashes={'traj_base.npz':digest(tape)},
                  schema='c1floor_label_adapter/1',variant=variant,source_job=j['id'],
                  actual_receipt_sha256=digest(source/'result.json')))
            labels.ROOT=view
            row=labels.one(j,'native')
            row.update(variant=variant,source_job=j['id'])
            all_rows.append(row)
    write(ROOT/'LABELS.json',dict(schema='c1floor_labels/1',rows=all_rows,
          source_sha256=digest(ROOT/'labels.py'),aggregates_sha256=digest(ROOT/'aggregates.py'),
          adapter_sha256=digest(__file__)))
    print(json.dumps(dict(rows=len(all_rows),source_sha256=digest(ROOT/'labels.py'))))


if __name__=='__main__':extract()
