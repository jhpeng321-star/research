"""Final read-only identity audit of only this workstream's recorded worker PIDs."""
import json
import time

from .operate import RECEIPTS,remote


def main():
    out=RECEIPTS/'OWNED_JOBS_FINAL.json'
    assert out.parent.is_dir() and not out.exists()
    workers=[]
    for path in sorted(RECEIPTS.glob('*_DISPATCH.json')):
        record=json.loads(path.read_text())
        assert record['status']=='DISPATCHED'
        assert record['root'].startswith('/home/ubuntu/rblane20261031_')
        workers.append(dict(receipt=path.name,**record['process']))
    code=f'''import pathlib,json
workers={workers!r}
for row in workers:
    stat=pathlib.Path('/proc/'+str(row['pid'])+'/stat')
    fields=stat.read_text().split() if stat.exists() else None
    row['matching_process_alive']=bool(fields and fields[21]==row['start_ticks'] and fields[2]!='Z')
    row['observed_state']=fields[2] if fields and fields[21]==row['start_ticks'] else 'absent_or_identity_changed'
assert not any(r['matching_process_alive'] for r in workers)
print(json.dumps(workers))
'''
    checked=json.loads(remote(code))
    result=dict(status='PASS',checked_at=time.time(),recorded_workers=len(checked),
                matching_owned_workers_alive=0,signals_sent=0,workers=checked)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',recorded_workers=len(checked),matching_owned_workers_alive=0,signals_sent=0)))


if __name__=='__main__': main()
