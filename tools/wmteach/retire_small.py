"""Retire only the obsolete small-corpus continuation AFTER exact quality closes."""
import json

from tools.wmteach.ops import remote,ROOT,REMOTE


def main():
    data=json.loads(remote(f'''import pathlib,json,time,sys
root=pathlib.Path({REMOTE!r})
if not (root/'QUALITY.json').is_file():
 print(json.dumps(dict(state='WAIT_QUALITY',stop_requested=False)));sys.exit()
rows=json.loads((root/'QUALITY_RECORDS.json').read_text())
assert len(rows)==24 and len({{(r['state_id'],r['horizon_s']) for r in rows}})==24
owner=json.loads((root/'DISPATCH.json').read_text());p=pathlib.Path(f'/proc/{{owner["pid"]}}/stat')
st=p.read_text().split() if p.exists() else None
identity=bool(st and st[21]==owner['start_ticks'] and
 pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()==owner['boot_id'])
assert identity,'ownership changed; no process action'
reason='OWNER1253 fixed milestone retirement: exact quality complete; prioritize5120state continuation, not small snapshots'
marker=root/'OWNER1253_RETIREMENT.json'
if not marker.exists():
 with marker.open('x') as f:json.dump(dict(unix=time.time(),owner=owner,reason=reason,
  complete_quality_states=12,retain_all_futures=True,cooperative_boundary_only=True),f)
if not (root/'STOP').exists():
 with (root/'STOP').open('x') as f:f.write(reason+'\\n')
print(json.dumps(dict(state='COOPERATIVE_RETIREMENT_REQUESTED',stop_requested=True,identity_matches=identity,
 owner=owner,receipt=json.loads(marker.read_text()),new_sumo_sim_s=0)))
'''))
    if data['stop_requested']:
        path=ROOT/'docs/paper_results/wmteach/SMALL_RETIREMENT.json'
        if not path.exists():
            with path.open('x',encoding='utf-8') as f:json.dump(data,f,sort_keys=True);f.write('\n')
    print(json.dumps(data))


if __name__=='__main__':main()
