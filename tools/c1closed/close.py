"""Read-only owned-resource audit and explicit encoding-stop resolution receipt."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import subprocess

from operate import HERE, REMOTE
from probe import ssh
from watch import creation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=1)


def main():
    hosts = {}
    for host in ('gpuA','gpuR'):
        hosts[host] = json.loads(ssh(host,f'''from pathlib import Path
import sys,json
sys.path.insert(0,{REMOTE!r});import expand_runner as r
p=Path({REMOTE!r});live=[];checked=0
receipts=list((p/'evidence').glob('*_OWNED.json'))
receipts += [p/n for n in ('DISPATCH.json','EXPAND_DISPATCH.json','evidence/TAIL_RECOVERY_LAUNCH.json') if (p/n).exists()]
for q in receipts:
 x=json.loads(q.read_text())
 if 'pid' not in x or 'start_ticks' not in x:continue
 checked+=1;stat=Path('/proc/'+str(x['pid'])+'/stat')
 if stat.exists():
  s=stat.read_text().split(') ',1)[1].split()
  if s[19]==x['start_ticks'] and s[0]!='Z':live.append(dict(pid=x['pid'],state=s[0],receipt=q.name))
free,owned=r.memory()
print(json.dumps(dict(checked_owned_receipts=checked,live_owned=live,owned_gpu_mib=owned,free_gpu_mib=free,panel_files=len(list((p/'panel').glob('*_h300.json'))))))
'''))
        assert not hosts[host]['live_owned'] and hosts[host]['owned_gpu_mib'] == 0
    local = {}
    for name in ('WATCH_LAUNCH.json','EXPAND_BRIDGE_LAUNCH.json'):
        x = json.loads((HERE/'evidence'/name).read_text())
        local[name] = dict(pid=x['pid'],alive=creation(x['pid']) == x['creation_time'])
        assert not local[name]['alive']
    for p in (HERE/'evidence').glob('PREFETCH_*_LAUNCH.json'):
        x = json.loads(p.read_text())
        assert creation(x['pid']) != x['creation_time']
    watch = json.loads((HERE/'evidence/WATCH_LAUNCH.json').read_text())
    assert all(digest(HERE/n) == v for n,v in watch['source_hashes'].items())
    check = json.loads((HERE/'evidence/DELIVERY_VERIFIED.json').read_text())
    assert check['passed'] and check['tasks'] == 480
    blocked = json.loads((HERE/'evidence/WATCH_BLOCKED.json').read_text())
    assert 'UnicodeEncodeError' in blocked['error'] and 'gbk' in blocked['error']
    root = HERE.parents[1]
    scientific = subprocess.check_output(['git','rev-parse','b40939aa'],cwd=root,text=True).strip()
    audit = dict(passed=True,at=datetime.now().astimezone().isoformat(),hosts=hosts,
                 local=local,all_frozen_collector_sources_unchanged=True,
                 delivery_sha256=digest(HERE/'evidence/DELIVERY_VERIFIED.json'))
    write(HERE/'evidence/OWNED_RESOURCES_CLOSED.json',audit)
    write(HERE/'evidence/WATCH_COMPLETE.json',dict(finished=audit['at'],commit=scientific,
        delivery_sha256=audit['delivery_sha256'],manual_resolution=True,
        reason='Original sole collector finished sync, tests, complete-panel analysis and delivery verification; GBK stdout failed while displaying staged diff before commit. Reviewed intended index and committed as b40939aa without rerunning analysis. Presentation-only footer/scope update subsequently reverified; all three metric CSVs unchanged.',
        blocked_receipt_retained=True,presentation_receipt_sha256=digest(HERE/'evidence/PRESENTATION_UPDATED.json'),
        owned_resource_receipt_sha256=digest(HERE/'evidence/OWNED_RESOURCES_CLOSED.json')))
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
