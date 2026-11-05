"""Install only the owner's single forced-command identity on cpuV; no RPC yet."""
import base64,hashlib,json,subprocess
from pathlib import Path
from planfix_ops import TEMP,remote

private=(TEMP/'id_ed25519_planfix_20261105').read_bytes()
public=(TEMP/'id_ed25519_planfix_20261105.pub').read_text().strip()
assert public.startswith('ssh-ed25519 ') and public.endswith('planfix-gpua-20261105')
host=subprocess.check_output(['ssh-keygen','-F','[connect.bjb1.seetacloud.com]:11000',
 '-f',str(Path.home()/'.ssh/known_hosts')],text=True)
known='\n'.join(line for line in host.splitlines() if line and not line.startswith('#'))+'\n'
assert known.strip() and all('ssh-' in line or 'ecdsa-' in line for line in known.splitlines())
config='''\n# PLANFIX owner-approved GPU RPC only identity, 2026-11-05
Host planfix-gpua
    HostName connect.bjb1.seetacloud.com
    Port 11000
    User root
    IdentityFile /home/ubuntu/.ssh/id_ed25519_planfix_20261105
    IdentitiesOnly yes
    UserKnownHostsFile /home/ubuntu/.ssh/known_hosts_planfix_20261105
    StrictHostKeyChecking yes
    BatchMode yes
    ConnectTimeout 20
    ConnectionAttempts 1
    ControlMaster auto
    ControlPersist 600
    ControlPath /home/ubuntu/.ssh/planfix-gpua-%C
    ServerAliveInterval 15
    ServerAliveCountMax 3
'''
code='''import base64,hashlib,json,os,subprocess,sys
from pathlib import Path
home=Path('/home/ubuntu/.ssh');assert home.is_dir()
payload=json.load(sys.stdin)
files={'id_ed25519_planfix_20261105':base64.b64decode(payload['private']),'id_ed25519_planfix_20261105.pub':__PUBLIC__,'known_hosts_planfix_20261105':__KNOWN__}
for name,data in files.items():
 p=home/name
 if p.exists():assert p.read_bytes()==data
 else:
  fd=os.open(p,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
  with os.fdopen(fd,'wb') as out:out.write(data);out.flush();os.fsync(out.fileno())
 assert p.stat().st_mode & 0o077==0
assert subprocess.check_output(['ssh-keygen','-y','-f',str(home/'id_ed25519_planfix_20261105')],text=True).split()[:2]==__PUBLIC_TEXT__.split()[:2]
p=home/'config';before=p.read_bytes() if p.exists() else b'';entry=__CONFIG__
backup=home/'config.bak_planfix_20261105'
if entry not in before:
 assert b'Host planfix-gpua' not in before
 if backup.exists():assert backup.read_bytes()==before
 else:
  with backup.open('xb') as out:out.write(before);out.flush();os.fsync(out.fileno())
  os.chmod(backup,0o600)
 with p.open('ab') as out:out.write(entry);out.flush();os.fsync(out.fileno())
 os.chmod(p,0o600)
receipt=dict(endpoint='root@planfix-gpua',port=11000,slot='gpuA-planfix1',key_scope='same single owner-approved forced warm_rpc key; no admin identity',
 public_sha256=hashlib.sha256(__PUBLIC__).hexdigest(),known_hosts_sha256=hashlib.sha256(__KNOWN__).hexdigest(),
 config_after_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),multiplexing='ControlMaster auto / ControlPersist600',new_sim_s=0)
p=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/CPU_GPU_ENDPOINT_PREPARED.json')
if p.exists():assert json.loads(p.read_text())==receipt
else:
 with p.open('x') as out:json.dump(receipt,out,sort_keys=True)
print(json.dumps(receipt),flush=True)
'''.replace('__PUBLIC_TEXT__',repr(public)).replace(
 '__PUBLIC__',repr((public+'\n').encode())).replace('__KNOWN__',repr(known.encode())).replace('__CONFIG__',repr(config.encode()))
remote('cpuV',code,timeout=25,stdin_data=json.dumps(dict(private=base64.b64encode(private).decode())).encode())
