"""Read-only process/resource receipt; never signals any process."""
from remote import ssh

SOURCE = r'''
import os, subprocess, json
print('LOAD', os.getloadavg())
print('CPU', os.cpu_count())
for cmd in (['ps','-eo','pid,ppid,ni,etime,args'],
            ['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv']):
 try:
  p = subprocess.run(cmd, capture_output=True, text=True)
  if cmd[0] == 'ps':
   print('\n'.join(x for x in p.stdout.splitlines()
                   if any(s in x.lower() for s in ('sumocore','obsstate','sumo ', 'wmdata'))
                   and 'base64' not in x))
  else: print(p.stdout, p.stderr)
 except FileNotFoundError as e: print(str(e))
'''

if __name__ == '__main__':
    for host in ('cpuQ', 'gpuR'):
        print(host, ssh(SOURCE, host).decode())
