"""Read exact remaining CUDA-only device/timing statements before CPU porting."""
from planfix_ops import remote
CODE=r'''
from pathlib import Path
r=Path('/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa/pytools/tools/wmteach')
for name in ('runtime.py','scale.py','marginal.py'):
 text=(r/name).read_text();print('MODULE',name,flush=True)
 for i,line in enumerate(text.splitlines(),1):
  if 'cuda' in line.lower():print(i,line,flush=True)
'''
if __name__=='__main__':remote('cpuV',CODE,timeout=25)
