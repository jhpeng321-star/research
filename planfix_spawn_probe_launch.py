from pathlib import Path
from planfix_ops import remote
source=Path(__file__).with_name('planfix_spawn_probe.py').read_text()
code=r'''import os,subprocess
from pathlib import Path
ops=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix/ops_v4_ownerstopped')
p=ops/'planfix_spawn_probe.py';text=__SOURCE__
if p.exists():assert p.read_text()==text
else:
 with p.open('x') as out:out.write(text)
source='/home/ubuntu/coopmwm20261105_clean1_src_7cc6edaa'
env=dict(os.environ,PYTHONPATH=str(ops)+':'+source+'/pytools:/home/ubuntu/e3final/runtime:/home/ubuntu/e3final/runtime/src:'+source+'/runtime_src:'+source+'/WM_PACKAGE',
 CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 LD_PRELOAD='/home/ubuntu/cpus_env/libmkl_intel_shim.so',PYTHONDONTWRITEBYTECODE='1')
env['SUMO_HOME']='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo'
env['PATH']=env['SUMO_HOME']+'/bin:/home/ubuntu/miniconda3/bin:'+env['PATH']
subprocess.run(['/home/ubuntu/miniconda3/bin/python','-B',str(p)],env=env,cwd=ops,check=True,timeout=25)
'''.replace('__SOURCE__',repr(source))
remote('cpuV',code,timeout=35)
