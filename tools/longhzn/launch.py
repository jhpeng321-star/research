"""Deploy own runner and detach CPU coordinator, sequentially per box."""
import argparse
from pathlib import Path
from operate import remote

p = argparse.ArgumentParser()
p.add_argument('host',choices=('cpuP','cpuQ'))
a = p.parse_args()
data = Path(__file__).with_name('run_cpu.py').read_text()
source = f'''import hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path.home()/'longhzn'; assert root.is_dir()
target=root/'run_cpu.py'; data={data!r}
assert not target.exists(); target.write_text(data)
assert not (root/'LAUNCH.json').exists()
env={{**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}}
with (root/'coordinator.log').open('x') as log:
 child=subprocess.Popen([sys.executable,'-B',str(target),'coordinate',{('0' if a.host=='cpuP' else '1')!r}],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
ticks=pathlib.Path(f'/proc/{{child.pid}}/stat').read_text().rsplit(')',1)[1].split()[19]
proof={{'pid':child.pid,'start_ticks':ticks,'source_sha256':hashlib.sha256(data.encode()).hexdigest(),'task':'S-LONGHZN-1'}}
(root/'LAUNCH.json').write_text(json.dumps(proof)); print(json.dumps(proof))
'''
remote(a.host, source)
