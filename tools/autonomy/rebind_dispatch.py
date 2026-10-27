"""Replace only this task's verified childless waiting successor after dispatch fix."""
from tools.loopresume.operate import remote
from tools.autonomy.transport import stage, launch


if __name__ == '__main__':
    stage('gpuA')
    remote('gpuA', '''import pathlib,json,os,signal,time
r=pathlib.Path('/root/autodl-tmp/c4/restart20261026')
d=json.loads((r/'AUTONOMY_PID.json').read_text()); pid=d['pid']
s=json.loads((r/'AUTONOMY_STATUS.json').read_text())
assert s['pid']==pid and s['status']=='WAIT_MAIN_C2H60B',s
stat=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
assert stat[19]==d['start_ticks'] and os.getsid(pid)==pid
assert not pathlib.Path(f'/proc/{pid}/task/{pid}/children').read_text().strip()
source=(r/'autonomy/remote.py').read_text(); assert 'loop.fast_module = lambda: fast' in source
assert not (r/'AUTONOMY_PID.before_dispatch_fix.json').exists()
os.kill(pid,signal.SIGTERM)
for i in range(100):
 p=pathlib.Path(f'/proc/{pid}/stat')
 if not p.exists() or p.read_text().rsplit(')',1)[1].split()[0]=='Z': break
 time.sleep(.1)
else: raise RuntimeError('owned waiter did not exit')
(r/'AUTONOMY_PID.json').rename(r/'AUTONOMY_PID.before_dispatch_fix.json')
(r/'arm/driver_autonomy.log').rename(r/'arm/driver_autonomy.before_dispatch_fix.log')
print(json.dumps({'retired_childless_waiter':d,'predecessor_untouched':s['predecessor']}))
''')
    launch('gpuA')
