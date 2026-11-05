"""Archive this exact dead preparation's local reservation; signal nothing."""
import fcntl,json,os,subprocess,time
from pathlib import Path

root=Path('/tmp/c4-cpuv-1000')
with (root/'gate.lock').open('r+b') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=root/'active.json';value=json.loads(p.read_text())
    assert value['parent']==3554
    assert value['result'].endswith('/9496fa43c019487ba6a1ac291b246825.result')
    assert 'cpu_prepare_' in value['argv'][-1] and 'ea559c952ff0' in value['argv'][-1]
    assert not Path('/proc/3554').exists() and time.monotonic()>value['deadline']
    # Match only this owned remote preparation command and the supervisor
    # named by its reservation. No other command or PID is selected/signalled.
    check=subprocess.run(['pgrep','-af',r'ea559c952ff0|_session /tmp/c4-cpuv-1000/active[.]json'],capture_output=True,text=True)
    assert check.returncode==1,(check.stdout,check.stderr)
    target=root/'PLANFIX_CPU_PREPARE_DEAD_1833_ACTIVE.json'
    assert not target.exists()
    os.rename(p,target)
    print(json.dumps(dict(archived_own_dead_reservation=str(target),parent=3554,remote_signals=0)))
