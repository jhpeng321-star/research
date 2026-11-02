"""Durable CPU-only release checks, independent of SSH lifetime."""
import json
import os
from pathlib import Path
import sys
import time


def run(root):
    from unittest.mock import patch
    import fcntl
    root=Path(root)
    with (root/'LS_CPU.claim').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'LS_CPU_RESULT.json').exists():return
        os.nice(max(0,19-os.nice(0)))
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',COOPMWM_MAPPO='/home/ubuntu/mappospeed/production2/policy_00169.pt',
            COOPMWM_COST_SOURCE=str(root/'FROZEN_COSTS.json'),COOPMWM_MOTION_SOURCE=str(root/'FROZEN_MOTION_SOURCE.py'))
        import torch,pytest,signal
        torch.set_num_threads(1)
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('CPU test exceeded 30 seconds')))
        class Counts:
            def __init__(self):self.tests=0;self.skips=0
            def pytest_runtest_setup(self,item):signal.alarm(30)
            def pytest_runtest_logreport(self,report):
                if report.when=='call':self.tests+=1
                if report.skipped:self.skips+=1
                if report.when=='teardown':signal.alarm(0)
        c=Counts();start=time.monotonic()
        selection=root/'LS_RELEASE_TESTS.json'
        paths=([str(root/'pytools/tools/coopmwm'/(n+'.py')) for n in json.loads(selection.read_text())]
            if selection.exists() else sorted(str(p) for p in (root/'pytools/tools/coopmwm').glob('test_*cpu.py')))
        settings=root/'LS_CPU_SELECTION.json'
        extra=json.loads(settings.read_text()) if settings.exists() else {}
        with patch('subprocess.Popen',side_effect=AssertionError('CPU qualification forbids physics/inference')):
            code=pytest.main(paths+['-v','--tb=short','-p','no:cacheprovider']+extra.get('pytest_args',[]),plugins=[c])
        v=dict(pass_=code==0,tests=c.tests,skips=c.skips,sim_s=0,wall_s=time.monotonic()-start,root=str(root),
            selection=extra,source_hashes=json.loads((root/'LS_SOURCE_HASHES.json').read_text()))
        with (root/'LS_CPU_RESULT.json').open('x') as f:json.dump(v,f)
        print(json.dumps({k:v[k] for k in ('pass_','tests','skips','sim_s','wall_s','root')}),flush=True)


if __name__=='__main__':run(sys.argv[1])
