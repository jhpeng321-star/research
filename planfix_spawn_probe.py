"""Exact spawn-path startup probe; every attempted SUMO Popen is blocked."""
import json,os,subprocess,time,traceback
from pathlib import Path
import torch
import planfix_run_bank as bank


def worker(spec,snapshot,root):
    import shutil
    print('SPAWN_ENV',os.environ.get('SUMO_HOME'),shutil.which('sumo'),flush=True)
    original=bank.sv.LocalBranchRunner.run
    def traced(self,*args,**kwargs):
        try:return original(self,*args,**kwargs)
        except BaseException:
            traceback.print_exc();raise
    bank.sv.LocalBranchRunner.run=traced
    def refuse(*args,**kwargs):
        print('SPAWN_POPEN_BLOCKED',repr(args[0]),flush=True)
        raise RuntimeError('ZERO_SUMO_SPAWN_PROBE')
    subprocess.Popen=refuse
    return bank.branch(spec,snapshot,root)


if __name__=='__main__':
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    os.nice(19-os.nice(0));torch.set_num_threads(1)
    work=Path('/home/ubuntu/coopmwm20261105_clean1/plan_fix')
    snapshot=torch.load(work/'bank_verification_v4/bank_02713af428706f46/FROZEN_SNAPSHOT.pt',weights_only=False,map_location='cpu')
    item=next(x for x in json.loads((work/'BANK_INVENTORY.json').read_text())['rows'] if x['id']=='bank_02713af428706f46')
    spec=bank.sv.build_branches(snapshot,[bank.library_plan(item['source_plan'],'zero_sumo_spawn')])[0]
    root=work/('diagnostic_spawn_'+str(time.time_ns()))
    with ProcessPoolExecutor(max_workers=1,mp_context=multiprocessing.get_context('spawn')) as pool:
        print('SPAWN_PROBE_RESULT',pool.submit(worker,spec,snapshot,str(root)).result(timeout=20),flush=True)
