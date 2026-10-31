"""End-to-end native scenario setup, forcibly stopped BEFORE any SUMO launch."""
import sys
import time
from pathlib import Path
ROOT=Path('/root/c1floor20261031')
R=Path('/root/autodl-tmp/c4/speedhead20261030/e3final')
sys.path[:0]=[str(ROOT),'/root/wmdata2',str(R/'runtime/src'),str(R/'runtime')]
from core import audit,write
sys.addaudithook(audit)
from traffic_shepherd import simulation
from tools.ctrlarm.zero import run_e3
from traffic_shepherd.c4.evalprep import Measurement


def stop_before_launch(*args,**kwargs):
    raise RuntimeError('DRY_SETUP_READY_NO_SUMO')


simulation.start_sumo=stop_before_launch
out=ROOT/'jobs'/f'setup_dry_{time.time_ns()}'
out.mkdir()
try:
    run_e3(dict(seed=969400,demand=1900,penetration=.05,arm='native',warmup=300,horizon=300,charge=0),
           {'policy':str(R/'inputs/policy_latest.pt')},out,measurement=Measurement())
except RuntimeError as exc:
    assert 'DRY_SETUP_READY_NO_SUMO' in str(exc),str(exc)
    write(out/'SETUP_PASS.json',dict(passed=True,actual_sim_s=0,sumo_launch_calls=0))
    print('SETUP_PASS: native scenario generated; SUMO launch intercepted; zero physics steps')
else:
    raise RuntimeError('dry setup failed to intercept SUMO')
