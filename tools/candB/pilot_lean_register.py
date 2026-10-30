"""Append the actual staged lean recipe/sunk-cost/observer receipt once."""
from datetime import datetime,timezone
import json
from pathlib import Path

from tools.candB.pilot_lean_observe import append,LEDGER,DIRECTORY


if __name__=='__main__':
    marker=DIRECTORY/'pilot_lean_DESIGN_REGISTERED.json'
    if marker.exists(): raise RuntimeError('already registered; do not duplicate')
    doc=dict(utc=datetime.now(timezone.utc).isoformat(),label='EXPLORATORY-LEAN',
        source='38c7e22f3f9da3d81026833e3ca3429b6267f330',
        authority_sha256='f3fc51795a94927009dc2d865c2a89c47f0d49b982fd890a40545e96868f84ec',
        sunk_original_dry_wall_s=8530.659535884857,original_dry_updates=0,original_frozen_baseline_durable=False,
        retired_original_pids=[985766,970521,999762],
        panel_prefix=[dict(index=39780,world='w963552m1900p05t300',tick=600),
            dict(index=45082,world='w963625m1900p10t300',tick=2240),
            dict(index=35565,world='w963495m2400p20t300',tick=900),
            dict(index=64945,world='w963904m2400p40t300',tick=1700)],
        original_panel_sha256='3eaa69a170c17302660d12dd4d7fddb4657e43fa334eece17844dd0fa6c0411e',
        dense_scoring_futures=8,antithetic_ES_future_indices=[0,1,2,3],long_horizon_s=300,
        forward_workers=2,forward_fraction=.28,rescore_every=5,reuse_panel=True,skip_extra_grad_diagnostics=True,
        memory_rule='measured coordinator+worker caps+contexts+sampled peers+2GiB<=min90% device/45GiB',
        observer=json.loads((DIRECTORY/'pilot_lean_OBSERVER_OWNERSHIP.json').read_text()),
        monitoring_through_phase3='PENDING; ACK only after actual third update',R4='original phase10 finite raw dense/300s; dense below frozen zero')
    append(LEDGER,'\n- S-PILOTB EXPLORATORY-LEAN DESIGN/RETIREMENT/OBSERVER ACK '+json.dumps(doc)+'\n')
    with marker.open('x') as stream: json.dump(doc,stream,indent=1)
    print(json.dumps(doc))
