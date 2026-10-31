"""Dry bundle/controller validation; no SUMO subprocess or step."""
from pathlib import Path
import json
import os
import sys
import torch

R=Path('/home/ubuntu/e3final/runtime')
sys.path[:0]=[str(R/'src'),str(R)]
from prepare_bundle import prepare
from supervisor import jobs,ROOT


def main():
    torch.set_num_threads(1)
    panel=jobs();assert len(panel)==144 and sum(j['charge'] for j in panel)==86544
    output=ROOT/'preflight';output.mkdir(exist_ok=False)
    bundles=[]
    for variant in ('default','inferred'):
        job=next(j for j in panel if j['variant']==variant)
        d=output/variant;d.mkdir()
        bundle,config=prepare(job,d);bundles.append(bundle)
        assert len(bundle['vehicles'])==71
        assert all(v['speed']>=0 for v in bundle['vehicles'])
        assert not any(v['lane_id'].startswith(':') for v in bundle['vehicles'])
    a,b=bundles
    assert a['receipt']['future_request_sha256']==b['receipt']['future_request_sha256']
    assert a['cav_ids']==b['cav_ids']
    assert a['receipt']['generated_routes_sha256']!=b['receipt']['generated_routes_sha256']
    from model_env import ObservableEnv
    from traffic_shepherd.controller_factory import create_pi_session
    from traffic_shepherd.config import load_config
    from wms.control.policy import load_policy_fn
    pi=create_pi_session(load_config(),a['cav_ids'],desired_speed_mps=31.29)
    policy=load_policy_fn('/home/ubuntu/mappospeed/production2/policy_latest.pt',device='cpu')
    result=dict(status='PASS',sim_s=0,jobs=len(panel),precharge=86544,
                common_demand=True,common_cavs=True,default_types=len({v['type'] for v in a['vehicles']}),
                inferred_types=len({v['type'] for v in b['vehicles']}),
                bundle_receipts=[x['receipt'] for x in bundles])
    with (output/'receipt.json').open('x') as f:json.dump(result,f,indent=1)
    print(json.dumps({k:v for k,v in result.items() if k!='bundle_receipts'}))


if __name__=='__main__':main()
