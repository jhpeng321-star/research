"""Real UNC reader contract: source-zero LC draws, strict-positive IDM, replay."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def load_reader():
    candidate = os.environ.get("RBFIT_LC_UNC_MODULE")
    if candidate:
        spec = importlib.util.spec_from_file_location("rbfit_lc_unc_candidate",candidate)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.load_boundary_fitted_state
    from wms.physics.lc_fit_for_unc import load_boundary_fitted_state
    return load_boundary_fitted_state


@pytest.fixture
def artifacts(tmp_path):
    root,lane = tmp_path/"idm",tmp_path/"lane"
    root.mkdir()
    lane.mkdir()
    idm_order = ["v0","time_headway","accel","decel","min_gap"]
    lc_order = ["lcStrategic","lcCooperative","lcSpeedGain","lcKeepRight","lcAssertive"]
    pop = {"schema":"wm_phys.idm_population.v1","parameter_order":idm_order,
        "distribution":"finite_support_mixture","support":[[30,1,1,2,2]],"weights":[1]}
    post = {"schema":"wm_phys.idm_posteriors.v1","conditioning_end_s":300,"drivers":{}}
    lp = {"schema":"wm_phys.lc2013_population.v1","parameter_order":lc_order,
        "distribution":"finite_support_mixture","support":[[0,0,0,0,1],[1,1,1,1,2]],"weights":[0.4,0.6]}
    ls = {"schema":"wm_phys.lc2013_posteriors.v1","conditioning_end_s":300,
        "drivers":{"969400:8":{"support_indices":[0],"weights":[1]}}}
    for path,payload in ((root/"idm_population.json",pop),(root/"idm_posteriors.json",post),
                         (lane/"lc2013_population.json",lp),(lane/"lc2013_posteriors.json",ls)):
        path.write_text(json.dumps(payload))
    return root,lane


def test_real_unc_boundary_draws_and_population_replay(load_reader,artifacts):
    import torch
    root,lane = artifacts
    reader = load_reader(root,lane_directory=lane)
    reader.check_time(300)
    draw = reader.draw(969400,8,lane=True,generator=torch.Generator().manual_seed(1))
    assert draw==dict(zip(reader.lane_population["parameter_order"],[0,0,0,0,1]))
    for vid in (None,999):
        a = reader.draw(969400,vid,lane=True,generator=torch.Generator().manual_seed(12))
        b = reader.draw(969400,vid,lane=True,generator=torch.Generator().manual_seed(12))
        assert a==b
    idm = reader.draw(969400,8,generator=torch.Generator().manual_seed(1))
    assert idm["v0"]==30 and idm["decel"]==2


@pytest.mark.parametrize("family,coordinate,value",[("idm",0,0),("lc",4,0),("lc",1,1.1)])
def test_idm_and_lc_domains_remain_distinct(load_reader,artifacts,family,coordinate,value):
    root,lane = artifacts
    path = root/"idm_population.json" if family=="idm" else lane/"lc2013_population.json"
    pop = json.loads(path.read_text())
    pop["support"][0][coordinate]=value
    path.write_text(json.dumps(pop))
    with pytest.raises(ValueError):
        load_reader(root,lane_directory=lane)


def test_real_unc_plans_keep_zero_lc_values_and_replay(load_reader,artifacts):
    import torch
    from wms.control.hidden_state import plan_hidden_state_futures
    from wms.model.driver_encoder import PARAMETER_NAMES
    root,lane = artifacts
    reader = load_reader(root,lane_directory=lane)
    vector = torch.tensor([30.,1.,2.,1.,2.,.4,.2,8.,1.])
    params = {name:vector[j].expand(1,3).clone() for j,name in enumerate(PARAMETER_NAMES)}
    class Model:
        physics = SimpleNamespace(dt_s=.5,entry_learned=False)
        def encode_drivers(self,state,**kwargs):
            return {},None,params
    state = {"present":torch.tensor([[True,True,False]]),"is_cav":torch.tensor([[False,True,False]])}
    options = dict(future_sampling="hidden_state",n_samples=4,seed=51,t0_s=300,
        row_keys=[969400],vehicle_ids=torch.tensor([[8,9,-1]]),fitted_state=reader)
    before = torch.get_rng_state().clone()
    a = plan_hidden_state_futures(Model(),state,{},**options)
    b = plan_hidden_state_futures(Model(),state,{},**options)
    assert torch.equal(before,torch.get_rng_state())
    for first,replay in zip(a,b):
        for name in reader.lane_population["parameter_order"][:4]:
            assert first.kwargs["driver_params"][name][0,0]==0
            assert torch.equal(first.kwargs["driver_params"][name][0,1:],torch.ones(2))
        for name,value in first.kwargs["driver_params"].items():
            assert torch.equal(value,replay.kwargs["driver_params"][name])
