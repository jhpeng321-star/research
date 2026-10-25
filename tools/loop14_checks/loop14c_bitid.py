"""Flags-off bit-identity probe: seeded Policy14 imagination (all + counterfactual rows) on a
synthetic first-passage WM (hybrid law, 13E veto, ramp lane-end stop).  usage: bitid.py ROOT OUT [flags]"""
import sys
from dataclasses import replace
root, out = sys.argv[1], sys.argv[2]
flags = sys.argv[3] if len(sys.argv) > 3 else ""
sys.path.insert(0, root + "/src")
import torch
from wms.control.train_cli import _read_config, GroupSignal
from wms.control.dreamer import DreamerConfig
from wms.control.contract import RewardWeights
from wms.control.counterfactual import CounterfactualConfig, build_variants
from wms.control.lanes import LaneRules
from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
from wms.control.obs import build_observations
from wms.control.safety import safe_accel
from wms.control.objective14 import authoritative_reward
from wms.control import policy14
from wms.model import ModelSpec, WmSModel
from wms.physics.config import PhysicsConfig
torch.manual_seed(0)
torch.set_num_threads(1)
m = WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5", merge_decision="first_passage",
                       lane_decision="sample", merge_passage_law="hybrid"),
             physics=PhysicsConfig(secure_gap_veto=True, ramp_lane_end_stop=True)).eval()
raw = _read_config(root + "/configs/c4_policy14.toml")
cfg = replace(DreamerConfig(**raw["trainer"]), real_return_mix=0.0, horizon=6, batch_size=2)
lanes = dict(raw["lanes"]); lanes.pop("enabled")
factory = policy14.controller_factory
if flags:
    p, n = flags.split(",")
    factory = policy14.make_controller_factory(policy_ramp_merge=p == "1", native_ramp_merge=n)
cf = dict(raw["counterfactual"]); cf.update(group_radius_m=200.0)
sampler = policy14.SmokeSampler(m)
trainer = policy14.Policy14Trainer([m], CavPolicy(accel_max=1.2, lane_head=True), GroupSignal(RsuGroupEncoder()),
                                   TeamCritic(entry_backlog=True), sampler, observation_builder=build_observations,
                                   reward_fn=authoritative_reward, safety=safe_accel, controller_factory=factory,
                                   config=cfg, reward_weights=RewardWeights(**raw["reward"]), device="cpu",
                                   lane_rules=LaneRules.from_mapping(lanes),
                                   counterfactual=CounterfactualConfig(**cf))
torch.manual_seed(1)
batch = sampler.sample(2)
batch = trainer._entry_slots(batch, torch.Generator().manual_seed(3))
variants = build_variants(batch.state, trainer.counterfactual, torch.Generator().manual_seed(4))
res = {}
for det in (True, False):
    torch.manual_seed(7)
    with torch.no_grad():
        imag = trainer.imagine(batch, deterministic=det, variants=variants, future_seed=11)
    for k, v in imag.items():
        if torch.is_tensor(v):
            res[f"{det}/{k}"] = v.detach().clone()
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if torch.is_tensor(vv):
                    res[f"{det}/{k}/{kk}"] = vv.detach().clone()
        elif isinstance(v, list):
            for i, vv in enumerate(v):
                if torch.is_tensor(vv):
                    res[f"{det}/{k}/{i}"] = vv.detach().clone()
res["n_rows"] = torch.tensor(variants.n_rows)
torch.save(res, out)
print("saved", len(res), "tensors; rows", variants.n_rows, "kinds", variants.row_kind.tolist())
