"""S-COOPW flags-off bit identity: seeded Policy14 imagination + two full ``update()`` calls on a
synthetic first-passage WM (hybrid law, 13E veto, ramp lane-end stop; same scene as
``tools/loop14_checks/loop14c_bitid.py``), run under one source tree.

    python tools/coopw/bitid_coopw.py ROOT CONFIG OUT.pt [--full]   # run (ROOT = source tree;
                                                    # --full: curriculum end, flags-on smoke)
    python tools/coopw/bitid_coopw.py --compare A.pt B.pt [allowed-new-key ...]  # exact

Saved: every imagination tensor (deterministic and sampled, all/base/individual/group/flip rows),
every finite numeric update metric except wall time, and the actor/critic parameters after the
two updates.  ``CONFIG`` is read from ROOT; an ``[uncertainty]`` section is passed only if present.
"""
import math
import sys


def flatten(prefix, value, out):
    import torch
    if torch.is_tensor(value):
        out[prefix] = value.detach().clone()
    elif isinstance(value, dict):
        for key, item in value.items():
            flatten(f"{prefix}/{key}", item, out)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            flatten(f"{prefix}/{index}", item, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        out[prefix] = torch.tensor(float(value), dtype=torch.float64)


def run(root, config, out, full=False):
    sys.path.insert(0, root + "/src")
    from dataclasses import replace

    import torch
    from wms.control import policy14
    from wms.control.contract import RewardWeights
    from wms.control.counterfactual import CounterfactualConfig, build_variants
    from wms.control.dreamer import DreamerConfig
    from wms.control.lanes import LaneRules
    from wms.control.objective14 import authoritative_reward
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder, TeamCritic
    from wms.control.safety import safe_accel
    from wms.control.train_cli import GroupSignal, _read_config
    from wms.model import ModelSpec, WmSModel
    from wms.physics.config import PhysicsConfig

    torch.manual_seed(0)
    torch.set_num_threads(1)
    m = WmSModel(ModelSpec(rollout_mode="driver_params", lane_model="hazard_v5",
                           merge_decision="first_passage", lane_decision="sample",
                           merge_passage_law="hybrid"),
                 physics=PhysicsConfig(secure_gap_veto=True, ramp_lane_end_stop=True)).eval()
    raw = _read_config(root + "/" + config)
    cfg = replace(DreamerConfig(**raw["trainer"]), real_return_mix=0.0, horizon=6, batch_size=1 if full else 2,
                  checkpoint_every=0, validation_every=0)
    lanes = dict(raw["lanes"])
    lanes.pop("enabled")
    switches = raw["policy14"]
    factory = policy14.make_controller_factory(switches.get("policy_ramp_merge", False),
                                               switches.get("native_ramp_merge", "gap6"))
    cf = dict(raw["counterfactual"])
    cf.update(group_radius_m=200.0)
    extra = {}
    if any(k in raw for k in ("uncertainty", "exploration", "horizon")):
        from wms.control import coopw as _coopw
    for section, cls in (("uncertainty", "UncertaintyConfig"), ("exploration", "ExplorationConfig"),
                         ("horizon", "HorizonConfig")):
        if section in raw:
            extra[section] = getattr(_coopw, cls)(**raw[section])  # noqa: F821
    sampler = policy14.SmokeSampler(m)
    trainer = policy14.Policy14Trainer(
        [m], CavPolicy(accel_max=1.2, lane_head=True), GroupSignal(RsuGroupEncoder()),
        TeamCritic(entry_backlog=True), sampler, observation_builder=build_observations,
        reward_fn=authoritative_reward, safety=safe_accel, controller_factory=factory, config=cfg,
        reward_weights=RewardWeights(**raw["reward"]), device="cpu",
        lane_rules=LaneRules.from_mapping(lanes), counterfactual=CounterfactualConfig(**cf), **extra)
    trainer.marginal_lane_weight = float(switches["sampled_marginal_lane_weight"])
    res = {}
    torch.manual_seed(1)
    batch = trainer._entry_slots(sampler.sample(2), torch.Generator().manual_seed(3))
    variants = build_variants(batch.state, trainer.counterfactual, torch.Generator().manual_seed(4))
    for det in (True, False):
        torch.manual_seed(7)
        with torch.no_grad():
            imagined = trainer.imagine(batch, deterministic=det, variants=variants, future_seed=11)
        for key, value in imagined.items():
            if key != "variants":
                flatten(f"imagine/{det}/{key}", value, res)
    torch.manual_seed(21)
    if full:                                   # flags-on smoke at the END of every curriculum
        trainer.coop_anchor = -10**6
    for step in range(2):
        metrics = trainer.update()
        metrics.pop("wall_time_s", None)
        flatten(f"update{step}", metrics, res)
    for name, module in (("policy", trainer.policy), ("group", trainer.group_encoder),
                         ("critic", trainer.critic)):
        for key, value in module.state_dict().items():
            res[f"params/{name}/{key}"] = value.detach().clone()
    torch.save(res, out)
    print("saved", len(res), "tensors; rows", variants.n_rows, "kinds", variants.row_kind.tolist())


def compare(a, b, allow_new=()):
    """``allow_new``: key suffixes that may exist only in ``b`` (additive metadata)."""
    import torch
    left, right = torch.load(a), torch.load(b)
    keys = sorted(set(left) | set(right))
    added = [k for k in keys if k not in left and k.rsplit("/", 1)[-1] in allow_new]
    if added:
        print(f"additive keys only in B (allowed): {added}")
    keys = [k for k in keys if k not in added]
    missing = [k for k in keys if k not in left or k not in right]
    diff = [k for k in keys if k in left and k in right
            and not (left[k].shape == right[k].shape and torch.equal(left[k], right[k]))]
    print(f"{len(keys)} keys; only-one-side {len(missing)}; different {len(diff)}")
    for k in (missing + diff)[:20]:
        print("  ", k)
    return 0 if not missing and not diff else 1


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        sys.exit(compare(sys.argv[2], sys.argv[3], tuple(sys.argv[4:])))
    run(sys.argv[1], sys.argv[2], sys.argv[3], full="--full" in sys.argv[4:])
