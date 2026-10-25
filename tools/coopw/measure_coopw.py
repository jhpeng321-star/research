"""S-COOPW measurement (CPU, no SUMO, no training): scales behind the COOPW weights (COOPW.md).

On real TRAIN start windows and a real WM-S, with the production POLICY14 trainer (fresh actor of
the production seed, critic warm-started on real returns exactly like ``policy14_cli``):

``--starts N``  per start with a counterfactual group (training sampling path, uncertainty
    futures on when ``[uncertainty]`` is enabled or ``--future-samples`` > 1):
    * the unit-weight objective terms of ``wms.control.coopw.term_objectives`` (``all`` = D_all,
      ``individual`` = sum_i D_i, ``group`` = sum_G D_G, ``coop`` = sum_G C_G, ``team`` =
      sum_{|G|>=3} sum_i M_i, ``uncertainty`` = -sum std on the all rows), all divided by the
      trainer's return scale exactly as in the actor loss;
    * the L2 norm over the actor parameters of each term's gradient and its cosine with ``all``;
    * with futures: the per-vehicle disagreement (std over futures of the per-step speed-deficit
      cost) by road zone, by horizon step and by penetration.
``--base-rates N``  N plain (target-distribution) start draws: the observable tilt features
    (``coopw.start_features``), the ``multi_cav`` flags and legacy ``group_rich``, by penetration
    and demand label (reporting only), plus the coverage / ESS the ``start_tilt`` would give.
``--profile N``  peak live CPU tensor bytes / wall time of N full ``trainer.update()`` calls at
    the curriculum position ``--profile-update-count`` (``scripts/c4_cfmem_profile.LiveBytes``).

    python tools/coopw/measure_coopw.py --config configs/c4_policy14_coop.toml \
        --wm-ckpt CK/wm_s.pt --wm-config CK/config.resolved.json --real-returns R.npz \
        --starts 24 --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

TERMS = ("all", "individual", "group", "coop", "team", "uncertainty")


def build_trainer(args, raw):
    from wms.control.contract import RewardWeights
    from wms.control.coopw import UncertaintyConfig
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.dreamer import DreamerConfig
    from wms.control.lanes import LaneRules
    from wms.control.objective14 import authoritative_reward
    from wms.control.obs import build_observations
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.policy14 import Policy14Trainer, make_controller_factory
    from wms.control.policy14_cli import load_members
    from wms.control.real_returns14 import (NPZRealReturnProvider, ScaledTeamCritic,
                                            warm_start_real_critic)
    from wms.control.safety import safe_accel
    from wms.control.train_cli import GroupSignal, _load_starts, policy_settings

    switches = raw["policy14"]
    cfg = replace(DreamerConfig(**raw["trainer"]), checkpoint_every=0, validation_every=0)
    if args.horizon_s:                     # profile a staged horizon directly (memory / time)
        cfg = replace(cfg, horizon=int(args.horizon_s), entry_pad_max=int(args.pad_max or cfg.entry_pad_max))
    models, receipts = load_members([args.wm_ckpt], [args.wm_config], "cpu")
    torch.manual_seed(args.seed)
    train, _ = _load_starts(raw["data"], models, args.seed, entries=True)
    settings = policy_settings(raw)
    policy = CavPolicy(accel_max=settings["accel_max"], lane_head=settings.get("lane_control") != "native",
                       init_zero_accel=settings.get("init_zero_accel", False))
    lane_section = dict(raw["lanes"])
    enabled = lane_section.pop("enabled", True)
    lane_rules = LaneRules.from_mapping(lane_section) if enabled else None
    factory = make_controller_factory(switches.get("policy_ramp_merge", False),
                                      switches.get("native_ramp_merge", "gap6"))
    cf = dict(raw["counterfactual"])
    for key in ("n_flip", "group_max_size", "n_group"):
        if getattr(args, key) is not None:
            cf[key] = getattr(args, key)
    unc = dict(raw.get("uncertainty", {}))
    if args.future_samples:
        unc.update(enabled=True, future_samples=args.future_samples)
        unc.setdefault("weights", [0.0])
        unc.setdefault("curriculum_updates", [0])
    uncertainty = UncertaintyConfig(**unc) if unc else None
    from wms.control.coopw import ExplorationConfig
    exploration = ExplorationConfig(**raw["exploration"]) if "exploration" in raw else None
    provider = NPZRealReturnProvider(args.real_returns, seed=args.seed)
    trainer = Policy14Trainer(models, policy, GroupSignal(RsuGroupEncoder()),
                              ScaledTeamCritic(entry_backlog=True,
                                               value_scale=float(switches["critic_value_scale"])),
                              train, observation_builder=build_observations,
                              reward_fn=authoritative_reward, safety=safe_accel,
                              controller_factory=factory, config=cfg,
                              reward_weights=RewardWeights(**raw["reward"]), device="cpu",
                              output_dir=None, lane_rules=lane_rules,
                              counterfactual=CounterfactualConfig(**cf),
                              real_return_provider=provider, uncertainty=uncertainty,
                              exploration=exploration)
    trainer.marginal_lane_weight = float(switches["sampled_marginal_lane_weight"])
    warm = warm_start_real_critic(trainer, updates=int(switches["critic_warmup_updates"]))
    return trainer, train, receipts, warm


def zone_of(s, lane):
    """Road zones from the shared contract geometry (RSU coverage + merge area)."""
    from wms.control.contract import MERGE_S0_M, MERGE_S1_M, RSU_MAINLINE_S_M

    if lane == 2:
        return "ramp"
    if s < RSU_MAINLINE_S_M[0]:
        return "main_far_upstream"
    if s < MERGE_S0_M:
        return "main_near_upstream"
    if s <= MERGE_S1_M:
        return "merge"
    return "downstream"


def disagreement_breakdown(imagined, n_base):
    """Per-vehicle std over futures of the per-control-step speed-deficit cost, all-CAV rows."""
    from wms.control.contract import SPEED_LIMIT_MPS

    states = imagined["states"]                       # [E'][H+1] dicts of [N,K]
    if len(states) < 2:
        return None
    steps = len(states[0]) - 1
    cost = torch.stack([torch.stack([torch.where(st["present"].bool(),
                                                 torch.relu(1 - st["v_mps"] / SPEED_LIMIT_MPS),
                                                 torch.zeros_like(st["v_mps"]))
                                     for st in member[1:]]) for member in states])   # [E',H,N,K]
    std = cost.detach().std(0, unbiased=False)[:, :n_base]                          # [H,B,K]
    start = states[0][0]
    zones = defaultdict(lambda: [0.0, 0])
    for b in range(n_base):
        for k in torch.nonzero(start["present"][b].bool()).flatten().tolist():
            name = zone_of(float(start["s_m"][b, k]), int(start["lane"][b, k]))
            zones[name][0] += float(std[:, b, k].mean())
            zones[name][1] += 1
    team = imagined["disagreement"].detach()[:, :n_base].mean(1)                     # [H]
    return {"per_vehicle_std_by_zone": {z: v[0] / v[1] for z, v in zones.items() if v[1]},
            "vehicles_by_zone": {z: v[1] for z, v in zones.items()},
            "team_std_by_step": [float(x) for x in team],
            "per_vehicle_std_by_step": [float(std[t].mean()) for t in range(steps)]}


def measure_scales(trainer, n_starts, seed):
    from wms.control.coopw import term_objectives
    from wms.control.counterfactual import build_variants

    parameters = [p for p in trainer.actor_parameters if p.requires_grad]
    rows = []
    generator = torch.Generator().manual_seed(seed)
    for index in range(n_starts):
        torch.manual_seed(seed + index)
        trainer.update_count = index
        batch = trainer._entry_slots(trainer._sample_starts(trainer.start_states),
                                     torch.Generator().manual_seed(1_000_003 * (index + 1)))
        variants = build_variants(batch.state, trainer.counterfactual, generator)
        penetration = trainer._penetration_buckets(batch).get(0)
        record = {"start": index, "groups": len(variants.groups), "rows": variants.n_rows,
                  "penetration": penetration, "meta": {k: batch.metadata[0].get(k) for k in
                                                       ("world", "tick", "start_source", "group_rich")}}
        if not variants.groups:
            rows.append(record)
            print(json.dumps(record), flush=True)
            continue
        started = time.perf_counter()
        trainer._training_imagination = True
        try:
            imagined = trainer.imagine(batch, variants=variants)
        finally:
            trainer._training_imagination = False
        n_base = imagined["n_base"]
        returns = imagined["returns"][..., :n_base].detach().float().flatten()
        low, high = map(float, torch.quantile(returns, torch.tensor([0.05, 0.95])))
        scale = max(1.0, high - low)        # first-update value of DreamerTrainer._return_scale
        terms, values = term_objectives(imagined, scale, trainer.config.lane_pg_weight)
        flat = {}
        for name, term in terms.items():
            grads = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
            flat[name] = torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                    for g, p in zip(grads, parameters)]).double()
        norms = {name: float(g.norm()) for name, g in flat.items()}
        cos = {name: float(g @ flat["all"]) / max(norms[name] * norms["all"], 1e-300)
               for name, g in flat.items() if name != "all"}
        record.update(group_sizes=[len(g["members"]) for g in variants.groups], scale=scale,
                      values=values, grad_norms=norms, cos_with_all=cos,
                      n_futures=imagined["n_futures"], seconds=time.perf_counter() - started,
                      disagreement=disagreement_breakdown(imagined, n_base))
        del imagined, terms, flat
        rows.append(record)
        print(json.dumps(record), flush=True)
    return rows


def summarise(rows):
    usable = [r for r in rows if r.get("grad_norms")]
    out = {"n_starts": len(rows), "n_with_groups": sum(1 for r in rows if r.get("groups")),
           "n_measured": len(usable)}
    for name in TERMS:
        have = [r for r in usable if name in r["grad_norms"] and r["grad_norms"][name] > 0]
        if not have:
            continue
        ratio = sorted(r["grad_norms"]["all"] / r["grad_norms"][name] for r in have)
        out[name] = {"n": len(have),
                     "grad_norm_median": statistics.median(r["grad_norms"][name] for r in have),
                     "abs_value_median": statistics.median(abs(r["values"][name]) for r in have),
                     "parity_all_over_term_median": statistics.median(ratio),
                     "parity_min_max": [ratio[0], ratio[-1]],
                     "parity_iqr": ([statistics.quantiles(ratio, n=4, method="inclusive")[i]
                                     for i in (0, 2)] if len(ratio) >= 2 else None),
                     "cos_with_all_median": (1.0 if name == "all" else
                                             statistics.median(r["cos_with_all"][name] for r in have))}
    dis = [r["disagreement"] for r in usable if r.get("disagreement")]
    if dis:
        zones = defaultdict(list)
        for d in dis:
            for z, v in d["per_vehicle_std_by_zone"].items():
                zones[z].append(v)
        steps = len(dis[0]["team_std_by_step"])
        by_pen = defaultdict(list)
        for r in usable:
            if r.get("disagreement"):
                by_pen[r["penetration"]].append(statistics.mean(r["disagreement"]["team_std_by_step"]))
        out["disagreement"] = {
            "per_vehicle_std_by_zone_mean": {z: statistics.mean(v) for z, v in sorted(zones.items())},
            "team_std_by_step_mean": [statistics.mean(d["team_std_by_step"][t] for d in dis)
                                      for t in range(steps)],
            "team_std_by_penetration_mean": {p: statistics.mean(v) for p, v in sorted(by_pen.items())}}
    return out


def base_rates(trainer, n, seed, tilt=None):
    """Observable start features under the TARGET distribution (plain draws) by world penetration
    and demand label (labels for REPORTING only; the features never read them), and the coverage
    the ``start_tilt`` weights would give: E_p[m f] / E_p[m] per feature and stratum, the stratum
    shares under the tilt, and the tilt's effective-sample-size fraction (E_p m)^2 / E_p m^2."""
    import re

    from wms.control.contract import PHYSICS_DT_S, STEPS_PER_CONTROL
    from wms.control.coopw import TILT_FEATURES, TiltSampler, multi_cav_flags, start_features
    from wms.control.counterfactual import group_rich

    cf = trainer.counterfactual
    horizon_s = trainer.config.horizon * STEPS_PER_CONTROL * PHYSICS_DT_S
    tilt = TiltSampler(tilt or cf.start_tilt or {f: 1.0 for f in TILT_FEATURES})   # m = prod(1 + w f)
    torch.manual_seed(seed)
    samples = []
    for _ in range(n):
        batch = trainer.start_states.sample(1)
        state = {k: v[:1] for k, v in batch.state.items() if torch.is_tensor(v) and v.ndim >= 1
                 and v.shape[0] == batch.state["s_m"].shape[0]}
        queue = batch.entry_queue[0] if batch.entry_queue is not None else None
        features = start_features(state, cf, horizon_s, entry_queue=queue, lane_rules=trainer.lane_rules)
        flags = multi_cav_flags(state, cf, horizon_s)
        world = str(batch.metadata[0].get("world", "")) if batch.metadata else ""
        match = re.search(r"m(\d+)p(\d+)", world)
        samples.append({**{k: bool(features[k]) for k in TILT_FEATURES},
                        **{f"flag_{k}": v for k, v in flags.items()},
                        "group_rich": group_rich(state, cf), "m": tilt.multiplier(features),
                        "demand_label": int(match[1]) if match else None,
                        "pen_label": f"p{match[2]}" if match else None,
                        "observed_share": features["observed_share"],
                        "observed_main_demand_vph": features["observed_main_demand_vph"]})
    names = list(TILT_FEATURES) + ["flag_group", "flag_near_merge", "flag_congested",
                                   "flag_ramp_encounter", "group_rich"]
    z = sum(r["m"] for r in samples) / len(samples)
    out = {"n": len(samples), "tilt_weights": tilt.weights, "Z": z, "samples": samples,
           "ess_fraction": z * z / (sum(r["m"] ** 2 for r in samples) / len(samples)),
           "target": {k: sum(r[k] for r in samples) / len(samples) for k in names},
           "tilted": {k: sum(r[k] * r["m"] for r in samples) / (z * len(samples)) for k in names}}
    for label in ("pen_label", "demand_label"):
        strata = defaultdict(list)
        for r in samples:
            strata[r[label]].append(r)
        out[f"by_{label}"] = {
            str(key): {"target_share": len(rows) / len(samples),
                       "tilted_share": sum(r["m"] for r in rows) / (z * len(samples)),
                       "target": {k: sum(r[k] for r in rows) / len(rows) for k in names},
                       "mean_observed_main_demand_vph": (
                           statistics.mean(r["observed_main_demand_vph"] for r in rows)
                           if all(r["observed_main_demand_vph"] is not None for r in rows) else None),
                       "mean_observed_share": statistics.mean(r["observed_share"] for r in rows)}
            for key, rows in sorted(strata.items(), key=lambda kv: str(kv[0]))}
    return out


def profile_updates(trainer, n, position):
    from c4_cfmem_profile import LiveBytes

    trainer.update_count = position
    trainer.coop_anchor = 0 if trainer.counterfactual.curriculum_anchor == "first_update" else None
    out = []
    for _ in range(n):
        tracker = LiveBytes()
        started = time.perf_counter()
        with tracker:
            metrics = trainer.update()
        keep = ("cf_rows", "n_groups", "mean_group_size", "cf_weights", "C_G", "M_i", "I_i",
                "cf_coverage", "start_oversampling", "cf_term_grad_norms", "cf_term_values",
                "uncertainty_weight_effective", "n_futures", "actor_grad_norm", "actor_grad_norms")
        out.append({"peak_bytes": tracker.peak, "s_per_update": time.perf_counter() - started,
                    **{k: metrics.get(k) for k in keep}})
        print(json.dumps(out[-1], default=str), flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/c4_policy14_coop.toml")
    p.add_argument("--wm-ckpt", required=True)
    p.add_argument("--wm-config", required=True)
    p.add_argument("--real-returns", required=True)
    p.add_argument("--starts", type=int, default=0)
    p.add_argument("--base-rates", type=int, default=0)
    p.add_argument("--seed", type=int, default=965140)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--n-group", dest="n_group", type=int, default=None)
    p.add_argument("--n-flip", dest="n_flip", type=int, default=None)
    p.add_argument("--group-max-size", dest="group_max_size", type=int, default=None)
    p.add_argument("--future-samples", type=int, default=0,
                   help="enable [uncertainty] futures for the measurement (weight 0 unless configured)")
    p.add_argument("--horizon-s", type=float, default=0, help="override the imagination horizon")
    p.add_argument("--pad-max", type=int, default=0, help="entrant-slot cap with --horizon-s")
    p.add_argument("--profile", type=int, default=0, help="profiled trainer.update() calls")
    p.add_argument("--profile-update-count", type=int, default=10_000,
                   help="curriculum position of the profiled updates (default: end weights)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    from wms.control.train_cli import _read_config

    raw = _read_config(args.config)
    trainer, train, receipts, warm = build_trainer(args, raw)
    record = {"args": vars(args), "wm": receipts, "critic_warmup": warm,
              "counterfactual": trainer.counterfactual.as_dict(),
              "uncertainty": trainer.uncertainty.as_dict() if trainer.uncertainty else None,
              "slots_K": int(train.sample(1).state["s_m"].shape[-1])}
    if args.base_rates:
        record["base_rates"] = base_rates(trainer, args.base_rates, args.seed)
        print(json.dumps(record["base_rates"], indent=1), flush=True)
    if args.starts:
        rows = measure_scales(trainer, args.starts, args.seed)
        record.update(rows=rows, summary=summarise(rows))
        print(json.dumps(record["summary"], indent=1), flush=True)
    if args.profile:
        record["profile"] = profile_updates(trainer, args.profile, args.profile_update_count)
    Path(args.out).write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
