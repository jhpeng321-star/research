"""S-BACKLOG step 4a: WM2 K=8 paired futures -> long-credit CQR rows (CPU, zero SUMO).

Rows are produced with the production building blocks of ``wms.control.long_credit``
(``frozen_imagination`` / ``pair_rows``): the collection actor of each real pair imagines K=8
coupled futures from the pair's causal start state; ``real_benefit`` is the recorded
native-minus-policy cost over the beyond-trusted tail, from exact per-tick cost clocks that were
reconciled to the authoritative ledger.

  loop : cycles c00/c01/c02 of the loop (re-exported datasets with entry_backlog clocks); each
         cycle uses ITS collection actor (checkpoint sha bound by the pairing receipt).
  e3   : S-E3DATA-1 'calibration'/'validation' worlds (ours_c1 / coopoff_c1 arms vs native);
         each arm uses its own checkpoint (sha bound by the E3 provenance).  Rows carry the
         registered E3 role ('calibration'/'validation') so no E3 fit world is ever used.

Efficiency (documented deviation, verified): ONE 300 s rollout per pair; the 60/120/180 s rows are
prefixes of the same K=8 coupled futures (production fits only the 300 s horizon).  ``--verify N``
additionally runs N explicit separate 120 s rollouts and records the prefix-vs-separate comparison.

Usage (one shard, one process):
  python cqr_rows.py --tasks loop,e3 --box cpuP --shard 0 --nshards 8 --threads 4 --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

HOME = Path(os.environ.get("BACKLOG_HOME", Path.home() / "backlog"))
os.environ.setdefault("C4_ALLOWED_ROOTS", str(HOME))
WM_CKPT = HOME / "inputs/wm/c02/wm_s.pt"
WM_CONFIG = "configs/c4_wm_loop14_r16_ft.toml"
POLICY_CONFIG = "configs/c4_policy14_c3longint.toml"
WM_SHA = "b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436"
TRUSTED_S = 90
HORIZONS = (60, 120, 180, 300)             # 60: probe_cqr full-window rows only; tail rows start after 90 s
LOOP = {   # cycle -> (policy ckpt, expected sha, dataset, pairing, policy returns, native returns)
    "c00": ("inputs/policy/c00/policy_latest.pt", "1b9949cc0eadb32db3f56b08684fa512fc2e70a26a0b7a360edb5599c6226ab5"),
    "c01": ("inputs/policy/c01.a3/policy_latest.pt", "64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c"),
    "c02": ("inputs/policy/c02.h60b.r2/policy_latest.pt", "44f3eb66aba9a8598a1fd8727074389cb8f797d1e2d2f6cdf223d29fb0e7f128"),
}
E3_POLICY = {
    "ours_c1": ("inputs/policy/c01.a3/policy_latest.pt", "64f70af46adc582b426e789f082c19a8dca0525952d5b84e8fb18618b5f9fb5c"),
    "coopoff_c1": ("inputs/policy/coopoff_c1/policy_latest.pt", "e1fa4881bbc5c89d05298e979d98b23284308331ad1bbb12f9608fa67e105675"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_trainer(threads):
    """Policy14Trainer on CPU, constructed exactly like ``policy14_cli.main`` (native-init actor,
    full-actor flags, WM2 as the single ensemble member) but with no data pack: the start sampler
    is supplied per pair source."""
    import torch
    from wms.control import policy14_cli as cli
    from wms.control.contract import RewardWeights
    from wms.control.counterfactual import CounterfactualConfig
    from wms.control.dreamer import DreamerConfig
    from wms.control.objective14 import authoritative_reward
    from wms.control.policy import CavPolicy, RsuGroupEncoder
    from wms.control.obs import build_observations
    from wms.control.policy14 import Policy14Trainer, make_controller_factory
    from wms.control.real_returns14 import ScaledTeamCritic
    from wms.control.safety import safe_accel
    from wms.control.train_cli import GroupSignal, _read_config, policy_settings
    from wms.control import coopw as coopw_module

    torch.set_num_threads(threads)
    raw = _read_config(POLICY_CONFIG)
    sw = raw["policy14"]
    cfg = replace(DreamerConfig(**raw["trainer"]), real_return_mix=0.0)
    models, receipts = cli.load_members([str(WM_CKPT)], [WM_CONFIG], "cpu")
    assert receipts[0]["checkpoint_sha256"] == WM_SHA, receipts[0]["checkpoint_sha256"]
    settings = policy_settings(raw)
    policy = CavPolicy(accel_max=settings["accel_max"], lane_head=False, init_zero_accel=False,
                       residual_on_native=True)
    factory = make_controller_factory(policy_ramp_merge=False, native_lane_change=True,
                                      native_ramp_merge="wm_law")
    coopw = {}
    for section, cls in (("uncertainty", "UncertaintyConfig"), ("exploration", "ExplorationConfig")):
        if section in raw:
            coopw[section] = getattr(coopw_module, cls)(**raw[section])
    trainer = Policy14Trainer(models, policy, GroupSignal(RsuGroupEncoder()),
                              ScaledTeamCritic(entry_backlog=True, value_scale=float(sw["critic_value_scale"])),
                              None, observation_builder=build_observations, reward_fn=authoritative_reward,
                              safety=safe_accel, controller_factory=factory, config=cfg,
                              reward_weights=RewardWeights(**raw["reward"]), device="cpu", output_dir=None,
                              lane_rules=None, counterfactual=CounterfactualConfig(**raw["counterfactual"]),
                              real_return_provider=None, **coopw)
    trainer.long_credit_requested = True
    trainer.long_credit_trusted_s = TRUSTED_S
    return trainer


def load_actor(trainer, ckpt, expect_sha):
    import torch
    path = HOME / ckpt if not str(ckpt).startswith("/") else Path(ckpt)
    if sha256(path) != expect_sha:
        raise SystemExit(f"REFUSED: actor checkpoint sha mismatch {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    trainer.policy.load_state_dict(payload["policy"])
    trainer.group_encoder.load_state_dict(payload["group_encoder"])
    trainer.policy.eval()
    trainer.group_encoder.eval()


def make_source(dataset):
    from wms.config.load import load_train_config
    from wms.control.start_states import StartStateSampler
    from wms.data.c4_pairs import PairedBranchSource
    cfg = load_train_config(WM_CONFIG)
    pairs = PairedBranchSource(Path(dataset), horizon_steps=600, history_s=cfg.data.history_s,
        history_stride_s=cfg.data.history_stride_s, entries_v2=True, junction_fix=cfg.data.junction_fix,
        merge_pivot_history=cfg.model.merge_decision == "first_passage", name="backlog_pairs",
        arm_kinds=("B", "POLICY"))
    return pairs, StartStateSampler(pairs.dataset)


def loop_curves(pairs, row, target_j):
    loader = pairs.dataset._loader
    ep_index, tick = loader.index[row]
    ep = loader.episode(ep_index)
    kin = ep.array("veh_kin")[tick + 1:tick + 601]
    mask = ep.array("veh_mask")[tick + 1:tick + 601]
    queue = ep.array("entry_backlog")[tick:tick + 600]
    costs = .5 * ((np.maximum(0., 1. - kin[..., 4] / 31.29) * mask).sum(-1) + queue.sum(-1))
    if not np.isclose(costs.sum(), -float(target_j), rtol=1e-6, atol=1e-3):
        raise ValueError("recorded cost clocks do not reconcile to authoritative J")
    return costs


def loop_tasks():
    from wms.control.real_returns14 import NPZRealReturnProvider
    tasks = []
    for cyc, (ckpt, sha) in LOOP.items():
        base = HOME / "stage/loop"
        data = HOME / f"reexport/pairs_{cyc}.a1.backlog"
        pairing = HOME / f"inputs/returns/pairing_{cyc}.a1.json"
        proof = json.loads(pairing.read_text())
        if proof["policy_checkpoint_sha256"] != sha:
            raise SystemExit(f"REFUSED: pairing receipt actor sha != expected for {cyc}")
        p = NPZRealReturnProvider(str(HOME / f"inputs/returns/returns_{cyc}.a1.npz"))
        b = NPZRealReturnProvider(str(HOME / f"inputs/returns/native_returns_{cyc}.a1.npz"))
        for i, item in enumerate(proof["pairs"]):
            tasks.append({"kind": "loop", "cycle": cyc, "i": i, "item": item, "data": str(data),
                          "ckpt": ckpt, "sha": sha, "jp": float(p._returns[i]), "jb": float(b._returns[i])})
    return tasks


def e3_tasks(box):
    tasks = []
    data = HOME / f"stage/e3cal.{box}"
    man = json.loads((data / "C4_EPISODES_MANIFEST.json").read_text())
    by_pair = {}
    for e in man["episodes"]:
        by_pair.setdefault(e["pair_key"], {})[e["arm"]] = e
    # Any completed prefix is role-balanced: hash-ranked worlds, calibration/validation alternating.
    rank = lambda pk: hashlib.sha256(("S-BACKLOG-E3ORDER:" + pk).encode()).hexdigest()
    per_role = {role: sorted((pk for pk, arms in by_pair.items() if arms["native"]["role_e3"] == role), key=rank)
                for role in ("calibration", "validation")}
    order = [pk for pair in zip(per_role["calibration"], per_role["validation"]) for pk in pair]
    for k, pk in enumerate(order):
        arms = by_pair[pk]
        for arm in ("ours_c1", "coopoff_c1"):
            ckpt, sha = E3_POLICY[arm]
            if arms[arm]["policy_sha256"] != sha:
                raise SystemExit(f"REFUSED: E3 provenance sha != expected actor for {arm}")
            tasks.append({"kind": "e3", "pair_key": pk, "arm": arm, "i": 100000 + 2 * k + (arm == "coopoff_c1"),
                          "world": str(arms[arm]["seed"]), "role": arms[arm]["role_e3"],
                          "data": str(data), "ckpt": ckpt, "sha": sha,
                          "native_ep": arms["native"]["episode"], "arm_ep": arms[arm]["episode"]})
    return tasks


def task_id(t):
    return f"{t['kind']}:{t.get('cycle', t.get('arm'))}:{t.get('pair_key', t['i'])}:{t['i']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", required=True, help="comma list of: loop,e3")
    ap.add_argument("--box", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke: stop after N tasks")
    ap.add_argument("--horizons", default="")
    ap.add_argument("--verify", type=int, default=0, help="explicit separate 120 s runs for the first N tasks")
    ap.add_argument("--loop-parity", type=int, default=-1,
                    help="process only loop tasks whose global index has this parity (two-box split)")
    a = ap.parse_args()
    horizons = tuple(int(x) for x in a.horizons.split(",")) if a.horizons else HORIZONS
    os.nice(10)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = []
    if "loop" in a.tasks:
        lt = loop_tasks()
        tasks += [t for n, t in enumerate(lt) if a.loop_parity < 0 or n % 2 == a.loop_parity]
    if "e3" in a.tasks:
        tasks += e3_tasks(a.box)
    verify_ids = {task_id(t) for t in tasks[:a.verify]}
    mine = [t for n, t in enumerate(tasks) if n % a.nshards == a.shard]
    if a.limit:
        mine = mine[:a.limit]
    target = out / f"rows_{a.box}_{a.shard:02d}.jsonl"
    done = set()
    if target.exists():
        done = {json.loads(l)["task"] for l in target.read_text().splitlines() if l.strip()}
    trainer = build_trainer(a.threads)
    from wms.control.long_credit import frozen_imagination, pair_benefit_curve
    cache = {}
    with frozen_imagination(trainer) as old, target.open("a") as sink:
        for t in mine:
            tid = task_id(t)
            if tid in done:
                continue
            start_time = time.time()
            if t["data"] not in cache:
                cache[t["data"]] = make_source(t["data"])
            pairs, sampler = cache[t["data"]]
            load_actor(trainer, t["ckpt"], t["sha"])
            loader = pairs.dataset._loader
            if t["kind"] == "loop":
                by_key = {g["pair_key"]: g for g in pairs.groups}
                group = by_key[t["item"]["pair_key"]]
                native_row = group["rows"][0]
                policy_row = next(r for r in group["rows"][1:] if loader.entries[r]["branch"]["kind"] == "POLICY")
                curves = [loop_curves(pairs, policy_row, t["jp"]), loop_curves(pairs, native_row, t["jb"])]
                world, role = str(t["item"]["seed"]), None
            else:
                if t["data"] + "#curves" not in cache:
                    cache[t["data"] + "#curves"] = dict(np.load(Path(t["data"]) / "cost_curves.npz"))
                curve_file = cache[t["data"] + "#curves"]
                group = next(g for g in pairs.groups if g["pair_key"] == t["pair_key"])
                native_row = group["rows"][0]
                policy_row = next(r for r in group["rows"][1:] if loader.entries[r]["episode"] == t["arm_ep"])
                assert loader.entries[native_row]["episode"] == t["native_ep"]
                curves = [curve_file[t["arm_ep"]], curve_file[t["native_ep"]]]
                world, role = t["world"], t["role"]
            diff = pair_benefit_curve(trainer, old, sampler, policy_row, t["i"], 300)         # [K, 300]
            trusted = TRUSTED_S
            gap = curves[1] - curves[0]                                                        # real, per 0.5 s tick
            rows = []
            for h in horizons:
                rows.append({"world": world, "horizon_s": h,
                             "samples": diff[:, trusted:h].sum(1).tolist() if h > trusted else [0.0] * diff.shape[0],
                             "real_benefit": float(gap[2 * trusted:2 * h].sum()) if h > trusted else 0.0,
                             "samples_full": diff[:, :h].sum(1).tolist(), "real_full": float(gap[:2 * h].sum())})
            check = None
            if tid in verify_ids:
                sep = pair_benefit_curve(trainer, old, sampler, policy_row, t["i"], 120)      # explicit 120 s run
                check = {"separate_120_tail": sep[:, trusted:].sum(1).tolist(),
                         "prefix_120_tail": diff[:, trusted:120].sum(1).tolist(),
                         "separate_120_full": sep.sum(1).tolist(), "prefix_120_full": diff[:, :120].sum(1).tolist()}
            for r in rows:
                r.update(task=tid, kind=t["kind"], cycle=t.get("cycle"), arm=t.get("arm"),
                         pair_key=loader.entries[policy_row]["pair_key"], role=role,
                         wm_sha256=WM_SHA, actor_sha256=t["sha"])
            sink.write(json.dumps({"task": tid, "seconds": time.time() - start_time, "rows": rows}) + "\n")
            sink.flush()
            print(f"{tid} {time.time() - start_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
