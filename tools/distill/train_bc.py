"""DISTILL student trainer: behaviour cloning of LEARNED_TRIGGER into ONE MAPPO_LANE-format actor.

Student = ``wms.control.policy.CavPolicy(lane_head=True)`` + ``RsuGroupEncoder`` exactly as the lane-capable
MAPPO comparator (S-MAPPOWZL, ``configs/c4_mappo_lane_wz3to2.toml``) builds them, run through the SAME
deployment computation as ``wms.control.mappo.agent.ActorStep`` / ``evaluate_actions`` on the wz3to2 N1 map
binding (``tools.nm_mappo.n1_adapter.install``: CAV-report-masked RSU field) with the LCEXEC lane-action
provider (``tools.nm_mappo.lane_actions:lcexec``) as the lane mask.  The export is a ``c4_dreamer/3``
``trainer="mappo"`` payload (``wms.control.mappo.agent.export_payload``), so
``tools.nm_mappo.sanity_eval.load_actor(path, lanes, lcexec)`` loads it as the MAPPO_LANE ``LaneActor``.

Init (shape-checked, recorded): every acceleration-path weight (own/report encoders, attention, accel head,
accel bounds [-3, 2.6]) and the RSU group encoder from the speed-only MAPPO u0 export (d9042cc5, the U0 the
teacher runs); lane head ``fresh_prior`` (default: seeded first layer, zero last layer, last bias = log of
the training-set lane-label prior -> argmax "stay" everywhere at init) or ``wzl`` (S-MAPPOWZL final
d13cd2f7 lane head).  At init the student's acceleration equals U0's exactly (tested).

Loss per CAV row r with weight w_r: ``accel_coef * w (a_mean - clip(a_applied))^2`` (MSE on the deterministic
decoded mean; or ``nll`` = Gaussian NLL of the pre-tanh latent) + ``lane_coef * w * CE(masked logits, label)``.
Weights are set ONCE from training-set counts: plan-related rows (plan_active, plus the ``pre_onset_s`` s
before a plan containing the CAV starts) share ``plan_share`` of the total weight; the other rows are a
per-epoch Bernoulli subsample (keep prob = min(1, nonplan_keep_ratio * N_plan / N_other)) sharing the rest.
Rows whose lane label is infeasible under the LCEXEC gate are dropped from the lane loss (counted).

Validation: 10 % of episodes by seed (``split_episodes``).  Checkpoint rule (pre-registered): the FINAL
epoch is exported; validation numbers are reported only, never used to select.

CLI: ``python -m tools.distill.train_bc --data DIR --out DIR --epochs N --seed S [--config JSON] [--device]``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from tools.distill import data as D  # noqa: E402

DEFAULT_CONFIG = {
    "map": "wz3to2",
    "lane_actions": "tools.nm_mappo.lane_actions:lcexec",
    "lanes_config": "configs/c4_mappo_lane_wz3to2.toml",      # [lanes] = the MAPPO_LANE rules
    "u0_path": "_inputs/mappo_u0_policy_00169.pt",
    "u0_sha256": "d9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca",
    "wzl_path": "_inputs/policy_00048.pt",
    "wzl_sha256": "d13cd2f75d3ff83dab7280be64ac5d0ec7cc05c24bf391a95e6c1ab6537015e9",
    "lane_init": "fresh_prior",          # fresh_prior | wzl
    "pre_onset_s": 10,
    "lane_target": "observed",           # observed | cmd_for_plan
    "plan_share": 0.5,                   # plan-related rows' share of the total loss weight
    "nonplan_keep_ratio": 4.0,           # kept other rows per plan-related row (per-epoch subsample)
    "val_fraction": 0.1,
    "batch_frames": 128,
    "lr": 3.0e-4,
    "weight_decay": 0.0,
    "grad_clip": 1.0,
    "accel_loss": "mse",                 # mse | nll
    "accel_coef": 1.0,
    "lane_coef": 1.0,
    "accel_target_clip": [-3.0, 1.2],    # registered CAV braking bound / CAV accel cap
    "require_receipts": True,
}


def file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------- runtime binding (= the MAPPO_LANE path)
def setup_runtime(cfg):
    """Bind the N1 map exactly like the MAPPO_LANE training/eval process; returns the module handles.

    ``wms.control.mappo.agent`` is imported BEFORE the install so ``map_runtime._rebind`` rebinds its
    ``build_rsu_field`` to the CAV-report-masked field, i.e. the student sees what ActorStep sees."""
    from wms.control.mappo import agent
    from wms.control.mappo.rollout import load_lane_actions
    from tools.nm_mappo import n1_adapter
    from tools.nm_mappo.sanity_eval import load_cfg
    from wms.control.lanes import LaneRules
    desc, record = n1_adapter.install(cfg["map"])
    if getattr(agent.build_rsu_field, "map_name", None) != cfg["map"]:
        raise RuntimeError("agent.build_rsu_field was not rebound to the map's masked RSU field")
    gate = load_lane_actions(cfg["lane_actions"])
    lanes = (load_cfg(resolve(cfg["lanes_config"])) or {}).get("lanes")
    rules = LaneRules.from_mapping(lanes)
    return dict(agent=agent, gate=gate, rules=rules, lanes_cfg=lanes,
                record=dict(map=desc.name, k_slots=int(desc.k_slots), rsu_field_rebound=record.get("rsu_field_rebound"),
                            gate_source=getattr(getattr(gate, "rules", None), "source", None)))


# ---------------------------------------------------------------- networks
def _load_payload(path, sha):
    path = resolve(path)
    got = file_sha(path)
    if sha and got != sha:
        raise ValueError(f"{path} sha256 {got} != expected {sha}")
    return torch.load(path, map_location="cpu", weights_only=True), got


def init_networks(cfg, lane_prior=None, *, seed: int = 0):
    """Student (policy, encoder, lane_rules_payload, init record)."""
    from wms.control.mappo.agent import build_networks
    torch.manual_seed(seed)
    policy, encoder, _ = build_networks(lane_control="learned")
    u0, u0_sha = _load_payload(cfg["u0_path"], cfg.get("u0_sha256"))
    if u0.get("lane_head") is not False or u0.get("lane_control") != "native":
        raise ValueError("u0 must be the speed-only (native-lane) MAPPO export")
    own = policy.state_dict()
    state = dict(u0["policy"])
    bad = {k: (tuple(v.shape), tuple(own[k].shape)) for k, v in state.items() if k in own and v.shape != own[k].shape}
    if bad:
        raise ValueError(f"u0 shape mismatch: {bad}")
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if unexpected or any(not k.startswith("lane_head.") for k in missing):
        raise ValueError(f"u0 load: missing {missing} unexpected {unexpected}")
    group = {k[len("encoder."):] if k.startswith("encoder.") else k: v for k, v in u0["group_encoder"].items()}
    encoder.load_state_dict(group, strict=True)
    record = dict(accel_and_encoder=dict(path=str(cfg["u0_path"]), sha256=u0_sha, iteration=u0.get("iteration"),
                                         keys_loaded=sorted(state), accel_bounds=[policy.accel_min, policy.accel_max]),
                  lane_init=cfg["lane_init"])
    wzl, wzl_sha = _load_payload(cfg["wzl_path"], cfg.get("wzl_sha256"))
    lane_rules_payload = wzl.get("lane_rules")
    record["lane_rules_from"] = dict(path=str(cfg["wzl_path"]), sha256=wzl_sha)
    if cfg["lane_init"] == "wzl":
        lane = {k[len("lane_head."):]: v for k, v in wzl["policy"].items() if k.startswith("lane_head.")}
        policy.lane_head.load_state_dict(lane, strict=True)
        record["lane_head"] = dict(source="S-MAPPOWZL final", path=str(cfg["wzl_path"]), sha256=wzl_sha)
    elif cfg["lane_init"] == "fresh_prior":
        prior = np.ones(3) if lane_prior is None else np.asarray(lane_prior, float) + 1.0
        logp = np.log(prior / prior.sum())
        with torch.no_grad():
            torch.nn.init.zeros_(policy.lane_head[-1].weight)
            policy.lane_head[-1].bias.copy_(torch.as_tensor(logp, dtype=policy.lane_head[-1].bias.dtype))
        record["lane_head"] = dict(source="fresh (seeded first layer, zero last layer, bias = log train prior +1)",
                                   counts=None if lane_prior is None else [int(x) for x in lane_prior],
                                   bias=[float(x) for x in logp])
    else:
        raise ValueError("lane_init must be fresh_prior or wzl")
    return policy, encoder, lane_rules_payload, record


# ---------------------------------------------------------------- device-resident dataset
class Batcher:
    def __init__(self, ds: D.Dataset, device):
        t = lambda x, dt=None: torch.as_tensor(np.ascontiguousarray(x), device=device, dtype=dt)  # noqa: E731
        self.device = device
        self.s, self.v, self.a = t(ds.s), t(ds.v), t(ds.a)
        self.lane = t(ds.lane.astype(np.int64))
        self.is_cav = t(ds.is_cav)
        self.length, self.cmd, self.intent = t(ds.length), t(ds.cmd), t(ds.intent)
        self.off = t(ds.frame_off)
        self.acc_tgt, self.acc_u0 = t(ds.acc_tgt), t(ds.acc_u0)
        self.lane_tgt = t(ds.lane_tgt.astype(np.int64))
        self.cat = t(ds.cat.astype(np.int64))

    def rows(self, frames):
        f = torch.as_tensor(frames, device=self.device, dtype=torch.long)
        start, count = self.off[f], self.off[f + 1] - self.off[f]
        kmax = max(1, int(count.max()))
        ar = torch.arange(kmax, device=self.device)
        valid = ar[None, :] < count[:, None]
        rows = torch.where(valid, start[:, None] + ar[None, :], torch.zeros_like(start[:, None]))
        return rows, valid

    def state(self, rows, valid):
        z = lambda x, fill=0.0: torch.where(valid, x[rows], torch.full_like(x[rows], fill))  # noqa: E731
        return {"present": valid, "s_m": z(self.s), "v_mps": z(self.v), "a_mps2": z(self.a),
                "lane": torch.where(valid, self.lane[rows], torch.full_like(self.lane[rows], -1)),
                "is_cav": valid & self.is_cav[rows], "length_m": z(self.length, 5.0),
                "cmd_accel_mps2": z(self.cmd), "lane_intent": z(self.intent)}


def lane_mask(rt, state):
    """[B,K,3] LCEXEC feasibility.  The vector gate's batch loop only drops absent slots, which the
    padded batch already marks ``present=False``, so the dense pass is exact (unit-tested)."""
    gate = rt["gate"]
    if hasattr(gate, "_dense"):
        return gate._dense(state, rt["rules"])
    return gate.feasibility(state, None, rt["rules"])


def forward(policy, encoder, rt, state):
    """ActorStep's deterministic computation, compacted to CAV slots like ``evaluate_actions``."""
    agent = rt["agent"]
    obs = agent.build_observations(state)
    field = agent.build_rsu_field(state)
    mask = lane_mask(rt, state)
    index = agent.compact_index(obs.cav_mask)
    obs_c = agent.compact_observation(obs, index)
    cav = obs_c.cav_mask
    positions = state["s_m"].gather(1, index)
    mask_c = agent._gather(mask.bool(), index)
    signal = encoder(obs_c, positions, field)
    with agent.capture_heads(policy) as heads:
        out = policy(obs_c, signal, deterministic=True, lane_mask=mask_c)
    mean, log_std = heads["head"].unbind(-1)
    feasible = agent._feasible(mask_c, cav)
    logits = torch.where(feasible, heads["lane_logits"], torch.full_like(heads["lane_logits"], -1e9))
    return dict(index=index, cav=cav, accel=out.accel, mean=mean, log_std=log_std,
                log_p=torch.log_softmax(logits, -1), feasible=feasible, choice=logits.argmax(-1))


def gather_labels(b: Batcher, rows, valid, index):
    r = rows.gather(1, index)
    ok = valid.gather(1, index)
    take = lambda x, fill: torch.where(ok, x[r], torch.full_like(x[r], fill))  # noqa: E731
    return dict(row=r, acc=take(b.acc_tgt, float("nan")), u0=take(b.acc_u0, float("nan")),
                lane=take(b.lane_tgt, D.IGNORE), cat=take(b.cat, D.CAT_NONE))


def accel_loss(cfg, policy, out, target):
    lo, hi = cfg["accel_target_clip"]
    y = target.clamp(lo, hi)
    if cfg["accel_loss"] == "mse":
        return (out["accel"] - y) ** 2
    if cfg["accel_loss"] == "nll":
        scale = (policy.accel_max - policy.accel_min) / 2.0
        mid = (policy.accel_max + policy.accel_min) / 2.0
        latent = torch.atanh(((y - mid) / scale).clamp(-0.999, 0.999))
        std = out["log_std"].clamp(-5.0, 1.0).exp()
        return -torch.distributions.Normal(out["mean"], std).log_prob(latent)
    raise ValueError("accel_loss must be mse or nll")


def lane_ce(out, target):
    """(per-row CE, usable mask, infeasible-label mask)."""
    has = (target >= 0) & out["cav"]
    t = target.clamp(0, 2)
    feasible_label = out["feasible"].gather(-1, t[..., None]).squeeze(-1)
    ce = -out["log_p"].gather(-1, t[..., None]).squeeze(-1)
    return ce, has & feasible_label, has & ~feasible_label


# ---------------------------------------------------------------- weights
def weight_plan(ds: D.Dataset, train_eps, cfg):
    """Counts on TRAINING episodes only -> (w_plan, w_other, keep_prob, counts)."""
    row_ep = np.repeat(ds.frame_ep, np.diff(ds.frame_off))
    in_train = np.isin(row_ep, np.asarray(train_eps))
    cat = ds.cat[in_train]
    n_plan = int(np.sum((cat == D.CAT_PLAN) | (cat == D.CAT_PRE)))
    n_other = int(np.sum(cat == D.CAT_OTHER))
    if n_plan == 0:
        keep, w_plan, w_other = 1.0, 0.0, 1.0
    else:
        keep = min(1.0, cfg["nonplan_keep_ratio"] * n_plan / max(1, n_other))
        used = n_plan + keep * n_other
        share = float(cfg["plan_share"])
        w_plan = share / n_plan * used
        w_other = (1.0 - share) / max(1e-9, keep * n_other) * used if n_other else 0.0
    counts = dict(train_rows_plan_active=int(np.sum(cat == D.CAT_PLAN)), train_rows_pre_onset=int(np.sum(cat == D.CAT_PRE)),
                  train_rows_other=n_other, train_rows_plan_related=n_plan)
    return w_plan, w_other, keep, counts


def lane_prior(ds: D.Dataset, train_eps):
    row_ep = np.repeat(ds.frame_ep, np.diff(ds.frame_off))
    lab = ds.lane_tgt[np.isin(row_ep, np.asarray(train_eps)) & (ds.lane_tgt >= 0)]
    return np.bincount(lab.astype(np.int64), minlength=3)[:3]


# ---------------------------------------------------------------- evaluation
CLASS_NAMES = ("stay", "left", "right")


@torch.no_grad()
def evaluate(policy, encoder, rt, b: Batcher, frames, batch_frames):
    policy.eval()
    encoder.eval()
    acc = {c: dict(n_accel=0, abs_err=0.0, sq_err=0.0, u0_abs_err=0.0, n_lane=0, correct=0, infeasible=0,
                   confusion=np.zeros((3, 3), np.int64)) for c in (D.CAT_PLAN, D.CAT_PRE, D.CAT_OTHER)}
    for i in range(0, len(frames), batch_frames):
        rows, valid = b.rows(frames[i:i + batch_frames])
        out = forward(policy, encoder, rt, b.state(rows, valid))
        lab = gather_labels(b, rows, valid, out["index"])
        _, lane_ok, lane_bad = lane_ce(out, lab["lane"])
        for c, m in acc.items():
            sel = out["cav"] & (lab["cat"] == c)
            fin = sel & torch.isfinite(lab["acc"])
            err = (out["accel"] - lab["acc"])[fin]
            m["n_accel"] += int(fin.sum())
            m["abs_err"] += float(err.abs().sum())
            m["sq_err"] += float((err ** 2).sum())
            u0 = fin & torch.isfinite(lab["u0"])
            m["u0_abs_err"] += float((lab["u0"] - lab["acc"])[u0].abs().sum())
            lsel = sel & lane_ok
            t, p = lab["lane"][lsel], out["choice"][lsel]
            m["n_lane"] += int(lsel.sum())
            m["correct"] += int((t == p).sum())
            m["infeasible"] += int((sel & lane_bad).sum())
            if t.numel():
                m["confusion"] += np.bincount((t * 3 + p).cpu().numpy(), minlength=9).reshape(3, 3)
    report = {}
    for c, m in acc.items():
        conf = m["confusion"]
        report[D.CAT_NAMES[c]] = dict(
            n_accel_rows=m["n_accel"], accel_mae=m["abs_err"] / m["n_accel"] if m["n_accel"] else None,
            accel_rmse=math.sqrt(m["sq_err"] / m["n_accel"]) if m["n_accel"] else None,
            teacher_u0_vs_applied_mae=m["u0_abs_err"] / m["n_accel"] if m["n_accel"] else None,
            n_lane_rows=m["n_lane"], lane_accuracy=m["correct"] / m["n_lane"] if m["n_lane"] else None,
            lane_label_infeasible=m["infeasible"],
            lane_recall={CLASS_NAMES[k]: (int(conf[k, k]) / int(conf[k].sum()) if conf[k].sum() else None)
                         for k in range(3)},
            lane_label_counts={CLASS_NAMES[k]: int(conf[k].sum()) for k in range(3)},
            lane_confusion_true_x_pred=conf.tolist())
    # plan-related (plan_active + pre-onset) vs non-plan summaries
    def merged(names):
        parts = [report[n] for n in names]
        na = sum(p["n_accel_rows"] for p in parts)
        nl = sum(p["n_lane_rows"] for p in parts)
        conf = sum(np.asarray(p["lane_confusion_true_x_pred"]) for p in parts)
        return dict(n_accel_rows=na,
                    accel_mae=(sum((p["accel_mae"] or 0) * p["n_accel_rows"] for p in parts) / na) if na else None,
                    n_lane_rows=nl, lane_accuracy=(int(np.trace(conf)) / nl) if nl else None,
                    lane_recall={CLASS_NAMES[k]: (int(conf[k, k]) / int(conf[k].sum()) if conf[k].sum() else None)
                                 for k in range(3)})
    report["plan_related"] = merged(("plan", "pre_onset"))
    report["non_plan"] = merged(("other",))
    return report


# ---------------------------------------------------------------- training
def set_determinism(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def export(policy, encoder, lane_rules_payload, extra: dict, path):
    from wms.control.lanes import LaneRules
    from wms.control.mappo.agent import export_payload
    rules = LaneRules.from_mapping({k: v for k, v in (lane_rules_payload or {}).items()
                                    if k in LaneRules.__dataclass_fields__})
    payload = export_payload(policy.cpu(), encoder.cpu(), rules, **extra)
    torch.save(payload, path)
    return file_sha(path)


def train(data_dir, out_dir, *, epochs: int, seed: int, cfg_overrides=None, device=None, log=print):
    cfg = {**DEFAULT_CONFIG, **(cfg_overrides or {})}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "student_final.pt").exists():
        raise FileExistsError(f"{out_dir}/student_final.pt exists: use a new output directory")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_determinism(seed)
    t0 = time.time()
    rt = setup_runtime(cfg)
    ds = D.load_dataset(data_dir, pre_onset_s=int(cfg["pre_onset_s"]), lane_target=cfg["lane_target"],
                        require_receipts=bool(cfg["require_receipts"]))
    seeds = [e["seed"] for e in ds.episodes]
    train_eps, val_eps = D.split_episodes(seeds, float(cfg["val_fraction"]), seed)
    w_plan, w_other, keep, counts = weight_plan(ds, train_eps, cfg)
    prior = lane_prior(ds, train_eps)
    policy, encoder, lane_rules_payload, init_record = init_networks(cfg, prior, seed=seed)
    policy.to(device)
    encoder.to(device)
    b = Batcher(ds, device)
    frame_ep = ds.frame_ep
    train_frames_all = np.flatnonzero(np.isin(frame_ep, train_eps))
    val_frames = np.flatnonzero(np.isin(frame_ep, val_eps))
    row_cat = ds.cat
    # per-row static weight (other rows are additionally Bernoulli-subsampled each epoch)
    log(json.dumps(dict(event="data", episodes=len(seeds), train_eps=len(train_eps), val_eps=len(val_eps),
                        frames=ds.n_frames, rows=int(len(ds.s)), counts=counts, w_plan=w_plan,
                        w_other=w_other, keep_other=keep, lane_prior=prior.tolist(),
                        load_s=round(time.time() - t0, 1), device=str(device))))
    params = list(policy.parameters()) + list(encoder.parameters())
    opt = torch.optim.AdamW(params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    metrics_path = out_dir / "metrics.jsonl"
    history = []
    val0 = evaluate(policy, encoder, rt, b, val_frames, int(cfg["batch_frames"])) if len(val_frames) else None
    history.append(dict(epoch=0, val=val0))
    with open(metrics_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(history[-1]) + "\n")
    w_row = torch.zeros(len(ds.s), device=device)
    plan_rel = torch.as_tensor((row_cat == D.CAT_PLAN) | (row_cat == D.CAT_PRE), device=device)
    other = torch.as_tensor(row_cat == D.CAT_OTHER, device=device)
    rng = np.random.default_rng(seed)
    in_train_row = torch.as_tensor(np.isin(np.repeat(frame_ep, np.diff(ds.frame_off)), train_eps), device=device)
    for epoch in range(1, epochs + 1):
        te = time.time()
        policy.train()
        encoder.train()
        keep_mask = torch.as_tensor(rng.random(len(ds.s)) < keep, device=device)
        w_row.zero_()
        w_row[plan_rel & in_train_row] = w_plan
        w_row[other & in_train_row & keep_mask] = w_other
        cs = np.concatenate(([0], np.cumsum((w_row > 0).cpu().numpy().astype(np.int64))))
        has = cs[ds.frame_off[1:]] - cs[ds.frame_off[:-1]]
        frames = train_frames_all[has[train_frames_all] > 0]
        frames = frames[rng.permutation(len(frames))]
        sums = dict(loss=0.0, accel=0.0, lane=0.0, w_accel=0.0, w_lane=0.0, infeasible=0, batches=0)
        for i in range(0, len(frames), int(cfg["batch_frames"])):
            rows, valid = b.rows(frames[i:i + int(cfg["batch_frames"])])
            out = forward(policy, encoder, rt, b.state(rows, valid))
            lab = gather_labels(b, rows, valid, out["index"])
            w = torch.where(out["cav"], w_row[lab["row"]], torch.zeros_like(out["accel"]))
            fin = torch.isfinite(lab["acc"]) & (w > 0)
            la = accel_loss(cfg, policy, out, torch.nan_to_num(lab["acc"]))
            ce, lane_ok, lane_bad = lane_ce(out, lab["lane"])
            lane_ok = lane_ok & (w > 0)
            wa, wl = (w * fin).sum(), (w * lane_ok).sum()
            loss_a = (w * la * fin).sum() / wa.clamp_min(1e-9)
            loss_l = (w * ce * lane_ok).sum() / wl.clamp_min(1e-9)
            loss = float(cfg["accel_coef"]) * loss_a + float(cfg["lane_coef"]) * loss_l
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, float(cfg["grad_clip"]))
            opt.step()
            sums["loss"] += float(loss.detach())
            sums["accel"] += float(loss_a.detach())
            sums["lane"] += float(loss_l.detach())
            sums["infeasible"] += int((lane_bad & (w > 0)).sum())
            sums["batches"] += 1
        nb = max(1, sums["batches"])
        val = evaluate(policy, encoder, rt, b, val_frames, int(cfg["batch_frames"])) if len(val_frames) else None
        rec = dict(epoch=epoch, train=dict(loss=sums["loss"] / nb, accel=sums["accel"] / nb, lane=sums["lane"] / nb,
                                           frames=int(len(frames)), batches=sums["batches"],
                                           lane_label_infeasible_weighted_rows=sums["infeasible"]),
                   val=val, epoch_s=round(time.time() - te, 2))
        history.append(rec)
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        pr = (val or {}).get("plan_related", {})
        npl = (val or {}).get("non_plan", {})
        log(f"epoch {epoch}/{epochs} loss {rec['train']['loss']:.4f} (a {rec['train']['accel']:.4f} "
            f"l {rec['train']['lane']:.4f}) val plan mae {pr.get('accel_mae')} acc {pr.get('lane_accuracy')} | "
            f"non-plan mae {npl.get('accel_mae')} acc {npl.get('lane_accuracy')} [{rec['epoch_s']} s]")
    data_manifest = [dict(e) for e in ds.episodes]
    manifest_sha = hashlib.sha256(json.dumps(data_manifest, sort_keys=True).encode()).hexdigest()
    extra = dict(iteration=int(epochs), student="distill_bc/1",
                 distill=dict(teacher="LEARNED_TRIGGER", data_manifest_sha256=manifest_sha,
                              train_seeds=[seeds[i] for i in train_eps], val_seeds=[seeds[i] for i in val_eps],
                              seed=int(seed), epochs=int(epochs), checkpoint_rule="final epoch (pre-registered)",
                              config=json.loads(json.dumps(cfg)), init=json.loads(json.dumps(init_record))))
    ck_sha = export(policy, encoder, lane_rules_payload, extra, out_dir / "student_final.pt")
    report = dict(checkpoint=str(out_dir / "student_final.pt"), checkpoint_sha256=ck_sha, config=cfg,
                  init=init_record, runtime=rt["record"], counts=counts, weights=dict(w_plan=w_plan, w_other=w_other,
                  keep_other=keep), lane_prior=prior.tolist(), episodes=data_manifest, skipped=ds.skipped,
                  data_manifest_sha256=manifest_sha, train_seeds=extra["distill"]["train_seeds"],
                  val_seeds=extra["distill"]["val_seeds"], epochs=epochs, seed=seed, device=str(device),
                  torch=torch.__version__, python=platform.python_version(), wall_s=round(time.time() - t0, 1),
                  final_val=history[-1].get("val"), init_val=history[0].get("val"))
    (out_dir / "REPORT.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    log(f"exported {out_dir / 'student_final.pt'} sha256 {ck_sha}")
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="teacher data dir (shards/ + receipts/)")
    p.add_argument("--out", required=True, help="NEW output directory")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--config", help="JSON file of DEFAULT_CONFIG overrides")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    overrides = json.loads(Path(a.config).read_text(encoding="utf-8")) if a.config else {}
    unknown = set(overrides) - set(DEFAULT_CONFIG)
    if unknown:
        raise SystemExit(f"unknown config keys {sorted(unknown)}")
    train(a.data, a.out, epochs=a.epochs, seed=a.seed, cfg_overrides=overrides, device=a.device)


if __name__ == "__main__":
    main()
