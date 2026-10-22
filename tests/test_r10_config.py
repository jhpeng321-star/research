"""Round 10 = round 9 + exactly fix B: the config difference is pinned, the loss and trainable set are real."""
from pathlib import Path

import pytest
import torch

from wms.cli.assemble import build_model, build_physics
from wms.config.load import load_raw, load_train_config
from wms.train.losses.compose import build_loss
from wms.train.settings import TrainSettings
from wms.train.trainer import build_optimizer

ROOT = Path(__file__).resolve().parents[1]
ROUND9 = ROOT / "configs/c4_wm_v10round9.toml"
ROUND10 = ROOT / "configs/c4_wm_v10round10.toml"

#: leaf paths allowed to differ between the effective round-9 and round-10 configs.
ALLOWED = {"name", "model.init_from", "optim.freeze_prefixes", "optim.lr_scale", "runtime.out_dir"}


def leaves(node, prefix=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{prefix}{key}.")
    else:
        yield prefix[:-1], node


def test_round10_differs_from_round9_only_by_fix_b():
    a, b = dict(leaves(load_raw(ROUND9))), dict(leaves(load_raw(ROUND10)))
    # The merged round-9 draft disables this adjoint for memory; the owner requires
    # it in both final presets. Compare against that explicit final-round-9 value.
    assert b["model.merge_pivot_state_st"] is True
    a["model.merge_pivot_state_st"] = True
    changed = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    changed = {"optim.lr_scale" if k.startswith("optim.lr_scale.") else k for k in changed}
    cutin = {k for k in changed if k.startswith("loss.cutin_reaction.")}
    assert changed - cutin <= ALLOWED, sorted(changed - cutin - ALLOWED)
    assert cutin == {f"loss.cutin_reaction.{k}" for k in
                     ("weight", "control_weight", "pre_s", "max_window_s", "huber_delta_mps2")}
    # the round-9 loss composition, data mix and curriculum are untouched
    for key in ("loss.merge_core.weight", "loss.merge_core.tf_weight", "curriculum.stage_steps",
                "optim.lr_per_stage", "optim.batch_size_per_stage", "data.c4_base"):
        assert a[key] == b[key]


def test_round10_loss_contract_and_composition():
    cfg = load_train_config(ROUND10)
    cut = cfg.loss.cutin_reaction
    assert cut.weight > 0 and cut.control_weight == 1.0 and cut.pre_s == 1.0
    assert cut.max_window_s == 6.0 and cut.huber_delta_mps2 == 0.5
    names9 = set(build_loss(TrainSettings.from_config(load_train_config(ROUND9))).names)
    names10 = set(build_loss(TrainSettings.from_config(cfg)).names)
    assert names10 - names9 == {"cutin_reaction"} and names9 <= names10
    assert "cutin_reaction" in cfg.required_log_terms()
    assert cfg.model.merge_decision == "first_passage" and cfg.model.merge_passage_law == "hybrid"
    assert not cfg.model.leader_change_features
    assert cfg.model.merge_pivot_state_st is True
    assert cfg.physics.ramp_lane_end_stop is True
    assert cfg.physics.ramp_risk_retain is True
    model = build_model(cfg, build_physics(cfg))
    assert model.spec.merge_pivot_state_st is True


def test_round10_trains_only_the_responsible_emitters_and_freezes_the_merge_pivot():
    cfg = load_train_config(ROUND10)
    model = build_model(cfg, build_physics(cfg))
    _, report = build_optimizer(model, lr=1e-4, weight_decay=0,
                                freeze_prefixes=tuple(cfg.optim.freeze_prefixes),
                                lr_scale=dict(cfg.optim.lr_scale))
    assert report["unmatched_freeze_prefixes"] == [] and report["unmatched_lr_scale_prefixes"] == []
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable
    assert all(n.startswith(("temporal", "gate_g_max_logit", "gate_tau_raw")) for n in trainable), sorted(trainable)
    assert not any(n.startswith(("merge_pivot", "merge_core", "driver_", "lane_", "residual_net", "history_proj", "k_embedding"))
                   for n in trainable)
    # round 9 trains merge_pivot only; round 10 leaves it bitwise frozen
    cfg9 = load_train_config(ROUND9)
    m9 = build_model(cfg9, build_physics(cfg9))
    build_optimizer(m9, lr=1e-4, weight_decay=0, freeze_prefixes=tuple(cfg9.optim.freeze_prefixes))
    assert {n for n, p in m9.named_parameters() if p.requires_grad} == {
        n for n, _ in m9.named_parameters() if n.startswith("merge_pivot")}


def test_round10_init_checkpoint_path_and_gate_chain():
    cfg = load_train_config(ROUND10)
    assert cfg.model.init_from.endswith("runs/c4_wm_v10_round9/wm_s_stage300.pt")
    gate = (ROOT / "scripts/round10_gate.sh").read_bytes()
    assert b"\r" not in gate and b"configs/c4_wm_v10round10.toml" in gate
    assert b"runs/c4_wm_v10_round10" in gate and b"repo_r10" in gate


def test_cutin_weight_comes_from_the_documented_rule_file():
    import json
    rule = json.loads((ROOT / "docs/r10_weight_rule.json").read_text(encoding="utf-8"))
    cfg = load_train_config(ROUND10)
    assert cfg.loss.cutin_reaction.weight == pytest.approx(rule["chosen_weight"], rel=1e-9)
    assert torch.isfinite(torch.tensor(rule["chosen_weight"])) and rule["chosen_weight"] > 0
