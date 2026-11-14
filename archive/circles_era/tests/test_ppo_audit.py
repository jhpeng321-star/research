"""Regression tests for PP-06 audit findings: all-arm comparator labeling,
missing-cell integrity, sampling-seed correction.

These cover the demonstrated defects:
  1. cmd_evaluate labeled ONLY the ppo arm — the fixed-wall 4117 beneficial
     and 4115-fixed harmful labels were missed; the rule/missing cells risked
     fabricated outcomes. The corrected grader classifies every arm with the
     SAME registered classifier and keeps missing cells missing.
  2. The executed SamplingActor v1 never used its dedicated generator.
     v2 must draw from the dedicated stream, respect masks, and store the
     matching log-probability.
"""
from pathlib import Path
import json
import torch

import pytest

from traffic_shepherd import ppo_contract as C
from traffic_shepherd.escalation_evidence import BenefitCriteria, benefit_label

CRITERIA = BenefitCriteria(
    primary_minimum_improvement={"system_seconds_per_request": 1.0,
                                 "braking_seconds_per_request": 0.02},
    maximum_cost_increase={"system_seconds_per_request": 0.5,
                           "source_wait_seconds_per_request": 0.1,
                           "braking_seconds_per_request": 0.005,
                           "effort_per_request": 0.5,
                           "ttc_seconds_per_request": 0.02})
KEYMAP = {"service": "system_seconds_per_request", "braking": "braking_seconds_per_request",
          "effort": "effort_per_request", "ttc": "ttc_seconds_per_request",
          "wait": "source_wait_seconds_per_request"}


def classify(diff, collisions=0, teleports=0, complete=True):
    effect = dict(control="damping", treatment="wall",
                  improvements={v: -diff[k] for k, v in KEYMAP.items()},
                  excess_collisions=collisions, excess_teleports=teleports,
                  excess_unfinished=0,
                  control_unfinished_requested=0, treatment_unfinished_requested=0,
                  control_collision_involvements=0, treatment_collision_involvements=collisions,
                  control_starting_teleports=0, treatment_starting_teleports=teleports,
                  complete_requested_service_control=complete,
                  complete_requested_service_treatment=complete,
                  baseline_safety_anomaly=collisions > 0)
    out = benefit_label(effect, CRITERIA)
    return out["label"], out.get("violated_guardrails", [])


# ---- comparator-label defect regressions (audit findings 1/2) -------------

def test_fixed_wall_4117_was_beneficial_not_nonbeneficial():
    diff = dict(service=-5.21625544267058, braking=-0.06458635703918723,
                effort=-4.71345205678607, ttc=-0.05624092888243832,
                wait=-0.05224963715529753)
    got, violated = classify(diff)
    assert got == "beneficial" and violated == []


def test_fixed_wall_4115_was_harmful_not_nonbeneficial():
    diff = dict(service=2.686622320768662, braking=0.0, effort=0.0,
                ttc=0.0, wait=0.0)
    got, violated = classify(diff)
    assert got == "harmful" and "system_seconds_per_request" in violated


def test_grader_labels_all_three_arms_not_only_ppo():
    """The executed grader (cmd_evaluate) classified only mode=='ppo'; the
    corrected grader must apply the identical classifier to every arm."""
    from pathlib import Path as _Path
    from traffic_shepherd.ppo_pilot import cmd_evaluate  # noqa: F401
    import inspect
    src = inspect.getsource(cmd_evaluate)
    assert "benefit_label" in src or "_audit_label" in src
    audit = json.loads((_Path("output/ppo_audit_v1/AUDIT_RESULTS.json"))
                       .read_text(encoding="utf-8"))
    labeled_modes = {v["mode"] for v in audit.get("cells", {}).values()
                     if "label" in v}
    assert {"ppo", "fixed_wall", "scripted_rule"} <= labeled_modes


def test_missing_cell_stays_missing_never_zero():
    audit = json.loads(Path("output/ppo_audit_v1/AUDIT_RESULTS.json").read_text())
    cell = audit["cells"]["s4118_scripted_rule"]
    assert cell["status"] == "MISSING"
    assert "diff" not in cell and cell.get("note", "").startswith("cell never completed")


def test_audit_labels_match_registered_classifier_on_all_cells():
    audit = json.loads(Path("output/ppo_audit_v1/AUDIT_RESULTS.json").read_text())
    for name, cell in audit["cells"].items():
        if cell.get("status") != "complete":
            assert cell["status"] == "MISSING"
            continue
        expect, violated = classify(cell["difference_per_request"],
                                 cell["treatment_collision_involvements"],
                                 cell["treatment_teleports"], cell["complete"])
        assert cell["label"] == expect, (name, cell["label"], expect)
        assert cell["violated_guardrails"] == violated


# ---- sampling v2: dedicated generator correction ---------------------------

def test_same_seed_reproduces_sampled_actions():
    from traffic_shepherd.ppo_net import ActorCritic
    from traffic_shepherd.ppo_features import RunningNormalizer
    from traffic_shepherd.ppo_worker import SamplingActor, SAMPLING_VERSION
    assert SAMPLING_VERSION == "dedicated_generator/2"
    torch.manual_seed(0)
    net = ActorCritic()
    norm = RunningNormalizer(C.OBS_DIM)
    norm.update([0.01] * C.OBS_DIM)
    seq_a = []
    actor = SamplingActor(net, norm.state_dict(), sample_seed=123)
    for _ in range(24):
        seq_a.append(actor([True, True], [0.01] * C.OBS_DIM, "inactive", 5.0))
    actor_b = SamplingActor(net, norm.state_dict(), sample_seed=123)
    actor_b.history = list(actor.history)  # same observation history
    seq_b = []
    for _ in range(24):
        seq_b.append(actor_b([True, True], [0.01] * C.OBS_DIM, "inactive", 5.0))
    assert seq_a == seq_b


def test_different_seeds_control_independent_streams():
    from traffic_shepherd.ppo_net import ActorCritic
    from traffic_shepherd.ppo_features import RunningNormalizer
    from traffic_shepherd.ppo_worker import SamplingActor
    torch.manual_seed(0)
    net = ActorCritic()
    norm = RunningNormalizer(C.OBS_DIM)
    norm.update([0.01] * C.OBS_DIM)
    seqs = []
    for seed in (1, 2):
        actor = SamplingActor(net, norm.state_dict(), sample_seed=seed)
        seq = []
        for _ in range(32):
            seq.append(actor([True, True], [0.01] * C.OBS_DIM, "inactive", 5.0))
        seqs.append(seq)
    assert seqs[0] != seqs[1]


def test_global_rng_use_does_not_alter_dedicated_stream():
    from traffic_shepherd.ppo_net import ActorCritic
    from traffic_shepherd.ppo_features import RunningNormalizer
    from traffic_shepherd.ppo_worker import SamplingActor
    torch.manual_seed(0)
    net = ActorCritic()
    norm = RunningNormalizer(C.OBS_DIM)
    norm.update([0.01] * C.OBS_DIM)
    outputs = []
    for _ in range(2):
        torch.manual_seed(999)            # perturb the GLOBAL stream
        actor = SamplingActor(net, norm.state_dict(), sample_seed=42)
        seq = [actor([True, True], [0.01] * C.OBS_DIM, "inactive", 5.0)
               for _ in range(16)]
        outputs.append(seq)
    assert outputs[0] == outputs[1]       # dedicated stream: identical


def test_sampled_actions_respect_masks_and_stored_logp_matches():
    from traffic_shepherd.ppo_net import ActorCritic
    from traffic_shepherd.ppo_features import RunningNormalizer
    from traffic_shepherd.ppo_worker import SamplingActor
    torch.manual_seed(0)
    net = ActorCritic()
    norm = RunningNormalizer(C.OBS_DIM)
    norm.update([0.01] * C.OBS_DIM)
    actor = SamplingActor(net, norm.state_dict(), sample_seed=7)
    for i in range(40):
        mask = [True, i % 2 == 0]         # ACTIVATE unavailable on odd rows
        action = actor(mask, [0.01] * C.OBS_DIM, "inactive", 5.0)
        if i % 2:
            assert action == C.A_WAIT     # masked action never sampled
    last_action, last_logp, _value = actor.rows[-1]
    obs = torch.tensor([norm.normalize(
        __import__("traffic_shepherd.ppo_worker", fromlist=["stacked_obs"]).stacked_obs(
            actor.history, len(actor.history) - 1))])
    with torch.no_grad():
        d = net.distribution(obs, torch.tensor([[True, False]]))
        lp = float(d.log_prob(torch.tensor([last_action])))
    assert lp == pytest.approx(last_logp, abs=1e-6)
