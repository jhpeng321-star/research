"""Offline decision replay for the bounded PPO pilot (PP-07; zero simulation).

Reconstructs, from saved artifacts only:
  * exact phase timelines and decision milestones for any completed dev arm;
  * a fixed registry of TRAINING decision rows (source artifact, scenario,
    episode, round, time, phase, mask, action);
  * deterministic action comparison between the initial actor (ckpt_init +
    fresh initial normalizer), the final evaluated actor (ckpt_u8 + the
    exact evaluated normalizer) and the predeclared feasible-activation rule.

No simulator is launched. Deterministic selection matches the development
evaluator: argmax over the masked logits (masked actions forced to -inf,
ties broken toward the lower action index by torch.argmax).
"""
import json
from pathlib import Path

import torch

from traffic_shepherd import ppo_contract as C
from traffic_shepherd.ppo_features import RunningNormalizer

A_WAIT, A_ACTIVATE = C.A_WAIT, C.A_ACTIVATE
A_RELEASE, A_CONTINUE = C.A_RELEASE, C.A_CONTINUE


def load_registry(training_root, schedule):
    """Every saved TRAINING decision row, identified by source."""
    rows = []
    for entry in schedule:
        i = entry["episode_index"]
        src = Path(training_root) / f"episode_{i:03d}" / "rollout.json"
        rollout = json.loads(src.read_text(encoding="utf-8"))
        for row in rollout:
            rows.append(dict(source=str(src), scenario_seed=entry["seed"],
                             scenario_loading=entry["loading"], episode_index=i,
                             round_index=i // 4, time_s=row["time_s"],
                             phase=row["phase"], mask=list(row["mask"]),
                             executed_action=row["action"],
                             obs_stacked_raw=list(row["obs_stacked_raw"])))
    return rows


def replay_round_normalizers(training_root, updates=8, episodes_per_update=4):
    """Deterministically replay the round-start normalizer states.

    train() starts each round with a fresh-then-accumulating Welford state
    and updates it once per round with that round's raw stacked observations
    in episode order. The per-round files were not retained; this replay
    reproduces them exactly because the update rule is deterministic.
    """
    norm = RunningNormalizer(C.OBS_DIM)
    states = {}
    for u in range(updates):
        states[u] = norm.state_dict()
        for i in range(u * episodes_per_update, (u + 1) * episodes_per_update):
            for row in json.loads((Path(training_root) / f"episode_{i:03d}" /
                                   "rollout.json").read_text(encoding="utf-8")):
                norm.update(row["obs_stacked_raw"])
    return states


def _argmax_action(net, obs, mask):
    with torch.no_grad():
        distribution = net.distribution(obs, torch.tensor([mask]))
        return int(distribution.logits.argmax(dim=-1).item()), distribution


def initial_actor_action(net_init, mask):
    """Initial deployed actor: fresh normalizer (count=0) maps every
    observation to the all-zero vector of the checkpoint's input width."""
    in_dim = getattr(getattr(net_init, 'body', None), '0', None)
    in_dim = in_dim.in_features if in_dim is not None else int(mask[1] and 0) or C.OBS_DIM
    obs = torch.zeros((1, in_dim), dtype=torch.float32)
    action, distribution = _argmax_action(net_init, obs, mask)
    p_act = float(distribution.probs[0, A_ACTIVATE])
    return action, p_act


def final_actor_action(net_final, final_normalizer_state, raw_stacked_obs, mask):
    state = RunningNormalizer.from_state(final_normalizer_state)
    obs = torch.tensor([state.normalize(raw_stacked_obs)], dtype=torch.float32)
    action, distribution = _argmax_action(net_final, obs, mask)
    p_act = float(distribution.probs[0, A_ACTIVATE])
    return action, p_act


def compare_registry(rows, net_init, net_final, final_normalizer_state):
    """A/B/C comparison over the fixed registry.

    A: initial actor + its initial (fresh) normalizer.
    B: final evaluated actor + the exact evaluated normalizer.
    C: feasible-activation rule on activation rows only; HOLD rows are
       reported without an invented duration rule.
    """
    stats = dict(
        rows_total=len(rows),
        activation_rows=0, hold_rows=0, single_option_rows=0,
        init=dict(activate_chosen=0, p_act_sum=0.0),
        final=dict(activate_chosen=0, p_act_sum=0.0,
                   wait_to_activate=0, activate_to_wait=0,
                   continue_to_release=0, release_to_continue=0),
        rule_agreement_activation=0,
        per_scenario={})
    for row in rows:
        mask = row["mask"]
        scen = f"s{row['scenario_seed']}"
        scen_stats = stats["per_scenario"].setdefault(scen, dict(
            rows=0, activation_rows=0, hold_rows=0, single_option=0,
            init_activate=0, final_activate=0, wait_to_activate=0,
            activate_to_wait=0, continue_to_release=0, release_to_continue=0,
            rule_agree=0))
        scen_stats["rows"] += 1
        if row["phase"] == "inactive" and mask[A_ACTIVATE]:
            stats["activation_rows"] += 1
            scen_stats["activation_rows"] += 1
        elif row["phase"] == "hold":
            stats["hold_rows"] += 1
            scen_stats["hold_rows"] += 1
        if mask != [True, True]:
            stats["single_option_rows"] += 1
            scen_stats["single_option"] += 1

        a_init, p_init = initial_actor_action(net_init, mask)
        a_final, p_final = final_actor_action(net_final, final_normalizer_state,
                                              row["obs_stacked_raw"], mask)
        stats["init"]["p_act_sum"] += p_init
        stats["final"]["p_act_sum"] += p_final

        if row["phase"] == "inactive" and mask[A_ACTIVATE]:
            stats["init"]["activate_chosen"] += a_init == A_ACTIVATE
            stats["final"]["activate_chosen"] += a_final == A_ACTIVATE
            scen_stats["init_activate"] += a_init == A_ACTIVATE
            scen_stats["final_activate"] += a_final == A_ACTIVATE
            # rule C: ACTIVATE whenever feasible
            if a_final == A_ACTIVATE:
                stats["rule_agreement_activation"] += 1
                scen_stats["rule_agree"] += 1
            if a_init == A_WAIT and a_final == A_ACTIVATE:
                stats["final"]["wait_to_activate"] += 1
                scen_stats["wait_to_activate"] += 1
            if a_init == A_ACTIVATE and a_final == A_WAIT:
                stats["final"]["activate_to_wait"] += 1
                scen_stats["activate_to_wait"] += 1
        elif row["phase"] == "hold":
            pair = (a_init, a_final)
            if pair == (A_CONTINUE, A_RELEASE):
                stats["final"]["continue_to_release"] += 1
                scen_stats["continue_to_release"] += 1
            if pair == (A_RELEASE, A_CONTINUE):
                stats["final"]["release_to_continue"] += 1
                scen_stats["release_to_continue"] += 1
    return stats


def reconstruct_timeline(decision_log_path, summary_path):
    """Milestones from source artifacts: phase spans from summary
    phase_seconds (the authoritative per-second grid) and requested/executed
    decisions from the policy decision log."""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    phase_seconds = summary["phase_seconds"]
    t, spans = 0, {}
    for phase in ("damping", "assembly", "hold", "taper", "release"):
        count = phase_seconds.get(phase, 0)
        if count:
            spans[phase] = (t, t + count)
        t += count
    log_path = Path(decision_log_path)
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    decisions = [dict(time_s=e["time_s"], phase=e["phase"],
                      requested=e["requested_action"],
                      executed=e["executed_action"],
                      rejection=e.get("rejection_reason"))
                 for e in log
                 if e["executed_action"] in ("activate", "continue", "release")
                 or e.get("rejection_reason") == "activation_refused_at_execution"]
    first_refusal = next((d["time_s"] for d in decisions
                          if d.get("rejection") == "activation_refused_at_execution"), None)
    # arms without a decision log (fixed wall): the first action opportunity
    # and executed activation are the registered decision time (damping end)
    activation = next((d["time_s"] for d in decisions
                       if d.get("executed") == "activate"), None)
    continuations = [d["time_s"] for d in decisions
                     if d.get("executed") == "continue"]
    release = next((d["time_s"] for d in decisions
                    if d.get("executed") == "release"), None)
    hold_start = spans["hold"][0] if "hold" in spans else None
    taper_start = spans["taper"][0] if "taper" in spans else None
    handback = spans["release"][0] if "release" in spans else None
    return dict(
        first_control_tick=0.0,
        first_refused_activation_s=first_refusal,
        activation_executed_s=activation,
        assembly_duration_s=(hold_start - activation
                             if activation is not None and hold_start is not None else None),
        hold_start_s=hold_start,
        admitted_hold_s=None,  # admitted duration is read from the committed
                               # profile at runtime, not persisted per tick;
                               # see TIMINGS reconstruction notes
        delivered_hold_s=(taper_start - hold_start
                          if hold_start is not None and taper_start is not None else None),
        release_request_s=release,
        handback_completed_s=handback,
        continuation_count=len(continuations),
        continuation_times_s=continuations,
        phase_spans=spans,
        decisions=decisions)
