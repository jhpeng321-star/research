"""FULLPOL learning-speed / learning-budget audit (CPU arithmetic only; report, never a gate).

Registered values are read, not changed: trigger KL cap 0.045 per update (T2D), imitation
optimiser steps per update (MAIN 19:5x: 20), 9 trigger slots (none + 8 candidates).

* ``concentration``: geometry of moving a categorical policy from its start distribution to
  ``target_p`` on one labelled slot set under a per-update KL cap. KL(old||new) <= cap allows
  a Fisher-Rao step of about sqrt(2*cap); the Fisher-Rao distance d = 2*arccos(sum sqrt(p q))
  gives the minimum number of capped updates ceil(d / sqrt(2*cap)) (straight-line bound;
  measured pace is slower - LEARNAUDIT: clean2 CE 2.13 -> 1.21 (u6) -> 0.39 (u12)).
* ``ppo_episodes``: paired-episode count for a two-sided z-test to resolve a mean episode gain
  of ``effect`` with per-episode SD ``sd`` (power 0.8, alpha 0.05) - a lower bound on what
  per-decision PPO needs, because the per-decision team-reward signal is diluted further.
* ``sd_from_ci``: per-episode SD implied by a paired-seed bootstrap CI (normal approximation).
"""
import math


def fisher_rao(p, q):
    bc = sum(math.sqrt(a*b) for a, b in zip(p, q))
    return 2.*math.acos(max(-1., min(1., bc)))


def kl(p, q):
    return sum(a*math.log(a/b) for a, b in zip(p, q) if a > 0)


def concentration(*, slots=9, target_p=.8, kl_cap=.045, start=None):
    """Minimum capped updates from ``start`` (uniform) to target_p on one slot (rest uniform)."""
    p0 = list(start) if start is not None else [1./slots]*slots
    rest = (1.-target_p)/(slots-1)
    p1 = [target_p]+[rest]*(slots-1)
    d = fisher_rao(p0, p1)
    step = math.sqrt(2.*kl_cap)
    return dict(slots=slots, target_p=target_p, kl_cap=kl_cap, kl_total_nats=kl(p0, p1),
                fisher_rao=d, max_step=step, min_updates=math.ceil(d/step),
                note='straight-line lower bound; the cap is a mean over supervised contexts')


def sd_from_ci(lo, hi, n):
    return (hi-lo)/(2*1.96)*math.sqrt(n)


def ppo_episodes(effect, sd, *, power=.8, alpha=.05):
    z = {(.8, .05): (1.959964, .841621)}[(power, alpha)]
    return math.ceil(((z[0]+z[1])*sd/abs(effect))**2)


def clean5a_estimate(*, effect_veh_s, ci95, n_seeds, episodes_per_update=16, updates=24, kl_cap=.045,
                     imitation_steps=20, slots=9):
    """Learning-budget numbers for READY (inputs: the S-REGIME row of the guided family)."""
    sd = sd_from_ci(*ci95, n_seeds)
    need = ppo_episodes(effect_veh_s, sd)
    conc = concentration(slots=slots, kl_cap=kl_cap)
    return dict(ev_bc=dict(conc, imitation_steps_per_update=imitation_steps,
                           labels='every decision row of every ordinary B episode (dense, per-condition)'),
                ppo=dict(per_episode_sd_veh_s=sd, effect_veh_s=effect_veh_s, paired_episodes_needed=need,
                         run_episodes_per_arm=episodes_per_update*updates,
                         note='z-test bound for an ALWAYS-ON strategy effect; per-decision credit is diluted '
                              'further, so PPO alone is not expected to discover the strategy in 24 updates'))
