"""c4/learnfix (MAIN 2026-11-08 ~22:3x, LEARNDIAG_20261108): an action-level learning signal for the trigger head.

Opt-in by the registered ``learnfix`` config value (absent = every path byte-identical to before). B = R code.

1. EV-BC labels (fullpol_learn.ev_bc_records): a row is labelled ``act`` when a qualifying guidance family is
   offered (and, for a privileged situational row, the teacher admits it); ``none`` ONLY where a privileged teacher
   explicitly rejects every offered situational family (bin observed, not admitted); every other row is
   UNLABELLED (no BC loss). Per update, act and none rows get equal total weight (per-row weight <= 1).
2. Trigger PPO (track2_backend.train):
   - fairness harm (paired same-world A episode, fullpol_fair.penalties) is attributed only to the rows where a
     non-``none`` trigger action was taken (spread evenly over them), not to every row;
   - the local delay reward stays per row; advantage = return - recorded value (+ G4 terms), then a per-cell
     baseline (mean over that cell's choice rows) is subtracted and the result is normalised over choice rows;
     rows without a choice get advantage 0;
   - ``epochs`` epochs over minibatches of ``minibatch_ticks`` decision ticks, order from a SEPARATE seeded
     generator (the shared episode RNG is never consumed);
   - the KL cap (layer kl_budget, 0.045) and the reported KL are the mean over CHOICE rows; a step over the cap
     is reverted and the PPO update stops;
   - entropy weight = min(registered, entropy_frac * RMS(adv) / mean H) so the entropy term is < 10% of |adv|.
"""
import math
from copy import deepcopy

import torch

RULE = ('c4/learnfix v1 (MAIN 2026-11-08, LEARNDIAG): EV-BC act / explicit-teacher-none / unlabelled labels with '
        'equal act:none total weight (row weight <= 1); trigger PPO with fairness harm on acting rows only, per-cell '
        'baseline + choice-row normalised advantages, epochs x minibatches of decision ticks (separate seeded '
        'order), choice-row KL cap and report, entropy weight capped at entropy_frac x RMS(adv) / mean H')
DEFAULT = dict(epochs=4, minibatch_ticks=32, entropy_frac=0.1, row_weight_cap=1.0)


def value(**overrides):
    return dict(DEFAULT, rule=RULE, **overrides)


def settings(cfg):
    """None when absent (unchanged paths); else the validated registration."""
    raw = (cfg or {}).get('learnfix')
    if raw is None:
        return None
    if (not isinstance(raw, dict) or raw.get('rule') != RULE or set(raw) != set(DEFAULT) | {'rule'}
            or type(raw['epochs']) is not int or not 1 <= raw['epochs'] <= 16
            or type(raw['minibatch_ticks']) is not int or not 1 <= raw['minibatch_ticks'] <= 4096
            or not 0 < float(raw['entropy_frac']) <= 1 or not 0 < float(raw['row_weight_cap']) <= 1):
        raise ValueError('learnfix needs the registered rule, epochs (1..16), minibatch_ticks, entropy_frac '
                         'and row_weight_cap in (0, 1]')
    return dict(raw)


# ---------------------------------------------------------------- 1. EV-BC label weights

def balance(kinds, cap=1.0):
    """{kind: per-row weight} for row kinds 'act' / 'none' (unlabelled rows are never passed): equal total weight
    per class, every row weight <= cap; a class without a partner keeps weight cap only if it is 'act' (lone
    'none' labels get 0, so EV-BC never pushes a whole update toward none)."""
    n_act, n_none = kinds.count('act'), kinds.count('none')
    if n_act and n_none:
        m = min(n_act, n_none)
        return dict(act=cap*m/n_act, none=cap*m/n_none)
    return dict(act=cap if n_act else 0., none=0.)


# ---------------------------------------------------------------- 2. fairness on acting rows

def acting_rows(by_key):
    """{(episode_id, time_s): [bool per member]}: a non-none trigger action was taken by that CAV at that tick."""
    out = {}
    for key, sample in by_key.items():
        trig = sample.get('track2_trigger')
        if trig is None:
            continue
        members = sample['context']['members']
        acts = dict(zip(trig['context']['members'], trig['actions']))
        out[key] = [int(acts.get(v, 0)) > 0 for v in members]
    return out


def apply_fair_acting(targets, harm_by_episode, acting):
    """Each episode's harm (veh-s, vs the paired A episode) spread evenly over the rows where a non-none action
    was taken; an episode without such a row gets no PPO harm term (receipted). Returns (targets, receipt)."""
    out = {k: list(v) for k, v in targets.items()}
    receipt = []
    for eid, harm in sorted(harm_by_episode.items()):
        rows = [(k, i) for k in out if k[0] == eid for i, a in enumerate(acting.get(k, ())) if a]
        receipt.append(dict(episode_id=eid, harm_veh_s=harm, acting_rows=len(rows),
                            attributed=bool(harm > 0 and rows)))
        if harm <= 0 or not rows:
            continue
        for k, i in rows:
            out[k][i] -= harm/len(rows)
    return out, receipt


# ---------------------------------------------------------------- 3. trigger PPO

def choice_mask(batch):
    return batch.mask[:, 1:].any(1)


def advantages(ticks):
    """In place: per-cell baseline over choice rows, then normalisation over all choice rows; non-choice 0.
    ``ticks``: [dict(batch, adv (list), cell)] -> adds 'norm_adv' (tensor). Returns a receipt."""
    raw = {}
    for t in ticks:
        a = torch.as_tensor(t['adv'], dtype=torch.float32)
        c = choice_mask(t['batch'])
        t['_a'], t['_c'] = a, c
        raw.setdefault(t['cell'], []).append(a[c])
    base = {cell: float(torch.cat(v).mean()) if sum(len(x) for x in v) else 0. for cell, v in raw.items()}
    centred = torch.cat([t['_a'][t['_c']]-base[t['cell']] for t in ticks]) if ticks else torch.zeros(0)
    std = float(centred.std()) if len(centred) > 1 else 1.
    std = std if math.isfinite(std) and std > 1e-8 else 1.
    for t in ticks:
        n = torch.zeros_like(t['_a'])
        n[t['_c']] = (t['_a'][t['_c']]-base[t['cell']])/std
        t['norm_adv'] = n
        del t['_a'], t['_c']
    return dict(choice_rows=int(len(centred)), cell_baseline=base, std=std,
                raw_rms=float(centred.square().mean().sqrt()) if len(centred) else 0.)


@torch.no_grad()
def _logits(policy, ticks):
    return [policy.distribution(t['batch'].context, t['batch'].descriptors, t['batch'].mask)[0] for t in ticks]


@torch.no_grad()
def choice_kl(policy, ticks, reference):
    num, den = 0., 0
    for t, ref in zip(ticks, reference):
        c = choice_mask(t['batch'])
        if not bool(c.any()):
            continue
        after = policy.distribution(t['batch'].context, t['batch'].descriptors, t['batch'].mask)[0]
        kl = torch.distributions.kl_divergence(torch.distributions.Categorical(logits=ref[c]),
                                               torch.distributions.Categorical(logits=after[c]))
        num += float(kl.sum())
        den += int(c.sum())
    return num/den if den else 0.


def trigger_ppo(policy, optimizer, ticks, *, settings, ppo, kl_budget, seed):
    """Choice-row PPO over decision ticks. ``ticks``: [dict(batch, adv, ret, cell)]; ``ppo``: registered
    trigger_ppo (clip_ratio, value_weight, entropy_weight, max_grad_norm). Returns the receipt."""
    adv_receipt = advantages(ticks)
    reference = _logits(policy, ticks)
    with torch.no_grad():
        ents = [torch.distributions.Categorical(logits=r[choice_mask(t['batch'])]).entropy()
                for t, r in zip(ticks, reference) if bool(choice_mask(t['batch']).any())]
    mean_h = float(torch.cat(ents).mean()) if ents else 0.
    rms = float(torch.cat([t['norm_adv'][choice_mask(t['batch'])] for t in ticks]).square().mean().sqrt()) \
        if adv_receipt['choice_rows'] else 0.
    ent_w = min(float(ppo['entropy_weight']), settings['entropy_frac']*rms/max(mean_h, 1e-6)) if rms else 0.
    gen = torch.Generator().manual_seed(int(seed))
    params = policy.learning_parameters()
    steps, stopped = [], None
    size = settings['minibatch_ticks']
    for epoch in range(settings['epochs']):
        order = torch.randperm(len(ticks), generator=gen).tolist()
        for start in range(0, len(order), size):
            mb = [ticks[i] for i in order[start:start+size]]
            before = deepcopy(policy.state_dict())
            opt = deepcopy(optimizer.state_dict())
            actor, value, ent, n_choice, n_rows = 0., 0., 0., 0, 0
            for t in mb:
                b = t['batch']
                logp, values, entropy = policy.evaluate(b)
                c = choice_mask(b)
                ret = torch.as_tensor(t['ret'], dtype=values.dtype)
                value = value+.5*(ret-values).square().sum()
                n_rows += len(values)
                if bool(c.any()):
                    ratio = (logp-b.logp)[c].exp()
                    a = t['norm_adv'][c]
                    obj = torch.minimum(ratio*a, ratio.clamp(1-ppo['clip_ratio'], 1+ppo['clip_ratio'])*a)
                    actor = actor-obj.sum()
                    ent = ent+entropy[c].sum()
                    n_choice += int(c.sum())
            loss = value*float(ppo['value_weight'])/max(n_rows, 1)
            if n_choice:
                loss = loss+(actor-ent_w*ent)/n_choice
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, ppo['max_grad_norm'], error_if_nonfinite=True)
            optimizer.step()
            kl = choice_kl(policy, ticks, reference)
            accepted = math.isfinite(kl) and kl <= kl_budget
            steps.append(dict(epoch=epoch, ticks=len(mb), choice_rows=n_choice, loss=float(loss.detach()),
                              grad_norm=float(norm), choice_kl=kl, accepted=accepted))
            if not accepted:
                policy.load_state_dict(before)
                optimizer.load_state_dict(opt)
                stopped = dict(epoch=epoch, step=len(steps)-1, choice_kl=kl)
                break
        if stopped:
            break
    optimizer.zero_grad(set_to_none=True)
    return dict(rule=RULE, ticks=len(ticks), advantages=adv_receipt, entropy_weight=ent_w, mean_entropy=mean_h,
                adv_rms=rms, steps=steps, accepted=sum(s['accepted'] for s in steps),
                stopped=stopped, final_choice_kl=choice_kl(policy, ticks, reference), kl_budget=kl_budget,
                kl_rule='mean KL over choice rows (>= 1 non-none valid slot), all ticks of the update')
