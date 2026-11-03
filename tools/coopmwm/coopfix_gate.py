"""COOPFIX GATE: offline CPU check that the fixes make the policy learn the cooperative plans.

Given a MAPPO u0 checkpoint (tools/coopmwm/accel_learner schema), a bank of pairs (with v2v
partner features, replay_bank.bank_pairs) and the training options, runs ONE imitation phase
offline - the registered bank_imitation.run loop, no PPO update, no SUMO - and reports:

- passes completed (full epochs without a registered KL stop);
- the final joint KL against the frozen pre-update policy, within the runner's registered
  limit (bank_imitation .045 - imported, never re-stated);
- the imitation L1 error on the COOPERATIVE pairs before/after, with a pair-cluster
  bootstrap 95% CI on the per-member-step fall (PASS: CI entirely below 0);
- per cooperative entry, the fraction of CAV-steps whose action moved toward the plan action;
- whether the zero-initialised v2v input weights moved (non-zero): the policy can use
  partner intent.

Intent extension (c4/intent-int, option ``v2v_intent``, DEFAULT OFF): with the option on the
phase also trains the zero-initialised intent head on the pairs' plan-future-command targets
(``context['intent_target']`` from replay_bank.bank_pairs(intent=True)) with the registered
weight, and the report additionally carries

- the masked plan-future intent error (m/s^2) on the cooperative pairs before/after
  (bank_imitation.intent_error estimator; PASS requires it to fall);
- whether the PARTNER-INTENT input columns of the widened actor moved (non-zero): the
  per-partner ``intent_0..intent_{H-1}`` channels inside the ``K*(6+H+1)`` v2v block,
  reported separately from the plain six-feature partner columns.

Verdict PASS iff all four hold, plus the two intent checks when ``v2v_intent`` is on (the
option-off verdict semantics are byte-identical to the pre-intent gate). Both training options are live and default off:
``bank_filter='statistical'`` (c4/bank-stat-filter, BANKFILT READY) is applied per entry
inside replay_bank.bank_pairs and disclosed here; ``imitation_trust='own_pass'``
(c4/imit-fullpass, IMITFIX READY) runs the phase on its own trust region with the
guaranteed cooperative-first first pass and is forwarded to the registered runner with the
bank's pair_cooperative flags. The READY paths are still polled (file read only) and the
option state is disclosed in the report. The v2v flag defaults OFF; no pre-existing module
behaviour changes with the options off (the gate itself is a new, self-contained module).
"""
import copy
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from . import bank_imitation
from . import v2v_model
from .acceleration import Policy as AccelerationPolicy

FEATURES = v2v_model.FEATURES
ACTOR_LATENT = v2v_model.ACTOR_LATENT
BOOTSTRAP_SEED = 20261101   # same estimator family seed as replay_bank._boot_ci
N_BOOT = 10000
READY_FILES = dict(imitation_trust=Path('E:/agent-work/c4-imit-fullpass/IMITFIX_READY.json'),
                   bank_filter=Path('E:/agent-work/c4-bank-stat-filter/BANKFILT_READY.json'))


class GatePolicy(AccelerationPolicy):
    """acceleration.Policy whose forward consumes the pair-context v2v block.

    The zero-extended input layers (v2v_model.extend_for_v2v) place the v2v channels at
    FEATURES, between the x/pooled block and the latent/rsu tails. With extra width 0 (v2v
    flag off, default) the computation is exactly the registered base forward.
    """

    def v2v_width(self):
        return int(self.actor[0].weight.shape[1])-FEATURES-ACTOR_LATENT

    def v2v_matrix(self, context, n):
        extra = self.v2v_width()
        if extra <= 0:
            return None
        block = context.get('v2v') or {}
        rows = torch.zeros((n, extra))
        for i, vid in enumerate(context['members']):
            row = block.get(vid)
            if row is not None:
                rows[i] = torch.as_tensor(row, dtype=torch.float32).reshape(-1)[:extra]
        return rows

    def forward(self, context):
        x = torch.tensor(context['x'], dtype=torch.float32).reshape(-1, FEATURES)
        rsu = torch.tensor(context['rsu'], dtype=torch.float32)
        rows = self.v2v_matrix(context, len(x))
        latent = self.broadcast(rsu).expand(len(x), -1)
        mean = self.actor(torch.cat((x, latent), -1) if rows is None else torch.cat((x, rows, latent), -1))
        pooled_x = x.mean(0) if len(x) else torch.zeros(FEATURES)
        pooled = pooled_x if rows is None else torch.cat(
            (pooled_x, rows.mean(0) if len(rows) else torch.zeros(rows.shape[1])))
        value = self.critic(torch.cat((pooled, rsu))).squeeze(-1)
        return mean, value


def load_gate_policy(checkpoint, v2v_dim, intent=False, intent_horizon=v2v_model.V2V_INTENT_H):
    """Registered v2v_model.load_checkpoint, rebound to the v2v-aware GatePolicy forward.

    Same state layout (the extension only widens the two input layers in place), so the
    rebind changes nothing about the loaded parameters, optimizer or RNG state.
    ``intent=True`` (DEFAULT OFF) loads through the intent binding config instead, so the
    zero-initialised intent head is attached exactly as in the live runner (the widened
    width is the K*(6+H+1) intent block)."""
    if intent:
        policy, optimizer, generator, payload = v2v_model.load_checkpoint(checkpoint, config=dict(
            v2v=True, v2v_intent=True, v2v_intent_horizon=int(intent_horizon),
            v2v_dim=v2v_dim, accel_bounds=v2v_model.INTENT_BOUNDS), v2v_dim=v2v_dim)
    else:
        policy, optimizer, generator, payload = v2v_model.load_checkpoint(checkpoint, v2v_dim=v2v_dim)
    policy.__class__ = GatePolicy
    return policy, optimizer, generator, payload


# ---------------------------------------------------------------- option interfaces (stubs until READY)

def option_ready(name):
    """Parsed READY json of an option branch, or None while it is not READY (stub active)."""
    path = READY_FILES[name]
    try:
        if path.is_file():
            return json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return None


def option_status():
    return {name: dict(ready_file=str(READY_FILES[name]), ready=option_ready(name) is not None,
                       active='stub' if option_ready(name) is None else 'ready')
            for name in sorted(READY_FILES)}


def filter_report(options, summary):
    """bank_filter option (c4/bank-stat-filter, BANKFILT READY): the statistical rule is
    applied INSIDE replay_bank.bank_pairs (per-entry replicated advantage + paired
    per-vehicle deficit bounds, bank_filter.RULE); the gate validates the option and
    discloses the bank's own summary['bank_filter'] report. It can never be applied to a
    bare pairs file - the rule needs the recorded receipts and branch traces."""
    spec = options.get('bank_filter')
    if not spec:
        return dict(option='bank_filter', enabled=False, active='stub', stage='identity (all admitted pairs)')
    if option_ready('bank_filter') is None:
        raise ValueError('bank_filter option configured but BANKFILT_READY.json is absent (stub active)')
    applied = (summary or {}).get('bank_filter')
    if applied is None:
        raise ValueError("bank_filter='statistical' requires pairs produced by "
                         "replay_bank.bank_pairs(bank_filter='statistical'); a bare pairs file "
                         'cannot apply the per-entry statistical rule')
    return dict(option='bank_filter', enabled=True, active='ready', value=spec,
                stage='applied inside replay_bank.bank_pairs (per-entry replicated advantage '
                      '+ paired per-vehicle deficit bounds)', report=applied)


def trust_kwargs(options, pairs=None, summary=None):
    """imitation_trust option (c4/imit-fullpass, IMITFIX READY): kwargs for the registered
    imitation runner. The registered value 'own_pass' is forwarded by its parameter name
    together with the bank's pair_cooperative flags when they align with the pairs; a runner
    without the parameter raises instead of silently ignoring the option."""
    spec = options.get('imitation_trust')
    if not spec:
        return {}, dict(option='imitation_trust', enabled=False, active='stub',
                        stage='registered KL bound only')
    ready = option_ready('imitation_trust')
    if ready is None:
        raise ValueError('imitation_trust option configured but IMITFIX_READY.json is absent (stub active)')
    parameters = inspect.signature(bank_imitation.run).parameters
    for name in ('imitation_trust', 'trust'):
        if name in parameters:
            kwargs = {name: spec}
            flags = (summary or {}).get('pair_cooperative')
            if 'cooperative' in parameters and flags is not None and len(flags) == len(pairs or ()):
                kwargs['cooperative'] = flags
            return kwargs, dict(option='imitation_trust', enabled=True, active='ready', value=spec,
                                parameter=name, cooperative_flags=kwargs.get('cooperative') is not None)
    raise ValueError('imitation_trust option configured but bank_imitation.run does not accept it')


# ---------------------------------------------------------------- cooperative pair bookkeeping

def is_cooperative(sample):
    """Structural rule (definitional, replay_bank.py: single-CAV plans are never cooperative):
    a pair is cooperative when its committed plan has more than one member."""
    return len(sample.get('behaviour', {}).get('proposal_members', [])) > 1


def cooperative_pairs(pairs, summary=None):
    """(unique cooperative pairs, count of raw cooperative samples in the bank summary).

    bank_pairs flattens to other+coop*(1+full)+coop[:rem]; the first
    summary['exposure']['cooperative_samples_raw'] cooperative positions are exactly one
    unduplicated copy. Without a summary every cooperative pair is kept (oversampling
    disclosed in the count)."""
    coop = [(s, w) for s, w in pairs if is_cooperative(s)]
    raw = None
    if summary and (summary.get('exposure') or {}).get('cooperative_samples_raw') is not None:
        raw = int(summary['exposure']['cooperative_samples_raw'])
        if 0 < raw <= len(coop):
            coop = coop[:raw]
    return coop, raw


def pair_plan_key(sample):
    """Entry identity of a pair: (plan_id, start_s) of its committed plan execution."""
    members = list(sample.get('behaviour', {}).get('proposal_members', []))
    for execution in sample.get('plan_executions', []) or []:
        if list(execution.get('proposal_members', [])) == members:
            return (execution.get('plan_id'), execution.get('start_s'))
    return (None, None)


def cooperative_entries(pairs, summary=None):
    """Unique cooperative pairs grouped into their bank entries.

    With a consistent bank summary the entry durations slice the unique list exactly;
    otherwise consecutive runs of the same (plan_id, start_s) form the groups."""
    coop, raw = cooperative_pairs(pairs, summary)
    durations = None
    if summary and summary.get('entries'):
        durations = [int(e['duration_s']) for e in summary['entries'] if e.get('cooperative')]
        if sum(durations) != len(coop):
            durations = None
    entries = []
    if durations:
        at = 0
        for d in durations:
            entries.append(coop[at:at+d]);at += d
    else:
        previous, run = object(), []
        for pair in coop:
            key = pair_plan_key(pair[0])
            if run and key != previous:
                entries.append(run);run = []
            run.append(pair);previous = key
        if run:
            entries.append(run)
    return entries, coop, raw


def member_steps(policy, sample):
    """(indices, targets) of the plan members present in the pair context."""
    js = bank_imitation.indices(sample)
    if not js:
        return [], None
    target = torch.as_tensor(sample['action'], dtype=torch.float32).reshape(-1, 1)
    return js, target[js]


@torch.no_grad()
def action_errors(policy, sample):
    """Per member-step L1 error |mean - plan action| of one pair (bank_imitation estimator)."""
    js, target = member_steps(policy, sample)
    if not js:
        return None
    mean, _ = policy(sample['context'])
    return (mean[js]-target).abs().reshape(-1)


@torch.no_grad()
def coop_error(policy, pairs):
    """Mean cooperative-pair L1 action error (m/s^2) over every member-step."""
    errors = [action_errors(policy, s) for s, _ in pairs]
    errors = [e for e in errors if e is not None]
    if not errors:
        return None
    return float(torch.cat(errors).mean())


@torch.no_grad()
def coop_fall_bootstrap(before, after, pairs, n_boot=N_BOOT, seed=BOOTSTRAP_SEED):
    """Pair-cluster bootstrap 95% CI of the mean per-member-step error fall (after - before).

    Negative = the imitation phase moved actions toward the plan. Clusters are pairs (the
    seconds of one committed plan share the entry); the seed belongs to the registered
    bootstrap estimator family. Returns (mean fall, [lo, hi] or None under 2 clusters, n)."""
    deltas = []
    for s, _ in pairs:
        e_before, e_after = action_errors(before, s), action_errors(after, s)
        if e_before is None:
            continue
        deltas.append(float((e_after-e_before).mean()))
    if not deltas:
        return None, None, 0
    mean = float(np.mean(deltas))
    if len(deltas) < 2:
        return mean, None, len(deltas)
    values = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    boots = values[rng.integers(0, len(values), (int(n_boot), len(values)))].mean(1)
    return mean, [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))], len(deltas)


@torch.no_grad()
def entry_toward_fractions(before, after, entries):
    """Per cooperative entry: fraction of CAV-steps whose action moved toward the plan action."""
    out = []
    for group in entries:
        toward = steps = 0
        plan_id, start_s = None, None
        for s, _ in group:
            plan_id, start_s = pair_plan_key(s)
            js, target = member_steps(before, s)
            if not js:
                continue
            mean_before, _ = before(s['context'])
            mean_after, _ = after(s['context'])
            e_before = (mean_before[js]-target).abs()
            e_after = (mean_after[js]-target).abs()
            toward += int((e_after < e_before).sum());steps += len(js)
        out.append(dict(plan_id=plan_id, start_s=start_s, pairs=len(group),
                        member_steps=steps, toward_steps=toward,
                        toward_fraction=toward/steps if steps else None))
    return out


def v2v_weight_report(policy):
    """Whether the zero-initialised v2v input columns moved (non-zero), per head.

    The imitation phase trains the actor only (registered bank_imitation loss), so the
    action-path actor input weights are the criterion; critic movement is reported but
    not required."""
    extra = policy.v2v_width()
    heads = {}
    for name, layer in (('actor', policy.actor[0]), ('critic', policy.critic[0])):
        block = layer.weight.detach()[:, FEATURES:FEATURES+extra] if extra > 0 else layer.weight.detach()[:, :0]
        heads[name] = dict(nonzero=bool(extra > 0 and bool((block != 0).any())),
                           l2=float(block.norm()), max_abs=float(block.abs().max()) if extra > 0 else 0.)
    return dict(extra_dim=extra, heads=heads,
                moved=bool(heads['actor']['nonzero']),
                rule='actor input columns non-zero (the imitation phase trains the actor only)')


def intent_columns(policy, horizon=None):
    """Per-partner partner-intent input columns of the widened actor, or None.

    For an intent block (extra == K*(6+H+1), v2v_obs.is_intent_block_width) the partner
    slot's planned-command channels sit at slot*(6+H+1)+6 .. +6+H inside the v2v block;
    returns the FEATURES-offset column index list. None for a plain six-feature v2v block
    (option off) so callers can skip the report."""
    extra = policy.v2v_width()
    from .v2v_obs import is_intent_block_width
    if extra <= 0 or not is_intent_block_width(extra):
        return None
    h = int(horizon) if horizon is not None else v2v_model.V2V_INTENT_H
    k, stride = v2v_model.V2V_INTENT_K, v2v_model.V2V_INTENT_BASE+int(h)+1
    if extra != k*stride:
        raise ValueError('intent block width mismatch: extra=%d, expected K*(6+H+1)=%d' % (extra, k*stride))
    return [FEATURES+slot*stride+v2v_model.V2V_INTENT_BASE+j
            for slot in range(k) for j in range(int(h))]


def intent_weight_report(policy, horizon=None):
    """Whether the PARTNER-INTENT input columns moved (non-zero), per head.

    The intent channels are the per-partner ``intent_0..intent_{H-1}`` values inside the
    ``K*(6+H+1)`` block (mask channel and the six plain partner features excluded)."""
    columns = intent_columns(policy, horizon)
    if columns is None:
        return None
    heads = {}
    for name, layer in (('actor', policy.actor[0]), ('critic', policy.critic[0])):
        block = layer.weight.detach()[:, columns]
        heads[name] = dict(nonzero=bool((block != 0).any()), l2=float(block.norm()),
                           max_abs=float(block.abs().max()))
    return dict(columns=len(columns), horizon=int(horizon if horizon is not None
                                                   else v2v_model.V2V_INTENT_H),
                heads=heads, moved=bool(heads['actor']['nonzero']),
                rule='actor partner-intent input columns non-zero (the imitation phase trains the actor only)')


@torch.no_grad()
def coop_intent_error(policy, pairs):
    """Masked mean absolute plan-future intent error (m/s^2) over the cooperative pairs.

    bank_imitation.intent_error estimator; None when no pair carries intent targets or
    the policy carries no intent head (option off)."""
    if not hasattr(policy, 'intent'):
        return None
    return bank_imitation.intent_error(policy, pairs)


# ---------------------------------------------------------------- the gate itself

def normalize_options(options=None):
    source = dict(options or {})
    bank_filter = source.get('bank_filter')
    if bank_filter not in (None, False, 'statistical'):
        raise ValueError("bank_filter must be None/False or the registered value 'statistical'")
    imitation_trust = source.get('imitation_trust')
    if imitation_trust not in (None, False, 'own_pass'):
        raise ValueError("imitation_trust must be None/False or the registered value 'own_pass'")
    v2v_intent = source.get('v2v_intent')
    if v2v_intent not in (None, False, True):
        raise ValueError('v2v_intent must be None/False or True')
    return dict(v2v=bool(source.get('v2v', False)), v2v_dim=source.get('v2v_dim'),
                passes=int(source.get('passes', 5)), coefficient=float(source.get('coefficient', 1.)),
                bank_filter=bank_filter, imitation_trust=imitation_trust,
                v2v_intent=bool(v2v_intent),
                v2v_intent_horizon=int(source.get('v2v_intent_horizon') or v2v_model.V2V_INTENT_H))


def run_gate(checkpoint, pairs, summary=None, options=None):
    """One offline imitation phase and the COOPFIX verdict (see module docstring)."""
    opts = normalize_options(options)
    filter_stage = filter_report(opts, summary)
    extra = v2v_model.v2v_extra({'v2v': opts['v2v']}, opts['v2v_dim'])
    intent_phase = None
    if opts['v2v_intent']:
        intent_phase = dict(weight=v2v_model.DEFAULT_INTENT_WEIGHT,
                            horizon=opts['v2v_intent_horizon'], bounds=v2v_model.INTENT_BOUNDS)
    policy, optimizer, generator, payload = load_gate_policy(
        checkpoint, extra, intent=opts['v2v_intent'], intent_horizon=opts['v2v_intent_horizon'])
    reference = copy.deepcopy(policy)
    kwargs, trust_stage = trust_kwargs(opts, pairs, summary)
    run_report = bank_imitation.run(policy, optimizer, pairs, generator, reference,
                                    coefficient=opts['coefficient'], passes=opts['passes'],
                                    **({'intent': intent_phase} if intent_phase else {}), **kwargs)
    entries, coop, raw = cooperative_entries(pairs, summary)
    fall_mean, fall_ci, fall_clusters = coop_fall_bootstrap(reference, policy, coop)
    with_v2v = sum(1 for s, _ in pairs if any(
        any(bool(x) for x in row) for row in ((s.get('context', {}).get('v2v') or {}).values())))
    report = dict(
        gate='coopfix', checkpoint=str(checkpoint), update=payload.get('update'),
        options=dict(opts, status=option_status()),
        bank=dict(pairs=len(pairs), cooperative_pairs=len(coop), cooperative_raw=raw,
                  entries=len(entries),
                  pairs_with_v2v_features=with_v2v,
                  v2v_feature_fraction=with_v2v/len(pairs) if pairs else None,
                  counts=(summary or {}).get('counts'), exposure=(summary or {}).get('exposure')),
        filter=filter_stage, trust=trust_stage,
        imitation=dict(passes_requested=run_report['requested_epochs'],
                       passes_completed=run_report['epochs_completed'],
                       optimizer_steps=run_report['optimizer_steps'],
                       pairs_seen=run_report['pairs_seen'],
                       kl=run_report['kl'], kl_limit=run_report['kl_limit'],
                       stop_reason=run_report.get('stop_reason')),
        cooperative=dict(error_before_mps2=coop_error(reference, coop),
                         error_after_mps2=coop_error(policy, coop),
                         fall_mean_mps2=fall_mean, fall_ci95=fall_ci, fall_clusters=fall_clusters,
                         entries=entry_toward_fractions(reference, policy, entries)),
        v2v_weights=v2v_weight_report(policy))
    if opts['v2v_intent']:
        report['intent'] = dict(enabled=True, horizon=opts['v2v_intent_horizon'],
                                weight=v2v_model.DEFAULT_INTENT_WEIGHT,
                                error_before_mps2=coop_intent_error(reference, coop),
                                error_after_mps2=coop_intent_error(policy, coop),
                                estimator='bank_imitation.intent_error: masked mean absolute '
                                          'plan-future command error (m/s^2) on the cooperative pairs',
                                pairs_with_intent_targets=sum(
                                    1 for s, _ in coop if (s.get('context', {}) or {}).get('intent_target')),
                                runner_report=run_report.get('intent'))
        report['v2v_weights']['partner_intent'] = intent_weight_report(policy, opts['v2v_intent_horizon'])
    report['verdict'] = verdict(report)
    return report


def verdict(report):
    """PASS iff: >=1 full pass, no registered KL stop and final KL within the runner limit,
    the cooperative error falls with the bootstrap CI entirely below 0, and the v2v input
    weights moved (non-zero). With ``v2v_intent`` on: the partner-intent error falls and
    the partner-intent input weights moved. Every failure is disclosed with its reason."""
    imitation, coop, weights = report['imitation'], report['cooperative'], report['v2v_weights']
    checks = dict(
        full_pass=imitation['passes_completed'] >= 1,
        kl_within_limit=imitation['kl'] <= imitation['kl_limit'] and not str(
            imitation.get('stop_reason', '')).startswith(('pre-imitation', 'imitation minibatch', 'full-bank')),
        cooperative_error_falls=bool(coop['fall_ci95']) and coop['fall_ci95'][1] < 0.,
        v2v_weights_moved=bool(weights['moved']))
    intent = report.get('intent')
    if intent is not None:
        checks['intent_error_falls'] = bool(intent['error_before_mps2'] is not None
            and intent['error_after_mps2'] is not None
            and intent['error_after_mps2'] < intent['error_before_mps2'])
        partner = (report['v2v_weights'].get('partner_intent') or {})
        checks['partner_intent_weights_moved'] = bool(partner.get('moved'))
    report['checks'] = checks
    reasons = [name for name, ok in checks.items() if not ok]
    report['reasons'] = reasons
    return 'PASS' if not reasons else 'FAIL'


# ---------------------------------------------------------------- inputs and CLI

def bank_inputs(bank_root, arm, training_seed, beta=1., max_decisions=None, v2v=True,
                v2v_cfg=None, bank_filter=None, pair_flags=False, intent=False,
                intent_horizon=v2v_model.V2V_INTENT_H):
    """(pairs, summary) from the live B-REPLAY bank (CPU-only read of recorded receipts);
    bank_filter='statistical' (BANKFILT READY) is applied per entry inside bank_pairs,
    pair_flags=True (IMITFIX READY) adds the summary['pair_cooperative'] imitation flags,
    intent=True (v2v_intent, DEFAULT OFF) adds the pairs' plan-future intent targets."""
    from . import replay_bank
    return replay_bank.bank_pairs(Path(bank_root), arm, int(training_seed), beta=float(beta),
                                  max_decisions=max_decisions, v2v=v2v, v2v_cfg=v2v_cfg,
                                  bank_filter=bank_filter, pair_flags=pair_flags,
                                  intent=bool(intent), intent_horizon=int(intent_horizon))


def pairs_file(path):
    payload = json.loads(Path(path).read_text())
    return [(sample, float(w)) for sample, w in payload['pairs']], payload.get('summary')


def main(argv=None):
    """python -m tools.coopmwm.coopfix_gate run --checkpoint CKPT --pairs-file P.json --v2v
    (offline) or --bank-root ROOT --arm B --training-seed N (live bank); writes the report."""
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('run', 'status'))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--pairs-file', type=Path)
    p.add_argument('--bank-root')
    p.add_argument('--arm', default='B', choices=('B', 'R'))
    p.add_argument('--training-seed', type=int, default=0)
    p.add_argument('--beta', type=float, default=1.)
    p.add_argument('--max-decisions', type=int)
    p.add_argument('--v2v', action='store_true')
    p.add_argument('--v2v-dim', type=int)
    p.add_argument('--bank-filter', dest='bank_filter', action='store_true',
                   help="apply the registered statistical bank filter (requires --bank-root)")
    p.add_argument('--imitation-trust', dest='imitation_trust', action='store_true',
                   help="run the imitation phase with the registered own_pass trust region")
    p.add_argument('--v2v-intent', dest='v2v_intent', action='store_true',
                   help='train and report the plan-future intent head (option v2v_intent, '
                        'requires pairs carrying context["intent_target"])')
    p.add_argument('--v2v-intent-horizon', dest='v2v_intent_horizon', type=int,
                   default=v2v_model.V2V_INTENT_H)
    p.add_argument('--passes', type=int, default=5)
    p.add_argument('--coefficient', type=float, default=1.)
    p.add_argument('--out', type=Path, default=Path('COOPFIX_GATE_REPORT.json'))
    a = p.parse_args(argv)
    if a.cmd == 'status':
        return option_status()
    if not a.checkpoint:
        p.error('--checkpoint is required for run')
    options = dict(v2v=a.v2v, v2v_dim=a.v2v_dim, passes=a.passes, coefficient=a.coefficient,
                   bank_filter='statistical' if a.bank_filter else None,
                   imitation_trust='own_pass' if a.imitation_trust else None,
                   v2v_intent=bool(a.v2v_intent), v2v_intent_horizon=a.v2v_intent_horizon)
    if a.pairs_file:
        if a.bank_filter:
            p.error('--bank-filter requires --bank-root (the rule needs the recorded receipts and traces)')
        pairs, summary = pairs_file(a.pairs_file)
    elif a.bank_root:
        pairs, summary = bank_inputs(a.bank_root, a.arm, a.training_seed, a.beta,
                                     a.max_decisions, v2v=a.v2v, bank_filter=options['bank_filter'],
                                     pair_flags=bool(options['imitation_trust']),
                                     intent=bool(options['v2v_intent']),
                                     intent_horizon=options['v2v_intent_horizon'])
    else:
        p.error('one of --pairs-file or --bank-root is required for run')
    report = run_gate(a.checkpoint, pairs, summary, options)
    a.out.write_text(json.dumps(report, indent=1, sort_keys=True, default=float))
    return report


if __name__ == '__main__':
    main()
