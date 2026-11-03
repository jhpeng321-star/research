"""COOPFIX EXACT GATE: one fixed update from MAPPO u0 (v2v+intent), then the isolated
policy-synergy check on the certified cooperative bank states.

OFFLINE (CPU only): load MAPPO u0 extended with the v2v+intent inputs
(v2v_model.load_checkpoint, zero-initialised at v2v_obs.V2V_DIM), build the STATISTICALLY
FILTERED bank (replay_bank.bank_pairs under the bank v2 rule, bank_statistics.view -
transient process-local adapter, restored immediately), and run ONE fixed update = one
registered full-pass imitation run (bank_imitation.run, >=5 passes, KL bound imported from
the runner report - never re-stated). Offline gate fields: passes, final KL, cooperative-pair
error change (paired per-pair cluster bootstrap 95% CI), intent-pair error change, and
partner-intent input weight movement (the zero-initialised v2v columns of actor/critic).

EXACT (planned here, executed by the live agent on cpuQ, consumed here): for every
cooperative candidate certified by the bank v2 rule (mandatory states 986066, 986048,
986003) the ISOLATED policy-synergy branches (policy_synergy.specs_for, isolated=True:
group-only NEW / one SOLO NEW per member / background-only NEW, all other CAVs u3, captured
t0 action retained, intervention t0+1) PLUS the SECOND-SEED solo branches
(replay_bank.replicate salt), so every solo delta is replicated on 2 seeds. Per state the
gate reports: group-only gain, best solo, synergy, per-vehicle paired bootstrap 95% CI over
the two paired seeds (plus the registered two-seed Student-t CI, bank_statistics
.replicated_ci), and the background-only delta. Verdict PASS iff the offline checks pass AND
the group-only synergy 95% CI lower bound > 0 on the 6-CAV state 986066.

Registered budget caps (isolated diagnostic registration, MAIN2026-11-03 12:26, imported as
constants - never behavioural): per state n<=6 group members, window end <= 1200 s,
(n+2) isolated branches <= 9600 sim-s; the second-seed solo cost is quoted separately.
Job/receipt formats are the existing policy_synergy sealed specs, QUOTE.json layout and
sumo_verify.run_branch graded receipts; the executor is the existing
policy_synergy.BranchRunner behind bank_v2v.enabled(v2v). No new SUMO wrapper is defined.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import bank_imitation, bank_statistics, bank_v2v, policy_synergy as ps, sumo_verify as sv
from . import replay_bank as rb, v2v_integrate as vi, v2v_model
from .exact_grade import _seal, _write_new, verify_sealed
from .protocol import digest, sha
from .v2v_model import FEATURES

GATE = 'coopfix_exact'
GATE_UPDATE = 0
MANDATORY_STATES = (986066, 986048, 986003)
GATE_STATE = 986066                    # the 6-CAV state the verdict is bound to
MAX_GROUP = 6                          # registered isolated diagnostic caps (MAIN 12:26)
MAX_WINDOW_END_S = 1200
ISOLATED_STATE_CAP_SIM_S = 9600
BOOTSTRAP_SEED = 20261101              # replay_bank._boot_ci estimator family seed
N_BOOT = 10000
PLAN_FILE = 'COOPFIX_EXACT_PLAN.json'
GATE_FILE = 'COOPFIX_GATE.json'


# ---------------------------------------------------------------- offline fixed update

def coop_pairs(pairs, summary=None):
    """(unique cooperative pairs, raw count): structural/definitional rule (single-CAV plans
    are never cooperative), deduplicated through the bank exposure record like the bank
    builder flattens it; without a summary every cooperative pair is kept."""
    coop = [(s, w) for s, w in pairs
            if len(s.get('behaviour', {}).get('proposal_members', [])) > 1]
    raw = None
    if summary and (summary.get('exposure') or {}).get('cooperative_samples_raw') is not None:
        raw = int(summary['exposure']['cooperative_samples_raw'])
        if 0 < raw <= len(coop):
            coop = coop[:raw]
    return coop, raw


def _present_bits(sample):
    """True when any partner presence bit is set in the sample's v2v blocks (partner-major
    blocks, presence last of each 6-wide partner row, v2v_obs layout)."""
    blocks = (sample.get('context') or {}).get('v2v') or {}
    for row in blocks.values():
        for slot in range(len(row) // 6):
            if row[slot * 6 + 5]:
                return True
    return False


def intent_pairs(pairs):
    """Pairs whose context actually carries partner-intent features (a present partner)."""
    return [(s, w) for s, w in pairs if _present_bits(s)]


def widen_pairs(pairs, width):
    """Append each member's v2v block to the context x rows (v2v_integrate.widen_context).
    Samples without a v2v payload get zero blocks at the extended width (the observation
    contract zero-pads absent partners), so every pair reaches the widened model."""
    out = []
    for s, w in pairs:
        blocks = s['context'].get('v2v') or {}
        if width > 0 and not blocks:
            blocks = {vid: [0.]*int(width) for vid in s['context'].get('members') or []}
        out.append((dict(s, context=vi.widen_context(s['context'], blocks)), w))
    return out


@torch.no_grad()
def per_pair_errors(policy, pairs):
    """Mean member-step L1 action error (m/s^2) of each pair (the bank_imitation estimator)."""
    out = []
    for sample, _w in pairs:
        js = bank_imitation.indices(sample)
        if not js:
            continue
        mean, _ = policy(sample['context'])
        target = torch.as_tensor(sample['action'], dtype=torch.float32).reshape(-1, 1)
        out.append(float((mean[js] - target[js]).abs().mean()))
    return out


def paired_fall_ci(before, after, n_boot=N_BOOT, seed=BOOTSTRAP_SEED):
    """Paired cluster bootstrap 95% CI of the mean per-pair error change (after - before);
    negative = the fixed update moved actions toward the plan. None CI under 2 pairs."""
    deltas = [float(a) - float(b) for b, a in zip(before, after)]
    if not deltas:
        return None, None, 0
    mean = float(np.mean(deltas))
    if len(deltas) < 2:
        return mean, None, len(deltas)
    values = np.asarray(deltas)
    rng = np.random.default_rng(seed)
    boots = values[rng.integers(0, len(values), (int(n_boot), len(values)))].mean(1)
    return mean, [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))], len(deltas)


def weight_report(policy, extra):
    """Whether the zero-initialised partner-intent input columns moved (non-zero) per head."""
    heads = {}
    for name in ('actor', 'critic'):
        layer = getattr(policy, name)[0]
        block = layer.weight.detach()[:, FEATURES:FEATURES + extra] if extra > 0 \
            else layer.weight.detach()[:, :0]
        heads[name] = dict(nonzero=bool(extra > 0 and bool((block != 0).any())),
                           l2=float(block.norm()),
                           max_abs=float(block.abs().max()) if extra > 0 else 0.)
    return dict(extra_dim=int(extra), heads=heads,
                moved=any(h['nonzero'] for h in heads.values()))


def fixed_update(checkpoint, pairs, summary=None, passes=5, coefficient=1., v2v=True,
                 v2v_dim=None):
    """ONE fixed update: one registered full-pass imitation run of the extended u0 on the
    bank pairs; returns (post-update policy state dict, offline gate fields)."""
    extra = v2v_model.v2v_extra({'v2v': v2v}, v2v_dim)
    policy, optimizer, generator, payload = v2v_model.load_checkpoint(checkpoint, v2v_dim=extra)
    reference = copy.deepcopy(policy)
    if extra > 0:
        pairs = widen_pairs(pairs, extra)
        policy, reference = vi.WidenedPolicy(policy), vi.WidenedPolicy(reference)
    report = bank_imitation.run(policy, optimizer, pairs, generator, reference,
                                coefficient=coefficient, passes=passes)
    module = policy.model if extra > 0 else policy
    coop, raw = coop_pairs(pairs, summary)
    intent = intent_pairs(pairs)
    coop_before, coop_after = per_pair_errors(reference, coop), per_pair_errors(policy, coop)
    intent_before, intent_after = per_pair_errors(reference, intent), per_pair_errors(policy, intent)
    fall_mean, fall_ci, fall_clusters = paired_fall_ci(coop_before, coop_after)
    intent_mean, intent_ci, intent_clusters = paired_fall_ci(intent_before, intent_after)
    offline = dict(
        gate=GATE, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        update=payload.get('update'), v2v=v2v, v2v_dim=extra,
        passes=dict(requested=report['requested_epochs'], completed=report['epochs_completed'],
                    optimizer_steps=report['optimizer_steps'], pairs_seen=report['pairs_seen'],
                    stop_reason=report.get('stop_reason')),
        kl=report['kl'], kl_limit=report['kl_limit'], kl_reference=report['kl_reference'],
        rule=report['rule'],
        cooperative_pairs=dict(count=len(coop), raw=raw,
                               error_before_mps2=float(np.mean(coop_before)) if coop_before else None,
                               error_after_mps2=float(np.mean(coop_after)) if coop_after else None,
                               change_mean_mps2=fall_mean, change_ci95=fall_ci,
                               clusters=fall_clusters),
        intent_pairs=dict(count=len(intent),
                          error_before_mps2=float(np.mean(intent_before)) if intent_before else None,
                          error_after_mps2=float(np.mean(intent_after)) if intent_after else None,
                          change_mean_mps2=intent_mean, change_ci95=intent_ci,
                          clusters=intent_clusters),
        partner_intent_weights=weight_report(module, extra))
    offline['checks'] = offline_checks(offline)
    return module.state_dict(), offline


def offline_checks(offline):
    """Registered offline verdict checks (coopfix gate family): >=1 full pass inside the
    runner's KL bound, the cooperative-pair error falls with its CI entirely below 0, and
    the partner-intent input weights moved (the policy can use partner intent)."""
    coop, weights = offline['cooperative_pairs'], offline['partner_intent_weights']
    return dict(
        full_pass=offline['passes']['completed'] >= 1,
        kl_within_limit=offline['kl'] <= offline['kl_limit']
        and offline['passes']['stop_reason'] == 'all registered imitation epochs complete',
        cooperative_error_falls=bool(coop['change_ci95']) and coop['change_ci95'][1] < 0.,
        partner_intent_weights_moved=bool(weights['moved']))


def statistical_bank_pairs(root, arm, training_seed, **kw):
    """bank_pairs under the bank v2 rule (bank_statistics.view): the transient process-local
    adapter is installed for the build and restored immediately afterwards."""
    original = rb.confirmations
    bank_statistics.install(rb)
    try:
        return rb.bank_pairs(root, arm, int(training_seed), v2v=True, **kw)
    finally:
        rb.confirmations = original


# ---------------------------------------------------------------- candidate states (bank v2 rule)

def certified_candidates(root, arm, sources=()):
    """Admitted positive decisions whose confirmation is COOPERATIVE under the bank v2 rule
    (bank_statistics.view: replicated Student-t synergy CI lower > 0), oldest first."""
    decisions = rb.bank_decisions(root, arm, sources=sources)
    confirmed = rb.confirmations(root, arm)
    out = []
    for r in decisions:
        c = confirmed.get(r['verification_id'])
        if r['status'] != 'accepted' or not (r['grade_veh_s'] or 0) > 0 or c is None:
            continue
        v = bank_statistics.view(c)
        co = v.get('cooperation') or {}
        if v['admitted'] and co.get('cooperative') is True:
            out.append((r, dict(c, statistical_view=v)))
    return out


def _quote(vid, specs, phase='bank_verification_ext'):
    return dict(verification_id=vid, branch_ids=[s['verification_id'] for s in specs],
                sim_s=sum(s['sim_s'] for s in specs), phase=phase)


# ---------------------------------------------------------------- job planning

def plan(root, arm, control, new_state, old_state, offline=None, training_seed=0,
         pairs_source=None, update=GATE_UPDATE, sources=(), v2v=True, v2v_dim=None):
    """Plan every SUMO job of the gate (sealed existing-format specs + snapshots + sim-s
    quotes) after the offline fixed update; writes COOPFIX_EXACT_PLAN.json atomically."""
    root, control = Path(root), Path(control)
    if (control/PLAN_FILE).exists():
        return load_plan(control)
    if offline is None:
        pairs, summary = statistical_bank_pairs(root, arm, training_seed) \
            if pairs_source is None else _pairs_file(pairs_source)
        _new, offline = fixed_update(u0_path(root, arm), pairs, summary, v2v=v2v, v2v_dim=v2v_dim)
    candidates = certified_candidates(root, arm, sources=sources)
    seeds = {r['episode']['world']['seed']: (r, c) for r, c in candidates}
    missing = [s for s in MANDATORY_STATES if s not in seeds]
    if missing:
        raise ValueError(f'bank v2 rule has not certified mandatory cooperative states {missing}; '
                         f'certified: {sorted(seeds)}')
    links = [json.loads(p.read_text()) for p in (rb.bank_root(root, arm)/'decisions').glob('*.json')]
    by_vid = {link['verification_id']: link for link in links}
    states, snapshots = [], {}
    for seed in sorted(seeds):
        r, c = seeds[seed]
        group = list(r['chosen_plan']['members'])
        end_s = int(r['window']['end_s'])
        if not 2 <= len(group) <= MAX_GROUP:
            raise ValueError(f'state {seed}: group size {len(group)} outside the registered '
                             f'isolated diagnostic range 2..{MAX_GROUP}')
        if end_s > MAX_WINDOW_END_S:
            raise ValueError(f'state {seed}: window end {end_s} > registered {MAX_WINDOW_END_S}')
        snap = rb._decision_snapshot(root, arm, r, by_vid)
        isolated = ps.specs_for(snap, r, update, arm, new_state, old_state, isolated=True)
        iso_cost = sum(s['sim_s'] for s, _ in isolated)
        if iso_cost > ISOLATED_STATE_CAP_SIM_S:
            raise ValueError(f'state {seed}: isolated cost {iso_cost} sim-s exceeds the '
                             f'registered per-state cap {ISOLATED_STATE_CAP_SIM_S}')
        receipt2 = dict(r, verification_id=r['verification_id'] + '_seed2')
        snap2 = rb.replicate(snap, f'{GATE}:{receipt2["verification_id"]}')
        seed2 = [(s, sn) for s, sn in ps.specs_for(snap2, receipt2, update, arm, new_state,
                                                   old_state, isolated=True)
                 if s['branch'].startswith('solo')]
        for spec, sn in isolated:
            snapshots[spec['verification_id']] = sn
        for spec, sn in seed2:
            snapshots[spec['verification_id']] = sn
        state = dict(seed=int(seed), verification_id=r['verification_id'], group=group,
                     window=dict(r['window']), j_ref_veh_s=float(r['branches'][0]['J_veh_s']),
                     grade_veh_s=float(r['grade_veh_s']),
                     plan_cooperation_gain_veh_s=(c.get('cooperation') or {})
                     .get('cooperation_gain_veh_s'),
                     statistical_reason=(c.get('cooperation') or {}).get('statistical_reason'),
                     isolated_quote=_quote(r['verification_id'], [s for s, _ in isolated]),
                     seed2_solo_quote=_quote(receipt2['verification_id'], [s for s, _ in seed2]),
                     jobs=[_job(s, 1, sn) for s, sn in isolated]
                          + [_job(s, 2, sn) for s, sn in seed2])
        states.append(state)
    plan_record = dict(gate=GATE, arm=arm, update=update, unix=time.time(),
                       new_state_sha256=sv._state_sha(new_state),
                       old_state_sha256=sv._state_sha(old_state),
                       rule='group-only gain, best solo, synergy = group-only gain - best solo; '
                            'per-vehicle paired bootstrap 95% CI over the two paired seeds; '
                            'PASS iff offline checks pass AND the synergy CI lower bound > 0 '
                            f'on state {GATE_STATE}',
                       caps=dict(max_group=MAX_GROUP, window_end_s=MAX_WINDOW_END_S,
                                 isolated_state_cap_sim_s=ISOLATED_STATE_CAP_SIM_S,
                                 source='isolated diagnostic registration MAIN2026-11-03 12:26'),
                       mandatory_states=list(MANDATORY_STATES), offline=offline, states=states,
                       totals=dict(states=len(states), jobs=sum(len(s['jobs']) for s in states),
                                   isolated_sim_s=sum(s['isolated_quote']['sim_s'] for s in states),
                                   seed2_solo_sim_s=sum(s['seed2_solo_quote']['sim_s'] for s in states),
                                   total_sim_s=sum(j['sim_s'] for s in states for j in s['jobs'])))
    (control/'jobs').mkdir(parents=True, exist_ok=True)
    for vid, snap in snapshots.items():                       # executor inputs (cpuQ runs these)
        with (control/'jobs'/f'{vid}.SNAPSHOT.pt').open('xb') as f:
            torch.save(snap, f)
    for state in states:
        for job in state['jobs']:
            spec = job.pop('spec')
            _write_new(control/'jobs'/f'{job["verification_id"]}.{job["branch"]}.spec.json', spec)
    _write_new(control/PLAN_FILE, _seal(plan_record, 'gate_plan_sha256'))
    return load_plan(control)


def u0_path(root, arm):
    path = Path(root)/f'{arm}_checkpoint_000.pt'
    if not path.exists():
        raise ValueError(f'MAPPO u0 checkpoint missing: {path}')
    return path


def _job(spec, seed, snapshot):
    return dict(branch=spec['branch'], verification_id=spec['verification_id'], seed=int(seed),
                sim_s=spec['sim_s'], snapshot_sha256=sv.snapshot_sha(snapshot),
                spec_sha256=spec['branch_sha256'], spec=spec)


def load_plan(control):
    path = Path(control)/PLAN_FILE
    if not path.exists():
        raise ValueError(f'no {PLAN_FILE} under {control}; run the plan stage first')
    plan_record = json.loads(path.read_text())
    verify_sealed(plan_record, 'gate_plan_sha256')
    return plan_record


def _pairs_file(path):
    payload = json.loads(Path(path).read_text())
    return [(sample, float(w)) for sample, w in payload['pairs']], payload.get('summary')


# ---------------------------------------------------------------- execution (cpuQ; never on this laptop)

def execute(control):
    """Run every planned branch through the EXISTING policy_synergy.BranchRunner behind
    bank_v2v.enabled(v2v), one branch at a time, and write the run_branch graded receipts.
    No retry: a failed/stopped branch is recorded and never imputed."""
    plan_record = load_plan(control)
    control = Path(control)
    runner = ps.BranchRunner(control/'branches', workers=1)
    quotes = []
    for state in plan_record['states']:
        specs = []
        for job in state['jobs']:
            spec = json.loads((control/'jobs'/f'{job["verification_id"]}.{job["branch"]}.spec.json')
                              .read_text())
            verify_sealed(spec, 'branch_sha256')
            if spec['branch_sha256'] != job['spec_sha256']:
                raise ValueError('planned spec differs from its sealed job record')
            snap = torch.load(control/'jobs'/f'{job["verification_id"]}.SNAPSHOT.pt',
                              weights_only=False)
            specs.append(spec)
            runner.prepare([spec], snap)
            with bank_v2v.enabled(plan_record['offline']['v2v']):
                receipt = sv.run_branch(spec, snap, runner)
            _write_new(control/'branches'/spec['verification_id']/
                       f'{spec["branch"]}.graded.json', receipt)
        quotes.append(_quote(state['verification_id'], specs))
    return quotes


# ---------------------------------------------------------------- consumption and verdict

class StoredRunner:
    """Regrades a stored raw branch result (BranchRunner-layout .result.json) through
    sumo_verify.run_branch without any SUMO."""

    def __init__(self, control):
        self.control = Path(control)

    def run(self, spec, snapshot):
        path = self.control/'branches'/spec['verification_id']/f'{spec["branch"]}.result.json'
        return json.loads(path.read_text())


def _receipt(control, job):
    """The graded receipt of one planned job: the executor's .graded.json, or a raw
    .result.json regraded through sumo_verify.run_branch (prefix check included)."""
    base = Path(control)/'branches'/job['verification_id']
    graded = base/f'{job["branch"]}.graded.json'
    if graded.exists():
        return json.loads(graded.read_text())
    spec = json.loads((Path(control)/'jobs'/f'{job["verification_id"]}.{job["branch"]}.spec.json')
                      .read_text())
    verify_sealed(spec, 'branch_sha256')
    snap = torch.load(Path(control)/'jobs'/f'{job["verification_id"]}.SNAPSHOT.pt',
                      weights_only=False)
    return sv.run_branch(spec, snap, StoredRunner(control))


def paired_seed_ci(values, student_t=False, n_boot=N_BOOT, seed=BOOTSTRAP_SEED):
    """Paired bootstrap 95% CI of the mean over the paired seeds (and, disclosed, the
    registered two-seed Student-t CI, bank_statistics.replicated_ci)."""
    mean = float(np.mean(values))
    boots = None
    if len(values) >= 2:
        v = np.asarray(values, float)
        rng = np.random.default_rng(seed)
        boots = v[rng.integers(0, len(v), (int(n_boot), len(v)))].mean(1)
    student = bank_statistics.replicated_ci(*values) if student_t and len(values) == 2 else None
    return dict(mean=mean,
                bootstrap_ci95=([float(np.quantile(boots, .025)), float(np.quantile(boots, .975))]
                                if boots is not None else None),
                student_t_ci95=student, seeds=len(values))


def state_metrics(state, receipts):
    """Per state: group-only gain, best solo, synergy, per-vehicle paired bootstrap 95% CI,
    background-only delta. J is vehicle-seconds (lower is better); every delta is
    J_u3_reference - J_branch, so positive means the NEW policy improved."""
    j_ref = state['j_ref_veh_s']
    members = list(state['group'])
    seeds = sorted({j['seed'] for j in state['jobs'] if j['branch'].startswith('solo')})
    all_receipt = receipts.get((1, 'all'))
    out = dict(seed=state['seed'], verification_id=state['verification_id'], group=members,
               j_ref_veh_s=j_ref, window=state['window'])
    if all_receipt is None or all_receipt['status'] != 'graded':
        out.update(status='incomplete', group_only_gain_veh_s=None, complete=False,
                   reason='the group-only branch is ungraded; never imputed')
        return out
    group_only_gain = j_ref - float(all_receipt['J_veh_s'])
    per_vehicle = {}
    for i, member in enumerate(members):
        deltas, js = [], {}
        for seed in seeds:
            receipt = receipts.get((seed, f'solo{i}'))
            if receipt is None or receipt['status'] != 'graded':
                continue
            js[seed] = float(receipt['J_veh_s'])
            deltas.append(j_ref - js[seed])
        ci = paired_seed_ci(deltas, student_t=True) if deltas else \
            dict(mean=None, bootstrap_ci95=None, student_t_ci95=None, seeds=0)
        per_vehicle[member] = dict(delta_veh_s=ci['mean'], ci95=ci['bootstrap_ci95'],
                                   student_t_ci95=ci['student_t_ci95'], seeds_graded=len(deltas),
                                   J_veh_s=js)
    graded = {key: value for key, value in receipts.items() if value['status'] == 'graded'}
    per_seed = {seed: {i: j_ref - float(graded[(seed, f'solo{i}')]['J_veh_s'])
                       for i in range(len(members)) if (seed, f'solo{i}') in graded}
                for seed in seeds}
    best_member, best_solo = None, None
    graded_means = {m: per_vehicle[m]['delta_veh_s'] for m in members
                    if per_vehicle[m]['seeds_graded']}
    if graded_means:
        best_member = max(graded_means, key=lambda m: graded_means[m])
        best_solo = float(graded_means[best_member])
    synergy_seeds = [group_only_gain - max(per_seed[seed].values()) for seed in seeds
                     if len(per_seed[seed]) == len(members)]
    synergy = paired_seed_ci(synergy_seeds, student_t=True) if synergy_seeds else \
        dict(mean=None, bootstrap_ci95=None, student_t_ci95=None, seeds=0)
    background = receipts.get((1, 'background'))
    out.update(
        group_only_gain_veh_s=group_only_gain,
        best_solo_member=best_member, best_solo_gain_veh_s=best_solo,
        synergy_veh_s=synergy['mean'], synergy_ci95=synergy['bootstrap_ci95'],
        synergy_student_t_ci95=synergy['student_t_ci95'], per_vehicle=per_vehicle,
        background_only_delta_veh_s=(j_ref - float(background['J_veh_s'])
                                     if background and background['status'] == 'graded' else None),
        background_status=(background or {}).get('status'),
        ungraded_branches=sorted(f'seed{seed}/{branch}' for seed, branch in receipts
                                 if receipts[(seed, branch)]['status'] != 'graded'))
    out['complete'] = not out['ungraded_branches'] and bool(synergy_seeds)
    out['status'] = 'graded' if out['complete'] else 'incomplete'
    return out


def collect(control, plan_record=None):
    """Consume every planned job output, compute the per-state synergy metrics and write
    COOPFIX_GATE.json (offline fields from the sealed plan; verdict per the gate rule)."""
    control = Path(control)
    plan_record = plan_record or load_plan(control)
    states = []
    for state in plan_record['states']:
        receipts = {(j['seed'], j['branch']): _receipt(control, j) for j in state['jobs']}
        states.append(state_metrics(state, receipts))
    report = dict(gate=GATE, arm=plan_record['arm'], update=plan_record['update'],
                  unix=time.time(), plan_sha256=digest(plan_record),
                  offline=plan_record['offline'], offline_checks=plan_record['offline']['checks'],
                  states=states, plan_totals=plan_record['totals'])
    report['verdict'] = verdict(report)
    _write_new(control/GATE_FILE, _seal(report, 'gate_report_sha256'))
    return report


def verdict(report):
    """PASS iff the offline checks pass AND the group-only synergy 95% CI lower bound > 0 on
    the 6-CAV state 986066 (which requires its two-seed solo replication to be graded)."""
    states = {s['seed']: s for s in report['states']}
    gate_state = states.get(GATE_STATE)
    lower = None if not (gate_state and gate_state.get('synergy_ci95')) \
        else gate_state['synergy_ci95'][0]
    checks = dict(offline_checks=all(report['offline_checks'].values()),
                  gate_state_present=gate_state is not None,
                  gate_state_complete=bool(gate_state and gate_state['complete']),
                  gate_state_synergy_ci_lower_gt_0=bool(lower is not None and lower > 0.))
    report['gate_state_synergy_lower_bound'] = lower
    report['gate_checks'] = checks
    report['reasons'] = [name for name, ok in checks.items() if ok is not True]
    return 'PASS' if not report['reasons'] else 'FAIL'


# ---------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('plan', 'execute', 'collect', 'status'))
    p.add_argument('--root', help='live run root (bank, decisions, u0/u3 checkpoints)')
    p.add_argument('--arm', default='B', choices=('B', 'R'))
    p.add_argument('--control', required=True, help='gate control directory (plan, jobs, receipts)')
    p.add_argument('--u0', type=Path, help='MAPPO u0 checkpoint (default {root}/{arm}_checkpoint_000.pt)')
    p.add_argument('--u3', type=Path, help='frozen reference checkpoint (default {root}/{arm}_checkpoint_003.pt)')
    p.add_argument('--pairs-file', type=Path, help='offline bank pairs json (default: the live bank)')
    p.add_argument('--training-seed', type=int, default=0)
    p.add_argument('--passes', type=int, default=5)
    p.add_argument('--coefficient', type=float, default=1.)
    p.add_argument('--no-v2v', action='store_true')
    p.add_argument('--v2v-dim', type=int, default=None)
    a = p.parse_args(argv)
    if a.cmd == 'status':
        control = Path(a.control)
        out = {name: (json.loads((control/name).read_text())
                      if (control/name).exists() else None) for name in (PLAN_FILE, GATE_FILE)}
        out['planned_jobs'] = out[PLAN_FILE].get('totals', {}).get('jobs') if out[PLAN_FILE] else None
        out['graded_jobs'] = len(list((control/'branches').glob('*/*.graded.json'))) \
            if (control/'branches').exists() else 0
        return out
    if a.cmd == 'execute':
        return execute(a.control)
    if a.cmd == 'collect':
        return collect(a.control)
    root = Path(a.root)
    pairs, summary = (_pairs_file(a.pairs_file) if a.pairs_file
                      else statistical_bank_pairs(root, a.arm, a.training_seed))
    _new, offline = fixed_update(a.u0 or u0_path(root, a.arm), pairs, summary, passes=a.passes,
                                 coefficient=a.coefficient, v2v=not a.no_v2v, v2v_dim=a.v2v_dim)
    old_state = torch.load(a.u3 or Path(root)/f'{a.arm}_checkpoint_003.pt',
                           map_location='cpu', weights_only=False)['policy']
    return plan(root, a.arm, a.control, _new, old_state, offline=offline,
                training_seed=a.training_seed, pairs_source=a.pairs_file, v2v=not a.no_v2v,
                v2v_dim=a.v2v_dim)


if __name__ == '__main__':
    print(json.dumps(main(), indent=1, default=str))
