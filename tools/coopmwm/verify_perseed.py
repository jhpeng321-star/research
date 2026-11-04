"""verify_mode='all8_both_seeds' (owner 2026-11-03 ~22:05; DEFAULT OFF, legacy byte-identical when off).

Plans must be judged on the seed they are used on; cooperation can differ per seed and scenario.
Per completed first-seed decision (accepted OR rejected - a plan may win only on seed 2):

  seed 1 = the existing first-seed receipt (K candidates + 'policy continues', already run);
  seed 2 = ALL the same K candidates + 'policy continues' on rb.replicate(snapshot, vid)
           (the existing seed-2 CRN salt convention: salt = first-seed verification_id);
  per seed, EVERY candidate whose within-seed per-second paired bootstrap lower bound
  (bank_statistics._stats: policy-continues minus plan, B=2000, RNG 20261103, circular 30 s
  blocks or per-vehicle units) is > 0 AND whose composite MAIN04:26 safety verdict passes
  is admitted - not only the best; seeds select their winners independently;
  each seed additionally needs the later-harm check on its OWN CRN (one extended-window
  verification of all that seed's LB>0 candidates, tail 600 s, also that seed's safety window;
  MAIN 2026-11-03 launch-final: seed 2 is now measured for new verifications and a measured
  failure rejects; ``later_harm_seed2=False`` reproduces the earlier not-measured records);
  own-seed safety deltas are required (missing -> data_missing, never another seed's window);
  per admitted (plan, seed) with >= 2 members: ONE batched solo verification on THAT seed
  (one member follows the plan, the others policy-continue; one shared policy branch),
  within-seed cooperative label = best-solo minus group per-second bootstrap LB > 0.

OPTION ``extra_seeds`` (owner 2026-11-04 02:10; DEFAULT 0 = byte-identical records, calls and charges):
for every decision with at least one within-seed COOPERATIVE example (seeds 1/2, after their solos),
each such plan (distinct candidate index; one batched verification per extra seed holds all of them)
is also run - together with the policy-continues reference - on the additional CRN seeds
3..(2+extra_seeds). Salt derivation (``extra_seed_salt``): seed 2 keeps ``replicate(snapshot, vid)``;
seed k >= 3 uses ``replicate(snapshot, f'{vid}#seed{k}')`` - a deterministic function of the first-seed
verification_id and k only (distinct from vid, from each other and from every evaluation world seed;
no evaluation seed 964000-964199/962xxx/977xxx is ever an input). Each (plan, seed) is admitted on its
own with the SAME per-seed rules (within-seed paired-bootstrap LB > 0, own-seed composite safety, own-seed
extended-window later-harm), then gets its own batched solo verification on THAT seed; admitted examples
are appended to ``per_seed_v1.examples`` with ``seed_index >= 3`` (own plan/tape/label), so the per_seed_v1
bank rule consumes them unchanged. No cross-seed requirement; seeds >= 3 do not alter the seed-1/2
examples, the legacy ``replicated`` flag (cooperative on BOTH seeds 1 and 2) or the exact gate.
``mean_replicated_advantage_veh_s`` stays the seeds-1/2 admission mean; ``winners``/admitted/cooperative
seed lists simply gain the extra seed indices. Every extra verification (group, extended, solos) is
charged through the SAME hooks as seeds 1/2. Both arms (B and R) use the identical code path.

Every verification is charged through the existing hooks before physics: sv.verify ->
runner.admit (disk guard) -> rb._confirm_charge(run_root, phase, kind) (looked up at call time,
so precharged/builder overrides apply); warm_health.assert_running(root, arm) is checked before
each verification. Receipts are ordinary sealed sv receipts in per-kind roots; the decision
record (bank/confirmations_all8/<vid>.json, sealed 'confirmation_sha256') carries the
per_seed_v1 example list (one example per (candidate, seed), each with its own plan, tape and
seed index) so the per_seed_v1 bank rule (c4/bank-perseed) can consume it directly.
"""
import copy
import json
from pathlib import Path
import time

ALL8 = 'all8_both_seeds'
LEGACY = 'legacy'
VERIFY_MODES = (LEGACY, ALL8)
SCHEMA = 'coopmwm/verify-perseed/v1'
RECORDS = 'confirmations_all8'
SEED2_ROOT = 'confirm_seed2_all8'
EXT_ROOT = 'confirm_extended_all8'
EXT2_ROOT = 'confirm_extended_seed2_all8'
SOLO_ROOTS = {1: 'confirm_solo_all8_s1', 2: 'confirm_solo_all8_s2'}
KIND_SEED2, KIND_EXT = 'confirm_seed2_all8', 'confirm_extended_all8'
KIND_EXT2 = 'confirm_extended_seed2_all8'
EXT_ROOTS = {1: EXT_ROOT, 2: EXT2_ROOT}
KIND_EXTS = {1: KIND_EXT, 2: KIND_EXT2}
KIND_SOLO = {1: 'confirm_solo_all8_s1', 2: 'confirm_solo_all8_s2'}
FIRST_EXTRA_SEED = 3
EVAL_SEED_RANGES = ((962000, 962999), (964000, 964199), (977000, 977999))
# Evaluation world seeds (964000-964199, 962xxx, 977xxx): never an input or output of an
# extra-seed CRN salt (see extra_seed_salt).


def seed_kinds(k):
    """(seed_kind, extended_kind, solo_kind) of CRN seed k; roots are named like the kinds.
    k=1 has no seed-kind (first-seed receipt already exists); k=2 are the existing constants."""
    if k == 1:
        return None, KIND_EXT, KIND_SOLO[1]
    if k == 2:
        return KIND_SEED2, KIND_EXT2, KIND_SOLO[2]
    return f'confirm_seed{k}_all8', f'confirm_extended_seed{k}_all8', f'confirm_solo_all8_s{k}'


def extra_seed_salt(vid, k):
    """CRN salt of extra seed k >= 3: a pure deterministic function of (first-seed vid, k).

    salt = f'{vid}#seed{k}' - distinct from the seed-2 salt (vid) and from every other extra
    seed's salt. The generator seed replicate() derives from it (int(digest(...)))
    is checked against the evaluation world-seed ranges above; in the collision case the salt is
    deterministically re-derived with a '#x<n>' suffix, so no evaluation seed is ever the CRN
    generator seed of an extra seed."""
    if int(k) < FIRST_EXTRA_SEED:
        raise ValueError(f'extra seeds start at {FIRST_EXTRA_SEED}, not {k}')
    from .protocol import digest

    def gen_seed(s):
        return int(digest(dict(replication=s))[:15], 16)
    base = f'{vid}#seed{int(k)}'
    for n in range(1000):
        candidate = base if not n else f'{base}#x{n}'
        if not any(lo <= gen_seed(candidate) <= hi for lo, hi in EVAL_SEED_RANGES):
            return candidate
    raise ValueError(f'extra_seed_salt: cannot avoid evaluation seeds for {vid!r} seed {k}')


def extra_seed_indices(extra_seeds):
    n = int(extra_seeds or 0)
    if n < 0:
        raise ValueError('extra_seeds must be >= 0')
    return list(range(FIRST_EXTRA_SEED, FIRST_EXTRA_SEED+n))


RULE = ('all8_both_seeds (owner 2026-11-03 22:05): all K candidates + policy-continues on BOTH CRN seeds; '
        'per seed every candidate with within-seed per-second paired bootstrap lower95 > 0 and composite '
        'safety pass is admitted (each seed also needs its own extended-window later-harm pass, measured '
        'on that seed\'s CRN); solos per admitted (plan, seed); within-seed cooperative label; each admitted '
        '(plan, seed) is its own imitation example from THAT seed\'s tape.')


def mode(value):
    """Normalise a verify_mode config value; None/'' -> legacy."""
    v = value or LEGACY
    if v not in VERIFY_MODES:
        raise ValueError(f'unknown verify_mode {value!r}')
    return v


# ---------------------------------------------------------------- cost quote

def _branch(t0, duration_s, tail_s, end_s=1200):
    from . import sumo_verify as sv
    return sv.branch_sim_s(min(int(t0)+int(duration_s)+int(tail_s), int(end_s)))


def quote_decision(t0, durations, members, admitted=None, first_seed_positive=None, end_s=1200,
                   extra_seeds=0, cooperative=None):
    """Charged sim-s of ONE decision: today (legacy) vs all8_both_seeds (vs + extra seeds).

    durations/members: per candidate (K entries). admitted: {1: [idx..], 2: [idx..]} (0-based
    candidate indices admitted per seed; seed-1 LB>0 set is assumed == admitted for the extended
    quote). Legacy: first-seed K+1 branches; if the first seed is positive, replication (2) +
    extended (2) + per-member solos (2n, run when replication + later-harm pass, here assumed).
    extra_seeds (BUILD option, DEFAULT 0 = output keys unchanged): per extra CRN seed every
    COOPERATIVE plan is re-verified + policy reference (group), gets its extended later-harm
    window (all plans assumed LB>0 on the extra seed, like the seed-1/2 assumption) and its
    solos (assumed admitted; cooperative plans always have >= 2 members). cooperative: candidate
    indices with a within-seed cooperative example on seeds 1/2; None = the union of the
    admitted seed-1/2 candidates with >= 2 members (upper bound, consistent with the other
    assumptions here)."""
    from .replay_bank import EXTENDED_TAIL_S
    from .sumo_verify import TAIL_S
    admitted = {int(k): list(v) for k, v in (admitted or {1: [], 2: []}).items()}
    K = len(durations)
    maxD = max(durations)
    first = (K+1)*_branch(t0, maxD, TAIL_S, end_s)
    best = (admitted.get(1) or [0])[0]
    positive = bool(admitted.get(1)) if first_seed_positive is None else bool(first_seed_positive)
    legacy_conf = 0
    if positive:
        n = len(members[best])
        legacy_conf = (2*_branch(t0, durations[best], TAIL_S, end_s)
                       + 2*_branch(t0, durations[best], EXTENDED_TAIL_S, end_s)
                       + (2*n*_branch(t0, durations[best], TAIL_S, end_s) if n >= 2 else 0))
    seed2 = (K+1)*_branch(t0, maxD, TAIL_S, end_s)
    s1 = admitted.get(1) or []
    ext = (len(s1)+1)*_branch(t0, max(durations[i] for i in s1), EXTENDED_TAIL_S, end_s) if s1 else 0
    # MAIN 2026-11-03 (launch-final): seed-2 later harm is MEASURED for new all8 verifications -
    # one extended-window verification of the seed-2 LB>0 set (assumed == admitted[2]) on seed 2.
    s2 = admitted.get(2) or []
    ext += (len(s2)+1)*_branch(t0, max(durations[i] for i in s2), EXTENDED_TAIL_S, end_s) if s2 else 0
    solos = sum((len(members[i])+1)*_branch(t0, durations[i], TAIL_S, end_s)
                for k in (1, 2) for i in admitted.get(k) or [] if len(members[i]) >= 2)
    legacy = first+legacy_conf
    new = first+seed2+ext+solos
    out = dict(t0=t0, K=K, first_seed_sim_s=first, legacy_confirmation_sim_s=legacy_conf,
               legacy_total_sim_s=legacy, all8_seed2_sim_s=seed2, all8_extended_sim_s=ext,
               all8_solo_sim_s=solos, all8_confirmation_sim_s=seed2+ext+solos, all8_total_sim_s=new,
               multiplier_total=new/legacy, multiplier_confirmation=((seed2+ext+solos)/legacy_conf
                                                                     if legacy_conf else None),
               admitted=admitted, note='branch sim-s charged from t=0 (warm-up included), as sv.quote')
    coop = (sorted({int(i) for i in cooperative}) if cooperative is not None
            else sorted({i for k in (1, 2) for i in admitted.get(k) or [] if len(members[i]) >= 2}))
    n_extra = int(extra_seeds or 0)
    if n_extra > 0:
        eg = (len(coop)+1)*_branch(t0, max((durations[i] for i in coop), default=maxD), TAIL_S, end_s) if coop else 0
        ee = (len(coop)+1)*_branch(t0, max((durations[i] for i in coop), default=maxD), EXTENDED_TAIL_S, end_s) if coop else 0
        es = sum((len(members[i])+1)*_branch(t0, durations[i], TAIL_S, end_s) for i in coop)
        extra_total = n_extra*(eg+ee+es)
        out.update(extra_seeds=n_extra, cooperative=coop, extra_seeds_group_sim_s=n_extra*eg,
                   extra_seeds_extended_sim_s=n_extra*ee, extra_seeds_solo_sim_s=n_extra*es,
                   extra_seeds_sim_s=extra_total, all8_total_with_extra_sim_s=new+extra_total,
                   multiplier_total_with_extra=((new+extra_total)/legacy if legacy else None))
    return out


def standard_quote():
    """Per-decision quote at representative windows (K=8, D=120 s, 2-member plans)."""
    D, M = [120]*8, [['a', 'b']]*8
    rows = {}
    for t0 in (420, 600, 900):
        rows[f't0_{t0}'] = dict(
            first_seed_negative_seed2_none=quote_decision(t0, D, M, {1: [], 2: []}),
            first_seed_negative_seed2_one=quote_decision(t0, D, M, {1: [], 2: [3]}),
            one_winner_each_seed=quote_decision(t0, D, M, {1: [0], 2: [0]}),
            different_winners_each_seed=quote_decision(t0, D, M, {1: [0], 2: [5]}),
            two_winners_each_seed=quote_decision(t0, D, M, {1: [0, 1], 2: [0, 2]}))
    return rows


def extra_seeds_quote(n=4):
    """Per-decision quote for BUILD option extra_seeds=n (CLI 'extra-seeds'; DEFAULT 4).

    Rows: no cooperative example on seeds 1/2 -> the extra seeds never run (0 extra sim-s);
    one/two cooperative winners -> group+reference, extended later-harm window and solos per
    extra seed (upper-bound admission assumption, see quote_decision)."""
    D, M = [120]*8, [['a', 'b']]*8
    rows = {}
    for t0 in (420, 600, 900):
        rows[f't0_{t0}'] = dict(
            no_cooperative=quote_decision(t0, D, M, {1: [], 2: []}, extra_seeds=n),
            one_cooperative_winner=quote_decision(t0, D, M, {1: [0], 2: [0]}, extra_seeds=n, cooperative=[0]),
            two_cooperative_winners=quote_decision(t0, D, M, {1: [0, 1], 2: [0, 2]}, extra_seeds=n,
                                                   cooperative=[0, 1, 2]))
    return rows


# ---------------------------------------------------------------- helpers

def first_seed_candidates(receipt, branches_dir, candidates=None):
    """The K first-seed candidate plans, identity-checked against the sealed receipt.

    ``candidates`` (in-memory, e.g. from the collector) or the branch spec files written by
    SubprocessBranchRunner (<branches>/<vid>/cand<i>.spec.json, sealed 'branch_sha256')."""
    from .exact_grade import verify_sealed
    from .protocol import digest
    want = receipt['candidate_plan_sha256']
    if candidates is None:
        candidates = []
        d = Path(branches_dir)/receipt['verification_id']
        for i in range(1, len(want)+1):
            p = d/f'cand{i}.spec.json'
            if not p.exists():
                raise FileNotFoundError(f'first-seed candidate spec missing: {p}')
            spec = json.loads(p.read_text(encoding='utf-8'))
            verify_sealed(spec, 'branch_sha256')
            if spec['branch_sha256'] != receipt['branch_specs_sha256'][i]:
                raise ValueError(f'candidate spec {p} differs from the sealed receipt')
            candidates.append(spec['plan'])
    candidates = [copy.deepcopy(c) for c in candidates]
    if [digest(c) for c in candidates] != list(want):
        raise ValueError('candidate plans differ from the first-seed receipt identity')
    return candidates


def _pair_safety(receipt, branches_dir, index):
    """Candidate ``index`` (1-based branch) minus policy deltas; MAIN04:26 verdict."""
    from . import replay_bank as rb
    from .bank_safety import verdict
    pol, cand = receipt['branches'][0], receipt['branches'][index]
    deltas = {k: float(cand['window'][k])-float(pol['window'][k]) for k in rb.SAFETY_KEYS}
    w = receipt['window']
    tr = {b['branch']: Path(branches_dir)/receipt['verification_id']/b['branch']/'observable.jsonl.gz'
          for b in (pol, cand)}
    deltas['ttc_below_1p5_events'] = (rb.ttc_events(tr[cand['branch']], w['start_s'], w['end_s'])
                                      - rb.ttc_events(tr[pol['branch']], w['start_s'], w['end_s']))
    v = verdict(deltas)
    return deltas, bool(v['new_combined_rule']), bool(v['old_no_metric_increase'])


def _complete(r):
    return r is not None and r.get('status') in ('accepted', 'rejected')


def _candidate_stats(receipt, branches_dir, index):
    """Within-seed paired bootstrap (policy-continues minus candidate) -> stats or raises."""
    from . import bank_statistics as bs
    if receipt['branches'][index]['status'] != 'graded' or receipt['branches'][0]['status'] != 'graded':
        raise bs._Missing(f'branch {receipt["branches"][index]["branch"]} ungraded')
    pol = bs._read_result(branches_dir, receipt['verification_id'], receipt['branches'][0])
    alt = bs._read_result(branches_dir, receipt['verification_id'], receipt['branches'][index])
    return bs._stats([(pol, alt, receipt['window'])]), alt


def _health(root, arm, health):
    if health is None:
        from .warm_health import assert_running
        assert_running(root, arm)
    elif health:
        health()


def _verify(kind, snapshot, candidates, root_dir, runner_for, run_root, phase, proposer, workers, tail_s,
            charges, root, arm, health):
    """One sv.verify through the existing hooks; appends the charge row."""
    from . import replay_bank as rb
    from . import sumo_verify as sv
    _health(root, arm, health)
    r = sv.verify(snapshot, candidates, runner_for(root_dir), rb._confirm_charge(run_root, phase, kind),
                  proposer, workers=workers, root=root_dir, tail_s=tail_s)
    charges.append(dict(kind=kind, phase=phase, verification_id=r['verification_id'], status=r['status'],
                        charged_sim_s=r.get('charged_sim_s', 0), actual_sim_s=r.get('actual_sim_s'),
                        n_branches=len(r.get('branches') or []), receipt_sha256=r['receipt_sha256']))
    return r


def _reference_agrees(ref_tape, solo_tape, window):
    """Bitwise per-second fingerprint agreement of two policy-reference tapes over [start_s, end_s].

    Missing fingerprints count as disagreement (never imputed)."""
    from tools.coopnative.storage import canonical

    def fp(tape, t):
        v = tape['prefixes'].get(str(t))
        if v is None and tape.get('final') is not None and float(tape['final']['t_s']) == float(t):
            v = tape['final']
        return v
    for t in range(int(window['start_s']), int(window['end_s'])+1):
        a, b = fp(ref_tape, t), fp(solo_tape, t)
        if a is None or b is None or canonical(a) != canonical(b):
            return False
    return True


def _cooperation(seed, plan, plan_tape, window, solo, solo_dir, reference_J, admitted, reference_tape=None):
    """Within-seed cooperation label from one batched solo verification (never pooled).

    The synergy is always computed on the COMMON window [t0, min(end_solo, end_group)]. A solo policy
    branch whose total J differs from the reference J only because its window is shorter is NOT a
    reference mismatch when the two policy tapes agree bit-for-bit over the common window
    (``reference_tape`` = the group verification's policy branch tape); only tapes that actually differ
    over the overlap are labelled reference_mismatch. Records whose J already match are unchanged."""
    from . import bank_statistics as bs
    if len(plan['members']) < 2:
        return dict(cooperative=False, cooperative_status='single_cav',
                    statistical_reason='single-CAV plan: cooperative=False by definition')
    if solo is None or not _complete(solo):
        return dict(cooperative=False, cooperative_status='incomplete',
                    statistical_reason=f'seed-{seed} solo verification incomplete; never imputed')
    graded = [i for i, J in enumerate(solo['J_candidates_veh_s']) if J is not None]
    if len(graded) != len(plan['members']):
        return dict(cooperative=False, cooperative_status='incomplete',
                    statistical_reason=f'seed-{seed} solo branch failed; never imputed')
    match = solo['J_policy_veh_s'] == reference_J
    advs = [solo['J_policy_veh_s']-J for J in solo['J_candidates_veh_s']]
    b = max(range(len(advs)), key=lambda i: (advs[i], -i))
    try:
        solo_tape = bs._read_result(solo_dir/'branches', solo['verification_id'], solo['branches'][b+1])
        common = bs.paired_window(window, solo['window'])
        st = bs._stats([(solo_tape, plan_tape, common)])
        agree = None
        if not match and reference_tape is not None:
            solo_ref = bs._read_result(solo_dir/'branches', solo['verification_id'], solo['branches'][0])
            agree = _reference_agrees(reference_tape, solo_ref, common)
    except bs._Missing as e:
        return dict(cooperative=False, cooperative_status='data_missing', statistical_reason=str(e),
                    solo_advantages_veh_s=advs)
    st.update(best_solo_member=plan['members'][b], best_solo_advantage_veh_s=advs[b],
              paired_window=dict(group_window=window, solo_window=solo['window'], used=common))
    positive = st['lower95_one_sided_veh_s'] > 0
    status = 'cooperative' if positive else 'not_cooperative'
    reason = f'seed-{seed} synergy lower95 ' + ('> 0' if positive else '<= 0')
    extra = {}
    if not match:
        if agree:
            extra = dict(solo_policy_matches_reference_on_common_window=True, synergy_window_used=common)
            reason += (f' (computed on common window [{common["start_s"]}, {common["end_s"]}]; policy reference '
                       'tapes bit-identical there)')
        else:
            status, positive = 'reference_mismatch', False
            reason = (f'seed-{seed} solo policy branch differs from the reference policy branch over the common '
                      'window; not counted')
            extra = dict(solo_policy_matches_reference_on_common_window=False)
    return dict(cooperative=bool(admitted and positive), cooperative_status=status, statistical_reason=reason,
                synergy=st, solo_advantages_veh_s=advs, solo_verification_id=solo['verification_id'],
                solo_receipt_sha256=solo['receipt_sha256'], solo_policy_matches_reference=match, **extra)


# ---------------------------------------------------------------- confirmation (one decision)

def confirm_all8(root, arm, receipt, snapshot, runner_for, run_root, phase, workers=2, version=None,
                 candidates=None, branches_dir=None, health=None, later_harm_seed1=True, later_harm_seed2=True,
                 slim_rejected_tapes=None, extra_seeds=0):
    """all8_both_seeds confirmation of ONE completed first-seed decision -> sealed record.

    slim_rejected_tapes (DEFAULT OFF = None -> the C4NM_SLIM_REJECTED_TAPES env var, unset
    means OFF; byte-identical when off): after the record is sealed, every explicitly rejected
    candidate branch tape (and its extended-window branch) is slimmed to SLIM.json and the full
    RESULT.json(.gz) removed (tools/coopmwm/slim_tapes.py). Admitted, policy-reference and solo
    tapes are never touched; the record itself is unchanged.

    extra_seeds (BUILD option, DEFAULT 0 = records, calls and charges byte-identical): after
    seeds 1/2 and their solos, every plan with a within-seed COOPERATIVE example is re-verified
    (+ policy-continues reference, one batched verification per extra seed) on CRN seeds
    3..(2+extra_seeds) with extra_seed_salt salts; each (plan, extra seed) is admitted
    independently by the SAME per-seed rules (LB > 0, own-seed safety, own-seed measured
    later harm), then gets its own solos on THAT seed; examples are appended with
    seed_index >= 3 (per_seed_v1 consumes them unchanged). Seeds >= 3 never alter the seed-1/2
    examples, mean_replicated_advantage_veh_s (seeds-1/2 admission mean), the legacy
    ``replicated`` flag (cooperative on BOTH seeds 1 and 2) or the exact gate."""
    from . import replay_bank as rb
    from . import bank_statistics as bs
    from . import exact_grade as xg
    from .protocol import digest
    if receipt['status'] not in ('accepted', 'rejected'):
        raise ValueError('all8 confirmation needs a completed first-seed decision')
    bank = Path(rb.bank_root(root, arm));vid = receipt['verification_id']
    target = bank/RECORDS/(vid+'.json')
    if target.exists():
        v = json.loads(target.read_text());xg.verify_sealed(v, 'confirmation_sha256');return v
    first_dir = Path(branches_dir if branches_dir is not None else receipt['_branches'])
    plans = first_seed_candidates(receipt, first_dir, candidates)
    rc = {k: v for k, v in receipt.items() if not k.startswith('_')}
    proposer = rc['proposer'];charges = []
    common = dict(root=root, arm=arm, health=health, runner_for=runner_for, run_root=run_root, phase=phase,
                  proposer=proposer, workers=workers, charges=charges)

    def verify(kind, snap, cands, root_dir, tail_s):
        return _verify(kind, snap, cands, root_dir, tail_s=tail_s, **common)

    # Seed 2: the SAME K candidates + policy continues, existing salt convention.
    s2_snapshot = rb.replicate(snapshot, vid)
    s2_dir = bank/SEED2_ROOT
    s2 = verify(KIND_SEED2, s2_snapshot, plans, s2_dir, rb.sv.TAIL_S)
    seeds = {1: (rc, first_dir), 2: (s2, s2_dir/'branches')}
    branch_of = {k: {i: i+1 for i in range(len(plans))} for k in seeds}

    def build_eval(k, i):
        """Within-seed paired bootstrap + REQUIRED own-seed safety deltas of candidate i on seed k."""
        r, bdir = seeds[k]
        if not _complete(r):
            return dict(missing=f'seed-{k} decision {r.get("status")}: {r.get("reason")}')
        try:
            st, tape = _candidate_stats(r, bdir, branch_of[k][i])
        except bs._Missing as e:
            return dict(missing=str(e))
        try:     # own-seed safety deltas are REQUIRED (no fallback to another seed's window)
            deltas, safe, old_safe = _pair_safety(r, bdir, branch_of[k][i])
        except (OSError, EOFError, KeyError, TypeError, ValueError) as e:
            return dict(missing=f'seed-{k} safety deltas unavailable: {e!r}')
        return dict(stats=st, tape=tape, window_deltas=deltas, window_safe=safe,
                    window_safe_old=old_safe,
                    advantage=r['J_policy_veh_s']-r['J_candidates_veh_s'][branch_of[k][i]-1])

    # Within-seed paired bootstrap of every candidate on each seed (independent selection).
    evals = {k: {} for k in seeds}
    for k in seeds:
        for i in range(len(plans)):
            evals[k][i] = build_eval(k, i)

    # Later harm, per seed: one extended-window verification of every LB>0 candidate of that seed
    # on THAT seed's CRN (seed 1 = decision snapshot, seed 2 = the seed-2 replicate); MAIN
    # 2026-11-03: new all8 verifications measure seed 2 too and a measured failure rejects.
    measure = {1: bool(later_harm_seed1), 2: bool(later_harm_seed2)}
    snaps = {1: snapshot, 2: s2_snapshot}
    exts, ext_index, lbs = {k: None for k in seeds}, {k: {} for k in seeds}, {}
    for k in sorted(snaps):
        lbs[k] = [i for i, e in evals[k].items() if 'stats' in e and e['stats']['lower95_one_sided_veh_s'] > 0]
        if measure[k] and lbs[k]:
            exts[k] = verify(KIND_EXTS[k], snaps[k], [plans[i] for i in lbs[k]], bank/EXT_ROOTS[k],
                             rb.EXTENDED_TAIL_S)
            ext_index[k] = {i: j+1 for j, i in enumerate(lbs[k])}
    ext, lb1 = exts[1], lbs[1]

    def build_example(k, i):
        e = evals[k][i]
        branch = seeds[k][0]['branches'][branch_of[k][i]] if _complete(seeds[k][0]) else None
        plan = plans[i]
        ex = dict(seed_index=k, candidate_index=i, example_id=f'{vid}#c{i+1}#s{k}', plan=copy.deepcopy(plan),
                  plan_id=plan['id'], plan_sha256=digest(plan), duration_s=int(plan['duration_s']),
                  members=list(plan['members']), group_size=len(plan['members']))
        if 'missing' in e:
            ex.update(admitted=False, status='data_missing', reason='per-unit data missing: '+e['missing'],
                      cooperative=False, cooperative_status='data_missing', later_harm='not_measured')
            return ex
        st = e['stats'];lower = st['lower95_one_sided_veh_s']
        later, safe, source = None, e['window_safe'], f'seed{k}_window'
        if measure[k] and lower > 0:
            j, x = ext_index[k][i], exts[k]
            if not _complete(x) or x['J_candidates_veh_s'][j-1] is None:
                later = None;ex['later_harm_missing'] = True
            else:
                try:
                    ed, safe, _ = _pair_safety(x, bank/(EXT_ROOTS.get(k) or seed_kinds(k)[1])/'branches', j)
                except (OSError, EOFError, KeyError, TypeError, ValueError):
                    later = None;ex['later_harm_missing'] = True
                else:
                    later = x['J_policy_veh_s']-x['J_candidates_veh_s'][j-1] > 0
                    ex['extended_deltas'] = ed;source = f'extended_window_seed{k}'
                    ex['extended_advantage_veh_s'] = x['J_policy_veh_s']-x['J_candidates_veh_s'][j-1]
        if measure[k] and lower > 0 and later is None:
            ok, reason = False, f'seed-{k} extended-window branch missing; never imputed'
        elif later is False:
            ok, reason = False, f'seed-{k} later-harm check failed (extended-window advantage <= 0)'
        elif not lower > 0:
            ok, reason = False, f'seed-{k} one-sided 95% lower bound {lower:.4g} <= 0'
        elif not safe:
            ok, reason = False, f'seed-{k} composite safety check failed ({source})'
        else:
            ok = True
            reason = f'seed-{k} lower95 > 0, safety pass' + (', later-harm pass' if later else
                                                              ', later-harm NOT MEASURED')
        ex.update(admitted=ok, status='admitted' if ok else 'rejected', reason=reason, admission=st,
                  lower95_one_sided_veh_s=lower, total_veh_s=st['pooled_total_veh_s'],
                  mean_per_unit_veh_s=st['per_seed'][0]['mean_per_unit_veh_s'],
                  n_units=st['per_seed'][0]['n_units'], advantage_veh_s=e['advantage'],
                  later_harm=('pass' if later else 'fail') if later is not None else 'not_measured',
                  safety_ok=safe, safety_source=source, window_deltas=e['window_deltas'],
                  window_safety_old_rule_ok=e['window_safe_old'],
                  tape=dict(branches_dir=str(seeds[k][1]), verification_id=seeds[k][0]['verification_id'],
                            branch=dict(branch), window=dict(seeds[k][0]['window']), seed_index=k,
                            intent_entry=(f'{vid}/{branch["branch"]}' if k == 1
                                          else f'{vid}/seed{k}/{branch["branch"]}')))
        return ex

    examples = []
    for k in sorted(seeds):
        for i in range(len(plans)):
            examples.append(build_example(k, i))

    # Solos per admitted (plan, seed) on THAT seed: one batched verification, shared policy branch.
    def run_solo(k, ex):
        s_kind = KIND_SOLO.get(k) or seed_kinds(k)[2]
        plan = ex['plan']
        solo = None
        if len(plan['members']) >= 2:
            solo = verify(s_kind, snaps[k], [rb.solo_candidate(plan, m) for m in plan['members']],
                          bank/s_kind, rb.sv.TAIL_S)
        ref_tape = None
        if solo is not None and solo['J_policy_veh_s'] != seeds[k][0]['J_policy_veh_s']:
            try:     # only needed when the solo window differs from the group window
                ref_tape = bs._read_result(seeds[k][1], seeds[k][0]['verification_id'],
                                           seeds[k][0]['branches'][0])
            except (bs._Missing, OSError, ValueError):
                ref_tape = None
        ex.update(_cooperation(k, plan, evals[k][ex['candidate_index']]['tape'], ex['tape']['window'], solo,
                               bank/s_kind, seeds[k][0]['J_policy_veh_s'], True, reference_tape=ref_tape))

    for ex in examples:
        if ex['status'] != 'admitted':
            if 'cooperative' not in ex:
                ex.update(cooperative=False, cooperative_status='not_admitted')
            continue
        run_solo(ex['seed_index'], ex)

    # Extra seeds (BUILD option, DEFAULT 0 = nothing below runs): seeds 3..(2+extra_seeds) for
    # the plans that carry a within-seed COOPERATIVE example on seeds 1/2. Each (plan, extra
    # seed) stands alone: independent admission by the same per-seed rules, own extended
    # later-harm window, own solos; examples are APPENDED (seed_index >= 3).
    coop_idx = sorted({e['candidate_index'] for e in examples if e.get('cooperative') is True})
    extras = {}
    for k in extra_seed_indices(extra_seeds):
        if not coop_idx:
            break
        kind, k_ext, k_solo = seed_kinds(k)
        salt = extra_seed_salt(vid, k)
        ksnap = rb.replicate(snapshot, salt)
        kdir = bank/kind
        kr = verify(kind, ksnap, [plans[i] for i in coop_idx], kdir, rb.sv.TAIL_S)
        seeds[k] = (kr, kdir/'branches');snaps[k] = ksnap;measure[k] = True
        branch_of[k] = {i: j+1 for j, i in enumerate(coop_idx)}
        evals[k] = {i: build_eval(k, i) for i in coop_idx}
        lbs[k] = [i for i, e in evals[k].items() if 'stats' in e and e['stats']['lower95_one_sided_veh_s'] > 0]
        if lbs[k]:
            exts[k] = verify(k_ext, ksnap, [plans[i] for i in lbs[k]], bank/k_ext, rb.EXTENDED_TAIL_S)
            ext_index[k] = {i: j+1 for j, i in enumerate(lbs[k])}
        for i in coop_idx:
            ex = build_example(k, i)
            examples.append(ex)
            if ex['status'] == 'admitted':
                run_solo(k, ex)
            elif 'cooperative' not in ex:     # same not_admitted fields the seed-1/2 loop sets
                ex.update(cooperative=False, cooperative_status='not_admitted')
        extras[k] = dict(kind=kind, ext_kind=k_ext, solo_kind=k_solo, salt=salt, receipt=kr, root=kdir)

    adm = [e for e in examples if e['admitted']]
    adm12 = [e for e in adm if e['seed_index'] in (1, 2)]
    winners = {str(k): [e['plan_id'] for e in adm if e['seed_index'] == k] for k in sorted(seeds)}
    coop = [e for e in examples if e.get('cooperative') is True]
    coop_seeds = sorted({e['seed_index'] for e in coop})
    by_plan = {}
    for e in coop:
        by_plan.setdefault(e['plan_id'], set()).add(e['seed_index'])
    replicated = sorted(p for p, s in by_plan.items() if {1, 2} <= s)
    group_sizes = [len(p['members']) for p in plans]
    out = dict(schema=SCHEMA, verify_mode=ALL8, bank_rule='per_seed_v1', rule=RULE, verification_id=vid, arm=arm,
               policy_version=version, t0=rc['t0'], K=len(plans), first_seed_status=rc['status'],
               first_seed_advantage_veh_s=rc.get('grade_veh_s'),
               first_seed=dict(verification_id=vid, receipt_sha256=rc['receipt_sha256'], branches_dir=str(first_dir)),
               seed2=dict(verification_id=s2['verification_id'], status=s2['status'], salt=vid,
                          snapshot_sha256=s2.get('snapshot_sha256'), receipt_sha256=s2['receipt_sha256'],
                          root=str(s2_dir), J_policy_veh_s=s2.get('J_policy_veh_s')),
               extended=(dict(verification_id=ext['verification_id'], status=ext['status'],
                              receipt_sha256=ext['receipt_sha256'], candidate_indices=lb1)
                         if ext is not None else None),
               extended_seed2=(dict(verification_id=exts[2]['verification_id'], status=exts[2]['status'],
                                    receipt_sha256=exts[2]['receipt_sha256'], candidate_indices=lbs[2])
                               if exts[2] is not None else None),
               later_harm_measured=dict(seed1=measure[1], seed2=measure[2]),
               candidates=[dict(index=i, plan_id=p['id'], plan_sha256=digest(p), members=list(p['members']),
                                duration_s=int(p['duration_s'])) for i, p in enumerate(plans)],
               winners=winners, distinct_winners_across_seeds=set(winners['1']) != set(winners['2']),
               admitted=bool(adm),
               mean_replicated_advantage_veh_s=(sum(e['advantage_veh_s'] for e in adm12)/len(adm12)
                                                if adm12 else None),
               per_seed_v1=dict(examples=examples, admitted_seeds=sorted({e['seed_index'] for e in adm}),
                                cooperative_seeds=coop_seeds, replicated=bool(replicated),
                                replicated_plans=replicated,
                                reason='; '.join(f'{e["example_id"]}: {e["reason"]}' for e in examples)),
               cooperation=dict(group_size=max(group_sizes) if group_sizes else 0, cooperative=bool(coop),
                                cooperative_seeds=coop_seeds, cooperative_examples=len(coop),
                                replicated=bool(replicated), needs_second_seed_solos=False,
                                rule='within-seed synergy per admitted (plan, seed); never pooled'),
               charges=charges, charged_sim_s=sum(c['charged_sim_s'] or 0 for c in charges),
               actual_sim_s=sum(c['actual_sim_s'] or 0 for c in charges), unix=time.time())
    if extras:     # only present with extra_seeds >= 1: never written when the option is off
        out['extra_seeds'] = int(extra_seeds)
        out['extra'] = {str(k): dict(
            seed_index=k, salt=x['salt'], kind=x['kind'],
            verification_id=x['receipt']['verification_id'], status=x['receipt']['status'],
            snapshot_sha256=x['receipt'].get('snapshot_sha256'), receipt_sha256=x['receipt']['receipt_sha256'],
            root=str(x['root']), candidates=coop_idx, J_policy_veh_s=x['receipt'].get('J_policy_veh_s'),
            extended=(dict(verification_id=exts[k]['verification_id'], status=exts[k]['status'],
                           receipt_sha256=exts[k]['receipt_sha256'], candidate_indices=lbs[k])
                      if exts.get(k) is not None else None),
            later_harm_measured=True, solo_root=str(bank/x['solo_kind'])) for k, x in sorted(extras.items())}
    xg._write_new(target, xg._seal(out, 'confirmation_sha256'))
    sealed = json.loads(target.read_text())
    from . import slim_tapes
    if slim_tapes.enabled(slim_rejected_tapes):
        slim_tapes.slim_record(bank, sealed)
    return sealed


# ---------------------------------------------------------------- readers (per_seed_v1 integration)

def relabel_reference_mismatch(record, bank):
    """Reader-side (no SUMO, nothing rewritten) correction of 'reference_mismatch' examples.

    A sealed example labelled reference_mismatch only because its solo window is shorter than the group
    window is re-derived from the stored solo receipt and tapes with ``_cooperation`` (synergy on the
    common window; policy tapes compared bit-for-bit there). Returns the record unchanged (same object)
    when no example changes, else a corrected deep copy with the per-seed aggregates recomputed and
    ``reader_relabel`` set on each changed example. Any unreadable data keeps the stored label."""
    from . import bank_statistics as bs
    exs = record.get('per_seed_v1', {}).get('examples') or []
    if not any(e.get('cooperative_status') == 'reference_mismatch' for e in exs):
        return record
    out = copy.deepcopy(record);changed = 0
    for e in out['per_seed_v1']['examples']:
        if e.get('cooperative_status') != 'reference_mismatch':
            continue
        try:
            k, t = e['seed_index'], e['tape']
            bdir = Path(t['branches_dir'])
            solo_root = Path(bank)/(SOLO_ROOTS.get(k) or seed_kinds(k)[2])
            solo = bs._receipt(solo_root, dict(verification_id=e['solo_verification_id'],
                                               receipt_sha256=e.get('solo_receipt_sha256')))
            grec = bs._receipt(bdir.parent, dict(verification_id=t['verification_id']))
            plan_tape = bs._read_result(bdir, t['verification_id'], t['branch'])
            ref_tape = bs._read_result(bdir, t['verification_id'], grec['branches'][0])
            members = next(c['members'] for c in out['candidates'] if c['index'] == e['candidate_index'])
            new = _cooperation(k, dict(members=members), plan_tape, t['window'], solo, solo_root,
                               grec['J_policy_veh_s'], True, reference_tape=ref_tape)
        except (bs._Missing, OSError, KeyError, ValueError, StopIteration, TypeError):
            continue
        if new['cooperative_status'] in ('reference_mismatch', 'data_missing', 'incomplete'):
            continue
        e.update(new);e['reader_relabel'] = 'reference_mismatch -> '+new['cooperative_status'];changed += 1
    if not changed:
        return record
    ps, exs = out['per_seed_v1'], out['per_seed_v1']['examples']
    coop = [e for e in exs if e.get('cooperative') is True]
    by_plan = {}
    for e in coop:
        by_plan.setdefault(e['plan_id'], set()).add(e['seed_index'])
    replicated = sorted(p for p, s in by_plan.items() if {1, 2} <= s)   # extra seeds (>= 3) never change it
    seeds = sorted({e['seed_index'] for e in coop})
    ps.update(cooperative_seeds=seeds, replicated=bool(replicated), replicated_plans=replicated)
    out['cooperation'].update(cooperative=bool(coop), cooperative_seeds=seeds, cooperative_examples=len(coop),
                              replicated=bool(replicated))
    return out


def records(root, arm):
    """{vid: sealed all8 record}; each record IS a per_seed_v1 view (bank_rule='per_seed_v1')."""
    from . import replay_bank as rb
    from .exact_grade import verify_sealed
    out = {}
    bank = rb.bank_root(root, arm)
    for p in sorted((Path(bank)/RECORDS).glob('*.json')):
        v = json.loads(p.read_text());verify_sealed(v, 'confirmation_sha256')
        out[v['verification_id']] = relabel_reference_mismatch(v, bank)
    return out


def overlay(conf, root, arm):
    """per_seed_v1 integration: all8 records replace that decision's legacy confirmation view
    (added for decisions without one, e.g. first-seed-rejected decisions with a seed-2 winner)."""
    out = dict(conf)
    out.update(records(root, arm))
    return out


def bank_examples(root, arm):
    """Admitted (plan, seed) imitation examples, standalone: each carries its own plan, t0,
    seed-specific tape location/digest, advantage and within-seed cooperative label."""
    rows = []
    for vid, c in sorted(records(root, arm).items()):
        for e in c['per_seed_v1']['examples']:
            if e['admitted']:
                rows.append(dict(e, verification_id=vid, t0=c['t0'], policy_version=c['policy_version']))
    return rows


def imitation_entry(example, t0, weight):
    """The (tape, entry) that per_seed_v1 bank_pairs turns into imitation pairs for one example
    (uses the example's OWN plan, not the first-seed chosen plan)."""
    from .accel_storage import read
    from .protocol import digest
    t = example['tape']
    tape = read(Path(t['branches_dir'])/t['verification_id']/t['branch']['branch']/'RESULT.json')
    if t['branch'].get('result_sha256') and digest(tape) != t['branch']['result_sha256']:
        raise ValueError('bank branch tape differs from its sealed verification receipt')
    return tape, dict(start_s=int(t0), duration_s=int(example['duration_s']), plan_id=example['plan_id'],
                      weight=weight)


# ---------------------------------------------------------------- pending driver

def pending(root, arm, sources=()):
    """Completed first-seed decisions (accepted AND rejected) without an all8 record, oldest first."""
    from . import replay_bank as rb
    done = set(records(root, arm))
    return [r for r in rb.bank_decisions(root, arm, sources=sources)
            if r['status'] in ('accepted', 'rejected') and r['verification_id'] not in done]


def confirm_pending_all8(root, arm, runner_for, run_root, phase, workers=2, sources=(), limit=None, health=None,
                         slim_rejected_tapes=None, extra_seeds=0):
    """extra_seeds (BUILD option, DEFAULT 0 = unchanged): extra CRN seeds 3..(2+extra_seeds) for
    the plans cooperative on seeds 1/2, per decision (arms B and R alike).
    slim_rejected_tapes (DEFAULT OFF, all8 only): after each record is sealed, rejected candidate
    branch tapes are slimmed to SLIM.json (tools/coopmwm/slim_tapes.py)."""
    from . import replay_bank as rb
    by_vid = {}
    for p in (Path(rb.bank_root(root, arm))/'decisions').glob('*.json'):
        v = json.loads(p.read_text());by_vid[v['verification_id']] = v
    out = []
    for r in pending(root, arm, sources):
        if limit is not None and len(out) >= limit:
            break
        snap = rb._decision_snapshot(root, arm, r, by_vid)
        out.append(confirm_all8(root, arm, r, snap, runner_for, run_root, phase, workers,
                                r['_policy_version'], health=health,
                                slim_rejected_tapes=slim_rejected_tapes,
                                extra_seeds=extra_seeds)['admitted'])
    return out


def main(argv=None):
    """python -m tools.coopmwm.verify_perseed quote|extra-seeds  (no SUMO; prints the quote).

    ``quote``: the per-decision all8_both_seeds quote. ``extra-seeds`` (BUILD option quote,
    default n=4): the same rows plus the extra-seed sim-s of the cooperative plans."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=('quote', 'extra-seeds'))
    ap.add_argument('--n', type=int, default=4, help='extra CRN seeds (extra-seeds command; default 4)')
    a = ap.parse_args(argv)
    print(json.dumps(standard_quote() if a.cmd == 'quote' else extra_seeds_quote(a.n), indent=1))


if __name__ == '__main__':
    main()
