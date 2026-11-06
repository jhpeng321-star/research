"""Clean4 (wz3to2 lane-drop) budget: per-phase u1-u24 quote, coverage and owner approval.

Pure, zero-physics and fail-closed. Nothing here reserves a charge or launches.

* Every phase is sized for its FULL u1-u24 schedule with >= 15 % margin (MAIN 02:43
  lesson from clean2 stopping at u18).
* Smoke and gate are pre-approved inside phase newmap2 as sub-phase newmap2_smoke,
  capped at 40,000 sim-s (runtime ledger phases track2_smoke + gate_bank_verification_ext).
* Training phases are charged ONLY after MAIN writes BUDGET_CLEAN4.json carrying the
  owner's approval of this exact quote (approved()).
"""
from copy import deepcopy
import math

from .exact_grade import _seal, verify_sealed

SCHEMA = 'coopmwm.clean4.quote.v1'
UPDATES = 24
MARGIN = 0.15
EPISODE_SIM_S = 1200
# Owner/MAIN 14:37: newmap2 <= 300,000 sim-s (probe/mining); smoke+gate pre-approved inside it.
SMOKE_PHASE = 'newmap2_smoke'
SMOKE_PHASE_CAP = 40_000
SMOKE_CAP = 20_000
RUNTIME_SMOKE_PHASES = ('track2_smoke', 'gate_bank_verification_ext')
FINAL_ARMS = ('A', 'B', 'R', 'base', 'native')
TRAIN_RANGE = (989200, 989599)
FINAL_RANGE = (989600, 989699)
FORBIDDEN_RANGES = ((964000, 964199), (962000, 962999), (977000, 977999))
# Training-run phases the runtime charges (names unchanged from clean3 so the existing
# reservation code is reused); interim u8 evaluation is OFF in clean4 (held-out panel
# is read once, at u24).
TRAINING_PHASES = ('v2v_train', 'v2v_lessons', 'early_probes', 'isolated_replicated',
                   'v2v_final', 'g4_synergy_credit')


def allocation(base, *, margin=MARGIN, unit=EPISODE_SIM_S):
    """Smallest multiple of unit >= base*(1+margin)."""
    if type(base) is not int or base < 0:
        raise ValueError('nonnegative integer base cost required')
    if base == 0:
        return 0
    return int(math.ceil(round(base*(1+margin), 6)/unit))*unit


def phase(schedule, base, *, basis, allocated=None, margin=MARGIN):
    allocated = allocation(base, margin=margin) if allocated is None else allocated
    if base and allocated < base*(1+MARGIN)-1e-9:
        raise ValueError('phase allocation below the 15% margin: '+schedule)
    return dict(schedule=schedule, base_sim_s=base, allocated_sim_s=allocated,
                margin_fraction=round(allocated/base-1, 4) if base else None, basis=basis)


def in_range(seed, bounds):
    return bounds[0] <= seed <= bounds[1]


def check_seeds(train, final):
    """TRAIN and final seeds inside their registered wz3to2 ranges, disjoint, never forbidden."""
    train, final = list(train), list(final)
    for seed in train+final:
        if type(seed) is not int or any(in_range(seed, r) for r in FORBIDDEN_RANGES):
            raise ValueError('forbidden or non-integer seed: '+repr(seed))
    if not all(in_range(s, TRAIN_RANGE) for s in train):
        raise ValueError('clean4 training worlds must come from TRAIN 989200-989599')
    if not all(in_range(s, FINAL_RANGE) for s in final):
        raise ValueError('clean4 final evaluation must use the registered 989600-989699 range')
    if set(train) & set(final):
        raise ValueError('TRAIN and final worlds overlap')
    return dict(train=len(set(train)), final=len(set(final)))


def quote(*, episode_sim_s=EPISODE_SIM_S, episodes_per_update=16, practice_refs_per_update=12,
          lesson_sim_s, probe_sim_s, replicated_sim_s, g4_base_sim_s, g4_cap_sim_s,
          final_worlds=100, final_arms=FINAL_ARMS, diagnostic_factor=1.0, basis=None):
    """Full u1-u24 per-phase quote. diagnostic_factor >= 1 inflates the diagnostic bases
    (lessons/probes/replicated) when they are carried over from another map."""
    if diagnostic_factor < 1:
        raise ValueError('diagnostic factor never discounts a measured base')
    basis = basis or {}
    def scaled(value):
        return int(math.ceil(value*diagnostic_factor))
    phases = dict(
        v2v_train_waves=phase('u1-u24 A/B/R training episodes',
            UPDATES*3*episodes_per_update*episode_sim_s,
            basis=basis.get('train', f'3 arms x {episodes_per_update} episodes x {episode_sim_s} s x 24')),
        v2v_practice_refs=phase('u1-u24 B/R practice A-references',
            UPDATES*practice_refs_per_update*episode_sim_s,
            basis=basis.get('refs', f'{practice_refs_per_update} matched A references/update x {episode_sim_s} s')),
        v2v_lessons=phase('u6,u12,u18,u24 B lessons', scaled(lesson_sim_s), basis=basis.get('lessons', 'lessons')),
        early_probes=phase('u2,u4 probes', scaled(probe_sim_s), basis=basis.get('probes', 'probes')),
        isolated_replicated=phase('u6,u12,u24 replicated synergy test (TRAIN states)', scaled(replicated_sim_s),
            basis=basis.get('replicated', 'replicated')),
        g4_synergy_credit=phase('u1-u24 G4 sparse synergy credit B=R', g4_base_sim_s,
            allocated=g4_cap_sim_s, basis=basis.get('g4', 'G4 registered hard cap')),
        v2v_final=phase('u24 paired evaluation on the registered final range, arms '+'/'.join(final_arms),
            len(final_arms)*final_worlds*episode_sim_s,
            basis=basis.get('final', f'{len(final_arms)} arms x {final_worlds} worlds x {episode_sim_s} s')),
    )
    return phases


def runtime_phases(phases):
    """Map quote phases onto the runtime ledger phase names (practice refs charge v2v_train)."""
    p = {k: v['allocated_sim_s'] for k, v in phases.items()}
    return dict(v2v_train=p['v2v_train_waves']+p['v2v_practice_refs'], v2v_lessons=p['v2v_lessons'],
                early_probes=p['early_probes'], isolated_replicated=p['isolated_replicated'],
                v2v_final=p['v2v_final'], v2v_interim=0, g4_synergy_credit=p['g4_synergy_credit'],
                track2_smoke=SMOKE_CAP, gate_bank_verification_ext=SMOKE_PHASE_CAP-SMOKE_CAP,
                teacher_verify=0, group_credit_verify=0, augment_verify=0)


def document(phases, *, prior, cap, created, notes=()):
    """Sealed CLEAN4_QUOTE.json body."""
    total = sum(p['allocated_sim_s'] for p in phases.values())
    prior_total = sum(prior.values())
    available = cap-prior_total
    for name, p in phases.items():
        if p['base_sim_s'] and p['allocated_sim_s'] < p['base_sim_s']*(1+MARGIN)-1e-9:
            raise ValueError('phase below margin: '+name)
    value = dict(schema=SCHEMA, created=created, updates=UPDATES, arms=['A', 'B', 'R'],
        final_arms=list(FINAL_ARMS), margin_min=MARGIN, phases=deepcopy(phases),
        runtime_phases=runtime_phases(phases), total_training_sim_s=total,
        smoke_gate=dict(phase=SMOKE_PHASE, cap_sim_s=SMOKE_PHASE_CAP, smoke_cap_sim_s=SMOKE_CAP,
                        runtime_ledger_phases=list(RUNTIME_SMOKE_PHASES),
                        authority='pre-approved inside phase newmap2 (owner/MAIN 2026-11-06 14:37)'),
        owner_cap_sim_s=cap, prior_registered_sim_s=deepcopy(prior), prior_total_sim_s=prior_total,
        available_sim_s=available, fits=total <= available, shortfall_sim_s=max(0, total-available),
        seeds=dict(train=list(TRAIN_RANGE), final=list(FINAL_RANGE)),
        approval_contract=('MAIN writes E:/agent-work/_sol/BUDGET_CLEAN4.json with owner_approval '
            '(verbatim owner text), total_cap_sim_s, approved_sim_s >= total_training_sim_s and '
            'quote_sha256 == this receipt_sha256; training phases are charged only after it exists'),
        notes=list(notes), new_sim_s=0)
    return _seal(value, 'receipt_sha256')


def coverage(phases, need):
    """Refuse unless every phase allocation covers its full u1-u24 need."""
    short = {k: (v, phases.get(k, 0)) for k, v in need.items() if v > phases.get(k, 0)}
    if short:
        raise ValueError('phase does not cover its u1-u24 schedule: '+repr(sorted(short.items())))
    return dict(need=dict(need), allocated={k: phases[k] for k in need})


def smoke_gate_spent(rows):
    """newmap2_smoke sub-phase spend from a clean4 ledger (smoke + gate)."""
    spent = sum(r['sim_s'] for r in rows if r['phase'] in RUNTIME_SMOKE_PHASES)
    smoke = sum(r['sim_s'] for r in rows if r['phase'] == 'track2_smoke')
    if smoke > SMOKE_CAP or spent > SMOKE_PHASE_CAP:
        raise ValueError('newmap2_smoke cap exceeded (smoke <= 20000, smoke+gate <= 40000)')
    return dict(smoke=smoke, gate=spent-smoke, total=spent)


def approved(budget, quote_doc):
    """Owner approval binding for training (BUDGET_CLEAN4.json written by MAIN)."""
    verify_sealed(quote_doc, 'receipt_sha256')
    text = budget.get('owner_approval')
    if not isinstance(text, str) or not text.strip():
        raise ValueError('BUDGET_CLEAN4.json must quote the owner approval verbatim')
    if budget.get('quote_sha256') != quote_doc['receipt_sha256']:
        raise ValueError('BUDGET_CLEAN4.json must bind this exact CLEAN4 quote')
    amount = budget.get('approved_sim_s')
    if type(amount) is not int or amount < quote_doc['total_training_sim_s']:
        raise ValueError('approved training budget below the full u1-u24 quote')
    cap = budget.get('total_cap_sim_s')
    if type(cap) is not int or cap-quote_doc['prior_total_sim_s'] < quote_doc['total_training_sim_s']:
        raise ValueError('approved owner cap does not cover prior registered phases plus the clean4 quote')
    return dict(owner_approval=text, approved_sim_s=amount, total_cap_sim_s=cap,
                quote_sha256=quote_doc['receipt_sha256'])
