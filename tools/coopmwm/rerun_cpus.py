"""Standalone rerun preparation; opt-in Track 2 production runtime targets cpuV.
with ITS OWN root and ITS OWN single ledger. Supersedes the c4/rerun_clean prep (which kept the
coop3 A as a remote control): here ALL THREE arms A, B and R run from MAPPO u0 with identical
code in ONE new root - B and R with the full fix set (V2V intent, per_seed_v1 cooperative_only
bank, imitation_step_mode kl_targeted, practice_starts 0.25/cap 2, imitation_target_space
'effective'), A the same runtime minus bank/practice (as in coop3) - so the B/R safety
composites read A's results from the SAME new root and no remote A read ever happens.

Preparation commands never dispatch or run SUMO. The explicit run_tracks/smoke boundary
consumes qualified injected CPU integrations (or test-only CPU plants). The new root owns CHARGES.jsonl +
ACCOUNTING.claim (lock) + AUTHORITY.md; it is created only if absent (refused otherwise), and
every charge is refused unless it lands in THIS root's own ledger (never a symlink or another
root's ledger). A read-only staging manifest lists every bank/tape/snapshot/MAPPO-checkpoint
file the run reads - resolved through the REAL bank readers (bank_decisions, the per_seed_v1
confirmations, temperature, practice links) - relative to /home/ubuntu so an operator can
rsync+sha them to cpuS. The dry preflight refuses unless OWNER_APPROVAL.json binds the budget
request SHA, the owner quote SHA and the verdict file SHA, and every manifest file SHA matches
the staged copies. Effective-imitation is pinned to f35b193a.

Track 2 / T2C is explicitly opt-in (track2=True). A stays plain MAPPO PPO; B/R use the frozen
MAPPO cooperation layer, trigger head, group credit, observable merge features, dense
WM/random teaching, augmentation and practice weighting from their own banks. coop_reward
stays OFF. The proposed evaluation is the
replicated isolated cooperation test at
u6/u12/u24: N=5 B/R replicates per update in paired worlds (same world seed across
arms by replicate index), bootstrap CIs against the matched common-base reference, and a
PRE-REGISTERED pass criterion (sealed document, bound by OWNER_APPROVAL.json and the run
root's AUTHORITY.md; the dry preflight refuses without all three bindings). The trigger
head/teachers code against sibling-track modules through explicit injected adapters;
only moves has a fixed symbol API in the shared spec. No sibling imports are needed to
produce a config or budget. The native production dispatcher is rerun_tracks_backend;
tests replace unavailable sibling adapters and the SUMO boundary with CPU fakes.
"""
import hashlib
import json
import math
import re
from copy import deepcopy
from pathlib import Path, PurePosixPath
from .rerun_tracks_bank import OPTIONS as BANK_TRAINING

UPDATES = 24
EPISODE_SIM_S = 1200
EPISODES_PER_UPDATE = 16              # budget-cut allocation; full 1200s episodes retained
PRACTICE_REFS_PER_UPDATE = 6          # one matched A-policy reference per practice episode (.25 of 24)
GATE_SIM_S = 35820                    # cooperative_only exact gate (7 states / 55 branches), measured x5
LESSON_SIM_S_MEASURED = 5280          # B u6 V2V lesson, measured
ISOLATED_SIM_S_MEASURED = 32610       # isolated synergy, measured (one replicate run)
TEST_EPISODES = 96                    # u8 interim + u24 final fixed episode tests, ALL THREE arms
SCHEDULE = dict(A=dict(lessons=[], isolated=[]),
                B=dict(lessons=[6, 12, 18, 24], isolated=[6, 12, 24]),
                R=dict(lessons=[], isolated=[6, 12, 24]))
# ---- Track 2 / T2C: trigger head + teachers + replicated isolated test ------------------
TRIGGER_ARMS = ('B', 'R')             # A stays plain MAPPO PPO: no head, no teacher
GROUP_CREDIT_VERIFY_SIM_S = GATE_SIM_S  # explicit phase ceiling, measured gate-sized allowance
CPU = dict(host='cpuV', cores=48, device='cpu', max_sumo_workers=40,
           workers=24, diagnostic_workers=16, torch_threads=1,
           shared_worker_limit=True)
TRACK_SLOTS = dict(
    coop_layer=dict(enabled=True, arms=['B', 'R'], module='tools.coopmwm.coop_layer',
                    frozen_base=True, zero_init=True, separate_optimizer=True,
                    kl_cap=.045, base_actions='byte-identical MAPPO before residual offsets'),
    group_credit=dict(enabled=False, arms=['B', 'R'], module='tools.coopmwm.group_credit',
                      layer_advantage_only=True, inputs='belief/observables only',
                      verification='existing all8/isolated machinery; sibling registered fraction',
                      verification_phase='group_credit_verify'),
    merge_features=dict(enabled=True, arms=['B', 'R'], module='tools.coopmwm.merge_features',
                        inputs='own sensing + timestamped t-1 V2V reports; missingness flags'),
      dense_teacher=dict(enabled=False, arms=['B', 'R'], module='tools.coopmwm.dense_teacher',
                         verify_fraction=.1, verified_through='all8', candidates=8,
                         calibrated_p_min=.8, calibration_module='tools.coopmwm.planner_calibration',
                         verified_bypass=True, same_confidence_filter=True,
                        distribution='same move candidates and proposal distribution for B/R'),
     plan_augment=dict(enabled=True, arms=['B', 'R'], module='tools.coopmwm.plan_augment',
                        verify_fraction=.1, examples_per_update=24, once_before_launch=True,
                      same_procedure=True, source='each arm own bank; moves/variety registered ranges'),
     practice_weighting=dict(enabled=True, arms=['B', 'R'], same_procedure=True,
                              fraction=.5, sources=['own_bank', 'own_augmented_states']),
    track2_reward=dict(enabled=True, arms=['B', 'R'], module='tools.coopmwm.track2_reward',
        weight=1.0, provenance='T2C proposed equal own/neighbour-mean weighting; MAIN 02:30; approval-bound, no tuning',
        formula='r_i + weight * mean(observable neighbour realised rewards)',
        inputs='real SUMO outcomes; own sensing and causal V2V only'))
# Proposed component-level registrations; approval binds these exact values and costs.
# Verification uses whole all8/isolated checks, never fractions of a truncated window.
TRACK_SLOTS['group_credit'].update(verify_fraction=.1)
for _name in ('dense_teacher', 'group_credit', 'plan_augment'):
    TRACK_SLOTS[_name].update(verification_mode='fixed_total_coordinator_only',
                             internal_sumo_dispatch=False)
# Fixed TOTAL samples, not a fraction of the decision stream. Both arms receive
# exactly one whole all8 check at each of these updates, with no carry/extra work.
VERIFY_UPDATES = (6, 12, 18, 24)
VERIFICATION = dict(updates=list(VERIFY_UPDATES), samples_per_arm_per_update=1,
                    dense_total_sim_s=0, credit_total_sim_s=0,
                    augmentation_samples_per_arm=2, augmentation_total_sim_s=4*GATE_SIM_S)
TOTAL_CAP_SIM_S = 5_000_000
EARLY_PROBES = dict(updates=[2, 4], states=3, arms=['A', 'B', 'R'],
                    sim_s_per_arm=3*ISOLATED_SIM_S_MEASURED)
LEARNING_MONITOR = dict(trigger_saturation=[0., 1.], history_updates=3,
                        accuracy_tolerance=1e-6, layer_kl_near_zero=1e-8,
                        action='DIAGNOSE; pause only for safety')
SMOKE = dict(updates=1, episodes_per_arm=2, episode_sim_s=1200,
             training_sim_s=6*1200, practice_reference_sim_s=2*1200,
             augmented_examples_per_arm=2,
               verification_sim_s=4*GATE_SIM_S, isolated_dry_run_sim_s=0)
SMOKE['total_sim_s'] = sum(SMOKE[k] for k in
    ('training_sim_s', 'practice_reference_sim_s', 'verification_sim_s', 'isolated_dry_run_sim_s'))
ISOLATED_REPLICATES = 5
ISOLATED_BOOTSTRAP_REPS = 10000
ISOLATED_BOOTSTRAP_LEVEL = 0.95
ISOLATED_BOOTSTRAP_SEED = 20261104    # fixed seed: the CIs are reproducible before any run
ISOLATED_TEST = dict(
    updates=[6, 12, 24], arms=['B', 'R'], replicates=ISOLATED_REPLICATES,
    sim_s_per_replicate=ISOLATED_SIM_S_MEASURED,
    pairing='same world seed across arms (replicate index)',
    metric='policy_cooperation_gain_veh_s', higher_is_better=True,
    decomposition='delta(all group new, background base) - max delta(one member new, background base)',
    state_selection='same pre-registered state/group/window and world seed for every arm',
    seed_source='explicit observable TRAIN-world registration; never infer seeds from results',
    bootstrap=dict(reps=ISOLATED_BOOTSTRAP_REPS, level=ISOLATED_BOOTSTRAP_LEVEL,
                   seed=ISOLATED_BOOTSTRAP_SEED, method='percentile'),
    cell_seed_rule='seed = bootstrap.seed*10000 + 100*update + pair_index (pair order B-A, R-A, B-R)',
    safety_counters=['collisions', 'teleports', 'hard_brakes', 'ttc_events'])
FINAL_EVALUATION = dict(module='tools.coopmwm.planner_eval', update=24,
    arms=['A', 'B', 'R'], comparisons=['B-A', 'R-A', 'B-R'],
    episodes_per_arm=TEST_EPISODES, horizon_s=EPISODE_SIM_S,
    metric='J_veh_s', higher_is_better=False,
    bootstrap=dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'),
    safety_counters=list(ISOLATED_TEST['safety_counters']),
    phase='v2v_final', sim_s=3*TEST_EPISODES*EPISODE_SIM_S,
    checkpoint_components=dict(A=['base', 'optimizer'], B=['base', 'layer', 'trigger', 'optimizer'],
                               R=['base', 'layer', 'trigger', 'optimizer']))
CRITERION = dict(
    schema='coopmwm.rerun_cpus.isolated_criterion.v1',
    metric=ISOLATED_TEST['metric'], higher_is_better=True,
    updates=list(ISOLATED_TEST['updates']), replicates=ISOLATED_REPLICATES,
    pairing=ISOLATED_TEST['pairing'],
    primary_update=24, secondary_updates=[6, 12],
    bootstrap=dict(ISOLATED_TEST['bootstrap']),
    pass_rule=('PASS iff, at the primary update u24, the paired B-vs-common-u0 mean cooperation gain '
                'difference vs common frozen u0 (its cooperation gain is identically zero) '
                'has a two-sided 95% bootstrap CI lying entirely above 0 AND B has '
                'no more total collisions/teleports/hard-brakes/TTC events than A across the replicates '
                '(tolerance 0, matched all-base reference). Secondary updates u6/u12 and the B-vs-R bank ablation '
               'are reported, never gated.'),
    safety_counters=list(ISOLATED_TEST['safety_counters']), safety_tolerance=0,
    r_arm=dict(kind='random_bank', gated=False,
     rule='R uses its own random bank (same move distribution); real local outcome reward; '
                    'reported, never gated'),
    registered='proposed T2C registration; effective only after SHA-bound OWNER_APPROVAL.json')
COOP_REWARD = dict(enabled=False, module='tools.coopmwm.coop_reward', weight=0.0,
                   built_on='c4/coop-reward', off_is_byte_identical=True,
                   note='T2C leaves the slot only; the shaper is built on its own branch and '
                        'must stay OFF (byte-identical reward path) in this registered run')
PINS = dict(v2v=True, v2v_dim=48, v2v_intent=True, v2v_intent_horizon=5, v2v_intent_loss_weight=1.0,
            v2v_intent_detach_trunk=True, bank_imitation='separate_full_pass', imitation_kl_cap=.045,
            bank_statistics='per_seed_v1', imitation_bank_filter='cooperative_only',
            imitation_step_mode='kl_targeted', v2v_intent_bank_term=True, practice_starts=True,
            practice_fraction=.25, practice_repeat_cap=2, imitation_target_space='effective',
            stop_after_update=UPDATES, A_stop_after_update=UPDATES, coop_reward=COOP_REWARD)
EFFECTIVE_IMITATION_COMMIT = 'f35b193aa539c7ddec1bb20acaa1f6b13ffb3d31'
HOME = '/home/ubuntu'                 # every staged path is listed relative to this prefix
CHECKPOINT_000_SOURCE = 'B_checkpoint_000.pt'      # the common MAPPO u0 in the original run
ISOLATED_REFERENCE_CHECKPOINTS = ('B_checkpoint_003.pt', 'R_checkpoint_003.pt')  # fresh legacy-u3 refs
RULE_FILE = 'SAFETY_RULE_0957.json'   # composite MAIN09:57 rule, copied into the new root
FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth')
_TRUTH = re.compile(r'(vtype|driver|vehicle_params)[^/]*truth|truth[^/]*(vtype|driver|vehicle_params)',
                    re.IGNORECASE)


def quote(*, track2=False):
    """Assert the complete budget-cut quote, including smoke and all phase ceilings.

    A/B/R retain full 1200s episodes; B/R verification is a fixed whole-run sample
    slate, augmentation is once, and isolated checks are B/R u6/u12/u24 N=5.
    """
    if not track2:
        from .rerun_cpus_compat import quote as legacy_quote
        return legacy_quote()
    a_train = UPDATES*EPISODES_PER_UPDATE*EPISODE_SIM_S
    practice_refs = math.ceil(EPISODES_PER_UPDATE*TRACK_SLOTS['practice_weighting']['fraction'])
    br_train = 2*UPDATES*(EPISODES_PER_UPDATE+practice_refs)*EPISODE_SIM_S
    lessons = len(SCHEDULE['B']['lessons'])*LESSON_SIM_S_MEASURED
    decisions = 2*UPDATES*EPISODES_PER_UPDATE*(EPISODE_SIM_S//60)
    teacher_verify = VERIFICATION['dense_total_sim_s']
    augment_verify = VERIFICATION['augmentation_total_sim_s']
    credit_verify = VERIFICATION['credit_total_sim_s']
    probes = len(EARLY_PROBES['updates'])*len(EARLY_PROBES['arms'])*EARLY_PROBES['sim_s_per_arm']
    cells = len(ISOLATED_TEST['arms'])*len(ISOLATED_TEST['updates'])
    isolated = cells*ISOLATED_TEST['replicates']*ISOLATED_SIM_S_MEASURED
    test_per_arm = TEST_EPISODES*EPISODE_SIM_S
    out = dict(a_train_sim_s=a_train, b_train_sim_s=br_train//2, r_train_sim_s=br_train//2,
               v2v_train=a_train+br_train, v2v_lessons=lessons,
               teacher_verify_sim_s=teacher_verify,
                group_credit_verify_sim_s=credit_verify,
                augment_verify_sim_s=augment_verify, early_probes_sim_s=probes,
                smoke_sim_s=SMOKE['total_sim_s'], verification_decisions=decisions,
               isolated_replicated_sim_s=isolated,
               isolated_replicated=dict(arms=list(ISOLATED_TEST['arms']),
                                        updates=list(ISOLATED_TEST['updates']),
                                        replicates=ISOLATED_TEST['replicates'], cells=cells,
                                        replicates_total=cells*ISOLATED_TEST['replicates'],
                                        per_replicate_sim_s=ISOLATED_SIM_S_MEASURED),
               gate_bank_verification_ext=GATE_SIM_S,
               interim_u8_sim_s=3*test_per_arm, final_u24_sim_s=3*test_per_arm,
               interim_u8=dict(per_arm=test_per_arm, arms=3, updates=8, episodes_per_arm=TEST_EPISODES),
                final_u24=dict(per_arm=test_per_arm, arms=3, updates=24, episodes_per_arm=TEST_EPISODES,
                               harness=FINAL_EVALUATION['module'], shared_worlds=True,
                               bootstrap=deepcopy(FINAL_EVALUATION['bootstrap'])))
    out['total_without_tests'] = (a_train+br_train+lessons+teacher_verify+isolated+GATE_SIM_S
                                  + credit_verify + augment_verify + probes + SMOKE['total_sim_s'])
    out['cost_basis'] = dict(isolated='B/R only u6/u12/u24 N=5; measured whole-replicate ceiling',
                             group_credit='OFF per MAIN 02:30; no verification work',
                               dense_teacher='OFF per MAIN 02:30; no verification work',
                               augmentation='ONCE before launch, two whole all8 samples per arm',
                              early_probes='two updates x three arms x three full-window state checks (conservative ceiling)',
                              wm_compute='CPU only; no extra SUMO')
    out['total'] = out['total_without_tests']+out['interim_u8_sim_s']+out['final_u24_sim_s']+720
    out['lesson_contingency_sim_s'] = 720
    out['verification'] = deepcopy(VERIFICATION)
    out['budget_cap_sim_s'] = TOTAL_CAP_SIM_S
    out['includes_smoke'] = True
    assert teacher_verify <= 300_000 and credit_verify <= 300_000
    assert augment_verify <= 200_000
    assert out['total'] <= TOTAL_CAP_SIM_S, 'total INCLUDING smoke exceeds owner cap'
    out['cap_asserted'] = True
    return out


def budget(*, track2=False):
    """The standalone experiment's own registered budget (one ledger, one total): the phase
    allocations cover the quote with the registered v2v_lessons floor (21,840) kept and the
    replicated isolated test carried by its own phase (it supersedes the single-run
    v2v_isolated plan of the original rerun)."""
    if not track2:
        from .rerun_cpus_compat import budget as legacy_budget
        return legacy_budget()
    q = quote(track2=True)
    phases = dict(v2v_train=q['v2v_train'], v2v_lessons=21840, teacher_verify=q['teacher_verify_sim_s'],
                   group_credit_verify=q['group_credit_verify_sim_s'],
                   augment_verify=q['augment_verify_sim_s'], early_probes=q['early_probes_sim_s'],
                   track2_smoke=q['smoke_sim_s'],
                  isolated_replicated=q['isolated_replicated_sim_s'],
                  gate_bank_verification_ext=GATE_SIM_S, v2v_interim=q['interim_u8_sim_s'],
                  v2v_final=q['final_u24_sim_s'])
    total = sum(phases.values())
    assert total == q['total'] and total <= TOTAL_CAP_SIM_S
    return dict(total=total, phases=phases, allocated=total, unallocated=0, updates=UPDATES,
                episodes_per_update=EPISODES_PER_UPDATE, arms=['A', 'B', 'R'],
                isolated_test='replicated B/R at u6/12/24, N=%d, bootstrap CIs, pre-registered '
                              'criterion (supersedes the single-run v2v_isolated plan)'
                              % ISOLATED_REPLICATES,
                quote_total_sim_s=q['total'], single_ledger=True,
                rule='rerun-cpus standalone: own root, own CHARGES.jsonl; never the coop3 ledger')


def wall_clock(pace_s=1800, measured_update_wall_s=1800, diagnostic_extra_s=900,
               test_episode_wall_s=300, test_workers=None, *, track2=False):
    """Paced updates (measured u7-u10 walls ~1,800 s each per arm, A the cheapest); B's
    diagnostics add ~diagnostic_extra_s each; R trails B by ~one update; the fixed-episode
    tests run 3 arms; the replicated isolated test runs at the measured update pace
    (36,000 sim-s per 1,800 s wall => 20 sim-s per wall-s)."""
    if not track2:
        from .rerun_cpus_compat import wall_clock as legacy_wall
        return legacy_wall(pace_s, measured_update_wall_s, diagnostic_extra_s,
                           test_episode_wall_s, 2 if test_workers is None else test_workers)
    test_workers = CPU['workers'] if test_workers is None else test_workers
    per_update = max(pace_s, measured_update_wall_s)
    b = UPDATES*per_update+len(SCHEDULE['B']['isolated'])*diagnostic_extra_s
    r = b+per_update
    tests = 2*3*TEST_EPISODES*test_episode_wall_s/test_workers      # u8 + u24, three arms
    q = quote(track2=True)
    isolated_s = q['isolated_replicated_sim_s']/20
    verification_s = (q['teacher_verify_sim_s']+q['group_credit_verify_sim_s']+
                      q['augment_verify_sim_s']+q['early_probes_sim_s']+q['smoke_sim_s'])/20
    return dict(estimate_only=True, target=deepcopy(CPU),
                per_update_s=per_update, A_s=UPDATES*per_update, B_s=b, R_s=r, tests_s=tests,
                isolated_s=isolated_s, verification_s=verification_s,
                total_h=round((r+isolated_s+verification_s+tests)/3600, 1))


# ---------------------------------------------------------------- own single ledger

AUTHORITY_LINE = ('S-COOPMWM RERUN-CPUS 2026-11-04: NEW STANDALONE A/B/R-v2v EXPERIMENT. This '
                  'root owns ONE SINGLE LEDGER (CHARGES.jsonl + ACCOUNTING.claim lock, real files '
                  'in THIS directory, never a symlink into another root) and its own registered '
                  'budget; every charge must land HERE. All three arms A/B/R run from the common '
                  'MAPPO u0 with identical code; B/R safety composites read A results from this '
                  'same root. No remote A reads, no shared accounting with any legacy root.\n')


def init_root(root, *, amendment_shas=(), track2=False):
    """Create the NEW standalone run root with its own ledger; refuses if the root exists.
    With explicit Track 2 opt-in, AUTHORITY.md also binds the isolated criterion SHA."""
    root = Path(guard_path(root))
    guard_path(root.resolve())
    if root.exists() or root.is_symlink():
        raise ValueError('REFUSED: run root already exists: '+str(root))
    root.mkdir(parents=True)
    for name in ('CHARGES.jsonl', 'ACCOUNTING.claim'):
        with (root/name).open('x'):
            pass
    text = AUTHORITY_LINE+'effective-imitation commit '+EFFECTIVE_IMITATION_COMMIT+'\n'
    if track2:
        text += 'isolated-test criterion '+criterion_document()['criterion_sha256']+'\n'
    for amendment in amendment_shas:
        text += 'amendmentSHA'+amendment+'\n'
    with (root/'AUTHORITY.md').open('x', encoding='utf-8') as f:
        f.write(text)
    return own_ledger(root, track2=track2)


def own_ledger(root, *, track2=False):
    """The OWN single-ledger binding: CHARGES.jsonl, ACCOUNTING.claim and AUTHORITY.md must be
    REAL files inside this root (a symlink or a path resolving elsewhere is another root's
    ledger and is refused), and AUTHORITY.md must carry the standalone rule."""
    guard_path(root)
    root_resolved = Path(guard_path(Path(root).resolve()))
    files = {}
    for name in ('CHARGES.jsonl', 'ACCOUNTING.claim', 'AUTHORITY.md'):
        p = Path(guard_path(Path(root)/name))
        if p.is_symlink() or not p.is_file():
            raise ValueError('REFUSED: not this root\'s own single ledger: '+str(p))
        if p.resolve().parent != root_resolved:
            raise ValueError('REFUSED: ledger file resolves outside the run root: '+str(p))
        if p.stat().st_nlink > 1:
            raise ValueError('REFUSED: ledger file shared with another directory entry '
                             '(another root\'s ledger): '+str(p))
        files[name] = p
    text = files['AUTHORITY.md'].read_text(encoding='utf-8')
    if 'SINGLE LEDGER' not in text:
        raise ValueError('REFUSED: AUTHORITY.md is not the standalone single-ledger rule: '+str(root))
    if EFFECTIVE_IMITATION_COMMIT not in text:
        raise ValueError('REFUSED: AUTHORITY.md does not pin the effective-imitation commit')
    if track2:
        expected = 'isolated-test criterion '+criterion_document()['criterion_sha256']
        if expected not in text:
            raise ValueError('REFUSED: AUTHORITY.md does not bind the pre-registered '
                             'isolated-test criterion ('+expected+')')
    return dict(root=str(root_resolved), **files, foreign=False)


def bind_ledger(root, cfg):
    """Standalone accounting hook (replaces v2v_live.accounting): binds accel_storage's reserve
    to THIS root's own ledger and the standalone budget; refuses any foreign ledger first."""
    track2 = cfg.get('track2', False)
    own_ledger(root, track2=track2)
    import types
    from tools.coopnative import storage
    from . import accel_storage
    from .accel_protocol import SUPERSEDED_CHARGE
    b = cfg['budget']
    accel_storage._reserve = types.FunctionType(storage.reserve.__code__,
        dict(storage.reserve.__globals__,
             CAP=b['total'] if track2 else b['total']-SUPERSEDED_CHARGE, CAPS=b['phases']))
    return dict(root=str(root), total=b['total'], phases=dict(b['phases']),
                rule='own single ledger of this root only; no other root\'s ledger accepted')


def charge(root, jobs, phase, *, budget, track2=False):
    """Reserve jobs against THIS root's own ledger (refuses any foreign ledger)."""
    bind_ledger(root, dict(budget=budget, track2=track2))
    from . import accel_storage
    return accel_storage.reserve(root, jobs, phase)


# ---------------------------------------------------------------- 3-arm runtime config

def head_teacher_config():
    """Prospective sibling bindings. Do not invent HEAD_ACTIONS/TEACHER_KINDS APIs:
    action buckets and teacher constructors belong to T2A/T2B. B uses WM labels and R
    uses random labels over the same proposals, both with registered verification."""
    head = dict(enabled=True, module='tools.coopmwm.trigger_policy', arms=list(TRIGGER_ARMS), v2v_dim=48,
                moves_module='tools.coopmwm.moves', families_source='moves.FAMILIES',
                candidates_api='moves.candidates(obs_state, rng, n)',
                bank_label_api='moves.from_plan(verified_plan)',
                decision_interval_s=60,
                low_level='frozen MAPPO V2V actor (same v2v_dim); executes '
                          'moves.offsets on top of the base actions (single writer, HDVs '
                          'never commanded)',
                update='PPO update for the discrete head only')
    teachers = dict(
        A=None,
        B=dict(enabled=False, module='tools.coopmwm.dense_teacher', kind='wm_dense',
               mode='DAgger: the WM planner (T1B scoring, imported) labels the best move per '
                    'state at decision points; cross-entropy imitation for the T2A head',
                verify_fraction=TRACK_SLOTS['dense_teacher']['verify_fraction'], verified_through='all8'),
        R=dict(enabled=False, module='tools.coopmwm.dense_teacher', kind='random',
               verify_fraction=TRACK_SLOTS['dense_teacher']['verify_fraction'], verified_through='all8',
               distribution='same move distribution as the WM teacher proposals (fairness)'))
    return head, teachers


def trigger_head_enabled(cfg, arm):
    """The T2A head applies to B and R only; A is plain MAPPO PPO."""
    head = cfg.get('trigger_head') or {}
    return head.get('enabled') is True and arm in head.get('arms', ())


def teacher_kind(cfg, arm):
    """'wm_dense' (B), 'random' (R) or None (A) from the teacher config slot."""
    t = (cfg.get('teacher') or {}).get(arm)
    return t.get('kind') if t and t.get('enabled') is True else None


def track2_components(cfg, arm, *, adapters):
    """Build qualified components via injected ``adapter(cfg=, arm=, moves=, components=)``.

    These are T2C's adapter arguments, not guessed exports from sibling modules. The
    integration supplies real adapters after merging those modules. A returns no layer
    components and never imports them. OFF slots never import or call a factory. The
    explicit move API is checked here, at integration time, rather than config time.
    """
    check_pins(cfg)
    if arm not in cfg['active_arms']:
        raise ValueError('REFUSED: unknown Track 2 arm')
    if arm == 'A' or not cfg.get('track2'):
        return {}
    names = component_names(cfg)
    missing = [n for n in names if not callable(adapters.get(n))]
    if missing:
        raise ValueError('REFUSED: Track 2 adapters missing: '+', '.join(missing))
    from . import moves
    if 'none' not in moves.FAMILIES:
        raise ValueError('REFUSED: the move library must offer the none family')
    if any(not callable(getattr(moves, n, None)) for n in ('candidates', 'offsets', 'from_plan')):
        raise ValueError('REFUSED: moves API needs candidates/offsets/from_plan')
    components = {}
    for name in names:
        components[name] = adapters[name](cfg=deepcopy(cfg), arm=arm, moves=moves,
                                          components=dict(components))
        if components[name] is None:
            raise ValueError('REFUSED: Track 2 adapter returned no component: '+name)
    return components


def component_names(cfg):
    return [n for n in ('merge_features', 'trigger_head', 'coop_layer', 'group_credit',
                       'dense_teacher', 'track2_reward', 'plan_augment', 'practice_weighting')
            if cfg.get(n, {}).get('enabled') is True]


def episode_options(cfg, arm):
    """Serializable worker extras for the qualified Track 2 episode integration.

    The old V2V worker does not consume these automatically; the integration must
    install the cooperation-layer controller/optimizer before admitting live work.
    """
    check_pins(cfg)
    if arm not in cfg['active_arms']:
        raise ValueError('REFUSED: unknown Track 2 arm')
    if arm == 'A' or not cfg.get('track2'):
        return {}                     # plain MAPPO's existing extras stay byte-identical
    keys = ('trigger_head', 'coop_layer', 'group_credit', 'merge_features',
            'dense_teacher', 'track2_reward', 'plan_augment', 'practice_weighting', 'coop_reward')
    return dict(track2={k: deepcopy(cfg[k]) for k in keys},
                track2_teacher=deepcopy(cfg['teacher'][arm]))


def runtime_config(base, *, root, reference_root, source_root, common_checkpoint_sha256, track2=False):
    """Registered producer of the standalone runtime: v2v_runner.prepare_intent with every B/R
    fix pinned from u0 (kl_targeted, practice_starts 0.25/cap 2, imitation_target_space
    'effective'), the 24-update cap, and the standalone bindings: accounting_root = run_root =
    the NEW root (its own ledger), legacy_root = the staged legacy reference dir (only the
    fresh legacy-u3 isolated reference checkpoints live there), control arm A in the SAME new
    root (no remote A reads), own budget. Track 2 clean profile: B/R trigger, frozen
    layer, credit, features, teaching, augmentation and weighting; extra shaper OFF.
    Diagnostic scheduling supersedes the v2v_runner default. The 1200 s episode horizon,
    intent horizon and caller's planner/search horizons are preserved (no speedup by truncation).
    This config requires qualified Track 2 adapters before any live run."""
    if not track2:
        from .rerun_cpus_compat import runtime_config as legacy_config
        return legacy_config(base, root=root, reference_root=reference_root,
                             source_root=source_root, common_checkpoint_sha256=common_checkpoint_sha256)
    from . import v2v_runner
    root, reference_root, source_root = (v2v_runner.remote_name(p) for p in
                                         (root, reference_root, source_root))
    if root == reference_root:
        raise ValueError('the run root must not be the staged legacy reference root')
    if root == source_root:
        raise ValueError('the run root must not be the deployed source root')
    if not re.fullmatch('[0-9a-f]{64}', common_checkpoint_sha256):
        raise ValueError('common original MAPPO u0 checkpoint hash required')
    base = dict(base)
    base['budget'] = budget(track2=True)
    base['track2'] = True
    cfg = v2v_runner.prepare_intent(base, root=root, legacy_root=reference_root,
        source_root=source_root, common_checkpoint_sha256=common_checkpoint_sha256,
        imitation_step_mode='kl_targeted', practice_starts=True, practice_fraction=.25,
        practice_repeat_cap=2, imitation_target_space='effective')
    head, teachers = head_teacher_config()
    cfg.update(run_root=root, legacy_root=reference_root, accounting_root=root,
               bank_source_root=root, source_root=source_root, standalone=True, rerun_cpus=True,
               rerun_arms=['A', 'B', 'R'],
               control_arm=dict(arm='A', root=root, update=UPDATES),
               effective_imitation_commit=EFFECTIVE_IMITATION_COMMIT,
               trigger_head=head, teacher=teachers, coop_reward=dict(COOP_REWARD),
                isolated_test=deepcopy(ISOLATED_TEST),
                final_evaluation=deepcopy(FINAL_EVALUATION),
                bank_training=deepcopy(BANK_TRAINING),
               criterion_sha256=criterion_document()['criterion_sha256'],
               diagnostic_schedule=deepcopy(SCHEDULE), # replicated isolated test, all arms
                diagnostic_phases=dict(lessons='v2v_lessons', isolated='isolated_replicated',
                                       probes='early_probes'),
                early_probes=deepcopy(EARLY_PROBES), learning_monitor=deepcopy(LEARNING_MONITOR),
                smoke=deepcopy(SMOKE), track2=True,
               single_ledger=dict(root=root, charges='CHARGES.jsonl', lock='ACCOUNTING.claim',
                                  authority='AUTHORITY.md', foreign_refused=True),
                budget=budget(track2=True),
               a_results=dict(root=root, rule='B/R safety composites read A_UPDATE_<u> and the '
                               'same-world A references from THIS root; no remote A reads'))
    cfg.update(deepcopy(TRACK_SLOTS))
    cfg.update(target_host='cpuV', device='cpu', cpu=deepcopy(CPU), workers=CPU['workers'],
                verification=deepcopy(VERIFICATION),
                job_namespace='t2c', launch_profile='track2_clean_run',
                integration_required=True, additional_cap_ceiling_sim_s=budget(track2=True)['total'],
                total_cap_ceiling_sim_s=budget(track2=True)['total'])
    check_pins(cfg)
    return cfg


def check_pins(cfg):
    """Pins for all three arms (identical code; A differs only by having no head/teacher, bank
    or practice) plus the standalone single-ledger and same-root A bindings and the Track-2
    trigger-head/teacher/isolated-test/coop_reward-slot registrations."""
    if not cfg.get('track2'):
        from .rerun_cpus_compat import check_pins as legacy_check
        return legacy_check(cfg)
    bad = {k: cfg.get(k) for k, v in PINS.items() if cfg.get(k) != v}
    if bad:
        raise ValueError('rerun-cpus pin mismatch: '+json.dumps(bad, sort_keys=True))
    if cfg.get('accounting_root') != cfg.get('run_root'):
        raise ValueError('REFUSED: the standalone experiment must account in its OWN root')
    if cfg.get('legacy_root') == cfg.get('run_root') or cfg.get('source_root') == cfg.get('run_root'):
        raise ValueError('REFUSED: the run root must not double as the reference/source root')
    if cfg.get('control_arm') != dict(arm='A', root=cfg['run_root'], update=UPDATES):
        raise ValueError('REFUSED: control arm A must run in the SAME new root (no remote A reads)')
    if list(cfg.get('rerun_arms', ())) != ['A', 'B', 'R']:
        raise ValueError('REFUSED: all three arms A/B/R rerun from MAPPO u0')
    if cfg.get('effective_imitation_commit') != EFFECTIVE_IMITATION_COMMIT:
        raise ValueError('REFUSED: effective-imitation commit not pinned to '+EFFECTIVE_IMITATION_COMMIT)
    if cfg.get('budget') != budget(track2=True):
        raise ValueError('REFUSED: standalone budget differs from the registered 3-arm quote')
    if (cfg.get('active_arms') != ['A', 'B', 'R']
            or cfg.get('final_updates') != dict(A=24, B=24, R=24)
            or not re.fullmatch('[0-9a-f]{64}', cfg.get('common_checkpoint_sha256', ''))
            or cfg.get('start_checkpoint_sha256') !=
            {a: cfg['common_checkpoint_sha256'] for a in ('A', 'B', 'R')}):
        raise ValueError('REFUSED: all three arms need the common MAPPO u0 and 24 updates')
    if cfg.get('bank_arms') != dict(B='B', R='R') or cfg.get('a_results', {}).get('root') != cfg['run_root']:
        raise ValueError('REFUSED: own B/R banks and same-root A results required')
    if (cfg.get('cpu') != CPU or cfg.get('target_host') != 'cpuV'
            or cfg.get('workers') != CPU['workers'] or cfg.get('device') != 'cpu'):
        raise ValueError('REFUSED: cpuV CPU-only shared 40-worker registration differs')
    if cfg.get('verification') != VERIFICATION:
        raise ValueError('REFUSED: fixed-total verification registration differs')
    if cfg.get('launch_profile') != 'track2_clean_run':
        raise ValueError('REFUSED: clean-run profile required')
    for key, slot in TRACK_SLOTS.items():
        if cfg.get(key) != slot:
            raise ValueError('REFUSED: clean-run slot differs: '+key)
    head = cfg.get('trigger_head') or {}
    if tuple(head.get('arms', ())) != TRIGGER_ARMS or head.get('enabled') is not True:
        raise ValueError('REFUSED: the trigger head applies to B and R only; A stays plain MAPPO PPO')
    if head.get('v2v_dim') != cfg.get('v2v_dim') or head.get('decision_interval_s') != 60:
        raise ValueError('REFUSED: the trigger head must reuse the V2V obs width and the '
                         'registered 60 s decision cadence')
    teachers = cfg.get('teacher') or {}
    if (teachers.get('A') is not None or teachers.get('B', {}).get('kind') != 'wm_dense'
            or teachers.get('R', {}).get('kind') != 'random'):
        raise ValueError('REFUSED: teachers must be A=none, B=wm_dense (WM DAgger), R=random')
    if teachers.get('B', {}).get('verify_fraction') != TRACK_SLOTS['dense_teacher']['verify_fraction'] \
            or teachers.get('B', {}).get('verified_through') != 'all8':
        raise ValueError('REFUSED: B dense teacher requires registered all8 machinery fraction')
    if not teachers.get('R', {}).get('distribution'):
        raise ValueError('REFUSED: R random teacher must reuse the SAME move distribution '
                         '(fairness)')
    if head != head_teacher_config()[0] or teachers != head_teacher_config()[1]:
        raise ValueError('REFUSED: trigger/teacher slots differ from clean-run registration')
    for key, value in (('early_probes', EARLY_PROBES), ('learning_monitor', LEARNING_MONITOR),
                        ('smoke', SMOKE), ('final_evaluation', FINAL_EVALUATION),
                        ('bank_training', BANK_TRAINING)):
        if cfg.get(key) != value:
            raise ValueError('REFUSED: registration differs: '+key)
    if cfg.get('isolated_test') != ISOLATED_TEST:
        raise ValueError('REFUSED: replicated isolated-test registration differs from the '
                         'registered u6/12/24 N=%d paired-worlds spec' % ISOLATED_REPLICATES)
    if cfg.get('criterion_sha256') != criterion_document()['criterion_sha256']:
        raise ValueError('REFUSED: config does not bind the current pre-registered criterion')
    for arm in ISOLATED_TEST['arms']:
        if list((cfg.get('diagnostic_schedule') or {}).get(arm, {}).get('isolated', ())) \
                != list(ISOLATED_TEST['updates']):
            raise ValueError('REFUSED: arm '+arm+' must run the replicated isolated test at '
                             'u6/12/24')
    if (cfg.get('diagnostic_phases') or {}).get('isolated') != 'isolated_replicated':
        raise ValueError('REFUSED: isolated diagnostics must charge the isolated_replicated phase')
    from .practice_starts import enabled
    if enabled(cfg, 'A') or not (enabled(cfg, 'B') and enabled(cfg, 'R')):
        raise ValueError('REFUSED: practice_starts applies to B and R only; A stays plain PPO')
    if cfg.get('diagnostic_schedule') != SCHEDULE:
        raise ValueError('REFUSED: diagnostic schedule differs from the registered plan')
    return True


# ---------------------------------------------------------------- read-only staging manifest

def guard_path(path):
    """HARD RULES: never open sealed/ or any path containing holdout/heldout/cal_split/chunk5/
    j4b/inc1; never read hetdata_truth or vType/driver/vehicle_params truth files."""
    s = str(path).replace('\\', '/').lower()
    hit = [t for t in FORBIDDEN if t in s]
    if hit:
        raise ValueError('REFUSED: forbidden path token '+repr(hit)+': '+str(path))
    if _TRUTH.search(s):
        raise ValueError('REFUSED: truth file path refused: '+str(path))
    return path


def sha(path):
    guard_path(path)
    guard_path(Path(path).resolve())
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


class _ReadRecorder:
    """Records every file the REAL bank readers open under the staged roots (read-only)."""

    def __init__(self, roots, home=None):
        self.roots = [Path(r).resolve() for r in roots]
        self.home = Path(home).resolve() if home is not None else None
        self.discovered = set()
        self.paths = set()
        self.lexical = set()
        self._saved = []

    def _under(self, path):
        try:
            resolved = Path(path).resolve()
        except (OSError, ValueError):
            return None
        for root in self.roots:
            if resolved == root or root in resolved.parents:
                return resolved
        return None

    def _record(self, path, mode=None):
        if isinstance(path, int):
            return path
        guard_path(path)                         # lexical guard BEFORE resolution/open
        # LS envelopes retain absolute deployment episode paths. Resolve them in
        # the mirror, exactly as deployment will, before the real reader opens.
        remote = str(path).replace('\\', '/')
        if self.home is not None and remote.startswith('/home/ubuntu/'):
            path = self.home.joinpath(*PurePosixPath(remote).parts[3:])
        resolved = Path(guard_path(Path(path).resolve()))
        mode = str(mode or 'r')
        if set(mode)-set('rb'):              # any write/create/append mode is never a read
            return path
        found = self._under(path)
        if self.home is not None and self.home in resolved.parents:
            found = resolved
            target_root = self.home/resolved.relative_to(self.home).parts[0]
            self.discovered.add(target_root)
        elif self.home is not None and any(r == Path(path).absolute() or
                r in Path(path).absolute().parents for r in self.roots):
            raise ValueError('REFUSED: reader link target outside staging home: '+str(resolved))
        elif self.home is not None and resolved.suffix.lower() in ('.json', '.jsonl', '.gz', '.pt'):
            raise ValueError('REFUSED: reader data outside staging home: '+str(resolved))
        if found is not None:
            guard_path(found)
            self.paths.add(found)
            self.lexical.add(Path(path).absolute())
        return path

    def __enter__(self):
        import builtins
        import gzip
        import pathlib
        def bind(original, record):
            def wrapper(path, *a, **k):
                path = record(path, k.get('mode') or (a[0] if a else None))
                return original(path, *a, **k)
            return wrapper
        self._saved.append((pathlib.Path, 'read_text', pathlib.Path.read_text))
        self._saved.append((pathlib.Path, 'read_bytes', pathlib.Path.read_bytes))
        self._saved.append((pathlib.Path, 'open', pathlib.Path.open))
        self._saved.append((builtins, 'open', builtins.open))
        self._saved.append((gzip, 'open', gzip.open))
        pathlib.Path.read_text = bind(pathlib.Path.read_text, lambda p, m=None: self._record(p, 'r'))
        pathlib.Path.read_bytes = bind(pathlib.Path.read_bytes, lambda p, m=None: self._record(p, 'r'))
        pathlib.Path.open = bind(pathlib.Path.open, self._record)
        builtins.open = bind(builtins.open, self._record)
        gzip.open = bind(gzip.open, self._record)
        return self

    def __exit__(self, *exc):
        import builtins
        import gzip
        import pathlib
        for owner, name, original in reversed(self._saved):
            setattr(owner, name, original)
        return False


def _local_of(remote_root):
    """The local mirror of a /home/ubuntu path: the mirror/stage directory is laid out like a
    filesystem root, so /home/ubuntu/<name> reads <mirror>/home/ubuntu/<name>."""
    guard_path(remote_root)
    p = PurePosixPath(str(remote_root).replace('\\', '/'))
    if not p.is_absolute() or p.parts[1:3] != ('home', 'ubuntu') or len(p.parts) < 4 or '..' in p.parts:
        raise ValueError('REFUSED: staged roots must live under '+HOME+': '+str(remote_root))
    local = _mirror[0].joinpath('home', 'ubuntu', *p.parts[3:])
    guard_path(local)
    guard_path(local.resolve())
    return local


def _rel(remote_root, resolved):
    """Manifest key: the file's path relative to /home/ubuntu (posix, no leading slash)."""
    base = PurePosixPath(str(remote_root).replace('\\', '/'))
    rel = resolved.relative_to(_local_of(remote_root))
    out = PurePosixPath(*base.parts[1:], *rel.parts)        # drop the leading '/'
    if not str(out).startswith('home/ubuntu/'):
        raise ValueError('REFUSED: staged path is not relative to '+HOME+': '+str(out))
    return str(out)


_mirror = [None]


def staging_manifest(*, remote_root, sources=(), training_seed=20261101):
    """READ-ONLY staging manifest of every bank/tape/snapshot/MAPPO-checkpoint file the new run
    reads, resolved through the REAL bank readers (rb.bank_decisions with the LS sources, the
    per_seed_v1 confirmations incl. replication/solo/all8 tapes, rb.temperature, the practice
    decision links and snapshots), plus the legacy-u3 isolated reference checkpoints and the
    composite rule file. Every entry is listed relative to /home/ubuntu with its SHA256 and
    size so an operator can rsync+sha it to cpuS. ``remote_root`` is the original run root
    under /home/ubuntu; its contents are read from the local mirror registered by
    ``manifest_source``. ``sources`` maps arm -> (remote_path, ...) LS-COLLECT label roots.

    Superset rule (disclosed in the manifest): every receipt's full branch set is staged, and
    the whole snapshot directory of each arm's own bank, because the frozen per-update practice
    plans and R's fallback rows are sampled at runtime from any admitted example.
    """
    from . import replay_bank as rb, bank_statistics, practice_starts
    if _mirror[0] is None:
        raise ValueError('REFUSED: register the local mirror first (manifest_source)')
    root_remote = str(remote_root).replace('\\', '/')
    root_local = _local_of(root_remote)
    if not root_local.is_dir():
        raise ValueError('REFUSED: local mirror of the staged root is missing: '+str(root_local))
    src_remote, src_local = {}, {}
    for arm, paths in (sources or {}).items():
        src_remote[arm] = []
        for p in paths:
            remote = str(p).replace('\\', '/')
            src_remote[arm].append(remote)
            src_local[remote] = _local_of(remote)
    watched = [root_local]+list(src_local.values())
    staging_home = (_mirror[0]/'home'/'ubuntu').resolve()
    discovered = set()
    entries, notes = {}, []
    recorded = set()
    snapshot_dependencies = []

    def under(base, resolved):
        base = Path(base)
        return resolved == base or base in resolved.parents

    def remote_of(resolved):
        if staging_home in resolved.parents:
            target = staging_home/resolved.relative_to(staging_home).parts[0]
            discovered.add(target)
            return '/home/ubuntu/'+target.name
        if under(root_local, resolved):
            return root_remote
        for remote, local in src_local.items():
            if under(local, resolved):
                return remote
        raise ValueError('REFUSED: file outside the staged roots: '+str(resolved))

    def add(local_path, kind):
        local_path = Path(guard_path(local_path))
        resolved = Path(guard_path(local_path.resolve()))
        if not local_path.is_file():
            return False
        key = _rel(remote_of(resolved), resolved)
        if key in entries:
            return True
        entries[key] = dict(sha256=sha(resolved), bytes=resolved.stat().st_size, kind=kind)
        return True

    def add_aliases(local_path, kind):
        """Keep both the reader path and target; staging needs no symlink privilege."""
        if not add(local_path, kind):
            return False
        lexical = Path(local_path).absolute()
        if lexical != lexical.resolve():
            key = 'home/ubuntu/'+lexical.relative_to(staging_home).as_posix()
            guard_path(key)
            entries[key] = dict(sha256=sha(local_path), bytes=lexical.stat().st_size, kind=kind)
        return True

    # 1) the REAL readers, under read recording (proves the resolution and catches every read).
    #    READ-ONLY: rb.temperature is invoked only when its sealed file already exists (the
    #    unfrozen rule is derived deterministically on cpuS from the same decisions; calling it
    #    with >=20 decisions would WRITE the file into this read-only mirror).
    with _ReadRecorder(watched, staging_home) as rec:
        for arm in ('B', 'R'):
            srcs = tuple(src_local.get(p, Path(p)) for p in src_remote.get(arm, ()))
            decisions = rb.bank_decisions(root_local, arm, sources=srcs)
            links = practice_starts._links(root_local, arm)
            # This is the reader used by lesson/isolated/practice consumers, not
            # an inferred inventory: it opens pending envelopes, gz snapshots,
            # and the envelope's episode RESULT (including compressed fallback).
            for decision in decisions:
                if decision.get('_policy_version') == 'u3' and decision['verification_id'] not in links:
                    # Capture shared dependencies even if a preceding decision
                    # opened the same episode result already.
                    with _ReadRecorder(watched, staging_home) as snapshot_rec:
                        rb._decision_snapshot(root_local, arm, decision, links)
                    dependencies = snapshot_rec.paths | snapshot_rec.lexical
                    snapshot_dependencies.append(dict(arm=arm, verification_id=decision['verification_id'],
                        files=sorted(_rel(remote_of(p), p) for p in dependencies)))
            if (rb.bank_root(root_local, arm)/f'TEMPERATURE_{arm}_{training_seed}.json').is_file():
                rb.temperature(root_local, arm, training_seed, decisions)
            bank_statistics.confirmations_per_seed(rb.confirmations, rb, root_local, arm, srcs)
            practice_starts._links(root_local, arm)
        recorded = set(rec.paths) | rec.lexical
        discovered.update(rec.discovered)

    # 2) deterministic walk of the reader-defined bank layout (the run may read any receipt's
    #    branches and any snapshot at a later update; disclosed superset)
    for arm in ('B', 'R'):
        bank = rb.bank_root(root_local, arm)
        for sub, kind in (('decisions', 'decision_link'), ('confirmations', 'confirmation'),
                          ('confirmations_safety0426', 'confirmation_overlay'),
                          ('confirmations_all8', 'confirmation_all8'),
                          ('snapshots', 'snapshot')):
            for p in sorted((bank/sub).glob('*.*')):
                add_aliases(p, kind)
        for temperature in sorted(bank.glob('TEMPERATURE_*.json')):
            add_aliases(temperature, 'temperature')
        lesson = bank/f'LESSON_LIST_{arm}.json'
        if lesson.exists():
            add_aliases(lesson, 'lesson_list')
        verification_roots = dict(sumo_verification='bank_receipt',
                                 confirm_replication='replication_receipt')
        # Include the real per-seed reader layout (extended windows and extra
        # seeds too), rather than a historically hand-written subset.
        verification_roots.update({p.name: 'confirmation_receipt'
                                   for p in bank.glob('confirm_*') if p.is_dir()})
        for name, kind in sorted(verification_roots.items()):
            verifications = bank/name/'verifications'
            if not verifications.is_dir():
                continue
            for p in sorted(verifications.glob('*.json')):
                add_aliases(p, kind)
                receipt = json.loads(p.read_text(encoding='utf-8'))
                for branch in receipt.get('branches') or ():
                    d = bank/name/'branches'/receipt['verification_id']/branch['branch']
                    add_aliases(d/'RESULT.json', 'tape')
                    add_aliases(d/'RESULT.json.gz', 'tape')
                    add_aliases(d/'observable.jsonl.gz', 'trace')
        for name in verification_roots:
            for p in sorted((bank/name/'links').glob('*.json')):
                add_aliases(p, 'solo_link')
    for arm, paths in src_remote.items():
        for remote in paths:
            src = src_local[remote]
            for p in sorted((src/'labels').glob('*.json')):
                add(p, 'label')
            for p in sorted((src/'branches').glob('*/*/RESULT.json')):
                add(p, 'tape')
                add(p.parent/'observable.jsonl.gz', 'trace')
            for p in sorted((src/'label_attempts').glob('*/branches/*/*/RESULT.json*')):
                add(p, 'tape')
                add(p.parent/'observable.jsonl.gz', 'trace')
    # 3) MAPPO checkpoints (u0 for all three arms; legacy-u3 isolated references) + rule file.
    #    All required: the run cannot start (u0), run its isolated synergy (u3 references) or
    #    arm the composite guard (rule) without them -> fail closed here, not on cpuS.
    required = [(root_local/CHECKPOINT_000_SOURCE, 'checkpoint_u0', 'common MAPPO u0 (staged as '
                '{A,B,R}_checkpoint_000.pt in the new root)'),
                (root_local/RULE_FILE, 'safety_rule', 'composite MAIN09:57 rule (copied into the new root)')]
    required += [(root_local/name, 'checkpoint_reference_u3', 'fresh legacy-u3 isolated reference')
                 for name in ISOLATED_REFERENCE_CHECKPOINTS]
    for path, kind, why in required:
        if not add(path, kind):
            raise ValueError('REFUSED: required staging input missing: '+str(path)+' ('+why+')')
    # 4) files the readers actually opened but the walk did not cover (fail-closed completeness)
    for p in sorted(recorded):
        if p.is_file():
            add_aliases(p, 'reader_opened')
    for key, entry in entries.items():
        if entry['kind'] == 'reader_opened':
            entry['kind'] = 'bank_input'
    if not entries:
        raise ValueError('REFUSED: the staging manifest resolved no files')
    roles = dict(common_checkpoint=[], safety_rule=[], reference_B=[], reference_R=[],
                 bank_B=[], bank_R=[])
    for key, entry in entries.items():
        if entry['kind'] == 'checkpoint_u0':
            roles['common_checkpoint'].append(key)
        if entry['kind'] == 'safety_rule':
            roles['safety_rule'].append(key)
        for arm in ('B', 'R'):
            if entry['kind'] == 'checkpoint_reference_u3' and key.endswith('/'+arm+'_checkpoint_003.pt'):
                roles['reference_'+arm].append(key)
            if '/replay_bank/'+arm+'/' in key:
                roles['bank_'+arm].append(key)
    from .exact_grade import _seal
    manifest = _seal(dict(schema='coopmwm.rerun_cpus.staging_manifest.v1', home=HOME,
        root=dict(remote=root_remote, local_mirror=str(root_local)),
        sources={arm: list(paths) for arm, paths in src_remote.items()},
        allowed_staged_roots=sorted('/home/ubuntu/'+p.name for p in discovered),
        training_seed=training_seed, entries=dict(sorted(entries.items())), roles=roles,
        snapshot_dependencies=snapshot_dependencies,
        counts={k: sum(1 for e in entries.values() if e['kind'] == k)
                for k in sorted({e['kind'] for e in entries.values()})},
        supersets=['every enumerated receipt\'s full branch set (RESULT.json + observable.jsonl.gz)',
                   'each arm\'s whole own-snapshot directory (practice plans sample at runtime)'],
        notes=notes, readers=['replay_bank.bank_decisions', 'replay_bank.temperature',
                              'replay_bank.confirmations', 'bank_statistics.confirmations_per_seed',
                              'practice_starts._links']), 'manifest_sha256')
    return manifest


def manifest_source(mirror):
    """Register the local mirror directory whose layout equals /home/ubuntu (read-only)."""
    guard_path(mirror)
    _mirror[0] = Path(guard_path(Path(mirror).resolve()))
    return _mirror[0]


def validate_manifest(manifest, *, checkpoint_sha256=None):
    """Validate the sealed inventory before any destination file is read."""
    from .exact_grade import verify_sealed
    try:
        verify_sealed(manifest, 'manifest_sha256')
    except (ValueError, KeyError) as e:
        raise SystemExit('REFUSED: staging manifest seal broken: '+str(e))
    if manifest.get('home') != HOME:
        raise SystemExit('REFUSED: staging manifest home differs from '+HOME)
    entries = manifest.get('entries')
    if not isinstance(entries, dict) or not entries:
        raise SystemExit('REFUSED: empty staging manifest')
    for key, entry in entries.items():
        p = PurePosixPath(key)
        guard_path(key)
        if (p.is_absolute() or p.parts[:2] != ('home', 'ubuntu') or len(p.parts) < 4
                or '..' in p.parts or '\\' in key or ':' in key or str(p) != key):
            raise SystemExit('REFUSED: unsafe staged destination: '+key)
        if (not re.fullmatch('[0-9a-f]{64}', str(entry.get('sha256', '')))
                or type(entry.get('bytes')) is not int or entry['bytes'] < 0):
            raise SystemExit('REFUSED: invalid staging entry: '+key)
    roles = manifest.get('roles', {})
    for dependency in manifest.get('snapshot_dependencies', []):
        files = dependency.get('files', [])
        if (not files or any(k not in entries for k in files)
                or not any('/pending/' in k and k.endswith('.json.gz') for k in files)
                or not any('/snapshots/' in k and k.endswith('.pt.gz') for k in files)
                or not any(k.endswith(('/RESULT.json', '/RESULT.json.gz')) for k in files)):
            raise SystemExit('REFUSED: incomplete LS snapshot dependencies')
    for role in ('common_checkpoint', 'safety_rule', 'reference_B', 'reference_R', 'bank_B', 'bank_R'):
        keys = roles.get(role)
        if not isinstance(keys, list) or not keys or any(k not in entries for k in keys):
            raise SystemExit('REFUSED: missing required staging role: '+role)
        kinds = {entries[k]['kind'] for k in keys}
        if role.startswith('bank_'):
            if not {'decision_link', 'snapshot', 'tape', 'trace', 'confirmation_all8',
                    'bank_receipt'}.issubset(kinds):
                raise SystemExit('REFUSED: incomplete required staging role: '+role)
        else:
            kind = {'common_checkpoint': 'checkpoint_u0', 'safety_rule': 'safety_rule',
                    'reference_B': 'checkpoint_reference_u3', 'reference_R': 'checkpoint_reference_u3'}[role]
            if len(keys) != 1 or kinds != {kind}:
                raise SystemExit('REFUSED: invalid staging role: '+role)
    if checkpoint_sha256 is not None and entries[roles['common_checkpoint'][0]]['sha256'] != checkpoint_sha256:
        raise SystemExit('REFUSED: staged common checkpoint/config u0 mismatch')
    return manifest


def verify_staged(manifest, stage_dir, *, checkpoint_sha256=None):
    """Hash every approved file, refusing empty inventories and escaped destinations."""
    validate_manifest(manifest, checkpoint_sha256=checkpoint_sha256)
    stage = Path(guard_path(stage_dir)).resolve()
    guard_path(stage)
    bad = []
    for key, entry in sorted(manifest['entries'].items()):
        guard_path(key)
        staged = stage.joinpath(*key.split('/'))
        resolved = Path(guard_path(staged.resolve()))
        if stage not in resolved.parents:
            raise SystemExit('REFUSED: staged destination escapes stage: '+key)
        if not staged.is_file() or staged.is_symlink():
            bad.append(key+': missing')
        elif sha(staged) != entry['sha256']:
            bad.append(key+': sha differs')
        elif staged.stat().st_size != entry['bytes']:
            bad.append(key+': bytes differ')
    if bad:
        raise SystemExit('REFUSED: %d staged file(s) do not match the manifest: %s'
                         % (len(bad), '; '.join(bad[:5])))
    return dict(files=len(manifest['entries']), bytes=sum(e['bytes'] for e in manifest['entries'].values()),
                manifest_sha256=manifest['manifest_sha256'])


def staging_files_command(manifest, *, track2=False):
    """Operator commands: write the file list, then rsync+sha it to cpuS (relative to /home/ubuntu)."""
    files = '\n'.join(sorted(manifest['entries']))
    target = 'cpuV' if track2 else 'cpuS'
    return ('cat > STAGING_FILES.txt <<\'EOF\'\n'+files+'\nEOF\n'
            'rsync -a --numeric-ids --files-from=STAGING_FILES.txt '
            '<original-host>:/ /home/ubuntu/   # then rsync /home/ubuntu to '+target+' unchanged')


# ---------------------------------------------------------------- pre-registered criterion + replicated isolated test

CRITERION_FILE = 'PREREGISTERED_CRITERION.json'


def criterion_document():
    """The PRE-REGISTERED isolated-cooperation-test criterion as a sealed (self-hash)
    document: the registered pass rule, the replicated isolated-test spec and the full quote.
    Any edit after registration breaks the seal; OWNER_APPROVAL.json binds the file SHA and
    the run root's AUTHORITY.md binds the document SHA."""
    from .exact_grade import _seal
    return _seal(dict(schema='coopmwm.rerun_cpus.preregistered_criterion.v1',
                       criterion=deepcopy(CRITERION), isolated_test=deepcopy(ISOLATED_TEST),
                       final_evaluation=deepcopy(FINAL_EVALUATION),
                       bank_training=deepcopy(BANK_TRAINING),
                      arms=dict(A='plain MAPPO PPO (no trigger head, no teacher)',
                                B='T2A trigger head + frozen cooperation layer + WM bank',
                                R='T2A trigger head + frozen cooperation layer + own random bank'),
                      slots=deepcopy(TRACK_SLOTS), cpu=deepcopy(CPU),
                      early_probes=deepcopy(EARLY_PROBES), learning_monitor=deepcopy(LEARNING_MONITOR),
                      smoke=deepcopy(SMOKE),
                      trigger_head=head_teacher_config()[0], teacher=head_teacher_config()[1],
                       quote=quote(track2=True)), 'criterion_sha256')


def isolated_jobs(registration):
    """Dry job grid only: explicit TRAIN worlds, state-set and full-window identities.

    Each replicate is one whole isolated check (together vs best solo, background base),
    not an individual branch. A qualified diagnostic adapter expands it using the existing
    isolated machinery and must quote the real branches before any charge. Repeated calls
    return the same IDs; no claims, dispatches or output writes occur here.
    """
    from .exact_grade import _seal
    spec = ISOLATED_TEST
    if registration.get('role') != 'train' or registration.get('metric') != spec['metric']:
        raise ValueError('REFUSED: isolated registration needs TRAIN worlds and registered metric')
    rows = registration.get('replicates', [])
    if len(rows) != spec['replicates']:
        raise ValueError('REFUSED: registration must list every isolated replicate')
    by_rep, seeds = {}, set()
    for row in rows:
        r, seed = row.get('replicate'), row.get('world_seed')
        if type(r) is not int or not 0 <= r < spec['replicates'] or r in by_rep:
            raise ValueError('REFUSED: duplicate or invalid registered replicate')
        if type(seed) is not int or seed < 0 or seed in seeds:
            raise ValueError('REFUSED: replicates need distinct explicit integer world seeds')
        for field in ('state_set_sha256', 'window_set_sha256'):
            if not re.fullmatch('[0-9a-f]{64}', row.get(field, '')):
                raise ValueError('REFUSED: registration missing '+field)
        by_rep[r], seeds = row, seeds | {seed}
    reg_sha = _seal(deepcopy(registration), 'registration_sha256')['registration_sha256']
    return [dict(id=f't2c_isolated_u{u:03}_{a}_r{r:03}', arm=a, update=u, replicate=r,
                 world_seed=by_rep[r]['world_seed'],
                 state_set_sha256=by_rep[r]['state_set_sha256'],
                 window_set_sha256=by_rep[r]['window_set_sha256'],
                 registration_sha256=reg_sha,
                 sim_s=spec['sim_s_per_replicate'], phase='isolated_replicated',
                 metric_name=spec['metric'])
            for u in spec['updates'] for a in spec['arms'] for r in range(spec['replicates'])]


def isolated_record(job, *, delta_all_new_veh_s, delta_only_member_new_veh_s, safety):
    """Adapter output in existing policy_synergy.metrics units, never a free-form score.
    Deltas must use the same base reference and base background, over the FULL window.
    """
    solos = list(delta_only_member_new_veh_s.values())
    if not 2 <= len(solos) <= 4:
        raise ValueError('REFUSED: isolated cooperation requires 2..4 group members')
    values = [float(delta_all_new_veh_s)] + [float(v) for v in solos]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('REFUSED: non-finite isolated deltas')
    if set(safety) != set(ISOLATED_TEST['safety_counters']):
        raise ValueError('REFUSED: isolated safety counters must be complete')
    return dict(job, metric=values[0]-max(values[1:]), status='graded', **safety)


def bootstrap_ci(values, *, reps=ISOLATED_BOOTSTRAP_REPS, seed=ISOLATED_BOOTSTRAP_SEED,
                 level=ISOLATED_BOOTSTRAP_LEVEL):
    """Deterministic percentile bootstrap CI of the mean (stdlib random, fixed seed): the
    registered analysis of the replicated isolated test - reproducible before any run."""
    import random
    from statistics import fmean
    vals = [float(v) for v in values]
    n = len(vals)
    if n < 2:
        raise ValueError('REFUSED: a bootstrap CI needs at least two paired replicates')
    if type(reps) is not int or reps < 1 or not 0 < level < 1:
        raise ValueError('REFUSED: bad bootstrap parameters')
    if not all(math.isfinite(v) for v in vals):
        raise ValueError('REFUSED: non-finite bootstrap values')
    rng = random.Random(int(seed))
    stats = sorted(fmean(vals[rng.randrange(n)] for _ in range(n)) for _ in range(reps))
    return dict(mean=fmean(vals), lo=stats[int((1-level)/2*reps)],
                hi=stats[min(reps-1, int((1-(1-level)/2)*reps))],
                n=n, reps=int(reps), level=float(level), seed=int(seed))


def _cell_seed(base_seed, update, pair_index):
    """The registered per-cell bootstrap seed (documented in ISOLATED_TEST['cell_seed_rule'])."""
    return int(base_seed)*10000 + 100*int(update) + int(pair_index)


def evaluate_isolated_test(records, *, registration):
    """Registered analysis of the replicated isolated cooperation test: refuses any non-registered
    arm/update/replicate, duplicates, non-finite metrics, safety counters or an INCOMPLETE
    grid (every arm x update x replicate cell must be present), computes the paired
    differences (same world seed across arms by replicate index) with per-cell bootstrap CIs,
    and applies the proposed pre-registered criterion: PASS iff at u24 the B-vs-A CI lies entirely
    above 0 AND B's summed safety counters do not exceed A's. R (own random bank) is reported
    (B-R ablation), never gated."""
    spec, updates, arms, reps = ISOLATED_TEST, list(ISOLATED_TEST['updates']), \
        list(ISOLATED_TEST['arms']), int(ISOLATED_TEST['replicates'])
    counters = list(spec['safety_counters'])
    jobs = {(j['arm'], j['update'], j['replicate']): j for j in isolated_jobs(registration)}
    index = {}
    for rec in records:
        arm, u, r = rec.get('arm'), rec.get('update'), rec.get('replicate')
        if arm not in arms:
            raise ValueError('REFUSED: isolated-test record for a non-registered arm: '+str(arm))
        if type(u) is not int or u not in updates:
            raise ValueError('REFUSED: isolated-test record at a non-registered update: '+str(u))
        if not isinstance(r, int) or isinstance(r, bool) or not 0 <= r < reps:
            raise ValueError('REFUSED: replicate index outside the registered range: '+str(r))
        key = (arm, u, r)
        if key in index:
            raise ValueError('REFUSED: duplicate isolated-test record: '+str(key))
        expected = jobs[key]
        if rec.get('status') != 'graded':
            raise ValueError('REFUSED: ungraded isolated result; never impute or drop failures')
        for field in ('id', 'metric_name', 'world_seed', 'state_set_sha256',
                      'window_set_sha256', 'registration_sha256'):
            if rec.get(field) != expected[field] or (field == 'world_seed' and
                                                    type(rec.get(field)) is not int):
                raise ValueError('REFUSED: unpaired or unregistered isolated '+field)
        try:
            metric = float(rec['metric'])
            safety_values = {c: float(rec[c]) for c in counters}
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError('REFUSED: missing or invalid isolated metric/safety counter') from e
        if not math.isfinite(metric):
            raise ValueError('REFUSED: non-finite isolated-test metric')
        cell = dict(metric=metric)
        for c in counters:
            v = safety_values[c]
            if not math.isfinite(v) or v < 0 or not v.is_integer():
                raise ValueError('REFUSED: bad isolated-test safety counter '+c+': '+str(v))
            cell[c] = v
        ref = rec.get('reference_safety', {})
        if any(type(ref.get(c)) not in (int, float) or not math.isfinite(ref[c])
               or ref[c] < 0 or int(ref[c]) != ref[c] for c in counters):
            raise ValueError('REFUSED: missing matched common-base reference safety counters')
        cell['reference_safety'] = ref
        index[key] = cell
    missing = [(a, u, r) for a in arms for u in updates for r in range(reps) if (a, u, r) not in index]
    if missing:
        raise ValueError('REFUSED: incomplete isolated-test grid: %d cell(s) missing, e.g. %s'
                         % (len(missing), missing[0]))
    # Common u0's all-group/each-solo policies are identical, so its isolated
    # cooperation gain is algebraically zero; no extra A simulation is scheduled.
    pairs = (('B', 'common_base'), ('R', 'common_base'), ('B', 'R'))
    bspec = spec['bootstrap']
    per_update = {}
    for u in updates:
        entry = {}
        for pi, (hi_arm, lo_arm) in enumerate(pairs):
            diffs = [index[(hi_arm, u, r)]['metric'] -
                     (0. if lo_arm == 'common_base' else index[(lo_arm, u, r)]['metric'])
                     for r in range(reps)]
            ci = bootstrap_ci(diffs, reps=bspec['reps'], seed=_cell_seed(bspec['seed'], u, pi),
                              level=bspec['level'])
            entry[hi_arm+'-'+lo_arm] = dict(mean=ci['mean'], ci_lo=ci['lo'], ci_hi=ci['hi'],
                                            n=ci['n'], level=ci['level'])
        per_update[str(u)] = entry
    pu = CRITERION['primary_update']
    primary = dict(per_update[str(pu)]['B-common_base'], update=pu, pair='B-common_base',
                   ci_excludes_zero_positive=bool(per_update[str(pu)]['B-common_base']['ci_lo'] > 0))
    tol = float(CRITERION['safety_tolerance'])
    safety = {}
    for c in counters:
        a_sum = sum(index[('B', pu, r)]['reference_safety'][c] for r in range(reps))
        b_sum = sum(index[('B', pu, r)][c] for r in range(reps))
        safety[c] = dict(common_base=a_sum, B=b_sum, ok=bool(b_sum <= a_sum + tol))
    primary['safety_ok'] = all(s['ok'] for s in safety.values())
    primary['pass'] = bool(primary['ci_excludes_zero_positive'] and primary['safety_ok'])
    return dict(schema='coopmwm.rerun_cpus.isolated_eval.v1',
                verdict='pass' if primary['pass'] else 'fail',
                registered=CRITERION['pass_rule'],
                r_arm='random-bank fairness control; random dense teacher; B-R reported, never gated',
                primary=primary, safety=safety, per_update=per_update,
                secondary_updates=list(CRITERION['secondary_updates']),
                cells=len(index), replicates=reps, arms=arms,
                criterion_sha256=criterion_document()['criterion_sha256'])


# ---------------------------------------------------------------- approval-gated dry preflight

BUDGET_REQUEST = 'RERUNCPUS_BUDGET_REQUEST.json'
APPROVAL = 'OWNER_APPROVAL.json'
OWNER_QUOTE = 'OWNER_QUOTE.txt'
VERDICT_FILE = 'EVENING_VERDICT.json'
MANIFEST_FILE = 'STAGING_MANIFEST.json'
CONFIG_FILE = 'T2C_CONFIG.json'
ISOLATED_REGISTRATION_FILE = 'ISOLATED_REGISTRATION.json'


def budget_request_document(cfg, registration, manifest):
    """Exact approval payload, with no model imports, charges or launches."""
    from .exact_grade import _seal
    check_pins(cfg)
    if not cfg.get('track2'):
        raise ValueError('REFUSED: Track 2 approval requires explicit opt-in')
    validate_manifest(manifest, checkpoint_sha256=cfg['common_checkpoint_sha256'])
    jobs = isolated_jobs(registration)
    final = final_world_registration(registration)
    return dict(quote=quote(track2=True), budget=budget(track2=True), new_sim_s=0,
                config_sha256=_seal(cfg, 'config_sha256')['config_sha256'],
                registration_sha256=jobs[0]['registration_sha256'],
                criterion_sha256=criterion_document()['criterion_sha256'],
                manifest_sha256=manifest['manifest_sha256'],
                final_evaluation=deepcopy(FINAL_EVALUATION),
                final_worlds_sha256=final['final_worlds_sha256'])


def final_world_registration(registration):
    """Explicit common-world registration shared with T1D; never discover seed files."""
    from .exact_grade import _seal
    final = deepcopy(registration.get('final_evaluation', {}))
    worlds = final.get('worlds', [])
    if final.get('role') != 'train' or len(worlds) != TEST_EPISODES:
        raise ValueError('REFUSED: final paired evaluation requires 96 explicit TRAIN worlds')
    ids, seeds = set(), set()
    for row in worlds:
        if set(row) != {'world_id', 'world'} or not isinstance(row['world_id'], str):
            raise ValueError('REFUSED: invalid final world identity')
        world = row['world']
        if (not isinstance(world, dict) or type(world.get('seed')) is not int or world['seed'] < 0
                or row['world_id'] in ids or world['seed'] in seeds):
            raise ValueError('REFUSED: duplicate/invalid final paired world')
        ids.add(row['world_id'])
        seeds.add(world['seed'])
    return _seal(final, 'final_worlds_sha256')


def preflight_dry(prep_dir, *, stage_dir, budget_request=BUDGET_REQUEST, approval=APPROVAL,
                  owner_quote=OWNER_QUOTE, verdict=VERDICT_FILE, manifest=MANIFEST_FILE,
                  criterion=CRITERION_FILE, config=CONFIG_FILE,
                  registration=ISOLATED_REGISTRATION_FILE, track2=False):
    """Zero-physics dry preflight: refuses unless OWNER_APPROVAL.json exists and binds the exact
    budget request SHA, the owner quote SHA, the verdict file SHA (verdict positive) and the
    PRE-REGISTERED criterion file SHA (sealed, carrying the registered criterion + replicated
    isolated-test spec) with the effective-imitation commit pinned, AND every staging-manifest
    file SHA matches the staged copies. The exact config, explicit TRAIN-world registration,
    full budget and run-root authority are also required. Passing qualifies the preparation
    artifacts only: sibling runtime integration remains required. Never dispatches; new_sim_s=0.

    Absent/false track2 preserves the approved legacy preflight and receipt contract.
    OWNER_APPROVAL criterion_sha256 is the FILE sha; AUTHORITY.md and config criterion_sha256
    use the DOCUMENT self-hash (separate digests, both returned in the receipt).
    """
    if not track2:
        from .rerun_cpus_compat import preflight_dry as legacy_preflight
        return legacy_preflight(prep_dir, stage_dir=stage_dir, budget_request=budget_request,
                                approval=approval, owner_quote=owner_quote, verdict=verdict,
                                manifest=manifest)
    from .exact_grade import verify_sealed
    guard_path(prep_dir)
    guard_path(stage_dir)
    prep, stage = (Path(guard_path(Path(p).resolve())) for p in (prep_dir, stage_dir))
    req = Path(guard_path((prep/budget_request).resolve()))
    if not req.is_file():
        raise SystemExit('REFUSED: budget request missing: '+str(req))
    request = json.loads(req.read_text(encoding='utf-8'))
    if request.get('quote', {}).get('total') != quote(track2=True)['total']:
        raise SystemExit('REFUSED: budget request total differs from the 3-arm quote')
    app = Path(guard_path((prep/approval).resolve()))
    if not app.is_file():
        raise SystemExit('REFUSED: no OWNER_APPROVAL.json (owner budget approval required)')
    a = json.loads(app.read_text(encoding='utf-8'))
    if a.get('budget_request_sha256') != sha(req):
        raise SystemExit('REFUSED: OWNER_APPROVAL budget sha differs from the request')
    if a.get('owner_approved') is not True:
        raise SystemExit('REFUSED: owner approval not positive')
    q = Path(guard_path((prep/owner_quote).resolve()))
    if not q.is_file() or a.get('owner_quote_sha256') != sha(q):
        raise SystemExit('REFUSED: owner quote missing or sha differs')
    v = Path(guard_path((prep/verdict).resolve()))
    if not v.is_file() or a.get('verdict_file_sha256') != sha(v):
        raise SystemExit('REFUSED: verdict file missing or sha differs')
    verdict_value = json.loads(v.read_text(encoding='utf-8'))
    if verdict_value.get('verdict') != 'positive':
        raise SystemExit('REFUSED: post-fix evening verdict not positive')
    if a.get('effective_imitation_commit') != EFFECTIVE_IMITATION_COMMIT:
        raise SystemExit('REFUSED: effective-imitation commit not pinned to '+EFFECTIVE_IMITATION_COMMIT)
    c = Path(guard_path((prep/criterion).resolve()))
    if not c.is_file():
        raise SystemExit('REFUSED: pre-registered criterion missing: '+str(c))
    if a.get('criterion_sha256') != sha(c):
        raise SystemExit('REFUSED: OWNER_APPROVAL criterion sha differs from the registered criterion')
    doc = json.loads(c.read_text(encoding='utf-8'))
    try:
        verify_sealed(doc, 'criterion_sha256')
    except ValueError as e:
        raise SystemExit('REFUSED: pre-registered criterion seal broken: '+str(e))
    if doc != criterion_document():
        raise SystemExit('REFUSED: criterion file does not carry the registered criterion '
                         'and isolated-test spec')
    config_path = Path(guard_path((prep/config).resolve()))
    reg_path = Path(guard_path((prep/registration).resolve()))
    for path, field in ((config_path, 'config_file_sha256'),
                        (reg_path, 'registration_file_sha256')):
        if not path.is_file() or a.get(field) != sha(path):
            raise SystemExit('REFUSED: missing or unbound '+field)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    reg = json.loads(reg_path.read_text(encoding='utf-8'))
    m = Path(guard_path((prep/manifest).resolve()))
    if not m.is_file():
        raise SystemExit('REFUSED: staging manifest missing: '+str(m))
    man = json.loads(m.read_text(encoding='utf-8'))
    validate_manifest(man, checkpoint_sha256=cfg['common_checkpoint_sha256'])
    if a.get('manifest_file_sha256') != sha(m):
        raise SystemExit('REFUSED: unbound manifest_file_sha256')
    try:
        expected = budget_request_document(cfg, reg, man)
        own_ledger(cfg['run_root'], track2=True)
    except ValueError as e:
        raise SystemExit('REFUSED: Track 2 config/registration/root: '+str(e))
    if request != expected:
        raise SystemExit('REFUSED: full budget request/config/registration differs from registration')
    staged = verify_staged(man, stage, checkpoint_sha256=cfg['common_checkpoint_sha256'])
    return dict(preflight_pass=True, ready_for_registered_launch=False,
                ready_for_qualified_integration=True, prepare_only=True, integration_required=True,
                 target_host='cpuV', new_sim_s=0, budget_request_sha256=sha(req),
                owner_quote_sha256=a.get('owner_quote_sha256'),
                verdict_file_sha256=a.get('verdict_file_sha256'),
                criterion_sha256=a.get('criterion_sha256'),
                criterion_document_sha256=doc['criterion_sha256'],
                config_file_sha256=sha(config_path), registration_file_sha256=sha(reg_path),
                manifest_sha256=man['manifest_sha256'], staged=staged)


def run_tracks(cfg, registration, backend, *, smoke=False, legacy_run=None):
    """Explicit opt-in call site. OFF calls the previous runner directly."""
    if not cfg.get('track2'):
        if legacy_run is None:
            raise ValueError('legacy runtime callable required')
        return legacy_run(cfg)
    from .rerun_tracks_runtime import run
    return run(cfg, registration, backend, smoke=smoke)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('cmd', choices=('quote', 'budget', 'wall_clock', 'criterion', 'isolated_plan',
                                  'isolated_eval', 'manifest', 'preflight', 'smoke', 'run', 'monitor'))
    p.add_argument('--prep', default='.')
    p.add_argument('--stage', default='.')
    p.add_argument('--mirror', default=None, help='local mirror of /home/ubuntu')
    p.add_argument('--legacy-root', default='/home/ubuntu/coopmwm20261102_warm1')
    p.add_argument('--source-b', default=None, help='LS-COLLECT label root for arm B')
    p.add_argument('--out', default=None)
    p.add_argument('--registration', default=ISOLATED_REGISTRATION_FILE)
    p.add_argument('--records', default=None)
    p.add_argument('--config', default=CONFIG_FILE)
    p.add_argument('--adapter', default=None, help='qualified backend factory module:callable (explicit)')
    p.add_argument('--root', default=None, help='monitor run root')
    p.add_argument('--track2', action='store_true', help='opt into strict Track 2 preflight')
    a = p.parse_args()
    if a.cmd == 'quote':
        print(json.dumps(dict(quote=quote(track2=a.track2), wall_clock=wall_clock(track2=a.track2),
                              budget=budget(track2=a.track2))))
    elif a.cmd in ('budget', 'wall_clock', 'criterion'):
        value = (criterion_document() if a.cmd == 'criterion' else
                 {'budget': budget, 'wall_clock': wall_clock}[a.cmd](track2=a.track2))
        print(json.dumps(value, indent=1, sort_keys=True))
    elif a.cmd in ('isolated_plan', 'isolated_eval'):
        reg = json.loads(Path(guard_path(Path(a.registration).resolve())).read_text(encoding='utf-8'))
        if a.cmd == 'isolated_plan':
            print(json.dumps(isolated_jobs(reg), indent=1, sort_keys=True))
        else:
            if not a.records:
                p.error('isolated_eval requires --records')
            records = json.loads(Path(guard_path(Path(a.records).resolve())).read_text(encoding='utf-8'))
            print(json.dumps(evaluate_isolated_test(records, registration=reg), indent=1, sort_keys=True))
    elif a.cmd in ('manifest', 'preflight') and a.mirror:
        manifest_source(a.mirror)
    if a.cmd == 'manifest':
        sources = {'B': [a.source_b]} if a.source_b else {}
        m = staging_manifest(remote_root=a.legacy_root, sources=sources)
        out = a.out or 'STAGING_MANIFEST.json'
        Path(out).write_text(json.dumps(m, indent=1, sort_keys=True)+'\n', encoding='utf-8')
        print(json.dumps(dict(entries=len(m['entries']), manifest_sha256=m['manifest_sha256'],
                              command=staging_files_command(m, track2=a.track2))))
    elif a.cmd == 'preflight':
        print(json.dumps(preflight_dry(a.prep, stage_dir=a.stage, track2=a.track2)))
    elif a.cmd == 'monitor':
        from .rerun_tracks_monitor import read_monitor
        if not a.root:
            p.error('monitor requires --root')
        print(json.dumps(read_monitor(a.root), indent=1, sort_keys=True))
    elif a.cmd in ('smoke', 'run'):
        import importlib
        cfg = json.loads(Path(guard_path(Path(a.config).resolve())).read_text(encoding='utf-8'))
        reg = json.loads(Path(guard_path(Path(a.registration).resolve())).read_text(encoding='utf-8'))
        if a.adapter:
            module, factory = a.adapter.split(':', 1)
            backend = getattr(importlib.import_module(module), factory)()
        else:
            from .rerun_tracks_backend import backend_factory
            backend = backend_factory(cfg, prep_dir=a.prep, stage_dir=a.stage)
        print(json.dumps(run_tracks(cfg, reg, backend, smoke=a.cmd == 'smoke'), indent=1, sort_keys=True))
