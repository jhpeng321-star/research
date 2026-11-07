"""Clean5b: clean5a recipe + the full per-CAV action set (LCEXEC free_program) on wz3to2. Zero-physics prepare.

MAIN 2026-11-06 23:40 / 23:48 (REGISTRATION; owner: run clean5b alongside clean5a):
* identical to clean5a (frozen u0 base, A = frozen evaluation arm, same 6 cells and TRAIN worlds, EV-BC solo
  targets from the same S-REGIMEU0 guidance doc, ev_bc_steps 60, trigger-PPO KL row_mean, R = no guidance,
  G2/G4/G4LD/G5/practice_offer, report-only B/R composite safety) PLUS FULLPOL ``recommended_5b``: LCEXEC per-CAV
  free_program slots (``lc_families`` cap 6 = FULLPOL TEMPLATE_CAP, FULLPOL generator/descriptor, NO templates) for BOTH B and R, so the
  action space is identical and the only B/R difference stays the EV-BC guidance.
* program families are offerable only: never BC/EV targets (no program BC, no templates); rolling_wall and
  gap_open_merge included as offerable classes (refs beyond 100 m sensing / closing-lane merges 0/15 - disclosed).
* G4 for programs OFF: the G4LD extension keeps clean5a's families (abreast_wall, group_hold, lane_drop_initiate);
  sparse_synergy_credit skips free_program_* selections (fail closed, disclosed).
* program members bypass the MAPPO pre-safety filter (LCEXEC runtime, MAIN-approved); speedMode 27 / env guards
  kept. Every child needs COOPMWM_G5_RUNTIME=1, COOPMWM_LC_RUNTIME=1, COOPMWM_MAP_SPEC=wz3to2.
* budget: separate phase clean5b <= 3,900,000 sim-s (BUDGET_CLEAN5B.json by MAIN; CLEAN5B_QUOTE.json); host cpuV
  at normal priority (WM-D nice-19 shards yield), registered 24 episode + 16 diagnostic workers (the pinned
  clean-run CPU registration), admission load cap raised for the shared host.
"""
from . import clean4_restart as c4, clean5a_restart as c5

RUN = 'clean5b'
ROOT = '/home/ubuntu/coopmwm20261107_clean5b'
QUOTE_FILE = 'CLEAN5B_QUOTE.json'
APPROVAL_FILE = 'BUDGET_CLEAN5B.json'
BUDGET_FILE = 'CLEAN5B_BUDGET.json'
APPROVED = c4.APPROVED_BY_RUN[RUN]
LC_CAP = 6          # = fullpol_head.TEMPLATE_CAP: the FULLPOL generator does not take cap; cap 3 raised on 4-6 member templates
# cpuV is shared with WM-D nice-19 shards (load ~55 before 5b) and N6/IMITREPLAY's 16 workers; the inherited
# admission cap (40) would block every 5b wave. Memory guard (>= 8 GB available) and disk guard unchanged.
LOAD_MAX = 120
# clean5a ran at load 2-4 (u1 ~55 min): arms are sequential and each wave had cpu.workers//3 = 8 workers.
# One arm's 16-episode update now runs as ONE wave of 16 (+16 diagnostic workers at u2/u4/u6/u12/u24 = 32 cores).
WAVE_WORKERS = 16
# r2 (cpuV, 92 GB): the B wave's 16 workers (~5-6 GB RSS each) OOM-killed the arm owner -> every wave waits for
# MemAvailable >= 30 GB (MAIN 2026-11-07); clean5b runs on cpuZ (184 GB).
MEM_MIN_GB = 30
DESIGN = ('A = frozen u0 base evaluated as an arm (no PPO); B/R identical code (clean5a: FULLPOL clean5_ev N2 '
          'slate, lane_drop features/moves/V2V 100 m, G2, G4 + G4LD extension, G5 vocabulary, practice offer) + LCEXEC '
          'per-CAV free_program slots (lc_families cap 6, FULLPOL generator/descriptor, no templates, program moves '
          'never BC/EV targets, G4 off for programs); only the guidance differs: B = regime EV-BC solo targets, R = none.')
AUTHORITY = ('MAIN 2026-11-06 23:40 phase clean5b <= 3,900,000 (owner: run clean5b alongside clean5a; BUDGET_CLEAN5B.json); '
             'MAIN 23:48 S-LCEXEC-SMOKE gate satisfied (c4/lcexec 0fe9574e); launch requires local suites + Linux '
             'qualification + N11 integration PASS + dry prepare + smoke + gate')
HOLD_LANE_PIN_NOTE = ('active N2 hold members get their own current lane as a G5 lane target every tick '
                      '(single writer; laneChangeMode 513 like the verified S-REGIMEU0 plant path); WMVERIFY found '
                      'unpinned N2 solo holders may leave lane 2')
BUDGET_RULE = ('MAIN 23:40 "phase clean5b <= 3,900,000 sim-s registered" (owner: run clean5b alongside clean5a), amended '
               '2026-11-07 ~03:5x to 3,930,000 after the failed r1/r2 spend (78,172); '
               'separate envelope; prior = the failed-root spend; smoke+gate inside the quote smoke_gate phase')


def keys(*, desc, guidance_doc, provenance):
    """Common B = R keys: clean5a keys + FULLPOL recommended_5b (identical templates=[] for both arms)."""
    from . import fullpol_ev
    out = c5.keys(desc=desc, guidance_doc=guidance_doc, provenance=provenance)
    out.update(fullpol_ev.recommended_5b(provenance=provenance, regime_guidance=guidance_doc, cap=LC_CAP,
                                         templates=[]))
    out['clean5_ev']['hold_lane_pin'] = True     # MAIN 2026-11-07: clean5b launches WITH hold_lane_pin ON
    return out


VARIANT = dict(run=RUN, quote_file=QUOTE_FILE, approval_file=APPROVAL_FILE, budget_file=BUDGET_FILE, design=DESIGN,
               approved=APPROVED, budget_rule=BUDGET_RULE, keys=keys, cfg_overrides=dict(load_max=LOAD_MAX, wave_workers=WAVE_WORKERS, mem_available_min_gb=MEM_MIN_GB),
               newmap_extra=dict(action_set='clean5a N2 slate + LCEXEC per-CAV free_program (lc_families cap 6)',
                                 program_targets='none (offerable only; no program BC/EV; G4 off for programs)',
                                 hold_lane_pin='ON (MAIN 2026-11-07; FULLPOL 5cbd616e): '+HOLD_LANE_PIN_NOTE,
                                 practice_offer_note='practice_offer ON (0.5); no exact-plan demonstration slots in '
                                 'clean5b (no templates, EV-BC only), so it is inert - disclosed'),
               requires_lc_runtime=True, authority=AUTHORITY, guidance_missing_ok=(), worlds=None)


def prepare(source, *, root=ROOT, **kwargs):
    return c5.prepare(source, root=root, variant=VARIANT, **kwargs)


# ---------------------------------------------------------------- clean5b-F (owner REGISTRATION 04:0x / 04:2x)

F_ROOT = '/home/ubuntu/coopmwm20261107_clean5bf'
F_QUOTE_FILE = 'CLEAN5BF_QUOTE.json'
F_NO_GUIDANCE_CELLS = ('D2400_p40',)    # no MULTICRIT rows: no EV-BC target there (registered, disclosed)
F_DESIGN = DESIGN+(' clean5b-F: the firm 15 m/s hold (wall_h) is banned as guidance and never offered; EV-BC targets '
                   '= fairness-eligible strategies only (every origin-lane/class group within +10 s/veh vs u0: '
                   'wall_g_solo2 / wall_g_solo2_gate0 / wall_g_duo12); fairness reward term (excess group harm vs '
                   'the frozen A arm on the same world beyond 10 s/veh, one-for-one veh-s, B = R) from the eval-only '
                   'per-vehicle logger (N6 port, fair_vehicle_log).')
F_AUTHORITY = AUTHORITY+('; owner REGISTRATION 2026-11-07 04:0x (clean5b-F fairness) + 04:2x (group label = A-arm origin '
                         'lane/class by vehicle id); CLEAN5BF_QUOTE covered re-quote of 1bb292e1')


def keys_f(*, desc, guidance_doc, provenance):
    """clean5b keys with FULLPOL's fairness slot (with_fairness: fair slot, fair guidance, parity, pin ON)."""
    from . import fullpol_ev
    out = c5.keys(desc=desc, guidance_doc=guidance_doc, provenance=provenance)
    out.update(fullpol_ev.with_fairness(
        fullpol_ev.recommended_5b(provenance=provenance, regime_guidance=guidance_doc, cap=LC_CAP, templates=[]),
        regime_guidance=guidance_doc))
    return out


VARIANT_F = dict(VARIANT, quote_file=F_QUOTE_FILE, keys=keys_f, design=F_DESIGN, authority=F_AUTHORITY,
                 guidance_missing_ok=F_NO_GUIDANCE_CELLS,
                 newmap_extra=dict(VARIANT['newmap_extra'], variant='clean5b-F',
                                   fairness='EV targets fairness-eligible only; reward term vs frozen A; wall_h banned',
                                   no_guidance_cells=list(F_NO_GUIDANCE_CELLS)))


def prepare_f(source, *, root=F_ROOT, **kwargs):
    return c5.prepare(source, root=root, variant=VARIANT_F, **kwargs)


# ---------------------------------------------------------------- clean5b-F8 (MAIN 06:37: memory restart, option a)

F8_ROOT = '/home/ubuntu/coopmwm20261107_clean5bf8'
F8_QUOTE_FILE = 'CLEAN5BF8_QUOTE.json'
F8_EPISODES = 8                 # per arm per update (A, B, R on the same worlds); was 16
F8_WAVE_WORKERS = 8             # MAIN 07:5x: one wave per arm per update (~60 min/update instead of ~110 with 6)
F8_MEM_MIN_GB = 40              # MAIN 10:4x: 95 deadlocked u1's paired G4 (B parent 86 GB + R 46 GB -> 54 GB available;
                                # G4 workers wait in runner.admission). With fresh per-update children: collection peak
                                # 41+46+8 x 5..10.6 = 127..172 GB, G4 ~86+46+small workers ~140..160 GB < 184.
F8_NOTE = ('MAIN 06:37 (registered, memory restart option a): 8 episodes per arm per update (B = R, A on the same '
           'worlds), wave_workers 8 + MemAvailable >= 40 GB before every wave (MAIN 10:4x; waits; fresh per-update arm children). u12 (primary) and u24 now hold '
           'HALF the episodes of the original 16-per-update design - disclosed.')


def worlds_by_update_8():
    """8 TRAIN worlds per update over the 6 staged cells (cell rotates with the update), seeds cycle in sixths."""
    from . import map_scenarios as ms
    cells = {c['cell_key']: c for c in ms.cells()}
    out = {}
    for u in range(1, 25):
        offset = ((u-1) % 6)*F8_EPISODES
        rows = []
        for k, seed in enumerate(c5.TRAIN_SEEDS[offset:offset+F8_EPISODES]):
            cell = cells[c5.CELL_KEYS[(k+u) % len(c5.CELL_KEYS)]]
            rows.append(ms.world(seed, cell['demand'], cell['penetration']))
        out[str(u)] = rows
    return out


def _fresh_rule():
    from .launch_training import FRESH_UPDATE_RULE    # each arm update in a fresh child: memory freed per update
    return FRESH_UPDATE_RULE


VARIANT_F8 = dict(VARIANT_F, quote_file=F8_QUOTE_FILE, worlds=worlds_by_update_8,
                  cfg_overrides=dict(load_max=LOAD_MAX, wave_workers=F8_WAVE_WORKERS, mem_available_min_gb=F8_MEM_MIN_GB,
                                     arm_update_process=_fresh_rule()),
                  design=F_DESIGN+' '+F8_NOTE, authority=F_AUTHORITY+'; '+F8_NOTE,
                  newmap_extra=dict(VARIANT_F['newmap_extra'], variant='clean5b-F8', episodes_per_update=F8_EPISODES,
                                    half_episode_note=F8_NOTE))


def prepare_f8(source, *, root=F8_ROOT, **kwargs):
    return c5.prepare(source, root=root, variant=VARIANT_F8, **kwargs)
