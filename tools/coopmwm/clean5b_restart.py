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
DESIGN = ('A = frozen u0 base evaluated as an arm (no PPO); B/R identical code (clean5a: FULLPOL clean5_ev N2 '
          'slate, lane_drop features/moves/V2V 100 m, G2, G4 + G4LD extension, G5 vocabulary, practice offer) + LCEXEC '
          'per-CAV free_program slots (lc_families cap 6, FULLPOL generator/descriptor, no templates, program moves '
          'never BC/EV targets, G4 off for programs); only the guidance differs: B = regime EV-BC solo targets, R = none.')
AUTHORITY = ('MAIN 2026-11-06 23:40 phase clean5b <= 3,900,000 (owner: run clean5b alongside clean5a; BUDGET_CLEAN5B.json); '
             'MAIN 23:48 S-LCEXEC-SMOKE gate satisfied (c4/lcexec 0fe9574e); launch requires local suites + Linux '
             'qualification + N11 integration PASS + dry prepare + smoke + gate')
BUDGET_RULE = ('MAIN 23:40 "phase clean5b <= 3,900,000 sim-s registered" (owner: run clean5b alongside clean5a); '
               'separate envelope, prior 0; smoke+gate inside the quote smoke_gate phase')


def keys(*, desc, guidance_doc, provenance):
    """Common B = R keys: clean5a keys + FULLPOL recommended_5b (identical templates=[] for both arms)."""
    from . import fullpol_ev
    out = c5.keys(desc=desc, guidance_doc=guidance_doc, provenance=provenance)
    out.update(fullpol_ev.recommended_5b(provenance=provenance, regime_guidance=guidance_doc, cap=LC_CAP,
                                         templates=[]))
    return out


VARIANT = dict(run=RUN, quote_file=QUOTE_FILE, approval_file=APPROVAL_FILE, budget_file=BUDGET_FILE, design=DESIGN,
               approved=APPROVED, budget_rule=BUDGET_RULE, keys=keys, cfg_overrides=dict(load_max=LOAD_MAX),
               newmap_extra=dict(action_set='clean5a N2 slate + LCEXEC per-CAV free_program (lc_families cap 6)',
                                 program_targets='none (offerable only; no program BC/EV; G4 off for programs)',
                                 practice_offer_note='practice_offer ON (0.5); no exact-plan demonstration slots in '
                                 'clean5b (no templates, EV-BC only), so it is inert - disclosed'),
               requires_lc_runtime=True, authority=AUTHORITY)


def prepare(source, *, root=ROOT, **kwargs):
    return c5.prepare(source, root=root, variant=VARIANT, **kwargs)
