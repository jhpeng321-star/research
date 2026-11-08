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
import json

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
               requires_lc_runtime=True, authority=AUTHORITY, guidance_missing_ok=(), worlds=None, cell_keys=c5.CELL_KEYS)


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
F8_G4_WORKERS = 4               # owner 2026-11-08 "fix + fresh relaunch" (F8R deadlock 01:04: 8 idle G4 workers kept
F8_G4_MEM_MIN_GB = 20           # 2.6 GB each, MemAvailable stuck at 36.5 < 40): G4 runs 4 one-job workers (memory
                                # returned after every job) and admits at 20 GB; collection waves keep 40. Worst cases in
                                # runs/active/C4-V7FT-20261017/clean5bf8s/MEMORY_PROOF.md.
F8_NOTE = ('MAIN 06:37 (registered, memory restart option a): 8 episodes per arm per update (B = R, A on the same '
           'worlds), wave_workers 8 + MemAvailable >= 40 GB before every wave (MAIN 10:4x; waits; fresh per-update arm children). u12 (primary) and u24 now hold '
           'HALF the episodes of the original 16-per-update design - disclosed.')


def worlds_by_update_8(cell_keys=None):
    """8 TRAIN worlds per update over the staged cells (default the 6 clean5 cells; cell rotates with the
    update), seeds cycle in sixths (the same 48 TRAIN seeds whatever the cells)."""
    from . import map_scenarios as ms
    keys = c5.CELL_KEYS if cell_keys is None else tuple(cell_keys)
    cells = {c['cell_key']: c for c in ms.cells()}
    out = {}
    for u in range(1, 25):
        offset = ((u-1) % 6)*F8_EPISODES
        rows = []
        for k, seed in enumerate(c5.TRAIN_SEEDS[offset:offset+F8_EPISODES]):
            cell = cells[keys[(k+u) % len(keys)]]
            rows.append(ms.world(seed, cell['demand'], cell['penetration']))
        out[str(u)] = rows
    return out


def _fresh_rule():
    from .launch_training import FRESH_UPDATE_RULE    # each arm update in a fresh child: memory freed per update
    return FRESH_UPDATE_RULE


VARIANT_F8 = dict(VARIANT_F, quote_file=F8_QUOTE_FILE, worlds=worlds_by_update_8,
                  cfg_overrides=dict(load_max=LOAD_MAX, wave_workers=F8_WAVE_WORKERS, mem_available_min_gb=F8_MEM_MIN_GB,
                                     g4_workers=F8_G4_WORKERS, g4_mem_available_min_gb=F8_G4_MEM_MIN_GB,
                                     arm_update_process=_fresh_rule()),
                  design=F_DESIGN+' '+F8_NOTE, authority=F_AUTHORITY+'; '+F8_NOTE,
                  newmap_extra=dict(VARIANT_F['newmap_extra'], variant='clean5b-F8', episodes_per_update=F8_EPISODES,
                                    half_episode_note=F8_NOTE))


def prepare_f8(source, *, root=F8_ROOT, **kwargs):
    return c5.prepare(source, root=root, variant=VARIANT_F8, **kwargs)


# ---------------------------------------------------------------- clean5c (MAIN 2026-11-08, L5C launch HELD 05:00)

C_RUN = 'clean5c'
C_ROOT = '/home/ubuntu/coopmwm20261108_clean5c'
C_QUOTE_FILE = 'CLEAN5C_QUOTE.json'
C_APPROVAL_FILE = 'BUDGET_CLEAN5C.json'
C_BUDGET_FILE = 'CLEAN5C_BUDGET.json'
C_NOTE = ('clean5c: B EV-BC targets = fullpol_bank.eligible_families(GUIDANCE_CLEAN5C) - WMVERIFY wmv_mc WM-sourced '
          'multi-CAV plans (N2 parity identical, offer_condition + slots/cooldown enforced at the offer) where verified, '
          'the labelled F8S rule rows elsewhere; B AND R get the same deduplicated bank templates (identical action set), '
          'R has no targets; '
          'otherwise identical to clean5b-F8S.')


def keys_c(*, desc, guidance_doc, provenance):
    """clean5b-F8S keys with the clean5c bank guidance: B targets = eligible families, B templates = the bank's."""
    from . import fullpol_ev, fullpol_bank
    clean5c = guidance_doc.get('clean5c') or {}
    if clean5c.get('rule') != fullpol_bank.RULE:
        raise ValueError('clean5c needs a fullpol_bank guidance doc (GUIDANCE_CLEAN5C)')
    out = c5.keys(desc=desc, guidance_doc=guidance_doc, provenance=provenance)
    unique = unique_templates(guidance_doc)
    templates = dict(B=unique, R=json.loads(json.dumps(unique)))   # same action set; arms differ only in targets
    out.update(fullpol_ev.with_fairness(
        fullpol_ev.recommended_5b(provenance=provenance, regime_guidance=guidance_doc, cap=LC_CAP, templates=templates),
        regime_guidance=guidance_doc))
    out['clean5_ev']['parity_families'] = fullpol_bank.eligible_families(guidance_doc)
    out['clean5c'] = parity_receipt(guidance_doc)
    return out


def unique_templates(doc):
    """Bank templates deduplicated across cells (same family + template + offer condition -> ONE entry, with
    the cells it was verified in), so no strategy is offered more often because it was verified in several cells."""
    out, seen = [], {}
    for t in doc['clean5c']['templates']:
        tpl = json.loads(json.dumps(t['template']))
        tpl['source'].pop('cell', None)
        key = json.dumps([t['family'], tpl, t['offer_condition']], sort_keys=True)
        if key in seen:
            seen[key]['cells'].append(t['cell'])
            continue
        seen[key] = dict(family=t['family'], cells=[t['cell']], template=tpl, offer_condition=t['offer_condition'])
        out.append(seen[key])
    return out


def parity_receipt(doc):
    """Bank-build parity verdicts + targets per cell, recorded in CONFIG / receipts."""
    from . import fullpol_bank
    c = doc['clean5c']
    rows = [r for r in doc['rows'] if r.get('source') == 'wm_bank']
    return dict(rule=c['rule'], guidance_sha256=doc['source']['sha256'], verified_cells=list(c['verified_cells']),
                targets=fullpol_bank.eligible_families(doc),
                bank_targets={f"{r['regime']}/{r['family']}": r.get('n2_parity') for r in rows},
                rejected={k: v['reason'] for k, v in sorted(c['rejected'].items())},
                templates=[dict(family=t['family'], cells=t['cells']) for t in unique_templates(doc)],
                fallback_cells=sorted({r['regime'] for r in doc['rows'] if r.get('source') == 'f8s_rule'}))


C_NO_GUIDANCE_CELLS = ('D2400_p40',)
C_G4_WORKERS = 8        # c4/speed (owner 2026-11-08): G4 one-job pool of 8 (configurable); memory proof
                        # clean5c/MEMORY_PROOF_SPEED.md: both parents 141 GB + 8 x 2.6 GB measured = 161.8 <= 165
# learn-step threads: OFF by default (1). Measured (c4/speed micro benchmark, real track2 train step, hidden 64): the
# per-record learning loop runs SLOWER with intra-op threads (1 thr 18.8 s, 4 thr 25.5 s, 8 thr 30.9 s, 16 thr 42.8 s
# on 22 cores), so the opt-in mechanism exists but is not enabled; pass learn_threads_max > 1 to turn it on.
C_LEARN_THREADS_MAX = 1


def _c_overrides(g4_workers=C_G4_WORKERS, learn_threads_max=C_LEARN_THREADS_MAX):
    from .launch_training import LEARN_THREADS_RULE
    out = dict(VARIANT_F8['cfg_overrides'], g4_workers=g4_workers)
    if learn_threads_max > 1:
        out['learn_threads'] = dict(rule=LEARN_THREADS_RULE, max=learn_threads_max,
                                    reserve=VARIANT_F8['cfg_overrides']['wave_workers'])
    return out
VARIANT_C = dict(VARIANT_F8, run=C_RUN, quote_file=C_QUOTE_FILE, approval_file=C_APPROVAL_FILE, budget_file=C_BUDGET_FILE,
                 approved=c4.APPROVED_BY_RUN[C_RUN], keys=keys_c, guidance_missing_ok=C_NO_GUIDANCE_CELLS,
                 cfg_overrides=None,
                 design=VARIANT_F8['design']+' '+C_NOTE, authority=VARIANT_F8['authority']+'; MAIN 2026-11-08 clean5c '
                 '(L5C) 3,600,000 sim-s; '+C_NOTE,
                 newmap_extra=dict(VARIANT_F8['newmap_extra'], variant='clean5c', clean5c_note=C_NOTE,
                                   program_targets='B: WM-verified bank templates (EV-BC on their strategy labels); R: none'))


def prepare_c(source, *, root=C_ROOT, g4_workers=C_G4_WORKERS, learn_threads_max=C_LEARN_THREADS_MAX, **kwargs):
    variant = dict(VARIANT_C, cfg_overrides=_c_overrides(g4_workers, learn_threads_max))
    return c5.prepare(source, root=root, variant=variant, **kwargs)


# ---------------------------------------------------------------- p20 (owner REGISTRATION 2026-11-08 17:22)
# "switch now and build it and restart": training cells -> 20% CAV for BOTH the WM-guided (clean5c, cpuW) and the
# rule-guided (F8, cpuZ) runs. Same speed code (c4/speed-env 64afeb8e), same TRAIN seeds (989220-989267, 8 per arm
# per update), B = R except the guidance. Guidance docs are files (PREPARE_INPUTS guidance_path) built by
# ops/p20_guidance.py, so a newly WMVERIFY-verified p20 bank (any cell) can be swapped in before launch.
# MAIN/OWNER (later 2026-11-08): include low congestion; per-cell priority by congestion; prefer plans that improve
# BOTH delay and fairness.

P20_CELLS = ('D2400_p20', 'D2800_p20', 'D3200_p20')
P20_PRIORITY = dict(D3200_p20='delay', D2800_p20='fair_and_delay', D2400_p20='fairness', D3200_p40='delay')
P20_RULE_FAMILIES = ('wall_g_duo12', 'wall_g_solo2')   # LOWCAV D3200_p20 PASS (multicrit/lowcav MULTICRIT_confirm)
P20_G4_WORKERS = 6        # REV-SPD 13:51: measured G4 worker peak 7.1 GB -> 6 workers (peak 163.8 GB); both p20 runs
P20_PRIORITY_RULE = (
    'p20 target priority by congestion (owner 2026-11-08): D3200 delay - lb95(gain) > 0 (over the 10 s group bound '
    'labelled, fairness penalty on); D2800 the best fair plan that also cuts delay - lb95 > 0 AND worst group '
    'increase <= 10 s; D2400 fairness - worst group increase < 0 AND CVaR90 delay below u0 AND delay not '
    'significantly worse (ub95(gain) >= 0); targets with lb95 <= 0 are admitted only by that fairness rule '
    '(row admission flag). Where a cell has plans improving BOTH delay (lb95 > 0) and fairness (worst <= 10 s; '
    'D2400 worst < 0), only those are kept. Cells without a target: "none" (B trains there like R).')
P20_NOTE = ('p20 (owner REGISTRATION 2026-11-08 17:22): training cells = 20% CAV only (D2400/D2800/D3200_p20); same '
            'TRAIN seeds, episodes, speed code and B = R code; only the guidance differs. '+P20_PRIORITY_RULE)


P20_HM_CELLS = ('D2800_p20', 'D3200_p20')   # owner 2026-11-08 22:49: heavy + medium only (4/4 per update)
#: owner 2026-11-08 ~23:0x: CLEAN5C5 = D3200_p20 + D2800_p20 + D3200_p40, 8 episodes/update split 3/3/2 (D3200_p40
#: starts without WM guidance; plans join at update boundaries via bank growth). Fixed per-update pattern.
C5_CELLS = ('D2800_p20', 'D3200_p20', 'D3200_p40')
C5_PATTERN = ('D3200_p20', 'D2800_p20', 'D3200_p40')*2+('D3200_p20', 'D2800_p20')


D2800_CELLS = ('D2800_p20',)   # owner 2026-11-08 ~23:1x: CLEAN5C5 and F8V train on D2800_p20 ONLY (8 episodes/update)
CELL_SETS = (P20_CELLS, P20_HM_CELLS, C5_CELLS, D2800_CELLS)
FINAL_EXTERNAL_RULE = ('MAIN 2026-11-08 ~23:1x: no in-run v2v_final phase; the final evaluation is the N6 sealed panel '
                       '(chain coverage needs v2v_final = 0)')


def _check_p20_cells(cells):
    got = tuple(sorted(cells))
    if got not in CELL_SETS:
        raise ValueError('p20 cells must be exactly one of '+repr(CELL_SETS))
    return got


def cells_view(doc, cells):
    """The guidance doc restricted to the training cells (rows, clean5c templates / verified cells); unchanged when
    nothing lies outside; otherwise re-hashed with the parent cited (MAIN: the UNION bank, only the training-cell
    rows are used). Fail closed when no row is left."""
    import copy
    from .fullpol_learn import guidance_doc
    cells = set(cells)
    c = doc.get('clean5c') or {}
    if {r['regime'] for r in doc['rows']} <= cells and all(t['cell'] in cells for t in c.get('templates', ())):
        return doc
    rows = [copy.deepcopy(r) for r in doc['rows'] if r['regime'] in cells]
    if not rows:
        raise ValueError('guidance has no rows in the training cells '+repr(sorted(cells)))
    out = guidance_doc(rows, sources=list(doc['source']['sources'])+[dict(
        name='parent guidance (training-cell view)', sha256=doc['source']['sha256'], baseline='mappo', role='parent')],
        provenance=doc['provenance']+' | training-cell view '+','.join(sorted(cells)), weight=doc['weight'],
        clip=doc['clip'])
    out['source']['rule'] = doc['source'].get('rule', out['source']['rule'])
    for k in set(doc)-set(out):
        out[k] = copy.deepcopy(doc[k])
    if 'clean5c' in out:
        out['clean5c']['templates'] = [t for t in c.get('templates', ()) if t['cell'] in cells]
        out['clean5c']['verified_cells'] = sorted(set(c.get('verified_cells', ())) & cells)
    if 'p20' in out:
        out['p20']['cells'] = {k: v for k, v in out['p20']['cells'].items() if k in cells}
    out['cells_view'] = dict(cells=sorted(cells), parent_rows_sha256=doc['source']['sha256'])
    return out


def cell_pattern(cells):
    """The per-update world pattern: C5 = 3/3/2 (C5_PATTERN), else the cells rotating."""
    return C5_PATTERN if tuple(sorted(cells)) == C5_CELLS else tuple(cells)


def p20_eligible(p, threshold_s=None):
    """(eligible, label, both) for a plan/row (cell or regime, lb95_veh_s, ub95_veh_s, worst_group_increase_s,
    optional cvar90_diff_s = CVaR90 per-vehicle delay strategy - u0, seconds)."""
    from .fullpol_fair import THRESHOLD_S
    th = THRESHOLD_S if threshold_s is None else threshold_s
    cell = p.get('cell') or p.get('regime')
    priority = P20_PRIORITY.get(cell)
    if priority is None:
        return False, 'not_a_p20_training_cell', False
    lb, ub, worst, cvar = p['lb95_veh_s'], p.get('ub95_veh_s'), p.get('worst_group_increase_s'), p.get('cvar90_diff_s')
    if priority == 'delay':
        if not lb > 0:
            return False, 'lb95<=0', False
        within = worst is not None and worst <= th
        return True, 'delay:'+('within_fair_bound' if within else 'over_fair_bound'), within
    if worst is None:
        return False, 'groups_missing', False
    if priority == 'fair_and_delay':
        if not lb > 0:
            return False, 'lb95<=0', False
        return (worst <= th), ('fair_and_delay' if worst <= th else 'over_fair_bound'), worst <= th
    if not worst < 0:
        return False, 'worst_group_not_lowered', False
    if cvar is None:
        return False, 'cvar90_missing', False
    if not cvar < 0:
        return False, 'cvar90_not_lowered', False
    if ub is None or ub < 0:
        return False, 'delay_significantly_worse', False
    return True, 'fairness', lb > 0


def p20_select(doc, cells, *, families=None, note):
    """Guidance doc for the p20 training cells: rows outside the cells (or, for the rule doc, the registered
    families) dropped; the p20 priority applied per row; prefer-both per cell; clean5c templates / verified
    cells restricted to the kept WM plans; per-cell labels (priority, source, targets); re-hashed with the parent
    doc as a cited source. Fail closed when no row is left."""
    import copy
    from .fullpol_learn import FAIRNESS_ADMISSION, guidance_doc
    cells = set(_check_p20_cells(cells))
    kept, rejected = [], {}
    for r in doc['rows']:
        if r['regime'] not in cells or (families is not None and r['family'] not in families):
            continue
        ok, label, both = p20_eligible(r)
        if not ok:
            rejected[f"{r['regime']}/{r['family']}"] = dict(reason=label, source=r.get('source'))
            continue
        row = dict(copy.deepcopy(r), fair=True, p20_label=label, p20_both=both)
        if not r['lb95_veh_s'] > 0:
            row['admission'] = FAIRNESS_ADMISSION          # D2400 fairness priority only
        kept.append(row)
    rows = []
    for cell in sorted(cells):
        mine = [r for r in kept if r['regime'] == cell]
        if any(r['p20_both'] for r in mine):
            for r in mine:
                if not r['p20_both']:
                    rejected[f"{cell}/{r['family']}"] = dict(reason='not_both_while_both_available', source=r.get('source'))
            mine = [r for r in mine if r['p20_both']]
        rows += mine
    if not rows:
        raise ValueError('p20 guidance has no eligible rows in the training cells')
    sources = list(doc['source']['sources'])+[dict(name='parent guidance (p20 selection)', sha256=doc['source']['sha256'],
                                                   baseline='mappo', role='parent')]
    out = guidance_doc(rows, sources=sources, provenance=doc['provenance']+' | '+note, weight=doc['weight'],
                       clip=doc['clip'])
    for k in set(doc)-set(out):
        out[k] = copy.deepcopy(doc[k])
    wm_kept = {(r['regime'], r['family']) for r in rows if r.get('source') == 'wm_bank'}
    if 'clean5c' in doc:
        c = out['clean5c']
        c['templates'] = [t for t in c['templates'] if (t['cell'], t['family']) in wm_kept]
        c['verified_cells'] = sorted({cell for cell, _ in wm_kept})
        c['rejected'] = dict({k: v for k, v in c['rejected'].items() if k.split('/')[0] in cells},
                             **{k: v['reason'] for k, v in rejected.items() if v.get('source') == 'wm_bank'})
        c['skipped'] = {k: v for k, v in c['skipped'].items() if k.split('/')[0] in cells}
    labels = {}
    for cell in sorted(cells):
        mine = [r for r in rows if r['regime'] == cell]
        srcs = sorted({'wm' if r.get('source') == 'wm_bank' else 'rule' for r in mine})
        labels[cell] = dict(priority=P20_PRIORITY[cell], source='+'.join(srcs) if srcs else 'none',
                            fallback=bool(mine) and all(r.get('fallback') for r in mine),
                            targets=[dict(family=r['family'], label=r['p20_label'], both=r['p20_both'],
                                          lb95_veh_s=r['lb95_veh_s'], worst_group_increase_s=r.get('worst_group_increase_s'),
                                          cvar90_diff_s=r.get('cvar90_diff_s')) for r in mine])
    out['p20'] = dict(rule=P20_PRIORITY_RULE, cells=labels, families=None if families is None else list(families),
                      rejected=rejected, parent_sha256=doc['source']['sha256'], note=note)
    return out


def check_p20_guidance(doc, cells, *, wm):
    """Prepare-time (zero physics): a p20-selected doc; every row, template and verified cell in the training
    cells; the rule doc carries only the registered LOWCAV families; returns the 'none' cells (no target)."""
    if doc.get('situational') is not None:         # CLEAN5C4 situational WM doc (fullpol_situational.check)
        from .fullpol_situational import check
        if not wm:
            raise ValueError('p20 rule guidance must be the LOWCAV families '+repr(P20_RULE_FAMILIES)+' only')
        return check(doc, _check_p20_cells(cells))
    cells = set(_check_p20_cells(cells))
    p20 = doc.get('p20') or {}
    regimes = {r['regime'] for r in doc['rows']}
    if (p20.get('rule') != P20_PRIORITY_RULE or not cells <= set(p20.get('cells') or ()) or not regimes <= cells
            or any(not r.get('p20_label') or r.get('fair') is not True for r in doc['rows'])):
        raise ValueError('p20 guidance must be p20-selected for the training cells (ops/p20_guidance.py)')
    if wm:
        c = doc.get('clean5c') or {}
        if not set(c.get('verified_cells', ())) <= cells or any(t['cell'] not in cells for t in c.get('templates', ())):
            raise ValueError('p20 clean5c bank templates outside the training cells')
    elif not {r['family'] for r in doc['rows']} <= set(P20_RULE_FAMILIES) or 'clean5c' in doc:
        raise ValueError('p20 rule guidance must be the LOWCAV families '+repr(P20_RULE_FAMILIES)+' only')
    none = tuple(sorted(cells-regimes))
    if none != tuple(c for c, v in sorted(p20['cells'].items()) if v['source'] == 'none' and c in cells):
        raise ValueError('p20 guidance labels disagree with its rows')
    return none


def _p20_variant(base, cells, g4_workers, *, label, wm, doc, overrides, learnfix=True, bank_growth_control=None,
                 final_external=False, learnfix2=False):
    none = check_p20_guidance(doc, cells, wm=wm)
    sources = ({c: v['source'] for c, v in doc['p20']['cells'].items() if c in cells} if doc.get('situational') is None else
               {c: ('wm_situational' if c not in none else 'none') for c in cells})
    note = P20_NOTE+f' {label}: per-cell guidance source {sources}.'
    return dict(base, cell_keys=cells, worlds=lambda: worlds_by_update_8(cell_pattern(cells)), guidance_missing_ok=none,
                cfg_overrides=dict(overrides, g4_workers=g4_workers,
                                   **({'learnfix': _learnfix_value()} if learnfix else {}),
                                   **({'learnfix2': _learnfix2_value()} if learnfix2 else {}),
                                   **({} if bank_growth_control is None else {'bank_growth': _bank_growth_value(
                                       bank_growth_control)}),
                                   **({'final_eval_external': FINAL_EXTERNAL_RULE} if final_external else {})),
                design=base['design']+' '+note+(' '+LEARNFIX_NOTE if learnfix else ''),
                authority=base['authority']+'; '+note,
                newmap_extra=dict(base['newmap_extra'], variant=label, p20_note=note, no_guidance_cells=list(none),
                                  training_cells=list(cells), p20_guidance_sources=sources,
                                  p20_priority=dict(P20_PRIORITY)))


LEARNFIX_NOTE = ('c4/learnfix (MAIN 2026-11-08 ~22:3x, LEARNDIAG): action-level trigger learning signal '
                 '(tools/coopmwm/learnfix.py RULE), B = R code; plant unchanged.')


def _learnfix_value():
    from . import learnfix
    return learnfix.value()


def _learnfix2_value():
    from . import learnfix
    return learnfix.value2()


def _bank_growth_value(control):
    from .bank_growth import RULE
    return dict(rule=RULE, control=str(control))


def prepare_f8_p20(source, *, root, guidance_doc, cells=P20_CELLS, g4_workers=P20_G4_WORKERS, learnfix=True,
                   final_external=False, learnfix2=False, **kwargs):
    """Rule-guided p20 (F8 code; B = LOWCAV duo12 / wall_g_solo2 EV-BC where eligible, R = none)."""
    cells = _check_p20_cells(cells)
    guidance_doc = cells_view(guidance_doc, cells)
    variant = _p20_variant(VARIANT_F8, cells, g4_workers, label='clean5b-F8-p20', wm=False, doc=guidance_doc,
                           overrides=VARIANT_F8['cfg_overrides'], learnfix=learnfix, final_external=final_external,
                           learnfix2=learnfix2)
    return c5.prepare(source, root=root, variant=variant, guidance_doc=guidance_doc, **kwargs)


def prepare_c_p20(source, *, root, guidance_doc, cells=P20_CELLS, g4_workers=P20_G4_WORKERS,
                  learn_threads_max=C_LEARN_THREADS_MAX, learnfix=True, bank_growth_control=None, final_external=False,
                  learnfix2=False, **kwargs):
    """WM-guided p20 (clean5c code; B = WMVERIFY-verified p20 WM plans, rule fallback labelled, R = none;
    B and R the same templates)."""
    cells = _check_p20_cells(cells)
    guidance_doc = cells_view(guidance_doc, cells)
    variant = _p20_variant(VARIANT_C, cells, g4_workers, label='clean5c-p20', wm=True, doc=guidance_doc,
                           overrides=_c_overrides(g4_workers, learn_threads_max), learnfix=learnfix,
                           bank_growth_control=bank_growth_control, final_external=final_external, learnfix2=learnfix2)
    return c5.prepare(source, root=root, variant=variant, guidance_doc=guidance_doc, **kwargs)
