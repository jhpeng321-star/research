"""AMENDMENT 41 (MAIN 2026-11-10 ~21:15; registered before any A41 SUMO / seed / outcome): D3200_p20 total-delay win
inside the 10 s group bound. Exploratory follow-up at a non-design cell; never pooled.

Question: is there a smaller bank built ONLY from the five A37-admitted D3200_p20 plans that keeps a total-delay
advantage over DUO_HOLD while every registered group stays within 10 s of U0, on worlds never used before?

- Candidates (fixed, mechanical, 8): A37 CELL_BANK priority order P (bank file f96f8c8e); S1..S5 = each plan of P alone
  (arm A37_PLAN on the A37 bank file, family = P[i]); N2, N3, N4 = the first 2/3/4 plans of P as a bank (arm CELL_BANK on
  a derived prefix bank, same rows / templates / priority). Executor = the unchanged A37 CELL_BANK / A37_PLAN executor
  (BANK_ALWAYS start rule; a37 digest 0a902437; LT 41fc8d1a and a32 1c51f22c unchanged).
- Stage 1 SELECTION (setting a41s_D3200_p20, 48 worlds 991724-991771): arms U0 + 8 candidates (NO DUO_HOLD, no LT),
  vehicle logs on. Per candidate vs U0 (paired): total delay J and the five registered groups (amendment 13, U0
  labels). Rule: eligible-A = every registered group's mean increase has CI95 upper <= 10 s (paired percentile bootstrap
  10,000, seed 20261104); if none, eligible-B = every group's mean increase (point) <= 10 s; if none, STOP ("no bank
  inside the bound", no stage 2). Among the eligible set the lowest mean J; ties -> fewer plans, then earlier in
  S1..S5, N2..N4. One pick -> sealed SELECTION_A41.json (one shot) before stage 2.
- Stage 2 CONFIRMATION (setting a41c_D3200_p20, 192 worlds 998400-998591): arms U0, DUO_HOLD, LEARNED_TRIGGER (context),
  SEL (the stage-1 pick). Refuses to build / reserve / run without the sealed SELECTION receipt (or with a STOP receipt).
- PRIMARY (conjunction): (a) SEL - DUO_HOLD total delay CI95 upper < 0 AND (b) SEL passes the registered group rule vs
  U0 (five groups, each mean increase <= 10 s, U0 labels) on the confirmation panel.
- Plant identity gate first (U0 on p20c 993100/1 and a35 D3200 998100/1 vs ops/a37/IDENTITY_REFS.json f049f546;
  |dJ| <= 1e-6 and identical safety). FAIL -> stop, MAIN decides.
- Budget: ONE ledger phase d3200bound_a41, cap = own-root allocation 1,560,000 sim-s (identity 4,800 + stage 1 518,400 +
  stage 2 921,600 + retry reserve 115,200); each episode precharged in <final_root>/CHARGES.jsonl when its dispatch is
  reserved. One retry per failed episode (same world, charged; REPLAY_AUTHORIZED_*); never an outcome-based retry.

Commands (python ops/a41_d3200bound.py <cmd> --final-root <root>):
  build-candidates  (laptop, zero physics) write ops/a41/CANDIDATES_A41.json from the A37 bank file (sha-verified)
  prepare           zero physics: a41_banks/ (A37 bank copy + N2..N4 prefix banks), shas == CANDIDATES_A41.json
  dry               zero physics: every spec of stage 1 (+ stage 2 when SELECTION exists), bindings, digests, budget
  budget            zero physics: the registered amount vs the cap and the ledger
  identity          U0 on the 4 identity worlds;  identity-compare -> IDENTITY_GATE.json (FAIL -> chain stops)
  select-run        stage 1 episodes (U0 + 8 candidates on 48 worlds)
  select            zero physics: the registered selection rule -> sealed SELECTION_A41.json (one shot)
  reselect          zero physics (laptop): recompute the selection and compare with the sealed receipt
  confirm-run       stage 2 episodes (refused without a sealed eligible SELECTION receipt)
  reports           zero physics: vehicles_report.json next to every published episode log (--stage 1|2)
  analyse           stage 2 primary conjunction + secondaries -> A41_RESULTS_D3200_p20.json (--out)
  status            coverage + ledger totals
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import nm_finaleval as N  # noqa: E402
import a32_ablation as A  # noqa: E402
import a35_transfer as T  # noqa: E402
import a37_celltailor as C37  # noqa: E402  (registers check_a37 for every arm used here)

A41_PHASE = 'd3200bound_a41'
A41_CAP_SIM_S = 1_560_000
EPISODE_SIM_S = 1200
STOP_A41 = 'STOP_A41'
HOST_BASE = '/root/a41db'
HOST_ROOT = f'{HOST_BASE}/root'
#: the A37 host bank file (MAIN registration: bank file f96f8c8e); a laptop copy is committed in ops/a41/
A37_HOST_BANK = '/root/a37ct/root/a37_banks/A37_CELL_BANK_D3200_p20.json'
A41_DIR = HERE/'a41'
A37_BANK_NAME = 'A37_CELL_BANK_D3200_p20.json'
A37_BANK_SHA256 = 'f96f8c8e16810ca5cea19ae7169a95013c8c091fb0fce718986525320edb32cd'
A37_ADMISSION_RECEIPT = '6ee853cc072115da61abb4045b759caeece49de6a468c3f8cfb3d02f854802b7'
#: frozen executors (registration): unchanged digests
A37_DIGEST = '0a9024371e0ea0686df8ac8ccdfa189fcceb1b413fb2604b31c0cf3a169707cf'
LT_DIGEST = '41fc8d1a2fb0ef547524a9d36488b7bbb4bc9f0e8c1e2d06f5fe85eb5903b54d'
A32_DIGEST = '1c51f22c1a0fc29b90a263504f97bd8f0bdc8ad3c4c5d84eb6c015ef46aa12cf'
CELL = 'D3200_p20'
BANK_DIR = 'a41_banks'
#: A37 CELL_BANK priority order P (registration text, prefixes of the full family names in the bank file)
P_SHORT = ('mut_ad6205c9', 'mut_b14f5c97', 'mut_82ad29f5', 'mut_a3534d24', 'mut_b1a94539')
LABELS = ('S1', 'S2', 'S3', 'S4', 'S5', 'N2', 'N3', 'N4')
SELECT_SETTING = 'a41s_D3200_p20'
CONFIRM_SETTING = 'a41c_D3200_p20'
SETTINGS = (SELECT_SETTING, CONFIRM_SETTING)
STAGE2_ARMS = ('U0', 'DUO_HOLD', 'LEARNED_TRIGGER', 'SEL')
FROZEN_ARMS = ('U0', 'DUO_HOLD', 'LEARNED_TRIGGER')
IDENT_WAVE = 'A41_IDENT_U0'
#: identity panels (registration: p20c 993100/1 and a35 D3200 998100/1) -> IDENTITY_REFS keys 'p20c:' / 'a35d3200:'
IDENT_PANELS = tuple(x for x in C37.IDENT_PANELS if x[0] in ('p20c', 'a35d3200'))
IDENTITY_REFS_SHA = 'f049f546bd344b91adb4a6a07933f99e029e703e5478b0296323b05303a2bbdf'   # receipt_sha256 of ops/a37/IDENTITY_REFS.json
GROUP_BOUND_S = 10.0
SELECTION_FILE = 'SELECTION_A41.json'
AUTHORITY = ('MAIN 2026-11-10 ~21:15 AMENDMENT 41 (owner "yes register the 3200 experiment"): D3200_p20 total-delay win '
             'inside the 10 s group bound; 8 mechanical subsets of the A37 bank f96f8c8e; stage 1 selection 991724-991771, '
             'stage 2 confirmation 998400-998591; phase d3200bound_a41 cap 1,560,000 sim-s')
SELECTION_RULE = ('eligible-A := every registered group (origin:L0/L1/L2, class:hdv, class:cav; U0 labels) mean '
                  'increase vs U0 has 95% CI upper <= 10 s (paired percentile bootstrap 10,000, seed 20261104); if no '
                  'candidate is eligible-A, eligible-B := every group mean increase (point) <= 10 s; if none, STOP ("no '
                  'bank inside the bound", no stage 2). Among the eligible set pick the lowest mean J; ties -> fewer '
                  'plans, then earlier in S1..S5, N2, N3, N4. No DUO_HOLD arm in stage 1.')
PRIMARY_RULE = ('SEL beats the hand-written hold inside the group bound at D3200_p20 iff (a) SEL - DUO_HOLD total delay '
                'CI95 upper < 0 (paired percentile bootstrap 10,000, seed 20261104; diff = treatment - reference, '
                'negative = less delay) AND (b) SEL passes the registered group rule vs U0 (five groups, each mean '
                'increase <= 10 s, U0 labels) on this panel; either part failing = NOT SUPPORTED')
REPLAY_AUTHORITY = ('MAIN AMENDMENT 41 registered rule: one retry per failed episode on the same world (charged); no '
                    'outcome-based retries')
SECONDARY = (('SEL', 'U0'), ('SEL', 'LEARNED_TRIGGER'), ('DUO_HOLD', 'U0'), ('LEARNED_TRIGGER', 'U0'),
             ('LEARNED_TRIGGER', 'DUO_HOLD'))


def planned_budget():
    """The registered amount (sim-s): identity 4 x 1,200 + stage 1 9 x 48 x 1,200 + stage 2 4 x 192 x 1,200."""
    ident = len(IDENT_PANELS)*N.IDENTITY_WORLDS*EPISODE_SIM_S
    s1 = (1+len(LABELS))*48*EPISODE_SIM_S
    s2 = len(STAGE2_ARMS)*192*EPISODE_SIM_S
    total = ident+s1+s2
    reserve = A41_CAP_SIM_S-total
    return dict(identity=ident, stage1=s1, stage2=s2, planned=total, cap=A41_CAP_SIM_S, retry_reserve=reserve,
                retry_episodes_covered=reserve//EPISODE_SIM_S, fits=total <= A41_CAP_SIM_S,
                note='own-root ledger: allocation (total over every phase) = phase cap = 1,560,000')


def setup(setting):
    if setting not in SETTINGS:
        raise N.Blocked(f'setting is one of {SETTINGS}')
    N.apply_setting(setting)
    from tools.coopmwm import map_scenarios
    from tools.coopmwm.nm_eval_runner import install_eval_ranges
    install_eval_ranges(map_scenarios)
    return N.load_panel(N.PANEL_FILE)


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# ------------------------------------------------------------------------------------------------ candidates / banks

def load_a37_bank(path=None):
    """The A37 CELL_BANK D3200_p20 bank (file sha f96f8c8e re-verified); rows in priority order == P."""
    path = Path(path or A41_DIR/A37_BANK_NAME)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != A37_BANK_SHA256:
        raise N.Blocked(f'{path}: not the A37 CELL_BANK bank f96f8c8e')
    bank = json.loads(raw)
    fams = [r['family'] for r in bank['rows']]
    if len(fams) != 5 or [f[:len(s)] for f, s in zip(fams, P_SHORT)] != list(P_SHORT):
        raise N.Blocked(f'A37 bank rows {fams} are not the registered priority order P')
    if [t['family'] for t in bank['clean5c']['templates']] != fams or any(r['regime'] != CELL for r in bank['rows']):
        raise N.Blocked('A37 bank templates / regime differ from its rows')
    if C37.bank_text(bank).encode('utf-8') != raw:
        raise N.Blocked('A37 bank text is not the bank_text form')
    return bank


def prefix_bank(bank, k):
    """The first ``k`` rows of P (same rows, templates, priority values and order; nothing else changed)."""
    fams = [r['family'] for r in bank['rows']][:k]
    return dict(bank, rows=deepcopy(bank['rows'][:k]),
                clean5c=dict(bank['clean5c'], templates=[deepcopy(t) for t in bank['clean5c']['templates']
                                                        if t['family'] in fams]))


def build_candidates(bank):
    """The 8 registered candidates (mechanical): S1..S5 single plans (A37_PLAN on the A37 bank), N2..N4 prefixes
    (CELL_BANK on derived prefix banks). -> sealed doc (doc_sha256)."""
    fams = [r['family'] for r in bank['rows']]
    cands, banks = [], {A37_BANK_NAME: A37_BANK_SHA256}
    for i, f in enumerate(fams):
        cands.append(dict(label=f'S{i+1}', kind='single', order=i, n_plans=1, families=[f], arm='A37_PLAN', family=f,
                          bank=A37_BANK_NAME, bank_file_sha256=A37_BANK_SHA256))
    for k in (2, 3, 4):
        name = f'A41_N{k}_{CELL}.json'
        text = C37.bank_text(prefix_bank(bank, k))
        banks[name] = sha_text(text)
        cands.append(dict(label=f'N{k}', kind='nested_prefix', order=len(cands), n_plans=k, families=fams[:k],
                          arm='CELL_BANK', family=None, bank=name, bank_file_sha256=banks[name]))
    if [c['label'] for c in cands] != list(LABELS):
        raise N.Blocked('candidate labels differ from the registration')
    body = dict(schema='coopmwm.a41.candidates.v1', amendment='41', cell=CELL, priority_order_P=fams,
                a37_bank=dict(name=A37_BANK_NAME, file_sha256=A37_BANK_SHA256, host_path=A37_HOST_BANK,
                              admission_receipt_sha256=A37_ADMISSION_RECEIPT),
                executor=dict(a37_sha256=A37_DIGEST, learned_trigger_sha256=LT_DIGEST, a32_sha256=A32_DIGEST,
                              start_rule='BANK_ALWAYS (a32_arms.AlwaysCore._gate)'),
                rule=('S1..S5 = each plan of P alone; N2, N3, N4 = the first 2, 3, 4 plans of P as a bank (same priority '
                      'order); the 5-plan bank is A37 CELL_BANK itself (not a candidate). No new plan, no parameter '
                      'change, no start cap or hand-set threshold.'),
                candidates=cands, banks=banks)
    return dict(body, doc_sha256=hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest())


def candidates(path=None):
    doc = N.read(path or A41_DIR/'CANDIDATES_A41.json')
    body = {k: v for k, v in doc.items() if k != 'doc_sha256'}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != doc['doc_sha256']:
        raise N.Blocked('CANDIDATES_A41: doc_sha256 differs (never edit a pre-outcome file)')
    if doc['a37_bank']['file_sha256'] != A37_BANK_SHA256 or [c['label'] for c in doc['candidates']] != list(LABELS):
        raise N.Blocked('CANDIDATES_A41: not the registered candidate set')
    if doc['executor']['a37_sha256'] != A37_DIGEST:
        raise N.Blocked('CANDIDATES_A41: executor digest differs')
    return doc


def candidate(label, doc=None):
    doc = doc or candidates()
    return next(c for c in doc['candidates'] if c['label'] == label)


def bank_path(final_root, name):
    return Path(final_root)/BANK_DIR/name


def prepare(final_root, a37_bank=None):
    """Zero physics: a41_banks/ = the A37 bank copy (sha f96f8c8e) + the N2..N4 prefix banks, each written once and
    checked against CANDIDATES_A41.json. ``a37_bank``: source file (host: /root/a37ct/root/a37_banks/..., else the
    committed ops/a41 copy)."""
    doc = candidates()
    src = Path(a37_bank) if a37_bank else (Path(A37_HOST_BANK) if Path(A37_HOST_BANK).is_file() else A41_DIR/A37_BANK_NAME)
    bank = load_a37_bank(src)
    if [r['family'] for r in bank['rows']] != doc['priority_order_P']:
        raise N.Blocked('A37 bank rows differ from CANDIDATES_A41 P')
    out = {}
    texts = {A37_BANK_NAME: C37.bank_text(bank)}
    texts.update({f'A41_N{k}_{CELL}.json': C37.bank_text(prefix_bank(bank, k)) for k in (2, 3, 4)})
    for name, text in texts.items():
        if sha_text(text) != doc['banks'][name]:
            raise N.Blocked(f'{name}: derived bank sha differs from CANDIDATES_A41')
        p = C37.write_once(bank_path(final_root, name), text)
        if N.file_sha(p) != doc['banks'][name]:
            raise N.Blocked(f'{p}: written bank differs')
        out[name] = dict(path=str(p), file_sha256=doc['banks'][name])
    return dict(source=str(src), banks=out)


def bank_entry(final_root, name, *, check_files=True):
    doc = candidates()
    want = doc['banks'][name]
    p = bank_path(final_root, name)
    if check_files and (not p.is_file() or N.file_sha(p) != want):
        raise N.Blocked(f'{p}: missing or not the registered bank (run prepare)')
    return dict(path=f'{HOST_ROOT}/{BANK_DIR}/{name}', file_sha256=want)


def candidate_specs(cand, panel, final_root, tag, *, check_files=True):
    """The A37 executor specs of one candidate (C37._a37_specs: the registered LT specs, arm -> A37 arm)."""
    entry = bank_entry(final_root, cand['bank'], check_files=check_files)
    specs = C37._a37_specs(cand['arm'], panel, final_root, entry, family=cand['family'], tag=tag)
    for s in specs:
        if s['checkpoint']['a37_sha256'] != A37_DIGEST:
            raise N.Blocked('a37_cellbank digest differs from the registered 0a902437')
    return specs


# ------------------------------------------------------------------------------------------------ waves

def ident_specs(final_root):
    u0, _ = A.bindings(final_root)
    plant = Path(final_root)/N.PRERUN_PLANT
    out = []
    for tag, name, pin in IDENT_PANELS:
        panel = N.verify(N.read(HERE/name), 'panel_sha256', name)
        if panel['panel_sha256'] != pin:
            raise N.Blocked(f'identity panel {name}: sha differs from the registered panel')
        specs = N.episode_specs('U0', panel, None, plant, 'mappo_frozen', None, u0)[:N.IDENTITY_WORLDS]
        out += [dict(s, job_id=f'a41i_{tag}_u0_{i:04}') for i, s in enumerate(specs)]
    return out


def stage1_waves(panel, final_root, *, check_files=True):
    """[(wave, 'episode', specs)]: U0 then S1..S5, N2..N4 on the 48 selection worlds. No DUO_HOLD / LT (registered)."""
    if panel['panel_sha256'] != N.SETTINGS_PINS[SELECT_SETTING]['PANEL_SHA256'] or len(panel['worlds']) != 48:
        raise N.Blocked('stage 1 runs on the registered 48-world selection panel only')
    u0, _ = A.bindings(final_root)
    plant = Path(final_root)/N.PRERUN_PLANT
    waves = [('A41S_U0', 'episode', C37._retag(N.episode_specs('U0', panel, None, plant, 'mappo_frozen', None, u0),
                                                'a41s_u0'))]
    for cand in candidates()['candidates']:
        waves.append((f'A41S_{cand["label"]}', 'episode',
                       candidate_specs(cand, panel, final_root, f'a41s_{cand["label"].lower()}', check_files=check_files)))
    arms = {s['arm'] for _, _, specs in waves for s in specs}
    if arms - {'U0', 'A37_PLAN', 'CELL_BANK'}:
        raise N.Blocked(f'stage 1 never runs {sorted(arms - {"U0", "A37_PLAN", "CELL_BANK"})}')
    return waves


def selection(final_root):
    p = Path(final_root)/SELECTION_FILE
    return N.verify(N.read(p), 'receipt_sha256', p.name) if p.is_file() else None


def require_selection(final_root):
    """Stage 2 refuses to start without the sealed SELECTION receipt of THIS registration (and on STOP)."""
    sel = selection(final_root)
    if sel is None:
        raise N.Blocked(f'stage 2 refused: no sealed {SELECTION_FILE} (run select-run + select first)')
    doc = candidates()
    if (sel.get('schema') != 'coopmwm.a41.selection.v1' or sel.get('candidates_doc_sha256') != doc['doc_sha256']
            or sel.get('panel_sha256') != N.SETTINGS_PINS[SELECT_SETTING]['PANEL_SHA256']):
        raise N.Blocked('stage 2 refused: SELECTION receipt is not the registered stage-1 selection')
    if sel['decision'] == 'STOP' or sel.get('pick') is None:
        raise N.Blocked('stage 2 refused: stage 1 decided STOP ("no bank inside the bound")')
    if sel['decision'] not in ('eligible-A', 'eligible-B') or sel['pick']['label'] not in LABELS:
        raise N.Blocked('stage 2 refused: SELECTION decision malformed')
    if candidate(sel['pick']['label'], doc) != sel['pick']['candidate']:
        raise N.Blocked('stage 2 refused: SELECTION pick differs from the registered candidate')
    return sel


def stage2_specs(arm, panel, final_root, *, check_files=True):
    if panel['panel_sha256'] != N.SETTINGS_PINS[CONFIRM_SETTING]['PANEL_SHA256'] or len(panel['worlds']) != 192:
        raise N.Blocked('stage 2 runs on the registered 192-world confirmation panel only')
    if arm in FROZEN_ARMS:
        require_selection(final_root)
        return C37._retag(T._arm_specs(arm, panel, final_root, check_files=check_files), f'a41c_{arm.lower()}')
    if arm == 'SEL':
        sel = require_selection(final_root)
        return candidate_specs(sel['pick']['candidate'], panel, final_root, 'a41c_sel', check_files=check_files)
    raise N.Blocked(f'no A41 stage-2 arm {arm!r}')


def stage2_waves(panel, final_root, *, check_files=True):
    require_selection(final_root)
    return [(f'A41C_{a}', 'episode', stage2_specs(a, panel, final_root, check_files=check_files)) for a in STAGE2_ARMS]


# ------------------------------------------------------------------------------------------------ runtime

def charged(final_root, prefix=None):
    rows = N.fe.ledger_rows(final_root)
    return sum(r['sim_s'] for r in rows if prefix is None or str(r['id']).startswith(prefix))


class A41Runtime(N.PrerunRuntime):
    """The pre-run plant of the mirror root, own ledger phase d3200bound_a41 (cap = allocation = 1,560,000); per-episode
    rows precharged at reservation; stage-2 jobs (a41c_*) refused without the sealed SELECTION receipt."""
    phase = A41_PHASE

    def _caps(self, amount):
        if charged(self.final_root)+amount > A41_CAP_SIM_S:
            raise N.Blocked(f'A41: {A41_PHASE} cap {A41_CAP_SIM_S} would be exceeded')

    def check_capacity(self, amount):
        self._caps(amount)

    def reserve(self, cfg, jobs):
        stages = {str(j['id']).split('_')[0] for j in jobs}
        if len(stages) != 1 or not stages <= {'a41i', 'a41s', 'a41c'}:
            raise N.Blocked(f'A41: a dispatch holds one stage only (got {sorted(stages)})')
        if stages == {'a41c'}:
            require_selection(self.final_root)
        self._caps(sum(j['sim_s'] for j in jobs))
        N.own_reserve(self.final_root, jobs, A41_PHASE, cap=A41_CAP_SIM_S, calls=self.calls, allocation=A41_CAP_SIM_S)


def execute(final_root, waves, *, workers, chunk, log, panel=None, runtime=None):
    """a37_celltailor.execute with the A41 runtime / STOP_A41: pending waves in chunks of ``chunk`` episodes."""
    final_root = Path(final_root)
    runtime = runtime or A41Runtime(N.Runtime(N.RUNNER), final_root)
    plant = final_root/N.PRERUN_PLANT
    cfg = runtime.config(plant)
    if runtime.physics:
        if not os.environ.get('COOPMWM_MAP_SPEC'):
            raise N.Blocked('set COOPMWM_MAP_SPEC before physics')
        status = runtime.runner_status()
        if not status['ok'] or not status.get('learned'):
            raise N.Blocked(f'runner cannot run the A41 arms: {status}')
        from tools.coopmwm import nm_eval_runner as R
        if getattr(R, 'A37_ARMS', None) != C37.A37_ARMS or getattr(R, 'A41_CONFIRM_RANGE', None) != (998400, 998599):
            raise N.Blocked('runner lacks the amendment-37 arms / amendment-41 range')
        if panel is not None:
            N.map_check(N.bind_map_api(None), panel)
    plan = N.prerun_plan(final_root)
    u0, _ = A.bindings(final_root)
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise N.Blocked('the U0 binding residual is not the pre-run u0 copy')

    def coverage():
        return {name: (N.published(final_root, name, specs, C37.check_a37)[2], len(specs)) for name, _, specs in waves}
    rounds = 0
    while True:
        cov = coverage()
        pending = [w for w in waves if cov[w[0]][0] < cov[w[0]][1]]
        if not pending:
            break
        if (final_root/STOP_A41).exists():
            log(dict(stopped=True, coverage=cov))
            return dict(status='STOPPED', coverage=cov)
        budget, limits, sel = chunk, {}, []
        for name, kind, specs in pending:
            take = min(cov[name][1]-cov[name][0], budget)
            sel.append((name, kind, specs))
            limits[name] = take
            budget -= take
            if budget == 0:
                break
        N.run_waves(sel, root=plant, final_root=final_root, cfg=cfg, runtime=runtime, workers=workers,
                    base='mappo_frozen', chunked=True, limit=limits)
        rounds += 1
        log(dict(round=rounds, published={k: v[0] for k, v in coverage().items()}, charged_sim_s=charged(final_root)))
    return dict(status='PASS', coverage=coverage(), charged_sim_s=charged(final_root))


def execute_with_retry(final_root, waves, *, log, **kw):
    """One retry per failed episode (registered): a failed dispatch leaves charged-unpublished jobs; if none of them
    was retried before, authorise ONE replay round (fresh <job>__r1 dirs, charged <job>#r1) and run again."""
    for attempt in range(3):
        try:
            return execute(final_root, waves, log=log, **kw)
        except Exception as e:  # noqa: BLE001
            jobs = N.interrupted_jobs(final_root)
            used = N._rounds_used(final_root)
            log(dict(failure=f'{type(e).__name__}: {str(e)[:500]}', interrupted=len(jobs), attempt=attempt))
            if not jobs or any(used.get(j, 0) >= 1 for j in jobs) or attempt == 2:
                raise
            value = N.authorize_replay(final_root=final_root, authority=REPLAY_AUTHORITY,
                                       reason=f'{type(e).__name__}: {str(e)[:300]}')
            log(dict(replay_authorized=len(value['jobs']), receipt=value['receipt_sha256']))
    raise N.Blocked('unreachable')


# ------------------------------------------------------------------------------------------------ identity

def identity_compare(final_root, refs_path=None):
    refs = N.read(refs_path or HERE/'a37'/'IDENTITY_REFS.json')
    if refs.get('receipt_sha256') != IDENTITY_REFS_SHA:
        raise N.Blocked('IDENTITY_REFS is not f049f546')
    N.verify(refs, 'receipt_sha256', 'IDENTITY_REFS')
    specs = ident_specs(final_root)
    got, _, cov = N.published(final_root, IDENT_WAVE, specs, C37.check_a37)
    if cov != len(specs):
        raise N.Blocked(f'{IDENT_WAVE}: {cov}/{len(specs)} published')
    rows = []
    for s, r in zip(specs, got):
        tag = s['job_id'].split('_')[1]
        ref = refs['worlds'][f'{tag}:{r["world_id"]}']
        rows.append(dict(panel=tag, world_id=r['world_id'], j_host=r['J_veh_s'], j_ref=ref['J_veh_s'],
                         abs_diff=abs(r['J_veh_s']-ref['J_veh_s']), safety_equal=r['safety'] == ref['safety'],
                         ref_source=ref['source']))
    ok = all(x['abs_diff'] <= 1e-6 and x['safety_equal'] for x in rows)
    return N.seal(dict(schema='coopmwm.a41.identity.v1', rule='|J diff| <= 1e-6 veh-s and identical safety, U0',
                       refs_receipt_sha256=refs['receipt_sha256'], verdict='PASS' if ok else 'FAILED', worlds=rows,
                       consequence='PASS: continue. FAILED: stop, MAIN decides (registration)'), 'receipt_sha256')


# ------------------------------------------------------------------------------------------------ selection

def group_table(groups):
    """{group: (increase_mean_s, ci95_upper)} for the five registered groups; None when a group is unavailable."""
    out = {}
    for g in N.CO_PRIMARY_GROUPS:
        v = (groups or {}).get('groups', {}).get(g) or {}
        out[g] = (float(v['increase_mean_s']), float(v['ci95'][1])) if 'increase_mean_s' in v and v.get('ci95') else None
    return out


def selection_rule(rows):
    """The registered rule on rows = [dict(label, order, n_plans, mean_J, groups={g: (mean, ci_upper) | None})].
    A group that is unavailable makes the candidate ineligible (every registered group must be shown inside the bound).
    -> dict(decision, eligible_A, eligible_B, pick)."""
    def ok(r, which):
        gs = [r['groups'].get(g) for g in N.CO_PRIMARY_GROUPS]
        if any(x is None for x in gs):
            return False
        return all((x[1] if which == 'A' else x[0]) <= GROUP_BOUND_S for x in gs)
    el_a = [r['label'] for r in rows if ok(r, 'A')]
    el_b = [r['label'] for r in rows if ok(r, 'B')]
    if el_a:
        decision, pool = 'eligible-A', el_a
    elif el_b:
        decision, pool = 'eligible-B', el_b
    else:
        return dict(decision='STOP', eligible_A=el_a, eligible_B=el_b, pick=None,
                    report='no bank inside the bound (no stage 2)')
    best = min((r for r in rows if r['label'] in pool), key=lambda r: (r['mean_J'], r['n_plans'], r['order']))
    return dict(decision=decision, eligible_A=el_a, eligible_B=el_b, pick=best['label'])


def select(final_root, panel):
    """Stage 1 -> sealed SELECTION_A41.json (one shot). Reads only the stage-1 waves (U0 + 8 candidates)."""
    final_root = Path(final_root)
    if selection(final_root) is not None:
        raise N.Blocked(f'{SELECTION_FILE} exists (one shot)')
    value = compute_selection(final_root, panel)
    N.publish(final_root/SELECTION_FILE, value)
    return value


def reselect(final_root, panel):
    """Zero physics: recompute the selection from the published stage-1 waves and compare with the sealed receipt
    (laptop re-analysis; the receipt is never rewritten)."""
    host = selection(final_root)
    v = compute_selection(final_root, panel)
    keys = ('decision', 'pick', 'eligible_A', 'eligible_B', 'candidates_doc_sha256', 'panel_sha256')
    return dict(recomputed_decision=v['decision'], recomputed_pick=(v['pick'] or {}).get('label'),
                host_receipt_sha256=(host or {}).get('receipt_sha256'),
                identical=host is not None and all(v[k] == host[k] for k in keys),
                recomputed_receipt_sha256=v['receipt_sha256'])


def compute_selection(final_root, panel):
    import numpy as np
    final_root = Path(final_root)
    doc = candidates()
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('panel bootstrap is not the registered one')
    waves = stage1_waves(panel, final_root)
    have, sources = {}, {}
    for (name, _, specs), arm in zip(waves, ['U0']+list(LABELS)):
        have[arm], sources[arm] = C37._load(final_root, name, specs)
    reports = C37._reports(final_root, have, worlds)
    rows, table = [], []
    for cand in doc['candidates']:
        lab = cand['label']
        cmp_ = N._bank_compare(have, reports, worlds, lab, 'U0', bootstrap)
        mean_j = float(np.mean([have[lab][w]['J_veh_s'] for w in worlds]))
        g = group_table(cmp_['groups'])
        rows.append(dict(label=lab, order=cand['order'], n_plans=cand['n_plans'], mean_J=mean_j, groups=g))
        acts = [have[lab][w].get('a37') or {} for w in worlds]
        fam = {}
        for a in acts:
            for f, n in (a.get('activations_by_family') or {}).items():
                fam[f] = fam.get(f, 0)+int(n)
        table.append(dict(label=lab, families=cand['families'], n_plans=cand['n_plans'], mean_J_veh_s=mean_j,
                          total_minus_u0=cmp_['total_minus_ref'], groups=cmp_['groups'],
                          group_increase_and_ci95_upper={k: (list(v) if v else None) for k, v in g.items()},
                          tail_cvar90_vs_u0=cmp_['tail_cvar90'], activations_by_family=fam,
                          safety_totals={k: int(sum(have[lab][w]['safety'][k] for w in worlds)) for k in N.SAFETY}))
    verdict = selection_rule(rows)
    pick = None
    if verdict['pick'] is not None:
        pick = dict(label=verdict['pick'], candidate=candidate(verdict['pick'], doc))
    value = N.seal(dict(schema='coopmwm.a41.selection.v1', amendment='41', stage=1, cell=CELL, rule=SELECTION_RULE,
                        authority=AUTHORITY, panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                        seeds=[panel['worlds'][0]['world']['seed'], panel['worlds'][-1]['world']['seed']],
                        candidates_doc_sha256=doc['doc_sha256'], bootstrap=bootstrap,
                        decision=verdict['decision'], eligible_A=verdict['eligible_A'], eligible_B=verdict['eligible_B'],
                        pick=pick, candidates=table,
                        u0=dict(mean_J_veh_s=float(np.mean([have['U0'][w]['J_veh_s'] for w in worlds])),
                                safety_totals={k: int(sum(have['U0'][w]['safety'][k] for w in worlds)) for k in N.SAFETY}),
                        comparator_note='no DUO_HOLD / LEARNED_TRIGGER arm in stage 1 (the pick cannot see the comparator)',
                        sources={k: v.get('receipt_sha256') for k, v in sources.items()}), 'receipt_sha256')
    return value


# ------------------------------------------------------------------------------------------------ analysis

def primary_verdict(sel_vs_duo, sel_vs_u0):
    """The registered conjunction from two N._bank_compare rows (SEL - DUO_HOLD, SEL - U0)."""
    ci = sel_vs_duo['total_minus_ref']['ci95']
    a = bool(ci[1] < 0)
    b = bool((sel_vs_u0.get('groups') or {}).get('all_pass') is True)
    return dict(rule=PRIMARY_RULE, a_total_vs_duo_ci95=ci, a_total_ci_upper_below_0=a, b_group_rule_vs_u0=b,
                supported=bool(a and b), reading='SUPPORTED' if a and b else 'NOT SUPPORTED',
                failed_parts=[p for p, v in (('a', a), ('b', b)) if not v])


def analyse(final_root, panel):
    """Stage 2: primary conjunction + secondaries (uncorrected). diff = treatment - reference, negative = less delay."""
    import numpy as np
    final_root = Path(final_root)
    sel = require_selection(final_root)
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    if bootstrap != dict(reps=10000, level=.95, seed=20261104, method='paired_percentile'):
        raise N.Blocked('panel bootstrap is not the registered one')
    have, sources = {}, {}
    for name, _, specs in stage2_waves(panel, final_root, check_files=False):
        arm = name[len('A41C_'):]
        have[arm], sources[arm] = C37._load(final_root, name, specs)
    reports = C37._reports(final_root, have, worlds)

    def contrast(arm, ref, kind):
        row = dict(treatment=arm, reference=ref, kind=kind,
                   convention='diff = treatment - reference; negative = treatment has less delay')
        row.update(N._bank_compare(have, reports, worlds, arm, ref, bootstrap))
        for k in ('hard_brakes', 'ttc_events', 'collisions', 'teleports'):
            row[f'{k}_minus_ref'] = N.fe.paired_stats(np.array([have[arm][w]['safety'][k] for w in worlds], float),
                                                      np.array([have[ref][w]['safety'][k] for w in worlds], float),
                                                      bootstrap, pct=False)
        return row
    sel_duo = contrast('SEL', 'DUO_HOLD', 'primary_a')
    sel_u0 = contrast('SEL', 'U0', 'primary_b')
    contrasts = [sel_duo, sel_u0] + [contrast(a, r, 'secondary') for a, r in SECONDARY if (a, r) != ('SEL', 'U0')]
    arms = {}
    for a in STAGE2_ARMS:
        arms[a] = dict(mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                       safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in N.SAFETY})
        rec = [have[a][w].get('a37') or have[a][w].get('a32') or have[a][w].get('learned_trigger') or {} for w in worlds]
        if any(rec):
            fam = {}
            for r in rec:
                for f, n in (r.get('activations_by_family') or {}).items():
                    fam[f] = fam.get(f, 0)+int(n)
            arms[a]['activations'] = dict(worlds_with_any=sum(1 for r in rec if r.get('activations')),
                                          total=sum(int(r.get('activations') or 0) for r in rec), by_family=fam)
    groups_vs_u0 = {c['treatment']: dict(all_pass=(c['groups'] or {}).get('all_pass'),
                                         groups={g: {k: v.get(k) for k in ('increase_mean_s', 'ci95', 'passes')}
                                                 for g, v in (c['groups'] or {}).get('groups', {}).items()})
                    for c in contrasts if c['reference'] == 'U0'}
    return N.seal(dict(schema='coopmwm.a41.analysis.v1', amendment='41', authority=AUTHORITY, cell=CELL,
                       panel_sha256=panel['panel_sha256'], n_worlds=len(worlds), bootstrap=bootstrap,
                       selection=dict(receipt_sha256=sel['receipt_sha256'], decision=sel['decision'],
                                      pick=sel['pick']['label'], families=sel['pick']['candidate']['families']),
                       primary=primary_verdict(sel_duo, sel_u0), contrasts=contrasts, arms=arms,
                       groups_vs_u0_all_arms=groups_vs_u0,
                       lt_group_result=groups_vs_u0.get('LEARNED_TRIGGER'),
                       sources={k: v.get('receipt_sha256') for k, v in sources.items()}, root=str(final_root),
                       label='amendment 41; exploratory follow-up at a non-design cell; one panel, one shot; never pooled'),
                  'analysis_sha256')


# ------------------------------------------------------------------------------------------------ CLI

def stage_waves(stage, panel, root, *, check_files=True):
    if stage == 1:
        return stage1_waves(panel, root, check_files=check_files)
    return stage2_waves(panel, root, check_files=check_files)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('build-candidates', 'prepare', 'dry', 'budget', 'identity', 'identity-compare',
                                       'select-run', 'select', 'reselect', 'confirm-run', 'reports', 'analyse', 'status'))
    p.add_argument('--final-root')
    p.add_argument('--stage', type=int, choices=(1, 2), default=1)
    p.add_argument('--workers', type=int, default=46)
    p.add_argument('--chunk', type=int, default=46)
    p.add_argument('--a37-bank', help='prepare / build-candidates: the A37 bank file (sha f96f8c8e re-verified)')
    p.add_argument('--out')
    a = p.parse_args(argv)
    if a.command == 'build-candidates':
        out = Path(a.out or A41_DIR/'CANDIDATES_A41.json')
        doc = build_candidates(load_a37_bank(a.a37_bank))
        if out.exists() and N.read(out) != doc:
            raise N.Blocked(f'{out} exists with different content (never overwrite)')
        out.write_text(json.dumps(doc, indent=1, sort_keys=True)+'\n', encoding='utf-8', newline='\n')
        print(json.dumps(dict(doc_sha256=doc['doc_sha256'], banks=doc['banks'])))
        return
    if a.command == 'budget' and not a.final_root:
        print(json.dumps(planned_budget(), indent=1))
        return
    if not a.final_root:
        p.error('--final-root required')
    root = Path(a.final_root)
    setting = CONFIRM_SETTING if a.command in ('confirm-run', 'analyse') or (
        a.command in ('reports', 'status') and a.stage == 2) else SELECT_SETTING
    panel = setup(setting)

    def log(x):
        line = json.dumps(dict(x, unix=time.time(), t=time.strftime('%F %T'), setting=setting), sort_keys=True,
                          default=str)
        print(line, flush=True)
        with (root/'A41_LOG.jsonl').open('a', encoding='utf-8') as f:
            f.write(line+'\n')
    if a.command == 'prepare':
        print(json.dumps(prepare(root, a.a37_bank), indent=1))
    elif a.command == 'budget':
        b = planned_budget()
        b.update(charged=charged(root), remaining=A41_CAP_SIM_S-charged(root))
        print(json.dumps(b, indent=1))
    elif a.command == 'dry':
        from tools.coopmwm import a37_cellbank as C, a32_arms as A32, learned_trigger as L
        digests = dict(a37=C.a37_digest(), a32=A32.a32_digest(), learned_trigger=L.learned_trigger_digest())
        if digests != dict(a37=A37_DIGEST, a32=A32_DIGEST, learned_trigger=LT_DIGEST):
            raise N.Blocked(f'frozen executor digests differ: {digests}')
        out = dict(setting=setting, panel_sha256=panel['panel_sha256'], digests=digests, budget=planned_budget(),
                   candidates_doc_sha256=candidates()['doc_sha256'], ident=[s['job_id'] for s in ident_specs(root)])
        out['stage1'] = {n: dict(n=len(s), first=s[0]['job_id'], seeds=[s[0]['world']['seed'], s[-1]['world']['seed']],
                                 arm=s[0]['arm'], family=s[0]['checkpoint'].get('a37_family'),
                                 bank=(s[0]['checkpoint'].get('a37_bank') or {}).get('path'))
                         for n, _, s in stage1_waves(panel, root)}
        out['stage1_sim_s'] = sum(v['n'] for v in out['stage1'].values())*EPISODE_SIM_S
        try:
            require_selection(root)
            out['stage2'] = 'SELECTION present: `status --stage 2` lists the stage-2 waves'
        except N.Blocked as e:
            out['stage2'] = str(e)
        print(json.dumps(out, indent=1))
    elif a.command in ('identity', 'select-run', 'confirm-run'):
        import fcntl
        with (root/'OWNER.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            N.publish(root/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()), replace=True)
            if a.command == 'identity':
                waves, mp = [(IDENT_WAVE, 'episode', ident_specs(root))], None
            elif a.command == 'select-run':
                waves, mp = stage1_waves(panel, root), panel
            else:
                waves, mp = stage2_waves(panel, root), panel
            log(dict(start=a.command, waves=[(w[0], len(w[2])) for w in waves], workers=a.workers, chunk=a.chunk,
                     pid=os.getpid(), charged_before=charged(root)))
            res = execute_with_retry(root, waves, workers=a.workers, chunk=a.chunk, log=log, panel=mp)
            log(dict(done=a.command, **res))
    elif a.command == 'identity-compare':
        v = identity_compare(root)
        N.publish(root/'IDENTITY_GATE.json', v, replace=True)
        print(json.dumps(dict(verdict=v['verdict'])))
        if v['verdict'] != 'PASS':
            sys.exit(2)
    elif a.command == 'select':
        v = select(root, panel)
        print(json.dumps(dict(decision=v['decision'], pick=(v['pick'] or {}).get('label'),
                              receipt_sha256=v['receipt_sha256'])))
    elif a.command == 'reselect':
        print(json.dumps(reselect(root, panel)))
    elif a.command == 'analyse':
        v = analyse(root, panel)
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        N.publish(out/f'A41_RESULTS_{CELL}.json', v, replace=True)
        print(json.dumps(dict(analysis_sha256=v['analysis_sha256'], reading=v['primary']['reading'])))
    elif a.command == 'status':
        out = dict(charged=charged(root), cap=A41_CAP_SIM_S, stage=a.stage)
        try:
            for n, _, s in stage_waves(a.stage, panel, root, check_files=False):
                out[n] = f'{N.published(root, n, s, C37.check_a37)[2]}/{len(s)}'
        except N.Blocked as e:
            out['note'] = str(e)
        print(json.dumps(out, indent=1))
    else:   # reports
        waves = stage_waves(a.stage, panel, root, check_files=False)
        if a.stage == 1:
            waves.append((IDENT_WAVE, 'episode', ident_specs(root)))
        n = 0
        for name, _, specs in waves:
            recs, _, _ = N.published(root, name, specs, C37.check_a37)
            for r in recs:
                rep = N.episode_log_report(root, r)
                if rep is None or not rep['ok']:
                    raise N.Blocked(f'{r["job_id"]}: report failed')
                n += 1
        print(json.dumps(dict(reports=n)))


if __name__ == '__main__':
    main()
