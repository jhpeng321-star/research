"""clean5a (wz3to2 lane-drop) final paired evaluation on the registered evaluation panel v2.

Port of ops/clean3_finaleval.py (c4/clean3-finaleval 47d7b376) to the wz3to2 map
(assets/research/newmap_candidates/wz3to2; newmap_spec.md N6). MAIN registrations 2026-11-06
(REGISTRATION.md 19:24/19:32 and the N6 block): the paper's final evaluation targets clean5a
(speed-only expected-value cooperation; base = a FROZEN S-MAPPOWZ snapshot, A = that base alone),
run on cpuZ right after clean5a u24; clean4 has no final evaluation. This launcher is the only
final-evaluation runner; its spend is debited from the run's ``v2v_final`` allocation (<= 1,040,000).

  * worlds: panel v2 (``nm_eval_panel_wz3to2_v2.json``, sealed ``panel_sha256`` 3acb2b56 pinned
    below): 96 held-out seeds 989600-989695, N1 cell D3200_p60, 1200 s windows; never TRAIN worlds;
  * arms (same paired worlds, windows and noise seeds): A, B, R at u24 (bound by sha at the gate),
    NATIVE, U0 (the transferred MAPPO u0: d9042cc5 executor + zero residual), MAPPO_SPEED (S-MAPPOWZ
    it168 only) and MAPPO_LANE (S-MAPPOWZL final, never aecd7d70); BASE is an alias (A for the
    frozen-snapshot base, U0 for a mappo_u0 base, NATIVE for a native base);
  * two registered analyses with separate seals: CLAIM 3a = A, B, R, NATIVE, U0 (B vs A, B vs R,
    vs NATIVE, vs U0), opened as soon as those arms finish; CLAIM 3b = each MAPPO comparator, run and
    opened only after that comparator is bound (or recorded as declared unavailable);
  * per-decision synergy: on 8 registered decision worlds (panel repeat % 12 == 0), descriptive
    capture now; branches when the runner provides them (F7);
  * per-seed reporting and a report-only regime split for every arm vs each comparator;
  * every wave is reserved whole in the run ledger (phase v2v_final, hard sub-cap, no foreign
    v2v_final spend) BEFORE dispatch; an interrupted wave is never replayed.
Map geometry/seed ranges/scenario worlds come from the N1 map API (``tools.coopmwm.map_api``);
physics is refused unless it matches the registered panel. Physics goes through ONE runner module
in the deployed source (``RUNNER``; contract in ``RUNNER_CONTRACT``); R = the no-guidance arm.

Commands: register-panel / verify-panel (zero physics, panel), quote, check, register, bind-arms,
bind-mappo, declare-unavailable, run, analyse, dry-run (fake physics, no simulator or process pool).
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean3_finaleval as fe  # noqa: E402  generic guard/read/publish/seal/estimator helpers
from clean3_finaleval import Blocked, NotReady, guard, read, publish, file_sha, seal, verify, digest  # noqa: E402

SCHEMA = 'coopmwm.nm.finaleval.v1'
MAP = 'wz3to2'
UPDATE = 24                     # the final training update (A/B/R COMPLETE receipts exist only there)
#: MAIN registration 2026-11-06 23:10 (pre-launch, pre-results): claim 3a/3c PRIMARY = clean5a u12 (B and R
#: bound by sha at u12; A = frozen U0), SECONDARY = u24 on the same panels; both reported, no selection.
EVAL_UPDATES = (12, 24)
PRIMARY_UPDATE = 12
UPDATE_AUTHORITY = ('MAIN registration 2026-11-06 23:10 (pre-launch, pre-results; owner: answers before noon): claim '
                    '3a/3c PRIMARY = clean5a u12 (B, R bound by sha at u12; A = frozen U0) on panel v2 + 3c with the '
                    'sealed NATIVE/U0/RULE pre-run arms; u24 SECONDARY on the same panels; both reported, no selection')
#: clean5a registered rule text (track2_backend.FROZEN_A_RULE): A = the frozen base, no PPO update
FROZEN_A_RULE = ('clean5a (MAIN 2026-11-06): "A = base alone" - A is the frozen bound base (BASE = the bound '
                 'S-MAPPOWZ snapshot or u0) evaluated as an arm every update on the same worlds; no PPO update; '
                 'only B and R train (their cooperation layer on the same frozen base).')
A_ALIAS_LABEL = ('A = U0: the frozen control arm (no PPO update) carries the u0 weights at every update - verified '
                 'zero-physics at the gate (A policy weights at the update == the common u0); not re-simulated')
ARMS = ('A', 'B', 'R')
DECISION_ARMS = ('B', 'R')
BASES = ('mappo_frozen', 'mappo_u0', 'native')   # clean5a: frozen S-MAPPOWZ snapshot base (A = base alone)
ORDER = ('B', 'A', 'R', 'RULE', 'RULE_FAIR', 'ORACLE', 'ORACLE_WM', 'LEARNED_TRIGGER', 'U0', 'MAPPO_SPEED', 'MAPPO_LANE', 'NATIVE',
         'MAPPO_CT')
#: MAIN 2026-11-06 (FAIRAUDIT F6): 7 arms; BASE is an alias (U0 for a mappo_u0 base, NATIVE for native)
EPISODE_ARMS = ('A', 'B', 'R', 'U0', 'RULE', 'RULE_FAIR', 'MAPPO_SPEED', 'MAPPO_LANE', 'NATIVE', 'ORACLE', 'ORACLE_WM',
                'LEARNED_TRIGGER', 'MAPPO_CT')
COMPARATOR_ARMS = ('MAPPO_SPEED', 'MAPPO_LANE')        # claim 3b, one seal per comparator
#: amendment 30 (MAIN 2026-11-09 15:29, owner-approved): U0's MAPPO speed policy continued for 96 episodes with its
#: original objective; evaluated on the p20c bank root exactly like MAPPO_SPEED (bound export + the bound zero-residual u0)
CT_ARM = 'MAPPO_CT'
BOUND_ARMS = ('U0',)+COMPARATOR_ARMS+(CT_ARM,)          # executors bound by path + sha256
CT_PHASE = 'mappo_ct'
CT_CAP_SIM_S = 192*1200                                 # the amendment-30 eval part (230,400) of the mappo_ct line
CT_AUTHORITY = ('MAIN 2026-11-09 15:29 amendment 30 (owner-approved): MAPPO_CT = U0 MAPPO speed policy + 96 episodes '
                'of its original MAPPO objective at D2800_p20 (training seeds only); sealed on the 192-world p20c '
                'panel against its U0; primary MAPPO_CT - U0; secondary LEARNED_TRIGGER - MAPPO_CT, fairness, CVaR90; '
                'budget line mappo_ct 450,000 (eval 230,400)')
CLAIM3A_ARMS = ('A', 'B', 'R', 'U0', 'RULE', 'RULE_FAIR', 'NATIVE', 'ORACLE', 'ORACLE_WM',
                'LEARNED_TRIGGER')                       # +25b oracle bounds, +26 learned trigger
#: records of these arms carry the executor sha actually loaded (bound executors)
EXECUTOR_ARMS = ('U0', 'RULE', 'RULE_FAIR', 'ORACLE', 'ORACLE_WM', 'LEARNED_TRIGGER')+COMPARATOR_ARMS+('MAPPO_CT',)
RULE_ARMS = ('RULE', 'RULE_FAIR')
#: amendments 25 + 25b (MAIN 2026-11-09 00:06, pre-outcome): privileged upper bounds, NOT deployable. Both = U0 (bound
#: u0 executor, A/0) + oracle_plan on the frozen end-of-training bank (CLEAN5C5 final-update snapshot, bound only
#: after training ends via a sealed ORACLE_BINDING.json). ORACLE: every admitted row (WM + rule, provenance-tagged);
#: ORACLE_WM: WM-sourced rows only. Per-family bank cooldown; unconditional rows act whenever a feasible group exists
#: (any bin, incl. unobserved); situational rows keep priority when admitted and feasible. Primary panel only.
ORACLE_ARMS = ('ORACLE', 'ORACLE_WM')
ORACLE_PLAN_SOURCE = dict(ORACLE='all', ORACLE_WM='wm')
ORACLE_SIM_S = 115_200                                 # per oracle arm: 96 primary worlds x 1,200 s
ORACLE_BINDING = 'ORACLE_BINDING.json'
#: amendment 26 (MAIN 2026-11-09 05:41, pre-outcome): LEARNED_TRIGGER = ORACLE_WM with the privileged-bin admission
#: replaced by the frozen CAV-observable classifier (p >= 0.378785); the insurance / deployability arm. Its model doc
#: (path in the deployed checkout, pinned sha) is sealed in the same ORACLE_BINDING.json as the bank.
LEARNED_ARMS = ('LEARNED_TRIGGER',)
ORACLE_SHA256 = '7409a7b4765651813c4771462b7acbc16386d626900d9a031e19d3a65a9f9d22'   # oracle_plan digest (25b, REV-SPD)
LEARNED_MODEL_SHA256 = '8e0ddd3e241ce0172736909133f0b745c58b6bf795e06ca5364e5755f7e3b720'
LEARNED_TRIGGER_SHA256 = '41fc8d1a2fb0ef547524a9d36488b7bbb4bc9f0e8c1e2d06f5fe85eb5903b54d'
BANK_ARMS = ORACLE_ARMS+LEARNED_ARMS                  # bound to the sealed end-of-training bank; p20x primary only
#: amendment 27 (MAIN 2026-11-09 06:08): the bank arms run NOW (before training ends) on the p20x primary panel in the
#: evaluation root's own ledger, phase v2v_final_bank, cap 3 x 115,200 (bank-run); the bank is frozen at fd385a2b
BANK_PHASE = 'v2v_final_bank'
BANK_CAP_SIM_S = 3*115_200
ORACLE_LABELS = dict(ORACLE='ORACLE: U0 + every admitted verified plan of the frozen bank (WM + rule), privileged '
                            'bins - upper bound, not deployable',
                     ORACLE_WM='ORACLE_WM: U0 + the WM-sourced admitted plans of the frozen bank only, privileged '
                               'bins - upper bound, not deployable',
                     LEARNED_TRIGGER='LEARNED_TRIGGER: insurance / deployability arm - ORACLE_WM plans admitted by a '
                                     'frozen CAV-observable classifier (p >= 0.378785; within-family helped-vs-harmful '
                                     'AUC ~0.5, onset AUC 0.60)')
#: MAIN registration (pre-results): RULE = U0 + the solo2 rule always on (one CAV holding in SUMO lane 2
#: with wall_h parameters, cooldown = hold) = the EXACT S-REGIMEU0 implementation (c4/imitreplay
#: tools/coopmwm/regimeu0.py RuleHook('wall_h_solo2')): the hand-designed cooperation baseline.
RULE_STRATEGY = 'wall_h_solo2'
RULE_LABEL = ('RULE: U0 + the S-REGIMEU0 wall_h_solo2 rule (one CAV holds in SUMO lane 2 at 15 m/s for 90 s, '
              'cooldown = hold), always on')
#: arms independent of the clean5a training: pre-run (sealed) before clean5a launches
PRERUN_ARMS = ('NATIVE', 'U0', 'RULE', 'RULE_FAIR', 'MAPPO_LANE')     # MAPPO_LANE: amendment 17 (MAIN-approved)
#: owner fairness decision (REGISTRATION 2026-11-07 04:0x, pre-results): RULE_FAIR = U0 + the EXACT S-MULTICRIT
#: wall_g_solo2 rule always on (one lane-2 CAV holds 22 m/s for 120 s, dmax 1.0; c4/multicrit blob 4ff09522)
RULE_FAIR_STRATEGY = 'wall_g_solo2'
RULE_FAIR_LABEL = ('RULE_FAIR: U0 + the S-MULTICRIT wall_g_solo2 rule (one CAV holds in SUMO lane 2 at 22 m/s for '
                   '120 s, dmax 1.0, cooldown = hold), always on')
RULE_FAIR_SHA256 = 'a5fdf1c3bf3a9d0dc82a6c296bf09ea6ad4bfa51b288e9c378b4c1ebc9e10376'   # multicrit_digest()
#: co-primary claim-3a criterion (owner decision 04:0x): B's total held-out delay lower than A's AND every group
#: (origin lane 0/1/2 = WM lane at first observation; HDV; CAV) has a mean per-vehicle delay increase <= 10 s vs A
GROUP_DELAY_MAX_S = 10.0
PRERUN_PHASE = 'v2v_final_prerun'
#: rule_digest() of the exact S-REGIMEU0 rule (sumo_lane, pick_wall, RuleHook, Recorder, constants);
#: identical at c4/imitreplay 385c4b8f, 52248a4b and d02228cf (blob 2e267403 / imitreplay d5042c33)
RULE_SHA256 = '887285923ca0b4ac1a7f728f3d687c57b49e944d3ec7abf62bddb4249d6690b4'
RULE_MODULE, RULE_SOURCE = 'tools.coopmwm.regimeu0', 'c4/imitreplay'
PRERUN_FILE = 'PRERUN_PLAN.json'
#: EVALUATION-ONLY ground-truth vehicle log written by every episode (runner VehicleLog)
VEHICLE_LOG = 'vehicles.jsonl.gz'
WARMUP_S = 300
#: one-episode vehicle-log check before any pre-run wave (MAIN: every new mode gets a one-episode check first)
LOGCHECK_PHASE = 'v2v_final_logcheck'
LOGCHECK_SEED, LOGCHECK_STOP_S = 989598, 360            # TRAIN world D3200_p60; 300 s warmup + 60 s logged
LOGCHECK_ARMS = ('NATIVE', 'RULE')                      # the native plant path and the policy (U0 + rule) path
SPEED_LIMIT_MPS = 31.29                                  # traffic_shepherd.c4.env.SPEED_LIMIT_MPS (team-delay term)
DROP_S_M = 75.44                                         # N1 descriptor lane drop (s)
NOMINAL_LENGTH_M = 5.0
#: operator stop: the pre-run finishes the chunk in flight, publishes it and stops (resume anywhere)
STOP_PRERUN = 'STOP_PRERUN'
PRERUN_PLANT = 'prerun_plant'
#: what NATIVE/U0/RULE episodes read from the plant CONFIG (runner + accel_plant + u0 loaders)
PLANT_KEYS = ('runtime', 'checkpoint', 'warm_history', 'accel_bounds', 'v2v', 'v2v_intent', 'common_checkpoint_sha256',
              'initial_checkpoint', 'launch_optimization')
PRERUN_AUTHORITY = ('MAIN registration 2026-11-06 (pre-results): PRE-RUN NATIVE, U0 and RULE for 3a and 3c from about '
                    '21:15 on cpuZ (off cpuZ by 00:30 for clean5a; else cpuV <= 16 workers); results stay SEALED (F12) '
                    'until clean5a A/B/R are bound and the 3a seal opens')
#: MAIN registration (pre-results): secondary claim 3c "condition-aware: no harm in light traffic" -
#: B, R, U0, NATIVE on 24 held-out worlds of the D2400_p60 cell (seeds 989600-989623, N1 world());
#: separate seal, opens with 3a. clean5a's base is u0 (d9042cc5), so A = U0 (BASE alias U0).
CLAIM3C_ARMS = ('B', 'R', 'U0', 'RULE', 'RULE_FAIR', 'NATIVE')
CLAIM3C_CELL = dict(demand=2400, penetration=0.6)
CLAIM3C_KEY = 'D2400_p60'
CLAIM3C_SEED_START = None
PANEL_SIZE_REASON = None
SEEDS_RESERVED = None
#: amendment 23: an evaluation-only held-out seed range added at runtime (None = N1 HELDOUT_RANGE only)
EVAL_EXTRA_RANGE = None
CLAIM3C_WORLDS = 24
FROZEN_MAPPO_SHA = 'd9042cc5a05ee1adc54a720b7482329a59850bddb00444ce29cc70e49cffa9ca'
#: Comparator executors are parameters (bind-mappo: path + sha256, re-verified at load). MAPPO_SPEED =
#: N7's wz3to2 speed-only retrain (S-MAPPOWZ), the transferred d9042cc5 export only as an explicitly
#: bound fallback named "u0 transferred, not retrained"; MAPPO_LANE = the lane-capable wz3to2 MAPPO
#: (S-MAPPOWZL, owner-approved). F12: no evaluation episode runs - so no result exists to be opened -
#: until every comparator is bound or explicitly declared unavailable.
MAPPO_KINDS = {
    'U0': {'u0_transferred_d9042cc5': 'MAPPO u0 transferred (frozen d9042cc5 executor, zero residual)'},
    'MAPPO_SPEED': {'retrained_wz3to2_it168': 'speed-only MAPPO retrained on wz3to2 (S-MAPPOWZ it168)'},
    'MAPPO_LANE': {'lane_retrained_wz3to2_final': 'lane-capable MAPPO retrained on wz3to2 (S-MAPPOWZL final)'},
    'MAPPO_CT': {'u0_continued_d2800_p20_96ep': 'U0 MAPPO speed policy continued 96 episodes, original objective '
                                                '(amendment 30)'}}
#: MAIN 2026-11-06 registered binding: MAPPO-speed = S-MAPPOWZ it168 ONLY (no it84/42/24 substitution);
#: MAPPO-lane = the S-MAPPOWZL final export, never aecd7d70; the transferred d9042cc5 u0 is the U0 arm.
REQUIRED_ITERATION = {'MAPPO_SPEED': 168}
REFUSED_SHA_PREFIXES = {'U0': (), 'MAPPO_SPEED': ('d9042cc5',), 'MAPPO_LANE': ('aecd7d70', 'd9042cc5'),
                        'MAPPO_CT': ('d9042cc5', 'aecd7d70')}
REQUIRED_SHA = {'U0': FROZEN_MAPPO_SHA}
FALLBACK_SHA_PREFIX = 'd9042cc5'
U0_LABEL = 'MAPPO u0 transferred (frozen d9042cc5 executor, zero residual)'
MAPPO_AUTHORITY = ('MAIN 2026-11-06 registered binding: comparators are parameters (path + sha256 verified at '
                   'bind and at every load); MAPPO_SPEED = S-MAPPOWZ it168 only (no it84 substitution); MAPPO_LANE = '
                   'S-MAPPOWZL final, never aecd7d70; U0 = the transferred d9042cc5 u0; F12: no wave runs (nothing '
                   'can be opened) until every comparator is bound or declared unavailable; v2v_final <= 1,040,000')
#: values written into panel v2 at registration (4-arm native quote); the amended quote is quote()
REGISTERED_PANEL_QUOTE = dict(quote_mappo_u0=672_000, quote_native=556_800)
#: F11 registered parity check (MAPPO in the eval plant vs its training env path): 2 TRAIN worlds x
#: 2 paths x 2 comparators x (300 s warmup + 120 s rollout)
PARITY_WORLDS, PARITY_ROLLOUT_S = (989598, 989599), 120
PARITY_SIM_S = len(PARITY_WORLDS)*2*len(COMPARATOR_ARMS)*(300+PARITY_ROLLOUT_S)
#: MAIN decision 2026-11-06 (F7): per-decision branches are NOT built; claims 3a/3c rest on B vs A/U0/RULE/R
DECISIONS_UNAVAILABLE = ('unavailable (not built): MAIN decision 2026-11-06 - F7 per-decision synergy branches are not built; '
                         'claims 3a/3c rest on B vs A/U0/RULE/R on the held-out panels (decision-level attribution not '
                         'required); B/R decision activity is reported descriptively')
PARITY_CELL = dict(demand=3200, penetration=0.6)      # the clean5a / S-MAPPOWZ training cell D3200_p60
PARITY_TAG = 'F11 parity (TRAIN worlds; not a panel)'
#: agreement tolerances (float32 env state vs float64 plant rows; deterministic executors)
PARITY_TOL = dict(s_m=1e-3, v_mps=1e-3, command_mps2=1e-4)
HORIZON_S = 1200                       # = clean4 training episode length (MAIN)
WINDOW = dict(start_s=0, end_s=HORIZON_S)
EVAL_RANGE = (989600, 989699)
TRAIN_RANGE = (989200, 989599)
N_WORLDS = 96
DECISION_STRIDE = 12                   # registered decision subset: panel repeat % 12 == 0 (8 worlds)
DECISION_H = 360
DECISION_MIN_T = 300
DECISION_MAX_T = HORIZON_S-DECISION_H  # 840
MIN_MEMBERS, MAX_MEMBERS = 2, 3
#: clean4 training cell (MAIN 2026-11-06: N3 NEWMAP2_REPORT recommendation, adopted)
CELL = dict(demand=3200, penetration=0.6)
CELL_AUTHORITY = ('MAIN 2026-11-06 (N6): clean4 training cell = demand 3200 veh/h x CAV share 0.6 '
                  '(N3 NEWMAP2_REPORT.md recommendation, adopted)')
NEWMAP2_REPORT_SHA256 = '70c119595ff3f391bbdebfa0a98d9c2f7ca02d2ae61f8acb53c126754adfdb05'
PHASE = 'v2v_final'
PANEL_ALLOCATION_SIM_S = 690_000       # value registered in panel v2's contract
ALLOCATION_SIM_S = 1_600_000           # v2v_final raised by MAIN 2026-11-07 (>= 15% over 1,357,680 incl. write-offs)
ALLOCATION_AUTHORITY = ('MAIN 2026-11-06 under the owner authorisation to act on its recommendations: v2v_final '
                        'raised 690,000 -> 1,040,000 (F6: 7 arms + branches) -> 1,200,000 (RULE arm in 3a and 3c, '
                        'claim 3c) -> 1,340,000 (N5: restore a >= 15% margin over the 1,164,960 ceiling) -> 1,600,000 (MAIN '
                        '2026-11-07: >= 15% over quote 1,241,760 + write-offs 115,200 + logcheck 720); every N6 charge '
                        'is in the evaluation root ledger (own-root accounting)')
BUDGET_KEYS = ('clean4_budget', 'clean3_budget')
SAFETY = fe.SAFETY
HARD = fe.HARD
BOOTSTRAP = dict(reps=10000, level=.95, seed=20261104, method='paired_percentile')
HERE = Path(__file__).resolve().parent
PANEL_FILE = HERE/'nm_eval_panel_wz3to2_v2.json'
#: sealed content digest of nm_eval_panel_wz3to2_v2.json (registered 2026-11-06 before clean4 launch)
PANEL_SHA256 = '3acb2b565fcb61f4978834b3ba47f0b574fe2d5da7ca028210ef75830c19a130'
SUPERSEDED_FILE = HERE/'nm_eval_panel_SUPERSEDED.json'
PANEL_3C_FILE = HERE/'nm_eval_panel_wz3to2_claim3c_D2400.json'
#: sealed content digest of the claim-3c panel (registered before any clean5a result)
PANEL_3C_SHA256 = 'd43510e9c75308b575cf97c4309c68da06555dfa442ffc73f3cbbec04dbef3fb'
#: superseded panels: never valid for a clean4 CONFIG binding (see SUPERSEDED_FILE)
SUPERSEDED = {'b5e1cd2a1d068c5e2bb66bb44005b2dd954e0419b4a6c99778af304aeae6dbc1':
              'v1 cell 2800 veh/h x 0.3 superseded by the 3200 x 0.6 clean4 cell (MAIN 2026-11-06)'}
PANEL_AUTHORITY = 'newmap_spec.md (MAIN 2026-11-06): HELD-OUT eval 989600-989699'
EXPOSURE_EXTRA = dict(checked=(), disclosed='')
SETTING = 'p60'
MAPPO_LANE_OOD = None

#: amendment 22 (owner 2026-11-08 17:22 "test 20% cav instances"; MAIN A 2026-11-08): the p20 setting. A launcher
#: profile: the same code, estimator and co-criteria; only cells, panels, the RULE strategy, the TRAIN-world check
#: cell, the default roots and the allocation differ. p60 (the default) is unchanged and stays sealed.
SETTINGS = dict(p20=dict(
    CELL=dict(demand=3200, penetration=0.2), CLAIM3C_CELL=dict(demand=2400, penetration=0.2), CLAIM3C_KEY='D2400_p20',
    PARITY_CELL=dict(demand=3200, penetration=0.2),
    CELL_AUTHORITY=('owner 2026-11-08 17:22 (REGISTRATION): switch to the 20% CAV setting; MAIN: claim-3a-equivalent '
                    'cell D3200_p20, claim-3c light cell D2400_p20 (N6 amendment 22)'),
    PANEL_AUTHORITY=('N6 amendment 22 (MAIN 2026-11-08, option A): held-out integers 989600-989695 at D3200_p20 and '
                     '989600-989623 at D2400_p20 (3c precedent)'),
    EXPOSURE_EXTRA=dict(
        checked=('N6 p60 panels: 989600-989695 x D3200_p60 and 989600-989623 x D2400_p60 (sealed, unopened); no p20 '
                 'held-out world was ever run (N1 guard: TRAIN roles refuse held-out seeds; N6 is the only held-out user)',),
        disclosed=('; p20 (amendment 22): requests and HDV parameters are seeded by map:seed, so each p20 world shares '
                   'its arrival stream and HDV parameter draws with its sealed, unopened p60 twin; the p20 panels are '
                   'NOT independent replications of the p60 panels; nothing leaks (no p60 outcome was opened)')),
    PANEL_FILE=HERE/'nm_eval_panel_wz3to2_p20.json', PANEL_3C_FILE=HERE/'nm_eval_panel_wz3to2_claim3c_D2400_p20.json',
    PANEL_SHA256=None, PANEL_3C_SHA256=None,
    RULE_STRATEGY='wall_g_duo12', RULE_MODULE='tools.coopmwm.regimeu0_multicrit', RULE_SOURCE='c4/multicrit 4ff09522',
    RULE_SHA256=RULE_FAIR_SHA256,
    RULE_LABEL=('RULE: U0 + the S-MULTICRIT wall_g_duo12 rule (one lane-1 + one lane-2 CAV hold 22 m/s for 120 s, '
                'dmax 1.0, cooldown = hold; the S-MULTICRIT-LOWCAV p20 rule; module blob 4ff09522 = the LOWCAV '
                'harness d89a3c2e), always on'),
    MAPPO_LANE_OOD=('MAPPO_LANE (S-MAPPOWZL) was trained at D3200_p60 only: at p20 it is OUT OF DISTRIBUTION '
                    '(disclosed; no retraining, no selection)'),
    ROOT=None, FINAL_ROOT=None, ALLOCATION_SIM_S=1_750_000,
    ALLOCATION_AUTHORITY=('N6 amendment 22: p20 evaluation root allocation 1,750,000 (own ledger): pre-run 720,000 + '
                          'checks 3,480 + 2 primary runs x B/R x 120 x 1200 = 576,000 (+ 288,000 if A is simulated) '
                          '= 1,587,480 worst case, ~10% margin')))


def apply_setting(name):
    """Rebind this module's profile globals (call before anything else; p60 = the registered default)."""
    g = globals()
    if name in (None, 'p60'):
        if g['SETTING'] != 'p60':
            raise Blocked('a process cannot switch back to p60 after selecting another setting')
        return 'p60'
    if name not in SETTINGS:
        raise Blocked(f'unknown setting {name!r}')
    if g['SETTING'] not in ('p60', name):
        raise Blocked('a process selects one setting')
    g.update(SETTINGS[name], SETTING=name)
    if name in SETTINGS_PINS:
        pin = SETTINGS_PINS.get(name, {})
        g.update(PANEL_SHA256=pin.get('PANEL_SHA256'), PANEL_3C_SHA256=pin.get('PANEL_3C_SHA256'))
    return name


#: amendment 23 (owner 22:49/22:51 "focus on heavy and medium ... push the cav percentage higher"; MAIN): a SECOND
#: p20-era evaluation root 'p20x' with two more PRIMARY panels on fresh held-out integers: D2800_p20 (48 worlds,
#: 993000-993047) in the claim-3a slot and D3200_p40 (48 worlds, 993048-993095) in the claim-3c slot (C3_ wave
#: prefix; in this root that slot is a PRIMARY panel, not the 3c light-traffic claim). Same arms, RULE duo12 /
#: RULE_FAIR solo2 (disclosed: rules from the p20 LOWCAV/MULTICRIT work, also applied at p40), estimator and
#: co-criteria. Its own ledger debits the SAME final-eval allocation: 1,750,000 - 723,480 (p20 root) = 1,026,520.
SETTINGS['p20x'] = dict(SETTINGS['p20'],
    CELL=dict(demand=2800, penetration=0.2), CLAIM3C_CELL=None, CLAIM3C_KEY=None,
    N_WORLDS=96, CLAIM3C_WORLDS=0, EVAL_RANGE=(993000, 993099), EVAL_EXTRA_RANGE=(993000, 993099),
    CLAIM3C_SEED_START=None, PARITY_CELL=dict(demand=2800, penetration=0.2),
    SEEDS_RESERVED=(993096, 993099), PANEL_SIZE_REASON=('MAIN amendment 23b (owner: PRIMARY = D2800_p20 only): 96 fresh '
                                                        'worlds; 993096-993099 reserved unused'),
    CELL_AUTHORITY=('owner 2026-11-08 (REGISTRATION): PRIMARY eval = D2800_p20 only; MAIN: N6 amendment 23b, '
                    'a new sealed 96-world D2800_p20 panel; D3200_p40 cancelled'),
    PANEL_AUTHORITY=('N6 amendment 23b (MAIN 2026-11-08): fresh audited held-out integers 993000-993095 (D2800_p20); '
                     'eval-only held-out range 993000-993099'),
    EXPOSURE_EXTRA=dict(
        checked=('amendment-23 audit: 993000-993099 absent from REGISTRATION.md, every c4/* branch (py/json/md/toml/sh), '
                 'protocols/ and runs/active/C4-V7FT-20261017 (only 993592 = a merge-map GENRAND seed, outside the '
                 'range), traffic_shepherd.c4.seeds sets, N1 TRAIN/EXTRA_TRAIN/HELDOUT ranges, WMVERIFY 989950-989999, '
                 'MULTICRIT/LOWCAV 991xxx; remote PLAN.json/job dirs audited by IMITREPLAY before the pre-run',),
        disclosed=('; amendment 23b: fresh integers, so no arrival stream is shared with any other panel; the range is '
                   'registered for evaluation only at runtime by the N6 runner (N1 guard otherwise unchanged; TRAIN '
                   'roles still refuse it)')),
    PANEL_FILE=HERE/'nm_eval_panel_wz3to2_p20x_D2800.json', PANEL_3C_FILE=None,
    ALLOCATION_SIM_S=1_974_720,
    ALLOCATION_AUTHORITY=('MAIN 2026-11-08 23:27 (owner delegation): final-eval allocation raised 1,750,000 -> 2,700,000 '
                          '(incl. ORACLE + ORACLE_WM 2 x 115,200, amendment 25b); the p20x roots debit it after the p20 '
                          'root (723,480) and the void p20x r1 checks (1,800): 1,974,720 (own ledgers)'))

#: amendment 28 (owner-approved; MAIN 2026-11-09 08:46, pre-outcome): a NEW sealed D2800_p20 CONFIRMATION panel of 192
#: fresh worlds 993100-993291 (993292-993299 reserved unused), evaluation-only, in its own roots ('p20c'). Arms: U0,
#: ORACLE_WM, LEARNED_TRIGGER (the amendment-27 frozen bank fd385a2b, model 8e0ddd3e, digests 7409a7b4 / 41fc8d1a) on
#: the p20x pre-run plant (bank root), plus the trained B arms of CLEAN5C5D and F8V4B at u12, each on its own training
#: plant (trained roots, c4/strongguide 905a5a02 source). PRIMARY: ORACLE_WM - U0; a SEPARATE pre-registered test,
#: never pooled with the amendment-27 (p20x) panel.
CONFIRM_SEEDS = (993100, 993291)
SETTINGS['p20c'] = dict(SETTINGS['p20x'],
    N_WORLDS=192, CLAIM3C_WORLDS=0, EVAL_RANGE=(993100, 993299), EVAL_EXTRA_RANGE=(993100, 993299),
    SEEDS_RESERVED=(993292, 993299), PANEL_SIZE_REASON=('MAIN amendment 28 (owner-approved 2026-11-09 08:46): a NEW '
                                                        'sealed 192-world D2800_p20 confirmation panel; 993292-993299 '
                                                        'reserved unused'),
    CELL_AUTHORITY=('owner 2026-11-09 08:46 (REGISTRATION): MAIN amendment 28, D2800_p20 confirmation panel'),
    PANEL_AUTHORITY=('N6 amendment 28 (MAIN 2026-11-09 08:46): fresh audited held-out integers 993100-993291 '
                     '(D2800_p20); eval-only held-out range 993100-993299; disjoint from 9896xx, the 993000-993095 '
                     'primary panel and every train/verify seed'),
    EXPOSURE_EXTRA=dict(
        checked=('amendment-28 audit: 993100-993299 as standalone integers absent from REGISTRATION.md, every local '
                 'branch/remote ref (952 refs; only ops/test_nm_finaleval.py asserts seed_scope(993100) is None), '
                 'protocols/, docs/; remote audit by IMITREPLAY on cpuZ/cpuW/gpuA and the cpuV migration archive: 0 '
                 'hits (runs/active/C4-V7FT-20261017/p20c_seed_audit/SEED_AUDIT_993100_993299.json f533438a)',),
        disclosed=('; amendment 28: fresh integers, so no arrival stream is shared with any other panel; registered '
                   'for evaluation only at runtime by the N6 runner (TRAIN roles still refuse it)')),
    PANEL_FILE=HERE/'nm_eval_panel_wz3to2_p20c_D2800.json', PANEL_3C_FILE=None,
    PRERUN_ARMS=('U0',), BANK_CAP_SIM_S=2*192*HORIZON_S,
    ALLOCATION_SIM_S=693_000,
    ALLOCATION_AUTHORITY=('MAIN 2026-11-09 08:46 amendment 28: final-eval allocation raised 2,700,000 -> 4,000,000; '
                          'p20c = 1,158,600 over three own-ledger roots: bank root 693,000 (U0 230,400 + ORACLE_WM and '
                          'LEARNED_TRIGGER 460,800 + logcheck 720 + seqcheck 1,080) and two trained roots 232,800 each '
                          '(B 230,400 + U0 plant-identity 2,400); used before: 1,651,080 (p20 723,480, p20x r1 1,800, '
                          'p20x r2 923,400, amendment-26 smoke 2,400)'))
CONFIRM_BANK_ARMS = ('ORACLE_WM', 'LEARNED_TRIGGER')
BANK_SETTINGS = ('p20x', 'p20c', 'a33', 'a33t')
#: trained arms (amendment 28): label -> registered description; evaluated as arm B at TRAINED_UPDATE on the training
#: root's own CONFIG/source (the exact training execution path), charged to the trained final root's own ledger
TRAINED_ARMS = dict(C5D='CLEAN5C5D-B u12 (cpuZ coopmwm20261109_clean5c5d: WM-only bank fd385a2b + DEMOX WM demo '
                        '6c0d040a; c4/strongguide 905a5a02, CONFIG de74ceb3)',
                    F8B='F8V4B-B u12 (cpuW coopmwm20261109_f8v4b: rule-only bank 15edd209 + rule demo 74fe9103; '
                        'c4/strongguide 905a5a02, CONFIG 0f4101f5)',
                    C5DR=('CLEAN5C5D-R (the unguided arm of the same run: same verified-only action set, no targets or '
                          'demos; MAIN 2026-11-09 09:26 addition to amendments 28/29)'))
#: the evaluated arm letter of each trained label (B = world-model guided, R = unguided)
TRAINED_ARM_LETTER = dict(C5D='B', F8B='B', C5DR='R')
#: labels that run their own U0 plant-identity episodes at u12; C5DR shares C5D's training CONFIG (inherited)
IDENTITY_LABELS = ('C5D', 'F8B')
TRAINED_UPDATE = 12
TRAINED_UPDATES = (12,)+tuple(range(13, 25))   # 28 = u12; 29 = u24, or the last sealed update k >= 13 under the
#: MAIN 2026-11-09 12:23 pre-registered disk-safety stop rule (label 'u<k>', disclosed)
TRAINED_PLAN = 'TRAINED_PLAN.json'
TRAINED_PHASE = 'v2v_final_confirm_trained'
TRAINED_CAP_SIM_S = 192*HORIZON_S
IDENTITY_PHASE = 'v2v_final_confirm_identity'
IDENTITY_WORLDS = 2                       # U0 on the trained plant, first 2 panel worlds: J + safety identical to bank-root U0
IDENTITY_CAP_SIM_S = IDENTITY_WORLDS*HORIZON_S
CONFIRM_TOTAL_SIM_S = 693_000+2*(TRAINED_CAP_SIM_S+IDENTITY_CAP_SIM_S)


#: sealed panel digests of the p20 panels (written at registration, amendment 22)
SETTINGS_PINS = dict(p20=dict(PANEL_SHA256='fe64aa3566fe156c94a03b5809f83df5f6be981e5e0fd079225aaab1abdb6ebf',
                               PANEL_3C_SHA256='711d22c9ac063ba09f466f7217a45a47991b14f3f94070d751a67133cb514c29'),
                     p20x=dict(PANEL_SHA256='a0603e5dacf26fb076adc2a3a1831fd96522b1f1a85844a83e3d500bdbcde0aa',
                               PANEL_3C_SHA256=None),
                     p20c=dict(PANEL_SHA256='d2ed7783ae9ea071f89aec4397d3ee155161878b44e1717b02188b16e7ff7fa2',
                               PANEL_3C_SHA256=None),
                     a33=dict(PANEL_SHA256='ab2933df19e84c2bb4c1f8ef2f97f3fdb2807763739b64a6943520941cfa47b5',
                              PANEL_3C_SHA256=None),
                     a33t=dict(PANEL_SHA256='7065a293321e0b9800d73cc18da5574a718bf2ef83937327694307d3a85a394a',
                               PANEL_3C_SHA256=None))
#: amendment 33 (MAIN 2026-11-10, owner "register option 1 and start it"; pre-outcome): a FRESH sealed D2800_p20
#: panel of 192 worlds 999500-999691 (999692-999699 reserved unused), evaluation-only ('a33'); arms U0, DUO_HOLD,
#: LEARNED_TRIGGER, COMBINED on the p20x/p20c pre-run plant (mirror of the bank-root bindings on gpuB). 'a33t' = the
#: fresh TRAIN-type classifier-data block 991600-991639 (40 worlds; U0 / DUO_HOLD / SOLO_HOLD verification tapes of the
#: COMBINED classifier; never evaluation, never pooled), executed through the same runner (scope label only).
A33_SEEDS = (999500, 999691)
SETTINGS['a33'] = dict(SETTINGS['p20c'],
    N_WORLDS=192, EVAL_RANGE=(999500, 999699), EVAL_EXTRA_RANGE=(999500, 999699), SEEDS_RESERVED=(999692, 999699),
    PANEL_SIZE_REASON=('MAIN amendment 33 (owner 2026-11-10): a FRESH sealed 192-world D2800_p20 panel; 999692-999699 '
                       'reserved unused'),
    CELL_AUTHORITY='owner 2026-11-10 (REGISTRATION): MAIN amendment 33, FRESH D2800_p20 panel',
    PANEL_AUTHORITY=('A33 amendment 33 (MAIN 2026-11-10): fresh audited held-out integers 999500-999691 (D2800_p20); '
                     'eval-only held-out range 999500-999699; disjoint from 9896xx, 993000-993299 and every '
                     'train/verify seed'),
    EXPOSURE_EXTRA=dict(
        checked=('amendment-33 audit: 999500-999699 as standalone integers absent from REGISTRATION.md seed use, '
                 'protocols/experiment_history.json, every local branch ref, the a28/a32 panels and run dirs '
                 '(E:/agent-work/a33/SEED_AUDIT.json)',),
        disclosed=('; amendment 33: fresh integers, so no arrival stream is shared with any other panel; registered '
                   'for evaluation only at runtime by the runner (TRAIN roles still refuse it)')),
    PANEL_FILE=HERE/'nm_eval_panel_wz3to2_a33_D2800.json', PANEL_3C_FILE=None)
SETTINGS['a33t'] = dict(SETTINGS['a33'],
    N_WORLDS=40, EVAL_RANGE=(991600, 991699), EVAL_EXTRA_RANGE=(991600, 991699), SEEDS_RESERVED=(991640, 991699),
    PANEL_SIZE_REASON=('MAIN amendment 33: COMBINED classifier data, 40 fresh TRAIN-type worlds x (U0, DUO_HOLD, '
                       'SOLO_HOLD) = 144,000 sim-s <= 150,000; 991640-991699 reserved unused'),
    CELL_AUTHORITY='owner 2026-11-10 (REGISTRATION): MAIN amendment 33, classifier training tapes (TRAIN-type)',
    PANEL_AUTHORITY=('A33 amendment 33: fresh audited TRAIN-type integers 991600-991639 (D2800_p20) for the COMBINED '
                     'classifier data ONLY (never evaluation); executed through the eval runner (scope label only)'),
    PANEL_FILE=HERE/'nm_eval_trainblock_wz3to2_a33t_D2800.json')
#: amendment 35 (MAIN 2026-11-10 ~04:30, owner-relayed reviewer request; pre-outcome): FROZEN-controller transfer.
#: U0, LEARNED_TRIGGER (bank fd385a2b, model 8e0ddd3e, threshold 0.378785) and DUO_HOLD (duo12_v20), unchanged, on four
#: FRESH 96-world panels, one per cell (exploratory; never pooled). The registered generator (N1 map_scenarios) builds
#: shares {0.2, 0.4, 0.6} only, so the preferred D2800_p10 / D2800_p30 are not constructible: substitutes (recorded
#: in REGISTRATION.md before any episode) D2800_p10 -> D2800_p60 and D2800_p30 -> D2800_p40. The bank rows are looked
#: up under their source cell D2800_p20 (runner bank_cell_key); same plant / bindings as the p20c bank root mirror.
A35_CELLS = (('a35_D2400_p20', 2400, 0.2, 998000, 'D2400_p20 (preferred)'),
             ('a35_D3200_p20', 3200, 0.2, 998100, 'D3200_p20 (preferred)'),
             ('a35_D2800_p60', 2800, 0.6, 998200, 'D2800_p60 (substitute for D2800_p10: no share < 0.2 is constructible)'),
             ('a35_D2800_p40', 2800, 0.4, 998300, 'D2800_p40 (substitute for D2800_p30: nearest constructible share '
                                                  'other than the D2800_p20 source cell)'))
for _name, _d, _p, _lo, _why in A35_CELLS:
    SETTINGS[_name] = dict(SETTINGS['p20c'],
        CELL=dict(demand=_d, penetration=_p), PARITY_CELL=dict(demand=_d, penetration=_p),
        N_WORLDS=96, CLAIM3C_WORLDS=0, EVAL_RANGE=(_lo, _lo+99), EVAL_EXTRA_RANGE=(_lo, _lo+99),
        SEEDS_RESERVED=(_lo+96, _lo+99),
        PANEL_SIZE_REASON=(f'MAIN amendment 35: a FRESH 96-world transfer panel {_lo}-{_lo+95} ({_why}); '
                           f'{_lo+96}-{_lo+99} reserved unused'),
        CELL_AUTHORITY=f'MAIN amendment 35 (REGISTRATION 2026-11-10): frozen-controller transfer cell {_why}',
        PANEL_AUTHORITY=(f'A35 amendment 35: fresh audited held-out integers {_lo}-{_lo+95}; eval-only range '
                         f'{_lo}-{_lo+99}; disjoint from 9896xx, 993000-993299, 999500-999699 and every train/verify seed'),
        EXPOSURE_EXTRA=dict(
            checked=('amendment-35 audit: 998000-998399 as standalone integers absent from REGISTRATION.md seed use, '
                     'protocols/experiment_history.json, every c4/* branch tip, the a28/a32/a33 panels and run dirs '
                     '(E:/agent-work/a35/SEED_AUDIT.json)',),
            disclosed=('; amendment 35: fresh integers, so no arrival stream is shared with any other panel; registered '
                       'for evaluation only at runtime by the runner (TRAIN roles still refuse it)')),
        PANEL_FILE=HERE/f'nm_eval_panel_wz3to2_{_name}.json', PANEL_3C_FILE=None)
BANK_SETTINGS = BANK_SETTINGS + tuple(c[0] for c in A35_CELLS)
#: amendment 35 panels (registered 2026-11-10 after the seed audit, before any amendment-35 episode)
SETTINGS_PINS['a35_D2400_p20'] = dict(PANEL_SHA256='456c5f3f9721b679b16132ffeea24b51b553530453c4f00e5bd670f5f72eb5ed', PANEL_3C_SHA256=None)
SETTINGS_PINS['a35_D3200_p20'] = dict(PANEL_SHA256='152dbfe93017f4b5c97f8925c832b332578fd66c53037208f4ad157733a6b280', PANEL_3C_SHA256=None)
SETTINGS_PINS['a35_D2800_p60'] = dict(PANEL_SHA256='425e8a8d0a15dd93bb915e7d49f58d427e807f18e3f523e3fd77df24d3dd2995', PANEL_3C_SHA256=None)
SETTINGS_PINS['a35_D2800_p40'] = dict(PANEL_SHA256='b9ad776dd866c6a4dd38f45520331ded5958a9202dd8d2ae8fdb9f416d358272', PANEL_3C_SHA256=None)
#: amendment 37 (MAIN 2026-11-10, condition tailoring; pre-outcome): the UNC-rule SUMO verification of the cell-searched
#: plans on FRESH audited TRAIN-type blocks (24 worlds each; never evaluation, never pooled): D3200_p20 991700-991723
#: (991724-991799 reserved unused) and D2400_p20 991800-991823 (991824-991899 reserved unused); executed through the
#: same eval runner (scope label only), same plant / bindings as the a35 transfer cells. The evaluation itself reuses
#: the amendment-35 panels (settings a35_D3200_p20 / a35_D2400_p20, pins unchanged).
A37_VERIFY_CELLS = (('a37v_D3200_p20', 'a35_D3200_p20', 991700), ('a37v_D2400_p20', 'a35_D2400_p20', 991800))
for _name, _src, _lo in A37_VERIFY_CELLS:
    SETTINGS[_name] = dict(SETTINGS[_src],
        N_WORLDS=24, EVAL_RANGE=(_lo, _lo+99), EVAL_EXTRA_RANGE=(_lo, _lo+99), SEEDS_RESERVED=(_lo+24, _lo+99),
        PANEL_SIZE_REASON=(f'MAIN amendment 37: UNC rule = 24 fresh TRAIN-type seeds {_lo}-{_lo+23} per cell; '
                           f'{_lo+24}-{_lo+99} reserved unused'),
        CELL_AUTHORITY=f'MAIN amendment 37 (REGISTRATION 2026-11-10): verification block of the cell plans ({_src[4:]})',
        PANEL_AUTHORITY=(f'A37 amendment 37: fresh audited TRAIN-type integers {_lo}-{_lo+23} ({_src[4:]}) for the UNC '
                         'verification of the cell-searched plans ONLY (never evaluation); executed through the eval '
                         'runner (scope label only)'),
        EXPOSURE_EXTRA=dict(
            checked=('amendment-37 audit: 991700-991899 as standalone integers absent from REGISTRATION.md seed use, '
                     'protocols/experiment_history.json, every c4/* branch tip, the wmverify / a32-a35 run dirs '
                     '(E:/agent-work/a37/SEED_AUDIT.json)',),
            disclosed='; amendment 37: fresh TRAIN-type integers (verification only), registered at runtime by the runner'),
        PANEL_FILE=HERE/f'nm_eval_trainblock_wz3to2_{_name}.json', PANEL_3C_FILE=None)
BANK_SETTINGS = BANK_SETTINGS + tuple(c[0] for c in A37_VERIFY_CELLS)
#: amendment 37 verification blocks (registered 2026-11-10 after the seed audit, before any amendment-37 episode)
SETTINGS_PINS['a37v_D3200_p20'] = dict(PANEL_SHA256='e4791cfcf1d9baee265c6a02cb78ac9092901f2fa24e513d7ed24963111a1d42', PANEL_3C_SHA256=None)
SETTINGS_PINS['a37v_D2400_p20'] = dict(PANEL_SHA256='b8fb13143ede9f2437bef5a0a731832658d11728c5f81b06c100be9c94df08e1', PANEL_3C_SHA256=None)
MAP_SPEC_REL = 'assets/research/newmap_candidates/wz3to2/MAP_SPEC.json'
#: MAIN 2026-11-06 (N11 F4): the panel binds the staged probe network bytes (sha256 of wz3to2.net.xml)
NET_SHA256 = 'b62778986347e8c21bca90d7ddcc8d9faeb288290a4dbdc3f53ee64ae8c21dfa'
ROOT = '/home/ubuntu/coopmwm_clean5a_wz3to2'            # cpuZ; pass --root explicitly
FINAL_ROOT = '/home/ubuntu/coopmwm_clean5a_wz3to2_finaleval'
MAX_WORKERS = 46                                          # cpuZ 48 cores (MAIN: run right after u24)
PREFIXES = ('nmfinal_', 'nmdec_', 'nmparity_', 'nm3c_', 'nmlogcheck_', 'nmenvpar_')
RUNNER = 'tools.coopmwm.nm_eval_runner'
RUNNER_CONTRACT = dict(
    module=RUNNER, MAP=MAP,
    episode=('episode(spec, cfg, work) -> record. spec keys: job_id, arm (A|B|R|U0|MAPPO_SPEED|MAPPO_LANE|NATIVE), '
             'world_id, '
             'world (panel world dict, map wz3to2), checkpoint ({path,file_sha256,update,...} or '
             "{kind:'native'}), sim_s 1200, window {0,1200}, noise_seed = world seed, deterministic True, "
             'panel_sha256, capture (B/R on the decision subset only, else None: at the first new 2-3-member '
             'cooperation plan with min_t_s <= start <= max_t_s, a clean3 sumo_verify decision snapshot). '
             'record = spec identity keys + status graded, J_veh_s (plant team_delay_vs, identical reducer for '
             'every arm), safety {collisions,teleports,hard_brakes,ttc_events} ints; B/R add decision_log (every '
             'new 2+-member cooperative move: t_s, move, members) and, when capture is set, decision = None or '
             '{t_s, move, members, plan, state_sha256, state_ref} (state None unless the runner can branch it). '
             'Refuse any world that is not a registered held-out wz3to2 world (the launcher builds specs only '
             'from the panel and re-checks every returned identity); HDVs never commanded.'),
    decision_branch=('decision_branch(spec, cfg, work) -> record. Load decision.state_ref (verify '
                     'state_sha256); clean3 CRN prefix replay from t=0 to t_s (fingerprints checked), then '
                     '[t_s, t_s+360): CAVs in new_members execute the decided plan with the arm u24 policy, '
                     'every other CAV the BASE controller (background). Charged t_s+360 sim-s (clean3 '
                     'sumo_verify.branch_sim_s). record = spec identity keys + status graded, J_veh_s over '
                     '[t_s, t_s+360), safety.'))
SEED_TOKEN = re.compile(r'(?<![\da-fA-F.])(9896\d\d)(?![\da-fA-F])')
SIX_DIGITS = re.compile(r'(?<![\da-fA-F.])(\d{6})(?![\da-fA-F])')


def _eval_tokens(text):
    """Six-digit tokens inside the setting's EVAL_RANGE (p60: exactly the SEED_TOKEN 9896xx hits; amendment 28: the
    p20c range, so declared 9896xx text in a p20 training CONFIG is not an exposure of the p20c panel)."""
    return [int(m) for m in SIX_DIGITS.findall(str(text)) if EVAL_RANGE[0] <= int(m) <= EVAL_RANGE[1]]
REPLICATED_FILE = 'track2_diagnostics/replicated_u024_live.json'


def replicated_file(update):
    return f'track2_diagnostics/replicated_u{int(update):03}_live.json'


def check_update(update):
    if int(update) not in EVAL_UPDATES:
        raise Blocked(f'evaluation update must be one of {EVAL_UPDATES} (MAIN 23:10)')
    return int(update)


def wave_name(arm, update, prefix=''):
    """A/B/R waves carry the evaluated update; U0/RULE/NATIVE/MAPPO waves are shared by every update."""
    return f'{prefix}{arm}_u{int(update):03}' if arm in ARMS else f'{prefix}{arm}'


def a_frozen(cfg):
    """clean5a frozen control arm (registered key): A is evaluated as the alias of U0, never re-simulated."""
    value = (cfg or {}).get('frozen_control_arm')
    if value is None:
        return False
    if value != FROZEN_A_RULE:
        raise Blocked('frozen_control_arm must carry the registered rule text')
    return True


def _world(seed, cell):
    """N1's registered world dict (tools.coopmwm.map_scenarios.world) for the panel cell."""
    from tools.coopmwm import map_scenarios
    if EVAL_EXTRA_RANGE is not None:                     # amendment 23: eval-only held-out range (runner patch)
        from tools.coopmwm.nm_eval_runner import install_eval_ranges
        install_eval_ranges(map_scenarios)
    return map_scenarios.world(int(seed), int(cell['demand_vph']['main']), float(cell['penetration']))


def in_decision_subset(world):
    return world['repeat'] % DECISION_STRIDE == 0


# ---------------------------------------------------------------- budget quote (zero physics)

def quote(base='mappo_u0', a_alias=True):
    """Ceiling sim-s against v2v_final (own ledger): the sealed pre-run (NATIVE/U0/RULE on 3a + 3c), B and R at
    each evaluation update (u12 primary, u24 secondary) on 3a + 3c, A only when it is not the frozen U0 alias,
    the two MAPPO comparators (shared by both updates) and F11 parity; decisions are not built (0)."""
    if base not in BASES:
        raise Blocked(f'unknown base {base!r}')
    episode, c3 = N_WORLDS*HORIZON_S, CLAIM3C_WORLDS*HORIZON_S
    need = {f'prerun_{a}': episode+c3 for a in PRERUN_ARMS}
    for u in EVAL_UPDATES:
        for a in ARMS:
            if a == 'A' and a_alias:
                continue
            need[f'u{u:03}_{a}'] = episode
            if a in CLAIM3C_ARMS:
                need[f'u{u:03}_C3_{a}'] = c3
    need.update({f'episodes_{a}': episode for a in COMPARATOR_ARMS if a not in PRERUN_ARMS})
    if SETTING == 'p20x':                # amendment 25b: ORACLE + ORACLE_WM on the p20x primary panel only
        need.update({f'oracle_{a}': episode for a in BANK_ARMS})
    need['parity_f11'] = PARITY_SIM_S
    need['decisions'] = 0
    total = sum(need.values())
    prerun = sum(v for k, v in need.items() if k.startswith('prerun_'))
    core = total-PARITY_SIM_S
    c3_total = len(PRERUN_ARMS)*c3+sum(v for k, v in need.items() if '_C3_' in k)
    return dict(base=base, phase=PHASE, allocation_sim_s=ALLOCATION_SIM_S, allocation_authority=ALLOCATION_AUTHORITY,
                updates=list(EVAL_UPDATES), primary_update=PRIMARY_UPDATE, a_alias_of_u0=bool(a_alias),
                need_ceiling=need, prerun_sim_s=prerun, prerun_arms=list(PRERUN_ARMS), claim3c_sim_s=c3_total,
                episodes_sim_s=total-PARITY_SIM_S-c3_total, decisions_ceiling_sim_s=0, parity_sim_s=PARITY_SIM_S,
                core_ceiling_sim_s=core, total_ceiling_sim_s=total, slack_sim_s=ALLOCATION_SIM_S-total,
                margin_over_core=round(ALLOCATION_SIM_S/core-1, 4), margin_over_total=round(ALLOCATION_SIM_S/total-1, 4),
                fits=total <= ALLOCATION_SIM_S,
                basis=dict(prerun=f'{len(PRERUN_ARMS)} arms (NATIVE, U0, RULE) x ({N_WORLDS} + {CLAIM3C_WORLDS}) worlds x '
                                  f'{HORIZON_S} s, shared by both updates',
                           per_update=(f'B, R{"" if a_alias else ", A"} x ({N_WORLDS} + {CLAIM3C_WORLDS}) worlds x '
                                       f'{HORIZON_S} s at each of u12 (primary) and u24 (secondary)'),
                           a=A_ALIAS_LABEL if a_alias else 'A simulated at each update',
                           comparators=f'{len(COMPARATOR_ARMS)} MAPPO comparators x {N_WORLDS} x {HORIZON_S} s (shared)',
                           parity_f11=(f'{len(PARITY_WORLDS)} TRAIN worlds x 2 paths (eval plant, training env) x '
                                       f'{len(COMPARATOR_ARMS)} comparators x {300+PARITY_ROLLOUT_S} s'),
                           decisions='F7 not built (MAIN decision): 0',
                           charging='planned sim-s precharged per job in the evaluation root ledger; whole wave reserved '
                                    'before dispatch'),
                wall_time=dict(workers=24, sim_s_per_worker_s='6-9 (newmap_probe cpuV)',
                               hours=[round(total/(24*9)/3600, 2), round(total/(24*6)/3600, 2)]))


# ---------------------------------------------------------------- panel registration (zero physics)

def build_panel(spec_path, registered_utc, *, demand=None, penetration=None):
    demand = CELL['demand'] if demand is None else demand
    penetration = CELL['penetration'] if penetration is None else penetration
    """The registered evaluation panel: N1 map descriptor + N1 world registry for the clean4 cell
    (no outcome of any run is read)."""
    from tools.coopmwm import map_descriptor as md, map_scenarios as ms
    spec_path = Path(spec_path)
    desc = md.load(spec_path)
    spec = json.loads(spec_path.read_text(encoding='utf-8'))
    net_sha = hashlib.sha256(Path(desc.network_path).read_bytes()).hexdigest()
    if net_sha != NET_SHA256:
        raise Blocked(f'network bytes {net_sha} are not the staged probe network {NET_SHA256}')
    if desc.name != MAP or set(spec['streams']) != {'main'}:
        raise Blocked('MAP_SPEC is not the single-stream wz3to2 map')
    registered = next((c for c in ms.cells() if c['demand'] == demand and c['penetration'] == penetration), None)
    if registered is None:
        raise Blocked(f'({demand}, {penetration}) is not a registered N1 wz3to2 cell')
    stream = spec['streams']['main']
    cell = dict(demand_vph={'main': int(demand)}, penetration=float(penetration), cell=registered['cell'],
                cell_key=registered['cell_key'], profile=stream['profile'], depart_lane=stream['depart_lane'],
                rate_profile_s=[[0, 300, 0.8], [300, 1500, 1.0], [1500, 1800, 0.6]],
                episode_s=[0, ms.EPISODE_S], authority=CELL_AUTHORITY, newmap2_report_sha256=NEWMAP2_REPORT_SHA256)
    seeds = list(range(EVAL_RANGE[0], EVAL_RANGE[0]+N_WORLDS))
    worlds = [dict(world_id=f'{MAP}_{s}', world=_world(s, cell)) for s in seeds]
    subset = [r['world_id'] for r in worlds if in_decision_subset(r['world'])]
    mde = lambda k: round(2.8016/math.sqrt(k), 3)   # noqa: E731  z(.975)+z(.80), paired two-sided
    contract = dict(
        update=UPDATE, arms=list(ARMS), comparators=dict(
            BASE='clean4 base controller (N4 choice): mappo_u0 = frozen speed-only MAPPO u0 '
                 '(A_checkpoint_000.pt == CONFIG common_checkpoint_sha256), or native (BASE aliases NATIVE)',
            NATIVE='native SUMO CAV driving (no commands), same plant/runtime'),
        comparisons=['B-A', 'R-A', 'B-R'], total_benefit=dict(vs=['BASE', 'NATIVE'], arms=['B', 'A', 'R']),
        metric='J_veh_s', higher_is_better=False,
        metric_definition=('team delay veh-s over the window: on-road speed deficit dt*max(0, 1-v/31.29) + '
                           'source waiting (requested, not yet inserted)'),
        horizon_s=HORIZON_S, window=dict(WINDOW), horizon_note='= clean4 training episode length (MAIN; N1 EPISODE_S)',
        noise_seed='world seed', deterministic=True, role='final_test (N1 HELDOUT_ROLES)',
        bootstrap=dict(BOOTSTRAP), estimator='tools.coopmwm.planner_eval._paired_ci (whole-world resampling, '
                                             'identical resample indices for every contrast)',
        safety_counters=list(SAFETY),
        primary=dict(contrast='B-BASE', rule=('mean(J_B - J_BASE) paired CI95 entirely below 0 AND B total '
                                              'collisions+teleports <= BASE total'),
                     note='descriptive generalisation check; the clean4 registered primary criterion (u24 '
                          'replicated synergy test on TRAIN states, N4) is unchanged; never gates training'),
        decisions=dict(arms=list(DECISION_ARMS), worlds=subset, subset_rule=f'panel repeat % {DECISION_STRIDE} == 0',
                       per_world=1,
                       rule=(f'first new cooperation plan with {MIN_MEMBERS}..{MAX_MEMBERS} members starting at '
                             f'{DECISION_MIN_T} <= t <= {DECISION_MAX_T} s in the evaluation episode; time order '
                             'only (outcome-blind); none -> world has no decision'),
                       horizon_s=DECISION_H, labels=['all', 'background', 'solo<i>'], background='BASE controller',
                       branching=('clean3 sumo_verify decision snapshot + CRN prefix replay from t=0; charged '
                                  't_s + 360 sim-s per branch (sumo_verify.branch_sim_s)'),
                       decomposition=('delta(all group new, background base) - max delta(one member new, '
                                      'background base) (clean3 ISOLATED_TEST)'),
                       metrics=['delta_all_new_veh_s', 'delta_only_member_new_veh_s', 'best_solo_veh_s',
                                'policy_cooperation_gain_veh_s'], higher_is_better=True),
        budget=dict(phase=PHASE, allocation_sim_s=PANEL_ALLOCATION_SIM_S, ledger='clean4 run root CHARGES.jsonl',
                    authority='MAIN decision 2026-11-06 for N6 (only clean4 final-evaluation runner)',
                    **REGISTERED_PANEL_QUOTE))
    power = dict(method='paired two-sided normal approximation, alpha .05, power .80; MDE in SD units of the '
                        'per-world paired difference = 2.8016/sqrt(N)',
                 mde_sd_units={str(k): mde(k) for k in (8, 16, 48, 64, 96, 100)},
                 noise_basis=('newmap_probe SUMMARY_stage3 (wz3to2 2800 x 0.3, 7 worlds 989000-989006, H=360 s): '
                              'paired SD solo-vs-base 3,022 veh-s (n=126), wall-vs-base T 3,633 (W2) - 5,259 (W1), '
                              'synergy S 3,008 (n=42); the 3200 x 0.6 cell is more congested, SDs likely larger'),
                 episode_assumption=('1200 s = 3.33 x 360 s windows; if independent, per-world SD ~ 1.83 x 3,022 '
                                     '~ 5,520 veh-s -> MDE(96) ~ 1,580 veh-s per world-episode'),
                 decisions=('<= 8 decisions per arm -> MDE ~ 0.99 SD (~3,000 veh-s): descriptive only, sized by '
                            'the 690,000 allocation with prefix-inclusive branch cost'),
                 chosen=N_WORLDS, reason=PANEL_SIZE_REASON or ('largest round panel inside the registered 100-seed '
                                                              'range; same 96-world shape as the clean3 registered '
                                                              'estimator; 4 seeds reserved unused'))
    panel = dict(
        schema=SCHEMA+'.panel', version=2 if SETTING == 'p60' else f'2_{SETTING}', map=MAP,
        role='evaluation only (registered by N6; never training, never selection)',
        registered_utc=registered_utc, authority=PANEL_AUTHORITY,
        supersedes=[dict(panel_sha256=k, reason=v) for k, v in sorted(SUPERSEDED.items())],
        seed_ranges=dict(evaluation=list(EVAL_RANGE), training=list(TRAIN_RANGE)),
        seeds_used=[seeds[0], seeds[-1]], seeds_reserved_unused=list(SEEDS_RESERVED or (seeds[-1]+1, EVAL_RANGE[1])),
        cell=cell, map_binding=dict(
            map_spec=MAP_SPEC_REL, net_file=Path(desc.network_path).name, net_sha256=net_sha,
            net_rule='sha256 of the staged probe network bytes (MAIN 2026-11-06, N11 F4)',
            map_spec_file_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            spec_sha256=desc.spec_sha256, network_semantic_sha256=desc.network_sha256,
            descriptor_sha256_at_registration=desc.digest,
            n1_registered={k: md.REGISTERED[MAP][k] for k in ('spec_sha256', 'network_sha256')},
            k_slots=desc.k_slots, bottleneck=desc.bottleneck, world_registry='tools.coopmwm.map_scenarios (N1)'),
        stream_rule=('map-namespaced streams (N1 map_scenarios == newmap probe/NEWMAP2): requests '
                     'newmap_requests_v1:{map}:{seed}:{stream}, CAV identity newmap_cav_v1:..., HDV parameters '
                     'sample_driver("{map}:{seed}", id)'),
        exposure=dict(
            checked=['newmap_probe ledger: seeds 989000-989006 only', 'NEWMAP2: TRAIN 989200-989202 only',
                     'tools/coopmwm, traffic_shepherd, ops code']+list(EXPOSURE_EXTRA['checked']),
            disclosed=('integers 989600-989635 are the OLD merge-map S-COOPMWM final_test worlds '
                       '(tools/coopmwm/protocol.py FINAL_START); wz3to2 worlds are map-namespaced, so no wz3to2 '
                       'traffic realisation shares draws with them; job/world ids carry the wz3to2_ prefix'
                       +EXPOSURE_EXTRA['disclosed'])),
        worlds=worlds, contract=contract, power=power)
    return seal(json.loads(json.dumps(panel)), 'panel_sha256')


def validate_panel(panel):
    verify(panel, 'panel_sha256', 'evaluation panel seal')
    if panel['panel_sha256'] in SUPERSEDED:
        raise Blocked(f'panel {panel["panel_sha256"]} is SUPERSEDED: {SUPERSEDED[panel["panel_sha256"]]}')
    worlds = panel.get('worlds') or []
    seeds = [r['world']['seed'] for r in worlds]
    if panel.get('map') != MAP or len(worlds) != N_WORLDS or len(set(seeds)) != N_WORLDS:
        raise Blocked('panel must hold 96 distinct wz3to2 worlds')
    for row in worlds:
        s = row['world']['seed']
        if not EVAL_RANGE[0] <= s <= EVAL_RANGE[1] or TRAIN_RANGE[0] <= s <= TRAIN_RANGE[1]:
            raise Blocked(f'panel seed {s} outside the registered evaluation range')
        if row != dict(world_id=f'{MAP}_{s}', world=_world(s, panel['cell'])):
            raise Blocked(f'panel world {s} does not follow the registered cell/identity rule')
    c = panel['contract']
    if panel['map_binding']['net_sha256'] != NET_SHA256:
        raise Blocked('panel does not bind the staged probe network')
    if (c['bootstrap'] != BOOTSTRAP or c['horizon_s'] != HORIZON_S or c['budget']['allocation_sim_s'] != PANEL_ALLOCATION_SIM_S
            or c['decisions']['worlds'] != [r['world_id'] for r in worlds if in_decision_subset(r['world'])]):
        raise Blocked('panel contract differs from the launcher constants')
    return panel


def load_panel(path=None, *, pinned=''):
    path, pinned = path or PANEL_FILE, PANEL_SHA256 if pinned == '' else pinned
    panel = validate_panel(read(path))
    if pinned is not None and panel['panel_sha256'] != pinned:
        raise Blocked(f'panel_sha256 {panel["panel_sha256"]} != pinned {pinned}')
    return panel


def build_panel_3c(spec_path, registered_utc):
    """Claim-3c panel: 24 held-out worlds of the D2400_p60 cell (N1 world()), same estimator."""
    v2 = load_panel()
    from tools.coopmwm import map_scenarios as ms
    cell_row = next(c for c in ms.cells() if c['demand'] == CLAIM3C_CELL['demand']
                    and c['penetration'] == CLAIM3C_CELL['penetration'])
    cell = dict(v2['cell'], demand_vph={'main': CLAIM3C_CELL['demand']}, penetration=CLAIM3C_CELL['penetration'],
                cell=cell_row['cell'], cell_key=cell_row['cell_key'],
                authority='MAIN registration (pre-results): claim 3c, condition-aware: no harm in light traffic'
                          +(f' ({SETTING} setting)' if SETTING != 'p60' else ''))
    seeds = list(range(CLAIM3C_SEED_START or EVAL_RANGE[0], (CLAIM3C_SEED_START or EVAL_RANGE[0])+CLAIM3C_WORLDS))
    worlds = [dict(world_id=f'{MAP}_{cell_row["cell_key"]}_{s}', world=_world(s, cell)) for s in seeds]
    contract = dict({k: v2['contract'][k] for k in ('update', 'metric', 'higher_is_better', 'metric_definition',
                                                  'horizon_s', 'window', 'horizon_note', 'noise_seed', 'deterministic',
                                                  'role', 'bootstrap', 'estimator', 'safety_counters')},
                    claim='3c: condition-aware - no harm in light traffic', arms=list(CLAIM3C_ARMS),
                    comparisons=['B-R'], total_benefit=dict(vs=['BASE', 'NATIVE'], arms=['B', 'R']),
                    primary=dict(contrast='B-BASE', rule='descriptive: no-harm check of B vs BASE (= U0) and NATIVE in '
                                                         'the light-traffic cell; CI and per-seed harm fraction reported',
                                 note='secondary claim; opens with 3a; separate seal'),
                    decisions=dict(arms=[], worlds=[], note='no decision capture on the 3c panel'))
    panel = dict(schema=SCHEMA+'.panel', version='claim3c' if SETTING == 'p60' else f'claim3c_{SETTING}', map=MAP,
                 role='evaluation only (claim 3c)',
                 registered_utc=registered_utc, authority='MAIN registration 2026-11-06 (claim 3c)',
                 seed_ranges=v2['seed_ranges'], seeds_used=[seeds[0], seeds[-1]], cell=cell,
                 map_binding=v2['map_binding'], stream_rule=v2['stream_rule'], exposure=v2['exposure'],
                 sibling_panel_sha256=v2['panel_sha256'], worlds=worlds, contract=contract, job_prefix='nm3c',
                 wave_prefix='C3_')
    return seal(json.loads(json.dumps(panel)), 'panel_sha256')


def load_panel_3c(path=None, *, pinned=''):
    if path is None and PANEL_3C_FILE is None:            # amendment 23b: a single-panel root (p20x)
        return None
    path, pinned = path or PANEL_3C_FILE, PANEL_3C_SHA256 if pinned == '' else pinned
    panel = verify(read(path), 'panel_sha256', 'claim-3c panel seal')
    if pinned is not None and panel['panel_sha256'] != pinned:
        raise Blocked(f'claim-3c panel_sha256 {panel["panel_sha256"]} != pinned {pinned}')
    seeds = [r['world']['seed'] for r in panel['worlds']]
    start = CLAIM3C_SEED_START or EVAL_RANGE[0]
    if seeds != list(range(start, start+CLAIM3C_WORLDS)) or panel['map_binding']['net_sha256'] != NET_SHA256:
        raise Blocked('claim-3c panel differs from its registration')
    for r in panel['worlds']:
        if r['world'] != _world(r['world']['seed'], panel['cell']) or r['world']['cell_key'] != CLAIM3C_KEY:
            raise Blocked('claim-3c world does not follow the registered cell')
    if panel['contract']['bootstrap'] != BOOTSTRAP or panel['contract']['horizon_s'] != HORIZON_S:
        raise Blocked('claim-3c contract differs from the launcher constants')
    return panel


# ---------------------------------------------------------------- N1 map API / clean4 binding

def default_spec_path():
    """COOPMWM_MAP_SPEC (N1 process selection) or the MAP_SPEC.json in this source tree."""
    from tools.coopmwm import map_descriptor as md
    return Path(os.environ.get(md.ENV) or Path(md.__file__).resolve().parents[2]/MAP_SPEC_REL)


class N1MapAPI:
    """The launcher's map interface over N1's adapter ``tools.coopmwm.map_api`` (N1_READY b2f30091):
    seed_ranges() {train, heldout}, descriptor(spec_path) -> MapDescriptor (raw net sha
    ``network_file_sha256``), world(seed, cell_key), check_world(world, role)."""
    STUB = False
    __name__ = 'tools.coopmwm.map_api'

    def __init__(self, spec_path=None):
        self.spec_path = spec_path

    def seed_ranges(self, name):
        from tools.coopmwm import map_api
        if name != map_api.MAP:
            return {}
        r = map_api.seed_ranges()
        return dict(train=tuple(r['train']), eval=tuple(r['heldout']))

    def descriptor(self, name):
        from tools.coopmwm import map_api, map_descriptor as md
        path = self.spec_path or (None if os.environ.get(md.ENV) else default_spec_path())
        desc = map_api.descriptor(path)
        if desc.name != name:
            raise ValueError(f'descriptor {desc.name!r} != {name!r}')
        return dict(name=desc.name, net_sha256=desc.network_file_sha256, spec_sha256=desc.spec_sha256,
                    network_sha256=desc.network_sha256, descriptor_sha256=desc.digest, stream_namespace=desc.name,
                    k_slots=desc.k_slots, spec_path=desc.spec_path)

    def world(self, name, seed, cell, repeat=None):
        from tools.coopmwm import map_api
        if name != map_api.MAP:
            raise ValueError(f'no world registry for {name!r}')
        return map_api.world(int(seed), cell['cell_key'])

    def check_world(self, world, role):
        from tools.coopmwm import map_api
        return map_api.check_world(world, 'final_test' if role == 'eval' else role)


def bind_map_api(module=None):
    """The N1 map API (default) or an injected module with the same four functions (tests)."""
    module = module or os.environ.get('NM_MAP_API')
    api = N1MapAPI() if module in (None, 'n1') else importlib.import_module(module)
    missing = [f for f in ('seed_ranges', 'descriptor', 'world', 'check_world') if not callable(getattr(api, f, None))]
    if missing:
        raise Blocked(f'map API {api.__name__} lacks {missing}')
    return api


def map_check(api, panel):
    """Compare the bound map API with the registered panel (zero physics)."""
    out = dict(module=api.__name__, stub=bool(getattr(api, 'STUB', False)), mismatches=[])
    ranges = api.seed_ranges(MAP)
    if tuple(ranges.get('eval', ())) != EVAL_RANGE or tuple(ranges.get('train', ())) != TRAIN_RANGE:
        out['mismatches'].append(f'seed ranges {ranges}')
    try:
        desc = api.descriptor(MAP)
        out.update({k: desc.get(k) for k in ('net_sha256', 'spec_sha256', 'network_sha256', 'descriptor_sha256',
                                             'spec_path')})
        if desc.get('net_sha256', desc.get('network_sha256')) != panel['map_binding']['net_sha256']:
            out['mismatches'].append('network sha differs from the panel map binding (staged probe network)')
        if desc.get('stream_namespace', MAP) != MAP:
            out['mismatches'].append('streams are not namespaced by the map')
    except (OSError, KeyError, TypeError, ValueError) as error:
        out['descriptor'] = f'unavailable: {error!r}'
    for row in panel['worlds']:
        w = row['world']
        try:
            if api.world(MAP, w['seed'], panel['cell'], w['repeat']) != w:
                out['mismatches'].append(f'world {w["seed"]} differs from the map API scenario world')
                break
            api.check_world(w, 'eval')
        except (TypeError, ValueError) as error:
            out['mismatches'].append(f'world {w["seed"]}: {error!r}')
            break
    out['ok'] = not out['mismatches'] and 'descriptor' not in out
    return out


def _cell_key(cell):
    """(demand, penetration) of a cell given as a dict ({demand_vph|demand, penetration}), an N1
    cell key ('D3200_p60') or an N1 cell id (8)."""
    if isinstance(cell, (str, int)) and not isinstance(cell, bool):
        from tools.coopmwm import map_scenarios as ms
        row = next(c for c in ms.cells() if cell in (c['cell_key'], c['cell']))
        return int(row['demand']), round(float(row['penetration']), 6)
    demand = cell.get('demand_vph') or ({'main': cell['demand']} if 'demand' in cell else {})
    if cell.get('profile', 'peak') != 'peak' or set(demand) != {'main'}:
        raise ValueError('only the single-stream peak profile is registered')
    return int(demand['main']), round(float(cell['penetration']), 6)


def clean4_binding(cfg, panel, declared_base=None):
    """N4 interface: CONFIG ``newmap`` = {map, base_controller, cell, eval_panel_sha256}. Present keys
    must match; absent keys are reported as unbound and covered by the sealed run plan."""
    nm = cfg.get('newmap') if isinstance(cfg.get('newmap'), dict) else {}
    bad, unbound = [], []
    if nm.get('map') is None:
        unbound.append('map')
    elif nm['map'] != MAP:
        bad.append(f'map {nm["map"]!r}')
    if nm.get('eval_panel_sha256') is None:
        unbound.append('eval_panel_sha256')
    elif nm['eval_panel_sha256'] in SUPERSEDED:
        bad.append(f'eval_panel_sha256 {nm["eval_panel_sha256"][:8]} is SUPERSEDED ({SUPERSEDED[nm["eval_panel_sha256"]]}); '
                   f'bind {panel["panel_sha256"]}')
    elif nm['eval_panel_sha256'] != panel['panel_sha256']:
        bad.append('eval_panel_sha256 differs from the registered panel')
    if nm.get('cell') is None:
        unbound.append('cell')
    else:
        try:
            if _cell_key(nm['cell']) != _cell_key(panel['cell']):
                bad.append(f'training cell {nm["cell"]} differs from the panel cell')
        except (KeyError, TypeError, ValueError, AttributeError, StopIteration):
            bad.append('unreadable training cell')
    base = nm.get('base_controller')
    if base is None:
        unbound.append('base_controller')
        base = declared_base
    elif declared_base is not None and declared_base != base:
        bad.append(f'declared base {declared_base} != CONFIG base {base}')
    if base not in BASES:
        bad.append(f'base controller {base!r} unknown (CONFIG newmap.base_controller or register --base)')
    if base == 'mappo_u0' and not cfg.get('common_checkpoint_sha256'):
        bad.append('mappo_u0 base needs CONFIG common_checkpoint_sha256')
    if bad:
        raise Blocked('clean4 CONFIG binding: '+'; '.join(bad))
    return dict(map=MAP, base=base, panel_sha256=panel['panel_sha256'], cell=deepcopy(panel['cell']),
                unbound=unbound, source='CONFIG.newmap' if not unbound else 'CONFIG.newmap + operator declaration')


def _seed_hits(value, skip=False):
    """Evaluation-range integers/tokens in a JSON value, ignoring declared ranges and the panel binding."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _seed_hits(v, skip or 'range' in str(k).lower() or str(k) == 'eval_panel_sha256')
    elif isinstance(value, list):
        for v in value:
            yield from _seed_hits(v, skip)
    elif skip or isinstance(value, bool):
        return
    elif isinstance(value, int) and EVAL_RANGE[0] <= value <= EVAL_RANGE[1]:
        yield value
    elif isinstance(value, str):
        yield from _eval_tokens(value)


def exposure_check(root, cfg):
    """No evaluation-range seed may appear as a world/job in the clean4 CONFIG, its explicit world
    registration or its ledger before this launcher (declared ``*range*`` values, the panel binding
    and this launcher's own job ids are allowed)."""
    found = dict(config=sorted(set(_seed_hits(cfg))),
                 ledger=sorted({int(m) for r in fe.ledger_rows(root) if not str(r.get('id', '')).startswith(PREFIXES)
                                for m in _eval_tokens(r.get('id', ''))}))
    registration = Path(root)/'ISOLATED_REGISTRATION.json'
    if registration.is_file():
        found['registration'] = sorted(set(_seed_hits(read(registration))))
    if any(found.values()):
        raise Blocked(f'evaluation seeds exposed in clean4: { {k: v[:5] for k, v in found.items() if v} }')
    return dict(status='PASS', scanned=['CONFIG.json (minus *range* keys and newmap.eval_panel_sha256)',
                                        'CHARGES.jsonl ids', 'ISOLATED_REGISTRATION.json if present'])


def allocation(cfg):
    """The clean4 sealed v2v_final allocation (first of BUDGET_KEYS present, else CONFIG budget)."""
    for key in BUDGET_KEYS:
        if isinstance(cfg.get(key), dict):
            phases = verify(deepcopy(cfg[key]), 'receipt_sha256', f'{key} seal')['phases']
            return key, phases.get(PHASE)
    return 'budget', ((cfg.get('budget') or {}).get('phases') or {}).get(PHASE)


def own_reserve(final_root, jobs, phase, *, cap=None, clean_root=None, calls=None, allocation=None):
    """The evaluation root's OWN ledger (<final_root>/CHARGES.jsonl): whole waves precharged before any
    dispatch; total over every phase <= ALLOCATION_SIM_S; ``cap`` bounds this phase. The clean run's
    root and ledger are never written (check_pins: a standalone experiment accounts in its own root)."""
    final_root = Path(final_root)
    lock = None
    try:
        import fcntl
        lock = (final_root/'LEDGER.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX)
    except ImportError:
        pass
    try:
        rows = fe.ledger_rows(final_root)
        ids = [j['id'] for j in jobs]
        known = {r['id'] for r in rows} | ({r['id'] for r in fe.ledger_rows(clean_root)} if clean_root else set())
        if len(set(ids)) != len(ids) or set(ids) & known:
            raise Blocked('duplicate charge/attempt')
        amount = sum(j['sim_s'] for j in jobs)
        allocation = ALLOCATION_SIM_S if allocation is None else allocation
        if sum(r['sim_s'] for r in rows)+amount > allocation:
            raise Blocked(f'{PHASE} allocation {allocation} would be exceeded (own ledger)')
        if cap is not None and sum(r['sim_s'] for r in rows if r.get('phase') == phase)+amount > cap:
            raise Blocked(f'{phase} cap {cap} would be exceeded')
        with (final_root/'CHARGES.jsonl').open('a', encoding='utf-8') as f:
            for j in jobs:
                f.write(json.dumps(dict(id=j['id'], phase=phase, sim_s=j['sim_s'], unix=time.time()), sort_keys=True)+'\n')
        if calls is not None:
            calls.append(('reserve', phase, amount))
    finally:
        if lock is not None:
            lock.close()


def budget_check(root, cfg, base, *, outstanding=None, final_root=None):
    """Zero physics: the registered allocation covers this launcher's ceiling (pre-run spend in the
    evaluation root ledger included); v2v_final carries no foreign spend."""
    source, allocated = allocation(cfg)
    q = quote(base, a_alias=a_frozen(cfg))
    rows = [r for r in fe.ledger_rows(root) if r.get('phase') == PHASE]
    prerun = sum(r['sim_s'] for r in fe.ledger_rows(final_root)) if final_root is not None and Path(final_root).is_dir() else 0
    mine = sum(r['sim_s'] for r in rows if str(r['id']).startswith(PREFIXES))+prerun
    foreign = [r['id'] for r in rows if not str(r['id']).startswith(PREFIXES)]
    cap = ALLOCATION_SIM_S        # the registered N6 cap; N6 charges only its own root ledger (amendment 7)
    need = max(0, q['total_ceiling_sim_s']-mine) if outstanding is None else outstanding
    out = dict(source=source, phase=PHASE, allocated=allocated, sub_cap=cap, spent_by_launcher=mine,
               foreign_rows=foreign[:5], foreign_count=len(foreign), ceiling=q['total_ceiling_sim_s'],
               outstanding_ceiling=need, remaining=cap-mine, eval_root_spent=prerun)
    if allocated is None:
        raise Blocked(f'the clean run CONFIG carries no sealed {PHASE} allocation: {out}')
    if foreign:
        raise Blocked(f'foreign {PHASE} charges in the clean4 ledger (in-run evaluation not off?): {out}')
    if mine+need > cap:
        raise Blocked(f'{PHASE} sub-cap would be exceeded: {out}')
    return out


# ---------------------------------------------------------------- comparators (zero physics)

def _arm(arm):
    if arm not in BOUND_ARMS:
        raise Blocked(f'comparator arm must be one of {BOUND_ARMS}')
    return arm


def mappo_binding(final_root, arm='MAPPO_SPEED'):
    path = Path(final_root)/f'BOUND_{_arm(arm)}.json'
    if not path.is_file():
        return None
    value = verify(read(path), 'binding_sha256', f'{arm} binding')
    if value.get('arm') != arm or value['kind'] not in MAPPO_KINDS[arm]:
        raise Blocked(f'{arm} binding names another arm/kind')
    return dict(value, binding_path=str(path))      # location only; the sealed record is unchanged


def declared_unavailable(final_root, arm):
    path = Path(final_root)/f'DECLARED_UNAVAILABLE_{_arm(arm)}.json'
    return verify(read(path), 'declaration_sha256', f'{arm} declaration') if path.is_file() else None


def comparators(final_root, arms=BOUND_ARMS):
    bound = {a: b for a in arms if (b := mappo_binding(final_root, a)) is not None}
    declared = {a: d for a in arms if a not in bound and (d := declared_unavailable(final_root, a)) is not None}
    return dict(bound=bound, declared=declared, pending=[a for a in arms if a not in bound and a not in declared])


def mappo_label(sha256, arm='MAPPO_SPEED'):
    """A comparator is named by its registered kind plus the sha actually loaded."""
    kind = next(iter(MAPPO_KINDS[arm]))
    return kind, f'{MAPPO_KINDS[arm][kind]} [executor {str(sha256)[:12]}]'


def export_meta(path):
    """Export metadata only (weights_only load, no network construction)."""
    import torch
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise Blocked(f'{path} is not a c4 export')
    policy = payload.get('policy') or {}
    return dict(format=payload.get('format'), trainer=payload.get('trainer'), iteration=payload.get('iteration'),
                lane_control=payload.get('lane_control'), lane_head=payload.get('lane_head') is True
                and any(str(k).startswith('lane_head.') for k in policy))


def check_export(arm, sha256, meta):
    """The registered binding rule per comparator arm (MAIN 2026-11-06)."""
    if any(sha256.startswith(x) for x in REFUSED_SHA_PREFIXES[arm]):
        raise Blocked(f'{arm} never substitutes {sha256[:8]} (registered binding)')
    if arm in REQUIRED_SHA and sha256 != REQUIRED_SHA[arm]:
        raise Blocked(f'{arm} is the frozen {REQUIRED_SHA[arm][:8]} executor only')
    if meta.get('trainer') != 'mappo':
        raise Blocked(f'{arm} must be a mappo export')
    if arm == 'MAPPO_SPEED':
        if meta.get('iteration') != REQUIRED_ITERATION[arm]:
            raise Blocked(f'MAPPO_SPEED is S-MAPPOWZ it168 only (export iteration {meta.get("iteration")})')
        if meta.get('lane_head'):
            raise Blocked('MAPPO_SPEED must be speed-only (export has a lane head)')
    if arm == CT_ARM and (meta.get('lane_head') or sha256 == FROZEN_MAPPO_SHA):
        raise Blocked('MAPPO_CT must be a speed-only export that differs from the frozen U0 executor')
    if arm == 'MAPPO_LANE' and not meta.get('lane_head'):
        raise Blocked('MAPPO_LANE must carry its lane head')
    return meta


def _wave_started(final_root, arm):
    root = Path(final_root)
    return any((root/f'{kind}_{name}.json').is_file() or any(root.glob(f'{kind}_{name}__*.json'))
               for kind in ('RESERVED', 'WAVE') for name in (arm, f'PARITY_{arm}'))


def bind_mappo(*, final_root, path, sha256, kind, authority, arm='MAPPO_SPEED', check=None, dry_run=False,
               residual_path=None, residual_sha256=None, train_config=None, train_src=None):
    """Bind a comparator executor (path + sha256, re-verified at every load). Rebinding is allowed
    only before that arm's wave is reserved; earlier bindings are kept."""
    final_root, arm = Path(final_root), _arm(arm)
    if run_plan(final_root) is None and prerun_plan(final_root) is None and trained_plan(final_root) is None:
        raise Blocked('RUN_PLAN.json / PRERUN_PLAN.json / TRAINED_PLAN.json missing: run register, prerun-register '
                      'or confirm-register first')
    residual = None
    if arm == 'U0':
        if not residual_path or not re.fullmatch('[0-9a-f]{64}', str(residual_sha256 or '')):
            raise Blocked('U0 needs the zero-residual u0 checkpoint: --residual-path and --residual-sha256')
        if file_sha(residual_path) != residual_sha256:
            raise Blocked(f'{residual_path} does not hash to {residual_sha256}')
        residual = dict(residual_path=str(residual_path), residual_sha256=residual_sha256)
    if train_config is not None:          # F11: the comparator's own training run config (N7 toml)
        if arm == 'U0':
            raise Blocked('U0 has no F11 parity (frozen executor, not trained on wz3to2)')
        residual = dict(residual or {}, train_config=str(train_config), train_config_sha256=file_sha(train_config))
    if kind not in MAPPO_KINDS[arm]:
        raise Blocked(f'{arm} kind must be one of {sorted(MAPPO_KINDS[arm])}')
    if not re.fullmatch('[0-9a-f]{64}', str(sha256)):
        raise Blocked('full lowercase sha256 required')
    if not str(authority).strip():
        raise Blocked('authority (verbatim decision/message) required')
    if file_sha(path) != sha256:
        raise Blocked(f'{path} does not hash to {sha256}')
    meta = export_meta(path)
    if dry_run and arm in REQUIRED_SHA:      # dry run only: a stand-in cannot hash to the frozen executor
        meta = dict(check_export(arm, REQUIRED_SHA[arm], meta), dry_run_stand_in=True)
    else:
        meta = check_export(arm, sha256, meta)
    if arm == 'MAPPO_LANE':          # amendment 17: the S-MAPPOWZL training execution path (tree manifest bound)
        if train_config is None or train_src is None:
            raise Blocked('MAPPO_LANE needs --train-config and --train-src (its training source tree)')
        from tools.coopmwm.nm_eval_runner import LANE_TRAIN_FILES
        missing = [f for f in LANE_TRAIN_FILES if not (Path(train_src)/f).is_file()]
        if missing:
            raise Blocked(f'--train-src lacks the training-path files {missing}')
        manifest = {f: file_sha(Path(train_src)/f) for f in LANE_TRAIN_FILES}
        residual = dict(residual or {}, train_src=str(train_src), train_src_files=manifest,
                        train_src_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
    if declared_unavailable(final_root, arm) is not None:
        raise Blocked(f'{arm} was declared unavailable; that declaration stands')
    if _wave_started(final_root, arm):
        raise Blocked(f'the {arm} wave already started; the comparator can no longer change')
    target = final_root/f'BOUND_{arm}.json'
    if target.is_file():
        n = len(list(final_root.glob(f'BOUND_{arm}_superseded_*.json')))+1
        publish(final_root/f'BOUND_{arm}_superseded_{n:03}.json', read(target))
    value = seal(dict(schema=SCHEMA+'.comparator', arm=arm, kind=kind, label=MAPPO_KINDS[arm][kind], path=str(path),
                      sha256=sha256, export=meta, authority=str(authority), rule=MAPPO_AUTHORITY, load_check=check,
                      dry_run=bool(dry_run), **(residual or {}),
                      bound_utc=datetime.now(timezone.utc).isoformat()), 'binding_sha256')
    publish(target, value, replace=True)
    return dict(value, binding_path=str(target))


def declare_unavailable(*, final_root, arm, authority, reason):
    """F12: an explicit, sealed 'comparator unavailable' declaration (never implied by a timeout)."""
    final_root, arm = Path(final_root), _arm(arm)
    if run_plan(final_root) is None and prerun_plan(final_root) is None:
        raise Blocked('RUN_PLAN.json / PRERUN_PLAN.json missing')
    if not str(authority).strip() or not str(reason).strip():
        raise Blocked('authority and reason required')
    if mappo_binding(final_root, arm) is not None:
        raise Blocked(f'{arm} is bound; a bound comparator cannot be declared unavailable')
    value = seal(dict(schema=SCHEMA+'.unavailable', arm=arm, authority=str(authority), reason=str(reason),
                      declared_utc=datetime.now(timezone.utc).isoformat()), 'declaration_sha256')
    publish(final_root/f'DECLARED_UNAVAILABLE_{arm}.json', value)
    return value


def bind_arms(*, final_root, shas, authority, update=PRIMARY_UPDATE):
    """MAIN: bind clean5a B/R (and optionally A) at the evaluated update by sha before the evaluation; the gate
    refuses any other file. One sealed ARMS_BOUND_u<update>.json per update, never rebound."""
    final_root, update = Path(final_root), check_update(update)
    if run_plan(final_root) is None:
        raise Blocked('RUN_PLAN.json missing: run register first')
    shas = {k: v for k, v in shas.items() if v is not None}
    if not {'B', 'R'} <= set(shas) <= set(ARMS) or not all(re.fullmatch('[0-9a-f]{64}', str(v)) for v in shas.values()):
        raise Blocked(f'B and R (A optional) u{update} checkpoint sha256 required')
    if not str(authority).strip():
        raise Blocked('authority required')
    if (final_root/f'ARMS_BOUND_u{update:03}.json').exists() or (final_root/f'GATE_PASS_u{update:03}.json').exists():
        raise Blocked(f'u{update} arms already bound (ARMS_BOUND or GATE_PASS exists); never rebound')
    value = seal(dict(schema=SCHEMA+'.arms', update=update, checkpoint_sha256=dict(shas), authority=str(authority),
                      bound_utc=datetime.now(timezone.utc).isoformat()), 'binding_sha256')
    publish(final_root/f'ARMS_BOUND_u{update:03}.json', value)
    return value


# ---------------------------------------------------------------- gate (zero physics)

def gate(root, cfg, runtime, binding, final_root=None, update=UPDATE):
    """The evaluated update sealed AND safety clean (A/B/R update receipts + checkpoints, monitors u1..u; COMPLETE
    only at the final u24). NotReady = wait; Blocked = never proceed automatically."""
    root, update = Path(root), int(update)
    names = [f'{{arm}}_UPDATE_{update:03}.json', f'{{arm}}_checkpoint_{update:03}.pt']+(
        ['{arm}_COMPLETE.json'] if update == UPDATE else [])
    missing = [n.format(arm=arm) for arm in ARMS for n in names if not (root/n.format(arm=arm)).is_file()]
    if not (root/f'LEARNING_MONITOR_u{update:03}.json').is_file():
        missing.append(f'LEARNING_MONITOR_u{update:03}.json')
    try:
        runtime.health(root)
    except Exception as error:   # registered health: FAULT / PAUSE_SAFETY / STOP / unreleased PAUSE_*
        raise NotReady(f'clean4 health not clear: {error!r}')
    if missing:
        raise NotReady(f'u{update} not sealed: missing '+', '.join(missing))
    updates = {arm: verify(read(root/f'{arm}_UPDATE_{update:03}.json'), 'update_sha256', f'{arm} u{update} update')
               ['update_sha256'] for arm in ARMS}
    bound, events, pauses = digest(cfg), {}, []
    for u in range(1, update+1):
        path = root/f'LEARNING_MONITOR_u{u:03}.json'
        if not path.is_file():
            raise NotReady(f'monitor receipt u{u:03} missing')
        row = verify(read(path), 'receipt_sha256', f'monitor u{u:03}')
        if row.get('config_sha256') != bound or row.get('update') != u:
            raise Blocked(f'monitor u{u:03} is not bound to this CONFIG')
        if sorted(a['arm'] for a in row['arms']) != list(ARMS):
            raise Blocked(f'monitor u{u:03} lacks an arm')
        for a in row['arms']:
            if a.get('action') == 'PAUSE_SAFETY':
                pauses.append((u, a['arm']))
            for k in HARD:
                if a['metrics']['safety'][k]:
                    events[f'u{u:03}_{a["arm"]}_{k}'] = a['metrics']['safety'][k]
    if pauses or events:
        raise Blocked(f'safety not clean: pauses={pauses} events={events}')
    checkpoints = {arm: file_sha(root/f'{arm}_checkpoint_{update:03}.pt') for arm in ARMS}
    bound_arms = Path(final_root)/f'ARMS_BOUND_u{update:03}.json' if final_root is not None else None
    if bound_arms is not None and bound_arms.is_file():
        expected = verify(read(bound_arms), 'binding_sha256', 'ARMS_BOUND')['checkpoint_sha256']
        if any(checkpoints[a] != sha for a, sha in expected.items()):
            raise Blocked(f'u{update} checkpoints differ from the bound shas: {expected} vs {checkpoints}')
    if binding['base'] == 'mappo_u0' and file_sha(root/'A_checkpoint_000.pt') != cfg['common_checkpoint_sha256']:
        raise Blocked('A_checkpoint_000.pt is not the registered common MAPPO u0')
    a_alias = None
    if a_frozen(cfg):
        a_alias = runtime.frozen_a_check(cfg, root, update)
        if a_alias.get('identical') is not True:
            raise Blocked(f'frozen control arm A differs from the common u0 at u{update}: {a_alias}')
    return dict(schema=SCHEMA+'.gate', status='PASS', root=str(root), config_sha256=bound, update=update,
                update_sha256=updates, checkpoint_sha256=checkpoints, monitors=update, collisions=0, teleports=0,
                base=binding['base'], panel_sha256=binding['panel_sha256'], a_alias_of_u0=a_alias, unix=time.time(),
                new_sim_s=0)


# ---------------------------------------------------------------- specs and records

def check_oracle_binding(binding):
    """The sealed ORACLE binding (amendment 25b): the frozen end-of-training bank (path + file sha256) and the
    oracle_plan digest; bound only after the CLEAN5C5 final update, never before."""
    if binding is None:
        raise Blocked(f'ORACLE arms not bound: no sealed {ORACLE_BINDING} (bound after training ends)')
    bank = binding.get('bank') or {}
    if (not bank.get('path') or not re.fullmatch('[0-9a-f]{64}', str(bank.get('file_sha256')))
            or not re.fullmatch('[0-9a-f]{64}', str(binding.get('oracle_sha256')))
            or not re.fullmatch('[0-9a-f]{64}', str(binding.get('receipt_sha256')))):
        raise Blocked(f'{ORACLE_BINDING}: bank path/file_sha256, oracle_sha256 and receipt_sha256 required')
    if binding['oracle_sha256'] != ORACLE_SHA256:
        raise Blocked(f'{ORACLE_BINDING}: oracle_sha256 is not the registered oracle_plan digest {ORACLE_SHA256[:8]}')
    banks = binding.get('banks') or {}            # amendment 27: per-arm bank override (ORACLE = the union bank)
    if not set(banks) <= set(ORACLE_ARMS) or any(
            not b.get('path') or not re.fullmatch('[0-9a-f]{64}', str(b.get('file_sha256'))) for b in banks.values()):
        raise Blocked(f'{ORACLE_BINDING}: banks overrides only for ORACLE arms, each with path + file_sha256')
    return binding


def arm_bank(binding, arm):
    """The bank an arm runs on: its amendment-27 override, else the common bank (= the applicability-label bank)."""
    bank = (binding.get('banks') or {}).get(arm) or binding['bank']
    return dict(path=bank['path'], file_sha256=bank['file_sha256'])


def oracle_binding(final_root):
    """The sealed ORACLE_BINDING.json of this root, else None. Its bank must be the bank of every sealed
    applicability label file (amendments 24/25: one frozen end-of-training bank)."""
    path = Path(final_root)/ORACLE_BINDING
    if not path.is_file():
        return None
    value = check_oracle_binding(verify(read(path), 'receipt_sha256', path.name))
    for labels in sorted(Path(final_root).glob('APPLICABILITY_*.json')):
        got = verify(read(labels), 'receipt_sha256', labels.name)['bank']['sha256']
        if got != value['bank']['file_sha256']:
            raise Blocked(f'{ORACLE_BINDING} bank {value["bank"]["file_sha256"][:8]} != {labels.name} bank {got[:8]}')
    return value


def episode_specs(arm, panel, cfg, root, base, binding=None, u0=None, update=UPDATE):
    root = Path(root)
    if arm in ARMS:
        path = root/f'{arm}_checkpoint_{int(update):03}.pt'
        checkpoint = dict(path=str(path), file_sha256=file_sha(path), update=int(update), arm=arm)
    elif arm in ('U0', 'RULE', 'RULE_FAIR', 'MAPPO_SPEED', CT_ARM):
        if u0 is None:
            raise Blocked(f'{arm} needs the U0 binding (frozen executor + zero-residual u0): bind-mappo --arm U0')
        exe = u0 if arm in ('U0', 'RULE', 'RULE_FAIR') else binding
        if exe is None:
            raise Blocked(f'{arm} executor not bound (bind-mappo --arm {arm})')
        # executor = the bound frozen/comparator export; residual = the bound zero-residual u0
        checkpoint = dict(kind=dict(U0='u0', RULE='rule', RULE_FAIR='rule', MAPPO_SPEED='mappo_comparator',
                                    MAPPO_CT='mappo_comparator')[arm],
                          comparator_kind=exe['kind'], label=exe['label'], executor_path=exe['path'],
                          executor_sha256=exe['sha256'], binding_sha256=exe['binding_sha256'],
                          binding_path=exe['binding_path'], path=u0['residual_path'], file_sha256=u0['residual_sha256'],
                          update=0, arm='A')
        if arm == 'RULE':
            checkpoint.update(rule=RULE_STRATEGY, rule_sha256=RULE_SHA256, rule_module=RULE_MODULE,
                              rule_source=RULE_SOURCE)
        if arm == 'RULE_FAIR':
            checkpoint.update(rule=RULE_FAIR_STRATEGY, rule_sha256=RULE_FAIR_SHA256,
                              rule_module='tools.coopmwm.regimeu0_multicrit', rule_source='c4/multicrit 4ff09522')
    elif arm == 'MAPPO_LANE':
        if binding is None:
            raise Blocked('MAPPO_LANE comparator not bound (bind-mappo --arm MAPPO_LANE)')
        if not binding.get('train_src') or not binding.get('train_config'):
            raise Blocked('MAPPO_LANE runs only on its training path: bind with --train-config and --train-src')
        checkpoint = dict(kind='mappo_lane', comparator_kind=binding['kind'], label=binding['label'],
                          executor_path=binding['path'], executor_sha256=binding['sha256'],
                          binding_sha256=binding['binding_sha256'], binding_path=binding['binding_path'],
                          train_config=binding['train_config'], train_config_sha256=binding['train_config_sha256'],
                          train_src=binding['train_src'], train_src_sha256=binding['train_src_sha256'])
    elif arm in BANK_ARMS:
        if SETTING not in BANK_SETTINGS:
            raise Blocked(f'{arm} is registered for the p20x primary panel (amendments 25b/26) and the p20c '
                          'confirmation panel (amendment 28) only')
        if u0 is None:
            raise Blocked(f'{arm} needs the U0 binding (frozen executor + zero-residual u0): bind-mappo --arm U0')
        check_oracle_binding(binding)
        checkpoint = dict(kind='oracle', comparator_kind=u0['kind'], label=u0['label'], executor_path=u0['path'],
                          executor_sha256=u0['sha256'], binding_sha256=u0['binding_sha256'],
                          binding_path=u0['binding_path'], path=u0['residual_path'], file_sha256=u0['residual_sha256'],
                          update=0, arm='A', cell_key=panel['cell']['cell_key'],
                          oracle_bank=arm_bank(binding, arm),
                          oracle_binding_sha256=binding['receipt_sha256'])
        if arm in ORACLE_ARMS:
            checkpoint.update(oracle_sha256=binding['oracle_sha256'], oracle_plan_source=ORACLE_PLAN_SOURCE[arm])
        else:
            model = binding.get('learned_trigger_model') or {}
            if not model.get('path') or model.get('file_sha256') != LEARNED_MODEL_SHA256:
                raise Blocked(f'{ORACLE_BINDING}: learned_trigger_model path + the registered file sha required')
            checkpoint.update(kind='learned_trigger', learned_trigger_sha256=LEARNED_TRIGGER_SHA256,
                              learned_trigger_model=dict(path=model['path'], file_sha256=model['file_sha256']))
    elif arm == 'NATIVE':
        checkpoint = dict(kind='native')
    else:
        raise Blocked(f'no {arm} wave')
    capture = dict(rule=panel['contract']['decisions'].get('rule'), min_t_s=DECISION_MIN_T, max_t_s=DECISION_MAX_T,
                   min_members=MIN_MEMBERS, max_members=MAX_MEMBERS)
    tag = arm.lower()+(f'_u{int(update):03}' if arm in ARMS else '')
    subset = set(panel['contract']['decisions']['worlds'])
    prefix = panel.get('job_prefix', 'nmfinal')
    return [dict(job_id=f'{prefix}_{tag}_{i:04}', arm=arm, world_id=r['world_id'], world=deepcopy(r['world']),
                 checkpoint=deepcopy(checkpoint), sim_s=HORIZON_S, window=dict(WINDOW), noise_seed=r['world']['seed'],
                 deterministic=True, panel_sha256=panel['panel_sha256'],
                 capture=deepcopy(capture) if arm in DECISION_ARMS and r['world_id'] in subset else None)
            for i, r in enumerate(panel['worlds'])]


def _safety(record, what):
    safety = record.get('safety')
    if not isinstance(safety, dict) or set(safety) != set(SAFETY) or any(
            type(v) is not int or v < 0 for v in safety.values()):
        raise Blocked(f'{what}: all four non-negative integer safety counters required')


def check_decision(decision, what):
    if decision is None:
        return None
    members = decision.get('members') if isinstance(decision, dict) else None
    t = decision.get('t_s') if isinstance(decision, dict) else None
    if (not isinstance(members, list) or not MIN_MEMBERS <= len(members) <= MAX_MEMBERS
            or len(set(members)) != len(members) or not all(isinstance(m, str) and m for m in members)
            or isinstance(t, bool) or not isinstance(t, (int, float)) or not DECISION_MIN_T <= t <= DECISION_MAX_T
            or not isinstance(decision.get('move'), str) or not decision['move']
            or (decision.get('state_sha256') is not None
                and not re.fullmatch('[0-9a-f]{64}', str(decision['state_sha256'])))
            or not isinstance(decision.get('state_ref'), (str, type(None)))
            or (decision.get('state_sha256') is None) != (decision.get('state_ref') is None)):
        raise Blocked(f'{what}: decision outside the registered capture rule')
    return decision


def check_decision_log(log, what):
    """Descriptive log of every new 2+-member cooperative move (B/R); bounded, types only."""
    if not isinstance(log, list) or len(log) > DECISION_LOG_LIMIT or any(
            not isinstance(r, dict) or not isinstance(r.get('move'), str) or not isinstance(r.get('members'), list)
            or len(r['members']) < 2 or isinstance(r.get('t_s'), bool) or not isinstance(r.get('t_s'), (int, float))
            for r in log):
        raise Blocked(f'{what}: malformed decision_log')
    return [dict(t_s=r['t_s'], move=r['move'], members=list(r['members'])) for r in log]


IDENTITY = ('arm', 'world_id', 'world', 'checkpoint', 'window', 'panel_sha256')
DECISION_LOG_LIMIT = 200


def check_episode(record, spec):
    what = f'{spec["arm"]} {spec["world_id"]}'
    if record.get('status') != 'graded':
        raise Blocked(f'{what}: ungraded record; never drop/impute/retry')
    if any(record.get(k) != spec[k] for k in IDENTITY):
        raise Blocked(f'{what}: record identity differs from its registered spec')
    _safety(record, what)
    if spec['capture'] is not None:
        if 'decision' not in record:
            raise Blocked(f'{what}: decision capture missing')
        check_decision(record['decision'], what)
    elif record.get('decision') is not None:
        raise Blocked(f'{what}: decision captured outside the registered B/R decision subset')
    if spec['arm'] in EXECUTOR_ARMS and record.get('executor_sha256_loaded') != spec['checkpoint']['executor_sha256']:
        raise Blocked(f'{what}: comparator executor actually loaded differs from its binding (never silent)')
    if spec['arm'] in DECISION_ARMS:
        log = check_decision_log(record.get('decision_log', []), what)
    elif record.get('decision_log'):
        raise Blocked(f'{what}: only B/R log cooperation decisions')
    else:
        log = None
    vlog = record.get('vehicle_log')
    if (not isinstance(vlog, dict) or vlog.get('file') != VEHICLE_LOG or vlog.get('ticks') != HORIZON_S-WARMUP_S
            or not re.fullmatch('[0-9a-f]{64}', str(vlog.get('sha256'))) or not isinstance(vlog.get('vehicles'), int)):
        raise Blocked(f'{what}: the ground-truth vehicle log (evaluation-only secondaries) is missing or incomplete')
    out = dict({k: deepcopy(record[k]) for k in IDENTITY}, status='graded', job_id=spec['job_id'],
               J_veh_s=fe._finite(record.get('J_veh_s'), f'{what} J_veh_s'), safety=dict(record['safety']),
               decision=deepcopy(record.get('decision')), vehicle_log=dict(vlog))
    if log is not None:
        out['decision_log'] = log
    if spec['arm'] in EXECUTOR_ARMS:
        out['executor_sha256_loaded'] = record['executor_sha256_loaded']
    if record.get('work_dir') is not None:
        if not re.fullmatch(re.escape(spec['job_id'])+r'__r[0-9]+', str(record['work_dir'])):
            raise Blocked(f'{what}: work_dir is not a replay dir of this job')
        out['work_dir'] = record['work_dir']
    if spec['arm'] in ORACLE_ARMS:
        if record.get('oracle_sha256_loaded') != spec['checkpoint']['oracle_sha256']:
            raise Blocked(f'{what}: oracle_plan actually loaded differs from its binding (never silent)')
        if not isinstance(record.get('oracle'), dict):
            raise Blocked(f'{what}: oracle receipts missing')
        out['oracle_sha256_loaded'] = record['oracle_sha256_loaded']
        out['oracle'] = deepcopy(record['oracle'])
    elif record.get('oracle') is not None:
        raise Blocked(f'{what}: oracle receipts outside an ORACLE arm')
    if spec['arm'] in LEARNED_ARMS:
        if record.get('learned_trigger_sha256_loaded') != spec['checkpoint']['learned_trigger_sha256']:
            raise Blocked(f'{what}: learned_trigger actually loaded differs from its registration (never silent)')
        if not isinstance(record.get('learned_trigger'), dict):
            raise Blocked(f'{what}: learned_trigger receipts missing')
        out['learned_trigger_sha256_loaded'] = record['learned_trigger_sha256_loaded']
        out['learned_trigger'] = deepcopy(record['learned_trigger'])
    elif record.get('learned_trigger') is not None:
        raise Blocked(f'{what}: learned_trigger receipts outside the LEARNED_TRIGGER arm')
    if spec['arm'] in RULE_ARMS:
        if record.get('rule_sha256_loaded') != spec['checkpoint']['rule_sha256']:
            raise Blocked(f'{what}: S-REGIMEU0 rule actually loaded differs from its registration')
        out['rule_sha256_loaded'] = record['rule_sha256_loaded']
        out['rule'] = deepcopy(record.get('rule'))
    elif record.get('rule') is not None:
        raise Blocked(f'{what}: rule telemetry outside the RULE arm')
    if spec['arm'] == 'NATIVE' and record.get('regime') is not None:
        regime = record['regime']
        v = regime.get('closing_lane_speed_mps') if isinstance(regime, dict) else 'bad'
        if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)):
            raise Blocked(f'{what}: malformed regime observable')
        out['regime'] = deepcopy(regime)
    return out


def decision_specs(arm, records, panel):
    """records: {world_id: checked episode record}. One branch set per captured decision."""
    out = []
    for i, row in enumerate(panel['worlds']):
        rec = records[row['world_id']]
        dec = rec['decision']
        if dec is None:
            continue
        if dec.get('state_sha256') is None:
            raise Blocked(f'{arm} {rec["world_id"]}: decision has no branchable state snapshot')
        labels = [('all', list(dec['members'])), ('background', [])]+[
            (f'solo{j}', [m]) for j, m in enumerate(dec['members'])]
        for label, new in labels:
            out.append(dict(job_id=f'nmdec_{arm.lower()}_u{int(rec["checkpoint"]["update"]):03}_{i:04}_{label}', arm=arm,
                            world_id=rec['world_id'], world=deepcopy(rec['world']),
                            checkpoint=deepcopy(rec['checkpoint']), decision=deepcopy(dec), label=label,
                            new_members=new, background='BASE', sim_s=int(dec['t_s'])+DECISION_H,
                            window=dict(start_s=dec['t_s'], end_s=dec['t_s']+DECISION_H),
                            panel_sha256=panel['panel_sha256']))
    return out


def check_branch(record, spec):
    what = f'{spec["job_id"]}'
    if record.get('status') != 'graded':
        raise Blocked(f'{what}: ungraded branch; never drop/impute/retry')
    if any(record.get(k) != spec[k] for k in IDENTITY+('decision', 'label', 'new_members')):
        raise Blocked(f'{what}: branch identity differs from its registered spec')
    _safety(record, what)
    return dict({k: deepcopy(record[k]) for k in IDENTITY+('decision', 'label', 'new_members')}, status='graded',
                job_id=spec['job_id'], J_veh_s=fe._finite(record.get('J_veh_s'), f'{what} J_veh_s'),
                safety=dict(record['safety']))


def decision_metrics(branches):
    """clean3 policy_synergy.metrics on one decision: {label: record} -> gains (higher is better)."""
    J = {k: v['J_veh_s'] for k, v in branches.items()}
    members = branches['all']['decision']['members']
    solos = [J[f'solo{j}'] for j in range(len(members))]
    delta_all = J['background']-J['all']
    only = {m: J['background']-s for m, s in zip(members, solos)}
    best = max(only.values())
    return dict(delta_all_new_veh_s=delta_all, delta_only_member_new_veh_s=only, best_solo_veh_s=best,
                policy_cooperation_gain_veh_s=delta_all-best, positive_policy_cooperation=bool(delta_all-best > 0),
                identity_error_veh_s=abs((min(solos)-J['all'])-(delta_all-best)))


# ---------------------------------------------------------------- runtimes

def process_groups(items):
    """Item indices grouped by arm (stable order of first appearance): one fresh pool per group."""
    groups = {}
    for i, (kind, spec, _) in enumerate(items):
        groups.setdefault(spec.get('arm'), []).append(i)
    return list(groups.values())


def _worker(kind, spec, cfg, work, runner):
    module = importlib.import_module(runner)
    return getattr(module, dict(episode='episode', branch='decision_branch', parity='parity', logcheck='logcheck',
                                envparity='envparity')[kind])(spec, cfg, work)


class Runtime:
    """Real cpuV bindings (clean4 deployed source). Tests/dry-run substitute FakeRuntime."""
    physics = True

    def __init__(self, runner=RUNNER):
        self.runner = runner

    def config(self, root):
        from tools.coopmwm import launch_training as t
        return t.checked_config(Path(root)/'CONFIG.json')

    def health(self, root):
        from tools.coopmwm import launch_training as t
        t.health(root)

    def runner_status(self):
        try:
            module = importlib.import_module(self.runner)
        except ImportError as error:
            return dict(ok=False, reason=f'{self.runner} not importable: {error!r}', native=False, decisions=False,
                        lane=False, rule=False, rule_fair=False, parity=False, oracle=False, learned=False)
        ok = getattr(module, 'MAP', None) == MAP and callable(getattr(module, 'episode', None))
        return dict(ok=ok, native=ok and getattr(module, 'NATIVE', True) is not False,
                    decisions=ok and callable(getattr(module, 'decision_branch', None)),
                    lane=ok and getattr(module, 'LANE_TRAIN_PATH', False) is True,
                    rule=ok and getattr(module, 'RULE', False) is True and RULE_STRATEGY in getattr(module, 'RULE_STRATEGIES', ())
                    and importlib.util.find_spec(RULE_MODULE) is not None,
                    rule_fair=ok and 'wall_g_solo2' in getattr(module, 'RULE_STRATEGIES', ())
                    and importlib.util.find_spec('tools.coopmwm.regimeu0_multicrit') is not None,
                    parity=ok and getattr(module, 'PARITY', False) is True and callable(getattr(module, 'parity', None)),
                    oracle=ok and set(ORACLE_ARMS) <= set(getattr(module, 'ORACLE_ARMS', ()))
                    and importlib.util.find_spec('tools.coopmwm.oracle_plan') is not None,
                    learned=ok and set(LEARNED_ARMS) <= set(getattr(module, 'LEARNED_ARMS', ()))
                    and importlib.util.find_spec('tools.coopmwm.learned_trigger') is not None,
                    reason=None if ok else f'{self.runner} lacks MAP={MAP!r} or episode()')

    def frozen_a_check(self, cfg, root, update):
        """Zero physics: A's policy weights at the update (production loader) equal the common u0's."""
        import torch
        from tools.coopmwm import launch_production as production
        native = dict(production.backend_config(cfg), run_root=str(root))
        at, u0 = (production.checkpoint_weights(native, 'A', u) for u in (int(update), 0))
        same = set(at) == set(u0) and all(torch.equal(at[k], u0[k]) if hasattr(at[k], 'shape') else at[k] == u0[k]
                                          for k in at)
        return dict(identical=bool(same), tensors=len(at), update=int(update), method='launch_production.checkpoint_weights')

    def plant_view(self, cfg):
        """What NATIVE/U0/RULE episodes read from the plant CONFIG (validated by the production loader)."""
        from tools.coopmwm import launch_production as production
        native = production.backend_config(cfg)
        return {k: deepcopy(native.get(k)) for k in PLANT_KEYS}

    def run(self, items, cfg, workers):
        """One FRESH process pool per arm (Python 3.10 has no max_tasks_per_child): a NATIVE episode imports the
        env before any G5 binding, so a worker that ran NATIVE must never run a policy/plant episode (the
        2026-11-07 01:09 3c failure). Groups run one after another; results keep the item order."""
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        os.environ['COOPMWM_G5_RUNTIME'] = '1'   # same bound runtime env as the clean3/clean4 owners
        if cfg.get('lc_families'):
            # clean5b (LCEXEC): the training chain requires COOPMWM_LC_RUNTIME=1; B/R lane programs need the env
            # knobs, and without a request the LCEXEC env step is byte-identical (lc_runtime golden test)
            os.environ['COOPMWM_LC_RUNTIME'] = '1'
        results = [None]*len(items)
        for group in process_groups(items):
            with ProcessPoolExecutor(max_workers=min(workers, len(group)),
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = {i: pool.submit(_worker, *items[i][:2], cfg, items[i][2], self.runner) for i in group}
                for i, f in futures.items():
                    results[i] = f.result(timeout=3600)
        return results


class MirrorRuntime(Runtime):
    """Evaluation host != training host (MAIN 23:10: the u12 primary runs on cpuV while clean5a trains on cpuZ):
    the clean run root is a read-only MIRROR at the same absolute path. CONFIG = the registered file (sha given),
    validated by the production loader; health = no FAULT/PAUSE_SAFETY/STOP receipt (processes live elsewhere);
    A/B/R checkpoints are bound by sha (bind-arms) and the monitors are sealed to this CONFIG digest."""

    def __init__(self, runner=RUNNER, config_sha256=None):
        super().__init__(runner)
        if not re.fullmatch('[0-9a-f]{64}', str(config_sha256 or '')):
            raise Blocked('mirror: --config-sha256 (the registered clean run CONFIG.json file sha256) required')
        self.config_sha256 = config_sha256

    def config(self, root):
        path = Path(root)/'CONFIG.json'
        if file_sha(path) != self.config_sha256:
            raise Blocked(f'mirror CONFIG.json does not hash to {self.config_sha256}')
        cfg = read(path)
        if Path(cfg['run_root']) != Path(root):
            raise Blocked('the mirror must sit at the clean run root path')
        self.plant_view(cfg)
        return cfg

    def health(self, root):
        for name in ('FAULT.json', 'PAUSE_SAFETY.json', 'STOP.json'):
            if (Path(root)/name).exists():
                raise ValueError('mirror health: '+name)


def _h(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


_FAKE_LOGS = {}


def _fake_vehicle_log(path, seed, arm, t0=WARMUP_S, t1=HORIZON_S):
    """Dry run: one small but complete log per (arm, world class) - written once, copied (fast)."""
    import gzip
    import shutil
    path.parent.mkdir(parents=True, exist_ok=True)
    shift = (_h(arm) + seed % 3) % 7
    key = (arm, seed % 3, t0, t1)
    if key in _FAKE_LOGS and Path(_FAKE_LOGS[key]).is_file():
        shutil.copyfile(_FAKE_LOGS[key], path)
        return dict(file=VEHICLE_LOG, sha256=file_sha(path), ticks=t1-t0, vehicles=6)
    _FAKE_LOGS[key] = str(path)
    with gzip.open(path, 'wt', encoding='utf-8', compresslevel=1) as f:
        for k in range(6):
            f.write(json.dumps(dict(veh=k, id=f'main.{k}', cav=k % 2 == 0, origin='main_in' if k < 4 else 'ramp_in',
                                    depart_delay_s=float(k), first_t=t0))+'\n')
        for t in range(t0, t1, 1):
            ks = list(range(6))
            f.write(json.dumps(dict(t=t, k=ks, s=[round(-300.+30*k+(12.+shift/7)*((t-t0) % 60), 2) for k in ks],
                                    v=[max(0., 12.-k-((t+shift) % 7)) for k in ks],
                                    a=[round(-.5+.25*((t+k+shift) % 5), 3) for k in ks], lane=[k % 2 for k in ks]))+'\n')
    return dict(file=VEHICLE_LOG, sha256=file_sha(path), ticks=t1-t0, vehicles=6)


def fake_episode(spec, work=None):
    w, arm = spec['world'], spec['arm']
    u = _h(w['seed'], arm)/0xffffffff
    factor = dict(NATIVE=1.06, U0=1.0, RULE=.985, RULE_FAIR=.99, MAPPO_SPEED=.995, MAPPO_LANE=.993, A=0.99, R=0.985, ORACLE=.97, ORACLE_WM=.98, LEARNED_TRIGGER=.985, MAPPO_CT=.996,
                  B=0.975)[arm]
    record = dict({k: deepcopy(spec[k]) for k in IDENTITY}, status='graded',
                  vehicle_log=(_fake_vehicle_log(Path(work)/VEHICLE_LOG, w['seed'], arm) if work is not None else
                               dict(file=VEHICLE_LOG, sha256=hashlib.sha256(spec['job_id'].encode()).hexdigest(),
                                    ticks=HORIZON_S-WARMUP_S, vehicles=1500+_h(w['seed'], arm) % 300)),
                  J_veh_s=float(100_000*factor*(1+0.03*(u-.5))),
                  safety=dict(collisions=0, teleports=0, hard_brakes=_h(w['seed'], arm, 'b') % 7,
                              ttc_events=_h(w['seed'], arm, 't') % 11))
    if arm in EXECUTOR_ARMS:
        record['executor_sha256_loaded'] = spec['checkpoint']['executor_sha256']
    if arm in ORACLE_ARMS:
        record['oracle_sha256_loaded'] = spec['checkpoint']['oracle_sha256']
        record['oracle'] = dict(plan_source=spec['checkpoint']['oracle_plan_source'], activations=_h(w['seed'], arm) % 4)
    if arm in LEARNED_ARMS:
        record['learned_trigger_sha256_loaded'] = spec['checkpoint']['learned_trigger_sha256']
        record['learned_trigger'] = dict(executor='learned_trigger', fires=_h(w['seed'], arm) % 4)
    if arm in RULE_ARMS:
        record['rule_sha256_loaded'] = spec['checkpoint']['rule_sha256']
        record['rule'] = dict(strategy=spec['checkpoint']['rule'], activations=_h(w['seed'], 'rule') % 9+1, issued=600,
                              lane_held_steps=600, members=_h(w['seed'], 'rule') % 9+1)
    if arm == 'NATIVE':
        record['regime'] = dict(closing_lane_speed_mps=4.+_h(w['seed'], 'regime') % 1500/100., n_obs=60)
    if arm in DECISION_ARMS:
        h = _h(w['seed'], arm, 'd')
        n = MIN_MEMBERS+h % 2
        first = None if h % 10 < 2 else dict(
            t_s=DECISION_MIN_T+h % (DECISION_MAX_T-DECISION_MIN_T+1), move='abreast_wall' if n == 3 else 'closing_pair',
            members=[f'main.{100+j+h % 50}' for j in range(n)])
        record['decision_log'] = [] if first is None else [deepcopy(first), dict(first, t_s=first['t_s']+120)]
        if spec['capture'] is not None:
            record['decision'] = None if first is None else dict(
                first, state_sha256=hashlib.sha256(str(h).encode()).hexdigest(), state_ref=f'states/{spec["job_id"]}.pt')
    return record


def fake_branch(spec):
    h = _h(spec['world']['seed'], spec['arm'], spec['label'])
    base = 32_000+(h % 2000)
    J = dict(background=base, all=base-(2_500 if spec['arm'] == 'B' else 1_200)).get(
        spec['label'], base-900-(h % 700))
    return dict({k: deepcopy(spec[k]) for k in IDENTITY+('decision', 'label', 'new_members')}, status='graded',
                J_veh_s=float(J), safety=dict(collisions=0, teleports=0, hard_brakes=h % 3, ttc_events=h % 4))


def fake_logcheck(spec, work):
    """Dry run: a small but complete ground-truth log (both classes, origins, depart delays, kinematics)."""
    import gzip
    path = Path(work)/VEHICLE_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for k in range(6):
            f.write(json.dumps(dict(veh=k, id=f'main.{k}', cav=k % 2 == 0, origin='main_in' if k < 4 else 'ramp_in',
                                    depart_delay_s=float(k), first_t=300 if k < 4 else 310))+'\n')
        for t in range(300, spec['sim_s']):
            ks = [k for k in range(6) if k < 4 or t >= 310]
            f.write(json.dumps(dict(t=t, k=ks, s=[round(-200.+30*k+12.*(t-300), 2) for k in ks],
                                    v=[max(0., 12.-k-(t % 7)) for k in ks], a=[round(-.5+.25*((t+k) % 5), 3) for k in ks],
                                    lane=[k % 2 for k in ks]))+'\n')
    return dict({k: deepcopy(spec[k]) for k in IDENTITY}, status='logged', kind='logcheck',
                vehicle_log=dict(file=VEHICLE_LOG, sha256=file_sha(path), ticks=spec['sim_s']-300, vehicles=6),
                work=str(work), pid=1000+len(spec['arm']))


def fake_envparity(spec, work):
    """Dry run: both paths write the same fake log (IDENTICAL) unless the world seed is odd and FAKE_ENVPAR_DIFF."""
    vlog = _fake_vehicle_log(Path(work)/VEHICLE_LOG, spec['world']['seed'], 'U0' if spec['path'] == 'plant' or not
                             FAKE_ENVPAR_DIFF else 'train', t1=spec['sim_s'])
    return dict({k: deepcopy(spec[k]) for k in IDENTITY}, status='traced', kind='envparity', path=spec['path'],
                J_veh_s=1000.0, safety=dict(collisions=0, teleports=0, hard_brakes=0), vehicle_log=vlog)


FAKE_ENVPAR_DIFF = False


def fake_parity(spec):
    """Dry run: both paths trace the same deterministic fake CAV states and commands (IDENTICAL)."""
    seed = spec['world']['seed']
    trace = []
    for t in range(300, spec['sim_s']):
        cav = {f'main.{j}': [round(-1500.+j*40+(t-300)*20., 4), 20., j % 3, round(-.5+(_h(seed, j, t) % 100)/100., 4),
                             (j % 3 if spec['arm'] == 'MAPPO_LANE' else None)] for j in range(5)}
        trace.append(dict(t=t, n=9, cav=cav))
    return dict({k: deepcopy(spec[k]) for k in IDENTITY}, status='traced', kind='parity', path=spec['path'],
                trace=trace, executor_sha256_loaded=spec['checkpoint']['executor_sha256'],
                safety=dict(collisions=0, teleports=0))


class FakeRuntime:
    """No simulator, no process pool, fake clean4 ledger: fake physics for the dry run/tests."""
    physics = False

    def __init__(self, runner=None):
        self.calls = []
        # decisions=False like the real runner (F7 not built, MAIN decision); tests may switch branches on
        self.status = dict(dict(ok=True, native=True, decisions=False, lane=True, rule=True, rule_fair=True, parity=True,
                                reason=None),
                           **(runner or {}))

    def config(self, root):
        cfg = read(Path(root)/'CONFIG.json')
        if Path(cfg['run_root']) != Path(root):
            raise Blocked('dry CONFIG must point at the dry root')
        return cfg

    def health(self, root):
        for name in ('FAULT.json', 'PAUSE_SAFETY.json', 'STOP.json'):
            if (Path(root)/name).exists():
                raise ValueError('dry health: '+name)

    def runner_status(self):
        return dict(self.status)

    frozen_a_identical = True

    def frozen_a_check(self, cfg, root, update):
        return dict(identical=self.frozen_a_identical, tensors=0, update=int(update), method='dry run')

    def plant_view(self, cfg):
        # the production loader's check_pins rule (rerun_cpus): a standalone run accounts in its OWN root
        if cfg.get('accounting_root') != cfg.get('run_root'):
            raise ValueError('REFUSED: the standalone experiment must account in its OWN root')
        return {k: deepcopy(cfg.get(k)) for k in PLANT_KEYS}

    def run(self, items, cfg, workers):
        self.calls.append(('run', len(items), workers))
        return [fake_logcheck(spec, work) if kind == 'logcheck' else fake_episode(spec, work) if kind == 'episode' else
                fake_envparity(spec, work) if kind == 'envparity' else
                dict(branch=fake_branch, parity=fake_parity)[kind](spec) for kind, spec, work in items]


# ---------------------------------------------------------------- waves (reserve whole wave, never replay)

# ---------------------------------------------------------------- F11 parity (registered; descriptive)

def parity_specs(arm, panel, cfg, root, base, binding, u0):
    """2 TRAIN worlds (D3200_p60) x 2 paths (eval plant, training env) for one bound MAPPO comparator."""
    from tools.coopmwm import map_scenarios as ms
    if arm not in COMPARATOR_ARMS:
        raise Blocked('F11 parity is for the MAPPO comparators only')
    if binding is None or not binding.get('train_config'):
        raise Blocked(f'F11: bind {arm} with --train-config (its training run config) first')
    checkpoint = episode_specs(arm, panel, cfg, root, base, binding, u0)[0]['checkpoint']
    out = []
    for seed in PARITY_WORLDS:
        world = ms.world(seed, PARITY_CELL['demand'], PARITY_CELL['penetration'])
        for path in ('plant', 'train'):
            out.append(dict(job_id=f'nmparity_{arm.lower()}_{path}_{seed}', kind='parity', arm=arm, path=path,
                            world_id=f'{MAP}_train_{seed}', world=deepcopy(world), checkpoint=deepcopy(checkpoint),
                            sim_s=300+PARITY_ROLLOUT_S, rollout_s=PARITY_ROLLOUT_S,
                            window=dict(start_s=300, end_s=300+PARITY_ROLLOUT_S), panel_sha256=PARITY_TAG,
                            noise_seed=seed, deterministic=True, capture=None, train_config=binding['train_config'],
                            train_config_sha256=binding['train_config_sha256']))
    return out


def check_parity(record, spec):
    what = f'parity {spec["arm"]} {spec["path"]} {spec["world_id"]}'
    if record.get('status') != 'traced' or record.get('kind') != 'parity' or record.get('path') != spec['path']:
        raise Blocked(f'{what}: untraced record; never drop/impute/retry')
    if any(record.get(k) != spec[k] for k in IDENTITY):
        raise Blocked(f'{what}: record identity differs from its registered spec')
    if record.get('executor_sha256_loaded') != spec['checkpoint']['executor_sha256']:
        raise Blocked(f'{what}: comparator executor actually loaded differs from its binding (never silent)')
    trace = record.get('trace')
    if not isinstance(trace, list) or not trace or any(not isinstance(x.get('t'), int) or not isinstance(x.get('cav'), dict)
                                                       for x in trace):
        raise Blocked(f'{what}: malformed trace')
    return dict({k: deepcopy(record[k]) for k in IDENTITY}, status='traced', kind='parity', path=spec['path'],
                job_id=spec['job_id'], trace=deepcopy(trace), executor_sha256_loaded=record['executor_sha256_loaded'],
                safety={k: int(v) for k, v in (record.get('safety') or {}).items()})


def parity_compare(plant, train, tol=PARITY_TOL):
    """Per tick and CAV: |ds|, |dv|, |d command| and lane targets between the two paths (zero physics)."""
    P, T = {x['t']: x['cav'] for x in plant['trace']}, {x['t']: x['cav'] for x in train['trace']}
    ticks = sorted(set(P) & set(T))
    worst = dict(s_m=0., v_mps=0., command_mps2=0.)
    pairs = agree = set_mismatch = presence = lane_mismatch = 0
    first = None
    for t in ticks:
        a, b = P[t], T[t]
        bad = set(a) != set(b)
        set_mismatch += bad
        for vid in sorted(set(a) & set(b)):
            (sa, va, la, ca, ta), (sb, vb, lb, cb, tb) = a[vid], b[vid]
            pairs += 1
            ds, dv = abs(sa-sb), abs(va-vb)
            dc = abs(ca-cb) if ca is not None and cb is not None else (0. if ca is None and cb is None else None)
            presence += dc is None
            lane_mismatch += (la != lb) or (ta != tb)
            worst['s_m'], worst['v_mps'] = max(worst['s_m'], ds), max(worst['v_mps'], dv)
            if dc is not None:
                worst['command_mps2'] = max(worst['command_mps2'], dc)
            ok = (ds <= tol['s_m'] and dv <= tol['v_mps'] and dc is not None and dc <= tol['command_mps2']
                  and la == lb and ta == tb)
            agree += ok
            bad = bad or not ok
        if bad and first is None:
            first = t
    identical = (bool(ticks) and set(P) == set(T) and not set_mismatch and agree == pairs and pairs > 0)
    return dict(verdict='IDENTICAL' if identical else 'DIFFERS', ticks_plant=len(P), ticks_train=len(T),
                ticks_common=len(ticks), cav_set_mismatch_ticks=set_mismatch, pairs=pairs, agree_pairs=agree,
                agree_fraction=round(agree/pairs, 6) if pairs else None, command_presence_mismatch=presence,
                lane_mismatch=lane_mismatch, max_abs=worst, first_divergence_t=first, tolerances=dict(tol))


def parity_report(arm, records, specs, binding):
    by = {(r['path'], r['world']['seed']): r for r in records}
    worlds = {}
    for seed in PARITY_WORLDS:
        worlds[str(seed)] = dict(compare=parity_compare(by[('plant', seed)], by[('train', seed)]),
                                 safety={p: by[(p, seed)]['safety'] for p in ('plant', 'train')})
    verdict = 'IDENTICAL' if all(w['compare']['verdict'] == 'IDENTICAL' for w in worlds.values()) else 'DIFFERS'
    return seal(dict(schema=SCHEMA+'.parity', check='F11', arm=arm, executor_sha256=binding['sha256'],
                     binding_sha256=binding['binding_sha256'], train_config_sha256=binding['train_config_sha256'],
                     worlds=worlds, verdict=verdict, descriptive=True,
                     rule=('FAIRAUDIT F11: the bound comparator in the evaluation plant vs its training env path on 2 '
                           'TRAIN worlds (native 0-300 s warmup + 120 s); reported, never a gate'),
                     jobs=[s['job_id'] for s in specs]), 'receipt_sha256')


def run_parity(*, root, final_root, runtime, panel, arm, cfg=None, base=None, workers=8):
    """Zero-physics bookkeeping around the registered F11 waves (reused if published)."""
    root, final_root = Path(root), Path(final_root)
    plan = run_plan(final_root)
    if plan is None:
        raise Blocked('RUN_PLAN.json missing: run register first')
    cfg = cfg if cfg is not None else runtime.config(root)
    base = base or plan['binding']['base']
    binding, u0 = mappo_binding(final_root, arm), mappo_binding(final_root, 'U0')
    if binding is None:
        raise Blocked(f'{arm} not bound')
    if not runtime.runner_status().get('parity'):
        raise Blocked('runner has no F11 parity()')
    specs = parity_specs(arm, panel, cfg, root, base, binding, u0)
    target = final_root/f'PARITY_{arm}.json'
    done, more = run_waves([(f'PARITY_{arm}', 'parity', specs)], root=root, final_root=final_root, cfg=cfg,
                           runtime=runtime, workers=workers, base=base)
    report = parity_report(arm, done[f'PARITY_{arm}'], specs, binding)
    if target.is_file():
        report = verify(read(target), 'receipt_sha256', target.name)
    else:
        publish(target, report)
    return report, more


def _chunk_files(final_root, name):
    out = []
    for path in Path(final_root).glob(f'WAVE_{name}__*.json'):
        m = re.fullmatch(rf'WAVE_{re.escape(name)}__(\d{{4}})-(\d{{4}})\.json', path.name)
        if m:
            out.append((int(m[1]), int(m[2]), path))
    return sorted(out)


def published(final_root, name, specs, check_):
    """(records, source, covered): the whole-wave receipt, else the contiguous prefix of chunk receipts
    WAVE_<name>__<first>-<last>.json (a pre-run stopped at a chunk boundary and resumed elsewhere)."""
    final_root = Path(final_root)
    target = final_root/f'WAVE_{name}.json'
    if target.is_file():
        value = verify(read(target), 'receipt_sha256', target.name)
        if [r['job_id'] for r in value['records']] != [s['job_id'] for s in specs]:
            raise Blocked(f'{name}: published wave does not match its registered specs')
        return ([check_(r, s) for r, s in zip(value['records'], specs)],
                dict(source=str(target), receipt_sha256=value['receipt_sha256']), len(specs))
    records, paths, shas, covered = [], [], [], 0
    for first, last, path in _chunk_files(final_root, name):
        if first != covered or last < first or last >= len(specs):
            raise Blocked(f'{name}: chunk receipts are not a contiguous partition of the registered specs')
        value = verify(read(path), 'receipt_sha256', path.name)
        part = specs[first:last+1]
        if [r['job_id'] for r in value['records']] != [s['job_id'] for s in part]:
            raise Blocked(f'{name}: chunk {path.name} does not match its registered specs')
        records += [check_(r, s) for r, s in zip(value['records'], part)]
        paths.append(str(path))
        shas.append(value['receipt_sha256'])
        covered = last+1
    return records, dict(source=paths, receipt_sha256=shas, chunks=len(paths)), covered


def run_waves(waves, *, root, final_root, cfg, runtime, workers, base, chunked=False, limit=None, canary=True):
    """waves: [(name, kind, specs)]. Published receipts are reused; an interrupted job (charged, nothing published)
    refuses unless MAIN authorised a replay round (then it runs in a fresh <job>__r<n> dir, charged <job>#r<n>).
    MAIN process rule (2026-11-07): ONE canary job per arm of a dispatch is charged, run and verified (record valid,
    vehicle log non-empty) before the rest of the dispatch is charged. chunked=True publishes WAVE_<name>__<a>-<b>."""
    root, final_root = Path(root), Path(final_root)
    out, sources, todo, done_before = {}, {}, [], {}
    chunked = chunked or bool(limit)
    prerun = isinstance(runtime, PrerunRuntime)
    phase = getattr(runtime, 'phase', PRERUN_PHASE) if prerun else PHASE
    used, rounds = _rounds_used(final_root, root), replay_rounds(final_root)

    def attempt(job):          # 0 = first run; n = authorised replay round n; None = interrupted, needs MAIN
        if job not in used:
            return 0
        return used[job]+1 if rounds.get(job, 0) > used[job] else None
    for name, kind, specs in waves:
        check_ = CHECKS[kind]
        records, source, covered = published(final_root, name, specs, check_)
        if covered == len(specs):
            out[name], sources[name] = records, source
            continue
        if any(attempt(s['job_id']) is None for s in specs[covered:]):
            raise Blocked(f'{name}: interrupted wave in the ledger; never replay without a MAIN authorisation')
        n = (limit.get(name) if isinstance(limit, dict) else limit) or len(specs)
        rest = specs[covered:covered+n]
        done_before[name] = (records, source, covered, len(specs))
        part = f'{name}__{covered:04}-{covered+len(rest)-1:04}' if chunked or covered else name
        todo.append((name, part, kind, rest))
    new = sum(s['sim_s'] for *_, specs in todo for s in specs)
    if new and prerun:              # the whole dispatch must fit BEFORE its canary is charged
        runtime.check_capacity(new)
    if new and not prerun:          # pre-run: PrerunRuntime.reserve enforces the registered pre-run cap
        mine = sum(r['sim_s'] for r in fe.ledger_rows(root) if r.get('phase') == PHASE and str(r['id']).startswith(PREFIXES))
        budget_check(root, cfg, base, outstanding=new, final_root=final_root)
        mine += sum(r['sim_s'] for r in fe.ledger_rows(final_root))
        if mine+new > ALLOCATION_SIM_S:
            raise Blocked(f'{PHASE} sub-cap {ALLOCATION_SIM_S} would be exceeded')

    def ledger_id(s):
        a = attempt(s['job_id'])
        return s['job_id']+(f'#r{a}' if a else '')

    def work_dir(s):
        a = attempt(s['job_id'])
        return final_root/'evaluation'/(s['job_id']+(f'__r{a}' if a else ''))
    for name, part, kind, specs in todo:
        if specs:
            ids = [ledger_id(s_) for s_ in specs]
            tag = max((attempt(s_['job_id']) for s_ in specs), default=0)
            publish(final_root/(f'RESERVED_{part}.replay{tag}.json' if tag else f'RESERVED_{part}.json'),
                    dict(name=part, wave=name, phase=phase, jobs=ids, sim_s=sum(s_['sim_s'] for s_ in specs),
                         ledger=str(final_root/'CHARGES.jsonl'), canary_first=bool(canary), unix=time.time()))
    items = [(kind, s_, str(work_dir(s_))) for _, _, kind, specs in todo for s_ in specs]
    by_item = {id(it[1]): it for it in items}

    def charge(sel):
        jobs = [dict(id=ledger_id(it[1]), sim_s=it[1]['sim_s']) for it in sel]
        if prerun:
            runtime.reserve(cfg, jobs)
        else:                         # the evaluation root's own ledger; the clean run root is never written
            own_reserve(final_root, jobs, PHASE, clean_root=root, calls=getattr(runtime, 'calls', None))
    results = [None]*len(items)
    first = [g[0] for g in process_groups(items)] if canary and items else []
    if first:
        sel = [items[i] for i in first]
        charge(sel)
        for i, rec in zip(first, runtime.run(sel, cfg, len(sel))):
            kind, spec_, work = items[i]
            CHECKS[kind](rec, spec_)                       # raises on any invalid canary record
            if kind == 'episode':
                path = Path(work)/VEHICLE_LOG
                if runtime.physics and (not path.is_file() or path.stat().st_size < 1000
                                        or file_sha(path) != rec['vehicle_log']['sha256']):
                    raise Blocked(f'canary {spec_["job_id"]}: vehicle log missing/empty; the rest is not charged')
            results[i] = rec
    rest_idx = [i for i in range(len(items)) if results[i] is None]
    if rest_idx:
        sel = [items[i] for i in rest_idx]
        charge(sel)
        for i, rec in zip(rest_idx, runtime.run(sel, cfg, workers)):
            results[i] = rec
    del by_item
    position = 0
    for name, part, kind, specs in todo:
        records = results[position:position+len(specs)]
        works = [Path(it[2]).name for it in items[position:position+len(specs)]]
        position += len(specs)
        check_ = CHECKS[kind]
        checked = [check_(r, s_) for r, s_ in zip(records, specs)]
        if kind == 'episode':
            checked = [dict(c, work_dir=w) if w != c['job_id'] else c for c, w in zip(checked, works)]
        if len(checked) != len(specs):
            raise Blocked(f'{name}: incomplete wave; never drop/impute/retry')
        value = seal(dict(schema=SCHEMA+'.wave', name=part, wave=name, phase=phase, kind=kind, physics=runtime.physics,
                          sim_s=sum(s_['sim_s'] for s_ in specs), records=checked), 'receipt_sha256')
        publish(final_root/f'WAVE_{part}.json', value)
        before, source, covered, n = done_before[name]
        if part == name:
            out[name] = checked
            sources[name] = dict(source=str(final_root/f'WAVE_{part}.json'), receipt_sha256=value['receipt_sha256'])
        elif covered+len(checked) == n:            # the last chunk completes the wave
            out[name] = before+checked
            sources[name] = dict(source=list(source.get('source') or [])+[str(final_root/f'WAVE_{part}.json')],
                                 receipt_sha256=list(source.get('receipt_sha256') or [])+[value['receipt_sha256']],
                                 chunks=(source.get('chunks') or 0)+1)
    return out, sources


CHECKS = dict(episode=check_episode, branch=check_branch, parity=check_parity)


# ---------------------------------------------------------------- analysis (zero physics)

def gain_stats(values, bootstrap):
    """Mean and registered whole-world percentile CI of a per-world gain (higher is better)."""
    from tools.coopmwm import planner_eval as pe
    v = np.asarray(values, float)
    if len(v) == 0:
        return dict(n=0, mean=None, ci95=[None, None], p_two_sided=None, n_positive=0)
    ci = pe._paired_ci(v, bootstrap['reps'], bootstrap['seed'], 1-bootstrap['level'])
    return dict(n=len(v), mean=ci['mean_delta_vs'], ci95=ci['ci95'], p_two_sided=ci['p_two_sided'],
                n_positive=int(np.sum(v > 0)))


def analyse_decisions(branches, bootstrap):
    """branches: {arm: [checked branch records]} -> per-decision decomposition tables."""
    out, by_world = {}, {}
    for arm, rows in branches.items():
        groups = {}
        for r in rows:
            groups.setdefault(r['world_id'], {})[r['label']] = r
        decisions = {}
        for world_id, labels in groups.items():
            members = labels['all']['decision']['members'] if 'all' in labels else []
            if not members or set(labels) != {'all', 'background'} | {f'solo{j}' for j in range(len(members))}:
                raise Blocked(f'{arm} {world_id}: incomplete branch set; never imputed')
            decisions[world_id] = dict(decision_metrics(labels), move=labels['all']['decision']['move'],
                                       t_s=labels['all']['decision']['t_s'], members=len(members))
        by_world[arm] = decisions
        rows_ = list(decisions.values())
        moves = sorted({d['move'] for d in rows_})
        safety = {group: {k: int(sum(r['safety'][k] for r in rows if (r['label'] == group or (
            group == 'solo' and r['label'].startswith('solo'))))) for k in SAFETY} for group in ('all', 'background', 'solo')}
        out[arm] = dict(
            decisions=len(rows_), worlds_without_decision=None,
            delta_all_new_veh_s=gain_stats([d['delta_all_new_veh_s'] for d in rows_], bootstrap),
            best_solo_veh_s=gain_stats([d['best_solo_veh_s'] for d in rows_], bootstrap),
            policy_cooperation_gain_veh_s=gain_stats([d['policy_cooperation_gain_veh_s'] for d in rows_], bootstrap),
            positive_cooperation=sum(d['positive_policy_cooperation'] for d in rows_),
            max_identity_error_veh_s=max([d['identity_error_veh_s'] for d in rows_], default=0.0),
            by_move={m: dict(n=sum(d['move'] == m for d in rows_),
                             mean_delta_all=float(np.mean([d['delta_all_new_veh_s'] for d in rows_ if d['move'] == m])),
                             mean_best_solo=float(np.mean([d['best_solo_veh_s'] for d in rows_ if d['move'] == m])),
                             mean_gain=float(np.mean([d['policy_cooperation_gain_veh_s'] for d in rows_ if d['move'] == m])))
                     for m in moves},
            branch_safety=safety,
            branch_collision_or_teleport=any(safety[g][k] for g in safety for k in HARD))
    if {'B', 'R'} <= set(by_world):
        common = sorted(set(by_world['B']) & set(by_world['R']))
        out['B-R_matched_worlds'] = dict(
            note='descriptive: same world, different decisions/states',
            policy_cooperation_gain_veh_s=gain_stats([by_world['B'][w]['policy_cooperation_gain_veh_s']
                                                      - by_world['R'][w]['policy_cooperation_gain_veh_s'] for w in common],
                                                     bootstrap))
    return out, by_world


def analyse(*, panel, base, episodes, decisions=None, replicated=None, sources=None, missing=None, comparators=None,
            scope='claim3a', update=UPDATE, a_alias=False, final_root=None, dry_run=False):
    """episodes: {arm: {world_id: record}} complete panels of EPISODE_ARMS; BASE is an alias of U0
    (mappo_u0 base) or NATIVE (native base), never a separate run. ``comparators``: {arm: binding}."""
    bootstrap = panel['contract']['bootstrap']
    worlds = [r['world_id'] for r in panel['worlds']]
    have = dict(episodes)
    for arm, rows in have.items():
        if arm not in EPISODE_ARMS or set(rows) != set(worlds):
            raise Blocked(f'{arm}: every registered world exactly once required')
    base_arm = alias_of(base)
    if base_arm in have:
        have['BASE'] = have[base_arm]
    J = {n: np.array([have[n][w]['J_veh_s'] for w in worlds]) for n in have}
    S = {n: {k: np.array([have[n][w]['safety'][k] for w in worlds], float) for k in SAFETY} for n in have}
    names = [n for n in ORDER if n in have]
    loaded = {}
    for arm in COMPARATOR_ARMS:
        if arm in have:
            shas = sorted({have[arm][w]['executor_sha256_loaded'] for w in worlds})
            if len(shas) > 1:
                raise Blocked(f'{arm} episodes loaded different executors: {shas}')
            loaded[arm] = shas[0]
    labels = dict(A=(A_ALIAS_LABEL if a_alias else f'A u{update} (the frozen base alone)' if base == 'mappo_frozen'
                     else f'A u{update} (base only)'),
                  B=f'B u{update} (cooperation layer + SUMO-verified guidance)',
                  R=f'R u{update} (cooperation layer, no guidance = the no-guidance arm)',
                  NATIVE='native SUMO CAV driving', U0=U0_LABEL,
                  RULE=RULE_LABEL, RULE_FAIR=RULE_FAIR_LABEL, **ORACLE_LABELS,
                  **{a: mappo_label(loaded[a], a)[1] if a in loaded else f'{a} (not run)' for a in COMPARATOR_ARMS})
    if MAPPO_LANE_OOD:                                   # amendment 22: disclosed in every table that carries the arm
        labels['MAPPO_LANE'] += ' - '+MAPPO_LANE_OOD
    summary = {n: dict(label=labels[n]+(' = BASE' if n == base_arm else ''), n=len(worlds), mean_J_veh_s=float(J[n].mean()),
                       safety_totals={k: int(S[n][k].sum()) for k in SAFETY},
                       any_collision_or_teleport=bool(S[n]['collisions'].sum()+S[n]['teleports'].sum() > 0))
               for n in names}
    tables = dict(arms=summary, base=dict(controller=base, alias_of=base_arm, comparators={
        a: dict({k: b[k] for k in ('kind', 'label', 'sha256', 'binding_sha256')},
                executor_sha256_loaded=loaded.get(a), kind_by_loaded_sha=mappo_label(loaded[a], a)[0] if a in loaded else None)
        for a, b in (comparators or {}).items()}))
    alias = lambda n, c: c == 'BASE' and n == base_arm  # noqa: E731
    total = {}
    for comparator in TOTAL_COMPARATORS:
        if comparator in have:
            total[f'vs_{comparator}'] = dict(
                convention=f'diff = X - {comparator}; negative = X has less delay',
                rows={n: fe.paired_stats(J[n], J[comparator], bootstrap) for n in names
                      if n != comparator and not alias(n, comparator)})
        else:
            total[f'vs_{comparator}'] = dict(status='unavailable', reason=(missing or {}).get(comparator, 'not run'))
    total['base_is_native'] = base == 'native'
    total['base_alias_of'] = base_arm
    tables['total_benefit'] = total
    tables['registered_B_A_R'] = dict(convention='diff = first - second; negative = first has less delay',
                                      rows={c: fe.paired_stats(J[c[0]], J[c[2]], bootstrap)
                                            for c in panel['contract']['comparisons']})
    primary = dict(panel['contract']['primary'])
    if 'BASE' in have:
        row = fe.paired_stats(J['B'], J['BASE'], bootstrap)
        hard = {n: sum(summary[a]['safety_totals'][k] for k in HARD) for n, a in (('B', 'B'), ('BASE', base_arm))}
        primary.update(row=row, hard_events=hard, meets_rule=bool(row['ci95'][1] < 0 and hard['B'] <= hard['BASE']))
    else:
        primary.update(status='unavailable')
    tables['primary'] = primary
    decomposition = {}
    train = ('A-BASE', 'A', 'BASE', 'A over BASE (identically 0 when BASE = A, the frozen base alone)')
    for comparator, chain in (('BASE', (train,)),
                              ('NATIVE', (('BASE-NATIVE', 'BASE', 'NATIVE', 'base controller over native'), train))):
        if not {comparator, 'BASE', 'A', 'B', 'R'} <= set(have):
            decomposition[f'B-{comparator}'] = dict(status=f'unavailable (needs {comparator})')
            continue
        steps = chain+(('R-A', 'R', 'A', 'cooperation layer without guidance (R = the no-guidance arm)'),
                       ('B-R', 'B', 'R', 'SUMO-verified guidance (B vs the no-guidance arm R)'))
        error = float(np.max(np.abs(sum(J[t]-J[c] for _, t, c, _ in steps)-(J['B']-J[comparator]))))
        decomposition[f'B-{comparator}'] = dict(
            identity=f'B-{comparator} = '+' + '.join(f'({k})' for k, *_ in steps)+', exact per world and resample',
            total=fe.paired_stats(J['B'], J[comparator], bootstrap),
            components={k: dict(fe.paired_stats(J[t], J[c], bootstrap, pct=False), meaning=m) for k, t, c, m in steps},
            max_identity_error_veh_s=error)
    per_decision = dict(status='unavailable', reason=(missing or {}).get('DECISIONS', 'not run'))
    if decisions is not None:
        per_decision, _ = analyse_decisions(decisions, bootstrap)
        subset = panel['contract']['decisions']['worlds']
        for arm in DECISION_ARMS:
            if arm in per_decision and arm in have:
                per_decision[arm]['worlds_without_decision'] = sum(have[arm][w]['decision'] is None for w in subset)
        per_decision.update(registered_worlds=len(subset),
                            convention=('gains in veh-s, positive = less delay; policy_cooperation_gain = delta_all - '
                                        'best_solo = min solo J - all J; background = BASE controller'))
    activity = {}
    for arm in DECISION_ARMS:
        if arm in have and all('decision_log' in have[arm][w] for w in worlds):
            logs = [have[arm][w]['decision_log'] for w in worlds]
            firsts = [min(r['t_s'] for r in log) for log in logs if log]
            kinds = {}
            for log in logs:
                for r in log:
                    kinds[r['move']] = kinds.get(r['move'], 0)+1
            activity[arm] = dict(
                worlds=len(worlds), worlds_with_cooperative_move=sum(bool(log) for log in logs),
                cooperative_moves=sum(len(log) for log in logs), by_move=dict(sorted(kinds.items())),
                median_first_move_t_s=float(np.median(firsts)) if firsts else None,
                subset_captures=sum(have[arm][w]['decision'] is not None for w in panel['contract']['decisions']['worlds']))
    tables['synergy'] = dict(panel_decomposition=decomposition, per_decision=per_decision,
                             decision_activity=dict(activity, descriptive=True,
                                                    note='every new 2+-member cooperative move logged in the episode'),
                             registered_u24_train_states=replicated or dict(status='unavailable'))
    safety = dict(totals={n: summary[n]['safety_totals'] for n in names},
                  any_collision_or_teleport={n: summary[n]['any_collision_or_teleport'] for n in names})
    for comparator in ('BASE', 'NATIVE'):
        if comparator in have:
            others = [n for n in names if n != comparator and not alias(n, comparator)]  # BASE row = its alias
            safety[f'vs_{comparator}'] = {n: {k: fe.paired_stats(S[n][k], S[comparator][k], bootstrap, pct=False)
                                              for k in SAFETY} for n in others}
            safety[f'worlds_more_hard_events_than_{comparator}'] = {
                n: [w for i, w in enumerate(worlds) if any(S[n][k][i] > S[comparator][k][i] for k in HARD)] for n in others}
    tables['safety'] = safety
    tables['per_seed'] = per_seed_tables(panel, J, names, worlds, base_arm)
    tables['regimes'] = regime_tables(panel, have, J, names, worlds)
    if final_root is not None:
        alias = {'BASE': base_arm}
        tables['secondary'] = secondary_tables(final_root=final_root, panel=panel, have=have, names=names, alias=alias,
                                               dry_run=dry_run)
    if scope == 'claim3a' and {'A', 'B'} <= set(have):
        # owner decision 04:0x (pre-results): co-primary = total delay B < A (paired CI95 upper < 0) AND groups
        total = fe.paired_stats(J['B'], J['A'], bootstrap)
        groups = (tables.get('secondary') or {}).get('co_primary_groups') or dict(status='unavailable')
        tail = (tables.get('secondary') or {}).get('co_primary_tail') or dict(status='unavailable')
        tables['co_primary'] = dict(
            rule=('B total held-out delay lower than A (paired whole-world CI95 upper bound < 0) AND every group '
                  f'(origin lane 0/1/2, HDV, CAV) mean per-vehicle delay increase <= {GROUP_DELAY_MAX_S} s vs A '
                  'AND (amendment 19) paired CVaR90(B) - CVaR90(A) 95% CI lower bound <= 0'),
            total_B_minus_A=total, total_lower=bool(total['ci95'][1] < 0), groups=groups, tail_cvar90=tail,
            meets=bool(total['ci95'][1] < 0 and groups.get('all_pass') is True and tail.get('passes') is True))
    if scope == 'claim3c':
        # amendment 19: the tail co-criterion applies to the 3c panel too (B vs A = U0 there)
        tail = (tables.get('secondary') or {}).get('co_primary_tail') or dict(status='unavailable')
        tables['co_criterion_tail_3c'] = dict(rule='amendment 19: paired CVaR90(B) - CVaR90(A=U0) 95% CI lower bound <= 0',
                                              tail_cvar90=tail, meets=tail.get('passes') is True)
    value = dict(schema=SCHEMA+'.tables', scope=scope, map=MAP, update=update, a_alias_of_u0=bool(a_alias),
                 evaluation='PRIMARY (u12)' if update == PRIMARY_UPDATE else 'SECONDARY (u24)',
                 panel_sha256=panel['panel_sha256'], base=base,
                 bootstrap=bootstrap, metric=panel['contract']['metric'], horizon_s=HORIZON_S, worlds=len(worlds),
                 present=names, missing=missing or {}, sources=sources or {}, tables=tables,
                 claim=dict(claim3a='does learned cooperation add on top of the frozen MAPPO base: B vs A, B vs R, vs NATIVE '
                                    'and U0 (opens when those arms finish)',
                            claim3b='vs MAPPO-speed it168 and MAPPO-lane, each opened only after it is bound',
                            claim3c='condition-aware: no harm in light traffic (D2400_p60; B, R, U0, NATIVE; opens with 3a)'
                            )[scope[:7]],
                 primary_criterion_training='u24 replicated synergy test on registered TRAIN states (run registration); unchanged')
    return seal(json.loads(json.dumps(value, allow_nan=False)), 'analysis_sha256')


PER_SEED_COMPARATORS = ('NATIVE', 'U0', 'RULE', 'RULE_FAIR', 'MAPPO_SPEED', 'MAPPO_LANE')
TOTAL_COMPARATORS = ('BASE', 'NATIVE', 'RULE', 'RULE_FAIR')+COMPARATOR_ARMS


def seed_stats(diff, worlds):
    """Per-seed view of X - comparator (negative = benefit): fractions, quartiles, sorted table."""
    d = np.asarray(diff, float)
    order = np.argsort(d, kind='stable')
    return dict(n=len(d), frac_benefit=float(np.mean(d < 0)), frac_harm=float(np.mean(d > 0)),
                frac_tie=float(np.mean(d == 0)), quartiles=[float(q) for q in np.quantile(d, [.25, .5, .75])],
                min=float(d.min()), max=float(d.max()),
                sorted=[dict(world_id=worlds[i], diff_veh_s=float(d[i])) for i in order])


def per_seed_tables(panel, J, names, worlds, base_arm):
    """MAIN: every arm vs each comparator (NATIVE, U0, MAPPO-speed, MAPPO-lane) - paired bootstrap mean CI
    plus per-seed benefit/harm fractions, quartiles and a sorted table; B-R and B-A per seed."""
    bootstrap = panel['contract']['bootstrap']
    out = dict(convention='diff = X - comparator per held-out seed; negative = benefit (less delay)', comparators={})
    for c in PER_SEED_COMPARATORS:
        if c not in J:
            out['comparators'][c] = dict(status='unavailable')
            continue
        out['comparators'][c] = {n: dict(mean_ci=fe.paired_stats(J[n], J[c], bootstrap), **seed_stats(J[n]-J[c], worlds))
                                 for n in names if n != c}
    for pair in ('B-R', 'B-A'):
        x, y = pair.split('-')
        out[pair] = (dict(mean_ci=fe.paired_stats(J[x], J[y], bootstrap), **seed_stats(J[x]-J[y], worlds))
                     if x in J and y in J else dict(status='unavailable'))
    cols = [c for c in ('A', 'R')+PER_SEED_COMPARATORS if c in J]
    order = np.argsort(J['B']-J[base_arm], kind='stable') if base_arm in J else range(len(worlds))
    out['table_B'] = dict(sorted_by=f'B-{base_arm}' if base_arm in J else 'panel order',
                          rows=[dict(world_id=worlds[i], seed=panel['worlds'][[r['world_id'] for r in panel['worlds']].index(worlds[i])]['world']['seed'],
                                     J_B=float(J['B'][i]), **{f'B-{c}': float(J['B'][i]-J[c][i]) for c in cols})
                                for i in order])
    return out


def regime_tables(panel, have, J, names, worlds):
    """Report-only regime split: demand cell, and the RSU-observable closing-lane speed tercile measured
    on the untreated NATIVE arm (WM lane 2, 300 m before the drop, t 300-360 s)."""
    bootstrap = panel['contract']['bootstrap']
    cells = sorted({r['world']['cell_key'] for r in panel['worlds']})
    pairs = [(n, c) for c in PER_SEED_COMPARATORS if c in J for n in names if n != c]
    pairs += [(x, y) for x, y in (('B', 'R'), ('B', 'A')) if x in J and y in J]

    def block(mask):
        idx = np.flatnonzero(mask)
        return dict(n=len(idx), rows={f'{n}-{c}': dict(mean=float(np.mean(J[n][idx]-J[c][idx])) if len(idx) else None,
                                                      frac_benefit=float(np.mean(J[n][idx]-J[c][idx] < 0)) if len(idx) else None)
                                      for n, c in pairs})
    keys = [next(r['world']['cell_key'] for r in panel['worlds'] if r['world_id'] == w) for w in worlds]
    out = dict(report_only=True, demand_cell={k: block(np.array([x == k for x in keys])) for k in cells})
    speeds = [((have.get('NATIVE') or {}).get(w) or {}).get('regime', {}).get('closing_lane_speed_mps') for w in worlds]
    known = np.array([v is not None for v in speeds])
    if 'NATIVE' not in have or not known.any():
        out['closing_lane_speed_tercile'] = dict(status='unavailable (needs the NATIVE regime observable)')
        return dict(out, bootstrap_note=f'means only; seed {bootstrap["seed"]} CIs are in per_seed')
    v = np.array([x if x is not None else np.nan for x in speeds], float)
    cuts = [float(c) for c in np.nanquantile(v, [1/3, 2/3])]
    tercile = np.where(~known, -1, np.where(v <= cuts[0], 0, np.where(v <= cuts[1], 1, 2)))
    out['closing_lane_speed_tercile'] = dict(
        cutpoints_mps=cuts, unknown=int((~known).sum()),
        definition='NATIVE arm, RSU-observable rows (CAV reports), WM lane 2, s in [-300, 75.44] m, t in [300, 360) s',
        terciles={name: block(tercile == i) for i, name in enumerate(('slow', 'middle', 'fast'))})
    return out


def _f(x, digits=1):
    return 'n/a' if x is None else f'{x:,.{digits}f}'


def _ci(pair, digits=1):
    return 'n/a' if pair is None or None in pair else f'[{_f(pair[0], digits)}, {_f(pair[1], digits)}]'


def render(value):
    t = value['tables']
    lines = [f'# clean5 wz3to2 final paired evaluation (u{value["update"]}) - {value["scope"]} - '
             f'panel {value["panel_sha256"][:8]}', '',
             f"Claim: {value['claim']}.", '',
             f"Panel: {value['worlds']} held-out wz3to2 worlds, panel_sha256 `{value['panel_sha256']}`; "
             f"base controller {value['base']} (BASE = {t['base']['alias_of']}); comparators "
             f"{', '.join(t['arms'][a]['label'] for a in COMPARATOR_ARMS if a in t['arms']) or 'none run'}; metric {value['metric']} (veh-s, lower is better), full "
             f"{value['horizon_s']} s; paired whole-world percentile bootstrap {value['bootstrap']['reps']} reps, seed "
             f"{value['bootstrap']['seed']}, 95% CI. analysis_sha256 `{value['analysis_sha256']}`.", '']
    if value.get('dry_run'):
        lines += ['**DRY RUN - FAKE PHYSICS - NUMBERS ARE SYNTHETIC AND MEANINGLESS**', '']
    if value['missing']:
        lines += ['Not run: '+'; '.join(f'{k}: {v}' for k, v in value['missing'].items()), '']
    lines += ['## T0 Arms', '', '| arm | label | mean J (veh-s) | collisions | teleports | hard brakes | TTC events |',
              '|---|---|---:|---:|---:|---:|---:|']
    for n, r in t['arms'].items():
        s = r['safety_totals']
        lines.append(f"| {n} | {r['label']} | {_f(r['mean_J_veh_s'])} | {s['collisions']} | {s['teleports']} | "
                     f"{s['hard_brakes']} | {s['ttc_events']} |")

    def block(title, rows, note):
        out = ['', title, '', note, '', '| contrast | diff (veh-s) | 95% CI | % | % 95% CI | p | worlds lower/higher |',
               '|---|---:|---|---:|---|---:|---|']
        for k, r in rows.items():
            out.append(f"| {k} | {_f(r['diff'])} | {_ci(r['ci95'])} | {_f(r.get('pct'), 2)} | {_ci(r.get('pct_ci95'), 2)} "
                       f"| {_f(r['p_two_sided'], 4)} | {r['worlds_lower']}/{r['worlds_higher']} |")
        return out
    p = t['primary']
    lines += ['', '## Primary (pre-registered, descriptive)', '', f"{p['contrast']}: {p['rule']}."]
    if 'row' in p:
        lines.append(f"diff {_f(p['row']['diff'])} veh-s, CI95 {_ci(p['row']['ci95'])}, hard events {p['hard_events']}: "
                     f"meets rule = **{p['meets_rule']}**.")
    for comparator in TOTAL_COMPARATORS:
        tb = t['total_benefit'][f'vs_{comparator}']
        if 'rows' in tb:
            lines += block(f'## T1 Total benefit vs {comparator}', {f'{k}-{comparator}': v for k, v in tb['rows'].items()},
                           tb['convention']+f"; BASE = {t['total_benefit']['base_alias_of']}"+'.')
        else:
            lines += ['', f'## T1 Total benefit vs {comparator}: unavailable ({tb["reason"]})']
    lines += block('## T2 B vs A vs R', t['registered_B_A_R']['rows'], t['registered_B_A_R']['convention']+'.')
    lines += ['', '## T3 Synergy decomposition', '']
    for key, dec in t['synergy']['panel_decomposition'].items():
        if 'components' not in dec:
            lines.append(f'Panel {key}: {dec["status"]}.')
            continue
        lines += [f"Panel identity {dec['identity']} (max per-world error {dec['max_identity_error_veh_s']:.2e} veh-s).", '',
                  '| component | diff (veh-s) | 95% CI | p |', '|---|---:|---|---:|',
                  f"| total {key} | {_f(dec['total']['diff'])} | {_ci(dec['total']['ci95'])} | {_f(dec['total']['p_two_sided'], 4)} |"]
        lines += [f"| {k} | {_f(r['diff'])} | {_ci(r['ci95'])} | {_f(r['p_two_sided'], 4)} |" for k, r in dec['components'].items()]
        lines.append('')
    pdx = t['synergy']['per_decision']
    if pdx.get('status') == 'unavailable':
        lines.append(f"Per-decision decomposition: unavailable ({pdx['reason']}).")
    else:
        lines += [f"Per-decision decomposition on the {pdx['registered_worlds']} registered decision worlds "
                  f"({pdx['convention']}).", '',
                  '| arm | decisions | worlds w/o | delta_all [CI] | best solo [CI] | cooperation gain [CI] | gain>0 | branch coll/tele |',
                  '|---|---:|---:|---|---|---|---:|---|']
        cell = lambda s: f"{_f(s['mean'])} {_ci(s['ci95'])}"  # noqa: E731
        for arm in DECISION_ARMS:
            if arm in pdx:
                r = pdx[arm]
                lines.append(f"| {arm} | {r['decisions']} | {r['worlds_without_decision']} | {cell(r['delta_all_new_veh_s'])} | "
                             f"{cell(r['best_solo_veh_s'])} | {cell(r['policy_cooperation_gain_veh_s'])} | "
                             f"{r['positive_cooperation']} | {r['branch_collision_or_teleport']} |")
        if 'B-R_matched_worlds' in pdx:
            s = pdx['B-R_matched_worlds']['policy_cooperation_gain_veh_s']
            lines.append(f"\nB-R cooperation gain on matched worlds (descriptive): {_f(s['mean'])} {_ci(s['ci95'])}, n={s['n']}.")
    act = {k: v for k, v in t['synergy'].get('decision_activity', {}).items() if isinstance(v, dict)}
    if any(arm in act for arm in DECISION_ARMS):
        lines += ['', 'Decision activity (descriptive): worlds with a cooperative move / moves / by move / median first t.', '',
                  '| arm | worlds with move | moves | by move | median first t (s) | subset captures |',
                  '|---|---:|---:|---|---:|---:|']
        for arm in DECISION_ARMS:
            if arm in act:
                a = act[arm]
                lines.append(f"| {arm} | {a['worlds_with_cooperative_move']}/{a['worlds']} | {a['cooperative_moves']} | "
                             f"{a['by_move']} | {_f(a['median_first_move_t_s'], 0)} | {a['subset_captures']} |")
    iso = t['synergy']['registered_u24_train_states']
    if 'synergy' in iso:
        lines += ['', f"clean4 registered u24 replicated test (TRAIN states): verdict **{iso['verdict']}**, "
                      f"safety_ok={iso['safety_ok']}, receipt `{iso['receipt_sha256']}`."]
    ps = t['per_seed']
    lines += ['', '## T5 Per-seed view (96 held-out seeds; diff = X - comparator, negative = benefit)', '',
              '| comparator | arm | mean [95% CI] | benefit | harm | quartiles |', '|---|---|---|---:|---:|---|']
    for c, rows in ps['comparators'].items():
        if 'status' in rows:
            lines.append(f"| {c} | - | {rows['status']} | | | |")
            continue
        for n, r in rows.items():
            lines.append(f"| {c} | {n} | {_f(r['mean_ci']['diff'])} {_ci(r['mean_ci']['ci95'])} | {r['frac_benefit']:.2f} | "
                         f"{r['frac_harm']:.2f} | {', '.join(_f(q) for q in r['quartiles'])} |")
    for k in ('B-R', 'B-A'):
        r = ps[k]
        if 'status' in r:
            lines.append(f"| {k[2:]} | B | {r['status']} | | | |")
            continue
        lines.append(f"| {k[2:]} | B | {_f(r['mean_ci']['diff'])} {_ci(r['mean_ci']['ci95'])} | {r['frac_benefit']:.2f} | "
                     f"{r['frac_harm']:.2f} | {', '.join(_f(q) for q in r['quartiles'])} |")
    tb = ps['table_B']
    cols = [k for k in tb['rows'][0] if k.startswith('B-')] if tb['rows'] else []
    lines += ['', f"Per-seed table for B (sorted by {tb['sorted_by']}):", '',
              '| seed | J_B | '+' | '.join(cols)+' |', '|---:|---:|'+'---:|'*len(cols)]
    for r in tb['rows']:
        lines.append(f"| {r['seed']} | {_f(r['J_B'])} | "+' | '.join(_f(r[c]) for c in cols)+' |')
    rg = t['regimes']
    lines += ['', '## T6 Regimes (report-only)', '']
    for k, b in rg['demand_cell'].items():
        lines.append(f"Demand cell {k}: n={b['n']}; "+'; '.join(f"{p} {_f(r['mean'])}" for p, r in b['rows'].items()
                                                                if p in ('B-R', 'B-A', 'B-NATIVE', 'B-U0')))
    ct = rg['closing_lane_speed_tercile']
    if 'terciles' in ct:
        lines += ['', f"Closing-lane speed terciles (cut {', '.join(_f(c, 2) for c in ct['cutpoints_mps'])} m/s; "
                      f"{ct['definition']}; unknown {ct['unknown']}):", '',
                  '| tercile | n | B-R | B-A | B-NATIVE | B-U0 |', '|---|---:|---:|---:|---:|---:|']
        for name, b in ct['terciles'].items():
            lines.append(f"| {name} | {b['n']} | "+' | '.join(_f((b['rows'].get(p) or {}).get('mean'))
                                                             for p in ('B-R', 'B-A', 'B-NATIVE', 'B-U0'))+' |')
    else:
        lines.append(f"Closing-lane speed terciles: {ct['status']}.")
    lines += ['', '## T4 Safety (panel totals and paired differences)', '',
              '| arm | collisions | teleports | hard brakes | TTC events | any collision/teleport |', '|---|---:|---:|---:|---:|---|']
    for n, s in t['safety']['totals'].items():
        lines.append(f"| {n} | {s['collisions']} | {s['teleports']} | {s['hard_brakes']} | {s['ttc_events']} | "
                     f"{t['safety']['any_collision_or_teleport'][n]} |")
    for comparator in ('BASE', 'NATIVE'):
        key = f'vs_{comparator}'
        if key in t['safety']:
            lines += ['', f'Paired safety differences vs {comparator} (X - {comparator}, per world):', '',
                      '| arm | counter | mean diff | 95% CI | worlds lower/higher |', '|---|---|---:|---|---|']
            for n, rows in t['safety'][key].items():
                for k, r in rows.items():
                    lines.append(f"| {n} | {k} | {_f(r['diff'], 3)} | {_ci(r['ci95'], 3)} | {r['worlds_lower']}/{r['worlds_higher']} |")
    return '\n'.join(lines)+'\n'


SECONDARY_METRICS = (
    ('fairness', 'delay_gini', lambda r, rec: r['delay']['gini']),
    ('fairness', 'delay_jain', lambda r, rec: r['delay']['jain']),
    ('fairness', 'delay_mean_s', lambda r, rec: r['delay']['mean_s']),
    ('fairness', 'worst10_delay_mean_s', lambda r, rec: r['delay']['worst10_mean_s']),
    ('fairness', 'hdv_cav_delay_ratio', lambda r, rec: r['delay_by_class']['hdv_cav_ratio']),
    ('safety', 'ttc_below_3_vs', lambda r, rec: r['ttc']['below_3_vs']),
    ('safety', 'min_ttc_s', lambda r, rec: r['ttc']['min_s']),
    ('safety', 'hard_brake_vs', lambda r, rec: r['accel']['hard_brake_vs']),
    ('safety', 'emergency_decel_vs', lambda r, rec: r['accel']['emergency_vs']),
    ('safety', 'collisions', lambda r, rec: rec['safety']['collisions']),
    ('safety', 'teleports', lambda r, rec: rec['safety']['teleports']),
    ('smoothness', 'accel_var', lambda r, rec: r['accel']['var']),
    ('smoothness', 'accel_var_per_vehicle_mean', lambda r, rec: r['accel_var_per_vehicle_mean']),
    ('smoothness', 'stop_and_go', lambda r, rec: r['stop_and_go']),
    ('throughput', 'crossings_drop', lambda r, rec: r['crossings_drop']))
SECONDARY_COMPARATORS = ('BASE', 'NATIVE', 'U0', 'RULE', 'RULE_FAIR')+COMPARATOR_ARMS
_REPORTS = {}


REPORT_VERSION = 'vehicle_log_report/4'
#: clean5b-F training convention (fair_vehicle_log.ORIG = MULTICRIT analyze.ORIG): WM lane -> origin SUMO lane label
ORIG = {2: 'L0', 1: 'L1', 0: 'L2'}


def episode_log_report(final_root, record):
    """The sealed episode's ground-truth log -> its report (file sha re-verified every time; the report is
    content-addressed: cached in-process and next to the log as vehicles_report.json, keyed by the log sha)."""
    path = Path(final_root)/'evaluation'/record.get('work_dir', record['job_id'])/VEHICLE_LOG
    if not path.is_file():
        return None
    sha = record['vehicle_log']['sha256']
    if file_sha(path) != sha:
        raise Blocked(f'{record["job_id"]}: vehicle log differs from its sealed record')
    if sha not in _REPORTS:
        cache = path.parent/'vehicles_report.json'
        cached = read(cache) if cache.is_file() else None
        if cached and cached.get('log_sha256') == sha and cached.get('version') == REPORT_VERSION:
            _REPORTS[sha] = cached['report']
        else:
            _REPORTS[sha] = vehicle_log_report(path, t0=WARMUP_S, t1=HORIZON_S)
            if not cache.exists():
                publish(cache, dict(log_sha256=sha, version=REPORT_VERSION, report=_REPORTS[sha]))
    return _REPORTS[sha]


def _group_means(per_vehicle, labels):
    """{group: mean per-vehicle delay} over the vehicles of ``per_vehicle`` that carry a label."""
    sums = {}
    for vid, (delay, _, _) in per_vehicle.items():
        for g in labels.get(vid, ()):
            n, total = sums.get(g, (0, 0.))
            sums[g] = (n+1, total+delay)
    return {g: total/n for g, (n, total) in sums.items()}


def _labels(per_vehicle):
    """Training convention (c4/clean5b 1caf5c1c fair_vehicle_log.groups): every vehicle is in its class group;
    it is in origin:L0/L1/L2 only if its origin is defined (first logged row upstream of the drop); otherwise it
    goes to the separately reported origin:undefined group (never part of the criterion; still in total delay)."""
    out = {}
    for vid, (_, origin, cav) in per_vehicle.items():
        out[vid] = ('class:cav' if cav else 'class:hdv',
                    f'origin:{origin}' if origin in ORIG.values() else 'origin:undefined')
    return out


CO_PRIMARY_GROUPS = ('origin:L0', 'origin:L1', 'origin:L2', 'class:hdv', 'class:cav')
REPORTED_GROUPS = CO_PRIMARY_GROUPS+('origin:undefined',)


def co_primary_groups(reports, worlds, bootstrap, reference='U0'):
    """Owner fairness criterion, part 2 (MAIN 04:2x + amendment 13): group labels from the REFERENCE arm (U0) on
    the same world, paired by vehicle id, with the clean5b-F training convention (origin = WM lane of the first
    logged row if upstream of the drop, else undefined; class groups hold every paired vehicle). Per group, the
    per-world mean per-vehicle delay of B minus A must average <= GROUP_DELAY_MAX_S over worlds (95% CI reported)
    for the five registered groups; origin:undefined is reported only. Vehicles of an arm absent from U0 and U0
    vehicles absent from an arm are counted separately; own-arm labels are a sensitivity only."""
    if not {'B', 'A', reference} <= set(reports):
        return dict(status='unavailable', reason=f'needs B, A and {reference} vehicle logs')

    def compare(label_of):
        groups = {}
        missing = {arm: dict(not_in_reference=0, reference_not_in_arm=0) for arm in ('B', 'A')}
        per = {g: [] for g in REPORTED_GROUPS}
        for w in worlds:
            means = {}
            ref = reports[reference][w]['per_vehicle']
            for arm in ('B', 'A'):
                pv = reports[arm][w]['per_vehicle']
                labels = label_of(arm, w)
                missing[arm]['not_in_reference'] += sum(1 for vid in pv if vid not in ref)
                missing[arm]['reference_not_in_arm'] += sum(1 for vid in ref if vid not in pv)
                means[arm] = _group_means(pv, labels)
            for g in REPORTED_GROUPS:
                if g in means['B'] and g in means['A']:
                    per[g].append((means['B'][g], means['A'][g]))
        for g, pairs in per.items():
            if len(pairs) < 2:
                groups[g] = dict(status='unavailable', n_worlds=len(pairs))
                continue
            b, a = (np.array(x, float) for x in zip(*pairs))
            stats = fe.paired_stats(b, a, bootstrap, pct=False)
            groups[g] = dict(n_worlds=len(pairs), increase_mean_s=stats['diff'], ci95=stats['ci95'],
                             passes=bool(stats['diff'] <= GROUP_DELAY_MAX_S), registered=g in CO_PRIMARY_GROUPS)
        ok = [groups[g] for g in CO_PRIMARY_GROUPS if 'passes' in groups[g]]
        return dict(groups=groups, missing=missing,
                    all_pass=len(ok) == len(CO_PRIMARY_GROUPS) and all(v['passes'] for v in ok))
    registered = compare(lambda arm, w: _labels(reports[reference][w]['per_vehicle']))
    own = compare(lambda arm, w: _labels(reports[arm][w]['per_vehicle']))
    return dict(rule=(f'every registered group (origin:L0/L1/L2, class:hdv, class:cav; labels from {reference} on the '
                      'same world by vehicle id; origin = WM lane of the first logged row if upstream of the drop) '
                      f'mean per-vehicle delay increase (B - A) <= {GROUP_DELAY_MAX_S} s (point estimate; 95% CI '
                      'reported); origin:undefined reported only'), reference=reference,
                convention='c4/clean5b 1caf5c1c tools/coopmwm/fair_vehicle_log.py per_vehicle/groups (amendment 13)',
                groups=registered['groups'], all_pass=registered['all_pass'],
                vehicles_missing=registered['missing'],
                sensitivity_own_arm_labels=dict(groups=own['groups'], all_pass=own['all_pass']))


#: amendment 19 (owner/professor, REGISTRATION 2026-11-08 00:14, pre-outcome): the worst-off 10% of vehicles by delay
TAIL_LEVEL = 0.90


def cvar90(delays):
    """Mean per-vehicle delay of the most-delayed 10% of vehicles: the ceil(0.10 n) largest delays (n >= 1)."""
    d = np.sort(np.asarray(delays, float))[::-1]
    return float(d[:max(1, math.ceil((1-TAIL_LEVEL)*len(d)-1e-9))].mean())


def p90(delays):
    return float(np.quantile(np.asarray(delays, float), TAIL_LEVEL))


def _paired_tail(reports, worlds, arm, ref, stat):
    """Per world: ``stat`` over the vehicles present in BOTH ``arm`` and ``ref`` (paired by vehicle id), each arm's
    own per-vehicle delay (the delay of the group criterion: vehicle_log_report per_vehicle[vid][0])."""
    x, y, n = [], [], []
    for w in worlds:
        a, r = reports[arm][w]['per_vehicle'], reports[ref][w]['per_vehicle']
        ids = sorted(set(a) & set(r))
        if not ids:
            return None
        x.append(stat([a[v][0] for v in ids]))
        y.append(stat([r[v][0] for v in ids]))
        n.append(len(ids))
    return np.array(x), np.array(y), n


def tail_criterion(reports, worlds, bootstrap, reference):
    """Claim 3a/3c co-criterion #3 (amendment 19): paired CVaR90(B) - CVaR90(A) per world (vehicles in both B and A);
    whole-world seed bootstrap (the registered BOOTSTRAP); PASSES iff the 95% CI LOWER bound <= 0 (B not
    significantly worse than A in the worst-off 10%)."""
    if not {'B', reference} <= set(reports):
        return dict(status='unavailable', reason=f'needs B and {reference} vehicle logs')
    got = _paired_tail(reports, worlds, 'B', reference, cvar90)
    if got is None or len(worlds) < 2:
        return dict(status='unavailable', reason='a world without paired vehicles, or fewer than 2 worlds')
    b, a, n = got
    stats = fe.paired_stats(b, a, bootstrap, pct=False)
    return dict(rule=('paired CVaR90 (mean delay of the most-delayed 10% of the vehicles present in both B and '
                      f'{reference} on the same world) B - {reference}: 95% CI lower bound <= 0'),
                reference=reference, n_worlds=len(worlds), paired_vehicles_per_world=dict(min=min(n), max=max(n)),
                diff_mean_s=stats['diff'], ci95=stats['ci95'], p_two_sided=stats['p_two_sided'],
                per_world=dict(B=b.tolist(), reference=a.tolist()), passes=bool(stats['ci95'][0] <= 0))


def tail_vs_u0(reports, worlds, bootstrap, names, reference='U0'):
    """SECONDARY (amendment 19): CVaR90 and p90 of per-vehicle delay, every arm vs U0 (paired vehicles), CI + %."""
    if reference not in reports:
        return dict(status='unavailable', reason=f'no {reference} vehicle logs')
    out = dict(convention='diff = arm - U0 per world over vehicles present in both; lower is better', reference=reference)
    for name, stat in (('cvar90_delay_s', cvar90), ('p90_delay_s', p90)):
        rows = {}
        for n in names:
            if n == reference:
                continue
            got = _paired_tail(reports, worlds, n, reference, stat)
            rows[n] = (dict(status='unavailable') if got is None or len(worlds) < 2 else
                       dict(mean_ci=fe.paired_stats(got[0], got[1], bootstrap, pct=True), per_world=got[0].tolist(),
                            per_world_reference=got[1].tolist()))
        out[name] = rows
    return out


APPLICABILITY_ARMS = ('B', 'ORACLE', 'ORACLE_WM', 'LEARNED_TRIGGER')   # 24 primary (B); 25/25b bounds; 26


def applicability_labels(final_root, cell_key):
    """The ONE sealed APPLICABILITY_*.json of this root that labels ``cell_key`` (amendment 24), else None."""
    found = []
    for path in sorted(Path(final_root).glob('APPLICABILITY_*.json')):
        value = verify(read(path), 'receipt_sha256', path.name)
        if cell_key in value['cells']:
            found.append(value)
    if len(found) > 1:
        raise Blocked(f'{cell_key}: more than one sealed applicability label file')
    return found[0] if found else None


def applicability_tables(final_root, panel, have, reports, worlds, names, bootstrap, reference='U0'):
    """Amendment 24 (owner main claim, pre-registered): on the APPLICABLE worlds of this cell (labels sealed from
    the U0 tapes before any B outcome), paired total delay X - U0 (PRIMARY for B: CI95 upper < 0) plus the
    unchanged fairness co-criteria (groups <= 10 s vs U0 labels; CVaR90 X - U0 CI95 lower <= 0) on that subset;
    reported: the applicability rate (worlds, vehicle-time), non-applicable worlds (must not be significantly
    worse: CI95 lower <= 0), all worlds; MAPPO_LANE as secondary arm. Pooled over cells: pooled_applicability()."""
    labels = applicability_labels(final_root, panel['cell']['cell_key']) if final_root is not None else None
    if labels is None:
        return dict(status='unavailable', reason='no sealed applicability labels for this cell')
    key = panel['cell']['cell_key']
    app = [w for w in worlds if labels['worlds'][w]['applicable']]
    non = [w for w in worlds if not labels['worlds'][w]['applicable']]
    out = dict(labels_receipt_sha256=labels['receipt_sha256'], bank_sha256=labels['bank']['sha256'],
               rate=labels['cells'][key], applicable_worlds=len(app), non_applicable_worlds=len(non), arms={})
    for arm in [a for a in APPLICABILITY_ARMS+('MAPPO_LANE',) if a in names and reference in names]:
        row = {}
        for subset, ws in (('applicable', app), ('non_applicable', non), ('all', worlds)):
            if len(ws) < 2:
                row[subset] = dict(status='unavailable', n_worlds=len(ws))
                continue
            x = np.array([have[arm][w]['J_veh_s'] for w in ws], float)
            y = np.array([have[reference][w]['J_veh_s'] for w in ws], float)
            total = fe.paired_stats(x, y, bootstrap)
            sub = {a: {w: reports[a][w] for w in ws} for a in (arm, reference)}
            sub['B'], sub['A'] = sub[arm], sub[reference]
            groups = co_primary_groups(dict(sub, U0=sub[reference]), ws, bootstrap)
            tail = tail_criterion(dict(B=sub[arm], A=sub[reference]), ws, bootstrap, 'A')
            row[subset] = dict(n_worlds=len(ws), total_minus_U0=total, total_lower=bool(total['ci95'][1] < 0),
                               not_worse=bool(total['ci95'][0] <= 0), groups=groups, tail_cvar90=tail)
        a = row.get('applicable', {})
        row['meets_primary'] = bool(a.get('total_lower') and (a.get('groups') or {}).get('all_pass') is True
                                    and (a.get('tail_cvar90') or {}).get('passes') is True)
        row['non_applicable_not_worse'] = (row.get('non_applicable') or {}).get('not_worse')
        row['role'] = dict(B='PRIMARY (amendment 24)', ORACLE='privileged upper bound (amendments 25/25b: WM + rule '
                           'rows), not deployable', ORACLE_WM='privileged upper bound (amendment 25b: WM rows only), '
                           'not deployable', LEARNED_TRIGGER='insurance / deployability arm (amendment 26)',
                           MAPPO_LANE='secondary')[arm]
        out['arms'][arm] = row
    return out


#: amendment 27 analysis (registered by N6 before any bank-arm outcome): BREAKDOWN worlds (secondary subset) = the
#: panel worlds whose U0 episode shows at least one stop-and-go event (v >= 5 -> v < 1, ground-truth vehicle log);
#: labelled from U0 logs only.
BREAKDOWN_RULE = 'U0 stop_and_go >= 1 (ground-truth vehicle log of the U0 episode on the same world)'


def _bank_compare(have, reports, worlds, arm, reference, bootstrap):
    """X - reference on ``worlds``: paired total delay (CI95), the five group co-criteria, CVaR90 tail."""
    if len(worlds) < 2:
        return dict(status='unavailable', n_worlds=len(worlds))
    x = np.array([have[arm][w]['J_veh_s'] for w in worlds], float)
    y = np.array([have[reference][w]['J_veh_s'] for w in worlds], float)
    total = fe.paired_stats(x, y, bootstrap)
    sub = {a: {w: reports[a][w] for w in worlds} for a in (arm, reference)}
    pair = dict(B=sub[arm], A=sub[reference])
    pair[reference] = sub[reference]
    groups = co_primary_groups(pair, worlds, bootstrap, reference=reference)
    tail = tail_criterion(dict(B=sub[arm], A=sub[reference]), worlds, bootstrap, 'A')
    return dict(n_worlds=len(worlds), total_minus_ref=total, total_lower=bool(total['ci95'][1] < 0),
                not_worse=bool(total['ci95'][0] <= 0), groups=groups, tail_cvar90=tail,
                meets=bool(total['ci95'][1] < 0 and (groups or {}).get('all_pass') is True
                           and (tail or {}).get('passes') is True))


def bank_analyse(*, final_root, panel, runtime=None):
    """Zero physics (amendments 24-27): the bank arms vs U0 (and vs MAPPO_LANE) on the p20x primary panel from the
    published waves of this root: all / applicable / non-applicable (amendment-24 labels) / breakdown worlds; total
    delay CI95, the group fairness co-criteria, CVaR90; applicability rate. Sealed BANK_TABLES.json."""
    final_root = Path(final_root)
    if SETTING != 'p20x':
        raise Blocked('bank arms: p20x only')
    plan = prerun_plan(final_root)
    if plan is None:
        raise Blocked('PRERUN_PLAN.json missing')
    plant = final_root/PRERUN_PLANT
    cfg = read(plant/'CONFIG.json')
    u0 = mappo_binding(final_root, 'U0')
    oracle = oracle_binding(final_root)
    if u0 is None or oracle is None:
        raise Blocked('bank analysis needs the U0 binding and the sealed ORACLE_BINDING')
    bound = dict(MAPPO_LANE=mappo_binding(final_root, 'MAPPO_LANE'), **{a: oracle for a in BANK_ARMS})
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    have, sources, missing = {}, {}, {}
    for arm in ('U0', 'MAPPO_LANE')+BANK_ARMS:
        if arm == 'MAPPO_LANE' and bound['MAPPO_LANE'] is None:
            missing[arm] = 'not bound'
            continue
        specs = episode_specs(arm, panel, cfg, plant, 'mappo_frozen', bound.get(arm), u0)
        records, source, covered = published(final_root, arm, specs, check_episode)
        if covered < len(specs):
            missing[arm] = f'{covered}/{len(specs)} published'
            continue
        have[arm], sources[arm] = {r['world_id']: r for r in records}, source
    if 'U0' not in have or not set(BANK_ARMS) & set(have):
        raise Blocked(f'bank analysis needs U0 and at least one complete bank arm: {missing}')
    reports = {}
    for arm in have:
        rows = {w: episode_log_report(final_root, have[arm][w]) for w in worlds}
        if any(r is None or not r['ok'] for r in rows.values()):
            raise Blocked(f'{arm}: vehicle logs missing or invalid on this host')
        reports[arm] = rows
    labels = applicability_labels(final_root, panel['cell']['cell_key'])
    if labels is None:
        raise Blocked('no sealed applicability labels for this cell (amendment 24)')
    if labels['bank']['sha256'] != oracle['bank']['file_sha256']:
        raise Blocked('applicability labels and ORACLE_BINDING bind different banks')
    app = [w for w in worlds if labels['worlds'][w]['applicable']]
    breakdown = [w for w in worlds if reports['U0'][w]['stop_and_go'] >= 1]
    subsets = dict(all=worlds, applicable=app, non_applicable=[w for w in worlds if w not in set(app)],
                   breakdown=breakdown, non_breakdown=[w for w in worlds if w not in set(breakdown)])
    arms = {}
    for arm in [a for a in BANK_ARMS if a in have]:
        row = dict(label=dict(ORACLE_LABELS)[arm], mean_J_veh_s=float(np.mean([have[arm][w]['J_veh_s'] for w in worlds])),
                   safety_totals={k: int(sum(have[arm][w]['safety'][k] for w in worlds)) for k in SAFETY})
        for reference in [r for r in ('U0', 'MAPPO_LANE') if r in have]:
            row[f'vs_{reference}'] = {name: _bank_compare(have, reports, ws, arm, reference, bootstrap)
                                       for name, ws in subsets.items()}
        receipts = [have[arm][w].get('oracle') or have[arm][w].get('learned_trigger') or {} for w in worlds]
        row['activations'] = dict(worlds_with_any=sum(1 for r in receipts if r.get('activations')),
                                  total=sum(int(r.get('activations') or 0) for r in receipts))
        arms[arm] = row
    value = seal(dict(schema=SCHEMA+'.bank_tables', amendments=['24', '25', '25b', '26', '27'],
                      panel_sha256=panel['panel_sha256'], convention='diff = X - reference; negative = X has less delay',
                      primary_criterion=('applicable worlds: X - U0 CI95 upper < 0 AND groups all_pass AND CVaR90 '
                                         'passes (amendment 24); bank arms are bounds (ORACLE, ORACLE_WM) or the '
                                         'insurance / deployability arm (LEARNED_TRIGGER), never the B primary'),
                      breakdown_rule=BREAKDOWN_RULE, oracle_binding_sha256=oracle['receipt_sha256'],
                      labels_receipt_sha256=labels['receipt_sha256'], bank_sha256=labels['bank']['sha256'],
                      applicability=dict(rate=labels['cells'][panel['cell']['cell_key']], applicable_worlds=len(app)),
                      breakdown_worlds=len(breakdown), reference_means={r: float(np.mean([have[r][w]['J_veh_s'] for w in worlds]))
                                                                      for r in ('U0', 'MAPPO_LANE') if r in have},
                      sources={k: v.get('receipt_sha256') if isinstance(v, dict) else v for k, v in sources.items()},
                      missing=missing, bootstrap=bootstrap, arms=arms), 'analysis_sha256')
    target = final_root/'BANK_TABLES.json'
    if target.exists():
        old = verify(read(target), 'analysis_sha256', target.name)
        if old['analysis_sha256'] != value['analysis_sha256']:
            publish(final_root/f'BANK_TABLES_{int(time.time())}.json', value)
        return value
    publish(target, value)
    return value


def secondary_tables(*, final_root, panel, have, names, alias, dry_run=False):
    """SECONDARY, report-only (amendment 5): every arm vs every comparator present, per metric: paired whole-world
    bootstrap CI + per-seed fractions/quartiles; per-world values per arm (the per-seed table)."""
    bootstrap = dict(panel['contract']['bootstrap'])
    if dry_run:                       # FAKE PHYSICS tables only: fewer resamples to keep the dry run fast
        bootstrap['reps'] = 200
    worlds = [r['world_id'] for r in panel['worlds']]
    reports = {}
    for n in names:
        rows = {w: episode_log_report(final_root, have[n][w]) for w in worlds}
        if any(r is None for r in rows.values()):
            return dict(status='unavailable', reason=f'{n}: ground-truth vehicle logs not present on this host')
        bad = [w for w, r in rows.items() if not r['ok']]
        if bad:
            return dict(status='unavailable', reason=f'{n}: vehicle log problems in {len(bad)} worlds',
                        problems={w: rows[w]['problems'][:3] for w in bad[:5]})
        reports[n] = rows
    out = dict(status='report-only (SECONDARY; primary outcome unchanged)', convention=(
        'diff = arm - comparator; per metric higher/lower is better as named (delay, TTC exposure, braking, '
        'variance, stop-and-go: lower; Jain, min TTC, crossings: higher)'), bootstrap=bootstrap, metrics={})
    for group, metric, get in SECONDARY_METRICS:
        per = {n: [get(reports[n][w], have[n][w]) for w in worlds] for n in names}
        entry = dict(group=group, per_world={n: v for n, v in per.items()})
        ok = {n: v for n, v in per.items() if all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)}
        if set(ok) != set(per):
            entry['undefined_in'] = sorted(set(per)-set(ok))
        cmp_ = {}
        for c in SECONDARY_COMPARATORS:
            ref = alias.get(c, c)
            if ref not in ok:
                continue
            cmp_[c] = {n: dict(mean_ci=fe.paired_stats(ok[n], ok[ref], bootstrap, pct=False),
                               **{k: v for k, v in seed_stats(np.asarray(ok[n], float)-np.asarray(ok[ref], float),
                                                              worlds).items() if k != 'sorted'})
                       for n in ok if n != ref}
        entry['vs'] = cmp_
        out['metrics'][metric] = entry
    out['co_primary_groups'] = co_primary_groups(reports, worlds, bootstrap)
    # amendment 19: A = U0 where A is not on the panel (claim 3c; A is the frozen U0 alias)
    reference = 'A' if 'A' in reports else 'U0'
    out['co_primary_tail'] = tail_criterion(reports, worlds, bootstrap, reference)
    out['tail_vs_u0'] = tail_vs_u0(reports, worlds, bootstrap, names)
    out['applicability'] = applicability_tables(final_root, panel, have, reports, worlds, names, bootstrap)
    origins = sorted({o for n in names for w in worlds for o in reports[n][w]['delay_by_origin_s']})
    out['delay_by_origin_s'] = {o: {n: [reports[n][w]['delay_by_origin_s'].get(o) for w in worlds] for n in names}
                                for o in origins}
    return json.loads(json.dumps(out, allow_nan=False))


def check_replicated(value, update):
    """clean3_finaleval.check_replicated for the evaluated update (u12 or u24 replicated test; descriptive)."""
    verify(value, 'receipt_sha256', f'replicated_u{int(update):03}')
    if value.get('update') != int(update) or value.get('kind') != 'replicated' or value.get('status') != 'complete':
        raise Blocked(f'replicated u{update} receipt is not the complete live u{update} test')
    reports = value['metric_reports']
    keep = ('mean', 'ci_lo', 'ci_hi', 'n', 'level', 'reps', 'seed')
    return dict(receipt_sha256=value['receipt_sha256'], verdict=value.get('verdict'), update=int(update),
                criterion=value.get('criterion'), safety=value.get('safety'), safety_ok=value.get('safety_ok'),
                synergy={k: {f: v.get(f) for f in keep} for k, v in reports[fe.SYNERGY]['comparisons'].items()},
                total_benefit={k: {f: v.get(f) for f in keep}
                               for k, v in reports[fe.TOTAL_BENEFIT].get('comparisons', {}).items()},
                arm_means=dict(synergy=reports[fe.SYNERGY].get('arm_means'),
                               total_benefit=reports[fe.TOTAL_BENEFIT].get('arm_means')))


def tables(*, root, final_root, panel, base, episodes, decisions, sources, missing, dry_run=False, comparators=None,
           scope='claim3a', stem='CLAIM3A_TABLES', update=UPDATE, a_alias=False):
    sources, missing = dict(sources), dict(missing)
    replicated = None
    path = Path(root)/replicated_file(update)
    key = f'replicated_u{int(update):03}'
    if path.is_file():
        replicated = check_replicated(read(path), update)
        sources[key] = dict(source=str(path), receipt_sha256=replicated['receipt_sha256'])
    else:
        missing[key] = 'clean run monitor receipt absent (descriptive context only)'
    if a_alias:
        sources['A'] = dict(alias_of='U0', rule=FROZEN_A_RULE, label=A_ALIAS_LABEL)
    value = analyse(panel=panel, base=base, episodes=episodes, decisions=decisions, replicated=replicated,
                    sources=sources, missing=missing, comparators=comparators, scope=scope, update=update,
                    a_alias=a_alias, final_root=final_root, dry_run=dry_run)
    if dry_run:
        value = seal(dict(value, dry_run=True), 'analysis_sha256')
    existing = Path(final_root)/f'{stem}.json'
    if existing.is_file() and read(existing).get('analysis_sha256') != value['analysis_sha256']:
        stem = f'{stem}_'+value['analysis_sha256'][:12]   # never overwrite a published table
    if not (Path(final_root)/f'{stem}.json').is_file():
        publish(Path(final_root)/f'{stem}.json', value)
    guard(Path(final_root)/f'{stem}.md').write_text(render(value), encoding='utf-8')
    return value


# ---------------------------------------------------------------- orchestration

class Status:
    def __init__(self, final_root):
        self.final_root = Path(final_root)
        self.serial = max([int(p.name.split('_')[1]) for p in self.final_root.glob('STATUS_[0-9][0-9][0-9]_*.json')],
                          default=0)

    def __call__(self, step, state, detail=None):
        self.serial += 1
        value = dict(schema=SCHEMA+'.status', step=step, status=state, sequence=self.serial, detail=detail,
                     unix=time.time(), timestamp=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
        publish(self.final_root/f'STATUS_{self.serial:03}_{step}_{state}.json', value)
        publish(self.final_root/'STATUS.json', value, replace=True)


def run_plan(final_root):
    path = Path(final_root)/'RUN_PLAN.json'
    return verify(read(path), 'receipt_sha256', 'RUN_PLAN seal') if path.is_file() else None


def preflight(root, final_root, runtime, panel, map_module=None):
    """Zero physics: everything that must hold before the gate is armed."""
    cfg = runtime.config(root)
    plan = run_plan(final_root) if Path(final_root).is_dir() else None
    binding = clean4_binding(cfg, panel, declared_base=(plan or {}).get('binding', {}).get('base'))
    if plan is not None and (plan['binding']['panel_sha256'] != panel['panel_sha256']
                             or plan['binding']['base'] != binding['base'] or plan['clean4_root'] != str(root)):
        raise Blocked('RUN_PLAN binds a different panel, base controller or clean4 root')
    exposure = exposure_check(root, cfg)
    budget = budget_check(root, cfg, binding['base'], final_root=final_root)
    maps = map_check(bind_map_api(map_module), panel)
    runner = runtime.runner_status()
    if runtime.physics and not os.environ.get('COOPMWM_MAP_SPEC'):
        raise Blocked('set COOPMWM_MAP_SPEC to the staged wz3to2 MAP_SPEC.json (N1 process map) before physics')
    if runtime.physics and (maps['stub'] or not maps['ok']):
        raise Blocked(f'map API not qualified for physics: {maps}')
    if runtime.physics and not runner['ok']:
        raise Blocked(f'runner unavailable: {runner["reason"]}')
    return cfg, binding, dict(binding=binding, exposure=exposure, budget=budget, map_api=maps, runner=runner)


def alias_of(base):
    return dict(native='NATIVE', mappo_u0='U0', mappo_frozen='A')[base]


def _settle(final_root, arms, status, step, *, poll, timeout, sleep):
    """F12: wait until every named executor arm is bound or explicitly declared unavailable."""
    def settled():
        state = comparators(final_root, arms)
        if state['pending']:
            raise NotReady(f'{step}: results stay sealed until bound or declared unavailable: '
                           'pending '+', '.join(state['pending']))
        return state
    return fe.wait(step, settled, status, poll=poll, timeout=timeout, sleep=sleep)


def execute(*, root, final_root, runtime, panel, panel_3c=None, allow_missing=(), workers=24, poll=60,
            gate_timeout=7*86400, replicated_timeout=0, comparator_timeout=7*86400, map_module=None, dry_run=False,
            sleep=time.sleep, update=PRIMARY_UPDATE):
    """One evaluation update (u12 PRIMARY, u24 SECONDARY): CONFIG -> GATE (update sealed, safety clean, B/R (A)
    bound by sha; frozen A == u0) -> U0 bound/declared -> claim 3a + 3c episodes (B, R at the update; A = U0 alias
    for the frozen control arm; U0/RULE/NATIVE shared, reused from the sealed pre-run) -> sealed CLAIM3A/3C tables
    of the update -> for each MAPPO comparator: bound/declared -> F11 parity -> its (shared) wave -> its own sealed
    CLAIM3B tables of the update. Returns {analysis: value}."""
    root, final_root, update = Path(root), Path(final_root), check_update(update)
    sfx = f'_u{update:03}'
    if run_plan(final_root) is None:
        raise Blocked('RUN_PLAN.json missing: run register first')
    status = Status(final_root)
    current = f'CONFIG{sfx}'
    try:
        status(current, 'RUNNING', dict(update=update))
        cfg, binding, report = preflight(root, final_root, runtime, panel, map_module)
        base, missing = binding['base'], {}
        a_alias = a_frozen(cfg)
        arms = [a for a in ARMS if not (a == 'A' and a_alias)]
        if report['runner']['native']:
            arms.append('NATIVE')
        else:
            missing['NATIVE'] = 'runner has no native episode'
        do_decisions = report['runner']['decisions']
        for name in list(missing):
            if name not in allow_missing:
                raise Blocked(f'{name}: {missing[name]} (pass --allow-missing {name} to proceed without it)')
        if not do_decisions:   # never blocks the evaluation (MAIN): tabulated as unavailable
            missing['DECISIONS'] = DECISIONS_UNAVAILABLE
        status(current, 'PASS', dict(report, update=update, arms=arms, a_alias_of_u0=a_alias, missing=missing,
                                     comparators=comparators(final_root)))
        current = f'GATE{sfx}'
        gated = fe.wait(current, lambda: gate(root, cfg, runtime, binding, final_root, update), status, poll=poll,
                        timeout=gate_timeout, sleep=sleep)
        if not (final_root/f'GATE_PASS{sfx}.json').is_file():
            publish(final_root/f'GATE_PASS{sfx}.json', seal(gated, 'receipt_sha256'))
        status(current, 'PASS', gated)
        kw = dict(root=root, final_root=final_root, cfg=cfg, runtime=runtime, workers=workers, base=base)
        # ---------------------------------------------------------------- claim 3a
        current = 'U0'
        u0 = _settle(final_root, ('U0',), status, current, poll=poll, timeout=comparator_timeout, sleep=sleep)
        bindings, compat = dict(u0['bound']), None
        if 'U0' in bindings:
            arms.append('U0')
            if cfg.get('common_checkpoint_sha256') not in (None, bindings['U0']['residual_sha256']):
                raise Blocked('the run common u0 differs from the bound zero-residual u0 (pre-run U0/RULE invalid)')
            for rarm, key, why in (('RULE', 'rule', 'S-REGIMEU0 rule (tools.coopmwm.regimeu0)'),
                                   ('RULE_FAIR', 'rule_fair', 'S-MULTICRIT rule (tools.coopmwm.regimeu0_multicrit)')):
                if report['runner'].get(key):
                    arms.append(rarm)
                elif rarm in allow_missing:
                    missing[rarm] = f'runner has no {why}'
                else:
                    raise Blocked(f'{rarm}: runner has no {why} (--allow-missing {rarm})')
            compat = prerun_compat(final_root, cfg, runtime)
            oracle = oracle_binding(final_root) if SETTING == 'p20x' else None
            for oarm in BANK_ARMS if SETTING == 'p20x' else ():
                flag = 'oracle' if oarm in ORACLE_ARMS else 'learned'
                if oracle is None:
                    missing[oarm] = f'not bound: no sealed {ORACLE_BINDING} (bound after training ends)'
                elif report['runner'].get(flag):
                    arms.append(oarm)
                elif oarm in allow_missing:
                    missing[oarm] = f'runner has no {oarm} arm'
                else:
                    raise Blocked(f'{oarm}: bound but the runner has no {oarm} arm (--allow-missing {oarm})')
        elif a_alias:
            raise Blocked('A is the frozen U0 alias but U0 was declared unavailable')
        else:
            missing['U0'] = f'declared unavailable: {u0["declared"]["U0"]["reason"]}'
            missing['RULE'] = missing['RULE_FAIR'] = 'needs the U0 executor (declared unavailable)'
        status(current, 'PASS', dict(bound=sorted(bindings), declared=sorted(u0['declared']),
                                     prerun=compat if 'U0' in bindings else None))
        current = f'EPISODES{sfx}'

        def specs_for(arm, pnl):
            bound = oracle_binding(final_root) if arm in BANK_ARMS else bindings.get(arm)
            return episode_specs(arm, pnl, cfg, root, base, bound, bindings.get('U0'), update)
        specs = {arm: specs_for(arm, panel) for arm in arms}
        if any(specs[arm][0]['checkpoint']['file_sha256'] != gated['checkpoint_sha256'][arm] for arm in ARMS if arm in arms):
            raise Blocked(f'u{update} checkpoint changed after the gate')
        arms_3c = [a for a in CLAIM3C_ARMS if a in arms] if panel_3c is not None else []
        specs_3c = {a: specs_for(a, panel_3c) for a in arms_3c}
        status(current, 'RUNNING', dict(update=update, arms=arms, claim3c_arms=arms_3c, workers=workers))
        # claim 3c (secondary claim, separate seal) runs in the same dispatch: it opens with 3a
        done, sources = run_waves([(wave_name(arm, update), 'episode', specs[arm]) for arm in arms]
                                  + [(wave_name(a, update, 'C3_'), 'episode', specs_3c[a]) for a in arms_3c], **kw)
        episodes = {arm: {r['world_id']: r for r in done[wave_name(arm, update)]} for arm in arms}
        episodes_3c = {a: {r['world_id']: r for r in done[wave_name(a, update, 'C3_')]} for a in arms_3c}
        if a_alias:                      # A = U0 (frozen control arm; identical weights verified at the gate)
            episodes['A'] = episodes['U0']
            if 'U0' in episodes_3c and 'A' in CLAIM3C_ARMS:
                episodes_3c['A'] = episodes_3c['U0']
        status(current, 'PASS', dict(update=update, arms=sorted(episodes), claim3c_arms=sorted(episodes_3c)))
        decisions = None
        if do_decisions:
            current = f'DECISIONS{sfx}'
            dspecs = {arm: decision_specs(arm, episodes[arm], panel) for arm in DECISION_ARMS}
            status(current, 'RUNNING', {arm: len(v) for arm, v in dspecs.items()})
            done, more = run_waves([(f'DECISIONS_{arm}{sfx}', 'branch', dspecs[arm]) for arm in DECISION_ARMS], **kw)
            sources.update(more)
            decisions = {arm: done[f'DECISIONS_{arm}{sfx}'] for arm in DECISION_ARMS}
            status(current, 'PASS', {arm: len(v) for arm, v in decisions.items()})
        if alias_of(base) in missing:
            missing['BASE'] = f'BASE is {alias_of(base)}: {missing[alias_of(base)]}'

        def replicated_receipt():
            if not (root/replicated_file(update)).is_file():
                raise NotReady(f'waiting for the clean run replicated u{update} receipt (descriptive context)')
        if replicated_timeout > 0:
            try:
                fe.wait('MONITOR_RECEIPTS', replicated_receipt, status, poll=poll, timeout=replicated_timeout, sleep=sleep)
            except NotReady:
                pass   # tabulated as missing; ``analyse`` re-tabulates later with zero physics
        current = f'TABLES_3A{sfx}'
        common = dict(root=root, final_root=final_root, base=base, dry_run=dry_run, update=update, a_alias=a_alias)
        out = {}
        out['claim3a'] = tables(panel=panel, episodes=episodes, decisions=decisions, sources=sources,
                                missing=dict(missing, **{a: 'claim 3b (separate seal)' for a in COMPARATOR_ARMS}),
                                comparators=bindings, scope='claim3a', stem=f'CLAIM3A_TABLES{sfx}', **common)
        status(current, 'PASS', dict(analysis_sha256=out['claim3a']['analysis_sha256'], present=out['claim3a']['present']))
        if panel_3c is not None:
            current = f'TABLES_3C{sfx}'
            out['claim3c'] = tables(panel=panel_3c, episodes=episodes_3c, decisions=None,
                                    sources={k: v for k, v in sources.items() if k.startswith('C3_')},
                                    missing={a: missing[a] for a in CLAIM3C_ARMS if a in missing},
                                    comparators={a: b for a, b in bindings.items() if a == 'U0'}, scope='claim3c',
                                    stem=f'CLAIM3C_TABLES{sfx}', **common)
            status(current, 'PASS', dict(analysis_sha256=out['claim3c']['analysis_sha256'], present=out['claim3c']['present']))
        # ---------------------------------------------------------------- claim 3b, one comparator at a time
        for arm in COMPARATOR_ARMS:
            current = f'COMPARATOR_{arm}{sfx}'
            state = _settle(final_root, (arm,), status, current, poll=poll, timeout=comparator_timeout, sleep=sleep)
            if arm in state['declared']:
                d = state['declared'][arm]
                note = seal(dict(schema=SCHEMA+'.claim3b', scope=f'claim3b_{arm}', status='declared unavailable',
                                 declaration=d, panel_sha256=panel['panel_sha256']), 'analysis_sha256')
                if not (final_root/f'CLAIM3B_{arm}_UNAVAILABLE.json').is_file():
                    publish(final_root/f'CLAIM3B_{arm}_UNAVAILABLE.json', note)
                out[f'claim3b_{arm}'] = note
                status(current, 'PASS', dict(declared=d['declaration_sha256']))
                continue
            bound = state['bound'][arm]
            if arm == 'MAPPO_LANE' and not report['runner'].get('lane'):
                if arm not in allow_missing:
                    raise Blocked('MAPPO_LANE bound but the runner has no lane episode (--allow-missing MAPPO_LANE)')
                status(current, 'SKIPPED', dict(reason='runner has no lane-capable MAPPO episode'))
                continue
            parity_note = None
            if bound.get('train_src'):
                parity_note = ('F11 identical by construction: the comparator runs its own training execution path '
                               '(amendment 17)')
            elif report['runner'].get('parity') and bound.get('train_config'):
                parity, more = run_parity(root=root, final_root=final_root, runtime=runtime, panel=panel, arm=arm,
                                          cfg=cfg, base=base, workers=workers)
                sources.update(more)
                sources[f'PARITY_{arm}_REPORT'] = dict(source=str(final_root/f'PARITY_{arm}.json'),
                                                       receipt_sha256=parity['receipt_sha256'], verdict=parity['verdict'])
            else:
                parity_note = ('F11 parity unavailable: '+('runner has no parity()' if not report['runner'].get('parity')
                                                            else 'comparator bound without --train-config'))
            done, more = run_waves([(arm, 'episode', episode_specs(arm, panel, cfg, root, base, bound,
                                                                     bindings.get('U0')))], **kw)
            sources.update(more)
            both = dict(episodes, **{arm: {r['world_id']: r for r in done[arm]}})
            out[f'claim3b_{arm}'] = tables(
                panel=panel, episodes=both, decisions=decisions, sources=sources,
                missing=dict(missing, **{a: 'claim 3b (separate seal)' for a in COMPARATOR_ARMS if a != arm},
                             **({f'PARITY_{arm}': parity_note} if parity_note else {})),
                comparators=dict(bindings, **{arm: bound}), scope=f'claim3b_{arm}', stem=f'CLAIM3B_{arm}_TABLES{sfx}',
                **common)
            if panel_3c is not None and arm in PRERUN_ARMS:      # its sealed 3c pre-run wave (amendment 17)
                done3, more3 = run_waves([(f'C3_{arm}', 'episode', episode_specs(arm, panel_3c, cfg, root, base, bound,
                                                                                   bindings.get('U0')))], **kw)
                sources.update(more3)
                out[f'claim3b_{arm}_3c'] = tables(
                    panel=panel_3c, episodes=dict(episodes_3c, **{arm: {r['world_id']: r for r in done3[f'C3_{arm}']}}),
                    decisions=None, sources={k: v for k, v in sources.items() if k.startswith('C3_')},
                    missing={a: missing[a] for a in CLAIM3C_ARMS if a in missing},
                    comparators={a: b for a, b in dict(bindings, **{arm: bound}).items() if a in ('U0', arm)},
                    scope=f'claim3b_{arm}_3c', stem=f'CLAIM3B_{arm}_3C_TABLES{sfx}', **common)
            status(current, 'PASS', dict(analysis_sha256=out[f'claim3b_{arm}']['analysis_sha256']))
        current = f'DONE{sfx}'
        status(current, 'PASS', {k: v['analysis_sha256'] for k, v in out.items()})
        return out
    except BaseException as error:
        status(current, 'FAILED', dict(error=repr(error), traceback=traceback.format_exc()))
        raise


# ---------------------------------------------------------------- pre-run (NATIVE, U0, RULE; sealed)

def prerun_plan(final_root):
    path = Path(final_root)/PRERUN_FILE
    return verify(read(path), 'receipt_sha256', 'PRERUN_PLAN seal') if path.is_file() else None


def prerun_cap():
    return len(PRERUN_ARMS)*(N_WORLDS+CLAIM3C_WORLDS)*HORIZON_S


PRERUN_CAP_FILE = 'PRERUN_CAP.json'
PRERUN_CAP_MAX = 800_000                     # MAIN 2026-11-07: ... -> 660,000 (RULE_FAIR) -> 800,000 (MAPPO_LANE, amend 17)


def _cap_files(final_root):
    return sorted(Path(final_root).glob('PRERUN_CAP*.json'))


def effective_prerun_cap(final_root, plan):
    cap = plan['cap_sim_s']
    for path in _cap_files(final_root):
        value = verify(read(path), 'receipt_sha256', path.name)
        if value['prerun_plan_sha256'] != plan['receipt_sha256'] or not plan['cap_sim_s'] <= value['cap_sim_s'] <= PRERUN_CAP_MAX:
            raise Blocked(f'{path.name} does not amend this pre-run plan within the registered maximum')
        cap = max(cap, value['cap_sim_s'])
    return cap


def amend_prerun_cap(*, final_root, cap_sim_s, authority):
    """MAIN cap decision for the pre-run sub-cap (sealed, in sequence; only raises, within the registered max)."""
    plan = prerun_plan(final_root)
    if plan is None:
        raise Blocked('PRERUN_PLAN.json missing')
    current = effective_prerun_cap(final_root, plan)
    if not str(authority).strip() or not current < int(cap_sim_s) <= PRERUN_CAP_MAX:
        raise Blocked(f'authority and a cap in ({current}, {PRERUN_CAP_MAX}] required')
    n = len(_cap_files(final_root))
    target = Path(final_root)/(PRERUN_CAP_FILE if n == 0 else f'PRERUN_CAP_{n+1:02}.json')
    value = seal(dict(schema=SCHEMA+'.prerun_cap', prerun_plan_sha256=plan['receipt_sha256'], cap_sim_s=int(cap_sim_s),
                      previous=current, authority=str(authority), utc=datetime.now(timezone.utc).isoformat()),
                 'receipt_sha256')
    publish(target, value)
    return value


def _published_jobs(final_root):
    jobs = set()
    for path in Path(final_root).glob('WAVE_*.json'):
        jobs |= {r['job_id'] for r in read(path).get('records', [])}
    return jobs


def _rounds_used(final_root, clean_root=None):
    """{base job id: highest charged round} (0 = the original charge, n = '<job>#rn')."""
    used = {}
    rows = fe.ledger_rows(final_root)+(fe.ledger_rows(clean_root) if clean_root else [])
    for r in rows:
        base, _, n = str(r['id']).partition('#r')
        used[base] = max(used.get(base, -1), int(n) if n else 0)
    return used


def replay_rounds(final_root):
    """{base job id: number of MAIN replay authorisations}."""
    rounds = {}
    for path in sorted(Path(final_root).glob('REPLAY_AUTHORIZED_*.json')):
        for job in verify(read(path), 'receipt_sha256', path.name)['jobs']:
            rounds[job] = rounds.get(job, 0)+1
    return rounds


def replay_authorized(final_root):
    return set(replay_rounds(final_root))


def interrupted_jobs(final_root):
    """Charged jobs (any round) with no published record and no unused authorisation: they need MAIN."""
    used, rounds, done = _rounds_used(final_root), replay_rounds(final_root), _published_jobs(final_root)
    return sorted(j for j, n in used.items() if j not in done and rounds.get(j, 0) <= n and not j.startswith('nmlogcheck_'))


def interrupted_parts(final_root):
    jobs = interrupted_jobs(final_root)
    return {'interrupted': jobs} if jobs else {}


def authorize_replay(*, final_root, authority, reason):
    """MAIN-authorised replay round of every interrupted job (charged, nothing published or opened): the next
    round runs in a FRESH job dir <job>__r<n> (old dirs untouched) and is charged as <job>#r<n>. Sealed."""
    final_root = Path(final_root)
    jobs = interrupted_jobs(final_root)
    if not jobs:
        raise Blocked('no interrupted part to replay')
    if not str(authority).strip() or not str(reason).strip():
        raise Blocked('authority and reason required')
    used = _rounds_used(final_root)
    n = len(list(final_root.glob('REPLAY_AUTHORIZED_*.json')))+1
    value = seal(dict(schema=SCHEMA+'.replay', jobs=jobs, rounds={j: used[j]+1 for j in jobs}, authority=str(authority),
                      reason=str(reason), results_opened=False, work_dirs='<job>__r<round> (fresh)',
                      utc=datetime.now(timezone.utc).isoformat()), 'receipt_sha256')
    publish(final_root/f'REPLAY_AUTHORIZED_{n:02}.json', value)
    return value


class PrerunRuntime:
    """The same runner/plant (real or fake) with the evaluation root's OWN ledger (phase
    v2v_final_prerun, cap = PRERUN_ARMS x (96 + 24) x 1200 s); the CONFIG is the registered plant copy."""

    def __init__(self, inner, final_root):
        self.inner, self.final_root = inner, Path(final_root)
        self.physics = inner.physics
        self.calls = getattr(inner, 'calls', None)

    def config(self, root):
        plan = prerun_plan(self.final_root)
        if plan is None:
            raise Blocked('PRERUN_PLAN.json missing: prerun-register first')
        if Path(root) != self.final_root/PRERUN_PLANT:
            raise Blocked('the pre-run plant root is <final_root>/prerun_plant')
        cfg = read(Path(root)/'CONFIG.json')
        if digest(cfg) != plan['plant']['config_sha256'] or self.inner.plant_view(cfg) != plan['plant']['view']:
            raise Blocked('pre-run plant CONFIG differs from PRERUN_PLAN')
        return cfg

    def health(self, root):
        return None

    def runner_status(self):
        return self.inner.runner_status()

    def plant_view(self, cfg):
        return self.inner.plant_view(cfg)

    def check_capacity(self, amount):
        """Whole-dispatch cap check BEFORE the canary is charged (a cap refusal never strands a canary)."""
        plan = prerun_plan(self.final_root)
        rows = fe.ledger_rows(self.final_root)
        cap = effective_prerun_cap(self.final_root, plan)
        if sum(r['sim_s'] for r in rows if r.get('phase') == PRERUN_PHASE)+amount > cap:
            raise Blocked(f'pre-run: {PRERUN_PHASE} cap {cap} would be exceeded')
        if sum(r['sim_s'] for r in rows)+amount > ALLOCATION_SIM_S:
            raise Blocked(f'pre-run: {PHASE} allocation {ALLOCATION_SIM_S} would be exceeded')

    def reserve(self, cfg, jobs):
        plan = prerun_plan(self.final_root)
        try:
            own_reserve(self.final_root, jobs, PRERUN_PHASE, cap=effective_prerun_cap(self.final_root, plan),
                        calls=self.calls)
        except Blocked as error:
            raise Blocked(f'pre-run: {error}') from None

    def run(self, items, cfg, workers):
        return self.inner.run(items, cfg, workers)


class MappoCtRuntime(PrerunRuntime):
    """Amendment 30: MAPPO_CT on the p20c bank root, own phase CT_PHASE, cap CT_CAP_SIM_S; the root allocation is
    raised by exactly that cap (the mappo_ct line, not the amendment-28 allocation)."""
    phase = CT_PHASE

    def check_capacity(self, amount):
        rows = fe.ledger_rows(self.final_root)
        if sum(r['sim_s'] for r in rows if r.get('phase') == CT_PHASE)+amount > CT_CAP_SIM_S:
            raise Blocked(f'MAPPO_CT: {CT_PHASE} cap {CT_CAP_SIM_S} would be exceeded')
        if sum(r['sim_s'] for r in rows)+amount > ALLOCATION_SIM_S+CT_CAP_SIM_S:
            raise Blocked(f'MAPPO_CT: root allocation {ALLOCATION_SIM_S}+{CT_CAP_SIM_S} would be exceeded')

    def reserve(self, cfg, jobs):
        try:
            own_reserve(self.final_root, jobs, CT_PHASE, cap=CT_CAP_SIM_S, calls=self.calls,
                        allocation=ALLOCATION_SIM_S+CT_CAP_SIM_S)
        except Blocked as error:
            raise Blocked(f'MAPPO_CT: {error}') from None


class BankRuntime(PrerunRuntime):
    """Amendment 27: the bank arms (ORACLE, ORACLE_WM, LEARNED_TRIGGER) on the pre-run plant of this root, charged to
    its own ledger under phase v2v_final_bank, cap BANK_CAP_SIM_S (the allocation bound still applies)."""
    phase = BANK_PHASE

    def check_capacity(self, amount):
        rows = fe.ledger_rows(self.final_root)
        if sum(r['sim_s'] for r in rows if r.get('phase') == BANK_PHASE)+amount > BANK_CAP_SIM_S:
            raise Blocked(f'bank arms: {BANK_PHASE} cap {BANK_CAP_SIM_S} would be exceeded')
        if sum(r['sim_s'] for r in rows)+amount > ALLOCATION_SIM_S:
            raise Blocked(f'bank arms: {PHASE} allocation {ALLOCATION_SIM_S} would be exceeded')

    def reserve(self, cfg, jobs):
        try:
            own_reserve(self.final_root, jobs, BANK_PHASE, cap=BANK_CAP_SIM_S, calls=self.calls)
        except Blocked as error:
            raise Blocked(f'bank arms: {error}') from None


def bind_oracle(*, final_root, bank_path, model_path, authority, banks=None):
    """Zero physics (amendment 27): seal ORACLE_BINDING.json once - the common bank (ORACLE_WM, LEARNED_TRIGGER and
    the applicability labels), per-arm bank overrides (ORACLE: the union bank), the LEARNED_TRIGGER model doc and
    the registered oracle_plan digest. Every file is hashed here; never overwritten."""
    final_root = Path(final_root)
    if SETTING not in BANK_SETTINGS:
        raise Blocked('the bank arms are registered for the p20x and p20c roots only')
    if (final_root/ORACLE_BINDING).exists():
        raise Blocked(f'{ORACLE_BINDING} exists; never overwrite')
    if not str(authority or '').strip():
        raise Blocked('authority required')

    def entry(path):
        path = Path(path).resolve()
        return dict(path=str(path), file_sha256=file_sha(path))
    model = entry(model_path)
    if model['file_sha256'] != LEARNED_MODEL_SHA256:
        raise Blocked(f'{model_path} is not the registered LEARNED_TRIGGER model {LEARNED_MODEL_SHA256[:8]}')
    value = seal(dict(schema=SCHEMA+'.oracle_binding', created=datetime.now(timezone.utc).isoformat(),
                      authority=str(authority), bank=entry(bank_path),
                      banks={arm: entry(path) for arm, path in (banks or {}).items()},
                      learned_trigger_model=model, oracle_sha256=ORACLE_SHA256,
                      learned_trigger_sha256=LEARNED_TRIGGER_SHA256), 'receipt_sha256')
    check_oracle_binding(value)
    publish(final_root/ORACLE_BINDING, value)
    return value


def prerun_register(*, final_root, plant_config, u0_path, u0_sha256, runtime, panel, panel_3c, authority):
    """Zero physics: a NEW evaluation root with the sealed PRERUN_PLAN, a byte-for-byte content copy of the
    clean run CONFIG (unchanged: run_root/accounting_root stay the clean root, which passes the production
    check_pins; the plant only READS it - runtime, executor, loaders) and the bound common u0 copy
    (A_checkpoint_000.pt, sha re-verified). Charges go to the evaluation root's own ledger."""
    import shutil
    final_root = guard(Path(final_root))
    if final_root.exists():
        raise Blocked('final root exists; never overwrite')
    if not str(authority).strip():
        raise Blocked('authority required')
    source = read(plant_config)
    if source.get('common_checkpoint_sha256') not in (None, u0_sha256):
        raise Blocked('plant CONFIG common u0 differs from the bound u0 sha')
    if file_sha(u0_path) != u0_sha256:
        raise Blocked(f'{u0_path} does not hash to {u0_sha256}')
    runtime.plant_view(source)                      # zero-physics refusal leaves no partial final root
    plant = final_root/PRERUN_PLANT
    plant.mkdir(parents=True)
    shutil.copyfile(u0_path, plant/'A_checkpoint_000.pt')
    if file_sha(plant/'A_checkpoint_000.pt') != u0_sha256:
        raise Blocked('u0 copy does not hash to the bound sha')
    cfg = deepcopy(source)
    view = runtime.plant_view(cfg)                  # the production loader validates it BEFORE any file is written
    publish(plant/'CONFIG.json', cfg)
    plan = seal(dict(schema=SCHEMA+'.prerun', created=datetime.now(timezone.utc).isoformat(), authority=str(authority),
                     registration=PRERUN_AUTHORITY, arms=list(PRERUN_ARMS), phase=PRERUN_PHASE, cap_sim_s=prerun_cap(),
                     **({} if SETTING == 'p60' else dict(setting=SETTING, cells=dict(claim3a=CELL, claim3c=CLAIM3C_CELL),
                                                         mappo_lane_note=MAPPO_LANE_OOD)),
                     debits=f'{PHASE} (counted by this launcher against {ALLOCATION_SIM_S}; ledger = this root)',
                     panel_sha256=panel['panel_sha256'], panel_3c_sha256=panel_3c and panel_3c['panel_sha256'],
                     plant=dict(source=str(plant_config), source_file_sha256=file_sha(plant_config),
                                source_sha256=digest(source), config_sha256=digest(cfg), view=view, root=str(plant)),
                     u0=dict(source=str(u0_path), sha256=u0_sha256, path=str(plant/'A_checkpoint_000.pt')),
                     rule=dict(strategy=RULE_STRATEGY, rule_sha256=RULE_SHA256,
                               **({} if SETTING == 'p60' else dict(module=RULE_MODULE, source=RULE_SOURCE))),
                     sealed=('F12: pre-run WAVE receipts are not opened, tabulated or summarised by anyone until clean5a '
                             'A/B/R are bound and the claim-3a seal opens; this launcher publishes no pre-run table'),
                     reuse=('the main run reuses each pre-run wave unchanged when the clean5a plant view and common u0 '
                            'equal this plan (prerun_compat); an interrupted pre-run wave is never replayed')),
                'receipt_sha256')
    publish(final_root/PRERUN_FILE, plan)
    return plan


def prerun_execute(*, final_root, runtime, panel, panel_3c, workers=16, arms=None, chunk=None):
    """Pre-run NATIVE, U0 and RULE on panel v2 (3a) and the 3c panel: waves only, no analysis. Dispatched
    in chunks of ``chunk`` episodes (default 2 x workers) published as WAVE_<wave>__<a>-<b>.json; the
    ``STOP_PRERUN`` file stops cleanly after the chunk in flight; a later call (any host with the same
    absolute paths) resumes at the first unpublished episode. Returns receipt shas and counts only
    (results stay sealed, F12)."""
    final_root = Path(final_root)
    arms = PRERUN_ARMS if arms is None else tuple(arms)      # the setting's pre-run arms (amendment 28: U0 only)
    plan = prerun_plan(final_root)
    if plan is None:
        raise Blocked('PRERUN_PLAN.json missing: prerun-register first')
    if (plan['panel_sha256'], plan['panel_3c_sha256']) != (panel['panel_sha256'], panel_3c and panel_3c['panel_sha256']):
        raise Blocked('PRERUN_PLAN binds other panels')
    ct = tuple(arms) == (CT_ARM,)
    if CT_ARM in arms and (not ct or SETTING != 'p20c'):
        raise Blocked('MAPPO_CT runs alone, on the p20c bank root only (amendment 30)')
    bank = set(arms) <= set(BANK_ARMS)
    if not (set(arms) <= set(PRERUN_ARMS) or bank or ct):
        raise Blocked(f'pre-run arms are {PRERUN_ARMS}; bank-run arms are {BANK_ARMS}')
    if bank and SETTING == 'p20c' and not set(arms) <= set(CONFIRM_BANK_ARMS):
        raise Blocked(f'amendment 28 bank arms are {CONFIRM_BANK_ARMS} only')
    if ct:
        pre = runtime if isinstance(runtime, MappoCtRuntime) else MappoCtRuntime(getattr(runtime, 'inner', runtime),
                                                                                  final_root)
        if oracle_binding(final_root) is None or mappo_binding(final_root, 'U0') is None:
            raise Blocked('MAPPO_CT runs on the sealed amendment-28 bank root (U0 + ORACLE binding present)')
    elif bank:
        pre = runtime if isinstance(runtime, BankRuntime) else BankRuntime(getattr(runtime, 'inner', runtime), final_root)
    else:
        pre = runtime if isinstance(runtime, PrerunRuntime) else PrerunRuntime(runtime, final_root)
    check_ = (final_root/'LOGCHECK.json')
    if not check_.is_file() or verify(read(check_), 'receipt_sha256', 'LOGCHECK')['verdict'] != 'PASS':
        raise Blocked('run prerun-logcheck first: a PASS of the one-episode vehicle-log check is required')
    status = Status(final_root)
    current = 'PRERUN'
    try:
        status(current, 'RUNNING', dict(arms=list(arms), workers=workers))
        plant = final_root/PRERUN_PLANT
        cfg = pre.config(plant)
        runner = pre.runner_status()
        if pre.physics and not os.environ.get('COOPMWM_MAP_SPEC'):
            raise Blocked('set COOPMWM_MAP_SPEC to the staged wz3to2 MAP_SPEC.json (N1 process map) before physics')
        if not runner['ok'] or ('NATIVE' in arms and not runner['native']) or ('RULE' in arms and not runner.get('rule')) \
                or ('RULE_FAIR' in arms and not runner.get('rule_fair')) or ('MAPPO_LANE' in arms and not runner.get('lane')) \
                or (set(ORACLE_ARMS) & set(arms) and not runner.get('oracle')) \
                or (set(LEARNED_ARMS) & set(arms) and not runner.get('learned')):
            raise Blocked(f'runner cannot run the pre-run arms: {runner}')
        if pre.physics:
            map_check(bind_map_api(None), panel)
        u0 = mappo_binding(final_root, 'U0')
        if {'U0', 'RULE', 'RULE_FAIR'} & set(arms) or bank or ct:
            if u0 is None:
                raise Blocked('bind-mappo --arm U0 (frozen d9042cc5 + --residual-path <final_root>/prerun_plant/'
                              'A_checkpoint_000.pt) before the pre-run')
            if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
                raise Blocked('the U0 binding residual is not the pre-run u0 copy')
        waves = []
        bound = {arm: mappo_binding(final_root, arm) for arm in arms if arm in COMPARATOR_ARMS+(CT_ARM,)}
        if bank:
            oracle = oracle_binding(final_root)
            if oracle is None:
                raise Blocked(f'bank-run needs a sealed {ORACLE_BINDING} (bind-oracle)')
            bound.update({arm: oracle for arm in arms})
        for arm in arms:
            if arm in bound and bound[arm] is None:
                raise Blocked(f'{arm} pre-run needs its binding first (bind-mappo --arm {arm})')
        for arm in arms:
            waves.append((arm, 'episode', episode_specs(arm, panel, cfg, plant, 'mappo_frozen', bound.get(arm), u0)))
        for arm in arms if panel_3c is not None else ():      # amendment 23b: a root may hold one panel only
            waves.append((f'C3_{arm}', 'episode', episode_specs(arm, panel_3c, cfg, plant, 'mappo_frozen', bound.get(arm),
                                                                 u0)))
        chunk = int(chunk or 2*workers)
        if chunk < 1:
            raise Blocked('chunk must be >= 1')
        stopped, rounds = False, 0

        def coverage():
            return {name: (published(final_root, name, specs, check_episode)[2], len(specs)) for name, _, specs in waves}
        while True:
            cov = coverage()
            pending = [(name, kind, specs) for name, kind, specs in waves if cov[name][0] < cov[name][1]]
            if not pending:
                break
            if (final_root/STOP_PRERUN).exists():
                stopped = True
                break
            budget, limits, selection = chunk, {}, []
            for name, kind, specs in pending:
                take = min(cov[name][1]-cov[name][0], budget)
                selection.append((name, kind, specs))
                limits[name] = take
                budget -= take
                if budget == 0:
                    break
            run_waves(selection, root=plant, final_root=final_root, cfg=cfg, runtime=pre, workers=workers,
                      base='mappo_frozen', chunked=True, limit=limits)
            rounds += 1
            status(current, 'CHUNK', dict(round=rounds, published={k: v[0] for k, v in coverage().items()}))
        cov = coverage()
        receipts = {name: published(final_root, name, specs, check_episode)[1] for name, _, specs in waves}
        out = dict(schema=SCHEMA+('.bank_done' if bank else '.prerun_done'), status='STOPPED' if stopped else 'PASS',
                   sealed=True, arms=list(arms),
                   prerun_plan_sha256=plan['receipt_sha256'], chunk=chunk, rounds=rounds,
                   coverage={k: dict(published=v[0], registered=v[1]) for k, v in cov.items()},
                   receipts={k: v['receipt_sha256'] for k, v in receipts.items()},
                   charged_sim_s=sum(r['sim_s'] for r in fe.ledger_rows(final_root)),
                   cap_sim_s=CT_CAP_SIM_S if ct else BANK_CAP_SIM_S if bank else effective_prerun_cap(final_root, plan),
                   resume=None if not stopped else ('remove STOP_PRERUN (or copy the whole final root to the same '
                                                    'absolute path on the other host) and run prerun-run again'))
        status(current, out['status'], out)
        return out
    except BaseException as error:
        status(current, 'FAILED', dict(error=repr(error), traceback=traceback.format_exc()))
        raise


def vehicle_log_report(path, *, t0=WARMUP_S, t1=HORIZON_S):
    """Zero physics: field-by-field check + the registered secondary quantities of one ground-truth vehicle log
    (vectorised: per-vehicle delay = sum max(0, 1 - v/SPEED_LIMIT) + depart delay if inserted after t0; same-lane
    consecutive TTC with the nominal 5 m length; stop-and-go v>=5 -> v<1; crossings of the drop s)."""
    import gzip
    heads, ticks, problems = {}, [], []
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            if 'veh' in x:
                heads[x['veh']] = x
            else:
                ticks.append(x)
    if [x.get('t') for x in ticks] != list(range(t0, t1)):
        problems.append(f'ticks {len(ticks)} != {t1-t0} consecutive seconds from {t0}')
    if not heads:
        problems.append('no vehicles')
    if any(set(h) != {'veh', 'id', 'cav', 'origin', 'depart_delay_s', 'first_t'} for h in heads.values()):
        problems.append('vehicle header fields differ from the registered set')
    cols = {c: [] for c in ('t', 'k', 's', 'v', 'a', 'lane')}
    for x in ticks:
        n = len(x.get('k', []))
        if any(len(x.get(c, [])) != n for c in ('s', 'v', 'a', 'lane')):
            problems.append(f't={x.get("t")}: column lengths differ')
            break
        cols['t'].append(np.full(n, x.get('t', -1), np.int64))
        for c in ('k', 's', 'v', 'a', 'lane'):
            cols[c].append(np.asarray(x[c], dtype=np.float64 if c in ('s', 'v', 'a') else np.int64)
                           if n else np.zeros(0, np.float64 if c in ('s', 'v', 'a') else np.int64))
    T, K, S, V, A, L = (np.concatenate(cols[c]) if cols[c] else np.zeros(0) for c in ('t', 'k', 's', 'v', 'a', 'lane'))
    K, L, T = K.astype(np.int64), L.astype(np.int64), T.astype(np.int64)
    if K.size and (K.min() < 0 or not set(np.unique(K).tolist()) <= set(heads)):
        problems.append('unknown vehicle index')
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(V)) and np.all(np.isfinite(A)) and np.all(V >= 0)
            and np.all(L >= 0)):
        problems.append('non-finite or invalid kinematics/lane')
    nveh = (max(heads)+1) if heads else 0
    deficit = np.maximum(0., 1.-V/SPEED_LIMIT_MPS)
    seen = np.bincount(K, minlength=nveh) if K.size else np.zeros(nveh, np.int64)
    delay_v = np.bincount(K, weights=deficit, minlength=nveh) if K.size else np.zeros(nveh)
    for k, h in heads.items():
        if seen[k] and h.get('first_t', -1) > t0 and isinstance(h.get('depart_delay_s'), (int, float)):
            delay_v[k] += float(h['depart_delay_s'])
    present = np.flatnonzero(seen > 0)
    d = np.sort(delay_v[present])
    # per-vehicle accel moments; stop-and-go and drop crossings on consecutive seconds of the same vehicle
    n_ = np.bincount(K, minlength=nveh).astype(float) if K.size else np.zeros(nveh)
    s1 = np.bincount(K, weights=A, minlength=nveh) if K.size else np.zeros(nveh)
    s2 = np.bincount(K, weights=A*A, minlength=nveh) if K.size else np.zeros(nveh)
    ok2 = n_ >= 2
    per_vehicle_var = (s2[ok2]/n_[ok2]-(s1[ok2]/n_[ok2])**2) if ok2.any() else np.zeros(0)
    order = np.lexsort((T, K))
    k_, t_, v_, s_ = K[order], T[order], V[order], S[order]
    same = (k_[1:] == k_[:-1]) & (t_[1:] == t_[:-1]+1)
    stops = int(np.sum(same & (v_[:-1] >= 5.0) & (v_[1:] < 1.0)))
    crossings = len(set(k_[1:][same & (s_[:-1] < DROP_S_M) & (s_[1:] >= DROP_S_M)].tolist()))
    # same-lane consecutive pairs per second (follower = lower s)
    order = np.lexsort((S, L, T))
    t2, l2, s2_, v2 = T[order], L[order], S[order], V[order]
    pair = (t2[1:] == t2[:-1]) & (l2[1:] == l2[:-1])
    gap = s2_[1:]-s2_[:-1]-NOMINAL_LENGTH_M
    closing = v2[:-1]-v2[1:]
    valid = pair & (closing > 0) & (gap > 0)
    ttc = gap[valid]/closing[valid]
    pairs = int(pair.sum())
    if heads and not pairs:
        problems.append('no same-lane vehicle pair (TTC not computable)')
    cav = sum(1 for h in heads.values() if h.get('cav') is True)
    hdv = sum(1 for h in heads.values() if h.get('cav') is False)
    origins = {}
    for h in heads.values():
        origins[h.get('origin')] = origins.get(h.get('origin'), 0)+1
    first_lane, first_s = {}, {}
    if K.size:                                   # the vehicle's FIRST logged row (training convention)
        o = np.lexsort((T, K))
        kk, ll, ss = K[o], L[o], S[o]
        first = np.r_[True, kk[1:] != kk[:-1]]
        first_lane = dict(zip(kk[first].tolist(), ll[first].tolist()))
        first_s = dict(zip(kk[first].tolist(), ss[first].tolist()))

    def origin_of(k):                            # c4/clean5b 1caf5c1c fair_vehicle_log.per_vehicle, verbatim rule
        return (ORIG.get(first_lane[k]) if first_s[k] < DROP_S_M else 'D') if k in first_lane else None
    with_origin = sum(v for o, v in origins.items() if isinstance(o, str) and o)
    with_delay = sum(1 for h in heads.values() if isinstance(h.get('depart_delay_s'), (int, float)))
    if heads and (not cav or not hdv):
        problems.append(f'both fleet classes required (CAV {cav}, HDV {hdv})')
    if heads and with_origin < .95*len(heads):
        problems.append(f'route origin missing for {len(heads)-with_origin}/{len(heads)} vehicles')
    if heads and with_delay < .95*len(heads):
        problems.append(f'depart delay missing for {len(heads)-with_delay}/{len(heads)} vehicles')
    if A.size and float(A.var()) == 0.:
        problems.append('acceleration identically constant')
    gini = float((2*np.arange(1, d.size+1)-d.size-1).dot(d)/(d.size*d.sum())) if d.size and d.sum() > 0 else None
    jain = float(d.sum()**2/(d.size*(d**2).sum())) if d.size and (d**2).sum() > 0 else None
    top = d[int(np.floor(.9*d.size)):] if d.size else d

    def mean_of(keys):
        vals = [delay_v[k] for k in keys if k < nveh and seen[k]]
        return float(np.mean(vals)) if vals else None
    cav_mean = mean_of([k for k, h in heads.items() if h.get('cav') is True])
    hdv_mean = mean_of([k for k, h in heads.items() if h.get('cav') is False])
    by_origin = {str(o): mean_of([k for k, h in heads.items() if h.get('origin') == o]) for o in origins}
    by_lane = {str(ln): mean_of([k for k, x in first_lane.items() if x == ln]) for ln in (0, 1, 2)}
    return dict(ok=not problems, problems=problems, ticks=len(ticks), vehicles=len(heads), cav=cav, hdv=hdv,
                delay_by_class=dict(cav_mean_s=cav_mean, hdv_mean_s=hdv_mean,
                                    hdv_cav_ratio=hdv_mean/cav_mean if cav_mean and hdv_mean is not None else None),
                delay_by_origin_s=by_origin, delay_by_origin_lane_s=by_lane,
                per_vehicle={str(heads[k]['id']): [round(float(delay_v[k]), 3), origin_of(k),
                                                   bool(heads[k].get('cav'))] for k in present.tolist() if k in heads},
                accel_var_per_vehicle_mean=float(per_vehicle_var.mean()) if per_vehicle_var.size else None,
                origins={str(k): v for k, v in sorted(origins.items(), key=lambda kv: str(kv[0]))},
                origin_coverage=round(with_origin/len(heads), 4) if heads else None,
                depart_delay_coverage=round(with_delay/len(heads), 4) if heads else None,
                delay=dict(mean_s=float(d.mean()) if d.size else None, max_s=float(d.max()) if d.size else None,
                           gini=gini, jain=jain, worst10_mean_s=float(top.mean()) if top.size else None),
                ttc=dict(pairs=pairs, min_s=float(ttc.min()) if ttc.size else None, below_3_vs=int(np.sum(ttc < 3.0))),
                accel=dict(n=int(A.size), var=float(A.var()) if A.size else None, hard_brake_vs=int(np.sum(A < -3.0)),
                           emergency_vs=int(np.sum(A < -4.5))),
                stop_and_go=stops, crossings_drop=crossings)


def prerun_logcheck(*, final_root, runtime, panel, arms=LOGCHECK_ARMS, workers=2):
    """ONE short episode per pre-run code path (NATIVE: native plant; RULE: U0 policy + rule) on a TRAIN world,
    own ledger phase v2v_final_logcheck; every log checked field by field (zero physics). Sealed
    LOGCHECK.json; prerun-run refuses without a PASS. Prints no outcome."""
    from tools.coopmwm import map_scenarios as ms
    final_root = Path(final_root)
    plan = prerun_plan(final_root)
    if plan is None:
        raise Blocked('PRERUN_PLAN.json missing: prerun-register first')
    target = final_root/'LOGCHECK.json'
    if target.is_file():
        return verify(read(target), 'receipt_sha256', 'LOGCHECK')
    pre = runtime if isinstance(runtime, PrerunRuntime) else PrerunRuntime(runtime, final_root)
    plant = final_root/PRERUN_PLANT
    cfg = pre.config(plant)
    u0 = mappo_binding(final_root, 'U0')
    if u0 is None and set(arms) != {'NATIVE'}:
        raise Blocked('bind-mappo --arm U0 before the logcheck (the RULE path needs it)')
    if u0 is not None and (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise Blocked('the U0 binding residual is not the pre-run u0 copy')
    world = ms.world(LOGCHECK_SEED, PARITY_CELL['demand'], PARITY_CELL['penetration'])
    specs = []
    for arm in arms:
        ck = episode_specs(arm, panel, cfg, plant, 'mappo_frozen', None, u0)[0]['checkpoint']
        specs.append(dict(job_id=f'nmlogcheck_{arm.lower()}_{LOGCHECK_SEED}', kind='logcheck', arm=arm,
                          world_id=f'{MAP}_train_{LOGCHECK_SEED}', world=deepcopy(world), checkpoint=ck,
                          sim_s=LOGCHECK_STOP_S, window=dict(start_s=WARMUP_S, end_s=LOGCHECK_STOP_S),
                          panel_sha256='logcheck (TRAIN world)', noise_seed=LOGCHECK_SEED, deterministic=True,
                          capture=None))
    rows = {r['id'] for r in fe.ledger_rows(final_root)}
    if any(sp['job_id'] in rows for sp in specs):
        raise Blocked('an interrupted logcheck is charged; never replay (register a fresh root)')
    own_reserve(final_root, [dict(id=sp['job_id'], sim_s=sp['sim_s']) for sp in specs], LOGCHECK_PHASE,
                cap=len(LOGCHECK_ARMS)*LOGCHECK_STOP_S, calls=pre.calls)
    items = [('logcheck', sp, str(final_root/'logcheck'/sp['job_id'])) for sp in specs]
    results = pre.run(items, cfg, workers)
    reports = {}
    for sp, rec, (_, _, work) in zip(specs, results, items):
        if rec.get('status') != 'logged' or any(rec.get(k) != sp[k] for k in IDENTITY):
            raise Blocked(f'logcheck {sp["arm"]}: unexpected record')
        path = Path(work)/VEHICLE_LOG
        if file_sha(path) != rec['vehicle_log']['sha256']:
            raise Blocked(f'logcheck {sp["arm"]}: vehicle log differs from its record')
        reports[sp['arm']] = dict(vehicle_log=rec['vehicle_log'],
                                  check=vehicle_log_report(path, t0=WARMUP_S, t1=LOGCHECK_STOP_S))
    verdict = 'PASS' if all(r['check']['ok'] for r in reports.values()) else 'FAIL'
    value = seal(dict(schema=SCHEMA+'.logcheck', verdict=verdict, world=world, stop_s=LOGCHECK_STOP_S,
                      phase=LOGCHECK_PHASE, sim_s=sum(sp['sim_s'] for sp in specs), arms=reports,
                      prerun_plan_sha256=plan['receipt_sha256'], rule='MAIN: one-episode check before any wave; '
                      'TRAIN world (not held-out); fields checked: ticks, fleet class, route origin, depart delay, '
                      'kinematics, per-vehicle delay, same-lane TTC, acceleration'), 'receipt_sha256')
    publish(target, value)
    return value


SEQCHECK_PHASE = 'v2v_final_seqcheck'
SEQCHECK_ARMS = ('NATIVE', 'U0', 'RULE')


def prerun_seqcheck(*, final_root, runtime, panel):
    """Regression check of the 01:09 failure: NATIVE -> U0 -> RULE short TRAIN episodes in ONE dispatch with ONE
    worker (the old pool would reuse the NATIVE process for U0); each log checked field by field. Sealed
    SEQCHECK.json; 3 x 360 = 1,080 sim-s in phase v2v_final_seqcheck."""
    from tools.coopmwm import map_scenarios as ms
    final_root = Path(final_root)
    plan = prerun_plan(final_root)
    if plan is None:
        raise Blocked('PRERUN_PLAN.json missing')
    target = final_root/'SEQCHECK.json'
    if target.is_file():
        return verify(read(target), 'receipt_sha256', 'SEQCHECK')
    pre = runtime if isinstance(runtime, PrerunRuntime) else PrerunRuntime(runtime, final_root)
    plant = final_root/PRERUN_PLANT
    cfg = pre.config(plant)
    u0 = mappo_binding(final_root, 'U0')
    if u0 is None or (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise Blocked('the U0 binding must be the pre-run u0 copy')
    world = ms.world(LOGCHECK_SEED, PARITY_CELL['demand'], PARITY_CELL['penetration'])
    specs = [dict(job_id=f'nmlogcheck_seq_{arm.lower()}_{LOGCHECK_SEED}', kind='logcheck', arm=arm,
                  world_id=f'{MAP}_train_{LOGCHECK_SEED}', world=deepcopy(world),
                  checkpoint=episode_specs(arm, panel, cfg, plant, 'mappo_frozen', None, u0)[0]['checkpoint'],
                  sim_s=LOGCHECK_STOP_S, window=dict(start_s=WARMUP_S, end_s=LOGCHECK_STOP_S),
                  panel_sha256='logcheck (TRAIN world)', noise_seed=LOGCHECK_SEED, deterministic=True, capture=None)
             for arm in SEQCHECK_ARMS]
    if {sp['job_id'] for sp in specs} & {r['id'] for r in fe.ledger_rows(final_root)}:
        raise Blocked('an interrupted sequence check is charged; never replay')
    own_reserve(final_root, [dict(id=sp['job_id'], sim_s=sp['sim_s']) for sp in specs], SEQCHECK_PHASE,
                cap=len(SEQCHECK_ARMS)*LOGCHECK_STOP_S, calls=pre.calls)
    items = [('logcheck', sp, str(final_root/'logcheck'/sp['job_id'])) for sp in specs]
    results = pre.run(items, cfg, 1)                     # ONE worker: the old pool reused it NATIVE -> U0 -> RULE
    reports = {}
    for sp, rec, (_, _, work) in zip(specs, results, items):
        path = Path(work)/VEHICLE_LOG
        if rec.get('status') != 'logged' or file_sha(path) != rec['vehicle_log']['sha256']:
            raise Blocked(f'sequence check {sp["arm"]}: unexpected record')
        reports[sp['arm']] = dict(vehicle_log=rec['vehicle_log'], pid=rec.get('pid'),
                                  check=vehicle_log_report(path, t0=WARMUP_S, t1=LOGCHECK_STOP_S))
    pids = [r['pid'] for r in reports.values()]
    verdict = 'PASS' if all(r['check']['ok'] for r in reports.values()) and len(set(pids)) == len(pids) else 'FAIL'
    value = seal(dict(schema=SCHEMA+'.seqcheck', verdict=verdict, order=list(SEQCHECK_ARMS), pids=pids,
                      world=world, sim_s=len(specs)*LOGCHECK_STOP_S, phase=SEQCHECK_PHASE, arms=reports,
                      rule='MAIN 2026-11-07: one arm per worker process; one-episode check of the NATIVE->U0->RULE '
                           'sequence (TRAIN world, <= 1,200 sim-s) before re-running 3c'), 'receipt_sha256')
    publish(target, value)
    return value


# ---------------------------------------------------------------- amendment 17 addendum (N5 REVIEW_N6_5)

ENVPAR_PHASE = 'v2v_final_envparity'
ENVPAR_SEEDS = PARITY_WORLDS                     # 2 TRAIN worlds (D3200_p60)
ENVPAR_STOP_S = 420


def _envpar_specs(final_root, cfg, plant, panel):
    from tools.coopmwm import map_scenarios as ms
    plan = prerun_plan(final_root)
    u0, lane = mappo_binding(final_root, 'U0'), mappo_binding(final_root, 'MAPPO_LANE')
    if u0 is None or lane is None or not lane.get('train_src'):
        raise Blocked('the addendum needs the U0 binding and the MAPPO_LANE training-path binding')
    if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
        raise Blocked('the U0 binding residual is not the pre-run u0 copy')
    ck = dict(episode_specs('U0', panel, cfg, plant, 'mappo_frozen', None, u0)[0]['checkpoint'],
              train_config=lane['train_config'], train_config_sha256=lane['train_config_sha256'],
              train_src=lane['train_src'], train_src_sha256=lane['train_src_sha256'])
    out = []
    for seed in ENVPAR_SEEDS:
        world = ms.world(seed, PARITY_CELL['demand'], PARITY_CELL['penetration'])
        for path in ('plant', 'train'):
            out.append(dict(job_id=f'nmenvpar_u0_{path}_{seed}', kind='envparity', arm='U0', path=path,
                            world_id=f'{MAP}_train_{seed}', world=deepcopy(world), checkpoint=deepcopy(ck),
                            sim_s=ENVPAR_STOP_S, window=dict(start_s=WARMUP_S, end_s=ENVPAR_STOP_S),
                            panel_sha256='envparity (TRAIN worlds)', noise_seed=seed, deterministic=True, capture=None))
    return out


def _receipt_diff(a, b, prefix=''):
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        if isinstance(x, dict) and isinstance(y, dict):
            out += _receipt_diff(x, y, f'{prefix}{k}.')
        elif x != y:
            out.append(dict(key=f'{prefix}{k}', plant=x, train=y))
    return out


def env_equivalence(*, final_root, panel, python=None, runner=RUNNER):
    """(a) Zero-SUMO: both paths run up to the SUMO start on TRAIN world 989598 (fresh interpreters: the plant from
    this source tree with the G5 runtime, the training env from the bound S-MAPPOWZL tree); compare the normalised
    command line, scenario files, env kwargs/config and module shas. Sealed ENV_EQUIV.json; never starts SUMO."""
    import subprocess
    final_root = Path(final_root)
    target = final_root/'ENV_EQUIV.json'
    if target.is_file():
        return verify(read(target), 'receipt_sha256', 'ENV_EQUIV')
    plant = final_root/PRERUN_PLANT
    cfg = read(plant/'CONFIG.json')
    spec = _envpar_specs(final_root, cfg, plant, panel)[0]
    work = final_root/'envequiv'
    work.mkdir(parents=True, exist_ok=True)
    (work/'spec.json').write_text(json.dumps(spec, sort_keys=True), encoding='utf-8')
    python = python or sys.executable
    out = {}
    for path in ('plant', 'train'):
        env = dict(os.environ)
        cwd = None
        if path == 'plant':
            env['COOPMWM_G5_RUNTIME'] = '1'
        else:
            from tools.coopmwm.nm_eval_runner import train_subprocess_env
            src = Path(spec['checkpoint']['train_src'])
            cwd, env = str(src), train_subprocess_env(src)
        dest = work/f'start_{path}.json'
        with (work/f'start_{path}.log').open('w') as log:
            subprocess.run([python, '-B', '-m', runner, 'start-receipt', '--path', path, '--spec', str(work/'spec.json'),
                            '--cfg', str(plant/'CONFIG.json'), '--work', str(work/f'work_{path}'), '--out', str(dest)],
                           cwd=cwd, env=env, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        out[path] = read(dest)
    diffs = _receipt_diff({k: out['plant'][k] for k in ('cmd', 'files', 'env')},
                          {k: out['train'][k] for k in ('cmd', 'files', 'env')})
    module_diff = _receipt_diff(out['plant']['modules'], out['train']['modules'])
    value = seal(dict(schema=SCHEMA+'.env_equivalence', world=spec['world'], sim_s=0, verdict='EQUIVALENT' if not diffs
                      else 'DIFFERS', differences=diffs, module_differences=module_diff,
                      sumo_binary=dict(plant=out['plant']['sumo_binary'], train=out['train']['sumo_binary']),
                      env_class=dict(plant=out['plant']['env_class'], train=out['train']['env_class']),
                      env_classes=dict(plant=out['plant']['env_classes'], train=out['train']['env_classes']),
                      module_paths=dict(plant=out['plant'].get('module_paths'), train=out['train'].get('module_paths')),
                      rule='N5 REVIEW_N6_5 fix (a): config/route/option/kwarg/module-sha equivalence on one TRAIN world'),
                 'receipt_sha256')
    publish(target, value)
    return value


def _log_ticks(path):
    import gzip
    heads, ticks = {}, {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            x = json.loads(line)
            if 'veh' in x:
                heads[x['veh']] = x['id']
            else:
                ticks[x['t']] = {heads[k]: (s_, v, ln) for k, s_, v, ln in zip(x['k'], x['s'], x['v'], x['lane'])}
    return ticks


def envparity_compare(plant_log, train_log):
    P, T = _log_ticks(plant_log), _log_ticks(train_log)
    first, worst, n = None, dict(s_m=0., v_mps=0., lane=0), 0
    for t in sorted(set(P) | set(T)):
        a, b = P.get(t, {}), T.get(t, {})
        bad = set(a) != set(b)
        for vid in set(a) & set(b):
            n += 1
            (sa, va, la), (sb, vb, lb) = a[vid], b[vid]
            worst['s_m'], worst['v_mps'] = max(worst['s_m'], abs(sa-sb)), max(worst['v_mps'], abs(va-vb))
            worst['lane'] += la != lb
            bad = bad or (sa, va, la) != (sb, vb, lb)
        if bad and first is None:
            first = dict(t=t, only_plant=sorted(set(a)-set(b))[:10], only_train=sorted(set(b)-set(a))[:10],
                         differing=sorted(v for v in set(a) & set(b) if a[v] != b[v])[:10])
    return dict(identical=first is None and set(P) == set(T), ticks=dict(plant=len(P), train=len(T)), pairs=n,
                max_abs=worst, first_divergence=first)


#: amendment 17c (MAIN): with exactly identical trajectories, J agrees if |dJ|/J <= 1e-9 (floating-point summation order)
J_REL_TOL = 1e-9


def j_within(a, b):
    return abs(a-b) <= J_REL_TOL*max(abs(a), abs(b), 1e-12)


def envparity_assess_17c(final_root):
    """Zero physics: the amendment-17c criterion on the sealed ENVPARITY_U0.json (trajectories identical every tick
    AND |dJ|/J <= 1e-9 on every world). Sealed ENVPARITY_U0_17C.json."""
    final_root = Path(final_root)
    target = final_root/'ENVPARITY_U0_17C.json'
    if target.is_file():
        return verify(read(target), 'receipt_sha256', target.name)
    base = verify(read(final_root/'ENVPARITY_U0.json'), 'receipt_sha256', 'ENVPARITY_U0')
    worlds = {k: dict(trajectories_identical=w['compare']['identical'], pairs=w['compare']['pairs'], J=w['J'],
                      rel_dJ=abs(w['J']['plant']-w['J']['train'])/max(abs(w['J']['plant']), 1e-12),
                      J_within=j_within(w['J']['plant'], w['J']['train'])) for k, w in base['worlds'].items()}
    value = seal(dict(schema=SCHEMA+'.envparity_17c', source_receipt=base['receipt_sha256'], tolerance=J_REL_TOL,
                      worlds=worlds, accepted=all(w['trajectories_identical'] and w['J_within'] for w in worlds.values()),
                      rule='amendment 17c: trajectories identical every tick AND |dJ|/J <= 1e-9'), 'receipt_sha256')
    publish(target, value)
    return value


def prerun_envparity(*, final_root, runtime, panel, workers=4):
    """(b) U0 plant vs S-MAPPOWZL training env: 2 TRAIN worlds x (300+120 s) x 2 paths = 1,680 sim-s in the
    evaluation root ledger (phase v2v_final_envparity), canary first; pass = every vehicle's s/v/lane per second
    and J identical. Sealed ENVPARITY_U0.json; reports the first divergence, never patches."""
    final_root = Path(final_root)
    target = final_root/'ENVPARITY_U0.json'
    if target.is_file():
        return verify(read(target), 'receipt_sha256', 'ENVPARITY_U0')
    pre = runtime if isinstance(runtime, PrerunRuntime) else PrerunRuntime(runtime, final_root)
    plant = final_root/PRERUN_PLANT
    cfg = pre.config(plant)
    specs = _envpar_specs(final_root, cfg, plant, panel)
    if {sp['job_id'] for sp in specs} & {r['id'] for r in fe.ledger_rows(final_root)}:
        raise Blocked('an interrupted addendum parity run is charged; never replay')
    items = [('envparity', sp, str(final_root/'evaluation'/sp['job_id'])) for sp in specs]
    own_reserve(final_root, [dict(id=specs[0]['job_id'], sim_s=specs[0]['sim_s'])], ENVPAR_PHASE,
                cap=len(specs)*ENVPAR_STOP_S, calls=pre.calls)
    first = pre.run(items[:1], cfg, 1)                         # canary: one plant job verified first
    if first[0].get('status') != 'traced' or first[0]['vehicle_log']['ticks'] != ENVPAR_STOP_S-WARMUP_S:
        raise Blocked('addendum parity canary failed; nothing else charged')
    own_reserve(final_root, [dict(id=sp['job_id'], sim_s=sp['sim_s']) for sp in specs[1:]], ENVPAR_PHASE,
                cap=len(specs)*ENVPAR_STOP_S, calls=pre.calls)
    records = first+pre.run(items[1:], cfg, workers)
    worlds = {}
    for seed in ENVPAR_SEEDS:
        rec = {r['path']: (r, it) for r, it in zip(records, items) if r['world']['seed'] == seed}
        logs = {p: Path(rec[p][1][2])/VEHICLE_LOG for p in ('plant', 'train')}
        for p in logs:
            if file_sha(logs[p]) != rec[p][0]['vehicle_log']['sha256']:
                raise Blocked(f'addendum parity {seed} {p}: vehicle log differs from its record')
        cmp_ = envparity_compare(logs['plant'], logs['train'])
        J = {p: rec[p][0]['J_veh_s'] for p in logs}
        worlds[str(seed)] = dict(compare=cmp_, J=J, J_identical=J['plant'] == J['train'],
                                 J_within_17c=j_within(J['plant'], J['train']),
                                 safety={p: rec[p][0]['safety'] for p in logs})
    verdict = 'IDENTICAL' if all(w['compare']['identical'] and w['J_identical'] for w in worlds.values()) else 'DIFFERS'
    accepted = all(w['compare']['identical'] and w['J_within_17c'] for w in worlds.values())
    value = seal(dict(schema=SCHEMA+'.envparity', verdict=verdict, accepted_17c=accepted, worlds=worlds,
                      sim_s=len(specs)*ENVPAR_STOP_S,
                      phase=ENVPAR_PHASE, rule=('N5 REVIEW_N6_5 fix (b): U0 plant vs S-MAPPOWZL training env; pass = every '
                                                'vehicle s/v/lane per second and J identical; on DIFFERS the MAPPO_LANE '
                                                'contrasts are labelled "different simulator path" or rerun in the plant')),
                 'receipt_sha256')
    publish(target, value)
    return value


HOSTID_PHASE = 'v2v_final_hostcheck'
HOSTID_SEED = ENVPAR_SEEDS[0]
#: amendment 21: per host move, one U0 plant replay of the TRAIN world envparity job (300 s warm-up + 120 s)
HOSTID_CAP_S = 4*ENVPAR_STOP_S
#: the N6 overlay (tmp/n6_deploy_<sha>/SHA256SUMS)
DEPLOY_FILES = ('ops/clean3_finaleval.py', 'ops/nm_eval_panel_SUPERSEDED.json', 'ops/nm_eval_panel_wz3to2_claim3c_D2400.json',
                'ops/nm_eval_panel_wz3to2_v2.json', 'ops/nm_finaleval.py', 'tools/coopmwm/imitreplay.py',
                'tools/coopmwm/nm_eval_runner.py', 'tools/coopmwm/regimeu0.py', 'tools/coopmwm/regimeu0_multicrit.py')
RECEIPT_KEYS = ('cmd', 'files', 'env', 'env_class', 'env_classes', 'sumo_binary')


def host_identity(*, final_root, runtime, panel, host, python=None, runner=RUNNER, sumo=True, reference=None):
    """Plant identity after a host move (amendment 21). Never reads a held-out outcome.
    (1) zero SUMO: the plant start receipt on this host (fresh interpreter, G5 runtime, traci.start intercepted) vs
        the reference plant receipt from the previous host (default <final_root>/envequiv/start_plant.json): command
        line, scenario files, env kwargs/config, env classes, SUMO binary path must be equal; module sha differences
        are listed and split into N6 overlay files (expected after an overlay update) and others (must be none);
        the SUMO binary sha256 and `sumo --version` on this host are recorded.
    (2) ``sumo``: the U0 plant envparity job on TRAIN world 989598 (420 sim-s, own ledger phase v2v_final_hostcheck,
        cap 1,680) vs the previous host's plant log of the same job (evaluation/nmenvpar_u0_plant_989598): every
        vehicle's s/v/lane identical every tick and J within 1e-9 of ENVPARITY_U0. Sealed HOST_IDENTITY_<host>.json."""
    import re
    import shutil
    import subprocess
    final_root = Path(final_root)
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,32}', host or ''):
        raise Blocked('host label: 1..32 of [A-Za-z0-9_-]')
    target = final_root/f'HOST_IDENTITY_{host}.json'
    if target.is_file():
        return verify(read(target), 'receipt_sha256', target.name)
    ref_path = Path(reference) if reference else final_root/'envequiv'/'start_plant.json'
    ref = read(ref_path)
    plant = final_root/PRERUN_PLANT
    pre = runtime if isinstance(runtime, PrerunRuntime) else PrerunRuntime(runtime, final_root)
    cfg = pre.config(plant)
    specs = [sp for sp in _envpar_specs(final_root, cfg, plant, panel)
             if sp['path'] == 'plant' and sp['world']['seed'] == HOSTID_SEED]
    work = final_root/f'hostid_{host}'
    work.mkdir(parents=True, exist_ok=False)
    (work/'spec.json').write_text(json.dumps(specs[0], sort_keys=True), encoding='utf-8')
    env = dict(os.environ, COOPMWM_G5_RUNTIME='1')
    dest = work/'start_plant.json'
    with (work/'start_plant.log').open('w') as log:
        subprocess.run([python or sys.executable, '-B', '-m', runner, 'start-receipt', '--path', 'plant', '--spec',
                        str(work/'spec.json'), '--cfg', str(plant/'CONFIG.json'), '--work', str(work/'work_plant'),
                        '--out', str(dest)], env=env, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
    here = read(dest)
    relabel = lambda ds: [dict(key=d['key'], previous_host=d['plant'], this_host=d['train']) for d in ds]  # noqa: E731
    diffs = relabel(_receipt_diff({k: ref.get(k) for k in RECEIPT_KEYS}, {k: here.get(k) for k in RECEIPT_KEYS}))
    overlay = {Path(f).with_suffix('').as_posix().replace('/', '.') for f in DEPLOY_FILES if f.endswith('.py')}
    mods = relabel(_receipt_diff(ref['modules'], here['modules']))
    changed = sorted(d['key'] for d in mods)
    binary = shutil.which(Path(here['sumo_binary']).name) if not Path(here['sumo_binary']).is_file() else here['sumo_binary']
    try:
        version = subprocess.run([binary, '--version'], capture_output=True, text=True, timeout=60).stdout.splitlines()[:1]
    except (OSError, TypeError, subprocess.SubprocessError):
        version = None
    zero = dict(reference=str(ref_path), reference_sha256=file_sha(ref_path), receipt_differences=diffs,
                module_differences=mods,
                overlay_modules_changed=[m for m in changed if m in overlay],
                other_modules_changed=[m for m in changed if m not in overlay],
                sumo_binary=dict(path=here['sumo_binary'], sha256=file_sha(binary) if binary and Path(binary).is_file()
                                 else None, version=version),
                module_paths=here.get('module_paths'))
    zero['identical'] = not diffs and not zero['other_modules_changed']
    replay = dict(status='not requested')
    if sumo and zero['identical']:
        spec = dict(specs[0], job_id=f'nmhostid_{host}_u0_plant_{HOSTID_SEED}')
        own_reserve(final_root, [dict(id=spec['job_id'], sim_s=spec['sim_s'])], HOSTID_PHASE, cap=HOSTID_CAP_S,
                    calls=pre.calls)
        rec = pre.run([('envparity', spec, str(work/spec['job_id']))], cfg, 1)[0]
        mine = work/spec['job_id']/VEHICLE_LOG
        if rec.get('status') != 'traced' or file_sha(mine) != rec['vehicle_log']['sha256']:
            raise Blocked('host identity replay did not trace or its log differs from its record')
        old = final_root/'evaluation'/specs[0]['job_id']/VEHICLE_LOG
        base = verify(read(final_root/'ENVPARITY_U0.json'), 'receipt_sha256', 'ENVPARITY_U0')
        J_old = base['worlds'][str(HOSTID_SEED)]['J']['plant']
        cmp_ = envparity_compare(old, mine)
        replay = dict(job_id=spec['job_id'], sim_s=spec['sim_s'], phase=HOSTID_PHASE, compare=cmp_,
                      J=dict(previous_host=J_old, this_host=rec['J_veh_s']),
                      J_within=j_within(J_old, rec['J_veh_s']),
                      identical=bool(cmp_['identical'] and j_within(J_old, rec['J_veh_s'])))
    elif sumo:
        replay = dict(status='skipped: the zero-SUMO receipt differs (no SUMO spent)')
    verdict = 'IDENTICAL' if zero['identical'] and (not sumo or replay.get('identical')) else 'DIFFERS'
    value = seal(dict(schema=SCHEMA+'.host_identity', host=host, verdict=verdict, zero_sumo=zero, replay=replay,
                      rule=('amendment 21: zero-SUMO plant start receipt equal to the previous host (module changes only '
                            'in N6 overlay files) AND the 420 s TRAIN-world U0 plant replay identical every tick with J '
                            'within 1e-9; no held-out outcome is read')), 'receipt_sha256')
    publish(target, value)
    return value


def prerun_compat(final_root, cfg, runtime):
    """The clean run may reuse the pre-run waves only if its plant view and common u0 equal the
    PRERUN_PLAN (zero physics; checked at register and again before the episodes)."""
    plan = prerun_plan(final_root)
    if plan is None:
        return None
    view = runtime.plant_view(cfg)
    differs = sorted(k for k in PLANT_KEYS if view.get(k) != plan['plant']['view'].get(k))
    if differs:
        raise Blocked(f'clean run plant differs from the pre-run plant on {differs}: the pre-run waves are not reusable')
    if cfg.get('common_checkpoint_sha256') != plan['u0']['sha256']:
        raise Blocked('clean run common u0 differs from the pre-run u0')
    if plan['rule']['rule_sha256'] != RULE_SHA256 or plan['rule'].get('strategy', RULE_STRATEGY) != RULE_STRATEGY:
        raise Blocked('pre-run RULE digest/strategy differs from this launcher')
    source = dict(cfg)
    return dict(status='compatible', prerun_plan_sha256=plan['receipt_sha256'], plant_keys=list(PLANT_KEYS),
                same_config=digest(source) == plan['plant']['config_sha256'])


def register(*, root, final_root, runtime, panel, base=None):
    """Zero physics: the output root and its sealed run plan (binding + budget check included)."""
    root, final_root = Path(root), Path(final_root)
    guard(final_root)
    if (final_root/'RUN_PLAN.json').exists():
        raise Blocked('RUN_PLAN.json exists; never overwrite')
    if final_root.exists() and prerun_plan(final_root) is None:
        raise Blocked('final root exists and is not a registered pre-run root; never overwrite')
    cfg = runtime.config(root)
    binding = clean4_binding(cfg, panel, declared_base=base)
    exposure_check(root, cfg)
    budget = budget_check(root, cfg, binding['base'], final_root=final_root)
    compat = prerun_compat(final_root, cfg, runtime) if final_root.exists() else None
    final_root.mkdir(parents=True, exist_ok=True)
    plan = seal(dict(schema=SCHEMA+'.plan', created=datetime.now(timezone.utc).isoformat(), clean4_root=str(root),
                     config_sha256=digest(cfg), binding=binding, panel_sha256=panel['panel_sha256'],
                     contract_sha256=digest(panel['contract']), budget=budget,
                     quote=quote(binding['base'], a_alias=a_frozen(cfg)),
                     prerun=compat,
                     updates=dict(evaluated=list(EVAL_UPDATES), primary=PRIMARY_UPDATE, authority=UPDATE_AUTHORITY,
                                  a_alias_of_u0=a_frozen(cfg)),
                     amendments=[dict(rule=MAPPO_AUTHORITY, allocation=ALLOCATION_AUTHORITY, arms=list(EPISODE_ARMS),
                                      base_alias=alias_of(binding['base']),
                                      quote_note='7 distinct arms for either base + 8-world decision branches + F11 '
                                                 'parity; supersedes the 4-arm native quote written into panel v2')],
                     authority=('MAIN decision 2026-11-06 (N6): only clean4 final-evaluation runner; debit clean4 '
                                'v2v_final; '+ALLOCATION_AUTHORITY),
                     rule=('frozen u24 arms + base/native comparators on the registered wz3to2 evaluation panel; '
                           'descriptive; never gates clean4; one attempt per wave (never replayed)')), 'receipt_sha256')
    publish(final_root/'RUN_PLAN.json', plan)
    return plan


def check(*, root, final_root, runtime, panel, map_module=None, update=PRIMARY_UPDATE):
    out = dict(schema=SCHEMA+'.check', panel_sha256=panel['panel_sha256'], new_sim_s=0)
    try:
        cfg, binding, report = preflight(root, final_root, runtime, panel, map_module)
        out.update(report, quote=quote(binding['base'], a_alias=a_frozen(cfg)), update=update)
        out['run_plan'] = (run_plan(final_root) or {}).get('receipt_sha256') if Path(final_root).is_dir() else None
        try:
            out['gate'] = gate(root, cfg, runtime, binding, final_root, check_update(update))
        except (NotReady, Blocked) as error:
            out['gate'] = dict(status=type(error).__name__, reason=str(error))
        out['status'] = 'PASS'
    except Blocked as error:
        out.update(status='BLOCKED', reason=str(error))
    return out


# ---------------------------------------------------------------- dry run (fake physics)

def build_fake_root(out, panel, base):
    """A disposable clean4-shaped root at u24 bound to the real registered panel; fake checkpoints and
    sealed fake receipts; no outcome of any real run is used."""
    root = Path(out)/'clean4_root'
    root.mkdir(parents=True)
    (root/'A_checkpoint_000.pt').write_bytes(b'dry-run fake common u0')
    cfg = dict(run_root=str(root), accounting_root=str(root), source_root=str(Path(out)/'no_source'), runtime='dry',
               common_checkpoint_sha256=file_sha(root/'A_checkpoint_000.pt'), frozen_control_arm=FROZEN_A_RULE,
               clean4_budget=seal(dict(phases=dict(v2v_train=6_000_000, isolated_replicated=165_600,
                                                   v2v_final=ALLOCATION_SIM_S)), 'receipt_sha256'),
               newmap=dict(map=MAP, base_controller=base, eval_panel_sha256=panel['panel_sha256'],
                           cell=dict(demand_vph=dict(panel['cell']['demand_vph']), penetration=panel['cell']['penetration'],
                                     profile=panel['cell']['profile']), train_seed_range=list(TRAIN_RANGE)))
    publish(root/'CONFIG.json', cfg)
    with (root/'CHARGES.jsonl').open('w', encoding='utf-8') as f:
        for i in range(3):
            f.write(json.dumps(dict(id=f'clean4_train_A_{989200+i}', phase='v2v_train', sim_s=1200))+'\n')
    for arm in ARMS:
        for u in EVAL_UPDATES:
            (root/f'{arm}_checkpoint_{u:03}.pt').write_bytes(f'dry-run fake {arm} u{u}'.encode())
            publish(root/f'{arm}_UPDATE_{u:03}.json', seal(dict(arm=arm, update=u, dry_run=True), 'update_sha256'))
        publish(root/f'{arm}_COMPLETE.json', dict(arm=arm, updates=UPDATE, dry_run=True))
    zero = {k: 0 for k in SAFETY}
    for u in range(1, UPDATE+1):
        publish(root/f'LEARNING_MONITOR_u{u:03}.json', seal(dict(update=u, config_sha256=digest(cfg), dry_run=True,
            arms=[dict(arm=a, action='CONTINUE', metrics=dict(safety=dict(zero))) for a in ARMS]), 'receipt_sha256'))
    return root, cfg


def fake_replicated(root, update=UPDATE):
    from tools.coopmwm import track2_monitor as p3
    records = []
    for arm in ('B', 'R'):
        for i in range(5):
            gain = (3.0 if arm == 'B' else 0.5)+i*0.4
            records.append(dict(arm=arm, replicate=i, world_seed=989200+i, state_set_sha256=f's{i}', window_set_sha256=f'w{i}',
                                metric=gain, **{fe.SYNERGY: gain, fe.TOTAL_BENEFIT: 10+gain}, safety={k: 0 for k in SAFETY},
                                reference_safety={k: 0 for k in SAFETY}))
    analysis = p3._analyse(records, update, dict(p3.BOOTSTRAP))
    publish(Path(root)/replicated_file(update), seal(dict(kind='replicated', update=update, status='complete', dry=False,
            records=records, dry_run=True, **analysis), 'receipt_sha256'))


def bind_fake_mappo(out, final_root, arm='MAPPO_SPEED', residual=None):
    """Dry run only: a stand-in executor file bound as a retrained comparator (U0: + the u0 residual)."""
    import torch
    fake = Path(out)/f'fake_{arm.lower()}_wz3to2.pt'
    if not fake.exists():
        policy = {'actor.0.weight': torch.zeros(1)}
        if arm == 'MAPPO_LANE':
            policy['lane_head.weight'] = torch.zeros(1)
        torch.save(dict(format='c4_dreamer/3', trainer='mappo', iteration=168 if arm == 'MAPPO_SPEED' else 999,
                        lane_control='native' if arm in ('MAPPO_SPEED', 'MAPPO_CT') else 'policy', lane_head=arm == 'MAPPO_LANE',
                        policy=policy, note='dry-run stand-in, not a MAPPO export'), fake)
    kind = next(iter(MAPPO_KINDS[arm]))
    extra = {}
    if arm == 'U0':
        extra = dict(residual_path=str(residual), residual_sha256=file_sha(residual))
    else:                                    # dry run: a stand-in training config for the F11 path
        config = Path(out)/f'fake_{arm.lower()}_train_config.toml'
        if not config.exists():
            config.write_text(f'# dry-run stand-in for the {arm} training run config' + chr(10), encoding='utf-8')
        extra = dict(train_config=config)
        if arm == 'MAPPO_LANE':              # dry run: a stand-in training source tree (amendment 17)
            from tools.coopmwm.nm_eval_runner import LANE_TRAIN_FILES
            src = Path(out)/'fake_mappo_lane_train_src'
            for f in LANE_TRAIN_FILES:
                (src/f).parent.mkdir(parents=True, exist_ok=True)
                if not (src/f).exists():
                    (src/f).write_text('# dry-run stand-in' + chr(10), encoding='utf-8')
            extra['train_src'] = src
    return bind_mappo(final_root=final_root, path=fake, sha256=file_sha(fake), kind=kind, arm=arm,
                      authority='DRY RUN ONLY - not N7', check=dict(status='dry run'), dry_run=arm in REQUIRED_SHA,
                      **extra)


def dry_run(*, out, panel, base='mappo_frozen', replicated=True, prerun=True):
    """Full launcher path with fake physics on the REAL registered panels: the sealed pre-run of
    NATIVE/U0/RULE (3a + 3c) first, then the main run reusing it. Writes only under ``out``."""
    out = guard(Path(out))
    if out.exists() and any(out.iterdir()):
        raise Blocked('dry-run output directory must be empty')
    out.mkdir(parents=True, exist_ok=True)
    root, cfg = build_fake_root(out, panel, base)
    if replicated:
        for u in EVAL_UPDATES:
            fake_replicated(root, u)
    runtime = FakeRuntime()
    final_root = out/'final_root'
    panel_3c = load_panel_3c()
    pre = None
    if prerun:
        prerun_register(final_root=final_root, plant_config=root/'CONFIG.json', u0_path=root/'A_checkpoint_000.pt',
                        u0_sha256=cfg['common_checkpoint_sha256'], runtime=runtime, panel=panel, panel_3c=panel_3c,
                        authority='DRY RUN ONLY')
        bind_fake_mappo(out, final_root, 'U0', residual=final_root/PRERUN_PLANT/'A_checkpoint_000.pt')
        bind_fake_mappo(out, final_root, 'MAPPO_LANE')
        logcheck = prerun_logcheck(final_root=final_root, runtime=runtime, panel=panel)
        if logcheck['verdict'] != 'PASS':
            raise Blocked(f'dry-run logcheck failed: {logcheck["arms"]}')
        pre = prerun_execute(final_root=final_root, runtime=runtime, panel=panel, panel_3c=panel_3c, workers=16)
    plan = register(root=root, final_root=final_root, runtime=runtime, panel=panel)
    for arm in BOUND_ARMS:
        if not prerun or arm not in ('U0', 'MAPPO_LANE'):
            bind_fake_mappo(out, final_root, arm, residual=root/'A_checkpoint_000.pt')
    value = execute(root=root, final_root=final_root, runtime=runtime, panel=panel, panel_3c=panel_3c,
                    poll=0, dry_run=True, update=PRIMARY_UPDATE)
    secondary = execute(root=root, final_root=final_root, runtime=runtime, panel=panel, panel_3c=panel_3c,
                        poll=0, dry_run=True, update=UPDATE)
    charged = sum(r['sim_s'] for r in fe.ledger_rows(final_root) if r['phase'] == PHASE)
    return dict(status='PASS', new_sim_s=0, physics=False, out=str(out), final_root=str(final_root), base=base,
                target='clean5a (dry)',
                analyses={k: v['analysis_sha256'] for k, v in value.items()},
                present={k: v.get('present') for k, v in value.items()}, calls=runtime.calls,
                secondary={k: v['analysis_sha256'] for k, v in secondary.items()},
                fake_v2v_final_sim_s=charged, quote=plan['quote'], prerun=pre,
                fake_prerun_sim_s=sum(r['sim_s'] for r in fe.ledger_rows(final_root) if r['phase'] == PRERUN_PHASE),
                tables=sorted(str(x) for x in final_root.glob('CLAIM3*_TABLES_u*.md')))


# ---------------------------------------------------------------- CLI

def _analyse_published(root, final_root, runtime, panel, panel_3c=None, update=PRIMARY_UPDATE):
    """Zero physics: re-tabulate every analysis of one update whose arms are published (3a, 3c; each bound 3b)."""
    update = check_update(update)
    sfx = f'_u{update:03}'
    cfg = runtime.config(root)
    plan = run_plan(final_root)
    if plan is None:
        raise Blocked('no RUN_PLAN.json')
    base = plan['binding']['base']
    a_alias = a_frozen(cfg)
    state = comparators(final_root)
    bindings = state['bound']

    def load(arm, pnl, prefix=''):
        name = wave_name(arm, update, prefix)
        if not (Path(final_root)/f'WAVE_{name}.json').is_file() and not _chunk_files(final_root, name):
            return None, 'no published wave receipt'
        if arm in BOUND_ARMS and arm not in bindings:
            raise Blocked(f'{arm} wave published without a binding')
        binding = bindings.get(arm)
        if arm in BANK_ARMS:
            if prefix:
                raise Blocked(f'{arm} runs on the primary panel only (amendment 25b)')
            binding = oracle_binding(final_root)
            if binding is None:
                raise Blocked(f'{arm} wave published without a sealed {ORACLE_BINDING}')
        specs = episode_specs(arm, pnl, cfg, root, base, binding, bindings.get('U0'), update)
        records, source, covered = published(final_root, name, specs, check_episode)
        if covered < len(specs):
            return None, f'wave incomplete ({covered}/{len(specs)} published)'
        return ({r['world_id']: r for r in records}, source), None
    episodes, sources, missing = {}, {}, {}
    for arm in EPISODE_ARMS:
        if (arm == 'A' and a_alias) or (arm in BANK_ARMS and SETTING != 'p20x'):
            continue
        got, why = load(arm, panel)
        if got is None:
            missing[arm] = why
        else:
            episodes[arm], sources[wave_name(arm, update)] = got
    if a_alias and 'U0' in episodes:
        episodes['A'] = episodes['U0']
    if any(a in missing for a in ARMS) or not {'A', 'B', 'R'} <= set(episodes):
        raise Blocked(f'A/B/R u{update} waves are not all published: {missing}')
    missing['DECISIONS'] = DECISIONS_UNAVAILABLE
    if alias_of(base) in missing:
        missing['BASE'] = f'BASE is {alias_of(base)}: {missing[alias_of(base)]}'
    common = dict(root=root, final_root=final_root, base=base, update=update, a_alias=a_alias)
    core = {a: v for a, v in episodes.items() if a in CLAIM3A_ARMS}
    out = dict(claim3a=tables(panel=panel, episodes=core, decisions=None, sources=sources,
                              missing=dict(missing, **{a: 'claim 3b (separate seal)' for a in COMPARATOR_ARMS}),
                              comparators={a: b for a, b in bindings.items() if a == 'U0'}, scope='claim3a',
                              stem=f'CLAIM3A_TABLES{sfx}', **common))
    if panel_3c is not None:
        e3c, s3c, m3c = {}, {}, {}
        for arm in CLAIM3C_ARMS:
            got, why = load(arm, panel_3c, 'C3_')
            if got is None:
                m3c[arm] = why
            else:
                e3c[arm], s3c[wave_name(arm, update, 'C3_')] = got
        if {'B', 'R'} <= set(e3c):
            out['claim3c'] = tables(panel=panel_3c, episodes=e3c, decisions=None, sources=s3c, missing=m3c,
                                    comparators={a: b for a, b in bindings.items() if a == 'U0'}, scope='claim3c',
                                    stem=f'CLAIM3C_TABLES{sfx}', **common)
    for arm in COMPARATOR_ARMS:
        if arm in episodes:
            out[f'claim3b_{arm}'] = tables(
                panel=panel, episodes=dict(core, **{arm: episodes[arm]}), decisions=None, sources=sources,
                missing=dict(missing, **{a: 'claim 3b (separate seal)' for a in COMPARATOR_ARMS if a != arm}),
                comparators={a: b for a, b in bindings.items() if a in ('U0', arm)}, scope=f'claim3b_{arm}',
                stem=f'CLAIM3B_{arm}_TABLES{sfx}', **common)
    return out


# ---------------------------------------------------------------- amendment 28: p20c confirmation (trained roots)

TRAINED_BOUND = 'TRAINED_BOUND_u012.json'


def _bound_name(update):
    return f'TRAINED_BOUND_u{int(update):03}.json'


def _amendment(update):
    return '28' if int(update) == 12 else '29'
CONFIRM_TABLES = 'CONFIRM_TABLES.json'
#: the registered contrasts (amendment 28): (treatment, reference, role); diff = treatment - reference
CONFIRM_CONTRASTS = (('ORACLE_WM', 'U0', 'PRIMARY'), ('LEARNED_TRIGGER', 'U0', 'SECONDARY'),
                     ('C5D', 'U0', 'SECONDARY'), ('C5D', 'F8B', 'SECONDARY'),
                     ('C5D', 'C5DR', 'SECONDARY'))      # MAIN 09:26: the effect of world-model guidance on learning
CONFIRM_PRIMARY_RULE = ('ORACLE_WM - U0 paired total delay (J veh-s per world, whole-world bootstrap) 95% CI upper < 0 '
                        'on the 192-world p20c panel; fairness co-criteria (every registered group mean per-vehicle '
                        'delay increase <= 10 s) and CVaR90 (95% CI lower <= 0) reported for every contrast; a '
                        'SEPARATE pre-registered confirmation test, never pooled with the amendment-27 p20x panel')


#: amendment 29 (owner 'go to 24', MAIN 2026-11-09 09:24, pre-outcome): CLEAN5C5D-B u24 and F8V4B-B u24 on the SAME
#: p20c panel with the amendment-28 U0; reported separately from u12
CONFIRM_SPECS = {
    '28': dict(update=12, bank_arms=('U0',)+CONFIRM_BANK_ARMS, contrasts=CONFIRM_CONTRASTS, rule=CONFIRM_PRIMARY_RULE,
               labels=('C5D', 'F8B', 'C5DR'),
               file=CONFIRM_TABLES,
               pooling='never pooled with the amendment-27 p20x panel (separate pre-registered test)'),
    '29': dict(update=24, bank_arms=('U0',), labels=('C5D', 'F8B', 'C5DR'),
               contrasts=(('C5D', 'U0', 'PRIMARY'), ('C5D', 'F8B', 'SECONDARY'), ('C5D', 'C5DR', 'SECONDARY')),
               rule=('CLEAN5C5D-B u24 - U0 (the amendment-28 U0 of the same panel) paired total delay 95% CI upper < 0; '
                     'secondary C5D u24 - F8B u24; fairness co-criteria and CVaR90 for both; plant identity inherited '
                     'from the amendment-28 u12 root of the same training CONFIG'),
               file='CONFIRM_TABLES_A29.json',
               pooling='reported separately from the u12 (amendment 28) tables; never pooled with p20x'),
    '30': dict(update=0, bank_arms=('U0', 'LEARNED_TRIGGER', CT_ARM), labels=(),
               contrasts=((CT_ARM, 'U0', 'PRIMARY'), ('LEARNED_TRIGGER', CT_ARM, 'SECONDARY')),
               rule=('MAPPO_CT - U0 (the amendment-28 U0 of the same panel) paired total delay (J veh-s per world, '
                     'whole-world bootstrap) 95% CI reported, PASS = upper < 0; secondary LEARNED_TRIGGER - MAPPO_CT; '
                     'fairness co-criteria and CVaR90 for both. ' + CT_AUTHORITY),
               file='CONFIRM_TABLES_A30.json',
               pooling='same panel and U0 as amendment 28; reported separately; never pooled with p20x'),
}


def _need_p20c():
    if SETTING != 'p20c':
        raise Blocked('amendment 28 commands run with --setting p20c only')


def _trained_label(label):
    _need_p20c()
    if label not in TRAINED_ARMS:
        raise Blocked(f'trained arm label must be one of {sorted(TRAINED_ARMS)}')
    return label


def trained_plan(final_root):
    path = Path(final_root)/TRAINED_PLAN
    return verify(read(path), 'receipt_sha256', 'TRAINED_PLAN seal') if path.is_file() else None


def trained_bound(final_root):
    plan = trained_plan(final_root)
    if plan is None:
        return None
    path = Path(final_root)/_bound_name(plan['update'])
    return verify(read(path), 'receipt_sha256', path.name) if path.is_file() else None


class TrainedRuntime(PrerunRuntime):
    """Amendment 28: one trained arm on its own training CONFIG/source (the exact training execution path), charged to
    the trained final root's own ledger: phase IDENTITY_PHASE (U0 plant identity, 2 worlds) or TRAINED_PHASE (B)."""

    def __init__(self, inner, final_root, phase):
        super().__init__(inner, final_root)
        if phase not in (IDENTITY_PHASE, TRAINED_PHASE):
            raise Blocked(f'unknown trained phase {phase}')
        self.phase = phase
        self.cap = IDENTITY_CAP_SIM_S if phase == IDENTITY_PHASE else TRAINED_CAP_SIM_S

    def config(self, root):
        plan = trained_plan(self.final_root)
        if plan is None:
            raise Blocked(f'{TRAINED_PLAN} missing: confirm-register first')
        if Path(root) != Path(plan['training_root']):
            raise Blocked('the trained plant root is the registered training root')
        cfg = self.inner.config(root)
        if digest(cfg) != plan['config_sha256']:
            raise Blocked('training CONFIG differs from TRAINED_PLAN')
        return cfg

    def check_capacity(self, amount):
        rows = fe.ledger_rows(self.final_root)
        if sum(r['sim_s'] for r in rows if r.get('phase') == self.phase)+amount > self.cap:
            raise Blocked(f'{self.phase} cap {self.cap} would be exceeded')
        if sum(r['sim_s'] for r in rows)+amount > TRAINED_CAP_SIM_S+IDENTITY_CAP_SIM_S:
            raise Blocked('trained root allocation would be exceeded')

    def reserve(self, cfg, jobs):
        try:
            own_reserve(self.final_root, jobs, self.phase, cap=self.cap, calls=self.calls)
        except Blocked as error:
            raise Blocked(f'trained root: {error}') from None


def confirm_register(*, root, final_root, runtime, panel, label, authority, update=TRAINED_UPDATE):
    """Zero physics: a NEW trained final root for one amendment-28 trained arm, bound to its training root and CONFIG
    (validated by the deployed source's production loader), the p20c panel, phases and caps. The B checkpoint is bound
    later (bind-trained, after the u12 export)."""
    label = _trained_label(label)
    if int(update) not in TRAINED_UPDATES:
        raise Blocked(f'trained update must be one of {TRAINED_UPDATES} (amendments 28/29)')
    update = int(update)
    root, final_root = Path(root), guard(Path(final_root))
    if final_root.exists():
        raise Blocked('final root exists; never overwrite')
    if not str(authority or '').strip():
        raise Blocked('authority required')
    cfg = runtime.config(root)
    nm_ = cfg.get('newmap') if isinstance(cfg.get('newmap'), dict) else {}
    cell = nm_.get('cell')
    if isinstance(cell, list):                 # clean5 p20 CONFIGs: the list of training cell keys
        same_cell = cell == [panel['cell']['cell_key']]
    else:
        try:
            same_cell = _cell_key(cell or {}) == _cell_key(panel['cell'])
        except (KeyError, TypeError, ValueError, AttributeError, StopIteration):
            same_cell = False
    if nm_.get('map') != MAP or not same_cell:
        raise Blocked(f'training CONFIG map/cell {nm_.get("map")}/{nm_.get("cell")} is not the panel D2800_p20 cell')
    if nm_.get('base_controller') not in BASES:
        raise Blocked('training CONFIG newmap.base_controller unknown')
    exposure = exposure_check(root, cfg)
    view = runtime.plant_view(cfg)
    final_root.mkdir(parents=True)
    if update == 12 and label in IDENTITY_LABELS:
        identity = dict(arm='U0', worlds=IDENTITY_WORLDS,
                        rule=('U0 (frozen d9042cc5 + the training root zero-residual u0) on this training plant for the '
                              'first 2 panel worlds must equal the bank-root U0 records: |J diff| <= 1e-6 veh-s and '
                              'identical safety counters; otherwise the '
                              f'{label} - U0 contrast is reported as plant-identity FAILED (not a claim)'))
        phases = {TRAINED_PHASE: TRAINED_CAP_SIM_S, IDENTITY_PHASE: IDENTITY_CAP_SIM_S}
    else:
        identity = dict(inherited=('the plant-identity verdict of the u12 identity root (C5D or F8B) with the same '
                                   'training CONFIG digest (same plant and source; no new identity episodes)'))
        phases = {TRAINED_PHASE: TRAINED_CAP_SIM_S}
    plan = seal(dict(schema=SCHEMA+'.trained_plan', created=datetime.now(timezone.utc).isoformat(),
                     amendment=_amendment(update),
                     authority=str(authority), label=label, description=TRAINED_ARMS[label], setting=SETTING,
                     training_root=str(root), config_sha256=digest(cfg), config_file_sha256=file_sha(root/'CONFIG.json'),
                     source_root=cfg.get('source_root'), source_manifest_sha256=cfg.get('source_manifest_sha256'),
                     newmap=deepcopy(nm_), training_eval_panel_note=('the training CONFIG binds its own panel sha; the '
                                                                     'p20c panel is new (amendment 28), cell checked'),
                     plant_view=view, exposure=exposure, panel_sha256=panel['panel_sha256'],
                     update=update, arm=TRAINED_ARM_LETTER[label], phases=phases, identity=identity,
                     sealed=('F12: no wave is opened, tabulated or summarised until confirm-analyse after every '
                             'registered arm is published')), 'receipt_sha256')
    publish(final_root/TRAINED_PLAN, plan)
    return plan


def _ended_training_health(root):
    """Amendment 28: the training ENDS at u12 (registered owner/MAIN stop), so a durable STOP is expected; a FAULT or a
    safety pause is not. Returns the stop files present (name + sha) for the binding receipt."""
    root = Path(root)
    for bad in ('FAULT.json', 'PAUSE_SAFETY.json'):
        if (root/bad).exists():
            raise Blocked(f'{bad} in the training root: diagnose before evaluation')
    return {p.name: file_sha(p) for p in sorted(root.glob('*STOP*.json'))}


class _EndedTraining:
    """The runtime with the running-training health check replaced by _ended_training_health (gate otherwise
    unchanged: u12 receipts/checkpoints, monitors u1..u12 bound to the CONFIG, safety clean, frozen A == u0)."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def health(self, root):
        _ended_training_health(root)


def bind_trained(*, final_root, runtime, panel, authority):
    """Zero physics, after the u12 export: the registered gate on the training root at u12 (A/B/R update receipts and
    checkpoints, monitors u1..u12 bound to the CONFIG, safety clean, frozen A == u0), then seal the B u12 checkpoint
    and evaluation-layer sha. Never rebound."""
    _need_p20c()
    final_root = Path(final_root)
    plan = trained_plan(final_root)
    if plan is None:
        raise Blocked(f'{TRAINED_PLAN} missing')
    if trained_bound(final_root) is not None:
        raise Blocked(f'{_bound_name(plan["update"])} exists; never rebound')
    update = int(plan['update'])
    if not str(authority or '').strip():
        raise Blocked('authority required')
    root = Path(plan['training_root'])
    pre = TrainedRuntime(runtime, final_root, TRAINED_PHASE)
    cfg = pre.config(root)
    binding = dict(base=plan['newmap']['base_controller'], panel_sha256=panel['panel_sha256'])
    stops = _ended_training_health(root)
    gated = gate(root, cfg, _EndedTraining(runtime), binding, None, update)
    gated['ended_training'] = stops
    letter = TRAINED_ARM_LETTER[plan['label']]
    side = root/f'{letter}_track2_{update:03}.pt'
    if not side.is_file():
        raise Blocked(f'{side.name} missing (the {letter} evaluation layer of u{update})')
    spec = episode_specs(letter, panel, cfg, root, binding['base'], update=update)[0]['checkpoint']
    if spec['file_sha256'] != gated['checkpoint_sha256'][letter]:
        raise Blocked(f'{letter} checkpoint changed between the gate and the binding')
    value = seal(dict(schema=SCHEMA+'.trained_bound', label=plan['label'], trained_plan_sha256=plan['receipt_sha256'],
                      authority=str(authority), update=update, amendment=_amendment(update), checkpoint=spec,
                      track2_layer=dict(path=str(side), file_sha256=file_sha(side)), gate=gated,
                      bound_utc=datetime.now(timezone.utc).isoformat()), 'receipt_sha256')
    publish(final_root/_bound_name(update), value)
    return value


def trained_specs(panel, bound, label):
    """B episodes of the trained arm from its SEALED binding (no training-root file is needed to rebuild them)."""
    u, letter = int(bound['update']), TRAINED_ARM_LETTER[label]
    return [dict(job_id=f'nmfinal_{label.lower()}_{letter.lower()}_u{u:03}_{i:04}', arm=letter, world_id=r['world_id'],
                 world=deepcopy(r['world']), checkpoint=deepcopy(bound['checkpoint']), sim_s=HORIZON_S,
                 window=dict(WINDOW), noise_seed=r['world']['seed'], deterministic=True,
                 panel_sha256=panel['panel_sha256'], capture=None)
            for i, r in enumerate(panel['worlds'])]


def identity_specs(panel, u0, cfg, root, label):
    specs = episode_specs('U0', panel, cfg, root, 'mappo_frozen', None, u0)[:IDENTITY_WORLDS]
    return [dict(s, job_id=f'nmfinal_{label.lower()}_ident_u0_{i:04}') for i, s in enumerate(specs)]


def _chunk_loop(final_root, waves, *, root, cfg, runtime, workers, chunk, status, step):
    """Chunked dispatch (STOP_PRERUN stops after the chunk in flight; a later call resumes)."""
    stopped, rounds = False, 0

    def coverage():
        return {name: (published(final_root, name, specs, check_episode)[2], len(specs)) for name, _, specs in waves}
    while True:
        cov = coverage()
        pending = [(n, k, s) for n, k, s in waves if cov[n][0] < cov[n][1]]
        if not pending:
            break
        if (final_root/STOP_PRERUN).exists():
            stopped = True
            break
        budget, limits, selection = chunk, {}, []
        for name, kind, specs in pending:
            take = min(cov[name][1]-cov[name][0], budget)
            selection.append((name, kind, specs))
            limits[name] = take
            budget -= take
            if budget == 0:
                break
        run_waves(selection, root=root, final_root=final_root, cfg=cfg, runtime=runtime, workers=workers,
                  base='mappo_frozen', chunked=True, limit=limits)
        rounds += 1
        status(step, 'CHUNK', dict(round=rounds, published={k: v[0] for k, v in coverage().items()}))
    return stopped, rounds, coverage()


def confirm_run(*, final_root, runtime, panel, workers=16, chunk=None):
    """The trained root's dispatch: (1) U0 plant identity on the first 2 panel worlds (phase IDENTITY_PHASE), then
    (2) the bound B u12 on all 192 worlds (phase TRAINED_PHASE), chunked. Waves only; results stay sealed (F12)."""
    _need_p20c()
    final_root = Path(final_root)
    plan, bound = trained_plan(final_root), trained_bound(final_root)
    if plan is None or bound is None:
        raise Blocked(f'{TRAINED_PLAN} and TRAINED_BOUND_u<update>.json (bind-trained after the export) required')
    if plan['panel_sha256'] != panel['panel_sha256']:
        raise Blocked('TRAINED_PLAN binds another panel')
    root, label = Path(plan['training_root']), plan['label']
    status = Status(final_root)
    current = 'CONFIRM'
    try:
        status(current, 'RUNNING', dict(label=label, workers=workers))
        ident = TrainedRuntime(runtime, final_root, IDENTITY_PHASE)
        cfg = ident.config(root)
        runner = ident.runner_status()
        if not runner['ok']:
            raise Blocked(f'runner unavailable: {runner}')
        if ident.physics and not os.environ.get('COOPMWM_MAP_SPEC'):
            raise Blocked('set COOPMWM_MAP_SPEC to the staged wz3to2 MAP_SPEC.json (N1 process map) before physics')
        if ident.physics:
            map_check(bind_map_api(None), panel)
        if file_sha(Path(bound['checkpoint']['path'])) != bound['checkpoint']['file_sha256'] or \
                file_sha(Path(bound['track2_layer']['path'])) != bound['track2_layer']['file_sha256']:
            raise Blocked('bound B u12 checkpoint / evaluation layer changed after the binding')
        update = int(plan['update'])
        u0 = mappo_binding(final_root, 'U0')
        own_identity = update == 12 and label in IDENTITY_LABELS
        if own_identity and (u0 is None or u0['residual_sha256'] != cfg.get('common_checkpoint_sha256')
                             or Path(u0['residual_path']) != root/'A_checkpoint_000.pt'):
            raise Blocked('bind-mappo --arm U0 with --residual-path <training root>/A_checkpoint_000.pt first')
        chunk = int(chunk or 2*workers)
        if chunk < 1:
            raise Blocked('chunk must be >= 1')
        if own_identity:
            s1, r1, cov1 = _chunk_loop(final_root, [(f'{label}_IDENT_U0', 'episode',
                                                     identity_specs(panel, u0, cfg, root, label))],
                                       root=root, cfg=cfg, runtime=ident, workers=min(workers, IDENTITY_WORLDS),
                                       chunk=IDENTITY_WORLDS, status=status, step=current)
        else:                                   # identity inherited from the u12 identity root (same CONFIG)
            s1, r1, cov1 = False, 0, {}
        stopped, rounds, cov2 = s1, r1, {}
        if not s1:
            train = TrainedRuntime(runtime, final_root, TRAINED_PHASE)
            s2, r2, cov2 = _chunk_loop(final_root, [(f'{label}_{TRAINED_ARM_LETTER[label]}_u{update:03}', 'episode',
                                                     trained_specs(panel, bound, label))],
                                       root=root, cfg=cfg, runtime=train, workers=workers, chunk=chunk, status=status,
                                       step=current)
            stopped, rounds = s2, r1+r2
        cov = dict(cov1, **cov2)
        out = dict(schema=SCHEMA+'.confirm_done', status='STOPPED' if stopped else 'PASS', sealed=True, label=label,
                   trained_plan_sha256=plan['receipt_sha256'], trained_bound_sha256=bound['receipt_sha256'],
                   chunk=chunk, rounds=rounds,
                   coverage={k: dict(published=v[0], registered=v[1]) for k, v in cov.items()},
                   charged_sim_s=sum(r['sim_s'] for r in fe.ledger_rows(final_root)))
        status(current, out['status'], out)
        return out
    except BaseException as error:
        status(current, 'FAILED', dict(error=repr(error), traceback=traceback.format_exc()))
        raise


def _wave_records(final_root, name):
    """Sealed records of a wave (whole receipt or contiguous chunk receipts), without rebuilding its specs."""
    final_root = Path(final_root)
    whole = final_root/f'WAVE_{name}.json'
    files = [whole] if whole.is_file() else [p for *_, p in _chunk_files(final_root, name)]
    out = []
    for path in files:
        out += verify(read(path), 'receipt_sha256', path.name)['records']
    return out


def _trained_records(final_root, panel, label):
    plan, bound = trained_plan(final_root), trained_bound(final_root)
    if plan is None or bound is None or plan['label'] != label or plan['panel_sha256'] != panel['panel_sha256']:
        raise Blocked(f'{final_root}: not the sealed {label} trained root of this panel')
    specs = trained_specs(panel, bound, label)
    records, source, covered = published(final_root, f'{label}_{TRAINED_ARM_LETTER[label]}_u{int(bound["update"]):03}',
                                         specs, check_episode)
    ident = _wave_records(final_root, f'{label}_IDENT_U0')
    want = [r['world_id'] for r in panel['worlds'][:IDENTITY_WORLDS]]
    if [r.get('world_id') for r in ident] != want or any(r.get('arm') != 'U0' or r.get('status') != 'graded'
                                                        or r.get('panel_sha256') != panel['panel_sha256'] for r in ident):
        ident = None
    return dict(plan=plan, bound=bound, records=records, source=source, covered=covered, n=len(specs), identity=ident)


def confirm_analyse(*, final_root, panel, trained_roots, amendment='28', identity_roots=None):
    """Zero physics (amendment 28): the registered contrasts on the p20c panel from the bank root (U0, ORACLE_WM,
    LEARNED_TRIGGER) and the two trained roots (C5D, F8B): paired total delay CI95, the group co-criteria, CVaR90;
    the U0 plant-identity check of each trained root. Sealed CONFIRM_TABLES.json in the bank root."""
    _need_p20c()
    if amendment not in CONFIRM_SPECS:
        raise Blocked(f'confirm analysis amendments are {sorted(CONFIRM_SPECS)}')
    spec_ = CONFIRM_SPECS[amendment]
    final_root = Path(final_root)
    plan = prerun_plan(final_root)
    if plan is None or plan['panel_sha256'] != panel['panel_sha256']:
        raise Blocked('bank root PRERUN_PLAN missing or binds another panel')
    plant = final_root/PRERUN_PLANT
    cfg = read(plant/'CONFIG.json')
    u0, oracle = mappo_binding(final_root, 'U0'), oracle_binding(final_root)
    if u0 is None or oracle is None:
        raise Blocked('confirm analysis needs the bank-root U0 binding and its sealed ORACLE_BINDING')
    worlds = [r['world_id'] for r in panel['worlds']]
    bootstrap = dict(panel['contract']['bootstrap'])
    have, sources, missing, roots = {}, {}, {}, {}
    for arm in spec_['bank_arms']:
        specs = episode_specs(arm, panel, cfg, plant, 'mappo_frozen',
                              oracle if arm in BANK_ARMS else mappo_binding(final_root, arm) if arm == CT_ARM else None,
                              u0)
        records, source, covered = published(final_root, arm, specs, check_episode)
        if covered < len(specs):
            missing[arm] = f'{covered}/{len(specs)} published'
            continue
        have[arm], sources[arm], roots[arm] = {r['world_id']: r for r in records}, source, final_root
    identity = {}
    loaded_updates = {}
    for label in spec_['labels']:
        path = (trained_roots or {}).get(label)
        if path is None:
            missing[label] = 'trained root not given'
            identity[label] = dict(verdict='UNAVAILABLE', reason='trained root not given')
            continue
        got = _trained_records(Path(path), panel, label)
        k = int(got['plan']['update'])
        if amendment == '29' and spec_['update'] == 24 and 13 <= k <= 24:
            loaded_updates[label] = k     # disk-safety stop: each training run's own last sealed update (disclosed)
            if 'C5D' in loaded_updates and 'C5DR' in loaded_updates and loaded_updates['C5D'] != loaded_updates['C5DR']:
                raise Blocked('C5D and C5DR come from one training run: their evaluated updates must be equal')
        elif k != spec_['update']:
            raise Blocked(f'{label}: amendment {amendment} evaluates u{spec_["update"]}, root is u{got["plan"]["update"]}')
        if spec_['update'] != 12 or label not in IDENTITY_LABELS:   # inherited from the u12 identity root
            src = (identity_roots or {}).get(label)
            if src is None:
                got['identity'] = None
            else:
                base_label = trained_plan(Path(src))['label']
                base = _trained_records(Path(src), panel, base_label)
                if base['plan']['config_sha256'] != got['plan']['config_sha256'] or int(base['plan']['update']) != 12 \
                        or base_label not in IDENTITY_LABELS:
                    raise Blocked(f'{label}: identity root is not a u12 identity root of the same training CONFIG')
                got['identity'] = base['identity']
        if got['covered'] < got['n']:
            missing[label] = f'{got["covered"]}/{got["n"]} published'
        else:
            have[label], sources[label], roots[label] = ({r['world_id']: r for r in got['records']}, got['source'],
                                                         Path(path))
        ref, rows = have.get('U0'), got['identity']
        if rows is None or ref is None:
            identity[label] = dict(verdict='UNAVAILABLE', reason='identity wave or bank-root U0 missing')
        else:
            cmp_ = [dict(world_id=r['world_id'], j_trained=r['J_veh_s'], j_bank=ref[r['world_id']]['J_veh_s'],
                         safety_equal=r['safety'] == ref[r['world_id']]['safety'],
                         vehicle_log_equal=r['vehicle_log']['sha256'] == ref[r['world_id']]['vehicle_log']['sha256'])
                    for r in rows]
            ok = all(abs(c['j_trained']-c['j_bank']) <= 1e-6 and c['safety_equal'] for c in cmp_)
            identity[label] = dict(verdict='PASS' if ok else 'FAILED', worlds=cmp_,
                                   rule='|J diff| <= 1e-6 veh-s and identical safety counters (vehicle-log sha reported)')
    reports = {}
    for arm in have:
        rows = {w: episode_log_report(roots[arm], have[arm][w]) for w in worlds}
        if any(r is None or not r['ok'] for r in rows.values()):
            raise Blocked(f'{arm}: vehicle logs missing or invalid on this host')
        reports[arm] = rows
    contrasts = []
    for arm, ref, role in spec_['contrasts']:
        row = dict(treatment=arm, reference=ref, role=role,
                   convention='diff = treatment - reference; negative = treatment has less delay')
        if arm not in have or ref not in have:
            row.update(status='unavailable', missing={a: missing.get(a) for a in (arm, ref) if a not in have})
        else:
            row.update(_bank_compare(have, reports, worlds, arm, ref, bootstrap))
            row['ci95_upper_below_0'] = row['total_lower']
            if ref == 'U0' and arm in TRAINED_ARMS:
                row['plant_identity'] = identity.get(arm, {}).get('verdict')
        contrasts.append(row)
    primary = contrasts[0]
    arms = {a: dict(label=TRAINED_ARMS.get(a) or dict(ORACLE_LABELS).get(a)
                    or (MAPPO_KINDS[CT_ARM]['u0_continued_d2800_p20_96ep'] if a == CT_ARM else None)
                    or 'U0 (frozen MAPPO d9042cc5 + u0)',
                    mean_J_veh_s=float(np.mean([have[a][w]['J_veh_s'] for w in worlds])),
                    safety_totals={k: int(sum(have[a][w]['safety'][k] for w in worlds)) for k in SAFETY})
            for a in have}
    for a in [x for x in CONFIRM_BANK_ARMS if x in spec_['bank_arms']]:
        if a in have:
            receipts = [have[a][w].get('oracle') or have[a][w].get('learned_trigger') or {} for w in worlds]
            arms[a]['activations'] = dict(worlds_with_any=sum(1 for r in receipts if r.get('activations')),
                                          total=sum(int(r.get('activations') or 0) for r in receipts))
    value = seal(dict(schema=SCHEMA+'.confirm_tables', amendment=amendment, setting=SETTING,
                      update=(spec_['update'] if not loaded_updates or set(loaded_updates.values()) == {24}
                              else dict(loaded_updates)),
                      stop_rule=(None if not loaded_updates or set(loaded_updates.values()) == {24} else
                                 'MAIN 2026-11-09 12:23 disk-safety stop: amendment 29 evaluates the last sealed '
                                 'update instead of u24 (' + ', '.join(f'{a}=u{u}' for a, u in loaded_updates.items())
                                 + '; disclosed)'),
                      panel_sha256=panel['panel_sha256'], n_worlds=len(worlds),
                      primary_rule=spec_['rule'],
                      primary=dict(contrast=f'{spec_["contrasts"][0][0]} - {spec_["contrasts"][0][1]}',
                                   verdict=('unavailable' if primary.get('status') == 'unavailable' else
                                            'PASS' if primary['ci95_upper_below_0'] else 'FAIL')),
                      contrasts=contrasts, plant_identity=identity, arms=arms, bootstrap=bootstrap,
                      oracle_binding_sha256=oracle['receipt_sha256'], u0_binding_sha256=u0['binding_sha256'],
                      sources={k: v.get('receipt_sha256') if isinstance(v, dict) else v for k, v in sources.items()},
                      roots={k: str(v) for k, v in roots.items()}, missing=missing,
                      pooling=spec_['pooling']),
                 'analysis_sha256')
    target = final_root/spec_['file']
    if target.exists():
        old = verify(read(target), 'analysis_sha256', target.name)
        if old['analysis_sha256'] != value['analysis_sha256']:
            publish(final_root/f'{target.stem}_{int(time.time())}.json', value)
        return value
    publish(target, value)
    return value


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('register-panel', 'register-panel-3c', 'verify-panel', 'quote', 'check', 'register',
                                       'bind-arms', 'prerun-register', 'prerun-logcheck', 'prerun-seqcheck',
                                       'prerun-amend-cap', 'authorize-replay', 'prerun-run', 'parity', 'env-equivalence',
                                       'prerun-envparity', 'envparity-assess', 'host-identity',
                                       'bind-mappo', 'declare-unavailable', 'run', 'analyse', 'dry-run',
                                       'bind-oracle', 'bank-run', 'bank-analyse',
                                       'confirm-register', 'bind-trained', 'confirm-run', 'confirm-analyse'))
    p.add_argument('--setting', choices=('p60',)+tuple(SETTINGS), default=os.environ.get('NM_SETTING', 'p60'),
                   help='evaluation setting (amendment 22): p60 = the registered default; p20 = the 20%% CAV panels')
    p.add_argument('--root', default=None)
    p.add_argument('--final-root', default=None)
    p.add_argument('--panel', default=None)
    p.add_argument('--map-spec', help='register-panel: wz3to2 MAP_SPEC.json (default: COOPMWM_MAP_SPEC or the source tree)')
    p.add_argument('--base', choices=BASES, help='quote/register/dry-run: base controller if CONFIG lacks it')
    p.add_argument('--map-api', help='map API module with seed_ranges/descriptor/world/check_world (default: N1)')
    p.add_argument('--runner', default=RUNNER)
    p.add_argument('--allow-missing', default='')
    p.add_argument('--workers', type=int, default=24)
    p.add_argument('--poll', type=int, default=60)
    p.add_argument('--replicated-wait', type=int, default=0, help='seconds to wait for the clean4 replicated u24 receipt')
    p.add_argument('--out', help='dry-run: empty output directory')
    p.add_argument('--mappo-path', help='bind-mappo: MAPPO executor checkpoint (N7 retrain or the d9042cc5 fallback)')
    p.add_argument('--mappo-sha256', help='bind-mappo: its full sha256 (verified now and at every load)')
    p.add_argument('--arm', choices=BOUND_ARMS, default='MAPPO_SPEED', help='bind-mappo/declare-unavailable: U0 or a comparator')
    p.add_argument('--a-sha', help='bind-arms: A checkpoint sha256 at --update (optional: A = U0 alias when frozen)')
    p.add_argument('--b-sha', help='bind-arms: B checkpoint sha256 at --update')
    p.add_argument('--r-sha', help='bind-arms: R checkpoint sha256 at --update')
    p.add_argument('--update', type=int, choices=TRAINED_UPDATES,
                   help='check/bind-arms/run/analyse: the evaluated update (12 = PRIMARY, 24 = SECONDARY); required')
    p.add_argument('--mirror-config-sha256', help='check/register/run/analyse/parity: the clean run root at --root is a '
                                                  'read-only mirror on this host; the registered CONFIG.json file sha256')
    p.add_argument('--mappo-kind', choices=sorted({k for v in MAPPO_KINDS.values() for k in v}),
                   help='bind-mappo: retrained_wz3to2 | u0_transferred_not_retrained (MAPPO_SPEED), lane_retrained_wz3to2 (MAPPO_LANE)')
    p.add_argument('--reason', help='declare-unavailable: why the comparator cannot be evaluated')
    p.add_argument('--residual-path', help='bind-mappo --arm U0: the common u0 A_checkpoint_000.pt (pre-run: '
                                           '<final_root>/prerun_plant/A_checkpoint_000.pt)')
    p.add_argument('--residual-sha256', help='bind-mappo --arm U0: its sha256 (= clean5a common_checkpoint_sha256)')
    p.add_argument('--plant-config', help='prerun-register: the clean5a CONFIG.json (plant source for NATIVE/U0/RULE)')
    p.add_argument('--u0-path', help='prerun-register: the common u0 checkpoint (A_checkpoint_000.pt)')
    p.add_argument('--u0-sha256', help='prerun-register: its sha256')
    p.add_argument('--train-src', help='bind-mappo --arm MAPPO_LANE: the S-MAPPOWZL training source tree (amendment 17)')
    p.add_argument('--train-config', help="bind-mappo (MAPPO_SPEED/MAPPO_LANE): the comparator's own training run "
                                          'config (N7 toml) used by the F11 training-env path')
    p.add_argument('--host', help='host-identity: label of this host (e.g. cpuW)')
    p.add_argument('--reference', help='host-identity: previous host plant start receipt (default envequiv/start_plant.json)')
    p.add_argument('--bank', help='bind-oracle: the common bank doc (ORACLE_WM, LEARNED_TRIGGER, applicability labels)')
    p.add_argument('--oracle-bank', help='bind-oracle: the ORACLE (union) bank doc override (amendment 27)')
    p.add_argument('--model', help='bind-oracle: the LEARNED_TRIGGER model doc (sha 8e0ddd3e)')
    p.add_argument('--bank-arms', default=','.join(BANK_ARMS), help='bank-run: comma-separated subset of the bank arms')
    p.add_argument('--label', choices=tuple(TRAINED_ARMS), help='amendment 28: trained arm label (confirm-register)')
    p.add_argument('--amendment', default='28', choices=('28', '29', '30'),
                   help='confirm-analyse: 28 (u12), 29 (u24 / disk-stop k) or 30 (MAPPO_CT)')
    p.add_argument('--identity-roots', help='confirm-analyse: label=<u12 identity root> for labels without their own '
                                            'identity episodes (C5DR=<C5D u12 root>; amendment 29: every label)')
    p.add_argument('--trained-roots', help='confirm-analyse: C5D=<trained root>,F8B=<trained root> (this host)')
    p.add_argument('--zero-sumo-only', action='store_true', help='host-identity: skip the 420 s replay')
    p.add_argument('--cap', type=int, help='prerun-amend-cap: the new pre-run sub-cap (MAIN decision)')
    p.add_argument('--chunk', type=int, help='prerun-run: episodes per dispatch (default 2 x workers); STOP_PRERUN '
                                             'in the final root stops after the chunk in flight')
    p.add_argument('--authority', help='bind-mappo: verbatim message/decision (e.g. N7 hand-off or MAIN fallback decision)')
    p.add_argument('--comparator-wait', type=int, default=7*86400,
                   help='run: seconds to wait (after the u24 gate) for every comparator to be bound or declared')
    a = p.parse_args(argv)
    apply_setting(a.setting)
    a.root = a.root or ROOT
    a.final_root = a.final_root or FINAL_ROOT
    a.panel = a.panel or str(PANEL_FILE)
    if a.root is None or a.final_root is None:
        p.error(f'setting {a.setting}: pass --root and --final-root explicitly')
    if EVAL_EXTRA_RANGE is not None:
        from tools.coopmwm import map_scenarios
        from tools.coopmwm.nm_eval_runner import install_eval_ranges
        install_eval_ranges(map_scenarios)
    if a.command in ('check', 'bind-arms', 'run', 'analyse') and a.update is None:
        p.error('--update 12 (PRIMARY) or 24 (SECONDARY) required')

    def runtime_():
        return MirrorRuntime(a.runner, a.mirror_config_sha256) if a.mirror_config_sha256 else Runtime(a.runner)
    if a.workers < 1 or a.workers > MAX_WORKERS:
        p.error(f'workers must be 1..{MAX_WORKERS} (cpuZ 48 cores)')
    if a.command == 'register-panel':
        path = guard(Path(a.panel))
        if path.exists():
            p.error('panel file exists; never overwrite a registration')
        value = build_panel(a.map_spec or default_spec_path(), datetime.now(timezone.utc).isoformat())
        publish(path, value)
        result = dict(panel_sha256=value['panel_sha256'], file_sha256=file_sha(path), path=str(path))
    elif a.command == 'register-panel-3c':
        path = guard(PANEL_3C_FILE)
        if path.exists():
            p.error('claim-3c panel exists; never overwrite a registration')
        value = build_panel_3c(a.map_spec or default_spec_path(), datetime.now(timezone.utc).isoformat())
        publish(path, value)
        result = dict(panel_sha256=value['panel_sha256'], file_sha256=file_sha(path), path=str(path))
    elif a.command == 'verify-panel':
        value = load_panel(a.panel)
        result = dict(status='PASS', panel_sha256=value['panel_sha256'], file_sha256=file_sha(a.panel),
                      worlds=len(value['worlds']), seeds=value['seeds_used'], cell=value['cell'],
                      decision_worlds=len(value['contract']['decisions']['worlds']))
    elif a.command == 'quote':
        result = {b: quote(b) for b in ([a.base] if a.base else BASES)}
    else:
        panel = load_panel(a.panel)
        if a.command == 'dry-run':
            result = dry_run(out=a.out, panel=panel, base=a.base or 'mappo_frozen')
        elif a.command == 'check':
            result = check(root=a.root, final_root=a.final_root, runtime=runtime_(), panel=panel,
                           map_module=a.map_api, update=a.update)
        elif a.command == 'register':
            result = register(root=a.root, final_root=a.final_root, runtime=runtime_(), panel=panel, base=a.base)
        elif a.command == 'bind-mappo':
            check_ = None
            try:
                from tools.coopmwm.nm_eval_runner import mappo_load_check
                check_ = mappo_load_check(a.mappo_path, a.mappo_sha256)
            except ImportError as error:
                check_ = dict(status='not run', reason=repr(error))
            result = bind_mappo(final_root=a.final_root, path=a.mappo_path, sha256=a.mappo_sha256, kind=a.mappo_kind,
                                arm=a.arm, authority=a.authority, check=check_, residual_path=a.residual_path,
                                residual_sha256=a.residual_sha256, train_config=a.train_config, train_src=a.train_src)
        elif a.command == 'parity':
            report, _ = run_parity(root=a.root, final_root=a.final_root, runtime=runtime_(), panel=panel,
                                   arm=a.arm, workers=min(a.workers, 8))
            result = dict(verdict=report['verdict'], receipt_sha256=report['receipt_sha256'],
                          worlds={k: v['compare'] for k, v in report['worlds'].items()})
        elif a.command == 'prerun-register':
            result = prerun_register(final_root=a.final_root, plant_config=a.plant_config, u0_path=a.u0_path,
                                     u0_sha256=a.u0_sha256, runtime=Runtime(a.runner), panel=panel,
                                     panel_3c=load_panel_3c(), authority=a.authority or PRERUN_AUTHORITY)
        elif a.command == 'prerun-logcheck':
            value = prerun_logcheck(final_root=a.final_root, runtime=Runtime(a.runner), panel=panel)
            result = dict(verdict=value['verdict'], receipt_sha256=value['receipt_sha256'],
                          arms={k: v['check'] for k, v in value['arms'].items()})
        elif a.command == 'prerun-seqcheck':
            value = prerun_seqcheck(final_root=a.final_root, runtime=Runtime(a.runner), panel=panel)
            result = dict(verdict=value['verdict'], receipt_sha256=value['receipt_sha256'], pids=value['pids'],
                          arms={k: v['check'] for k, v in value['arms'].items()})
        elif a.command == 'env-equivalence':
            value = env_equivalence(final_root=a.final_root, panel=panel, runner=a.runner)
            result = {k: value[k] for k in ('verdict', 'differences', 'module_differences', 'env_class', 'receipt_sha256')}
        elif a.command == 'host-identity':
            value = host_identity(final_root=a.final_root, runtime=Runtime(a.runner), panel=panel, host=a.host,
                                  runner=a.runner, sumo=not a.zero_sumo_only, reference=a.reference)
            result = dict(verdict=value['verdict'], receipt_sha256=value['receipt_sha256'],
                          zero_sumo_identical=value['zero_sumo']['identical'],
                          other_modules_changed=value['zero_sumo']['other_modules_changed'],
                          receipt_differences=value['zero_sumo']['receipt_differences'][:20],
                          replay_identical=value['replay'].get('identical'))
        elif a.command == 'envparity-assess':
            value = envparity_assess_17c(a.final_root)
            result = {k: value[k] for k in ('accepted', 'worlds', 'receipt_sha256')}
        elif a.command == 'prerun-envparity':
            value = prerun_envparity(final_root=a.final_root, runtime=Runtime(a.runner), panel=panel)
            result = {k: value[k] for k in ('verdict', 'worlds', 'receipt_sha256')}
        elif a.command == 'prerun-amend-cap':
            result = amend_prerun_cap(final_root=a.final_root, cap_sim_s=a.cap, authority=a.authority)
        elif a.command == 'authorize-replay':
            result = authorize_replay(final_root=a.final_root, authority=a.authority, reason=a.reason)
        elif a.command == 'prerun-run':
            import fcntl
            if a.workers > 16 and os.environ.get('NM_PRERUN_HOST') == 'cpuV':
                p.error('cpuV pre-run: <= 16 workers (MAIN)')
            with (Path(a.final_root)/'OWNER.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                publish(Path(a.final_root)/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()),
                        replace=True)
                result = prerun_execute(final_root=a.final_root, runtime=Runtime(a.runner), panel=panel,
                                        panel_3c=load_panel_3c(), workers=a.workers, chunk=a.chunk)
        elif a.command == 'bind-oracle':
            value = bind_oracle(final_root=a.final_root, bank_path=a.bank, model_path=a.model, authority=a.authority,
                                banks=dict(ORACLE=a.oracle_bank) if a.oracle_bank else None)
            result = dict(receipt_sha256=value['receipt_sha256'], bank=value['bank'], banks=value['banks'])
        elif a.command == 'confirm-register':
            value = confirm_register(root=a.root, final_root=a.final_root, runtime=Runtime(a.runner), panel=panel,
                                     label=a.label, authority=a.authority, update=a.update or TRAINED_UPDATE)
            result = dict(receipt_sha256=value['receipt_sha256'], label=value['label'],
                          config_sha256=value['config_sha256'], training_root=value['training_root'])
        elif a.command == 'bind-trained':
            value = bind_trained(final_root=a.final_root, runtime=Runtime(a.runner), panel=panel, authority=a.authority)
            result = dict(receipt_sha256=value['receipt_sha256'], label=value['label'],
                          checkpoint_sha256=value['checkpoint']['file_sha256'],
                          track2_layer_sha256=value['track2_layer']['file_sha256'],
                          ended_training=value['gate'].get('ended_training'))
        elif a.command == 'confirm-run':
            import fcntl
            with (Path(a.final_root)/'OWNER.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                publish(Path(a.final_root)/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()),
                        replace=True)
                result = confirm_run(final_root=a.final_root, runtime=Runtime(a.runner), panel=panel,
                                     workers=a.workers, chunk=a.chunk)
        elif a.command == 'confirm-analyse':
            roots = dict(x.split('=', 1) for x in (a.trained_roots or '').split(',') if x)
            ident = dict(x.split('=', 1) for x in (a.identity_roots or '').split(',') if x)
            value = confirm_analyse(final_root=a.final_root, panel=panel, trained_roots=roots,
                                    amendment=a.amendment, identity_roots=ident)
            result = dict(analysis_sha256=value['analysis_sha256'], primary=value['primary'], missing=value['missing'],
                          plant_identity={k: v.get('verdict') for k, v in value['plant_identity'].items()})
        elif a.command == 'bank-analyse':
            value = bank_analyse(final_root=a.final_root, panel=panel)
            result = dict(analysis_sha256=value['analysis_sha256'], missing=value['missing'])
        elif a.command == 'bank-run':
            import fcntl
            arms = tuple(x for x in a.bank_arms.split(',') if x)
            with (Path(a.final_root)/'OWNER.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                publish(Path(a.final_root)/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()),
                        replace=True)
                result = prerun_execute(final_root=a.final_root, runtime=Runtime(a.runner), panel=panel,
                                        panel_3c=None, workers=a.workers, chunk=a.chunk, arms=arms)
        elif a.command == 'bind-arms':
            result = bind_arms(final_root=a.final_root, shas=dict(A=a.a_sha, B=a.b_sha, R=a.r_sha), authority=a.authority,
                               update=a.update)
        elif a.command == 'declare-unavailable':
            result = declare_unavailable(final_root=a.final_root, arm=a.arm, authority=a.authority, reason=a.reason)
        elif a.command == 'analyse':
            value = _analyse_published(a.root, a.final_root, runtime_(), panel, load_panel_3c(), update=a.update)
            result = {k: dict(analysis_sha256=v['analysis_sha256'], present=v.get('present'), missing=v.get('missing'))
                      for k, v in value.items()}
        else:
            import fcntl
            if run_plan(a.final_root) is None:
                p.error('RUN_PLAN.json missing: run register first (the final root also holds the outputs)')
            with (Path(a.final_root)/'OWNER.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                publish(Path(a.final_root)/'LAUNCHER.json', dict(pid=os.getpid(), argv=sys.argv, unix=time.time()),
                        replace=True)
                value = execute(root=a.root, final_root=a.final_root, runtime=runtime_(), panel=panel,
                                panel_3c=load_panel_3c(), update=a.update,
                                allow_missing=tuple(x for x in a.allow_missing.split(',') if x), workers=a.workers,
                                poll=a.poll, replicated_timeout=a.replicated_wait, comparator_timeout=a.comparator_wait,
                                map_module=a.map_api)
                result = {k: dict(analysis_sha256=v['analysis_sha256'], present=v.get('present'), missing=v.get('missing'))
                          for k, v in value.items()}
    print(json.dumps(result, indent=1, sort_keys=True, default=str))


if __name__ == '__main__':
    main()
