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
UPDATE = 24
ARMS = ('A', 'B', 'R')
DECISION_ARMS = ('B', 'R')
BASES = ('mappo_frozen', 'mappo_u0', 'native')   # clean5a: frozen S-MAPPOWZ snapshot base (A = base alone)
ORDER = ('B', 'A', 'R', 'RULE', 'U0', 'MAPPO_SPEED', 'MAPPO_LANE', 'NATIVE')
#: MAIN 2026-11-06 (FAIRAUDIT F6): 7 arms; BASE is an alias (U0 for a mappo_u0 base, NATIVE for native)
EPISODE_ARMS = ('A', 'B', 'R', 'U0', 'RULE', 'MAPPO_SPEED', 'MAPPO_LANE', 'NATIVE')
COMPARATOR_ARMS = ('MAPPO_SPEED', 'MAPPO_LANE')        # claim 3b, one seal per comparator
BOUND_ARMS = ('U0',)+COMPARATOR_ARMS                    # executors bound by path + sha256
CLAIM3A_ARMS = ('A', 'B', 'R', 'U0', 'RULE', 'NATIVE')
#: records of these arms carry the executor sha actually loaded (bound executors)
EXECUTOR_ARMS = ('U0', 'RULE')+COMPARATOR_ARMS
#: MAIN registration (pre-results): RULE = U0 + the solo2 rule always on (one CAV holding in SUMO lane 2
#: with wall_h parameters, cooldown = hold) = the EXACT S-REGIMEU0 implementation (c4/imitreplay
#: tools/coopmwm/regimeu0.py RuleHook('wall_h_solo2')): the hand-designed cooperation baseline.
RULE_STRATEGY = 'wall_h_solo2'
RULE_LABEL = ('RULE: U0 + the S-REGIMEU0 wall_h_solo2 rule (one CAV holds in SUMO lane 2 at 15 m/s for 90 s, '
              'cooldown = hold), always on')
#: arms independent of the clean5a training: pre-run (sealed) before clean5a launches
PRERUN_ARMS = ('NATIVE', 'U0', 'RULE')
PRERUN_PHASE = 'v2v_final_prerun'
#: rule_digest() of the exact S-REGIMEU0 rule (sumo_lane, pick_wall, RuleHook, Recorder, constants);
#: identical at c4/imitreplay 385c4b8f, 52248a4b and d02228cf (blob 2e267403 / imitreplay d5042c33)
RULE_SHA256 = '887285923ca0b4ac1a7f728f3d687c57b49e944d3ec7abf62bddb4249d6690b4'
PRERUN_FILE = 'PRERUN_PLAN.json'
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
CLAIM3C_ARMS = ('B', 'R', 'U0', 'RULE', 'NATIVE')
CLAIM3C_CELL = dict(demand=2400, penetration=0.6)
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
    'MAPPO_LANE': {'lane_retrained_wz3to2_final': 'lane-capable MAPPO retrained on wz3to2 (S-MAPPOWZL final)'}}
#: MAIN 2026-11-06 registered binding: MAPPO-speed = S-MAPPOWZ it168 ONLY (no it84/42/24 substitution);
#: MAPPO-lane = the S-MAPPOWZL final export, never aecd7d70; the transferred d9042cc5 u0 is the U0 arm.
REQUIRED_ITERATION = {'MAPPO_SPEED': 168}
REFUSED_SHA_PREFIXES = {'U0': (), 'MAPPO_SPEED': ('d9042cc5',), 'MAPPO_LANE': ('aecd7d70', 'd9042cc5')}
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
ALLOCATION_SIM_S = 1_340_000           # v2v_final raised by MAIN 2026-11-06 (>= 15% margin over 1,164,960)
ALLOCATION_AUTHORITY = ('MAIN 2026-11-06 under the owner authorisation to act on its recommendations: v2v_final '
                        'raised 690,000 -> 1,040,000 (F6: 7 arms + branches) -> 1,200,000 (RULE arm in 3a and 3c, '
                        'claim 3c) -> 1,340,000 (N5: restore a >= 15% margin over the 1,164,960 ceiling); pre-run '
                        'spend in the evaluation root ledger counts against the same cap')
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
MAP_SPEC_REL = 'assets/research/newmap_candidates/wz3to2/MAP_SPEC.json'
#: MAIN 2026-11-06 (N11 F4): the panel binds the staged probe network bytes (sha256 of wz3to2.net.xml)
NET_SHA256 = 'b62778986347e8c21bca90d7ddcc8d9faeb288290a4dbdc3f53ee64ae8c21dfa'
ROOT = '/home/ubuntu/coopmwm_clean5a_wz3to2'            # cpuZ; pass --root explicitly
FINAL_ROOT = '/home/ubuntu/coopmwm_clean5a_wz3to2_finaleval'
MAX_WORKERS = 46                                          # cpuZ 48 cores (MAIN: run right after u24)
PREFIXES = ('nmfinal_', 'nmdec_', 'nmparity_', 'nm3c_')
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
REPLICATED_FILE = 'track2_diagnostics/replicated_u024_live.json'


def _world(seed, cell):
    """N1's registered world dict (tools.coopmwm.map_scenarios.world) for the panel cell."""
    from tools.coopmwm import map_scenarios
    return map_scenarios.world(int(seed), int(cell['demand_vph']['main']), float(cell['penetration']))


def in_decision_subset(world):
    return world['repeat'] % DECISION_STRIDE == 0


# ---------------------------------------------------------------- budget quote (zero physics)

def quote(base='mappo_u0'):
    """Ceiling sim-s against the clean4 v2v_final allocation: 7 distinct arms for either base
    controller (BASE is an alias), the 8-world decision branches and the F11 parity check."""
    if base not in BASES:
        raise Blocked(f'unknown base {base!r}')
    episode = N_WORLDS*HORIZON_S
    arms = list(EPISODE_ARMS)
    subset = N_WORLDS//DECISION_STRIDE
    need = {f'episodes_{a}': episode for a in arms}
    # clean3 prefix-replay branches are charged from t=0 (sumo_verify.branch_sim_s): <= HORIZON_S each
    need.update({f'decisions_{a}': subset*(2+MAX_MEMBERS)*(DECISION_MAX_T+DECISION_H) for a in DECISION_ARMS})
    need['parity_f11'] = PARITY_SIM_S
    need.update({f'claim3c_{a}': CLAIM3C_WORLDS*HORIZON_S for a in CLAIM3C_ARMS})
    prerun = len(PRERUN_ARMS)*(N_WORLDS+CLAIM3C_WORLDS)*HORIZON_S
    total = sum(need.values())
    core = total-PARITY_SIM_S-len(CLAIM3C_ARMS)*CLAIM3C_WORLDS*HORIZON_S
    return dict(base=base, phase=PHASE, allocation_sim_s=ALLOCATION_SIM_S, allocation_authority=ALLOCATION_AUTHORITY,
                need_ceiling=need, episodes_sim_s=len(arms)*episode,
                claim3c_sim_s=len(CLAIM3C_ARMS)*CLAIM3C_WORLDS*HORIZON_S, prerun_sim_s=prerun,
                prerun_arms=list(PRERUN_ARMS),
                decisions_ceiling_sim_s=core-len(arms)*episode, parity_sim_s=PARITY_SIM_S,
                core_ceiling_sim_s=core, total_ceiling_sim_s=total, slack_sim_s=ALLOCATION_SIM_S-total,
                margin_over_core=round(ALLOCATION_SIM_S/core-1, 4), margin_over_total=round(ALLOCATION_SIM_S/total-1, 4),
                fits=total <= ALLOCATION_SIM_S,
                basis=dict(episodes=f'{len(arms)} arms x {N_WORLDS} worlds x {HORIZON_S} s',
                           decisions=(f'{len(DECISION_ARMS)} arms x {subset} registered worlds x <=1 decision x '
                                      f'(all + background + <={MAX_MEMBERS} solos) x <= {HORIZON_S} s (prefix replay '
                                      'from 0 to t_s+360); only captured decisions are reserved'),
                           parity_f11=(f'{len(PARITY_WORLDS)} TRAIN worlds x 2 paths (eval plant, training env) x '
                                       f'{len(COMPARATOR_ARMS)} comparators x {300+PARITY_ROLLOUT_S} s'),
                           claim3c=(f'{len(CLAIM3C_ARMS)} arms x {CLAIM3C_WORLDS} held-out D2400_p60 worlds x {HORIZON_S} s'),
                           charging='planned sim-s precharged per job; whole wave reserved before dispatch'),
                wall_time=dict(workers=24, sim_s_per_worker_s='6-9 (newmap_probe cpuV: ~148k sim-s in ~22-25 min '
                                                              'on <=12 workers)',
                               hours=[round(total/(24*9)/3600, 2), round(total/(24*6)/3600, 2)]))


# ---------------------------------------------------------------- panel registration (zero physics)

def build_panel(spec_path, registered_utc, *, demand=CELL['demand'], penetration=CELL['penetration']):
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
                 chosen=N_WORLDS, reason=('largest round panel inside the registered 100-seed range; same 96-world '
                                          'shape as the clean3 registered estimator; 4 seeds reserved unused'))
    panel = dict(
        schema=SCHEMA+'.panel', version=2, map=MAP,
        role='evaluation only (registered by N6; never training, never selection)',
        registered_utc=registered_utc, authority='newmap_spec.md (MAIN 2026-11-06): HELD-OUT eval 989600-989699',
        supersedes=[dict(panel_sha256=k, reason=v) for k, v in sorted(SUPERSEDED.items())],
        seed_ranges=dict(evaluation=list(EVAL_RANGE), training=list(TRAIN_RANGE)),
        seeds_used=[seeds[0], seeds[-1]], seeds_reserved_unused=[seeds[-1]+1, EVAL_RANGE[1]],
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
                     'tools/coopmwm, traffic_shepherd, ops code'],
            disclosed=('integers 989600-989635 are the OLD merge-map S-COOPMWM final_test worlds '
                       '(tools/coopmwm/protocol.py FINAL_START); wz3to2 worlds are map-namespaced, so no wz3to2 '
                       'traffic realisation shares draws with them; job/world ids carry the wz3to2_ prefix')),
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


def load_panel(path=PANEL_FILE, *, pinned=PANEL_SHA256):
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
                authority='MAIN registration (pre-results): claim 3c, condition-aware: no harm in light traffic')
    seeds = list(range(EVAL_RANGE[0], EVAL_RANGE[0]+CLAIM3C_WORLDS))
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
    panel = dict(schema=SCHEMA+'.panel', version='claim3c', map=MAP, role='evaluation only (claim 3c)',
                 registered_utc=registered_utc, authority='MAIN registration 2026-11-06 (claim 3c)',
                 seed_ranges=v2['seed_ranges'], seeds_used=[seeds[0], seeds[-1]], cell=cell,
                 map_binding=v2['map_binding'], stream_rule=v2['stream_rule'], exposure=v2['exposure'],
                 sibling_panel_sha256=v2['panel_sha256'], worlds=worlds, contract=contract, job_prefix='nm3c',
                 wave_prefix='C3_')
    return seal(json.loads(json.dumps(panel)), 'panel_sha256')


def load_panel_3c(path=PANEL_3C_FILE, *, pinned=PANEL_3C_SHA256):
    panel = verify(read(path), 'panel_sha256', 'claim-3c panel seal')
    if pinned is not None and panel['panel_sha256'] != pinned:
        raise Blocked(f'claim-3c panel_sha256 {panel["panel_sha256"]} != pinned {pinned}')
    seeds = [r['world']['seed'] for r in panel['worlds']]
    if seeds != list(range(EVAL_RANGE[0], EVAL_RANGE[0]+CLAIM3C_WORLDS)) or panel['map_binding']['net_sha256'] != NET_SHA256:
        raise Blocked('claim-3c panel differs from its registration')
    for r in panel['worlds']:
        if r['world'] != _world(r['world']['seed'], panel['cell']) or r['world']['cell_key'] != 'D2400_p60':
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
        yield from (int(m) for m in SEED_TOKEN.findall(value))


def exposure_check(root, cfg):
    """No evaluation-range seed may appear as a world/job in the clean4 CONFIG, its explicit world
    registration or its ledger before this launcher (declared ``*range*`` values, the panel binding
    and this launcher's own job ids are allowed)."""
    found = dict(config=sorted(set(_seed_hits(cfg))),
                 ledger=sorted({int(m) for r in fe.ledger_rows(root) if not str(r.get('id', '')).startswith(PREFIXES)
                                for m in SEED_TOKEN.findall(str(r.get('id', '')))}))
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


def budget_check(root, cfg, base, *, outstanding=None, final_root=None):
    """Zero physics: the registered allocation covers this launcher's ceiling (pre-run spend in the
    evaluation root ledger included); v2v_final carries no foreign spend."""
    source, allocated = allocation(cfg)
    q = quote(base)
    rows = [r for r in fe.ledger_rows(root) if r.get('phase') == PHASE]
    prerun = sum(r['sim_s'] for r in fe.ledger_rows(final_root)) if final_root is not None and Path(final_root).is_dir() else 0
    mine = sum(r['sim_s'] for r in rows if str(r['id']).startswith(PREFIXES))+prerun
    foreign = [r['id'] for r in rows if not str(r['id']).startswith(PREFIXES)]
    cap = min(allocated or 0, ALLOCATION_SIM_S)
    need = max(0, q['total_ceiling_sim_s']-mine) if outstanding is None else outstanding
    out = dict(source=source, phase=PHASE, allocated=allocated, sub_cap=cap, spent_by_launcher=mine,
               foreign_rows=foreign[:5], foreign_count=len(foreign), ceiling=q['total_ceiling_sim_s'],
               outstanding_ceiling=need, remaining=cap-mine, prerun_spent=prerun)
    if allocated is None or allocated < ALLOCATION_SIM_S:
        raise Blocked(f'clean4 {PHASE} allocation {allocated} < registered {ALLOCATION_SIM_S}: {out}')
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
    if arm == 'MAPPO_LANE' and not meta.get('lane_head'):
        raise Blocked('MAPPO_LANE must carry its lane head')
    return meta


def _wave_started(final_root, arm):
    root = Path(final_root)
    return ((root/f'RESERVED_{arm}.json').is_file() or (root/f'WAVE_{arm}.json').is_file()
            or any(root.glob(f'RESERVED_{arm}__*.json')) or any(root.glob(f'WAVE_{arm}__*.json')))


def bind_mappo(*, final_root, path, sha256, kind, authority, arm='MAPPO_SPEED', check=None, dry_run=False,
               residual_path=None, residual_sha256=None):
    """Bind a comparator executor (path + sha256, re-verified at every load). Rebinding is allowed
    only before that arm's wave is reserved; earlier bindings are kept."""
    final_root, arm = Path(final_root), _arm(arm)
    if run_plan(final_root) is None and prerun_plan(final_root) is None:
        raise Blocked('RUN_PLAN.json / PRERUN_PLAN.json missing: run register or prerun-register first')
    residual = None
    if arm == 'U0':
        if not residual_path or not re.fullmatch('[0-9a-f]{64}', str(residual_sha256 or '')):
            raise Blocked('U0 needs the zero-residual u0 checkpoint: --residual-path and --residual-sha256')
        if file_sha(residual_path) != residual_sha256:
            raise Blocked(f'{residual_path} does not hash to {residual_sha256}')
        residual = dict(residual_path=str(residual_path), residual_sha256=residual_sha256)
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


def bind_arms(*, final_root, shas, authority):
    """MAIN: bind clean5a A/B/R u24 by sha before the evaluation; the gate refuses any other file."""
    final_root = Path(final_root)
    if run_plan(final_root) is None:
        raise Blocked('RUN_PLAN.json missing: run register first')
    if set(shas) != set(ARMS) or not all(re.fullmatch('[0-9a-f]{64}', str(v)) for v in shas.values()):
        raise Blocked('A, B and R u24 checkpoint sha256 required')
    if not str(authority).strip():
        raise Blocked('authority required')
    if (final_root/'ARMS_BOUND.json').exists() or (final_root/'GATE_PASS.json').exists():
        raise Blocked('A/B/R already bound (ARMS_BOUND or GATE_PASS exists); never rebound')
    value = seal(dict(schema=SCHEMA+'.arms', update=UPDATE, checkpoint_sha256=dict(shas), authority=str(authority),
                      bound_utc=datetime.now(timezone.utc).isoformat()), 'binding_sha256')
    publish(final_root/'ARMS_BOUND.json', value)
    return value


# ---------------------------------------------------------------- gate (zero physics)

def gate(root, cfg, runtime, binding, final_root=None):
    """clean4 u24 sealed AND safety clean. NotReady = wait; Blocked = never proceed automatically."""
    root = Path(root)
    missing = [n for arm in ARMS for n in (f'{arm}_UPDATE_{UPDATE:03}.json', f'{arm}_checkpoint_{UPDATE:03}.pt',
                                           f'{arm}_COMPLETE.json') if not (root/n).is_file()]
    if not (root/f'LEARNING_MONITOR_u{UPDATE:03}.json').is_file():
        missing.append(f'LEARNING_MONITOR_u{UPDATE:03}.json')
    try:
        runtime.health(root)
    except Exception as error:   # registered health: FAULT / PAUSE_SAFETY / STOP / unreleased PAUSE_*
        raise NotReady(f'clean4 health not clear: {error!r}')
    if missing:
        raise NotReady('u24 not sealed: missing '+', '.join(missing))
    updates = {arm: verify(read(root/f'{arm}_UPDATE_{UPDATE:03}.json'), 'update_sha256', f'{arm} u24 update')
               ['update_sha256'] for arm in ARMS}
    bound, events, pauses = digest(cfg), {}, []
    for u in range(1, UPDATE+1):
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
    checkpoints = {arm: file_sha(root/f'{arm}_checkpoint_{UPDATE:03}.pt') for arm in ARMS}
    bound_arms = Path(final_root)/'ARMS_BOUND.json' if final_root is not None else None
    if bound_arms is not None and bound_arms.is_file():
        expected = verify(read(bound_arms), 'binding_sha256', 'ARMS_BOUND')['checkpoint_sha256']
        if expected != checkpoints:
            raise Blocked(f'u24 checkpoints differ from the bound A/B/R shas: {expected} vs {checkpoints}')
    if binding['base'] == 'mappo_u0' and file_sha(root/'A_checkpoint_000.pt') != cfg['common_checkpoint_sha256']:
        raise Blocked('A_checkpoint_000.pt is not the registered common MAPPO u0')
    return dict(schema=SCHEMA+'.gate', status='PASS', root=str(root), config_sha256=bound, update_sha256=updates,
                checkpoint_sha256=checkpoints, monitors=UPDATE, collisions=0, teleports=0, base=binding['base'],
                panel_sha256=binding['panel_sha256'], unix=time.time(), new_sim_s=0)


# ---------------------------------------------------------------- specs and records

def episode_specs(arm, panel, cfg, root, base, binding=None, u0=None):
    root = Path(root)
    if arm in ARMS:
        path = root/f'{arm}_checkpoint_{UPDATE:03}.pt'
        checkpoint = dict(path=str(path), file_sha256=file_sha(path), update=UPDATE, arm=arm)
    elif arm in ('U0', 'RULE', 'MAPPO_SPEED'):
        if u0 is None:
            raise Blocked(f'{arm} needs the U0 binding (frozen executor + zero-residual u0): bind-mappo --arm U0')
        exe = u0 if arm in ('U0', 'RULE') else binding
        if exe is None:
            raise Blocked(f'{arm} executor not bound (bind-mappo --arm {arm})')
        # executor = the bound frozen/comparator export; residual = the bound zero-residual u0
        checkpoint = dict(kind=dict(U0='u0', RULE='rule', MAPPO_SPEED='mappo_comparator')[arm],
                          comparator_kind=exe['kind'], label=exe['label'], executor_path=exe['path'],
                          executor_sha256=exe['sha256'], binding_sha256=exe['binding_sha256'],
                          binding_path=exe['binding_path'], path=u0['residual_path'], file_sha256=u0['residual_sha256'],
                          update=0, arm='A')
        if arm == 'RULE':
            checkpoint.update(rule=RULE_STRATEGY, rule_sha256=RULE_SHA256, rule_module='tools.coopmwm.regimeu0',
                              rule_source='c4/imitreplay')
    elif arm == 'MAPPO_LANE':
        if binding is None:
            raise Blocked('MAPPO_LANE comparator not bound (bind-mappo --arm MAPPO_LANE)')
        checkpoint = dict(kind='mappo_lane', comparator_kind=binding['kind'], label=binding['label'],
                          executor_path=binding['path'], executor_sha256=binding['sha256'],
                          binding_sha256=binding['binding_sha256'], binding_path=binding['binding_path'])
    elif arm == 'NATIVE':
        checkpoint = dict(kind='native')
    else:
        raise Blocked(f'no {arm} wave')
    capture = dict(rule=panel['contract']['decisions'].get('rule'), min_t_s=DECISION_MIN_T, max_t_s=DECISION_MAX_T,
                   min_members=MIN_MEMBERS, max_members=MAX_MEMBERS)
    tag = arm.lower()+(f'_u{UPDATE:03}' if arm in ARMS else '')
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
    out = dict({k: deepcopy(record[k]) for k in IDENTITY}, status='graded', job_id=spec['job_id'],
               J_veh_s=fe._finite(record.get('J_veh_s'), f'{what} J_veh_s'), safety=dict(record['safety']),
               decision=deepcopy(record.get('decision')))
    if log is not None:
        out['decision_log'] = log
    if spec['arm'] in EXECUTOR_ARMS:
        out['executor_sha256_loaded'] = record['executor_sha256_loaded']
    if spec['arm'] == 'RULE':
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
            out.append(dict(job_id=f'nmdec_{arm.lower()}_u{UPDATE:03}_{i:04}_{label}', arm=arm,
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

def _worker(kind, spec, cfg, work, runner):
    module = importlib.import_module(runner)
    return getattr(module, 'episode' if kind == 'episode' else 'decision_branch')(spec, cfg, work)


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
                        lane=False, rule=False)
        ok = getattr(module, 'MAP', None) == MAP and callable(getattr(module, 'episode', None))
        return dict(ok=ok, native=ok and getattr(module, 'NATIVE', True) is not False,
                    decisions=ok and callable(getattr(module, 'decision_branch', None)),
                    lane=ok and getattr(module, 'MAPPO_LANE', False) is True,
                    rule=ok and getattr(module, 'RULE', False) is True and importlib.util.find_spec('tools.coopmwm.regimeu0') is not None,
                    reason=None if ok else f'{self.runner} lacks MAP={MAP!r} or episode()')

    def plant_view(self, cfg):
        """What NATIVE/U0/RULE episodes read from the plant CONFIG (validated by the production loader)."""
        from tools.coopmwm import launch_production as production
        native = production.backend_config(cfg)
        return {k: deepcopy(native.get(k)) for k in PLANT_KEYS}

    def reserve(self, cfg, jobs):
        """The clean4 ledger, through the clean4 owner's own whole-wave reservation (as clean3 lane-on)."""
        from tools.coopmwm import launch_production as production, track2_backend as p1
        p1._reserve(production.backend_config(cfg), jobs, PHASE)

    def run(self, items, cfg, workers):
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing
        os.environ['COOPMWM_G5_RUNTIME'] = '1'   # same bound runtime env as the clean3/clean4 owners
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(_worker, kind, spec, cfg, work, self.runner) for kind, spec, work in items]
            return [f.result(timeout=3600) for f in futures]


def _h(*parts):
    return int(hashlib.sha256(':'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def fake_episode(spec):
    w, arm = spec['world'], spec['arm']
    u = _h(w['seed'], arm)/0xffffffff
    factor = dict(NATIVE=1.06, U0=1.0, RULE=.985, MAPPO_SPEED=.995, MAPPO_LANE=.993, A=0.99, R=0.985, B=0.975)[arm]
    record = dict({k: deepcopy(spec[k]) for k in IDENTITY}, status='graded',
                  J_veh_s=float(100_000*factor*(1+0.03*(u-.5))),
                  safety=dict(collisions=0, teleports=0, hard_brakes=_h(w['seed'], arm, 'b') % 7,
                              ttc_events=_h(w['seed'], arm, 't') % 11))
    if arm in EXECUTOR_ARMS:
        record['executor_sha256_loaded'] = spec['checkpoint']['executor_sha256']
    if arm == 'RULE':
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


class FakeRuntime:
    """No simulator, no process pool, fake clean4 ledger: fake physics for the dry run/tests."""
    physics = False

    def __init__(self, runner=None):
        self.calls = []
        self.status = dict(dict(ok=True, native=True, decisions=True, lane=True, rule=True, reason=None), **(runner or {}))

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

    def plant_view(self, cfg):
        return {k: deepcopy(cfg.get(k)) for k in PLANT_KEYS}

    def reserve(self, cfg, jobs):
        root = Path(cfg['run_root'])
        rows = fe.ledger_rows(root)
        ids = [j['id'] for j in jobs]
        if len(set(ids)) != len(ids) or set(ids) & {r['id'] for r in rows}:
            raise Blocked('duplicate charge/attempt')
        amount = sum(j['sim_s'] for j in jobs)
        if sum(r['sim_s'] for r in rows if r['phase'] == PHASE)+amount > allocation(cfg)[1]:
            raise Blocked('whole-wave budget blocked')
        with (root/'CHARGES.jsonl').open('a', encoding='utf-8') as f:
            for j in jobs:
                f.write(json.dumps(dict(id=j['id'], phase=PHASE, sim_s=j['sim_s']), sort_keys=True)+'\n')
        self.calls.append(('reserve', PHASE, amount))

    def run(self, items, cfg, workers):
        self.calls.append(('run', len(items), workers))
        return [fake_episode(spec) if kind == 'episode' else fake_branch(spec) for kind, spec, _ in items]


# ---------------------------------------------------------------- waves (reserve whole wave, never replay)

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


def run_waves(waves, *, root, final_root, cfg, runtime, workers, base, chunked=False, limit=None):
    """waves: [(name, kind, specs)]. Every NEW wave (or the unpublished rest of a chunked wave) is
    reserved whole in the ledger BEFORE one shared dispatch; published receipts are reused, an
    interrupted one (charges, no receipt) refuses. chunked=True publishes WAVE_<name>__<a>-<b>.json."""
    root, final_root = Path(root), Path(final_root)
    out, sources, todo, done_before = {}, {}, [], {}
    chunked = chunked or bool(limit)
    prerun = isinstance(runtime, PrerunRuntime)
    phase = PRERUN_PHASE if prerun else PHASE
    # the clean-run ledger AND the evaluation root's pre-run ledger: an interrupted wave never replays
    charged = {r['id'] for r in fe.ledger_rows(root)} | {r['id'] for r in fe.ledger_rows(final_root)}
    for name, kind, specs in waves:
        check_ = check_episode if kind == 'episode' else check_branch
        records, source, covered = published(final_root, name, specs, check_)
        if covered == len(specs):
            out[name], sources[name] = records, source
            continue
        if {s['job_id'] for s in specs[covered:]} & charged:
            raise Blocked(f'{name}: interrupted wave in the ledger; never replay')
        n = (limit.get(name) if isinstance(limit, dict) else limit) or len(specs)
        rest = specs[covered:covered+n]
        done_before[name] = (records, source, covered, len(specs))
        part = f'{name}__{covered:04}-{covered+len(rest)-1:04}' if chunked or covered else name
        todo.append((name, part, kind, rest))
    new = sum(s['sim_s'] for *_, specs in todo for s in specs)
    if new and not prerun:          # pre-run: PrerunRuntime.reserve enforces the registered pre-run cap
        mine = sum(r['sim_s'] for r in fe.ledger_rows(root) if r.get('phase') == PHASE and str(r['id']).startswith(PREFIXES))
        budget_check(root, cfg, base, outstanding=new, final_root=final_root)
        mine += sum(r['sim_s'] for r in fe.ledger_rows(final_root))
        if mine+new > ALLOCATION_SIM_S:
            raise Blocked(f'{PHASE} sub-cap {ALLOCATION_SIM_S} would be exceeded')
    for name, part, kind, specs in todo:
        if specs:
            jobs = [dict(id=s['job_id'], sim_s=s['sim_s']) for s in specs]
            runtime.reserve(cfg, jobs)
            publish(final_root/f'RESERVED_{part}.json', dict(name=part, wave=name, phase=phase, jobs=[j['id'] for j in jobs],
                    sim_s=sum(j['sim_s'] for j in jobs), ledger=str((final_root if prerun else root)/'CHARGES.jsonl'),
                    unix=time.time()))
    items = [(kind, s, str(final_root/'evaluation'/s['job_id'])) for _, _, kind, specs in todo for s in specs]
    results = runtime.run(items, cfg, workers) if items else []
    position = 0
    for name, part, kind, specs in todo:
        records = results[position:position+len(specs)]
        position += len(specs)
        check_ = check_episode if kind == 'episode' else check_branch
        checked = [check_(r, s) for r, s in zip(records, specs)]
        if len(checked) != len(specs):
            raise Blocked(f'{name}: incomplete wave; never drop/impute/retry')
        value = seal(dict(schema=SCHEMA+'.wave', name=part, wave=name, phase=phase, kind=kind, physics=runtime.physics,
                          sim_s=sum(s['sim_s'] for s in specs), records=checked), 'receipt_sha256')
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
            scope='claim3a'):
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
    labels = dict(A='A u24 (the frozen S-MAPPOWZ base alone)' if base == 'mappo_frozen' else 'A u24 (base only)',
                  B='B u24 (cooperation layer + SUMO-verified guidance)',
                  R='R u24 (cooperation layer, no guidance)', NATIVE='native SUMO CAV driving', U0=U0_LABEL,
                  RULE=RULE_LABEL,
                  **{a: mappo_label(loaded[a], a)[1] if a in loaded else f'{a} (not run)' for a in COMPARATOR_ARMS})
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
    value = dict(schema=SCHEMA+'.tables', scope=scope, map=MAP, update=UPDATE, panel_sha256=panel['panel_sha256'], base=base,
                 bootstrap=bootstrap, metric=panel['contract']['metric'], horizon_s=HORIZON_S, worlds=len(worlds),
                 present=names, missing=missing or {}, sources=sources or {}, tables=tables,
                 claim=dict(claim3a='does learned cooperation add on top of the frozen MAPPO base: B vs A, B vs R, vs NATIVE '
                                    'and U0 (opens when those arms finish)',
                            claim3b='vs MAPPO-speed it168 and MAPPO-lane, each opened only after it is bound',
                            claim3c='condition-aware: no harm in light traffic (D2400_p60; B, R, U0, NATIVE; opens with 3a)'
                            )[scope[:7]],
                 primary_criterion_training='u24 replicated synergy test on registered TRAIN states (run registration); unchanged')
    return seal(json.loads(json.dumps(value, allow_nan=False)), 'analysis_sha256')


PER_SEED_COMPARATORS = ('NATIVE', 'U0', 'RULE', 'MAPPO_SPEED', 'MAPPO_LANE')
TOTAL_COMPARATORS = ('BASE', 'NATIVE', 'RULE')+COMPARATOR_ARMS


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
    lines = [f'# clean5a wz3to2 final paired evaluation (u{value["update"]}) - {value["scope"]} - '
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


def tables(*, root, final_root, panel, base, episodes, decisions, sources, missing, dry_run=False, comparators=None,
           scope='claim3a', stem='CLAIM3A_TABLES'):
    sources, missing = dict(sources), dict(missing)
    replicated = None
    path = Path(root)/REPLICATED_FILE
    if path.is_file():
        replicated = fe.check_replicated(read(path))
        sources['replicated_u024'] = dict(source=str(path), receipt_sha256=replicated['receipt_sha256'])
    else:
        missing['replicated_u024'] = 'clean4 monitor receipt absent (descriptive context only)'
    value = analyse(panel=panel, base=base, episodes=episodes, decisions=decisions, replicated=replicated,
                    sources=sources, missing=missing, comparators=comparators, scope=scope)
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
            sleep=time.sleep):
    """CONFIG -> GATE (u24 sealed, safety clean, A/B/R bound by sha) -> U0 bound/declared -> claim 3a
    episodes (A, B, R, U0, NATIVE) + decisions -> sealed CLAIM3A tables -> for each MAPPO comparator:
    bound/declared -> its wave -> its own sealed CLAIM3B tables. Returns {analysis: value}."""
    root, final_root = Path(root), Path(final_root)
    if run_plan(final_root) is None:
        raise Blocked('RUN_PLAN.json missing: run register first')
    status = Status(final_root)
    current = 'CONFIG'
    try:
        status(current, 'RUNNING')
        cfg, binding, report = preflight(root, final_root, runtime, panel, map_module)
        base, missing = binding['base'], {}
        arms = list(ARMS)
        if report['runner']['native']:
            arms.append('NATIVE')
        else:
            missing['NATIVE'] = 'runner has no native episode'
        do_decisions = report['runner']['decisions']
        for name in list(missing):
            if name not in allow_missing:
                raise Blocked(f'{name}: {missing[name]} (pass --allow-missing {name} to proceed without it)')
        if not do_decisions:   # never blocks the evaluation (MAIN): tabulated as unavailable
            missing['DECISIONS'] = 'runner has no decision_branch (per-decision branches not built)'
        status(current, 'PASS', dict(report, arms=arms, missing=missing, comparators=comparators(final_root)))
        current = 'GATE'
        gated = fe.wait(current, lambda: gate(root, cfg, runtime, binding, final_root), status, poll=poll,
                        timeout=gate_timeout, sleep=sleep)
        if not (final_root/'GATE_PASS.json').is_file():
            publish(final_root/'GATE_PASS.json', seal(gated, 'receipt_sha256'))
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
            if report['runner'].get('rule'):
                arms.append('RULE')
            elif 'RULE' in allow_missing:
                missing['RULE'] = 'runner has no S-REGIMEU0 rule (tools.coopmwm.regimeu0)'
            else:
                raise Blocked('RULE: runner has no S-REGIMEU0 rule (--allow-missing RULE)')
            compat = prerun_compat(final_root, cfg, runtime)
        else:
            missing['U0'] = f'declared unavailable: {u0["declared"]["U0"]["reason"]}'
            missing['RULE'] = 'needs the U0 executor (declared unavailable)'
        status(current, 'PASS', dict(bound=sorted(bindings), declared=sorted(u0['declared']),
                                     prerun=compat if 'U0' in bindings else None))
        current = 'EPISODES_3A'
        specs = {arm: episode_specs(arm, panel, cfg, root, base, bindings.get(arm), bindings.get('U0')) for arm in arms}
        if any(specs[arm][0]['checkpoint']['file_sha256'] != gated['checkpoint_sha256'][arm] for arm in ARMS):
            raise Blocked('u24 checkpoint changed after the gate')
        arms_3c = [a for a in CLAIM3C_ARMS if a in arms] if panel_3c is not None else []
        specs_3c = {a: episode_specs(a, panel_3c, cfg, root, base, bindings.get(a), bindings.get('U0')) for a in arms_3c}
        status(current, 'RUNNING', dict(arms=arms, claim3c_arms=arms_3c, workers=workers))
        # claim 3c (secondary, separate seal) runs in the same dispatch: it opens with 3a
        done, sources = run_waves([(arm, 'episode', specs[arm]) for arm in arms]
                                  + [(f'C3_{a}', 'episode', specs_3c[a]) for a in arms_3c], **kw)
        episodes = {arm: {r['world_id']: r for r in done[arm]} for arm in arms}
        episodes_3c = {a: {r['world_id']: r for r in done[f'C3_{a}']} for a in arms_3c}
        status(current, 'PASS', dict(arms=sorted(episodes), claim3c_arms=sorted(episodes_3c)))
        decisions = None
        if do_decisions:
            current = 'DECISIONS'
            dspecs = {arm: decision_specs(arm, episodes[arm], panel) for arm in DECISION_ARMS}
            status(current, 'RUNNING', {arm: len(v) for arm, v in dspecs.items()})
            done, more = run_waves([(f'DECISIONS_{arm}', 'branch', dspecs[arm]) for arm in DECISION_ARMS], **kw)
            sources.update(more)
            decisions = {arm: done[f'DECISIONS_{arm}'] for arm in DECISION_ARMS}
            status(current, 'PASS', {arm: len(v) for arm, v in decisions.items()})
        if alias_of(base) in missing:
            missing['BASE'] = f'BASE is {alias_of(base)}: {missing[alias_of(base)]}'

        def replicated_receipt():
            if not (root/REPLICATED_FILE).is_file():
                raise NotReady('waiting for the clean run replicated u24 receipt (descriptive context)')
        if replicated_timeout > 0:
            try:
                fe.wait('MONITOR_RECEIPTS', replicated_receipt, status, poll=poll, timeout=replicated_timeout, sleep=sleep)
            except NotReady:
                pass   # tabulated as missing; ``analyse`` re-tabulates later with zero physics
        current = 'TABLES_3A'
        out = {}
        out['claim3a'] = tables(root=root, final_root=final_root, panel=panel, base=base, episodes=episodes,
                                decisions=decisions, sources=sources, missing=dict(missing, **{
                                    a: 'claim 3b (separate seal)' for a in COMPARATOR_ARMS}),
                                dry_run=dry_run, comparators=bindings, scope='claim3a', stem='CLAIM3A_TABLES')
        status(current, 'PASS', dict(analysis_sha256=out['claim3a']['analysis_sha256'], present=out['claim3a']['present']))
        if panel_3c is not None:
            current = 'TABLES_3C'
            out['claim3c'] = tables(root=root, final_root=final_root, panel=panel_3c, base=base, episodes=episodes_3c,
                                    decisions=None, sources={f'C3_{a}': sources[f'C3_{a}'] for a in arms_3c},
                                    missing={a: missing[a] for a in CLAIM3C_ARMS if a in missing},
                                    dry_run=dry_run, comparators={a: b for a, b in bindings.items() if a == 'U0'},
                                    scope='claim3c', stem='CLAIM3C_TABLES')
            status(current, 'PASS', dict(analysis_sha256=out['claim3c']['analysis_sha256'], present=out['claim3c']['present']))
        # ---------------------------------------------------------------- claim 3b, one comparator at a time
        for arm in COMPARATOR_ARMS:
            current = f'COMPARATOR_{arm}'
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
            done, more = run_waves([(arm, 'episode', episode_specs(arm, panel, cfg, root, base, bound,
                                                                     bindings.get('U0')))], **kw)
            sources.update(more)
            both = dict(episodes, **{arm: {r['world_id']: r for r in done[arm]}})
            out[f'claim3b_{arm}'] = tables(
                root=root, final_root=final_root, panel=panel, base=base, episodes=both, decisions=decisions,
                sources=sources, missing=dict(missing, **{a: 'claim 3b (separate seal)' for a in COMPARATOR_ARMS if a != arm}),
                dry_run=dry_run, comparators=dict(bindings, **{arm: bound}), scope=f'claim3b_{arm}',
                stem=f'CLAIM3B_{arm}_TABLES')
            status(current, 'PASS', dict(analysis_sha256=out[f'claim3b_{arm}']['analysis_sha256']))
        current = 'DONE'
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

    def reserve(self, cfg, jobs):
        plan = prerun_plan(self.final_root)
        rows = fe.ledger_rows(self.final_root)
        ids = [j['id'] for j in jobs]
        if len(set(ids)) != len(ids) or set(ids) & {r['id'] for r in rows}:
            raise Blocked('duplicate pre-run charge/attempt')
        amount = sum(j['sim_s'] for j in jobs)
        if sum(r['sim_s'] for r in rows)+amount > plan['cap_sim_s']:
            raise Blocked(f'pre-run cap {plan["cap_sim_s"]} would be exceeded')
        with (self.final_root/'CHARGES.jsonl').open('a', encoding='utf-8') as f:
            for j in jobs:
                f.write(json.dumps(dict(id=j['id'], phase=PRERUN_PHASE, sim_s=j['sim_s'], unix=time.time()),
                                   sort_keys=True)+'\n')
        if self.calls is not None:
            self.calls.append(('reserve', PRERUN_PHASE, amount))

    def run(self, items, cfg, workers):
        return self.inner.run(items, cfg, workers)


def prerun_register(*, final_root, plant_config, u0_path, u0_sha256, runtime, panel, panel_3c, authority):
    """Zero physics: a NEW evaluation root with the sealed PRERUN_PLAN, the plant CONFIG copy (run_root =
    <final_root>/prerun_plant) and the bound common u0 copy (A_checkpoint_000.pt, sha re-verified)."""
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
    plant = final_root/PRERUN_PLANT
    plant.mkdir(parents=True)
    shutil.copyfile(u0_path, plant/'A_checkpoint_000.pt')
    if file_sha(plant/'A_checkpoint_000.pt') != u0_sha256:
        raise Blocked('u0 copy does not hash to the bound sha')
    cfg = dict(deepcopy(source), run_root=str(plant))
    view = runtime.plant_view(cfg)
    publish(plant/'CONFIG.json', cfg)
    plan = seal(dict(schema=SCHEMA+'.prerun', created=datetime.now(timezone.utc).isoformat(), authority=str(authority),
                     registration=PRERUN_AUTHORITY, arms=list(PRERUN_ARMS), phase=PRERUN_PHASE, cap_sim_s=prerun_cap(),
                     debits=f'{PHASE} (counted by this launcher against {ALLOCATION_SIM_S}; ledger = this root)',
                     panel_sha256=panel['panel_sha256'], panel_3c_sha256=panel_3c['panel_sha256'],
                     plant=dict(source=str(plant_config), source_file_sha256=file_sha(plant_config),
                                source_sha256=digest(source), config_sha256=digest(cfg), view=view, root=str(plant)),
                     u0=dict(source=str(u0_path), sha256=u0_sha256, path=str(plant/'A_checkpoint_000.pt')),
                     rule=dict(strategy=RULE_STRATEGY, rule_sha256=RULE_SHA256),
                     sealed=('F12: pre-run WAVE receipts are not opened, tabulated or summarised by anyone until clean5a '
                             'A/B/R are bound and the claim-3a seal opens; this launcher publishes no pre-run table'),
                     reuse=('the main run reuses each pre-run wave unchanged when the clean5a plant view and common u0 '
                            'equal this plan (prerun_compat); an interrupted pre-run wave is never replayed')),
                'receipt_sha256')
    publish(final_root/PRERUN_FILE, plan)
    return plan


def prerun_execute(*, final_root, runtime, panel, panel_3c, workers=16, arms=PRERUN_ARMS, chunk=None):
    """Pre-run NATIVE, U0 and RULE on panel v2 (3a) and the 3c panel: waves only, no analysis. Dispatched
    in chunks of ``chunk`` episodes (default 2 x workers) published as WAVE_<wave>__<a>-<b>.json; the
    ``STOP_PRERUN`` file stops cleanly after the chunk in flight; a later call (any host with the same
    absolute paths) resumes at the first unpublished episode. Returns receipt shas and counts only
    (results stay sealed, F12)."""
    final_root = Path(final_root)
    plan = prerun_plan(final_root)
    if plan is None:
        raise Blocked('PRERUN_PLAN.json missing: prerun-register first')
    if (plan['panel_sha256'], plan['panel_3c_sha256']) != (panel['panel_sha256'], panel_3c['panel_sha256']):
        raise Blocked('PRERUN_PLAN binds other panels')
    if not set(arms) <= set(PRERUN_ARMS):
        raise Blocked(f'pre-run arms are {PRERUN_ARMS}')
    pre = runtime if isinstance(runtime, PrerunRuntime) else PrerunRuntime(runtime, final_root)
    status = Status(final_root)
    current = 'PRERUN'
    try:
        status(current, 'RUNNING', dict(arms=list(arms), workers=workers))
        plant = final_root/PRERUN_PLANT
        cfg = pre.config(plant)
        runner = pre.runner_status()
        if pre.physics and not os.environ.get('COOPMWM_MAP_SPEC'):
            raise Blocked('set COOPMWM_MAP_SPEC to the staged wz3to2 MAP_SPEC.json (N1 process map) before physics')
        if not runner['ok'] or ('NATIVE' in arms and not runner['native']) or ('RULE' in arms and not runner.get('rule')):
            raise Blocked(f'runner cannot run the pre-run arms: {runner}')
        if pre.physics:
            map_check(bind_map_api(None), panel)
        u0 = mappo_binding(final_root, 'U0')
        if {'U0', 'RULE'} & set(arms):
            if u0 is None:
                raise Blocked('bind-mappo --arm U0 (frozen d9042cc5 + --residual-path <final_root>/prerun_plant/'
                              'A_checkpoint_000.pt) before the pre-run')
            if (u0['residual_path'], u0['residual_sha256']) != (plan['u0']['path'], plan['u0']['sha256']):
                raise Blocked('the U0 binding residual is not the pre-run u0 copy')
        waves = []
        for arm in arms:
            waves.append((arm, 'episode', episode_specs(arm, panel, cfg, plant, 'mappo_frozen', None, u0)))
        for arm in arms:
            waves.append((f'C3_{arm}', 'episode', episode_specs(arm, panel_3c, cfg, plant, 'mappo_frozen', None, u0)))
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
        out = dict(schema=SCHEMA+'.prerun_done', status='STOPPED' if stopped else 'PASS', sealed=True,
                   prerun_plan_sha256=plan['receipt_sha256'], chunk=chunk, rounds=rounds,
                   coverage={k: dict(published=v[0], registered=v[1]) for k, v in cov.items()},
                   receipts={k: v['receipt_sha256'] for k, v in receipts.items()},
                   charged_sim_s=sum(r['sim_s'] for r in fe.ledger_rows(final_root)), cap_sim_s=plan['cap_sim_s'],
                   resume=None if not stopped else ('remove STOP_PRERUN (or copy the whole final root to the same '
                                                    'absolute path on the other host) and run prerun-run again'))
        status(current, out['status'], out)
        return out
    except BaseException as error:
        status(current, 'FAILED', dict(error=repr(error), traceback=traceback.format_exc()))
        raise


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
    if plan['rule']['rule_sha256'] != RULE_SHA256:
        raise Blocked('pre-run RULE digest differs from this launcher')
    source = dict(cfg)
    return dict(status='compatible', prerun_plan_sha256=plan['receipt_sha256'], plant_keys=list(PLANT_KEYS),
                same_config_except_run_root=digest(dict(source, run_root=plan['plant']['root'])) == plan['plant']['config_sha256'])


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
                     contract_sha256=digest(panel['contract']), budget=budget, quote=quote(binding['base']),
                     prerun=compat,
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


def check(*, root, final_root, runtime, panel, map_module=None):
    out = dict(schema=SCHEMA+'.check', panel_sha256=panel['panel_sha256'], new_sim_s=0)
    try:
        cfg, binding, report = preflight(root, final_root, runtime, panel, map_module)
        out.update(report, quote=quote(binding['base']))
        out['run_plan'] = (run_plan(final_root) or {}).get('receipt_sha256') if Path(final_root).is_dir() else None
        try:
            out['gate'] = gate(root, cfg, runtime, binding, final_root)
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
    cfg = dict(run_root=str(root), source_root=str(Path(out)/'no_source'), runtime='dry',
               common_checkpoint_sha256=file_sha(root/'A_checkpoint_000.pt'),
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
        (root/f'{arm}_checkpoint_{UPDATE:03}.pt').write_bytes(f'dry-run fake {arm} u24'.encode())
        publish(root/f'{arm}_UPDATE_{UPDATE:03}.json', seal(dict(arm=arm, update=UPDATE, dry_run=True), 'update_sha256'))
        publish(root/f'{arm}_COMPLETE.json', dict(arm=arm, updates=UPDATE, dry_run=True))
    zero = {k: 0 for k in SAFETY}
    for u in range(1, UPDATE+1):
        publish(root/f'LEARNING_MONITOR_u{u:03}.json', seal(dict(update=u, config_sha256=digest(cfg), dry_run=True,
            arms=[dict(arm=a, action='CONTINUE', metrics=dict(safety=dict(zero))) for a in ARMS]), 'receipt_sha256'))
    return root, cfg


def fake_replicated(root):
    from tools.coopmwm import track2_monitor as p3
    records = []
    for arm in ('B', 'R'):
        for i in range(5):
            gain = (3.0 if arm == 'B' else 0.5)+i*0.4
            records.append(dict(arm=arm, replicate=i, world_seed=989200+i, state_set_sha256=f's{i}', window_set_sha256=f'w{i}',
                                metric=gain, **{fe.SYNERGY: gain, fe.TOTAL_BENEFIT: 10+gain}, safety={k: 0 for k in SAFETY},
                                reference_safety={k: 0 for k in SAFETY}))
    analysis = p3._analyse(records, UPDATE, dict(p3.BOOTSTRAP))
    publish(Path(root)/REPLICATED_FILE, seal(dict(kind='replicated', update=UPDATE, status='complete', dry=False,
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
                        lane_control='native' if arm == 'MAPPO_SPEED' else 'policy', lane_head=arm == 'MAPPO_LANE',
                        policy=policy, note='dry-run stand-in, not a MAPPO export'), fake)
    kind = next(iter(MAPPO_KINDS[arm]))
    extra = {}
    if arm == 'U0':
        extra = dict(residual_path=str(residual), residual_sha256=file_sha(residual))
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
        fake_replicated(root)
    runtime = FakeRuntime()
    final_root = out/'final_root'
    panel_3c = load_panel_3c()
    pre = None
    if prerun:
        prerun_register(final_root=final_root, plant_config=root/'CONFIG.json', u0_path=root/'A_checkpoint_000.pt',
                        u0_sha256=cfg['common_checkpoint_sha256'], runtime=runtime, panel=panel, panel_3c=panel_3c,
                        authority='DRY RUN ONLY')
        bind_fake_mappo(out, final_root, 'U0', residual=final_root/PRERUN_PLANT/'A_checkpoint_000.pt')
        pre = prerun_execute(final_root=final_root, runtime=runtime, panel=panel, panel_3c=panel_3c, workers=16)
    plan = register(root=root, final_root=final_root, runtime=runtime, panel=panel)
    for arm in BOUND_ARMS:
        if arm != 'U0' or not prerun:
            bind_fake_mappo(out, final_root, arm, residual=root/'A_checkpoint_000.pt')
    value = execute(root=root, final_root=final_root, runtime=runtime, panel=panel, panel_3c=panel_3c,
                    poll=0, dry_run=True)
    charged = sum(r['sim_s'] for r in fe.ledger_rows(root) if r['phase'] == PHASE)
    return dict(status='PASS', new_sim_s=0, physics=False, out=str(out), final_root=str(final_root), base=base,
                target='clean5a (dry)',
                analyses={k: v['analysis_sha256'] for k, v in value.items()},
                present={k: v.get('present') for k, v in value.items()}, calls=runtime.calls,
                fake_v2v_final_sim_s=charged, quote=plan['quote'], prerun=pre,
                fake_prerun_sim_s=sum(r['sim_s'] for r in fe.ledger_rows(final_root)),
                tables=sorted(str(x) for x in final_root.glob('CLAIM3*_TABLES.md')))


# ---------------------------------------------------------------- CLI

def _analyse_published(root, final_root, runtime, panel, panel_3c=None):
    """Zero physics: re-tabulate every analysis whose arms are published (3a; each bound 3b)."""
    cfg = runtime.config(root)
    plan = run_plan(final_root)
    if plan is None:
        raise Blocked('no RUN_PLAN.json')
    base = plan['binding']['base']
    state = comparators(final_root)
    bindings = state['bound']
    episodes, sources, missing = {}, {}, {}
    for arm in EPISODE_ARMS:
        if not (Path(final_root)/f'WAVE_{arm}.json').is_file() and not _chunk_files(final_root, arm):
            missing[arm] = 'no published wave receipt'
            continue
        if arm in BOUND_ARMS and arm not in bindings:
            raise Blocked(f'{arm} wave published without a binding')
        specs = episode_specs(arm, panel, cfg, root, base, bindings.get(arm), bindings.get('U0'))
        records, source, covered = published(final_root, arm, specs, check_episode)
        if covered < len(specs):
            missing[arm] = f'wave incomplete ({covered}/{len(specs)} published)'
            continue
        episodes[arm] = {r['world_id']: r for r in records}
        sources[arm] = source
    if any(a in missing for a in ARMS):
        raise Blocked(f'A/B/R waves are not all published: {missing}')
    decisions = {}
    for arm in DECISION_ARMS:
        path = Path(final_root)/f'WAVE_DECISIONS_{arm}.json'
        if path.is_file():
            value = verify(read(path), 'receipt_sha256', path.name)
            specs = decision_specs(arm, episodes[arm], panel)
            decisions[arm] = [check_branch(r, s) for r, s in zip(value['records'], specs)]
            sources[f'DECISIONS_{arm}'] = dict(source=str(path), receipt_sha256=value['receipt_sha256'])
    decisions = decisions if len(decisions) == len(DECISION_ARMS) else None
    if decisions is None:
        missing['DECISIONS'] = 'no published decision waves'
    if alias_of(base) in missing:
        missing['BASE'] = f'BASE is {alias_of(base)}: {missing[alias_of(base)]}'
    core = {a: v for a, v in episodes.items() if a in CLAIM3A_ARMS}
    out = dict(claim3a=tables(root=root, final_root=final_root, panel=panel, base=base, episodes=core, decisions=decisions,
                              sources=sources, missing=dict(missing, **{a: 'claim 3b (separate seal)' for a in COMPARATOR_ARMS}),
                              comparators={a: b for a, b in bindings.items() if a == 'U0'}, scope='claim3a',
                              stem='CLAIM3A_TABLES'))
    if panel_3c is not None and all((Path(final_root)/f'WAVE_C3_{a}.json').is_file() for a in ('B', 'R')):
        e3c, s3c = {}, {}
        for arm in CLAIM3C_ARMS:
            if (Path(final_root)/f'WAVE_C3_{arm}.json').is_file() or _chunk_files(final_root, f'C3_{arm}'):
                specs = episode_specs(arm, panel_3c, cfg, root, base, bindings.get(arm), bindings.get('U0'))
                records, source, covered = published(final_root, f'C3_{arm}', specs, check_episode)
                if covered == len(specs):
                    e3c[arm] = {r['world_id']: r for r in records}
                    s3c[f'C3_{arm}'] = source
        out['claim3c'] = tables(root=root, final_root=final_root, panel=panel_3c, base=base, episodes=e3c, decisions=None,
                                sources=s3c, missing={a: 'no published wave receipt' for a in CLAIM3C_ARMS if a not in e3c},
                                comparators={a: b for a, b in bindings.items() if a == 'U0'}, scope='claim3c',
                                stem='CLAIM3C_TABLES')
    for arm in COMPARATOR_ARMS:
        if arm in episodes:
            out[f'claim3b_{arm}'] = tables(
                root=root, final_root=final_root, panel=panel, base=base, episodes=dict(core, **{arm: episodes[arm]}),
                decisions=decisions, sources=sources,
                missing=dict(missing, **{a: 'claim 3b (separate seal)' for a in COMPARATOR_ARMS if a != arm}),
                comparators={a: b for a, b in bindings.items() if a in ('U0', arm)}, scope=f'claim3b_{arm}',
                stem=f'CLAIM3B_{arm}_TABLES')
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('register-panel', 'register-panel-3c', 'verify-panel', 'quote', 'check', 'register',
                                       'bind-arms', 'prerun-register', 'prerun-run',
                                       'bind-mappo', 'declare-unavailable', 'run', 'analyse', 'dry-run'))
    p.add_argument('--root', default=ROOT)
    p.add_argument('--final-root', default=FINAL_ROOT)
    p.add_argument('--panel', default=str(PANEL_FILE))
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
    p.add_argument('--a-sha', help='bind-arms: A u24 checkpoint sha256')
    p.add_argument('--b-sha', help='bind-arms: B u24 checkpoint sha256')
    p.add_argument('--r-sha', help='bind-arms: R u24 checkpoint sha256')
    p.add_argument('--mappo-kind', choices=sorted({k for v in MAPPO_KINDS.values() for k in v}),
                   help='bind-mappo: retrained_wz3to2 | u0_transferred_not_retrained (MAPPO_SPEED), lane_retrained_wz3to2 (MAPPO_LANE)')
    p.add_argument('--reason', help='declare-unavailable: why the comparator cannot be evaluated')
    p.add_argument('--residual-path', help='bind-mappo --arm U0: the common u0 A_checkpoint_000.pt (pre-run: '
                                           '<final_root>/prerun_plant/A_checkpoint_000.pt)')
    p.add_argument('--residual-sha256', help='bind-mappo --arm U0: its sha256 (= clean5a common_checkpoint_sha256)')
    p.add_argument('--plant-config', help='prerun-register: the clean5a CONFIG.json (plant source for NATIVE/U0/RULE)')
    p.add_argument('--u0-path', help='prerun-register: the common u0 checkpoint (A_checkpoint_000.pt)')
    p.add_argument('--u0-sha256', help='prerun-register: its sha256')
    p.add_argument('--chunk', type=int, help='prerun-run: episodes per dispatch (default 2 x workers); STOP_PRERUN '
                                             'in the final root stops after the chunk in flight')
    p.add_argument('--authority', help='bind-mappo: verbatim message/decision (e.g. N7 hand-off or MAIN fallback decision)')
    p.add_argument('--comparator-wait', type=int, default=7*86400,
                   help='run: seconds to wait (after the u24 gate) for every comparator to be bound or declared')
    a = p.parse_args(argv)
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
            result = check(root=a.root, final_root=a.final_root, runtime=Runtime(a.runner), panel=panel,
                           map_module=a.map_api)
        elif a.command == 'register':
            result = register(root=a.root, final_root=a.final_root, runtime=Runtime(a.runner), panel=panel, base=a.base)
        elif a.command == 'bind-mappo':
            check_ = None
            try:
                from tools.coopmwm.nm_eval_runner import mappo_load_check
                check_ = mappo_load_check(a.mappo_path, a.mappo_sha256)
            except ImportError as error:
                check_ = dict(status='not run', reason=repr(error))
            result = bind_mappo(final_root=a.final_root, path=a.mappo_path, sha256=a.mappo_sha256, kind=a.mappo_kind,
                                arm=a.arm, authority=a.authority, check=check_, residual_path=a.residual_path,
                                residual_sha256=a.residual_sha256)
        elif a.command == 'prerun-register':
            result = prerun_register(final_root=a.final_root, plant_config=a.plant_config, u0_path=a.u0_path,
                                     u0_sha256=a.u0_sha256, runtime=Runtime(a.runner), panel=panel,
                                     panel_3c=load_panel_3c(), authority=a.authority or PRERUN_AUTHORITY)
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
        elif a.command == 'bind-arms':
            result = bind_arms(final_root=a.final_root, shas=dict(A=a.a_sha, B=a.b_sha, R=a.r_sha), authority=a.authority)
        elif a.command == 'declare-unavailable':
            result = declare_unavailable(final_root=a.final_root, arm=a.arm, authority=a.authority, reason=a.reason)
        elif a.command == 'analyse':
            value = _analyse_published(a.root, a.final_root, Runtime(a.runner), panel, load_panel_3c())
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
                value = execute(root=a.root, final_root=a.final_root, runtime=Runtime(a.runner), panel=panel,
                                panel_3c=load_panel_3c(),
                                allow_missing=tuple(x for x in a.allow_missing.split(',') if x), workers=a.workers,
                                poll=a.poll, replicated_timeout=a.replicated_wait, comparator_timeout=a.comparator_wait,
                                map_module=a.map_api)
                result = {k: dict(analysis_sha256=v['analysis_sha256'], present=v.get('present'), missing=v.get('missing'))
                          for k, v in value.items()}
    print(json.dumps(result, indent=1, sort_keys=True, default=str))


if __name__ == '__main__':
    main()
