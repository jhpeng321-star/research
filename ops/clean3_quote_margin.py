"""CLEAN3 full u1-u24 per-phase quote with >=15% margin (MAIN 02:43 rule).

Zero physics. Base costs come from measured clean2 ledger rates (cpuV, read
2026-11-06 02:47 +08) and the cpuV native-manifest quote
CLEAN3_QUOTE_STOPPED_SESSION4.json. Every phase is sized for the full u1..u24
schedule; allocation = ceil(base*1.15 / 1200)*1200 unless a registered hard cap
already covers the worst case with >=15% over expected cost.
"""
import hashlib, json, math
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
MARGIN = 0.15
EP = 1200
UPDATES = 24

def alloc(base):
    return int(math.ceil(base * (1 + MARGIN) / EP) * EP)

native = json.loads((repo/'CLEAN3_QUOTE_STOPPED_SESSION4.json').read_text())['quote']
dq, lq = native['diagnostic_costs'], native['lesson_costs']
assert sorted(lq) == ['lesson_u006', 'lesson_u012', 'lesson_u018', 'lesson_u024']
budget = json.loads((repo/'BUDGET_CLEAN3.json').read_text())

phases = {}
def phase(name, schedule, base, *, allocated=None, basis):
    allocated = alloc(base) if allocated is None else allocated
    assert allocated >= base * (1 + MARGIN) - 1e-9, name
    phases[name] = dict(schedule=schedule, base_sim_s=base, allocated_sim_s=allocated,
                        margin_fraction=round(allocated/base - 1, 4), basis=basis)

# clean2 measured: v2v_train 72000 sim-s/update for A+B+R, of which 14400 practice refs.
phase('v2v_train_waves', 'u1-u24 A/B/R', UPDATES*57600,
      basis='clean2 measured 57600/update (A+B+R training episodes), u1..u18 constant')
phase('v2v_practice_refs', 'u1-u24 B/R practice references', UPDATES*14400,
      basis='clean2 measured 14400/update practiceref rows')
phase('v2v_lessons', 'u6,u12,u18,u24', sum(lq.values()),
      basis='cpuV native manifests 4 x 8760 (CLEAN3_QUOTE_STOPPED_SESSION4)')
phase('early_probes', 'u2,u4', dq['probe_u002']+dq['probe_u004'], basis='native manifests 2 x 22740')
phase('isolated_replicated', 'u6,u12,u24 (N=5)',
      dq['replicated_u006']+dq['replicated_u012']+dq['replicated_u024'], basis='native manifests 3 x 47920')
phase('v2v_interim', 'u8 paired evaluation', 3*96*EP, basis='3 arms x 96 test episodes x 1200 (clean2 measured 345600)')
phase('v2v_final', 'u24 paired evaluation A/B/R', 3*96*EP, basis='3 arms x 96 test episodes x 1200')
phase('v2v_final_laneon_mappo_aecd7d70', 'u24 extra frozen baseline (descriptive)', 96*EP,
      basis='owner 01:16: lane-on MAPPO aecd7d70 on the same 96 paired worlds')
phase('g4_synergy_credit', 'u1-u24 sparse joint-macro counterfactuals B=R', 153600, allocated=230400,
      basis='G4 two-actor full quote 153600; registered hard cap 230400 covers worst case')
phase('g1_certification', 'pre-launch, 8 ctx/arm x 2 arms x 2 seeds', 153600, allocated=230400,
      basis='G1_BUDGET preferred two-actor 153600; registered cap 230400 = worst case (4 solos)')
phase('gate_bank_verification_ext', 'pre-launch, run once, no retry', native['phases']['gate_bank_verification_ext'],
      basis='native gate manifest 26 branches')
phase('track2_smoke', 'pre-launch, all levers ON', 20000, basis='registered smoke cap 20000 (clean2 actual 14400)')

total = sum(p['allocated_sim_s'] for p in phases.values())
prior = 3098640           # clean1 1248480 (incl. plan_fix 33600) + clean2 1849200 + parity 960, cpuV ledgers 02:47
plan_fix_reserved = 0 if budget.get('plan_fix_reservation_released') else 566400  # owner 03:0x 'approve' released it
mining = 500000
cap = budget['total_cap_sim_s']
available = cap - prior - plan_fix_reserved - mining
fits = total <= available
quote = dict(schema='coopmwm.clean3.quote_margin.v1', created=__import__('datetime').datetime.now().astimezone().isoformat(timespec='seconds'),
    rule='MAIN 02:43: every phase sized for full u1-u24 with >=15% margin; total <= cap - prior - mining - plan_fix reservation',
    budget_sha256=hashlib.sha256((repo/'BUDGET_CLEAN3.json').read_bytes()).hexdigest(),
    native_quote_sha256=hashlib.sha256((repo/'CLEAN3_QUOTE_STOPPED_SESSION4.json').read_bytes()).hexdigest(),
    updates=UPDATES, arms=['A', 'B', 'R'], margin_min=MARGIN, phases=phases,
    total_allocated_sim_s=total, cap_sim_s=cap, prior_charged_sim_s=prior,
    plan_fix_reserved_unspent_sim_s=plan_fix_reserved, mining_phase_sim_s=mining,
    available_for_clean3_sim_s=available, fits=fits, shortfall_sim_s=max(0, total-available),
    alt_if_full_600000_plan_fix_reserved=dict(available_sim_s=cap-prior-600000-mining,
                                              shortfall_sim_s=max(0, total-(cap-prior-600000-mining))),
    status='PASS' if fits else 'BLOCKED_BUDGET', launch_authorized=fits, new_sim_s=0)
out = repo/'CLEAN3_QUOTE.json'
with out.open('x') as f:
    json.dump(quote, f, indent=2)
print(json.dumps({k: quote[k] for k in ('total_allocated_sim_s', 'available_for_clean3_sim_s', 'fits', 'shortfall_sim_s', 'status')}))
assert fits, 'CLEAN3 quote exceeds remaining cap: refuse launch'
