"""CLEAN5A_QUOTE.json: full u1-u24 per-phase quote (>=15% margin) inside the owner-approved 4M clean-training
envelope (BUDGET_CLEAN4.json), after the clean4 spend (sanity 28,800 + Plan A 75,802). Zero physics.
Design (MAIN GO 19:54): base u0 d9042cc5 FROZEN; A = frozen evaluation arm (no PPO), only B/R train; 6 staged
cells; G4LD at its 230,400 hard cap (budget-derived start caps); u6/u12/u24 replicated test; final eval v2v_final
cap 1,200,000 (claims 3a + 3c, MAIN 19:57); own smoke+gate 40,000.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.coopmwm import clean4_budget as b  # noqa: E402

CLEAN4_SPENT = 28_800+75_802
ENVELOPE = 4_000_000
PARENT = '903e2be27145f7da0e25cf7dfcd0303f0b0db0ce327badd007d0bad6a3a8a7eb'   # owner-approved clean4 quote (BUDGET_CLEAN4.json binds it)
phases = b.quote(lesson_sim_s=35_040, probe_sim_s=45_480, replicated_sim_s=143_760, g4_base_sim_s=200_000,
                 g4_cap_sim_s=230_400, diagnostic_factor=1.5,
                 basis=dict(train='3 arms (A frozen evaluation arm + B + R) x 16 episodes x 1200 s x 24',
                            g4='G4LD expected 192,000-230,400 within the registered 230,400 hard cap'))
phases['v2v_final'] = b.phase('u24 final evaluation (N6): claims 3a (96 x arms) + 3c light-traffic panel', 1_020_960,
                              allocated=1_200_000, basis='MAIN 19:54/19:57: total ~1,020,960; v2v_final cap 1,200,000')
phases['smoke_gate'] = b.phase('pre-launch smoke (<=20,000) + one safety gate (<=20,000)', 34_000, allocated=40_000,
                               basis='clean3/clean4 measured smoke 14,400 + gate 15,160 = 29,560 -> base 34,000')
total = sum(p['allocated_sim_s'] for p in phases.values())
doc = b.document(phases, prior=dict(clean4_actual=CLEAN4_SPENT), cap=ENVELOPE,
                 created=datetime.now().astimezone().isoformat(timespec='seconds'), parent_quote_sha256=PARENT,
                 notes=['clean5a inside the owner-approved 4M clean-training envelope (BUDGET_CLEAN4.json, owner 14:58); '
                        'available = 4,000,000 - clean4 actual 104,602 = 3,895,398',
                        'A = frozen evaluation arm (registered frozen_control_arm): its episodes are still collected every '
                        'update as references/evaluation; no PPO',
                        'base u0 d9042cc5 frozen; bound-trunk shim (frozen_trunk) available as fallback only'])
doc_path = Path(sys.argv[1])
with doc_path.open('x', encoding='utf-8') as f:
    json.dump(doc, f, indent=1, sort_keys=True); f.write('\n')
print(json.dumps(dict(total=total, available=doc['available_sim_s'], fits=doc['fits'], sha=doc['receipt_sha256'])))
