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
# Bases from the zero-physics dry prepare on cpuZ (c20965f0, config b5afa5ff): native probe manifests 2 x 36,003,
# replicated 3 x 43,524; EV-only mode has no bank lessons (0). G4 at its registered hard cap.
phases = b.quote(lesson_sim_s=0, probe_sim_s=72_006, replicated_sim_s=130_572, g4_base_sim_s=200_000,
                 g4_cap_sim_s=230_400, diagnostic_factor=1.0,
                 basis=dict(train='3 arms (A frozen evaluation arm + B + R) x 16 episodes x 1200 s x 24',
                            lessons='EV-only guidance: no verified bank lessons (registered; 0 sim-s)',
                            probes='native dry-prepare manifests 2 x 36,003', replicated='native dry-prepare manifests 3 x 43,524',
                            g4='G4LD expected 192,000-230,400 within the registered 230,400 hard cap'))
phases['v2v_final'] = b.phase('u24 final evaluation (N6): claims 3a + 3c incl. RULE arm', 1_165_200, allocated=1_340_000,
                              basis='MAIN-registered final-eval cap 1,340,000 (N6 launcher budget_check); base = cap/1.15')
phases['smoke_gate'] = b.phase('pre-launch smoke (<=20,000) + one safety gate (<=35,820 registered gate limit)', 48_000,
                               allocated=56_000, basis='smoke cap 20,000 + rerun_cpus.GATE_SIM_S 35,820 (3-member '
                               'Plan A walls: the dry prepare showed the full-window gate exceeds 20,000)')
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
