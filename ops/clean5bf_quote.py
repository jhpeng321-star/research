"""CLEAN5BF_QUOTE.json (covered re-quote of CLEAN5B_QUOTE 1bb292e1 after the failed r1/r2/r3 spend 85,372; envelope 3,930,000).
Base docstring: CLEAN5B_QUOTE.json: full u1-u24 per-phase quote (>=15% margin) inside MAIN's separate clean5b phase
(<= 3,900,000 sim-s, REGISTRATION 2026-11-06 23:40; BUDGET_CLEAN5B.json). Zero physics.
Same phase structure and bases as CLEAN5A_QUOTE b80fbd66 (same frozen u0 base, 3 arms x 16 episodes x 1200 s x 24,
same Plan A registration states, same 6 cells/TRAIN worlds); prior 0 (separate envelope).
"""
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.coopmwm import clean4_budget as b  # noqa: E402

ENVELOPE = 3_930_000
PARENT = '1bb292e1085fad10e2d064cf6f34caf646af9e44fd32d90a35c3d1725f73097f'   # CLEAN5B_QUOTE (BUDGET_CLEAN5B binds it)
PRIOR = dict(clean5b_failed_roots=2_400+75_772+7_200)   # r1 + r2 (cpuV OOM) + r3 (owner stop, smoke only)
phases = b.quote(lesson_sim_s=0, probe_sim_s=72_006, replicated_sim_s=130_572, g4_base_sim_s=200_000,
                 g4_cap_sim_s=230_400, diagnostic_factor=1.0,
                 basis=dict(train='3 arms (A frozen evaluation arm + B + R) x 16 episodes x 1200 s x 24',
                            lessons='EV-only guidance: no verified bank lessons (registered; 0 sim-s)',
                            probes='clean5a wz3to2 dry-prepare manifests 2 x 36,003 (same registration states)',
                            replicated='clean5a wz3to2 dry-prepare manifests 3 x 43,524 (same registration states)',
                            g4='G4LD (programs OFF) within the registered 230,400 hard cap'))
phases['v2v_final'] = b.phase('u24 final evaluation (claims 3a + 3c incl. RULE arm), if registered for clean5b', 1_165_200,
                              allocated=1_340_000, basis='mirror of the clean5a v2v_final allocation 1,340,000')
phases['smoke_gate'] = b.phase('pre-launch smoke (<=20,000) + one safety gate (<=35,820 registered gate limit)', 40_000,
                               allocated=46_000, basis='measured wz3to2 smoke 15,770 + gate 24,002 = 39,772 (r2 receipts)')
total = sum(p['allocated_sim_s'] for p in phases.values())
doc = b.document(phases, prior=PRIOR, cap=ENVELOPE, parent_quote_sha256=PARENT, created=datetime.now().astimezone().isoformat(timespec='seconds'),
                 notes=['clean5b = clean5a recipe + LCEXEC per-CAV free_program slots (MAIN 23:40/23:48); separate phase '
                        'clean5b <= 3,900,000 sim-s, prior 0',
                        'approval contract: MAIN writes E:/agent-work/_sol/BUDGET_CLEAN5B.json (owner_approval, '
                        'approved_sim_s 3,900,000, total_cap_sim_s, quote_sha256 = this receipt_sha256)'])
doc_path = Path(sys.argv[1])
with doc_path.open('x', encoding='utf-8') as f:
    json.dump(doc, f, indent=1, sort_keys=True); f.write('\n')
print(json.dumps(dict(total=total, available=doc['available_sim_s'], fits=doc['fits'], sha=doc['receipt_sha256'])))
