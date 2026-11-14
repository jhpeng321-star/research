"""LEARNED_TRIGGER 2-world SUMO smoke (amendment 26; TRAIN seeds, D2800_p20): U0 (S-REGIMEU0 u0, clean4 plant) +
learned_trigger hook (ORACLE_WM executor, frozen classifier admission) on the frozen bank. Mechanics only. Checks:
episodes finish; only CAVs commanded (the hook raises otherwise); digest + model sha as registered; receipts
(scores, fires, per-act prob, refusals); fire rate over scored offers; every act's prob >= threshold.
Usage: python ops/learned_trigger_smoke.py OUT_DIR BANK.json MODEL.json SEED [SEED ...]   (run from the checkout root)"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.coopmwm import accel_plant, clean4_verify as c4v, learned_trigger as L, map_scenarios as ms  # noqa: E402
from tools.coopmwm.v2v_runtime import load_checkpoint  # noqa: E402

MODEL_SHA256 = '8e0ddd3e241ce0172736909133f0b745c58b6bf795e06ca5364e5755f7e3b720'
DIGEST = '41fc8d1a2fb0ef547524a9d36488b7bbb4bc9f0e8c1e2d06f5fe85eb5903b54d'
out, bank_path, model_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
seeds = [int(s) for s in sys.argv[4:]]
L.assert_seeds(seeds)
if L.learned_trigger_digest() != DIGEST:
    raise SystemExit('learned_trigger differs from its registration (amendment 26)')
model = L.load_model(model_path, MODEL_SHA256)
REG = Path(os.environ.get('ORACLE_REGIMEU0', '/root/autodl-fs/wmverify_20261108/regimeu0_ref'))  # inputs/CONFIG.json, u0.pt
cfg = json.load(open(REG/'inputs'/'CONFIG.json'))
base = load_checkpoint(REG/'inputs'/'u0.pt', config=cfg)[0]
weights = {k: v.detach().clone() for k, v in base.state_dict().items()}
doc = json.load(open(bank_path))
report = dict(learned_trigger_sha256=DIGEST, model_sha256=MODEL_SHA256, bank=str(bank_path), worlds=[])
for seed in seeds:
    world = ms.world(seed, 2800, 0.2)
    work = out/f'D2800_p20_s{seed}_LEARNED_TRIGGER'
    work.mkdir(parents=True, exist_ok=False)
    hook = L.make_hook(doc, 'D2800_p20', model)
    job = c4v.job_fields(ident=work.name, path=work/'plant', world=world, arm='A', weights=weights, noise_seed=0,
                         runtime=c4v.RUNTIME_DIR, checkpoint=c4v.CHECKPOINT, sim_s=1200, v2v=True, deterministic=True)
    job.update(stop=1200, queue_root=str(out))
    t0 = time.time()
    res = accel_plant.episode(job, controller_hook=hook)
    rec = hook.receipts()
    probs = [a.get('prob') for a in rec['acts']]
    ok_probs = all(p is not None and p >= rec['threshold'] for p in probs)
    report['worlds'].append(dict(seed=seed, world=world, team_delay_vs=res['totals']['team_delay_vs'],
                                 actual_sim_s=res['actual_sim_s'], wall_s=round(time.time()-t0, 1), receipts=rec,
                                 fire_rate=rec['fires']/rec['scores'] if rec['scores'] else None,
                                 act_probs_ge_threshold=ok_probs))
    print(seed, res['totals']['team_delay_vs'], rec['activations'], rec['scores'], rec['fires'], ok_probs, flush=True)
(out/'SMOKE.json').write_text(json.dumps(report, indent=1, default=str))
