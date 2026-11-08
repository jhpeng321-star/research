"""ORACLE 2-world SUMO smoke (training seeds, D2800_p20): U0 (S-REGIMEU0 u0, clean4 plant) + oracle_plan hook on the
union bank. Checks: episodes finish; every activation's runtime privileged bin equals the bin recomputed from the
observable tape at that second (runtime rows == tape rows); only CAVs commanded (hook raises otherwise); receipts.
Usage: python smoke.py OUT_DIR BANK.json SEED [SEED ...]"""
import gzip, importlib.util, json, sys, time
from pathlib import Path
out, bank_path, seeds = Path(sys.argv[1]), Path(sys.argv[2]), [int(s) for s in sys.argv[3:]]
spec = importlib.util.spec_from_file_location('oracle_plan', str(Path(__file__).with_name('oracle_plan.py')))
O = importlib.util.module_from_spec(spec); sys.modules['oracle_plan'] = O; spec.loader.exec_module(O)
from tools.coopmwm import imitreplay as ir, accel_plant, clean4_verify as c4v, map_scenarios as ms
from tools.coopmwm.v2v_runtime import load_checkpoint
REG = Path('/root/autodl-fs/wmverify_20261108/regimeu0_ref')
cfg = json.load(open(REG/'inputs'/'CONFIG.json'))
base = load_checkpoint(REG/'inputs'/'u0.pt', config=cfg)[0]
weights = {k: v.detach().clone() for k, v in base.state_dict().items()}
doc = json.load(open(bank_path))
report = dict(oracle_sha256=O.oracle_digest(), vendored=O.vendored_check(), bank=str(bank_path), worlds=[])
for seed in seeds:
    world = ms.world(seed, 2800, 0.2)
    work = out/f'D2800_p20_s{seed}_ORACLE'
    work.mkdir(parents=True, exist_ok=False)
    hook = O.make_hook(doc, 'D2800_p20')
    job = c4v.job_fields(ident=work.name, path=work/'plant', world=world, arm='A', weights=weights, noise_seed=0,
                         runtime=c4v.RUNTIME_DIR, checkpoint=c4v.CHECKPOINT, sim_s=1200, v2v=True, deterministic=True)
    job.update(stop=1200, queue_root=str(out))
    t0 = time.time()
    res = accel_plant.episode(job, controller_hook=hook)
    rec = hook.receipts()
    acts = {a['t']: a for a in rec['acts']}
    tape = {}
    with gzip.open(work/'plant'/'observable.jsonl.gz', 'rt') as fh:
        for line in fh:
            d = json.loads(line)
            t = int(round(d['time_s']))
            if t in acts:
                tape[t] = [dict(vid=r['vid'], is_cav=bool(r.get('is_cav')), lane=int(r['lane']), s_m=float(r['s_m']),
                                v_mps=None if r.get('v_mps') is None else float(r['v_mps'])) for r in d['vehicles']]
    checks = []
    for t, a in acts.items():
        rows = tape.get(t) or []
        own = next((r for r in rows if r['vid'] == a['holder']), None)
        checks.append(dict(t=t, family=a['family'], runtime_bin=a['bin'], tape_bin=O.onset_bin(own, rows, 'privileged')))
    report['worlds'].append(dict(seed=seed, world=world, team_delay_vs=res['totals']['team_delay_vs'],
                                 actual_sim_s=res['actual_sim_s'], wall_s=round(time.time() - t0, 1), receipts=rec,
                                 bin_checks=checks, bins_equal=all(c['runtime_bin'] == c['tape_bin'] for c in checks)))
    print(seed, res['totals']['team_delay_vs'], rec['activations'], rec['activations_by_family'],
          all(c['runtime_bin'] == c['tape_bin'] for c in checks), flush=True)
(out/'SMOKE.json').write_text(json.dumps(report, indent=1, default=str))
