"""Zero-physics clean3 qualification: G1 selection, mined admission, full prepare on a disposable root."""
import json, sys, time, traceback, faulthandler
faulthandler.enable()
from pathlib import Path
import types
out = Path(sys.argv[1]); src = sys.argv[2]
from tools.coopmwm import clean3_restart as c3, rerun_cpus as rc, launch_preparation as stage
from tools.coopmwm import track2_backend as p1, launch_production as production, launch_training as training
from tools.coopmwm.exact_grade import _seal
from tools.coopmwm.protocol import digest
def forbidden(*a, **k): raise AssertionError('dry qualification must not reserve or dispatch physics')
p1._reserve = forbidden; production.run_branch = forbidden
import tools.coopnative.storage as old; old.reserve = forbidden
receipt = dict(started=time.time(), source=src)
try:
    work = out/'g1work'; rc.init_root(work)
    for arm in ('B', 'R'):
        stage.readonly_bank_view(Path(c3.CLEAN1)/'replay_bank'/arm, work/'replay_bank'/arm)
    cfg, runtime = c3.g1_config(src, work)
    from tools.coopmwm import g1_certification as g1
    selected, info = g1.prepare(cfg, runtime)
    receipt['g1'] = dict(candidates=info['candidates'], selected=info['selected'], quote=info['quote'],
        jobs=[dict(arm=i['job']['lesson']['arm'], family=i['job']['lesson']['macro']['family'],
                   members=len(i['job']['lesson']['macro']['members']), sim_s=i['job']['sim_s']) for i in selected],
        diagnostics=info['diagnostics'][:20], runtime_sha256=runtime)
    cfg['run_root'] = str(work)
    receipt['mined'] = c3.admit_mined(Path(c3.MINING_ROOT)/'MINED_MOVES.jsonl', cfg, out/'mined_admitted.jsonl')
    fake = out/'g1fake'; rc.init_root(fake)
    training.write(fake/'CLEAN3_GO_NOGO.json', _seal(dict(decision='NO-GO', runtime_sha256=runtime,
        qualification_only=True), 'receipt_sha256'))
    cfg3 = c3.prepare(Path(src), root=out/'root', g1_root=fake, mined=receipt['mined'])
    receipt['prepare'] = dict(status='PASS', config_sha256=digest(cfg3), restart={k: v for k, v in cfg3['clean3_restart'].items()
        if k in ('diagnostic_quotes', 'gate_sim_s', 'lesson_quotes', 'g1_go_nogo')},
        budget=dict(remaining=cfg3['clean3_budget']['remaining_sim_s'], prior=cfg3['clean3_budget']['protected_prior_sim_s']))
    sys.path.insert(0, src+'/ops'); import clean3_chain as chain
    from tools.coopmwm.storage import ledger
    r = cfg3['clean3_restart']
    receipt['coverage'] = chain.coverage(cfg3, ledger(Path(c3.CLEAN2)), gate_sim_s=r['gate_sim_s'],
        diagnostic_quotes=r['diagnostic_quotes'], lesson_quotes=r['lesson_quotes'])
    receipt['status'] = 'PASS'
except BaseException as error:
    receipt.update(status='FAILED', error=repr(error), traceback=traceback.format_exc())
receipt['finished'] = time.time()
(out/'DRYQ_RESULT.json').write_text(json.dumps(receipt, indent=1, default=str))
