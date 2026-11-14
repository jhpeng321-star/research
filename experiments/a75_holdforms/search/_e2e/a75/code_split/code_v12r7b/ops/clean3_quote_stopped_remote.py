"""Read-only actual stopped-ledger quote; operator sources supplied in memory."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import types

from tools.coopmwm import launch_training as training, launch_production as production
from tools.coopmwm import track2_backend as p1
from tools.coopmwm.protocol import digest
from tools.coopmwm.storage import ledger

root = Path('/home/ubuntu/coopmwm20261105_clean2')
cfg = training.checked_config(root/'CONFIG.json')
launched = training.read(root/'LAUNCHED.json')
assert digest(cfg) == launched['runtime_sha256']
assert hashlib.sha256((root/'CONFIG.json').read_bytes()).hexdigest() == launched['config_file_sha256']
assert (root/'STOP.json').is_file() and (root/'FAULT.json').is_file()
for owner in launched['owners'].values():
    path = Path('/proc')/str(owner['pid'])/'stat'
    fields = path.read_text().split() if path.exists() else []
    assert not (fields and fields[2] != 'Z' and fields[21] == str(owner['start_ticks']))


def forbidden(*args, **kwargs):
    raise AssertionError('stopped-ledger native quote must not load a current policy or dispatch physics')


p1._reserve = forbidden
production.run_branch = forbidden
production.checkpoint_weights = forbidden
sources = OPERATOR_SOURCES
modules = {}
for name in ('launch_evaluation', 'clean3_budget'):
    module = types.ModuleType('tools.coopmwm._operator_'+name)
    module.__package__ = 'tools.coopmwm'
    module.__file__ = '<operator-only-'+name+'>'
    exec(compile(sources[name], module.__file__, 'exec'), module.__dict__)
    modules[name] = module
modules['launch_evaluation'].ProcessPoolExecutor = forbidden
lessons = modules['launch_evaluation'].lesson_quotes(cfg)
before = hashlib.sha256((root/'CHARGES.jsonl').read_bytes()).hexdigest()
clean1_rows = ledger(Path(cfg['clean2_restart']['clean1_root']))
clean2_rows = ledger(root)
quote = modules['clean3_budget'].prospective(clean1_rows=clean1_rows,
    clean2_rows=clean2_rows, owner_budget=OWNER_BUDGET,
    diagnostic_quotes=cfg['clean2_budget']['diagnostic_quotes'],
    gate_manifest=cfg['clean2_budget']['gate_manifest'], lesson_quotes=lessons)
assert before == hashlib.sha256((root/'CHARGES.jsonl').read_bytes()).hexdigest()
mining = Path('/home/ubuntu/coopmwm20261106_s_mining')
mining_rows = [json.loads(line) for line in (mining/'CHARGES.jsonl').read_text().splitlines() if line]
assert len({row['id'] for row in mining_rows}) == len(mining_rows)
assert all(type(row['sim_s']) is int and row['sim_s'] >= 0 for row in mining_rows)
mining_spent = sum(row['sim_s'] for row in mining_rows)
assert mining_spent <= 500000
print(json.dumps(dict(schema='coopmwm.clean3.stopped_quote_operator.v1',
    timestamp=datetime.now(timezone.utc).isoformat(), read_only=True,
    operator_only_not_deployed=True, native_quote_qualification='PASS',
    forbidden_current_policy_load_and_reservation_and_dispatch=True,
    operator_source_sha256={name: hashlib.sha256(code.encode()).hexdigest()
        for name, code in sources.items()}, quote=quote, clean2_ledger_sha256=before,
    fault_sha256=hashlib.sha256((root/'FAULT.json').read_bytes()).hexdigest(),
    mining=dict(charged_sim_s=mining_spent, unspent_reserved_sim_s=500000-mining_spent,
        allocation_sim_s=500000,
        registration_sha256=hashlib.sha256((mining/'MINING_REGISTRATION.json').read_bytes()).hexdigest()),
    overall_cap_sim_s=6500000,
    overall_cumulative_bound_sim_s=quote['cumulative_upper_bound_sim_s']+500000,
    overall_shortfall_sim_s=max(0, quote['cumulative_upper_bound_sim_s']+500000-6500000),
    launch_authorized=False), sort_keys=True, allow_nan=False))
