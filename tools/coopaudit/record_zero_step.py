"""Reconstruct the setup-only failure from owned traceback; retain full charge."""
import json
from pathlib import Path
root=Path('/root/coopaudit20261031')
log=(root/'canary_1793433398.log').read_text()
assert 'CavEnvError: SUMO binary not found on PATH' in log
assert 'runner.py\", line 133, in execute' in log
assert not (root/'jobs/canary_s969699/scenarios').exists()
assert not (root/'jobs/canary_s969699/result.json').exists()
receipt=dict(actual_sim_s=0,charged_sim_s=300,reason='SUMO executable missing from noninteractive SSH PATH; constructor before reset/start_sumo',
             original_attempt='canary_s969699',original_log='canary_1793433398.log',
             repair='one manual zero-step attempt2, original charge retained, same seed, no world replacement')
(root/'jobs/canary_s969699/failure.json').write_text(json.dumps(receipt,indent=1)+'\n')
(root/'ZERO_STEP_REPAIR.json').write_text(json.dumps(receipt,indent=1)+'\n')
print(json.dumps(receipt,indent=1))
