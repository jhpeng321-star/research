"""Verify controller delivery and repeated scientific outputs from owned receipts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
raw = ROOT/'evidence_entrant/raw'
totals = {}
repeated = []
counts = {}
for host in ('cpuP', 'cpuQ'):
    for path in (raw/host/'results_v3').glob('e3_*_s600.json'):
        doc = json.loads(path.read_text())
        counts[doc['variant']] = counts.get(doc['variant'], 0)+1
        if not doc['entrant_control']:
            continue
        audit = doc['controller_audit']
        for i, arm in enumerate(doc['arms']):
            row = totals.setdefault((doc['variant'], arm), [0, 0])
            row[0] += audit['entrant_cav_steps'][i]
            row[1] += audit['non_native_command_steps'][i]
            if arm == 'native':
                assert audit['non_native_command_steps'][i] == 0
        if doc.get('repeat_identical'):
            repeated.append(str(path.relative_to(raw)))
for (_, arm), (exposure, commands) in totals.items():
    if arm != 'native' and exposure:
        assert commands > 0
off = json.loads((raw/'cpuP/OFF_BIT_IDENTITY.json').read_text())
assert off['arms_identical'] and off['both_repeat_identical']
result = dict(counts=counts, off_full_horizon_bit_identity=off, repeated_final_source=repeated,
              exposure={variant+'/'+arm: dict(cav_vehicle_steps=a, non_native_command_steps=b)
                        for (variant, arm), (a, b) in totals.items()})
(ROOT/'evidence_entrant/CONTROLLER_AUDIT.json').write_text(json.dumps(result, indent=1))
print(json.dumps(result))
