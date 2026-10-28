"""Verify controller delivery and repeated scientific outputs from owned receipts."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
raw = ROOT/'evidence_entrant/raw'
totals = {}
repeated = []
counts = {}
secondary = {'max_absolute_J_difference': 0., 'draw_dJ_sign_mismatches': [],
             'complete_K8_group_dJ_sign_mismatches': []}
groups = {}
sign = lambda x: (x > 0)-(x < 0)
for host in ('cpuP', 'cpuQ'):
    for path in (raw/host/'results_v4').glob('e3_*_s600.json'):
        doc = json.loads(path.read_text())
        counts[doc['variant']] = counts.get(doc['variant'], 0)+1
        if not doc['entrant_control']:
            continue
        groups.setdefault((doc['pair_key'], doc['variant']), []).append(doc)
        names = list(doc['arms'])
        for h in (10, 30, 60, 90, 180, 240):
            computed = {}
            for arm in names:
                a = doc['arms'][arm]
                computed[arm] = math.fsum(r[3] for r in a['timeline'] if r[1] == 'HDV' and r[2] <= h)
                secondary['max_absolute_J_difference'] = max(secondary['max_absolute_J_difference'],
                    abs(computed[arm]-a['delay'][str(h)][0]))
            for arm in names[1:]:
                direct = doc['arms'][arm]['delay'][str(h)][0]-doc['arms'][names[0]]['delay'][str(h)][0]
                derived = computed[arm]-computed[names[0]]
                if sign(direct) != sign(derived):
                    secondary['draw_dJ_sign_mismatches'].append([doc['pair_key'], doc['variant'], doc['draw'], arm, h, direct, derived])
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
for (key, variant), docs in groups.items():
    if len(docs) != 8:
        continue
    names = list(docs[0]['arms'])
    for h in (10, 30, 60, 90, 180, 240):
        for arm in names[1:]:
            direct = math.fsum(d['arms'][arm]['delay'][str(h)][0]-d['arms'][names[0]]['delay'][str(h)][0]
                               for d in docs)/8
            derived = math.fsum(math.fsum(r[3] for r in d['arms'][arm]['timeline'] if r[1] == 'HDV' and r[2] <= h)-
                                math.fsum(r[3] for r in d['arms'][names[0]]['timeline'] if r[1] == 'HDV' and r[2] <= h)
                                for d in docs)/8
            if sign(direct) != sign(derived):
                secondary['complete_K8_group_dJ_sign_mismatches'].append([key, variant, arm, h, direct, derived])
off = json.loads((raw/'cpuP/OFF_BIT_IDENTITY.json').read_text())
assert off['arms_identical'] and off['both_repeat_identical']
result = dict(counts=counts, off_full_horizon_bit_identity=off, repeated_final_source=repeated,
              exposure={variant+'/'+arm: dict(cav_vehicle_steps=a, non_native_command_steps=b)
                        for (variant, arm), (a, b) in totals.items()})
(ROOT/'evidence_entrant/CONTROLLER_AUDIT.json').write_text(json.dumps(result, indent=1))
(ROOT/'evidence_entrant/REDUCTION_PARITY.json').write_text(json.dumps(secondary, indent=1))
print(json.dumps(result))
print(json.dumps({k: len(v) if isinstance(v, list) else v for k, v in secondary.items()}))
