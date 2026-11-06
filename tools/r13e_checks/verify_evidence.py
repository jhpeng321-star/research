"""Validate copied study evidence and independently reconstruct its clustered CIs."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.r13e_checks import offline as study  # noqa: E402
import numpy as np  # noqa: E402

directory = ROOT/'docs/r13e_offline'


def load(name):
    return json.loads((directory/name).read_text(encoding='utf-8'))


def text_digest(path):
    # Git's Windows checkout may use CRLF; deployed source/evidence uses LF.
    return hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()


audit = load('final_audit.json')
assert audit['status'] == 'passed' and audit['checkpoint_unchanged'] and not audit['cuda_initialized']
for name,digest in audit['evidence_sha256'].items():
    if name == 'linux_tests.log':
        log = load('linux_tests.json')
        assert log['exit_code'] == 0 and log['passed'] == 70
        assert hashlib.sha256(log['stdout'].encode('utf-8')).hexdigest() == digest == log['raw_log_sha256']
    else:
        assert text_digest(directory/name) == digest, name
report, panel, base, gs = [load(name) for name in ('report.json','panel_receipt.json','base_checks.json','gs_equivalence.json')]
assert report['status'] == 'complete' and report['strict_exact_weights']
assert report['workers'] == 4 and report['threads'] == 1 and report['linux_nice'] == 19 and report['device'] == 'cpu'
assert report['states_sha256'] == panel['output']['sha256']
assert panel['role_counts'] == {'train':48,'val':0} and len(panel['selected_episodes']) == 48
assert panel['original_panel_audit']['role_counts'] == {'train':0,'val':48,'unknown':0}
assert report['one_step_windows'] == base['one_step_windows'] and len(base['one_step_windows']) == 48
rows = report['rollouts']['rows']
assert len(rows) == 96 and {(r['state'],r['replicate']) for r in rows} == {(i,k) for i in range(48) for k in range(2)}
for row in rows:
    assert row['seed'] == study.RNG_SEED+2*row['state']+row['replicate']
    assert all(arm in row for arm in study.ARMS)
groups = np.array([r['state'] for r in rows])
checked = 0
for arm,stats in report['rollouts']['state_cluster_run_mean'].items():
    later,earlier = study.CONTRASTS.get(arm,(arm,arm))
    for metric,stored in stats.items():
        values = []
        for row in rows:
            a,b = row[later][metric],row[earlier][metric]
            values.append(float('nan') if a is None or b is None else float(a)-float(b) if arm in study.CONTRASTS else float(a))
        actual = study.json_safe(study._estimate(values,groups))
        assert actual == stored, (arm,metric,actual,stored)
        checked += 1
assert gs['same_event_inputs'] and gs['recorded_events'] == 256 and gs['verdict'] == 'NOT_EQUIVALENT'
for dtype in ('torch.float32','torch.float64'):
    assert gs['events']['all'][dtype]['boundary_different'] == 120
    assert gs['events']['hdv'][dtype]['boundary_different'] == 97
for arm in study.ARMS:
    assert base['verdicts'][arm]['rules']['BASE-1'][0] == 'REJECT'
    assert all(base['verdicts'][arm]['rules'][name][0] == 'PASS' for name in ('BASE-2','BASE-3','BASE-5'))
    for cls,stats in base['classes'][arm].items():
        assert stats['clusters'] >= 2 and stats['n'] >= 5, (arm,cls)
assert audit['emergency_vs_emergency_gs_one_step_arrays_exact']
assert text_digest(ROOT/'tools/r13e_checks/offline.py') == audit['source_provenance']['tooling_overlay_sha256']['tools/r13e_checks/offline.py']
assert text_digest(ROOT/'tools/r13e_checks/gs_equivalence.py') == audit['gs_comparator_sha256']
assert text_digest(ROOT/'tools/r13e_checks/postprocess_remote.py') == audit['postprocess_script_sha256']
markdown = (ROOT/'R13E_OFFLINE.md').read_text(encoding='utf-8')
for key,label in [('recorded_braking','Recorded braking'),('last3_target_follower','Last-3-s target follower'),('all_lane1','All lane 1')]:
    fields=[]
    for unit in ('bias','mae'):
        value=base['contrasts']['emergency_minus_off'][key][unit]
        prefix='+' if unit == 'mae' else ''
        fields.append((f"{prefix}{value['mean']:.3f} [{value['lo']:.3f}, {value['hi']:.3f}]").replace('-', '−'))
    assert all(field in markdown for field in fields), (label,fields)
print(f'Evidence verified: 288 rollouts, 48 same-panel BASE windows, {checked} reconstructed cluster estimates/CIs, 256 same-event gs decisions, exact hashes and tensor/source identities.')
