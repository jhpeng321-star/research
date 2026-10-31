"""Verify private config inheritance preserves the original WMCALIB WM semantics."""
import json
from pathlib import Path

from .audit import digest, safe


def main():
    root = Path(__file__).resolve().parents[2]
    reference = safe('E:/agent-work/c4-wmcalib/tools/wmcalib/evidence/gpuR_canary_IDENTITY.json')
    worker = root/'_out/wmvalue_fit_worker_gpuR_v2.json'
    old, new = json.loads(reference.read_text()), json.loads(worker.read_text())
    fields = ('checkpoint_sha256', 'config_sha256', 'courtesy_state_grad',
              'courtesy_intent', 'courtesy_off', 'physics')
    differences = [k for k in fields if old['wm'][0][k] != new['wm'][0][k]]
    assert not differences, 'frozen WM semantics differ: '+repr(differences)
    assert old['candidate_sha256'] == new['candidate_sha256']
    assert old['export_sha256'] == new['export_sha256']
    doc = dict(schema='wmvalue_original_WM_binding/1', verdict='PASS',
        reference_sha256=digest(reference), worker_sha256=digest(worker),
        exact_fields=list(fields), candidate_exact=True, mappo_export_exact=True,
        all_resolved_physics_fields_identical=True, test_metrics_examined=False)
    with (root/'tools/wmvalue/evidence/original_wm_binding.json').open('x') as stream:
        json.dump(doc, stream, indent=1)
    print(json.dumps(doc), flush=True)


if __name__ == '__main__':
    main()
