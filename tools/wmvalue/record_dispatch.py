"""Bind frozen reserved arrays and the actual one-time dispatch, no test metrics."""
import json
from pathlib import Path

from .audit import digest


def main():
    root = Path(__file__).resolve().parents[2]
    out = root/'_out'
    launch = json.loads((out/'wmvalue_inference_v1/LAUNCH.json').read_text())
    real = json.loads((out/'wmvalue_reserved_real_v1.json').read_text())
    assert digest(out/'wmvalue_reserved_real_v1.npz') == real['arrays_sha256']
    deltas = [abs(v) for r in real['records'] for h in r['clock_differences'].values() for v in h.values()]
    source = [abs(h['source']) for r in real['records'] for h in r['clock_differences'].values()]
    doc = dict(schema='wmvalue_actual_dispatch/1', launch=launch,
        source_commit='226b0a8e', reserved_states=real['states'], reserved_worlds=real['worlds'],
        objective_checks=3*len(real['records']), max_real_clock_difference=max(deltas),
        max_source_clock_difference=max(source), arrays_sha256=real['arrays_sha256'],
        real_export_receipt_sha256=digest(out/'wmvalue_reserved_real_v1.json'),
        prefix_archive_sha256=digest(out/'wmvalue_reserved_prefix_v1.zip'),
        adapter_gate_sha256=digest(out/'wmvalue_cpu_audit_v3.json'),
        real_fit_or_selection=False, frozen_efficacy_test_executed=False,
        live_receipts='_out/wmvalue_inference_v1/{LAUNCH.json,DISPATCH.log,claims,tasks}',
        sumo_seconds=0, peer_signals=0, policy_training=False)
    target = root/'tools/wmvalue/evidence/inference_dispatch.json'
    with target.open('x') as stream:
        json.dump(doc, stream, indent=1, allow_nan=False)
    print(json.dumps(doc, indent=1), flush=True)


if __name__ == '__main__':
    main()
