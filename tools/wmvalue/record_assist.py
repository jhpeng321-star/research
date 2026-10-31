"""Snapshot the admitted disjoint CPU partition, preserving all live processes."""
import json
from pathlib import Path

from .audit import digest


def main():
    repo = Path(__file__).resolve().parents[2]
    root = repo/'_out/wmvalue_inference_v1'
    launch = json.loads((root/'ASSIST_CPU_R_LAUNCH.json').read_text())
    ack = json.loads((root/'ASSIST_CPU_R_ACK.json').read_text())
    assert launch['pid'] == ack['pid'] and len(ack['tasks']) == 40
    assert all(t[0] == 969439 for t in ack['tasks'])
    doc = dict(schema='wmvalue_actual_cpu_assist/1', source_commit='aba9ed4a',
        pid=launch['pid'], host='gpuR', device='cpu', lanes=8, reserved_world=969439,
        exclusive_tasks=40, worker_snapshot='inference_v2',
        primary_dispatcher_pid=28924, original_workers_preserved=True,
        launch_sha256=digest(root/'ASSIST_CPU_R_LAUNCH.json'),
        ack_sha256=digest(root/'ASSIST_CPU_R_ACK.json'),
        authorization='MAIN19:52 CPU release to C1/WM',
        initial_admission='10cores +40GiB spare', evaluator_changed=False,
        value_refit=False, sumo_seconds=0, peer_signals=0,
        progress_receipt='inference_cpu_assist_progress.json')
    with (repo/'tools/wmvalue/evidence/assist_dispatch.json').open('x') as stream:
        json.dump(doc, stream, indent=1)
    print(json.dumps(doc), flush=True)


if __name__ == '__main__':
    main()
