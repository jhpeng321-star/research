"""Read-only, explicitly scoped pilot timing collector (no model/GPU imports)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time


REMOTE = r'''
import json, os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
forbidden = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth')
assert root.is_absolute() and not any(s in str(root).lower() for s in forbidden)
out = dict(root=str(root), epoch=time.time(), stages={})
for name in ('OWNERSHIP.json', 'dry_OWNERSHIP.json', 'pilot_OWNERSHIP.json',
             'dry_EXIT.json', 'pilot_EXIT.json', 'GPU_ADMISSION.jsonl'):
    path = root/name
    if path.is_file():
        out[name] = path.read_text()
        if name.endswith('OWNERSHIP.json'):
            owned = json.loads(out[name])
            if 'pid' in owned:
                proc = Path('/proc')/str(owned['pid'])
                if (proc/'stat').is_file():
                    stat = (proc/'stat').read_text().rsplit(')', 1)[1].split()
                    if stat[19] == owned.get('start_ticks'):
                        out[name+'_proc'] = dict(user_ticks=int(stat[11]), system_ticks=int(stat[12]),
                            threads=int(stat[17]), uptime_s=float(Path('/proc/uptime').read_text().split()[0]),
                            ticks_per_second=os.sysconf('SC_CLK_TCK'), start_ticks=int(stat[19]))
                        env = (proc/'environ').read_bytes().split(b'\0')
                        out[name+'_env'] = [v.decode(errors='replace') for v in env if
                            v.startswith((b'WMS_', b'OMP_NUM_THREADS=', b'MKL_NUM_THREADS=', b'OPENBLAS_NUM_THREADS='))]
for mode in ('dry', 'pilot'):
    stage = root/(mode+'_EXPLORATORY')
    if not stage.is_dir():
        continue
    doc = {}
    for name in ('RUN_SUMMARY.json', 'frozen_scoring_baseline.json', 'gate_panel_result.json',
                 'updates.jsonl', 'rescores.jsonl', 'memory_execution.jsonl', 'native_curve.jsonl',
                 'speedprof_execution.jsonl'):
        path = stage/name
        if path.is_file():
            doc[name] = dict(bytes=path.stat().st_size, mtime=path.stat().st_mtime,
                             text=path.read_text())
    log = root/(mode+'.log')
    if log.is_file():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-32000))
            doc['log_tail'] = stream.read().decode(errors='replace')
    out['stages'][mode] = doc
print(json.dumps(out))
'''


def collect(host, root):
    result = subprocess.run(['ssh', host, '/root/miniconda3/bin/python', '-', root],
                            input=REMOTE, text=True, capture_output=True, check=True)
    return json.loads(result.stdout)


def summarize(doc):
    result = dict(epoch=doc['epoch'], root=doc['root'], stages={})
    for mode, stage in doc['stages'].items():
        rows = [json.loads(line) for line in stage.get('updates.jsonl', {}).get('text', '').splitlines()]
        events = [json.loads(line) for line in stage.get('memory_execution.jsonl', {}).get('text', '').splitlines()]
        spans, starts = [], {}
        for event in events:
            key = (event['update'], event.get('start'), event.get('future'))
            name = event['event']
            if name == 'full_forward_start':
                starts[key] = event['t']
            elif name == 'full_forward_done' and key in starts:
                spans.append(dict(update=event['update'], phase='full_K_forward', seconds=event['t']-starts[key]))
            elif name == 'future_start':
                starts[key] = event['t']
            elif name == 'future_vjp_done' and key in starts:
                spans.append(dict(update=event['update'], phase='dense_replay', future=event.get('future'),
                                  seconds=event['t']-starts[key]))
        result['stages'][mode] = dict(rows=rows, spans=spans, events=len(events),
            last_event=events[-1] if events else None,
            log_tail=stage.get('log_tail'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='gpuA')
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    doc = collect(args.host, args.root)
    doc.update(host=args.host, collected_epoch=time.time())
    args.output.write_text(json.dumps(doc, indent=1)+'\n', encoding='utf-8')
    print(json.dumps(summarize(doc), indent=1))


if __name__ == '__main__':
    main()
