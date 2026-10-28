from pathlib import Path
import json

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
for run in (base/'out/nine_N2', base/'failed_shared_monitor/nine_N2'):
    for path in sorted(run.glob('future_workers/*/[0-7]/memory_execution.jsonl')):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        print(str(path.relative_to(base)))
        for event in ('future_start', 'future_forward_done', 'future_vjp_done'):
            selected = [r for r in rows if r['event'] == event]
            if selected:
                print(event, 'max_allocated', max(r['allocated_bytes'] for r in selected),
                      'max_reserved', max(r['reserved_bytes'] for r in selected))
        peak = path.with_name('memory.json')
        if peak.exists():
            print('peak', peak.read_text())
