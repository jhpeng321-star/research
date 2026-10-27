"""Wait only for the task's completion receipt (never signal any process)."""
import json
from pathlib import Path
import time

root = Path.home()/'longhzn'
deadline = time.monotonic()+600
while not (root/'DONE.json').exists():
    rows = list((root/'rows').glob('*.json'))
    print(json.dumps({'rows':len(rows),'waiting':True}),flush=True)
    if time.monotonic() >= deadline:
        raise TimeoutError('CPU task not complete; retained running')
    time.sleep(30)
print((root/'DONE.json').read_text())
