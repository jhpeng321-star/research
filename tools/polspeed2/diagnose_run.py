from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('name', choices=('nine_N8', 'nine_N8_gate', 'nine_N4', 'nine_N4_s8', 'six_N4', 'six_N8'))
args = parser.parse_args()
root = Path('/root/autodl-tmp/c4/polspeed2_20261028/out')/args.name
print('console', (root/'console.log').read_text()[-12000:], flush=True)
logs = [p for p in root.glob('future_workers/*/worker*.log') if p.stat().st_size]
print('nonempty worker logs', len(logs), flush=True)
if logs:
    print(logs[0], logs[0].read_text()[-12000:], flush=True)
errors = sorted(root.glob('future_workers/*/[0-7]/*.error'))
print('error files', len(errors), flush=True)
if errors:
    print(errors[0], errors[0].read_text()[-12000:], flush=True)
