"""One-shot public-receipt audit of the owner-paused r12; no runtime imports."""
import argparse
import json
from pathlib import Path
import subprocess


REMOTE = r'''
ROOT = Path('/home/ubuntu/coopmwm20261106_clean3_r6')
CONTROL = Path('/home/ubuntu/cpuv_setup_20261105/clean3_chain_3e5bc9f9')
SOURCE = Path('/home/ubuntu/coopmwm20261106_clean3_src_3e5bc9f9')
def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def public(path):
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError('receipt escaped paused root')
    return json.loads(path.read_text())
out = probe(str(ROOT), str(CONTROL))
stop_path = ROOT/'STOPPED_BY_OWNER.json'
assert file_sha(stop_path) == 'dfb53061922f922073bba37ce312cb8a528a5c38e54836f66fe7c61415f1a584'
out['owner_stop'] = public(stop_path)
out['owner_stop_sha256'] = file_sha(stop_path)
authority = (ROOT/'AUTHORITY.md').read_text()
assert 'pause' in authority.lower() and 'measure' in authority.lower()
out['authority_sha256'] = file_sha(ROOT/'AUTHORITY.md')
out['authority_pause_excerpt'] = authority[-6000:]
assert 'STOP.json' in out['faults'] and 'FAULT.json' in out['faults']
assert all(not x['alive'] and not x['complete'] for x in out['owners'].values())
assert out['chain']['status'] == 'FAILED' and out['chain']['step'] == 'MONITOR'
chain = json.loads((CONTROL/'CHAIN_STARTED.json').read_text())
stat = Path('/proc')/str(chain['pid'])/'stat'
fields = stat.read_text().rsplit(')', 1)[1].split() if stat.exists() else []
chain['alive'] = bool(fields and fields[0] != 'Z' and fields[19] == str(chain['start_ticks']))
assert not chain['alive']
out['chain_owner'] = chain
out['sealed_updates'] = {}
for arm in ('A', 'B', 'R'):
    updates = []
    for u in range(1, 25):
        path = ROOT/f'{arm}_UPDATE_{u:03}.json'
        if not path.is_file():
            continue
        row = sealed(public(path), 'update_sha256')
        assert row['config_sha256'] == out['launch']['runtime_sha256']
        checkpoint = Path(row['checkpoint'])
        assert checkpoint.resolve().is_relative_to(ROOT)
        assert file_sha(checkpoint) == row['checkpoint_sha256']
        updates.append(dict(update=u, update_sha256=row['update_sha256'],
            checkpoint_sha256=row['checkpoint_sha256'], safety=row['safety']))
    out['sealed_updates'][arm] = updates
assert {a: [x['update'] for x in v] for a, v in out['sealed_updates'].items()} == dict(A=[1], B=[], R=[])
out['u0_checkpoints'] = {a: file_sha(ROOT/f'{a}_checkpoint_000.pt') for a in ('A', 'B', 'R')}
manifest = json.loads((SOURCE/'SOURCE_MANIFEST.json').read_text())
assert manifest['commit'] == out['launch']['runtime_commit']
changed = [p for p, expected in manifest['files'].items() if file_sha(SOURCE/p) != expected]
assert not changed, changed
out['source_files_verified'] = len(manifest['files'])
out['source_manifest_sha256'] = file_sha(SOURCE/'SOURCE_MANIFEST.json')
ledger = ROOT/'CHARGES.jsonl'
rows = [json.loads(line) for line in ledger.read_bytes().splitlines()]
assert len(rows) == 118 and out['charged_sim_s'] == 120760
out['ledger_sha256'] = file_sha(ledger)
out['ledger_rows'] = len(rows)
g4 = [r for r in rows if r['phase'] == 'g4_synergy_credit']
assert len(g4) == 16 and sum(r['sim_s'] for r in g4) == 19200
out['g4_reservation'] = dict(jobs=len(g4), charged_sim_s=19200,
    result_files_present=[r['id'] for r in g4 if (ROOT/'jobs'/r['id']/'RESULT.json').exists()])
assert not out['g4_reservation']['result_files_present']
out['preserved_hashes'] = {name: file_sha(ROOT/name) for name in
    ('STOP.json', 'FAULT.json', 'CONFIG.json', 'LAUNCHED.json')}
out.update(schema='coopmwm.clean3.owner_paused_audit.v1', status='VERIFIED_OWNER_PAUSED',
    new_dispatch=False, new_physics=False, process_signals=False, remote_writes=False)
print(json.dumps(out, sort_keys=True))
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    target = Path(args.receipt)
    if not target.parent.is_dir() or target.exists():
        raise ValueError('existing parent and new receipt required')
    code = Path(__file__).with_name('clean3_follow_remote.py').read_text() + REMOTE
    helper = (
        "import sys;sys.dont_write_bytecode=True;"
        "sys.path.insert(0,'/mnt/e/agent-work/c4-rerun-launch');"
        "from tools.coopmwm.cpuv_ssh import Transport;"
        "r=Transport().run('python3 -B -',timeout=40,"
        "input=sys.stdin.buffer.read(),capture_output=True);"
        "sys.stdout.buffer.write(r.stdout);sys.stderr.buffer.write(r.stderr);"
        "sys.exit(r.returncode)")
    result = subprocess.run(
        ['wsl', '-d', 'Ubuntu-22.04', '--', 'python3', '-c', helper],
        input=code.encode(), capture_output=True, timeout=55)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace').replace('\x00', ''))
    value = json.loads(result.stdout.decode().replace('\x00', ''))
    with target.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
    print(json.dumps({k: value[k] for k in ('timestamp', 'status', 'charged_sim_s',
        'ledger_rows', 'ledger_sha256', 'owner_stop_sha256', 'source_files_verified',
        'sealed_updates', 'g4_reservation')}, sort_keys=True))


if __name__ == '__main__':
    main()
