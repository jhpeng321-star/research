"""Save an exclusive r12 recovery receipt through the existing bounded F6 transport."""
import argparse
import json
from pathlib import Path
import subprocess

from clean3_follow import fetch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kind', choices=('launch', 'progress'), required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    target = Path(args.receipt)
    if not target.parent.is_dir() or target.exists():
        raise ValueError('existing parent and new receipt required')
    root = '/home/ubuntu/coopmwm20261106_clean3_r6'
    control = '/home/ubuntu/cpuv_setup_20261105/clean3_chain_3e5bc9f9'
    if args.kind == 'progress':
        value = fetch(root, control)
    else:
        code = (repo/'ops/clean3_resume_w4_remote.py').read_text(encoding='utf-8')
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
    print(json.dumps(dict(receipt=str(target), timestamp=value['timestamp'],
        root=value['root'], status=value.get('status', value.get('chain')),
        owners=value.get('launch_owners', value.get('owners')),
        charged_sim_s=value.get('charged_sim_s', value.get('root_charges', {}).get('total_sim_s')),
        monitors=sorted(value.get('monitors', {}), key=int), faults=value.get('faults')),
        sort_keys=True))


if __name__ == '__main__':
    main()
