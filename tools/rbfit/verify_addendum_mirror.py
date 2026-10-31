"""Small stdlib-only local publication check against the cpuQ verification."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/rbfit'))
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    verification = json.loads(args.verification.read_bytes())
    assert verification['status'] == 'PASS'
    sizes = {}
    for relative, expected in verification['artifact_sha256'].items():
        path = args.root/relative
        payload = path.read_bytes()
        assert hashlib.sha256(payload).hexdigest() == expected, relative
        sizes[relative] = len(payload)
    paper = Path(__file__).resolve().parents[2]/'docs/paper_results/RBFIT.md'
    lines = len(paper.read_text(encoding='utf-8').splitlines())
    assert lines <= 40
    report = dict(status='PASS', verified_artifacts=len(sizes), sizes_bytes=sizes,
                  paper_lines=lines, source_verification_sha256=hashlib.sha256(
                      args.verification.read_bytes()).hexdigest())
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
