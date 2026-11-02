"""Small stdlib-only local publication check against the cpuQ verification."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/rbfit'))
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--publication', type=Path, action='append', default=[],
                        help='Also check a later LC publication against its VERIFIED.json')
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
    publications = []
    for directory in args.publication:
        receipt_bytes = (directory/'VERIFIED.json').read_bytes()
        receipt = json.loads(receipt_bytes)
        assert receipt['status'] == 'PASS', str(directory)
        artifact_hashes = receipt['artifact_sha256']
        assert artifact_hashes, str(directory)
        for relative, expected in artifact_hashes.items():
            payload = (directory/relative).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == expected, str(directory/relative)
        publications.append(dict(publication=directory.name,
                                 verified_artifacts=len(artifact_hashes),
                                 verification_sha256=hashlib.sha256(receipt_bytes).hexdigest()))
    paper = Path(__file__).resolve().parents[2]/'docs/paper_results/RBFIT.md'
    lines = len(paper.read_text(encoding='utf-8').splitlines())
    assert lines <= 40
    report = dict(status='PASS', verified_artifacts=len(sizes), sizes_bytes=sizes,
                  paper_lines=lines, source_verification_sha256=hashlib.sha256(
                      args.verification.read_bytes()).hexdigest())
    if publications:
        report['later_publications'] = publications
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
