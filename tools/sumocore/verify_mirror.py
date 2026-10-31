"""Local receipt integrity and requested report-length check; no compute dispatch."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]


def main():
    report=ROOT/'docs/paper_results/SUMOCORE.md'
    lines=len(report.read_text(encoding='utf-8').splitlines());assert lines<=50
    data=report.parent/'sumocore'
    hashes=json.loads((data/'MIRROR_SHA256.json').read_text())
    for name,expected in hashes.items():
        path=data/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,name
    result=json.loads((data/'scores/verification.json').read_text())
    assert result['status']=='PASS' and result['actual_sim_s']==87517.5
    assert result['scientific_validity']=='FAIL' and len(result['invalid_attempts'])==18
    assert result['precharged_sim_s']==90150<100000
    print(json.dumps(dict(status='PASS',mirrored_files=len(hashes),report_lines=lines,
                         actual_sim_s=result['actual_sim_s'],scientific_validity=result['scientific_validity'])))


if __name__=='__main__':main()
