"""Verify compact engineering evidence without simulation, training, or remote access."""
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]


def main():
    report = ROOT / 'LOOPLONG.md'
    words = len(report.read_text(encoding='utf-8').split())
    assert words <= 500, words
    evidence = ROOT / 'tools/looplong'
    dry = json.loads((evidence / 'DRY_RUN.json').read_text())
    tests = (evidence / 'TESTS.log').read_text()
    passed = int(re.search(r'(\d+) passed', tests).group(1))
    assert 'failed' not in tests.lower() and dry['cuda_used'] is False
    assert len(dry['wm_updates']) == 2 and len(dry['policy_updates']) == 3
    assert all(math.isfinite(row['loss']) for row in dry['wm_updates'])
    assert dry['actual_ft_speed']['objective_checkpoint_rng_byte_equal']
    assert dry['acceptance_rollout_probe']['metrics_identical']
    paths = [report, *(evidence / name for name in ('DRY_RUN.json', 'TESTS.log', 'CYCLE1_TIMING.json'))]
    print(json.dumps({'report_words': words, 'tests_passed': passed,
        'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}, indent=1))


if __name__ == '__main__':
    main()
