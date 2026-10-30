"""Check every numeric report table against retained evidence and bind fast adapter."""
import ast
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def main():
    report = ROOT / 'docs/paper_results/WMREQUAL2.md'
    lines = report.read_text(encoding='utf-8').splitlines()
    assert len(lines) <= 60
    data = json.loads((HERE / 'evidence/ANALYSIS.json').read_text(encoding='utf-8'))
    runtime = json.loads((HERE / 'evidence/RUNTIME.json').read_text(encoding='utf-8'))
    text_runtime = json.loads((HERE / 'evidence/RUNTIME_TEXT.json').read_text(encoding='utf-8'))
    assert sha(ROOT / 'tools/wmrequal/evidence/SOURCE.json') == runtime['source_sha256']
    for name, receipt in runtime['files'].items():
        # Original staging preserved mixed Windows/LF bytes. A fresh checkout
        # normalizes those bytes, so bind remote raw receipt AND canonical text.
        text_receipt = text_runtime['files'][name]
        assert text_receipt['sha256'] == receipt['sha256'], name
        assert sha(ROOT / name) == text_receipt['normalized_lf_sha256'], name
    fast_hash = hashlib.sha256((runtime['source_sha256'] +
        sha(ROOT / 'tools/wmrequal/grouped_feedback.py') +
        sha(ROOT / 'tools/wmrequal/score_fast.py')).encode()).hexdigest()
    for i in range(36):
        batch = json.loads((ROOT / f'tools/wmrequal/evidence/validation/batch_{i:03d}.json').read_text(encoding='utf-8'))
        assert batch['receipt']['source_sha256'] == fast_hash
    arms = [data['per_arm'][a] for a in ('mappo', 'ours_c1', 'coopoff_c1')]
    sets = [*arms, data['all_arms'], data['no_mappo_sensitivity']]
    table = None
    checked = dict(accuracy_cells=0, component_tuples=0, gain_harm_cells=0)
    for line in lines:
        if line.startswith('| h s | MAPPO |'):
            table = 'accuracy'
        elif line.startswith('| h s | MAPPO: S;'):
            table = 'decomposition'
        elif line.startswith('| h s | MAPPO G;'):
            table = 'calls'
        elif re.match(r'^\| (60|90|120|180|300) \|', line):
            cells = [v.strip() for v in line.strip('|').split('|')]
            h = cells[0]
            if table == 'accuracy':
                assert len(cells) == 6
                for cell, values in zip(cells[1:], sets):
                    numbers = [float(v.replace('−', '-')) for v in cell.split('/')]
                    j = values['ON'][h]['J_median']
                    expected = [j[k] for k in ('sign', 'mae', 'correlation', 'bias')]
                    assert np.all(abs(np.array(numbers) - expected) <= [.000501,.050001,.000501,.050001]), (h, numbers, expected)
                    checked['accuracy_cells'] += 1
            elif table == 'decomposition':
                assert len(cells) == 4
                for cell, values in zip(cells[1:], arms):
                    parts = cell.split(';')
                    assert len(parts) == 3
                    for part, initial, metric in zip(parts, 'SUD', ('source', 'upstream', 'downstream')):
                        assert part.strip().startswith(initial + ' ')
                        numbers = [float(v.replace('−', '-').strip('%')) for v in part.strip()[2:].split('/')]
                        d = values['ON'][h]['decomposition'][metric]
                        expected = [d['bias'], d['mae'], 100*d['mse_covariance_share']]
                        np.testing.assert_allclose(numbers, expected, rtol=0, atol=.050001)
                        checked['component_tuples'] += 1
            elif table == 'calls':
                assert len(cells) == 6
                for cell, values in zip(cells[1:], sets):
                    c = values['ON'][h]['conditional_zero_replicate_sigma']
                    expected = '; '.join(f"{c[k]}:{c[k + '_correct']}/{c[k + '_sign_n']}"
                        if c[k] else '0:—' for k in ('gain','harm'))
                    assert cell == expected, (h, cell, expected)
                    checked['gain_harm_cells'] += 1
    assert checked == dict(accuracy_cells=25, component_tuples=45, gain_harm_cells=25)
    for name in ('audit_runtime.py', 'analyze.py', 'probe_cpu.py', 'verify_report.py'):
        tree = ast.parse((HERE / name).read_text(encoding='utf-8'))
        for n in tree.body:
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SCRIPT' for t in n.targets):
                ast.parse(ast.literal_eval(n.value))
    print(json.dumps(dict(report_lines=len(lines), numeric_tables_verified=checked,
        frozen_sources_and_fast_receipts_match=True, scripts_parse=4)))


if __name__ == '__main__':
    main()
