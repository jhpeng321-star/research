"""Snapshot existing reference functions and observable reductions; no inference."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {
    'breakdown': ('E:/agent-work/c4-entrant/tools/lrdiag/replay.py', ['breakdown']),
    'worker': ('E:/Temp/opencode/rbeval_wm_worker.py', ['history_tensors', 'callback_for']),
    'starts': ('E:/agent-work/c4-wmcalib/tools/wmcalib/wm_gpu_wmcalib.py', ['make_starts_wmdata']),
    'costs': ('E:/agent-work/c4-speedhead/tools/setupaudit/panel2.py', ['expand', 'Entries', 'costs']),
    'motion': ('E:/agent-work/c4-speedhead/tools/setupaudit/measure.py', ['parity_motion', 'native_background_motion']),
}


def main():
    refs = {}
    for key, (path, names) in SOURCES.items():
        text = Path(path).read_text(encoding='utf-8')
        nodes = {n.name: n for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        code = '\n\n'.join(ast.get_source_segment(text, nodes[n]) for n in names)
        refs[key] = dict(source=path, source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                         code=code, code_sha256=hashlib.sha256(code.encode()).hexdigest())
    target = HERE/'references.json'
    with target.open('x', encoding='utf-8') as f:
        json.dump(refs, f, indent=1)
    source = Path('E:/agent-work/c4-wmbreakdown/docs/paper_results/wmbreakdown/wmdata_observables.json')
    raw = source.read_bytes()
    data = json.loads(raw)
    assert len(data['episodes']) == 200
    # Already reduced observables only: this receipt contains no assignment/truth tables.
    out = HERE/'real_observables.json'
    with out.open('xb') as f:
        f.write(raw)
    receipt = dict(source=str(source), sha256=hashlib.sha256(raw).hexdigest(), episodes=200,
                   detector_sha256=refs['breakdown']['code_sha256'])
    with (HERE/'PREPARED.json').open('x') as f:
        json.dump(receipt, f, indent=1)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
