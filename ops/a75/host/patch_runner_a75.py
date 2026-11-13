"""AMENDMENT 75 runner edit (idempotent, zero SUMO): register the A75 hold-form executor arms (A75_PLAN / A75_BANK)
in tools/coopmwm/nm_eval_runner.py, exactly like the A44 arms. Additive only: every existing arm keeps its code path.
usage: patch_runner_a75.py arms <nm_eval_runner.py>      (LF files only; prints patched / unchanged)
The checkpoint of an A75 spec carries kind='a75', a75_arm, a75_sha256 (= a75_holdforms.a75_digest()), a75_bank
(path + file_sha256), a75_family (A75_PLAN only); the record carries a75_sha256_loaded and the a75 receipts.
"""
from pathlib import Path
import sys

HOOK_ANCHOR = "def a33_hook(ck, world):\n"
HOOK_ADD = '''A75_ARMS = ('A75_PLAN', 'A75_BANK')     # amendment 75 (MAIN 2026-11-13): hold-form variant of the A44 executor


def a75_hook(ck, world, arm):
    """Amendment 75: a75_holdforms.make_hook (the A44 executor; plain plans unchanged, extended hold forms: short /
    gap-start / zone / relative / taper / pulse) for this world's cell_key; bank doc file sha re-verified; executed
    code == registration (a75_sha256). A75_PLAN runs the single family ck['a75_family']; A75_BANK every row."""
    from . import a75_holdforms
    if arm not in A75_ARMS or ck.get('a75_arm') != arm:
        raise ValueError(f'A75: ck a75_arm {ck.get("a75_arm")!r} is not the arm {arm!r}')
    cell = world.get('cell_key')
    if not cell or cell != ck.get('cell_key') or 'bank_cell_key' in ck:
        raise ValueError('A75: world cell_key missing / not the bound cell_key (cell plans run in their own cell)')
    loaded = a75_holdforms.a75_digest()
    if loaded != ck.get('a75_sha256'):
        raise ValueError('A75: a75_holdforms differs from its registration (a75_sha256)')
    entry = ck.get('a75_bank') or {}
    path = Path(entry.get('path', ''))
    if not path.is_file() or _sha(path) != entry.get('file_sha256'):
        raise ValueError('A75: a75_bank changed after registration')
    bank = json.loads(path.read_text(encoding='utf-8'))
    family = ck.get('a75_family') if arm == 'A75_PLAN' else None
    if arm == 'A75_PLAN' and not family:
        raise ValueError('A75_PLAN needs ck a75_family')
    return a75_holdforms.make_hook(arm, cell, bank=bank, family=family), loaded


'''
EDITS = (
    ("+ A37_ARMS + A44_ARMS:\n        kind = dict(", "+ A37_ARMS + A44_ARMS + A75_ARMS:\n        kind = dict("),
    ("                    **{a: 'a44' for a in A44_ARMS})[spec['arm']]",
     "                    **{a: 'a44' for a in A44_ARMS}, **{a: 'a75' for a in A75_ARMS})[spec['arm']]"),
    ("        if spec['arm'] in A44_ARMS:\n            oracle = a44_hook(ck, world, spec['arm'])\n            rule = oracle\n",
     "        if spec['arm'] in A44_ARMS:\n            oracle = a44_hook(ck, world, spec['arm'])\n            rule = oracle\n"
     "        if spec['arm'] in A75_ARMS:\n            oracle = a75_hook(ck, world, spec['arm'])\n            rule = oracle\n"),
    ("    if oracle is not None and spec['arm'] in A44_ARMS:\n",
     "    if oracle is not None and spec['arm'] in A75_ARMS:\n        extra['a75_sha256_loaded'] = oracle[1]\n"
     "        extra['a75'] = oracle[0].receipts()\n    elif oracle is not None and spec['arm'] in A44_ARMS:\n"),
    ("+ A32_ARMS + A33_ARMS + A37_ARMS + A44_ARMS:\n        raise ValueError(f'unknown evaluation arm",
     "+ A32_ARMS + A33_ARMS + A37_ARMS + A44_ARMS + A75_ARMS:\n        raise ValueError(f'unknown evaluation arm"),
)


def once(s, old, new):
    if s.count(old) != 1:
        raise SystemExit(f'anchor not found exactly once: {old[:70]!r}')
    return s.replace(old, new)


def patch(s):
    if 'A75_ARMS' in s:
        return s
    s = once(s, HOOK_ANCHOR, HOOK_ADD + HOOK_ANCHOR)
    for old, new in EDITS:
        s = once(s, old, new)
    return s


def main():
    if len(sys.argv) != 3 or sys.argv[1] != 'arms':
        raise SystemExit(__doc__)
    p = Path(sys.argv[2])
    s = p.read_bytes().decode('utf-8')
    if '\r\n' in s:
        raise SystemExit(f'{p}: CRLF file; convert first')
    t = patch(s)
    if t != s:
        p.write_bytes(t.encode('utf-8'))
    print('patched' if t != s else 'unchanged', p)


if __name__ == '__main__':
    main()
