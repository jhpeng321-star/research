"""Metadata-only exposure audit; prune restricted paths BEFORE directory traversal."""
import os
import re

from tools.c1switch.design import FORBIDDEN, checked, sha

from .design import canary_world, worlds

SEED = re.compile(r'(?<![A-Za-z0-9.])(?:TRAIN|seed|s|w)?(9\d{5})(?!\d)')
TAIL = re.compile(r'^\s*(?:-|–|—|\.\.|~|锟\?)\s*(\d{3,6})(?!\d)')


def registry_hits(text):
    proposed = {w['seed'] for w in worlds()+[canary_world()]}
    hits = []
    for index, line in enumerate(text.splitlines(), 1):
        # Our own prospective reservation is not a prior exposure.
        if 'S-C1SWITCH2 DESIGN REGISTERED' in line:
            continue
        # This exact durable STOP report refers to our already allocated failed
        # world, not a new reservation. Remove only its known job identity;
        # every other seed/range on the line must still be audited.
        if ('S-C1SWITCH2 06:00 INFRASTRUCTURE STOP / GPU RELEASE' in line and
                'gpuR own763520 EXITED durableSTOP on wm__s969147' in line and
                'BB0physics/GPU' in line):
            line = line.replace('wm__s969147', 'registered-failed-job')
        # The per-CAV owner explicitly leaves its evaluation unused and consumes
        # OUR registered panel. Remove only that exact cross-reference, never
        # another allocation on the same line or a generic mention of this task.
        if ('S-SELMAPPO2 DESIGN REGISTERED' in line and 'Per S-C1SWITCH2 registration' in line
                and 'leave local14400evaluation reservation UNSTARTED' in line):
            line = line.replace('panel969136..969151/J600 receipts', 'registered-panel/J600 receipts')
        for match in SEED.finditer(line):
            lo = hi = int(match[1])
            tail = TAIL.match(line[match.end():])
            if tail:
                digits = tail[1]
                hi = int(str(lo)[:6-len(digits)]+digits)
            # This is the parent TRAIN eligibility range, not an allocated panel.
            if (lo, hi) in {(966000, 969999), (965184, 969999)}:
                continue
            # Historical negative exposure evidence, not a world reservation.
            if (lo, hi) == (966000, 969699) and re.search(r'none\s+(?:used|in)', line, re.I):
                continue
            overlap = proposed.intersection(range(lo, hi+1))
            if overlap:
                hits.append(dict(line=index, interval=[lo, hi], seeds=sorted(overlap)))
    return hits


def safe_filename_hits(root):
    root = checked(root)
    wanted = {str(w['seed']) for w in worlds()+[canary_world()]}
    hits, stack, visited = [], [root], 0
    while stack:
        current = stack.pop()
        with os.scandir(current) as entries:
            for entry in entries:
                if FORBIDDEN.search(entry.path) or entry.is_symlink():
                    continue
                visited += 1
                if any(re.search(r'(?<!\d)'+s+r'(?!\d)', entry.name) for s in wanted):
                    hits.append(entry.path)
                if entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
    return dict(root=str(root), visited_safe_names=visited, hits=hits)


def registration_audit(path):
    return dict(registration_sha256=sha(path),
                hits=registry_hits(checked(path).read_text(encoding='utf-8')))
