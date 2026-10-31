"""Prospective allocation, strict input paths and exact approval boundary."""
import hashlib
import re
import random
from pathlib import Path

PROTOCOL = 'S-C1SWITCH2/v2'
LIBRARY = ('native', 'ctrl0', 'mappo', 'pi', 'circles', 'wall')
CADENCE = 20
TIMES = tuple(range(300, 900, CADENCE))
BLOCKS = {'fleet': (18000000, 18007999)}
# Evaluation is supplied by the registered S-C1SWITCH2 manifest, never guessed.
PANEL = ()
CELLS = tuple((p, d) for p in (.05, .1, .2, .4) for d in (1900, 2400, 2700))
PANEL_CELLS = CELLS + ((.2, 2400), (.2, 2700), (.4, 2400), (.4, 2700))
ITERATIONS, EPISODES = 169, 46
CAP = 7_012_800
APPROVAL = 'S-SELMAPPO BUDGET APPROVED'
FORBIDDEN = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|'
                       r'vehicle_params|driver_params|vtype|\.rou\.xml', re.I)


def safe_path(path):
    path = Path(path)
    if FORBIDDEN.search(str(path)):
        raise ValueError('prohibited input path')
    resolved = path.resolve()
    if FORBIDDEN.search(str(resolved)):
        raise ValueError('prohibited resolved input path')
    return resolved


def digest(path):
    return hashlib.sha256(safe_path(path).read_bytes()).hexdigest()


def guard_seed(seed, variant, *, evaluation=False, smoke=False, panel=()):
    if type(seed) is not int or 964000 <= seed <= 964199 or seed // 1000 in (962, 977):
        raise ValueError('forbidden seed')
    lo, hi = BLOCKS[variant]
    if evaluation and smoke:
        raise ValueError('ambiguous seed role')
    allowed = panel if evaluation else (range(hi-1, hi+1) if smoke else range(lo, lo+ITERATIONS*EPISODES))
    if seed not in allowed:
        raise ValueError('seed outside exact task allocation')
    return seed


def training_world(seed, variant='fleet'):
    guard_seed(seed, variant)
    index = seed-BLOCKS[variant][0]
    iteration, within = divmod(index, EPISODES)
    cells = list(CELLS)
    random.Random(iteration).shuffle(cells)
    p, demand = cells[within % len(cells)]
    return dict(seed=seed, penetration=p, demand=demand)


def guard_world(world, variant, *, evaluation=False, smoke=False, panel=()):
    guard_seed(world['seed'], variant, evaluation=evaluation, smoke=smoke, panel=panel)
    if evaluation:
        if (world['penetration'], world['demand']) not in CELLS:
            raise ValueError('unknown evaluation cell')
    else:
        expected = (dict(seed=world['seed'], penetration=.4, demand=2700) if smoke else
                    training_world(world['seed'], variant))
        if world != expected:
            raise ValueError('seed/cell differs from prospective schedule')


def approval_line(registration):
    """A mere quoted proposal/negative sentence does not authorize simulation."""
    lines = safe_path(registration).read_text(encoding='utf-8').splitlines()
    return next((line for line in reversed(lines)
                 if (re.match(r'^-\s*(?:\d\d:\d\d\s+)?MAIN\s*:?\s*'
                              r'S-SELMAPPO BUDGET APPROVED\b', line)
                     or re.match(r'^-\s*23:42\s+S-SELMAPPO BUDGET APPROVED IN ADVANCE by the owner\b', line))), None)


def audit_registration(registration):
    """Conservative token/range audit of the entire shared file, including umbrellas.

    Expands shortened endpoints (e.g. 981000-099); ignores hex hashes. Own
    design-registration lines are identified explicitly, never broadly ignored.
    """
    hits = []
    for number, line in enumerate(safe_path(registration).read_text(encoding='utf-8').splitlines(), 1):
        if line.startswith('- S-SELMAPPO DESIGN REGISTERED '):
            continue
        for match in re.finditer(r'(?<![\da-fA-F])(\d{6,10})(?![\da-fA-F])', line):
            start = int(match[1])
            end = start
            tail = line[match.end():]
            interval = re.match(r'\s*(?:\.\.|-|–|—|锟\?)\s*(\d{1,10})(?!\d)', tail)
            if interval:
                token = interval[1]
                end = int(str(start)[:-len(token)] + token) if len(token) < len(str(start)) else int(token)
            if any(start <= hi and end >= lo for lo, hi in BLOCKS.values()):
                hits.append(dict(line=number, start=start, end=end))
    return dict(registration_sha256=digest(registration), blocks=BLOCKS,
                 conflicts=hits, passed=not hits)
