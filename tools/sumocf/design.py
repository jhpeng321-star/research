"""Immutable arithmetic/world allocation; no simulator imports."""
CELLS = [(p, d) for p in (.05, .1, .2, .4) for d in (1900, 2400, 2700)]
TRAIN_CAP = 8_000_000
DEV_CAP = 300_000
UPDATES = 32
SAMPLES = 8
DEV_UPDATES = (3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 32)


def train_worlds(update):
    assert 1 <= update <= UPDATES
    u = update - 1
    return [dict(seed=969200+6*u+j, penetration=CELLS[(2*j+u%2)%12][0],
                 demand=CELLS[(2*j+u%2)%12][1]) for j in range(6)]


def dev_worlds():
    return [dict(seed=969600+j, penetration=CELLS[j%12][0], demand=CELLS[j%12][1])
            for j in range(24)]


CANARY = dict(seed=969624, penetration=.4, demand=2700)
