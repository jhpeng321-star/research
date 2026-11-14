"""E1 Random arm: an independent, reproducible permutation of the pool.

ISOLATION CONTRACT (tested in tests/test_e1.py): this module imports only hashlib and random, contains no file,
path, network or subprocess access, and its single function receives nothing but the pool's descriptor hashes and
the search seed. It therefore cannot depend on a world-model score or a SUMO outcome, whatever exists on disk.
"""
import hashlib
import random

STREAM = 'claim_validation.e1_random_arm.v1'


def random_order(pool_hashes, seed):
    """The Random arm's ordering of the pool: the hashes are first sorted (so the result does not depend on the
    generation order or on any other ordering of the pool) and then shuffled by a stream that is named differently
    from the pool-generation stream."""
    hashes = [str(h) for h in pool_hashes]
    if len(set(hashes)) != len(hashes):
        raise ValueError('pool hashes are not distinct')
    out = sorted(hashes)
    rng = random.Random(int(hashlib.sha256(f'{STREAM}|{int(seed)}'.encode()).hexdigest()[:16], 16))
    rng.shuffle(out)
    return out
