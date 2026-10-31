"""Inline byte transport: no torch storage/file descriptors cross process queues."""
import hashlib
import pickle

from .protocol import safe_path


def pack(value):
    # Standard Pickler embeds CPU storage bytes. ForkingPickler instead exports
    # one shared-storage FD per old_logp tensor, exhausting a long PPO wave.
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def unpack(blob):
    if not isinstance(blob, bytes):
        raise TypeError('only inline bytes may cross the task process boundary')
    # Only our own just-serialized state/trajectory is accepted by the caller.
    return pickle.loads(blob)


def retain(root, transitions):
    blob = pack(transitions)
    path = safe_path(root)/'trajectory.pkl'
    with path.open('xb') as stream:
        stream.write(blob)
        stream.flush()
        __import__('os').fsync(stream.fileno())
    return blob, hashlib.sha256(blob).hexdigest()
