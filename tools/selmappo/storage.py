"""Append-only task receipts: precharge full episodes, never refund failures."""
import json
import math
import os

from .protocol import CAP, guard_seed, safe_path


def append(path, value):
    with safe_path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


class Budget:
    """Single-supervisor ledger. Existing unresolved START blocks resumption."""
    def __init__(self, root):
        self.root = safe_path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root/'ledger.jsonl'
        self.pending = {}
        self.charged = 0
        self.seen = set()
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                row = json.loads(line)
                if row['event'] == 'HISTORY':
                    from .protocol import digest
                    ids = row['consumed_ids']
                    consumed = {38700: 41, 43200: 46}.get(row['charged_sim_s'])
                    if (self.seen or self.charged or self.pending or
                            consumed is None or len(ids) != consumed+2 or len(set(ids)) != consumed+2 or
                            set(ids) != {f'fleet:{s}' for s in
                                         (*range(18000000, 18000000+consumed), 18007998, 18007999)} or
                            digest(self.root/'inherited_ledger.jsonl') != row['ledger_sha256']):
                        raise ValueError('invalid immutable transport history charge')
                    if consumed == 46:
                        from .continuation import TRANSPORT_LEDGER_SHA
                        if (row['transport_ledger_sha256'] != TRANSPORT_LEDGER_SHA or
                                digest(self.root/'inherited_transport_ledger.jsonl') != TRANSPORT_LEDGER_SHA):
                            raise ValueError('changed transport ancestor ledger')
                    self.charged = row['charged_sim_s']
                    self.seen.update(ids)
                elif row['event'] == 'START':
                    if row['id'] in self.seen or row['reserved_sim_s'] != 900:
                        raise ValueError('duplicate/unclosed reservation')
                    self.pending[row['id']] = row
                    self.seen.add(row['id'])
                    self.charged += row['reserved_sim_s']
                elif row['event'] == 'END' and row['id'] in self.pending:
                    if not math.isfinite(row['actual_sim_s']) or not 0 <= row['actual_sim_s'] <= 900:
                        raise ValueError('invalid actual ledger debit')
                    self.pending.pop(row['id'])
                else:
                    raise ValueError('invalid ledger ordering')
        if self.pending:
            raise ValueError('unclosed episode: inspect owned PID/actual receipt; no auto retry')
        if self.charged > CAP:
            raise ValueError('historical cap exceeded')

    def reserve(self, world, variant, *, evaluation=False, smoke=False, panel=()):
        guard_seed(world['seed'], variant, evaluation=evaluation, smoke=smoke, panel=panel)
        ident = f'{variant}:{world["seed"]}'
        if ident in self.seen or self.charged+900 > CAP:
            raise ValueError('duplicate episode or budget exhaustion')
        append(self.path, dict(event='START', id=ident, world=world, reserved_sim_s=900))
        self.pending[ident] = world
        self.seen.add(ident)
        self.charged += 900

    def finish(self, actual, ident=None):
        if ident is None and len(self.pending) == 1:
            ident = next(iter(self.pending))
        if ident not in self.pending or not math.isfinite(actual) or not 0 <= actual <= 900:
            raise ValueError('invalid terminal accounting')
        append(self.path, dict(event='END', id=ident, actual_sim_s=actual))
        self.pending.pop(ident)


class PrechargedEpisode:
    """Worker accounting; the supervisor alone owns the scientific budget ledger."""
    def __init__(self, root, world, variant, *, evaluation=False, smoke=False, panel=()):
        self.root, self.world, self.variant = safe_path(root), world, variant
        self.role = evaluation, smoke, tuple(panel)
        self.actual, self.reserved = None, False

    def reserve(self, world, variant, *, evaluation=False, smoke=False, panel=()):
        if (self.reserved or world != self.world or variant != self.variant or
                (evaluation, smoke, tuple(panel)) != self.role):
            raise ValueError('worker differs from precharged episode')
        guard_seed(world['seed'], variant, evaluation=evaluation, smoke=smoke, panel=panel)
        ident = f'{variant}:{world["seed"]}'
        records = [json.loads(line) for line in
                   (self.root.parent/'ledger.jsonl').read_text(encoding='utf-8').splitlines()]
        owned = [row for row in records if row.get('id') == ident]
        if (len(owned) != 1 or owned[0]['event'] != 'START' or
                owned[0]['reserved_sim_s'] != 900 or owned[0]['world'] != world):
            raise ValueError('worker has no unique supervisor precharge')
        self.reserved = True

    def finish(self, actual):
        if not self.reserved or self.actual is not None or not math.isfinite(actual) or not 0 <= actual <= 900:
            raise ValueError('invalid worker terminal accounting')
        append(self.root/'actual.jsonl', dict(actual_sim_s=actual))
        self.actual = actual
