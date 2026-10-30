"""Integer per-entrance conservation. No clipping, demand loss, or queue offset."""
from __future__ import annotations

from dataclasses import dataclass
import numbers


def count(value) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 0:
        raise ValueError("counts must be nonnegative integers")
    return int(value)


@dataclass
class SourceLedger:
    due: int
    inserted: int = 0

    def __post_init__(self):
        self.due, self.inserted = count(self.due), count(self.inserted)
        if self.inserted > self.due:
            raise ValueError("insertions exceed requests")

    @property
    def queue(self) -> int:
        return self.due - self.inserted

    def advance(self, arrivals: int, admitted: int) -> int:
        arrivals, admitted = count(arrivals), count(admitted)
        if admitted > self.queue + arrivals:
            raise ValueError("admission exceeds available demand")
        self.due += arrivals
        self.inserted += admitted
        return self.queue
