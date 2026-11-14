"""The G1-G6 gate registry (single source of truth for the CLI and reports)."""
from __future__ import annotations

from .gates import ALL

GATES = {g.NAME: g for g in ALL}
GATE_ORDER = tuple(g.NAME for g in ALL)


def entry_points_for(name: str) -> tuple:
    gate = GATES.get(name)
    return tuple(getattr(gate, "ENTRY_POINTS", ())) if gate else ()


def describe() -> list:
    """Ordered description of every gate and the entry points it needs."""
    out = []
    for i, name in enumerate(GATE_ORDER, start=1):
        gate = GATES[name]
        out.append({
            "gate": name,
            "id": f"G{i}",
            "title": getattr(gate, "TITLE", name),
            "entry_points": list(getattr(gate, "ENTRY_POINTS", ())),
        })
    return out
