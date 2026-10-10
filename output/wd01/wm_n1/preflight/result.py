"""Gate result + JSON-able report payloads."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GateResult:
    """One gate's verdict."""

    gate: str
    title: str
    passed: bool
    detail: str
    entry_points: tuple = ()
    checks: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_json(self) -> dict:
        return {
            "schema": "wm_n1_preflight_gate/1",
            "gate": self.gate,
            "title": self.title,
            "passed": bool(self.passed),
            "verdict": "PASS" if self.passed else "FAIL",
            "detail": self.detail,
            "entry_points": list(self.entry_points),
            "checks": self.checks,
            "error": self.error,
        }


def fail(gate: str, title: str, detail: str, entry_points=(),
         checks=None, error=None) -> GateResult:
    return GateResult(gate, title, False, detail, tuple(entry_points),
                      dict(checks or {}), error)


def ok(gate: str, title: str, detail: str, entry_points=(),
       checks=None) -> GateResult:
    return GateResult(gate, title, True, detail, tuple(entry_points),
                      dict(checks or {}), None)
