"""Frozen numeric/vocabulary constants shared across ``wms``."""
from __future__ import annotations

#: Kinematic feature width per slot: s, lane one-hot (3), v, a, length.
KIN_F: int = 7
#: Short per-driver history: last 5 s at 0.5 s.
HISTORY_H: int = 10
#: History feature width: v/30, a/3, lane one-hot (3), min(gap,250)/100.
HISTORY_F: int = 6

#: Channels of a long driver-history sample (rolling features + 3 extras).
LONG_HISTORY_F: int = 9
#: Length (m) assumed for a slot that carries no ``length_m``.
DEFAULT_LENGTH_M: float = 5.0

#: Controller-id vocabulary. ``none`` is the HDV slot; ``other`` collects
#: open-loop training policies so they are never mislabelled as native.
CONTROLLER_VOCAB: tuple[str, ...] = ("none", "native", "pi", "damping", "wall", "other", "direct")
N_CONTROLLERS: int = len(CONTROLLER_VOCAB)

#: ``format`` value written into (and required from) every checkpoint.
CHECKPOINT_FORMAT: str = "wm_s/1"

_CONTROLLER_ALIASES = {
    "fleet-pi": "pi", "pi-external-u": "pi", "fleet-pi-external-u": "pi",
    "direct-output": "direct",
}


def controller_index(policy: object) -> int:
    """Map a pack ``policy``/``arm`` string to an index of :data:`CONTROLLER_VOCAB`."""
    text = str(policy or "").strip().lower()
    text = _CONTROLLER_ALIASES.get(text, text)
    if text in CONTROLLER_VOCAB:
        return CONTROLLER_VOCAB.index(text)
    return CONTROLLER_VOCAB.index("other")
