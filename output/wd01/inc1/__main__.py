"""``python -m inc1`` -> the gated INC1 launcher (refuses without the gates)."""
from __future__ import annotations

from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
