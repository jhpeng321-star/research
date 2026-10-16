"""WM-S: physics-informed next-state world model for mixed CAV/HDV traffic.

Sub-packages (dependency order, see docs/ARCHITECTURE_V4.md section 2.1):
``core < config < physics < model < train < cli``; ``data`` and ``eval`` sit beside
``train``; ``legacy`` may import anything below it.
"""
from __future__ import annotations

__version__ = "0.1.0"
